from dataclasses import replace
import importlib
import importlib.util
import random
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from temporal_model_zoo import (
    CausalTransformerTemporalPolicy,
    LargeRecurrentTemporalPolicy,
    decision_safe_tokens,
)
from temporal_observation import (
    DecisionInput,
    EventKind,
    PublicEventToken,
    build_legal_actions,
    build_private_observation,
)
from temporal_puffer_agent import TrajectoryStep
from temporal_zoo_rl import (
    PPOConfig,
    DEFAULT_TEMPORAL_LEAGUE_MIXTURE,
    configured_league_mixture,
    context_compatible_episodes,
    cpu_rollout_thread_scope,
    evaluate_ppo_drift_packed,
    gae_returns,
    pack_ppo_episodes,
    ppo_update_epochs,
    ppo_update_packed,
    run_rollout_operation,
    validate_development_seed,
)


PUBLIC = {
    "current_suit": "harten",
    "current_rank": 5,
    "public_hand_sizes": (3, 4),
    "observer_index": 0,
    "current_player_index": 0,
    "direction": 1,
    "penalty_amount": 0,
    "public_sorts_played": (2, 1, 0, 1, 0),
}


def decision(own_cards=(4, 16, 49), events=()):
    observation = build_private_observation(
        own_cards=own_cards,
        **PUBLIC,
    )
    legal = build_legal_actions(
        own_cards=own_cards,
        current_suit=PUBLIC["current_suit"],
        current_rank=PUBLIC["current_rank"],
        penalty_amount=PUBLIC["penalty_amount"],
    )
    return DecisionInput(observation, legal, tuple(events))


def models():
    return (
        LargeRecurrentTemporalPolicy(
            hidden_size=24, num_layers=1, cell="gru"
        ),
        CausalTransformerTemporalPolicy(
            d_model=24,
            num_layers=1,
            num_heads=4,
            feedforward_size=48,
            max_context=32,
        ),
    )


def rollout(policy, decisions):
    state = policy.initial_state()
    steps = []
    policy.eval()
    with torch.no_grad():
        for item in decisions:
            result = policy.step(item, state)
            state = result.state
            action = int(result.logits.argmax())
            distribution = torch.distributions.Categorical(
                logits=result.logits
            )
            steps.append(
                TrajectoryStep(
                    decision=item,
                    action=action,
                    old_log_prob=float(
                        distribution.log_prob(
                            torch.tensor(
                                action, device=result.logits.device
                            )
                        )
                    ),
                    value=float(result.value),
                )
            )
    return tuple(steps)


def sample_decisions():
    return (
        decision(
            events=(PublicEventToken(EventKind.EPISODE_START),)
        ),
        decision(
            own_cards=(4, 16),
            events=(
                PublicEventToken(
                    EventKind.DRAW_COUNT,
                    actor_relative=1,
                    count=2,
                ),
            ),
        ),
    )


@pytest.mark.parametrize("policy", models())
def test_packed_forward_matches_online_history_and_resets_episode(policy):
    first = sample_decisions()
    second = (
        decision(
            own_cards=(4,),
            events=(PublicEventToken(EventKind.EPISODE_START),),
        ),
    )
    episodes = [
        (rollout(policy, first), True),
        (rollout(policy, second), False),
    ]
    packed = pack_ppo_episodes(episodes, device="cpu")
    policy.eval()
    with torch.no_grad():
        packed_logits, packed_values = policy.forward_packed(packed.safe)

    scalar_logits = []
    scalar_values = []
    for decisions in (first, second):
        state = policy.initial_state()
        with torch.no_grad():
            for item in decisions:
                result = policy.step(item, state)
                state = result.state
                scalar_logits.append(result.logits)
                scalar_values.append(result.value)
    torch.testing.assert_close(
        packed_logits, torch.stack(scalar_logits), atol=1e-5, rtol=1e-5
    )
    torch.testing.assert_close(
        packed_values, torch.stack(scalar_values), atol=1e-5, rtol=1e-5
    )
    assert packed.episode_decision_offsets == (0, 2, 3)
    assert packed.safe.valid_tokens[1].sum() == len(
        decision_safe_tokens(second[0])
    )


def test_gae_is_terminal_and_never_crosses_episode_boundaries():
    fake = decision(own_cards=(4,))
    two_steps = (
        TrajectoryStep(fake, 4, -0.2, 0.0),
        TrajectoryStep(fake, 4, -0.2, 0.0),
    )
    one_step = (TrajectoryStep(fake, 4, -0.2, 0.0),)
    config = PPOConfig(
        gamma=1.0,
        gae_lambda=1.0,
        step_reward=0.0,
        normalize_advantages=False,
    )

    win_returns, win_advantages = gae_returns(
        two_steps, won=True, config=config
    )
    loss_returns, loss_advantages = gae_returns(
        one_step, won=False, config=config
    )

    torch.testing.assert_close(win_returns, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(win_advantages, torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(loss_returns, torch.tensor([-1.0]))
    torch.testing.assert_close(loss_advantages, torch.tensor([-1.0]))


@pytest.mark.parametrize("policy", models())
def test_initial_ratio_is_one_kl_zero_and_update_is_finite(policy):
    episodes = [(rollout(policy, sample_decisions()), True)]
    before = evaluate_ppo_drift_packed(
        policy,
        episodes,
        config=PPOConfig(normalize_advantages=False),
        minibatch_episodes=1,
    )
    assert before["ratio_mean"] == pytest.approx(1.0, abs=1e-6)
    assert before["approx_kl"] == pytest.approx(0.0, abs=1e-7)
    batch = pack_ppo_episodes(
        episodes,
        device="cpu",
        config=PPOConfig(normalize_advantages=False),
    )
    logits, _ = policy.forward_packed(batch.safe)
    assert torch.isfinite(logits[batch.safe.legal_actions]).all()
    assert torch.all(
        logits[~batch.safe.legal_actions]
        == torch.finfo(logits.dtype).min
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    metrics = ppo_update_packed(
        policy,
        batch,
        optimizer,
        config=PPOConfig(normalize_advantages=False),
    )
    for key in (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "grad_norm",
    ):
        assert math_is_finite(metrics[key])
    assert metrics["samples"] == 2


def math_is_finite(value):
    return not (value != value or value in (float("inf"), float("-inf")))


def test_hidden_worlds_cannot_change_packed_bytes_or_update():
    hidden_a = {"opponent_cards": (7, 8, 9), "draw_order": (31, 32)}
    hidden_b = {"opponent_cards": (40, 41, 42), "draw_order": (47, 48)}
    assert hidden_a != hidden_b
    public_event = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=1, count=2
    )
    public_decision_a = decision(events=(public_event,))
    public_decision_b = decision(events=(public_event,))
    assert public_decision_a == public_decision_b

    torch.manual_seed(71)
    first = CausalTransformerTemporalPolicy(
        d_model=24,
        num_layers=1,
        num_heads=4,
        feedforward_size=48,
        max_context=16,
    )
    second = CausalTransformerTemporalPolicy(
        d_model=24,
        num_layers=1,
        num_heads=4,
        feedforward_size=48,
        max_context=16,
    )
    second.load_state_dict(first.state_dict())
    episodes_a = [(rollout(first, (public_decision_a,)), True)]
    episodes_b = [(rollout(second, (public_decision_b,)), True)]
    packed_a = pack_ppo_episodes(episodes_a, device="cpu")
    packed_b = pack_ppo_episodes(episodes_b, device="cpu")
    assert torch.equal(packed_a.safe.tokens, packed_b.safe.tokens)
    assert torch.equal(
        packed_a.safe.legal_actions, packed_b.safe.legal_actions
    )
    optimizer_a = torch.optim.AdamW(first.parameters(), lr=1e-3)
    optimizer_b = torch.optim.AdamW(second.parameters(), lr=1e-3)
    metrics_a = ppo_update_packed(first, packed_a, optimizer_a)
    metrics_b = ppo_update_packed(second, packed_b, optimizer_b)
    assert metrics_a["loss"] == pytest.approx(metrics_b["loss"], abs=1e-7)
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)


def test_epochs_preserve_whole_episodes_and_context_filter_drops_whole_rows():
    policy = LargeRecurrentTemporalPolicy(hidden_size=16, num_layers=1)
    short = rollout(policy, (decision(own_cards=(4,)),))
    long = rollout(policy, sample_decisions())
    accepted, metadata = context_compatible_episodes(
        [(short, True), (long, False)],
        max_context=1,
    )
    assert accepted == [(short, True)]
    assert metadata["dropped_over_context"] == 1

    accepted, _ = context_compatible_episodes(
        [(short, True), (long, False)],
        max_context=3,
    )
    assert accepted == [(short, True)]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    result = ppo_update_epochs(
        policy,
        accepted,
        optimizer,
        update_epochs=2,
        minibatch_episodes=1,
        seed=81,
        target_kl=None,
    )
    assert result["epochs_completed"] == 2
    assert result["samples"] == 1


def test_sealed_final_seed_range_is_rejected():
    validate_development_seed(91_999_999)
    with pytest.raises(ValueError, match="development seeds"):
        validate_development_seed(92_000_000)
    with pytest.raises(ValueError, match="development seeds"):
        validate_development_seed(91_999_999, span=2)


def test_cpu_rollout_thread_scope_is_narrow_and_exception_safe():
    original = torch.get_num_threads()
    requested = 1 if original != 1 else 2
    with cpu_rollout_thread_scope("cpu", requested):
        assert torch.get_num_threads() == requested
    assert torch.get_num_threads() == original

    with pytest.raises(RuntimeError, match="rollout failure"):
        with cpu_rollout_thread_scope(torch.device("cpu"), requested):
            assert torch.get_num_threads() == requested
            raise RuntimeError("rollout failure")
    assert torch.get_num_threads() == original


def test_thread_limited_rollout_preserves_seeded_random_streams_and_result():
    observed_threads = []

    def seeded_operation(seed, *, label):
        observed_threads.append(torch.get_num_threads())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return {
            "label": label,
            "python": [random.random() for _ in range(4)],
            "numpy": np.random.randint(0, 1000, size=4).tolist(),
            "torch": torch.randint(0, 1000, (4,)).tolist(),
        }

    expected = seeded_operation(53_991, label="same")
    observed_threads.clear()
    actual = run_rollout_operation(
        seeded_operation,
        53_991,
        label="same",
        rollout_device="cpu",
        cpu_threads=1,
    )

    assert actual == expected
    assert observed_threads == [1]


def test_cuda_rollout_scope_does_not_change_cpu_thread_configuration():
    original = torch.get_num_threads()
    with cpu_rollout_thread_scope("cuda", 1):
        assert torch.get_num_threads() == original
    assert torch.get_num_threads() == original


def _import_temporal_league(monkeypatch):
    if importlib.util.find_spec("pufferlib") is None:
        pufferlib = types.ModuleType("pufferlib")
        models = types.ModuleType("pufferlib.models")
        pufferlib.models = models
        monkeypatch.setitem(sys.modules, "pufferlib", pufferlib)
        monkeypatch.setitem(sys.modules, "pufferlib.models", models)
    return importlib.import_module("temporal_league")


def test_default_league_weights_preserve_historical_schedule(monkeypatch):
    league = _import_temporal_league(monkeypatch)
    defaults = SimpleNamespace(
        league_current_weight=4,
        league_historical_weight=2,
        league_champion_weight=2,
        league_tree_weight=1,
        league_smart_weight=1,
        league_smarter_weight=0,
    )

    mixture = configured_league_mixture(defaults)

    assert mixture == DEFAULT_TEMPORAL_LEAGUE_MIXTURE
    assert league.league_schedule(mixture) == (
        "current_selfplay",
        "current_selfplay",
        "current_selfplay",
        "current_selfplay",
        "historical_temporal",
        "historical_temporal",
        "ppo_champion",
        "ppo_champion",
        "bayesian_tree",
        "smart_rule",
    )
    assert league.league_schedule() == league.league_schedule(mixture)


def test_stronger_league_schedule_is_deterministic_and_validated(monkeypatch):
    league = _import_temporal_league(monkeypatch)
    stronger = (
        ("current_selfplay", 3),
        ("historical_temporal", 3),
        ("ppo_champion", 2),
        ("bayesian_tree", 1),
        ("smart_rule", 1),
        ("smarter_rule", 2),
    )
    expected = (
        "current_selfplay",
        "current_selfplay",
        "current_selfplay",
        "historical_temporal",
        "historical_temporal",
        "historical_temporal",
        "ppo_champion",
        "ppo_champion",
        "bayesian_tree",
        "smart_rule",
        "smarter_rule",
        "smarter_rule",
    )

    assert league.league_schedule(stronger) == expected
    assert league.league_schedule(stronger) == expected
    with pytest.raises(ValueError, match="unknown league"):
        league.league_schedule((("opponent_hidden_cards", 1),))
    with pytest.raises(ValueError, match="non-negative integers"):
        league.league_schedule((("smarter_rule", -1),))
    with pytest.raises(ValueError, match="at least one positive"):
        league.league_schedule(
            tuple((name, 0) for name in league.LEAGUE_OPPONENT_TYPES)
        )


def test_smarter_only_collection_is_seed_exact_and_stays_safe(monkeypatch):
    league = _import_temporal_league(monkeypatch)
    torch.manual_seed(74_201)
    policy = LargeRecurrentTemporalPolicy(
        hidden_size=12,
        num_layers=1,
        cell="gru",
    ).eval()
    mixture = (("smarter_rule", 1),)

    with cpu_rollout_thread_scope("cpu", 1):
        first = league.collect_league_batch(
            policy,
            None,
            [policy],
            games=2,
            seed=74_201,
            mixture=mixture,
        )
        second = league.collect_league_batch(
            policy,
            None,
            [policy],
            games=2,
            seed=74_201,
            mixture=mixture,
        )

    assert first == second
    episodes, contribution = first
    assert contribution["games"] == {"smarter_rule": 2}
    assert episodes
    for trajectory, _ in episodes:
        for step in trajectory:
            assert isinstance(step.decision, DecisionInput)
            assert all(
                isinstance(event, PublicEventToken)
                for event in step.decision.public_events
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_causal_packed_update_runs_entirely_on_cuda():
    policy = CausalTransformerTemporalPolicy(
        d_model=32,
        num_layers=1,
        num_heads=4,
        feedforward_size=64,
        max_context=32,
    ).cuda()
    episodes = [(rollout(policy, sample_decisions()), True)]
    batch = pack_ppo_episodes(
        episodes,
        device="cuda",
        dtype=next(policy.parameters()).dtype,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    metrics = ppo_update_packed(policy, batch, optimizer)
    assert batch.safe.tokens.is_cuda
    assert math_is_finite(metrics["loss"])
