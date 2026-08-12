import json

import pytest
import torch

pufferlib = pytest.importorskip("pufferlib")
if not str(getattr(pufferlib, "__version__", "")).startswith("4"):
    pytest.skip("requires pinned PufferLib 4 runtime", allow_module_level=True)

import game
import pufferlib.models

from temporal_league import ImitationStep, imitation_update_targeted
from temporal_observation import (
    DecisionInput,
    EventKind,
    PublicEventToken,
    build_legal_actions,
    build_private_observation,
    serialize_safe_history,
)
from temporal_puffer_agent import temporal_agent_factory
from packed_puffer_rollout import pack_puffer_episode
from temporal_puffer_v4 import (
    HawkRGLRUTemporalPolicy,
    NativePufferTemporalPolicy,
    PUFFERLIB_REVISION,
    build_temporal_policy,
    policy_model_config,
)
from temporal_training import (
    collect_smoke_episodes,
    evaluate_ppo_drift,
    evaluate_ppo_drift_batched,
    evaluate_ppo_drift_batched_minibatches,
    ppo_update,
    ppo_update_batched,
    ppo_update_batched_epochs,
)


def decision(events=()):
    own_cards = (4, 16, 49)
    observation = build_private_observation(
        own_cards=own_cards,
        current_suit="harten",
        current_rank=5,
        public_hand_sizes=(3, 4),
        observer_index=0,
        current_player_index=0,
        direction=1,
        penalty_amount=0,
        public_sorts_played=(2, 1, 0, 0, 0),
    )
    legal = build_legal_actions(
        own_cards=own_cards,
        current_suit="harten",
        current_rank=5,
        penalty_amount=0,
    )
    return DecisionInput(observation, legal, tuple(events))


def state_bytes(state):
    return b"".join(
        part.detach().cpu().contiguous().numpy().tobytes()
        for part in state
    )


def test_real_v4_policy_and_mingru_are_used():
    policy = NativePufferTemporalPolicy(hidden_size=24)
    assert policy.native_pufferlib
    assert policy.pufferlib_revision == PUFFERLIB_REVISION
    assert isinstance(policy.puffer_policy, pufferlib.models.Policy)
    assert isinstance(policy.puffer_policy.network, pufferlib.models.MinGRU)


def test_mode_conditioned_heads_preserve_native_55_action_contract():
    policy = NativePufferTemporalPolicy(
        hidden_size=16,
        num_layers=1,
        mode_conditioned_heads=True,
    )
    safe_tokens = torch.zeros(2, 3, 100)
    logits, values = policy.forward_native_sequence(safe_tokens)
    assert logits.shape == (6, 55)
    assert values.shape == (2, 3)
    assert hasattr(policy.puffer_policy.decoder, "card_and_pass")
    assert hasattr(policy.puffer_policy.decoder, "suit")


def test_hawk_rglru_parallel_training_matches_recurrent_evaluation():
    torch.manual_seed(29)
    policy = HawkRGLRUTemporalPolicy(
        hidden_size=20,
        num_layers=2,
        recurrent_size=24,
        mlp_expansion=2,
    )
    safe_tokens = torch.randn(3, 41, 100)
    parallel_logits, parallel_values = policy.forward_native_sequence(
        safe_tokens
    )
    state = policy.initial_state(batch_size=3)
    recurrent_logits = []
    recurrent_values = []
    for index in range(safe_tokens.shape[1]):
        logits, values, state = policy.puffer_policy.forward_eval(
            safe_tokens[:, index], state
        )
        recurrent_logits.append(logits)
        recurrent_values.append(values.reshape(-1))

    assert torch.allclose(
        parallel_logits.reshape(3, 41, 55),
        torch.stack(recurrent_logits, dim=1),
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.allclose(
        parallel_values,
        torch.stack(recurrent_values, dim=1),
        atol=2e-6,
        rtol=2e-6,
    )
    loss = parallel_logits.square().mean() + parallel_values.square().mean()
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in policy.parameters()
    )


def test_hawk_rglru_model_config_round_trip():
    policy = HawkRGLRUTemporalPolicy(
        hidden_size=24,
        num_layers=3,
        recurrent_size=28,
        mlp_expansion=3,
    )
    clone = build_temporal_policy(policy_model_config(policy))
    assert isinstance(clone, HawkRGLRUTemporalPolicy)
    assert policy_model_config(clone) == policy_model_config(policy)


def test_native_hidden_world_invariance_is_byte_exact():
    hidden_a = {"opponent_hand": (6, 7, 8, 9), "drawn_card": 31}
    hidden_b = {"opponent_hand": (32, 33, 34, 35), "drawn_card": 47}
    assert hidden_a != hidden_b
    public_draw = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=1, count=1
    )
    safe_a = decision((public_draw,))
    safe_b = decision((public_draw,))
    assert serialize_safe_history((safe_a,)) == serialize_safe_history((safe_b,))

    torch.manual_seed(17)
    policy = NativePufferTemporalPolicy(hidden_size=24).eval()
    with torch.no_grad():
        result_a = policy.step(safe_a, policy.initial_state())
        result_b = policy.step(safe_b, policy.initial_state())
    assert state_bytes(result_a.state) == state_bytes(result_b.state)
    assert torch.equal(result_a.logits, result_b.logits)
    assert torch.equal(result_a.value, result_b.value)


def test_native_batched_training_path_is_real_v4_api():
    policy = NativePufferTemporalPolicy(hidden_size=16)
    safe_tokens = torch.zeros(2, 3, 100)
    logits, values = policy.forward_native_sequence(safe_tokens)
    assert logits.shape == (6, 55)
    assert values.shape == (2, 3)
    (logits.square().mean() + values.square().mean()).backward()
    assert any(
        parameter.grad is not None
        for parameter in policy.puffer_policy.network.parameters()
    )


def test_targeted_distillation_supports_safe_card_and_suit_modes():
    card_decision = decision()
    suit_observation = build_private_observation(
        own_cards=(4, 16, 49),
        current_suit="harten",
        current_rank=5,
        public_hand_sizes=(3, 4),
        observer_index=0,
        current_player_index=0,
        direction=1,
        penalty_amount=0,
        public_sorts_played=(2, 1, 0, 0, 0),
        change_suit_mode=True,
    )
    suit_legal = build_legal_actions(
        own_cards=(4, 16, 49),
        current_suit="harten",
        current_rank=5,
        penalty_amount=0,
        change_suit_mode=True,
    )
    suit_decision = DecisionInput(suit_observation, suit_legal)

    def teacher_logits(safe_decision, chosen):
        values = [torch.finfo(torch.float32).min] * 55
        for index, legal in enumerate(safe_decision.legal_actions):
            if legal:
                values[index] = -1.0
        values[chosen] = 1.0
        return tuple(values)

    card_action = next(
        index
        for index, legal in enumerate(card_decision.legal_actions)
        if legal
    )
    suit_action = 52
    episodes = (
        (
            ImitationStep(
                card_decision,
                card_action,
                teacher_logits(card_decision, card_action),
            ),
            ImitationStep(
                suit_decision,
                suit_action,
                teacher_logits(suit_decision, suit_action),
            ),
        ),
    )
    policy = NativePufferTemporalPolicy(hidden_size=16)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    logs = imitation_update_targeted(
        policy,
        episodes,
        optimizer,
        epochs=1,
        seed=31,
        batch_episodes=1,
    )
    assert logs[0]["samples"] == 2
    assert logs[0]["optimizer_steps"] == 1
    assert torch.isfinite(torch.tensor(logs[0]["objective"]))
    assert torch.isfinite(torch.tensor(logs[0]["distillation_kl"]))


def test_native_batched_ppo_matches_scalar_recurrent_update():
    torch.manual_seed(19)
    behavior = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=2
    )
    episodes, collection = collect_smoke_episodes(
        behavior, seed=19, episode_count=2, max_turns=80
    )
    assert collection["decisions"] > 0

    scalar = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=2
    )
    batched = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=2
    )
    padded = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=2
    )
    scalar.load_state_dict(behavior.state_dict())
    batched.load_state_dict(behavior.state_dict())
    padded.load_state_dict(behavior.state_dict())
    scalar_optimizer = torch.optim.SGD(scalar.parameters(), lr=1e-4)
    batched_optimizer = torch.optim.SGD(batched.parameters(), lr=1e-4)
    padded_optimizer = torch.optim.SGD(padded.parameters(), lr=1e-4)
    scalar_metrics = ppo_update(
        scalar, episodes, scalar_optimizer
    )
    batched_metrics = ppo_update_batched(
        batched, episodes, batched_optimizer
    )
    padded_metrics = ppo_update_batched(
        padded,
        episodes,
        padded_optimizer,
        padded_batch_size=4,
        power_of_two_sequence_padding=True,
    )

    assert batched_metrics["samples"] == scalar_metrics["samples"]
    for name in (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "old_approx_kl",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
    ):
        assert batched_metrics[name] == pytest.approx(
            scalar_metrics[name], abs=2e-5
        )
        assert padded_metrics[name] == pytest.approx(
            batched_metrics[name], abs=2e-5
        )
    for scalar_parameter, batched_parameter, padded_parameter in zip(
        scalar.parameters(), batched.parameters(), padded.parameters()
    ):
        assert torch.allclose(
            scalar_parameter, batched_parameter, atol=2e-5, rtol=2e-5
        )
        assert torch.allclose(
            batched_parameter, padded_parameter, atol=2e-5, rtol=2e-5
        )
    scalar_drift = evaluate_ppo_drift(scalar, episodes)
    batched_drift = evaluate_ppo_drift_batched(scalar, episodes)
    chunked_drift = evaluate_ppo_drift_batched_minibatches(
        scalar, episodes, minibatch_trajectories=1
    )
    padded_drift = evaluate_ppo_drift_batched(
        scalar,
        episodes,
        padded_batch_size=4,
        power_of_two_sequence_padding=True,
    )
    assert batched_drift["samples"] == scalar_drift["samples"]
    for name in (
        "approx_kl",
        "old_approx_kl",
        "clip_fraction",
        "entropy",
        "value_mse",
        "explained_variance",
    ):
        assert batched_drift[name] == pytest.approx(
            scalar_drift[name], abs=2e-5
        )
        assert chunked_drift[name] == pytest.approx(
            batched_drift[name], abs=2e-5
        )
        assert padded_drift[name] == pytest.approx(
            batched_drift[name], abs=2e-5
        )


def test_persistent_prepared_replay_matches_one_batched_epoch():
    torch.manual_seed(29)
    behavior = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=2, mode_conditioned_heads=True
    )
    episodes, collection = collect_smoke_episodes(
        behavior, seed=29, episode_count=3, max_turns=80
    )
    assert collection["decisions"] > 0
    ordinary = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=2, mode_conditioned_heads=True
    )
    persistent = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=2, mode_conditioned_heads=True
    )
    worker_packed = NativePufferTemporalPolicy(
        hidden_size=16, num_layers=2, mode_conditioned_heads=True
    )
    ordinary.load_state_dict(behavior.state_dict())
    persistent.load_state_dict(behavior.state_dict())
    worker_packed.load_state_dict(behavior.state_dict())
    ordinary_optimizer = torch.optim.SGD(ordinary.parameters(), lr=1e-4)
    persistent_optimizer = torch.optim.SGD(
        persistent.parameters(), lr=1e-4
    )
    worker_packed_optimizer = torch.optim.SGD(
        worker_packed.parameters(), lr=1e-4
    )
    arguments = {
        "update_epochs": 1,
        "minibatch_trajectories": 4,
        "seed": 8029,
        "target_kl": None,
        "gae_lambda": 0.95,
        "normalize_advantages": True,
        "static_compile_padding": True,
        "length_bucketed_minibatches": True,
    }
    ordinary_metrics = ppo_update_batched_epochs(
        ordinary, episodes, ordinary_optimizer, **arguments
    )
    persistent_metrics = ppo_update_batched_epochs(
        persistent,
        episodes,
        persistent_optimizer,
        persistent_prepared_batches=True,
        **arguments,
    )
    packed_episodes = [
        pack_puffer_episode(behavior, episode) for episode in episodes
    ]
    worker_packed_metrics = ppo_update_batched_epochs(
        worker_packed,
        packed_episodes,
        worker_packed_optimizer,
        persistent_prepared_batches=True,
        **arguments,
    )
    assert persistent_metrics["persistent_prepared_batches"] is True
    assert persistent_metrics["samples"] == ordinary_metrics["samples"]
    assert persistent_metrics["optimizer_steps"] == ordinary_metrics[
        "optimizer_steps"
    ]
    for name in (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "old_approx_kl",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
    ):
        assert persistent_metrics[name] == pytest.approx(
            ordinary_metrics[name], abs=2e-5
        )
        assert worker_packed_metrics[name] == pytest.approx(
            persistent_metrics[name], abs=2e-5
        )
    for ordinary_parameter, persistent_parameter, packed_parameter in zip(
        ordinary.parameters(), persistent.parameters(), worker_packed.parameters()
    ):
        assert torch.allclose(
            ordinary_parameter,
            persistent_parameter,
            atol=2e-5,
            rtol=2e-5,
        )
        assert torch.allclose(
            persistent_parameter,
            packed_parameter,
            atol=2e-5,
            rtol=2e-5,
        )


def test_selfplay_seats_have_isolated_observation_and_memory():
    torch.manual_seed(23)
    policy = NativePufferTemporalPolicy(hidden_size=16)
    current_agent = temporal_agent_factory(
        policy, deterministic=True, seed=23
    )
    match = game.Game([current_agent, current_agent])
    match.reset()
    seat_zero, seat_one = match.players
    assert seat_zero.state[0].data_ptr() != seat_one.state[0].data_ptr()

    match.simulate_turn()
    assert state_bytes(seat_zero.state) != state_bytes(seat_one.state)
    assert seat_zero.safe_history
    assert not seat_one.safe_history

    match.simulate_turn()
    assert seat_one.safe_history
    assert seat_zero.state[0].data_ptr() != seat_one.state[0].data_ptr()
    first_zero = seat_zero.safe_history[0].observation.values[:50]
    first_one = seat_one.safe_history[0].observation.values[:50]
    assert first_zero != first_one
    for seat in (seat_zero, seat_one):
        serialized = serialize_safe_history(seat.safe_history)
        assert b"drawn_card" not in serialized
        assert b"opponent_hand" not in serialized
