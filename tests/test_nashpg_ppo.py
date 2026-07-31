import copy

import numpy as np
import pytest
import torch

from ai_observation import OBSERVATION_SIZE, PASS_ACTION
from nashpg_ppo import (
    SELF_PLAY_AGENT_SETS,
    assert_two_player_zero_sum,
    clipped_nashpg_objective,
    generate_audited_self_play_data,
    masked_current_reference_kl,
    module_sha256,
    nashpg_inner_iteration,
    refresh_reference,
    terminal_zero_sum_rewards,
    validate_development_seed,
    validate_development_seed_plan,
    validate_two_player_self_play,
)
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def test_masked_kl_is_current_to_reference_and_ignores_illegal_logits():
    current = torch.tensor([[0.0, 1.0, 10_000.0]], requires_grad=True)
    reference = torch.tensor([[2.0, -1.0, -10_000.0]])
    legal = torch.tensor([[True, True, False]])
    actual = masked_current_reference_kl(current, reference, legal)
    current_log = torch.log_softmax(current[:, :2], dim=-1)
    reference_log = torch.log_softmax(reference[:, :2], dim=-1)
    expected = (
        current_log.exp() * (current_log - reference_log)
    ).sum(dim=-1).mean()
    reverse = (
        reference_log.exp() * (reference_log - current_log)
    ).sum(dim=-1).mean()

    assert torch.equal(actual, expected)
    assert not torch.isclose(actual, reverse)
    actual.backward()
    assert current.grad is not None
    assert current.grad[0, 2] == 0


def test_identical_policy_has_exact_zero_masked_kl():
    logits = torch.tensor([[1000.0, -1000.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    assert masked_current_reference_kl(logits, logits, mask).item() == 0.0


def test_masked_kl_matches_rowwise_legal_distributions_under_fuzz():
    generator = torch.Generator().manual_seed(884_201)
    current = torch.randn(32, 55, generator=generator, dtype=torch.float64) * 50
    reference = (
        torch.randn(32, 55, generator=generator, dtype=torch.float64) * 50
    )
    legal = torch.rand(32, 55, generator=generator) < 0.2
    legal[:, 54] = True

    # Illegal logits can be arbitrarily extreme without affecting the result.
    current = current.masked_fill(~legal, 1e200)
    reference = reference.masked_fill(~legal, -1e200)
    actual = masked_current_reference_kl(current, reference, legal)
    expected_rows = []
    for row in range(len(current)):
        current_legal = current[row, legal[row]]
        reference_legal = reference[row, legal[row]]
        current_log = torch.log_softmax(current_legal, dim=-1)
        reference_log = torch.log_softmax(reference_legal, dim=-1)
        expected_rows.append(
            (current_log.exp() * (current_log - reference_log)).sum()
        )
    expected = torch.stack(expected_rows).mean()

    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    assert torch.isfinite(actual)


def test_masked_kl_rejects_missing_or_non_boolean_legal_actions():
    logits = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="boolean"):
        masked_current_reference_kl(
            logits,
            logits,
            torch.ones(2, 3),
        )
    with pytest.raises(ValueError, match="legal action"):
        masked_current_reference_kl(
            logits,
            logits,
            torch.tensor([[True, False, False], [False, False, False]]),
        )


def test_alpha_zero_is_exactly_the_ppo_clipped_surrogate():
    ratios = torch.tensor([0.7, 1.0, 1.3], requires_grad=True)
    advantages = torch.tensor([1.0, -2.0, 3.0])
    kl = torch.tensor(float("nan"))
    objective, surrogate = clipped_nashpg_objective(
        ratios,
        advantages,
        clip=0.1,
        alpha=0.0,
        reference_kl=kl,
    )
    assert torch.equal(objective, surrogate)
    assert objective is surrogate
    objective.backward()
    actual_gradient = ratios.grad.detach().clone()

    baseline_ratios = ratios.detach().clone().requires_grad_(True)
    baseline = torch.minimum(
        baseline_ratios * advantages,
        torch.clamp(baseline_ratios, 0.9, 1.1) * advantages,
    ).mean()
    baseline.backward()
    assert torch.equal(actual_gradient, baseline_ratios.grad)


def test_terminal_reward_conversion_restores_two_player_zero_sum():
    actions = torch.tensor([PASS_ACTION, 1, 2, PASS_ACTION])
    shaped = torch.tensor([-0.1, 1.0, 0.0, -1.1])
    terminal = terminal_zero_sum_rewards(actions, shaped)
    assert torch.allclose(terminal, torch.tensor([0.0, 1.0, 0.0, -1.0]))
    full = assert_two_player_zero_sum(terminal, [2, 2], expected_games=1)
    assert full == {
        "populated_episodes": 2,
        "implicit_zero_decision_seats": 0,
        "full_pair_zero_sum_verified": True,
        "terminal_payoffs_verified": True,
    }

    with pytest.raises(RuntimeError, match="not two-player zero-sum"):
        assert_two_player_zero_sum(
            torch.tensor([0.0, 1.0, 0.0, 0.0]),
            [2, 2],
            expected_games=1,
        )
    partial = assert_two_player_zero_sum(
        torch.tensor([0.0, 1.0]),
        [2],
        expected_games=1,
    )
    assert partial["implicit_zero_decision_seats"] == 1
    assert not partial["full_pair_zero_sum_verified"]
    # Once one seat is omitted, later flattened episodes shift left and must
    # not be paired by adjacency: [g0s0, g1s0, g1s1].
    shifted = assert_two_player_zero_sum(
        torch.tensor([1.0, 1.0, -1.0]),
        [1, 1, 1],
        expected_games=2,
    )
    assert shifted["implicit_zero_decision_seats"] == 1
    assert not shifted["full_pair_zero_sum_verified"]
    with pytest.raises(RuntimeError, match="terminal win/draw/loss"):
        assert_two_player_zero_sum(torch.tensor([0.5]), [1], expected_games=1)
    with pytest.raises(ValueError, match="one or two populated seats"):
        assert_two_player_zero_sum(
            torch.tensor([1.0]),
            [1],
            expected_games=2,
        )
    with pytest.raises(ValueError, match="populated"):
        assert_two_player_zero_sum(
            torch.tensor([1.0, -1.0]),
            [0, 2],
            expected_games=1,
        )
    with pytest.raises(ValueError, match="populated"):
        assert_two_player_zero_sum(
            torch.tensor([0.0, 0.0]),
            [0, 2],
            expected_games=1,
        )


class TinyPPO:
    def __init__(self):
        self.actor_device = torch.device("cpu")
        self.device = torch.device("cpu")
        self.actor = torch.nn.Linear(OBSERVATION_SIZE, 3)
        self.critic = torch.nn.Linear(OBSERVATION_SIZE, 1)
        self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=0.02)
        self.critic_optim = torch.optim.SGD(
            self.critic.parameters(), lr=0.02
        )
        self.clip = 0.1
        self.past_policy_probability = 0.0
        self.last_rollout_metrics = {
            "completed_games": 2,
            "policy_decisions": 8,
        }
        self.sync_calls = 0

    def generate_data(
        self,
        *,
        temperature,
        agent_sets,
        num_episodes,
        **_kwargs,
    ):
        assert agent_sets == SELF_PLAY_AGENT_SETS
        np.random.randint(
            0,
            DEVELOPMENT_SEED_LIMIT,
            size=num_episodes,
            dtype=np.uint32,
        )
        observations = torch.zeros(8, OBSERVATION_SIZE)
        observations[:, 0] = torch.arange(1, 9, dtype=torch.float)
        observations[:, 100:103] = torch.tensor(
            [
                [0.0, 1.0, -1.0],
                [1.0, 0.0, 2.0],
                [-1.0, 2.0, 0.0],
                [0.5, -0.5, 1.5],
                [1.5, 0.5, -0.5],
                [-0.5, 1.5, 0.5],
                [0.2, -0.1, 0.4],
                [-0.4, 0.1, 0.2],
            ]
        )
        actions = torch.tensor([0, 1, 2, 1, 1, 2, 0, 2])
        with torch.inference_mode():
            distribution = torch.distributions.Categorical(
                logits=self.actor(observations) / temperature
            )
            old_log_probs = distribution.log_prob(actions)
        # Adjacent episodes are the two seats of one game.
        rewards = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0])
        return observations, actions, old_log_probs, rewards, [2, 2, 2, 2]

    def get_action_masks(self, observations):
        return torch.ones(len(observations), 3, dtype=torch.bool)

    def _sync_rollout_actor(self):
        self.actor.eval()
        self.sync_calls += 1


class TinyPartialSeatPPO(TinyPPO):
    def generate_data(self, *, temperature, agent_sets, **kwargs):
        observations, actions, _, rewards, _ = super().generate_data(
            temperature=temperature,
            agent_sets=agent_sets,
            **kwargs,
        )
        observations = observations[:2]
        actions = actions[:2]
        rewards = rewards[:2]
        with torch.inference_mode():
            old_log_probs = torch.distributions.Categorical(
                logits=self.actor(observations) / temperature
            ).log_prob(actions)
        self.last_rollout_metrics["completed_games"] = 1
        self.last_rollout_metrics["policy_decisions"] = 2
        return observations, actions, old_log_probs, rewards, [2]


def test_audited_rollout_records_exact_deal_seeds_and_rng_state():
    np.random.seed(912_304)
    before = np.random.get_state()
    replay = np.random.RandomState()
    replay.set_state(before)
    expected = replay.randint(
        0,
        DEVELOPMENT_SEED_LIMIT,
        size=2,
        dtype=np.uint32,
    )
    expected_after = replay.get_state()

    _, audit = generate_audited_self_play_data(
        TinyPPO(),
        episodes=2,
        temperature=0.8,
        agent_sets=SELF_PLAY_AGENT_SETS,
    )

    assert audit["deal_seeds"] == [
        {"deal_seed": int(seed)}
        for seed in expected
    ]
    assert audit["deal_seed_rng_replay_verified"]
    actual_after = np.random.get_state()
    assert actual_after[0] == expected_after[0]
    assert np.array_equal(actual_after[1], expected_after[1])
    assert actual_after[2:] == expected_after[2:]


def test_audited_rollout_rejects_unaccounted_numpy_rng_draw():
    class ExtraDrawPPO(TinyPPO):
        def generate_data(self, **kwargs):
            result = super().generate_data(**kwargs)
            np.random.randint(0, DEVELOPMENT_SEED_LIMIT, dtype=np.uint32)
            return result

    with pytest.raises(RuntimeError, match="RNG consumption"):
        generate_audited_self_play_data(
            ExtraDrawPPO(),
            episodes=2,
            temperature=0.8,
            agent_sets=SELF_PLAY_AGENT_SETS,
        )


def test_partial_seat_iteration_distinguishes_structural_from_paired_proof():
    torch.manual_seed(4321)
    ppo = TinyPartialSeatPPO()
    reference = copy.deepcopy(ppo.actor).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    result = nashpg_inner_iteration(
        ppo,
        reference,
        alpha=0.2,
        episodes=1,
        update_epochs=1,
        batch_size=2,
        temperature=0.8,
        agent_sets=SELF_PLAY_AGENT_SETS,
        permutation_seed=80,
    )

    assert result["structural_two_player_zero_sum"]
    assert not result["zero_sum_verified"]
    assert result["payoff_audit"]["implicit_zero_decision_seats"] == 1
    assert not result["payoff_audit"]["full_pair_zero_sum_verified"]


def test_reference_stays_frozen_for_inner_steps_then_refreshes():
    torch.manual_seed(1234)
    ppo = TinyPPO()
    reference = copy.deepcopy(ppo.actor).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reference_before = module_sha256(reference)
    actor_before = module_sha256(ppo.actor)

    first = nashpg_inner_iteration(
        ppo,
        reference,
        alpha=0.2,
        episodes=2,
        update_epochs=1,
        batch_size=4,
        temperature=0.8,
        agent_sets=SELF_PLAY_AGENT_SETS,
        permutation_seed=77,
    )
    assert first["initial_reference_kl_current_to_reference"] == 0.0
    assert first["zero_sum_verified"]
    assert module_sha256(reference) == reference_before
    assert module_sha256(ppo.actor) != actor_before
    assert all(parameter.grad is None for parameter in reference.parameters())

    nashpg_inner_iteration(
        ppo,
        reference,
        alpha=0.2,
        episodes=2,
        update_epochs=1,
        batch_size=4,
        temperature=0.8,
        agent_sets=SELF_PLAY_AGENT_SETS,
        permutation_seed=78,
    )
    assert module_sha256(reference) == reference_before
    refreshed = refresh_reference(reference, ppo.actor)
    assert refreshed == module_sha256(ppo.actor)
    assert all(not parameter.requires_grad for parameter in reference.parameters())
    third = nashpg_inner_iteration(
        ppo,
        reference,
        alpha=0.2,
        episodes=2,
        update_epochs=1,
        batch_size=4,
        temperature=0.8,
        agent_sets=SELF_PLAY_AGENT_SETS,
        permutation_seed=79,
    )
    assert third["initial_reference_kl_current_to_reference"] == 0.0
    assert all(parameter.grad is None for parameter in reference.parameters())
    assert ppo.sync_calls == 3


def test_pilot_restrictions_reject_non_self_play_and_sealed_seed():
    validate_two_player_self_play([["SELF", "self"]])
    for invalid in (
        [["smart", "rl"]],
        [["self", "self", "self"]],
        [["self"], ["self"]],
    ):
        with pytest.raises(ValueError, match="two-player self-play"):
            validate_two_player_self_play(invalid)
    validate_development_seed(DEVELOPMENT_SEED_LIMIT - 1)
    with pytest.raises(ValueError, match="development"):
        validate_development_seed(DEVELOPMENT_SEED_LIMIT)
    validate_development_seed_plan(
        DEVELOPMENT_SEED_LIMIT - 10_001,
        outer_rounds=3,
        inner_iterations=3,
    )
    with pytest.raises(ValueError, match="development"):
        validate_development_seed_plan(
            DEVELOPMENT_SEED_LIMIT - 10_000,
            outer_rounds=3,
            inner_iterations=3,
        )
    with pytest.raises(ValueError, match="development"):
        validate_development_seed_plan(
            DEVELOPMENT_SEED_LIMIT - 10_001,
            outer_rounds=12,
            inner_iterations=3,
        )


@pytest.mark.parametrize("alpha", [-0.1, float("nan")])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        clipped_nashpg_objective(
            torch.ones(1),
            torch.ones(1),
            clip=0.1,
            alpha=alpha,
            reference_kl=torch.zeros(()),
        )
