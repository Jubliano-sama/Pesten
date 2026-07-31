import copy
import inspect
import math
import random

import numpy as np
import pytest
import torch

import ppo2 as ppo2_module
from ai_observation import HAND_END, OBSERVATION_SIZE
from anchored_ppo import (
    anchored_iteration,
    enforce_initial_ratio_invariant,
    full_policy_kl,
    monte_carlo_returns,
    own_hand_potential,
    potential_shaped_rewards,
)
from ppo2 import PPO_Supervised


def test_full_policy_kl_is_zero_for_identical_masked_policies():
    logits = torch.tensor([[1.0, 2.0, -3.0], [0.2, -0.1, 4.0]])
    legal = torch.tensor([[True, True, False], [True, False, True]])
    assert torch.allclose(
        full_policy_kl(logits, logits.clone(), legal),
        torch.tensor(0.0),
        atol=1e-7,
    )


def test_past_policy_probability_default_remains_backward_compatible():
    parameter = inspect.signature(PPO_Supervised).parameters[
        "past_policy_probability"
    ]
    assert parameter.default == pytest.approx(0.2)


def test_full_policy_kl_ignores_illegal_logits_and_is_positive_on_drift():
    reference = torch.tensor([[2.0, 0.0, 1000.0]])
    candidate = torch.tensor([[0.0, 2.0, -1000.0]])
    legal = torch.tensor([[True, True, False]])
    changed_illegal = reference.clone()
    changed_illegal[0, 2] = -1000.0
    assert torch.allclose(
        full_policy_kl(reference, changed_illegal, legal),
        torch.tensor(0.0),
        atol=1e-7,
    )
    assert full_policy_kl(reference, candidate, legal) > 0


def test_monte_carlo_returns_respect_episode_boundaries():
    rewards = torch.tensor([0.0, 1.0, 2.0, 3.0])
    returns = monte_carlo_returns(rewards, [2, 2], gamma=0.5)
    assert torch.allclose(returns, torch.tensor([0.5, 1.0, 3.5, 3.0]))


def observations_with_hand_sizes(hand_sizes):
    observations = torch.zeros(len(hand_sizes), OBSERVATION_SIZE)
    observations[:, 0] = torch.tensor(hand_sizes)
    return observations


def test_potential_shaping_exact_math_and_terminal_episode_boundaries():
    observations = observations_with_hand_sizes([7, 6, 5, 4, 3])
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    shaped, metrics = potential_shaped_rewards(
        rewards,
        observations,
        [2, 3],
        gamma=0.5,
        coefficient=0.7,
    )
    shaping = shaped - rewards

    assert torch.allclose(
        shaping,
        torch.tensor([0.4, 0.6, 0.3, 0.25, 0.3]),
        atol=1e-7,
    )
    # Discounted shaping telescopes to -coefficient * Phi(initial) in
    # each episode because terminal Phi is exactly zero.
    assert shaping[0] + 0.5 * shaping[1] == pytest.approx(0.7)
    assert (
        shaping[2] + 0.5 * shaping[3] + 0.25 * shaping[4]
    ) == pytest.approx(0.5)
    assert metrics["coefficient"] == pytest.approx(0.7)
    assert metrics["max_abs_shaping_reward"] == pytest.approx(0.6)


def test_zero_potential_coefficient_preserves_rewards_exactly():
    rewards = torch.tensor([-1.0, 0.0, 1.0])
    shaped, metrics = potential_shaped_rewards(
        rewards,
        observations_with_hand_sizes([7, 6, 5]),
        [3],
        gamma=0.97,
        coefficient=0.0,
    )
    assert shaped is rewards
    assert metrics["shaping_reward_mean"] == 0.0
    assert metrics["max_abs_shaping_reward"] == 0.0


def test_own_hand_potential_has_no_private_or_context_dependency():
    observations = observations_with_hand_sizes([7, 3])
    changed_context = observations.clone()
    changed_context[:, HAND_END:] = torch.arange(
        OBSERVATION_SIZE - HAND_END,
        dtype=torch.float,
    )

    assert torch.equal(
        own_hand_potential(observations),
        own_hand_potential(changed_context),
    )
    assert torch.allclose(
        own_hand_potential(observations),
        torch.tensor([-1.0, -3.0 / 7.0]),
    )


def test_initial_ratio_invariant_records_small_error_and_rejects_mismatch():
    assert enforce_initial_ratio_invariant(
        torch.tensor([1.0, 0.995, 1.01])
    ) == pytest.approx(0.01)
    with pytest.raises(
        RuntimeError,
        match=r"unchanged-policy ratio error reached 0\.10000",
    ):
        enforce_initial_ratio_invariant(torch.tensor([1.0, 1.1]))


class TinyPPO:
    def __init__(self, *, old_log_prob_offset: float = 0.0):
        self.actor_device = torch.device("cpu")
        self.device = torch.device("cpu")
        self.actor = torch.nn.Linear(OBSERVATION_SIZE, 3)
        self.critic = torch.nn.Linear(OBSERVATION_SIZE, 1)
        self.actor_optim = torch.optim.SGD(self.actor.parameters(), lr=0.01)
        self.critic_optim = torch.optim.SGD(self.critic.parameters(), lr=0.01)
        self.clip = 0.1
        self.old_log_prob_offset = old_log_prob_offset
        self.last_rollout_metrics = {"policy_decisions": 4}
        self.sync_calls = 0

    def generate_data(self, **_kwargs):
        observations = observations_with_hand_sizes([7, 6, 5, 4])
        observations[:, HAND_END] = torch.tensor(
            [0.0, 1.0, -1.0, 2.0]
        )
        actions = torch.tensor([0, 1, 2, 1])
        with torch.inference_mode():
            distribution = torch.distributions.Categorical(
                logits=self.actor(observations) / 0.8
            )
            old_log_probs = (
                distribution.log_prob(actions) + self.old_log_prob_offset
            )
        rewards = torch.tensor([0.0, 0.5, -0.25, 1.0])
        return observations, actions, old_log_probs, rewards, [4]

    def get_action_masks(self, observations):
        return torch.ones(
            len(observations), 3, dtype=torch.bool
        )

    def _sync_rollout_actor(self):
        self.sync_calls += 1


def run_tiny_anchored_iteration(
    ppo, *, potential_shaping_coefficient=0.0
):
    return anchored_iteration(
        ppo,
        copy.deepcopy(ppo.actor),
        episodes=1,
        update_epochs=1,
        batch_size=2,
        temperature=0.8,
        agent_sets=[["rl", "smart"]],
        kl_coefficient=1.0,
        entropy_coefficient=0.0,
        target_kl=1.0,
        potential_shaping_coefficient=potential_shaping_coefficient,
    )


def test_anchored_iteration_records_first_optimizer_ratio_error():
    ppo = TinyPPO()
    metrics = run_tiny_anchored_iteration(
        ppo, potential_shaping_coefficient=0.1
    )
    assert metrics["max_initial_ratio_error"] < 1e-6
    assert metrics["rollout"]["policy_decisions"] == 4
    assert metrics["potential_shaping"]["coefficient"] == pytest.approx(0.1)
    assert metrics["potential_shaping"]["max_abs_shaping_reward"] > 0
    assert ppo.sync_calls == 1


def test_anchored_iteration_rejects_rollout_policy_mismatch_before_update():
    ppo = TinyPPO(old_log_prob_offset=math.log(2.0))
    actor_before = copy.deepcopy(ppo.actor.state_dict())
    critic_before = copy.deepcopy(ppo.critic.state_dict())

    with pytest.raises(
        RuntimeError,
        match=r"rollout/update policy mismatch",
    ):
        run_tiny_anchored_iteration(ppo)

    assert all(
        torch.equal(actor_before[key], value)
        for key, value in ppo.actor.state_dict().items()
    )
    assert all(
        torch.equal(critic_before[key], value)
        for key, value in ppo.critic.state_dict().items()
    )


def checkpoint_rollout(
    tmp_path,
    monkeypatch,
    *,
    probability,
    seed,
    games=4,
):
    monkeypatch.chdir(tmp_path)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    trainer = PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device="cpu",
        training_actor_device="cpu",
        rollout_batch_size=games,
        compile_updates=False,
        past_policy_probability=probability,
    )
    torch.save(trainer.actor.state_dict(), tmp_path / "ppo_actor_0.pth")
    trainer.saved_checkpoints = [0]

    captured = []
    real_collect = ppo2_module.collect_batched_rollout

    def inspect_collect(engine, actor_policies, **kwargs):
        captured.append(
            {
                "actor_keys": engine.actor_keys.copy(),
                "collect_data": {
                    key: policy.collect_data
                    for key, policy in actor_policies.items()
                },
            }
        )
        return real_collect(engine, actor_policies, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(
            ppo2_module, "collect_batched_rollout", inspect_collect
        )
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            _, actions, _, _, lengths = trainer.generate_data(
                num_episodes=games,
                agent_sets=[["self", "self"]],
            )
        finally:
            torch.set_num_threads(previous_threads)
    return trainer, captured, actions, lengths


def test_zero_past_probability_assigns_only_collecting_current_actor(
    tmp_path, monkeypatch
):
    trainer, captured, actions, lengths = checkpoint_rollout(
        tmp_path,
        monkeypatch,
        probability=0.0,
        seed=71_001,
    )

    assert len(captured) == 1
    assert captured[0]["collect_data"] == {"current": True}
    assert set(captured[0]["actor_keys"].flat) == {"current"}
    assert len(lengths) == 8
    assert sum(lengths) == len(actions)
    assert trainer.last_rollout_metrics["past_policy_games"] == 0
    assert trainer.last_rollout_metrics["actor_key_counts"] == {
        "current": 8
    }


def test_unit_past_probability_uses_one_frozen_checkpoint_zero_per_game(
    tmp_path, monkeypatch
):
    trainer, captured, actions, lengths = checkpoint_rollout(
        tmp_path,
        monkeypatch,
        probability=1.0,
        seed=71_002,
    )

    assert len(captured) == 1
    assignment = captured[0]
    assert assignment["collect_data"] == {
        "current": True,
        "checkpoint:0": False,
    }
    assert all(
        list(row).count("current") == 1
        and list(row).count("checkpoint:0") == 1
        for row in assignment["actor_keys"]
    )
    # Only the current-policy seat contributes one trajectory per game.
    assert len(lengths) == 4
    assert sum(lengths) == len(actions)
    assert trainer.last_rollout_metrics["past_policy_games"] == 4
    assert trainer.last_rollout_metrics["actor_key_counts"] == {
        "current": 4,
        "checkpoint:0": 4,
    }
    assert all(
        not parameter.requires_grad
        for parameter in trainer._checkpoint_actor(0).parameters()
    )


def test_checkpoint_zero_actor_key_layout_is_deterministic_under_seed(
    tmp_path, monkeypatch
):
    first, first_capture, _, _ = checkpoint_rollout(
        tmp_path,
        monkeypatch,
        probability=1.0,
        seed=71_003,
        games=8,
    )
    second, second_capture, _, _ = checkpoint_rollout(
        tmp_path,
        monkeypatch,
        probability=1.0,
        seed=71_003,
        games=8,
    )

    assert np.array_equal(
        first_capture[0]["actor_keys"],
        second_capture[0]["actor_keys"],
    )
    assert (
        first.last_rollout_metrics["actor_key_layout_sha256"]
        == second.last_rollout_metrics["actor_key_layout_sha256"]
    )


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
def test_past_policy_probability_must_be_a_probability(probability):
    with pytest.raises(
        ValueError, match="past_policy_probability must be in"
    ):
        PPO_Supervised(
            use_pretrained=False,
            old_save=(),
            rollout_device="cpu",
            training_actor_device="cpu",
            compile_updates=False,
            past_policy_probability=probability,
        )
