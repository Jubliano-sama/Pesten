import copy

import pytest
import torch

from ai_observation import OBSERVATION_SIZE
from qboosted_ppo import (
    PublicObservationQCritic,
    build_argument_parser,
    expected_action_values,
    fit_q_critic,
    module_sha256,
    qboost_estimate,
    qboosted_anchored_iteration,
    validate_development_seed,
    validate_public_observations,
)
from ppo2 import DEFAULT_TORCH_THREADS


def test_qboost_estimator_matches_hand_computed_expected_sarsa_trace():
    estimate = qboost_estimate(
        rewards=torch.tensor([0.0, 1.0]),
        q_values=torch.tensor([[1.0, 3.0], [2.0, 4.0]]),
        actions=torch.tensor([0, 1]),
        policy_logits=torch.zeros(2, 2),
        legal_actions=torch.ones(2, 2, dtype=torch.bool),
        episode_lengths=[2],
        gamma=0.9,
        trace_lambda=0.5,
    )

    assert torch.allclose(
        estimate.expected_values, torch.tensor([2.0, 3.0])
    )
    assert torch.allclose(
        estimate.expected_sarsa_residuals,
        torch.tensor([1.7, -3.0]),
    )
    assert torch.allclose(
        estimate.advantages, torch.tensor([-0.65, -2.0])
    )
    assert torch.allclose(
        estimate.q_targets, torch.tensor([1.35, 1.0])
    )


def test_exact_expected_sarsa_q_has_zero_trace_residual():
    estimate = qboost_estimate(
        rewards=torch.tensor([0.2, 1.0]),
        q_values=torch.tensor([[2.0, 4.0], [1.0, 3.0]]),
        actions=torch.tensor([0, 0]),
        policy_logits=torch.zeros(2, 2),
        legal_actions=torch.ones(2, 2, dtype=torch.bool),
        episode_lengths=[2],
        gamma=0.9,
        trace_lambda=0.95,
    )

    assert torch.allclose(
        estimate.expected_sarsa_residuals,
        torch.zeros(2),
        atol=1e-7,
    )
    assert torch.allclose(estimate.advantages, torch.tensor([-1.0, -1.0]))
    assert torch.allclose(estimate.q_targets, torch.tensor([2.0, 1.0]))


def test_qboost_trace_never_crosses_episode_boundaries():
    arguments = {
        "rewards": torch.tensor([1.0, 2.0]),
        "q_values": torch.zeros(2, 2),
        "actions": torch.tensor([0, 0]),
        "policy_logits": torch.zeros(2, 2),
        "legal_actions": torch.ones(2, 2, dtype=torch.bool),
        "gamma": 1.0,
        "trace_lambda": 1.0,
    }
    separated = qboost_estimate(
        **arguments, episode_lengths=[1, 1]
    )
    joined = qboost_estimate(**arguments, episode_lengths=[2])

    assert torch.equal(separated.q_targets, torch.tensor([1.0, 2.0]))
    assert torch.equal(joined.q_targets, torch.tensor([3.0, 2.0]))


def test_policy_expectation_excludes_illegal_actions():
    values = expected_action_values(
        torch.tensor([[1.0, 10_000.0, 3.0]]),
        torch.tensor([[0.0, 10_000.0, 0.0]]),
        torch.tensor([[True, False, True]]),
    )
    assert values.item() == pytest.approx(2.0)


def test_qboost_estimator_rejects_an_illegal_sampled_action():
    with pytest.raises(ValueError, match="sampled action is not legal"):
        qboost_estimate(
            rewards=torch.tensor([0.0]),
            q_values=torch.tensor([[1.0, 2.0]]),
            actions=torch.tensor([1]),
            policy_logits=torch.zeros(1, 2),
            legal_actions=torch.tensor([[True, False]]),
            episode_lengths=[1],
        )


def test_qcritic_contract_accepts_only_existing_human_visible_observation():
    critic = PublicObservationQCritic()
    observations = torch.zeros(2, OBSERVATION_SIZE)
    validate_public_observations(observations)
    assert critic(observations).shape == (2, 55)

    with pytest.raises(ValueError, match="human-visible"):
        validate_public_observations(
            torch.zeros(2, OBSERVATION_SIZE + 1)
        )


class TinyPPO:
    def __init__(self):
        self.actor_device = torch.device("cpu")
        self.actor = torch.nn.Linear(OBSERVATION_SIZE, 3)
        self.actor_optim = torch.optim.SGD(
            self.actor.parameters(), lr=0.02
        )
        self.clip = 0.1
        self.last_rollout_metrics = {"policy_decisions": 6}
        self.sync_calls = 0

    def generate_data(self, *, temperature, **_kwargs):
        observations = torch.zeros(6, OBSERVATION_SIZE)
        observations[:, 0] = torch.tensor(
            [7.0, 6.0, 5.0, 7.0, 5.0, 3.0]
        )
        observations[:, 100:103] = torch.tensor(
            [
                [0.0, 1.0, -1.0],
                [1.0, 0.0, 2.0],
                [-1.0, 2.0, 0.0],
                [0.5, -0.5, 1.5],
                [1.5, 0.5, -0.5],
                [-0.5, 1.5, 0.5],
            ]
        )
        actions = torch.tensor([0, 1, 2, 2, 1, 0])
        with torch.inference_mode():
            distribution = torch.distributions.Categorical(
                logits=self.actor(observations) / temperature
            )
            old_log_probs = distribution.log_prob(actions)
        rewards = torch.tensor([0.0, 0.2, 1.0, 0.0, -0.1, -1.0])
        return observations, actions, old_log_probs, rewards, [3, 3]

    def get_action_masks(self, observations):
        return torch.ones(len(observations), 3, dtype=torch.bool)

    def _sync_rollout_actor(self):
        self.sync_calls += 1


def test_qboosted_iteration_updates_actor_and_qcritic_on_on_policy_data():
    torch.manual_seed(27_601)
    ppo = TinyPPO()
    reference = copy.deepcopy(ppo.actor).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    q_critic = torch.nn.Linear(OBSERVATION_SIZE, 3)
    q_optimizer = torch.optim.SGD(q_critic.parameters(), lr=0.02)
    actor_before = {
        name: value.detach().clone()
        for name, value in ppo.actor.state_dict().items()
    }
    critic_before = {
        name: value.detach().clone()
        for name, value in q_critic.state_dict().items()
    }

    metrics = qboosted_anchored_iteration(
        ppo,
        reference,
        q_critic,
        q_optimizer,
        episodes=2,
        actor_epochs=1,
        critic_epochs=2,
        batch_size=3,
        temperature=0.8,
        agent_sets=[["self", "self"]],
        kl_coefficient=0.1,
        entropy_coefficient=0.0,
        target_kl=1.0,
        seed=27_602,
    )

    assert metrics["algorithm"] == "information_safe_qboosted_ppo"
    assert not metrics["paper_exact_vrpo"]
    assert metrics["max_initial_ratio_error"] < 1e-6
    assert metrics["decisions"] == 6
    assert len(metrics["actor_epochs"]) == 1
    assert len(metrics["critic_epochs"]) == 2
    assert ppo.sync_calls == 1
    assert any(
        not torch.equal(actor_before[name], value)
        for name, value in ppo.actor.state_dict().items()
    )
    assert any(
        not torch.equal(critic_before[name], value)
        for name, value in q_critic.state_dict().items()
    )
    assert all(not parameter.requires_grad for parameter in reference.parameters())


def test_qcritic_warmup_updates_only_qcritic_and_records_actor_hash():
    torch.manual_seed(27_610)
    ppo = TinyPPO()
    q_critic = torch.nn.Linear(OBSERVATION_SIZE, 3)
    q_optimizer = torch.optim.SGD(q_critic.parameters(), lr=0.02)
    actor_before = {
        name: value.detach().clone()
        for name, value in ppo.actor.state_dict().items()
    }
    actor_hash = module_sha256(ppo.actor)
    critic_before = {
        name: value.detach().clone()
        for name, value in q_critic.state_dict().items()
    }

    history = fit_q_critic(
        ppo,
        q_critic,
        q_optimizer,
        iterations=2,
        episodes=2,
        critic_epochs=2,
        batch_size=3,
        temperature=0.8,
        agent_sets=[["self", "self"]],
        seed=27_611,
    )

    assert len(history) == 2
    assert all(item["actor_unchanged"] for item in history)
    assert all(
        item["actor_sha256_before"]
        == item["actor_sha256_after"]
        == actor_hash
        for item in history
    )
    assert all(
        torch.equal(actor_before[name], value)
        for name, value in ppo.actor.state_dict().items()
    )
    assert all(parameter.requires_grad for parameter in ppo.actor.parameters())
    assert any(
        not torch.equal(critic_before[name], value)
        for name, value in q_critic.state_dict().items()
    )


def test_zero_qcritic_warmup_is_a_backward_compatible_noop():
    ppo = TinyPPO()
    q_critic = torch.nn.Linear(OBSERVATION_SIZE, 3)
    optimizer = torch.optim.SGD(q_critic.parameters(), lr=0.02)
    assert fit_q_critic(
        ppo,
        q_critic,
        optimizer,
        iterations=0,
        episodes=0,
        critic_epochs=0,
        batch_size=0,
        temperature=0.8,
        agent_sets=[["self", "self"]],
    ) == []

    parser = build_argument_parser()
    defaults = parser.parse_args(["--output-dir", "pilot"])
    assert defaults.q_warmup_iterations == 0
    assert defaults.torch_threads == DEFAULT_TORCH_THREADS
    configured = parser.parse_args(
        [
            "--output-dir",
            "pilot",
            "--q-warmup-iterations",
            "3",
            "--q-warmup-games",
            "64",
            "--q-warmup-epochs",
            "5",
            "--torch-threads",
            "3",
        ]
    )
    assert (
        configured.q_warmup_iterations,
        configured.q_warmup_games,
        configured.q_warmup_epochs,
    ) == (3, 64, 5)
    assert configured.torch_threads == 3


class MutatingTinyPPO(TinyPPO):
    def generate_data(self, **kwargs):
        with torch.no_grad():
            self.actor.bias.add_(0.01)
        return super().generate_data(**kwargs)


def test_qcritic_warmup_detects_actor_mutation():
    ppo = MutatingTinyPPO()
    q_critic = torch.nn.Linear(OBSERVATION_SIZE, 3)
    optimizer = torch.optim.SGD(q_critic.parameters(), lr=0.02)
    with pytest.raises(
        RuntimeError, match="warm-up changed frozen actor"
    ):
        fit_q_critic(
            ppo,
            q_critic,
            optimizer,
            iterations=1,
            episodes=2,
            critic_epochs=1,
            batch_size=3,
            temperature=0.8,
            agent_sets=[["self", "self"]],
            seed=27_612,
        )


def test_qboosting_pilot_rejects_sealed_seed():
    with pytest.raises(ValueError, match="sealed"):
        validate_development_seed(92_000_000)
