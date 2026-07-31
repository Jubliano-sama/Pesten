import copy

import json
import pytest
import torch

from ai_observation import OBSERVATION_SIZE
from seed_integrity import DEVELOPMENT_SEED_LIMIT, scan_recorded_seed_interval
from trpo_selfplay import (
    conjugate_gradient,
    flat_parameters,
    masked_old_new_kl,
    recorded_deal_seeds,
    trainable_parameters,
    trpo_actor_step,
    validate_development_seeds,
)


def test_masked_kl_is_old_to_new_and_ignores_illegal_extremes():
    old = torch.tensor([[2.0, -1.0, 1e30]], dtype=torch.float64)
    new = torch.tensor([[-0.5, 1.0, -1e30]], dtype=torch.float64)
    legal = torch.tensor([[True, True, False]])
    actual = masked_old_new_kl(old, new, legal)
    old_log = torch.log_softmax(old[:, :2], dim=-1)
    new_log = torch.log_softmax(new[:, :2], dim=-1)
    expected = (old_log.exp() * (old_log - new_log)).sum()
    reverse = (new_log.exp() * (new_log - old_log)).sum()

    assert torch.equal(actual, expected)
    assert not torch.isclose(actual, reverse)
    assert masked_old_new_kl(old, old, legal).item() == 0.0


def test_conjugate_gradient_solves_spd_system():
    matrix = torch.tensor(
        [[4.0, 1.0], [1.0, 3.0]],
        dtype=torch.float64,
    )
    right_hand_side = torch.tensor([1.0, 2.0], dtype=torch.float64)
    solution, log = conjugate_gradient(
        lambda vector: matrix @ vector,
        right_hand_side,
        iterations=10,
        residual_tolerance=1e-14,
    )
    expected = torch.linalg.solve(matrix, right_hand_side)

    torch.testing.assert_close(solution, expected, rtol=1e-12, atol=1e-12)
    assert log["iterations"] <= 2
    assert log["final_residual_norm"] <= 1e-12


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(OBSERVATION_SIZE, 3)

    def forward(self, observations):
        return self.linear(observations)


def _tiny_batch():
    torch.manual_seed(84_603_001)
    actor = TinyActor().eval()
    observations = torch.randn(64, OBSERVATION_SIZE)
    legal = torch.ones(64, 3, dtype=torch.bool)
    with torch.inference_mode():
        logits = actor(observations)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        behavior_log_probabilities = distribution.log_prob(actions)
    advantages = torch.where(
        actions == 0,
        torch.ones_like(actions, dtype=torch.float32),
        -torch.ones_like(actions, dtype=torch.float32),
    )
    advantages = (advantages - advantages.mean()) / advantages.std(
        unbiased=False
    )
    return (
        actor,
        observations,
        actions,
        behavior_log_probabilities,
        advantages,
        legal,
    )


def test_trpo_step_improves_surrogate_inside_measured_trust_region():
    actor, observations, actions, old_log_probs, advantages, legal = _tiny_batch()
    before = flat_parameters(trainable_parameters(actor)).clone()
    result = trpo_actor_step(
        actor,
        observations,
        actions,
        old_log_probs,
        advantages,
        legal,
        temperature=1.0,
        target_kl=0.01,
        conjugate_gradient_iterations=10,
        fisher_damping=0.1,
        backtrack_iterations=10,
    )

    assert result["accepted"]
    assert result["actual_improvement"] > 0
    assert 0 <= result["kl_after"] <= result["target_kl"]
    assert result["rollback_verified"]
    assert not torch.equal(
        flat_parameters(trainable_parameters(actor)),
        before,
    )


def test_rejected_line_search_restores_parameters_exactly():
    actor, observations, actions, old_log_probs, advantages, legal = _tiny_batch()
    actor_before = copy.deepcopy(actor.state_dict())
    result = trpo_actor_step(
        actor,
        observations,
        actions,
        old_log_probs,
        advantages,
        legal,
        temperature=1.0,
        target_kl=0.01,
        conjugate_gradient_iterations=5,
        fisher_damping=0.1,
        backtrack_iterations=2,
        minimum_improvement_ratio=1e9,
    )

    assert not result["accepted"]
    assert result["reason"] == "line_search_rejected"
    assert result["rollback_verified"]
    for name, value in actor.state_dict().items():
        assert torch.equal(value, actor_before[name])


def test_behavior_policy_ratio_mismatch_fails_closed():
    actor, observations, actions, old_log_probs, advantages, legal = _tiny_batch()
    with pytest.raises(RuntimeError, match="maximum ratio error"):
        trpo_actor_step(
            actor,
            observations,
            actions,
            old_log_probs + 0.1,
            advantages,
            legal,
            temperature=1.0,
            target_kl=0.01,
        )


def test_seed_guard_rejects_sealed_namespace():
    validate_development_seeds([0, DEVELOPMENT_SEED_LIMIT - 1])
    with pytest.raises(ValueError, match="development-only"):
        validate_development_seeds([DEVELOPMENT_SEED_LIMIT])
    with pytest.raises(ValueError, match="development-only"):
        validate_development_seeds([-1])


def test_recorded_deal_seeds_are_visible_to_recursive_scanner(tmp_path):
    seed = 84_603_100
    artifact = tmp_path / "trpo_dev.json"
    artifact.write_text(
        json.dumps({"deal_seeds": recorded_deal_seeds([seed, seed + 1])}),
        encoding="utf-8",
    )
    uses = scan_recorded_seed_interval(
        [tmp_path],
        start=seed,
        end=seed + 2,
        allow_sealed_results=False,
    )
    assert [(use.key_path, use.seed) for use in uses] == [
        ("deal_seeds.[0].deal_seed", seed),
        ("deal_seeds.[1].deal_seed", seed + 1),
    ]


def test_trpo_rejects_nonpublic_observation_width():
    actor, observations, actions, old_log_probs, advantages, legal = _tiny_batch()
    with pytest.raises(ValueError, match="public observation"):
        trpo_actor_step(
            actor,
            torch.cat([observations, torch.zeros(len(observations), 1)], dim=1),
            actions,
            old_log_probs,
            advantages,
            legal,
            temperature=1.0,
            target_kl=0.01,
        )
