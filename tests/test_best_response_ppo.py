import copy

import numpy as np
import pytest
import torch

from ai_observation import OBSERVATION_SIZE, PASS_ACTION
from best_response_ppo import (
    CANDIDATE_KEY,
    INCUMBENT_KEY,
    allocate_fresh_seed_plan,
    audit_terminal_candidate_payoffs,
    build_matched_fixed_opponent_layout,
    clipped_response_objective,
    collect_fixed_opponent_response,
    module_sha256,
    ppo_best_response_update,
    terminal_only_rewards,
)
from ppo2 import build_action_masks


class TinyActor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(OBSERVATION_SIZE, 55)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, observations):
        return self.linear(observations)


def test_matched_fixed_opponent_layout_rotates_candidate_on_common_deals():
    policy_kinds, actor_keys, seeds, candidate_seats = (
        build_matched_fixed_opponent_layout([101, 202, 303])
    )

    assert policy_kinds.shape == (6, 2)
    assert actor_keys.tolist() == [
        [CANDIDATE_KEY, INCUMBENT_KEY],
        [INCUMBENT_KEY, CANDIDATE_KEY],
        [CANDIDATE_KEY, INCUMBENT_KEY],
        [INCUMBENT_KEY, CANDIDATE_KEY],
        [CANDIDATE_KEY, INCUMBENT_KEY],
        [INCUMBENT_KEY, CANDIDATE_KEY],
    ]
    assert seeds.tolist() == [101, 101, 202, 202, 303, 303]
    assert candidate_seats.tolist() == [0, 1, 0, 1, 0, 1]


def test_seed_plan_skips_recorded_values_and_stays_unique():
    plan = allocate_fresh_seed_plan(
        requested_start=100,
        deal_count=3,
        policy_seed_count=2,
        shuffle_seed_count=2,
        recorded_seeds=[101, 104],
    )

    allocated = [
        plan["training_seed"],
        *plan["deal_seeds"],
        *plan["policy_sampling_seeds"],
        *plan["shuffle_seeds"],
    ]
    assert allocated == [100, 102, 103, 105, 106, 107, 108, 109]
    assert plan["skipped_recorded_seeds"] == [101, 104]
    assert len(set(allocated)) == len(allocated)
    assert plan["all_seeds_below_92m"] is True


def test_fixed_opponent_rollout_collects_candidate_only_and_audits_payoff():
    candidate = TinyActor().eval()
    champion = copy.deepcopy(candidate).eval()
    champion_before = module_sha256(champion)

    rollout, engine, candidate_seats = collect_fixed_opponent_response(
        candidate,
        champion,
        deal_seeds=[7001, 7002],
        policy_sampling_seed=8001,
        temperature=1.0,
        max_turns=500,
        device=torch.device("cpu"),
    )
    terminal_rewards = terminal_only_rewards(
        rollout.actions, rollout.rewards
    )
    audit = audit_terminal_candidate_payoffs(
        terminal_rewards,
        rollout.episode_lengths,
        engine=engine,
        candidate_seats=candidate_seats,
    )

    assert rollout.completed_games == 4
    assert rollout.policy_decisions > 0
    assert rollout.observations.shape[1] == OBSERVATION_SIZE
    assert audit["games"] == 4
    assert audit["matched_deal_pairs"] == 2
    assert audit["candidate_seat_0_games"] == 2
    assert audit["candidate_seat_1_games"] == 2
    assert audit["terminal_payoffs_verified"] is True
    assert audit["environment_zero_sum_verified"] is True
    assert module_sha256(champion) == champion_before


def test_terminal_only_rewards_removes_pass_shaping():
    actions = torch.tensor([PASS_ACTION, 3, PASS_ACTION])
    collected = torch.tensor([-0.1, 1.0, -1.1])

    terminal = terminal_only_rewards(actions, collected)

    assert torch.allclose(terminal, torch.tensor([0.0, 1.0, -1.0]))


def test_clipped_response_objective_applies_entropy_and_champion_anchor():
    ratios = torch.tensor([1.0, 1.2])
    advantages = torch.tensor([1.0, -1.0])
    objective, surrogate = clipped_response_objective(
        ratios,
        advantages,
        torch.tensor(0.7),
        torch.tensor(0.2),
        clip=0.1,
        entropy_coefficient=0.01,
        champion_kl_coefficient=0.5,
    )

    assert surrogate.item() == pytest.approx(-0.1)
    assert objective.item() == pytest.approx(-0.193)


def test_ppo_update_enforces_unchanged_policy_ratio_guard():
    candidate = TinyActor()
    champion = copy.deepcopy(candidate)
    observations = torch.zeros((4, OBSERVATION_SIZE))
    actions = torch.full((4,), PASS_ACTION, dtype=torch.long)
    masks = build_action_masks(observations)
    with torch.no_grad():
        logits = candidate(observations)
        log_probabilities = torch.log_softmax(
            logits.masked_fill(~masks, torch.finfo(logits.dtype).min),
            dim=-1,
        )
        behavior = log_probabilities.gather(
            1, actions[:, None]
        ).squeeze(1)
    optimizer = torch.optim.Adam(candidate.parameters(), lr=1e-3)
    before_champion = module_sha256(champion)

    metrics = ppo_best_response_update(
        candidate,
        champion,
        observations,
        actions,
        behavior,
        torch.tensor([1.0, -1.0, 1.0, -1.0]),
        optimizer=optimizer,
        temperature=1.0,
        clip=0.1,
        entropy_coefficient=0.0,
        champion_kl_coefficient=0.02,
        epochs=1,
        batch_size=4,
        shuffle_seed=99,
    )

    assert metrics["unchanged_policy_ratio_guard_passed"] is True
    assert metrics["maximum_initial_ratio_error"] < 1e-6
    assert module_sha256(champion) == before_champion

    with pytest.raises(RuntimeError, match="rollout/update policy mismatch"):
        ppo_best_response_update(
            candidate,
            champion,
            observations,
            actions,
            behavior + 1.0,
            torch.ones(4),
            optimizer=optimizer,
            temperature=1.0,
            clip=0.1,
            entropy_coefficient=0.0,
            champion_kl_coefficient=0.0,
            epochs=1,
            batch_size=4,
            shuffle_seed=100,
        )


def test_duplicate_common_deal_seed_is_rejected():
    with pytest.raises(ValueError, match="must be unique"):
        build_matched_fixed_opponent_layout([11, 11])


def test_seed_plan_rejects_retired_92m_boundary():
    with pytest.raises(ValueError, match="below 92 million"):
        allocate_fresh_seed_plan(
            requested_start=92_000_000,
            deal_count=1,
            policy_seed_count=1,
            shuffle_seed_count=1,
            recorded_seeds=[],
        )
