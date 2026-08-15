import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from benchmark_rlvr_pool import OPPONENTS, PLAYER_WEIGHTS
from benchmark_rlvr_pool_batched import _weighted_outcome_summary
from packed_puffer_rollout import pack_puffer_episode
from run_teacher_reward_sweep import (
    DEFAULT_ARMS,
    PLAYER_COUNT_SCHEDULE,
    ranking_key,
    training_command,
    validate_arms,
)
from temporal_training import _discounted_returns, _gae_targets_from_values


def test_default_design_is_l9_plus_control_and_schedule_matches_teacher():
    arms = validate_arms(DEFAULT_ARMS)
    assert len(arms) == 10
    assert {arm["gamma"] for arm in arms} == {0.99, 0.995, 0.9975}
    assert {arm["decision_reward"] for arm in arms} == {-0.02, -0.01, 0.0}
    assert {arm["draw_reward"] for arm in arms} == {-1.0, -0.25, 0.0}
    assert len(PLAYER_COUNT_SCHEDULE) == 63
    assert {count: PLAYER_COUNT_SCHEDULE.count(count) for count in range(2, 9)} == {
        2: 14,
        3: 14,
        4: 14,
        5: 6,
        6: 5,
        7: 5,
        8: 5,
    }


def test_arm_validation_rejects_duplicate_and_invalid_contracts():
    with pytest.raises(ValueError, match="unique parameter"):
        validate_arms(
            [
                {"name": "a", "gamma": 0.99, "decision_reward": 0, "draw_reward": 0},
                {"name": "b", "gamma": 0.99, "decision_reward": 0, "draw_reward": 0},
            ]
        )
    with pytest.raises(ValueError, match="nonpositive"):
        validate_arms(
            [
                {"name": "a", "gamma": 0.99, "decision_reward": 0.1, "draw_reward": 0},
                {"name": "b", "gamma": 0.995, "decision_reward": 0, "draw_reward": 0},
            ]
        )


def test_training_command_uses_teacher_and_exact_reward_contract():
    args = argparse.Namespace(
        teacher=Path("teacher.pt"),
        champion=Path("champion.pth"),
        epochs=12,
        games_per_epoch=2048,
        rollout_workers=10,
        learning_rate=5e-5,
        update_epochs=4,
        gae_lambda=0.95,
        target_kl=0.015,
        training_seed=123,
        campaign_offset=9000,
    )
    arm = {
        "name": "neutral",
        "gamma": 0.995,
        "decision_reward": -0.01,
        "draw_reward": 0.0,
    }
    command = training_command(args, arm, Path("out"))
    assert command[command.index("--resume") + 1] == "teacher.pt"
    assert command[command.index("--gamma") + 1] == "0.995"
    assert command[command.index("--decision-reward") + 1] == "-0.01"
    assert command[command.index("--draw-reward") + 1] == "0.0"
    assert "--fresh-optimizer" in command
    assert "--no-privileged-critic" in command


def test_reward_targets_distinguish_draw_from_loss_and_use_preferred_default():
    loss = _discounted_returns(2, won=False, gamma=1.0)
    preferred_draw = _discounted_returns(2, won=None, gamma=1.0)
    legacy_draw = _discounted_returns(
        2,
        won=None,
        gamma=1.0,
        draw_reward=-1.0,
    )
    neutral_draw = _discounted_returns(
        2,
        won=None,
        gamma=1.0,
        decision_reward=-0.01,
        draw_reward=0.0,
    )
    assert torch.equal(loss, legacy_draw)
    assert torch.allclose(loss, torch.tensor([-1.02, -1.01]))
    assert torch.allclose(preferred_draw, torch.tensor([-0.77, -0.76]))
    assert torch.allclose(neutral_draw, torch.tensor([-0.02, -0.01]))

    values = torch.tensor([0.2, -0.1])
    returns, advantages = _gae_targets_from_values(
        values,
        won=None,
        gamma=1.0,
        gae_lambda=1.0,
        decision_reward=0.0,
        draw_reward=0.0,
    )
    assert torch.allclose(returns, torch.zeros(2), atol=2e-8)
    assert torch.allclose(advantages, -values, atol=2e-8)


def test_packed_episode_preserves_draw_sentinel():
    class Policy:
        @staticmethod
        def _event_features(decision):
            return []

        @staticmethod
        def _decision_features(decision):
            return np.zeros(100, dtype=np.float32)

    class Decision:
        public_events = ()
        legal_actions = (True,) * 55

    class Step:
        decision = Decision()
        action = 0
        old_log_prob = 0.0
        value = 0.0

    _, outcome = pack_puffer_episode(Policy(), ((Step(),), None))
    assert outcome is None


def test_weighted_outcomes_report_matched_loss_avoidance():
    strata = []
    for opponent in OPPONENTS:
        for players in PLAYER_WEIGHTS:
            strata.append(
                {
                    "opponent": opponent,
                    "players": players,
                    "before_win_shares": [0.2, 0.4],
                    "after_win_shares": [0.3, 0.5],
                    "before_loss_shares": [0.7, 0.5],
                    "after_loss_shares": [0.5, 0.3],
                    "before_draw_shares": [0.1, 0.1],
                    "after_draw_shares": [0.2, 0.2],
                }
            )
    result = _weighted_outcome_summary(strata)
    assert result["win"]["delta_after_minus_before"] == pytest.approx(0.1)
    assert result["loss_avoidance"]["delta_before_minus_after"] == pytest.approx(0.2)
    assert result["draw"]["delta_after_minus_before"] == pytest.approx(0.1)


def test_ranking_is_loss_first_then_win():
    def row(loss, win, draw):
        return {
            "screen_weighted_outcomes": {
                "loss": {"after_rate": loss},
                "win": {"after_rate": win},
                "draw": {"after_rate": draw},
            }
        }

    assert ranking_key(row(0.4, 0.2, 0.4)) < ranking_key(row(0.41, 0.5, 0.09))
    assert ranking_key(row(0.4, 0.3, 0.3)) < ranking_key(row(0.4, 0.2, 0.4))


def test_checked_in_sweep_summary_is_complete_and_not_a_promotion():
    path = Path("benchmarks/teacher_reward_sweep_20260815.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "complete_no_confirmed_gain"
    assert payload["protocol"]["total_training_games"] == 245_760
    assert len(payload["screen_ranking"]) == 10
    assert payload["screen_ranking"][0]["name"] == "l9_03"
    assert payload["confirmation"]["winner_vs_control"][
        "loss_reduction_confirmed"
    ] is False
    assert payload["confirmation"]["winner_vs_teacher"][
        "loss_reduction_confirmed"
    ] is False
    assert payload["conclusion"]["promotion_performed"] is False
