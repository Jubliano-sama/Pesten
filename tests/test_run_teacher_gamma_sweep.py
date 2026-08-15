import argparse
from pathlib import Path

import pytest

from run_teacher_gamma_sweep import (
    PLAYER_COUNT_SCHEDULE,
    classify_decision,
    confirmation_reference,
    gamma_slug,
    parse_gammas,
    training_command,
)


def _arm(gamma, delta):
    return {
        "gamma": gamma,
        "checkpoint": f"gamma-{gamma}.pt",
        "screen_weighted_result": {"delta_normalized_win_share": delta},
    }


def test_gamma_parser_and_schedule_contract():
    assert parse_gammas("0.9,0.97,1") == (0.9, 0.97, 1.0)
    assert gamma_slug(0.9330329915368074) == "gamma_0p933032991537"
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
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gammas("0.97")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gammas("0.97,0.97")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_gammas("0,0.97")


def test_training_command_is_matched_and_disables_critic_confound():
    args = argparse.Namespace(
        teacher=Path("teacher.pt"),
        champion=Path("champion.pth"),
        epochs=60,
        games_per_epoch=2048,
        rollout_workers=10,
        learning_rate=5e-5,
        update_epochs=4,
        gae_lambda=0.95,
        target_kl=0.015,
        training_seed=123,
        campaign_offset=5000,
    )
    command = training_command(args, 0.9, Path("output"))
    assert command[1] == "train_puffer_fast.py"
    assert "--fresh-optimizer" in command
    assert "--no-privileged-critic" in command
    assert "--compile-updates" in command
    assert command[command.index("--gamma") + 1] == "0.9"
    assert command[command.index("--campaign-offset") + 1] == "5000"


def test_confirmation_uses_historical_gamma_when_it_did_not_win():
    ranking = [_arm(0.933, 0.03), _arm(0.99, 0.02), _arm(0.97, 0.01)]
    assert confirmation_reference(ranking)["gamma"] == 0.97


def test_confirmation_uses_best_alternative_when_historical_gamma_wins():
    ranking = [_arm(0.97, 0.03), _arm(0.933, 0.02), _arm(0.99, 0.01)]
    assert confirmation_reference(ranking)["gamma"] == 0.933


def test_decision_requires_fresh_direct_and_baseline_confidence():
    positive = {"delta_normalized_ci99_9": [0.001, 0.02]}
    uncertain = {"delta_normalized_ci99_9": [-0.001, 0.02]}
    result = classify_decision(
        winner_gamma=0.933,
        reference_gamma=0.97,
        direct_result=positive,
        baseline_result=positive,
    )
    assert result["outcome"] == "alternate_gamma_confirmed_better"
    assert result["alternate_gamma_confirmed_helpful"] is True
    assert result["promotion_authorized"] is False

    result = classify_decision(
        winner_gamma=0.933,
        reference_gamma=0.97,
        direct_result=uncertain,
        baseline_result=positive,
    )
    assert result["outcome"] == "continued_training_helped_but_gamma_preference_unconfirmed"
    assert result["alternate_gamma_confirmed_helpful"] is False
