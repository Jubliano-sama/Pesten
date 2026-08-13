import math
from pathlib import Path
import sys

from run_factorized_league_until_plateau import (
    LEAGUE_SCHEDULES,
    PLAYER_COUNT_CURRICULA,
    _append_history,
    _confirmation_passed,
    _plateau_screen,
    _restore_campaign,
    _select_history,
    parse_args,
)


def test_every_curriculum_covers_2_through_8_without_schedule_lockstep():
    assert len(LEAGUE_SCHEDULES) == len(PLAYER_COUNT_CURRICULA) == 3
    for league, players in zip(LEAGUE_SCHEDULES, PLAYER_COUNT_CURRICULA):
        assert set(players) == set(range(2, 9))
        assert math.gcd(len(league), len(players)) == 1
        assert {
            "current_selfplay",
            "historical_temporal",
            "ppo_champion",
            "canonical_random",
            "smart_rule",
            "smarter_rule",
        } <= set(league)


def test_supervisor_defaults_to_bounded_training_but_not_evaluation(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_factorized_league_until_plateau.py",
            "--current",
            "current.pt",
            "--teacher",
            "teacher.pt",
            "--mlp",
            "mlp.pt",
            "--output-dir",
            "output",
        ],
    )

    args = parse_args()

    assert args.training_max_turns == 500


def test_bounded_history_preserves_oldest_newest_and_spans_lineage():
    paths = [Path(f"generation_{index}.pt") for index in range(12)]
    catalog = []
    for path in paths:
        catalog = _append_history(catalog, path)
    pool = _select_history(catalog, limit=8)

    assert len(pool) == 8
    assert pool[0] == paths[0].resolve()
    assert pool[-1] == paths[-1].resolve()
    assert len(set(pool)) == len(pool)
    assert any(path.resolve() in pool for path in paths[3:7])


def test_restore_campaign_uses_last_accepted_checkpoint_and_marks_partial_round(tmp_path):
    initial = tmp_path / "initial.pt"
    teacher = tmp_path / "teacher.pt"
    mlp = tmp_path / "mlp.pt"
    accepted = tmp_path / "accepted.pt"
    for path in (initial, teacher, mlp, accepted):
        path.touch()
    manifest = {
        "initial_checkpoint": str(initial),
        "teacher": {"path": str(teacher)},
        "mlp": {"path": str(mlp)},
        "current_checkpoint": str(accepted),
        "completed_phases": 589,
        "initial_historical_pool": [{"path": str(teacher)}],
        "rounds": [
            {
                "before": str(initial),
                "status": "complete",
                "plateau_signal": True,
            },
            {
                "before": str(accepted),
                "status": "monitoring",
            },
        ],
    }

    restored, catalog, phases, consecutive, window = _restore_campaign(
        manifest,
        current=initial,
        teacher=teacher,
        mlp=mlp,
    )

    assert restored == accepted.resolve()
    assert catalog[0] == teacher.resolve()
    assert catalog[-1] == accepted.resolve()
    assert phases == 589
    assert consecutive == 1
    assert window == initial.resolve()
    assert manifest["rounds"][-1]["status"] == "abandoned_after_restart"


def test_plateau_screen_uses_point_estimates_only_as_nomination():
    fixed = {
        "delta_normalized_win_share": 0.004,
        "delta_normalized_ci99_9": [-0.03, 0.04],
    }
    rules = {
        "delta_normalized_win_share": 0.009,
        "delta_normalized_ci99_9": [-0.04, 0.06],
    }

    passed, detail = _plateau_screen(
        fixed,
        rules,
        materiality=0.01,
        history_noninferior=True,
    )

    assert passed
    assert detail["passed"]
    assert detail["confidence_claim_deferred_to_disjoint_confirmation"]
    rules["delta_normalized_win_share"] = 0.011
    assert not _plateau_screen(
        fixed,
        rules,
        materiality=0.01,
        history_noninferior=True,
    )[0]


def test_plateau_confirmation_requires_both_powered_pools():
    fixed = {"delta_normalized_ci99_9": [-0.01, 0.025]}
    rules = {"delta_normalized_ci99_9": [-0.02, 0.029]}

    passed, detail = _confirmation_passed(
        fixed,
        rules,
        maximum_gain=0.03,
    )

    assert passed
    assert detail["passed"]
    rules["delta_normalized_ci99_9"][1] = 0.031
    assert not _confirmation_passed(
        fixed,
        rules,
        maximum_gain=0.03,
    )[0]
