from __future__ import annotations

from pathlib import Path

import pytest
import torch

import canonical_random_recovery as recovery
from random_specialist_workflow import PhaseZeroTracePolicy
from temporal_batched_speed import collect_league_batch_batched


def test_canonical_scalar_batched_parity_covers_both_seats_and_coupled_rng():
    result = recovery.run_canonical_random_parity_gate(
        PhaseZeroTracePolicy().eval(),
        deal_seeds=tuple(range(4_028_451_531, 4_028_451_535)),
        max_turns=1_000,
    )

    assert result["passed"]
    assert [case["absolute_game_index"] for case in result["cases"]] == [
        0,
        1,
        2,
        3,
    ]
    assert all(case["mismatches"] == [] for case in result["cases"])
    assert all(case["scalar_trace_sha256"] == case["batched_trace_sha256"] for case in result["cases"])


def test_canonical_role_has_no_independent_opponent_rng_bank():
    trace = []
    _, metadata = collect_league_batch_batched(
        PhaseZeroTracePolicy().eval(),
        None,
        (),
        games=2,
        seed=4_028_451_531,
        schedule=("canonical_random",),
        max_turns=180,
        trace_sink=trace,
    )

    assert metadata["games"] == {"canonical_random": 2}
    assert all("opponent_rng_seed" not in item for item in trace)
    assert all(
        item["canonical_random_rng"]["independent_opponent_rng"] is False
        for item in trace
    )
    with pytest.raises(ValueError, match="must not receive"):
        collect_league_batch_batched(
            PhaseZeroTracePolicy().eval(),
            None,
            (),
            games=1,
            seed=4_028_451_531,
            schedule=("canonical_random",),
            random_opponent_seeds=(1,),
            max_turns=20,
        )


def test_raw_binary_pair_scoring_makes_draw_zero_not_half():
    scores = recovery.pair_mean_from_rotated_binary_wins(
        [True, False, False],
        [False, False, True],
    )

    assert scores.tolist() == [0.5, 0.0, 0.5]
    assert recovery.raw_win_rate(scores) == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="draws must be encoded as 0"):
        recovery.pair_mean_from_rotated_binary_wins([0.5], [1])


def test_fixed_seed_plan_stays_below_sealed_and_has_no_opponent_rng_ranges():
    plan = recovery.planned_intervals()

    assert sum(interval.count for interval in plan.values()) == 160_520
    assert all("opponent_rng" not in name for name in plan)
    assert all(
        interval.end_exclusive <= recovery.CONSUMED_LEGACY_SEALED_INTERVAL.start_inclusive
        for interval in plan.values()
    )
    assert plan["calibration_parity_deals"].start_inclusive == recovery.CAMPAIGN_GAP.start_inclusive
    assert plan["final_audit_canonical_deals"].end_exclusive == 4_028_612_051


def test_warm_start_loads_only_weights_and_requires_fresh_optimizer(monkeypatch):
    class CapturingPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.loaded = None

        def load_state_dict(self, state_dict, strict=True):
            self.loaded = (state_dict, strict)
            return torch.nn.modules.module._IncompatibleKeys([], [])

    checkpoint = {
        "model_state_dict": {"weight": torch.tensor([7.0])},
        "optimizer_state_dict": {"would": "be inherited if this test failed"},
    }
    monkeypatch.setattr(
        recovery, "file_sha256", lambda path: recovery.PARENT_CHECKPOINT_SHA256
    )
    monkeypatch.setattr(recovery.torch, "load", lambda *args, **kwargs: checkpoint)
    policy = CapturingPolicy()

    evidence = recovery.load_warm_start_weights_only(policy)

    assert policy.loaded == (checkpoint["model_state_dict"], True)
    assert evidence["model_weights_loaded"] is True
    assert evidence["optimizer_state_loaded"] is False
    assert evidence["optimizer_reset_required"] is True


def test_bootstrap_dry_run_neither_writes_nor_scans_banks(monkeypatch, tmp_path):
    output = tmp_path / "campaign"
    monkeypatch.setattr(
        recovery,
        "build_collision_scan_evidence",
        lambda: (_ for _ in ()).throw(AssertionError("scan must not run")),
    )
    monkeypatch.setattr(
        recovery,
        "require_canonical_parity",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("parity must not run")),
    )

    result = recovery.bootstrap_campaign(output_dir=output, dry_run=True)

    assert result["dry_run"] is True
    assert not output.exists()


def test_admission_fails_closed_without_calibration_and_commitment(tmp_path):
    with pytest.raises(FileNotFoundError):
        recovery.validate_pretraining_admission(output_dir=tmp_path / "missing")


def test_admission_fails_closed_when_native_smoke_gate_is_false(monkeypatch, tmp_path):
    manifest = {"source_sha256": recovery.campaign_source_hashes()}
    commitment = {
        "campaign_id": recovery.CAMPAIGN_ID,
        "source_sha256": recovery.campaign_source_hashes(),
        "admission": {
            "canonical_scalar_batched_parity": True,
            "privacy_boundary": True,
            "native_cuda_smoke": False,
            "source_unchanged": True,
            "opponent_rng_bank": False,
        },
        "candidate_training": {"started": False},
    }
    monkeypatch.setattr(recovery, "load_frozen_manifest", lambda output: manifest)
    monkeypatch.setattr(recovery, "_assert_manifest_source_hashes", lambda value: None)
    monkeypatch.setattr(recovery, "_load_calibration_admission", lambda output: {"canonical_parity": {"passed": True}})
    monkeypatch.setattr(recovery, "_read_json", lambda path: commitment)

    with pytest.raises(RuntimeError, match="not complete"):
        recovery.validate_pretraining_admission(output_dir=tmp_path)


def test_canonical_audit_paths_and_order_are_fixed():
    claim, result = recovery.canonical_final_audit_paths(Path("benchmarks") / "x")
    order = recovery.audit_claim_order()

    assert claim.name == "final_audit_claim.json"
    assert result.name == "final_audit_result.json"
    assert order.index("exclusive-create one-use canonical final-audit claim") < order.index(
        "load selected candidate weights after claim only"
    ) < order.index("iterate registered final-audit deal seeds in ascending order")


def test_train_and_audit_dry_runs_do_not_write_or_touch_banks(tmp_path):
    campaign = tmp_path / "campaign"
    training = tmp_path / "training"

    train = recovery.train_canonical_campaign(
        output_dir=campaign, training_dir=training, dry_run=True
    )
    audit = recovery.audit_canonical_winner(
        output_dir=campaign, training_dir=training, dry_run=True
    )

    assert train["dry_run"] is True
    assert train["candidate_training_started"] is False
    assert train["final_audit_seed_bank_touched"] is False
    assert audit["dry_run"] is True
    assert audit["candidate_model_loaded"] is False
    assert audit["final_seed_iterator_constructed"] is False
    assert not campaign.exists()
    assert not training.exists()


def test_training_budget_is_exact_and_has_six_25k_checkpoints():
    plan = recovery.training_microbatch_plan()
    contract = recovery.training_budget_contract()

    assert sum(row["games"] for row in plan) == 150_000
    assert [row["end_offset"] for row in plan if row["checkpoint_after"]] == [
        25_000,
        50_000,
        75_000,
        100_000,
        125_000,
        150_000,
    ]
    assert contract["checkpoint_count"] == 6
    assert contract["wall_clock_cap_seconds"] == 5 * 60 * 60
    assert contract["ppo_epochs"] == 4


def test_development_bank_is_exclusive_once(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = {
        "seed_registry": {
            "intervals": {
                "development_screening_canonical_01": {
                    "start_inclusive": 10,
                    "end_exclusive": 510,
                    "count": 500,
                }
            }
        },
        "development": {
            "bootstrap_seed_interval": {
                "start_inclusive": 99,
                "end_exclusive": 100,
                "count": 1,
            }
        },
    }
    checkpoint_record = {
        "checkpoint_sha256": "checkpoint-sha",
        "cumulative_training_games": 25_000,
        "admission": {"eligible": True},
    }
    monkeypatch.setattr(
        recovery,
        "_load_canonical_checkpoint_policy",
        lambda **kwargs: (object(), {}),
    )
    monkeypatch.setattr(
        recovery,
        "evaluate_canonical_seat_balanced_pairs",
        lambda policy, *, deal_seeds: {
            "pair_scores": [1.0],
            "raw_win_rate": 1.0,
            "outcomes": {"wins": 2, "losses": 0, "draws": 0},
        },
    )
    monkeypatch.setattr(
        recovery,
        "_bootstrap_summary",
        lambda evaluation, *, seed: {"raw_win_rate": 1.0, "lower_95": 1.0},
    )

    first = recovery._evaluate_development_checkpoint(
        training_dir=tmp_path,
        checkpoint_index=1,
        checkpoint_path=checkpoint,
        checkpoint_record=checkpoint_record,
        manifest=manifest,
        device=torch.device("cpu"),
        commitment_sha256="commitment",
    )

    assert first["one_use"] is True
    with pytest.raises(FileExistsError, match="already consumed"):
        recovery._evaluate_development_checkpoint(
            training_dir=tmp_path,
            checkpoint_index=1,
            checkpoint_path=checkpoint,
            checkpoint_record=checkpoint_record,
            manifest=manifest,
            device=torch.device("cpu"),
            commitment_sha256="commitment",
        )


def test_development_ranking_is_raw_win_then_lower_ci_then_games_and_sha():
    ranked = recovery.rank_development_checkpoints(
        [
            {
                "checkpoint_sha256": "z",
                "training_games": 50_000,
                "pair_scores": [1.0, 0.0],
                "bootstrap_lower_95": 0.8,
            },
            {
                "checkpoint_sha256": "a",
                "training_games": 25_000,
                "pair_scores": [1.0, 1.0],
                "bootstrap_lower_95": 0.7,
            },
            {
                "checkpoint_sha256": "b",
                "training_games": 25_000,
                "pair_scores": [1.0, 1.0],
                "bootstrap_lower_95": 0.7,
            },
        ]
    )

    assert [row["checkpoint_sha256"] for row in ranked] == ["a", "b", "z"]


def test_confirmation_failure_is_terminal_and_never_creates_frozen_winner(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint.pt"
    development_winner = {
        "selection": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": "checkpoint-sha",
            "training_games": 25_000,
        }
    }
    manifest = {
        "confirmation": {
            "pair_interval": {
                "start_inclusive": 10,
                "end_exclusive": 2_510,
                "count": 2_500,
            },
            "bootstrap_seed_interval": {
                "start_inclusive": 20,
                "end_exclusive": 21,
                "count": 1,
            },
            "required_raw_win_rate": 0.8,
            "required_lower_95_confidence_bound": 0.8,
        }
    }
    monkeypatch.setattr(
        recovery,
        "_load_canonical_checkpoint_policy",
        lambda **kwargs: (object(), {}),
    )
    monkeypatch.setattr(
        recovery,
        "evaluate_canonical_seat_balanced_pairs",
        lambda policy, *, deal_seeds: {
            "pair_scores": [0.0],
            "raw_win_rate": 0.0,
            "outcomes": {"wins": 0, "losses": 2, "draws": 0},
        },
    )
    monkeypatch.setattr(
        recovery,
        "_bootstrap_summary",
        lambda evaluation, *, seed: {"raw_win_rate": 0.0, "lower_95": 0.0},
    )

    result = recovery._run_confirmation(
        training_dir=tmp_path,
        development_winner=development_winner,
        manifest=manifest,
        device=torch.device("cpu"),
        commitment_sha256="commitment",
        manifest_sha256="manifest",
    )

    assert result["frozen_winner"] is None
    assert result["failed"]["audit_forbidden"] is True
    assert (tmp_path / "confirmation_failed.json").is_file()
    assert not (tmp_path / "frozen_winner.json").exists()


def test_final_audit_claim_is_exclusive_and_cannot_retry(monkeypatch, tmp_path):
    claim, result = recovery.canonical_final_audit_paths(tmp_path)
    manifest = {
        "final_audit": {
            "claim_path": str(claim),
            "result_path": str(result),
            "pair_interval": {"start_inclusive": 10, "end_exclusive": 11, "count": 1},
            "pairs": 1,
        }
    }
    monkeypatch.setattr(recovery, "load_frozen_manifest", lambda output: manifest)

    first = recovery.create_final_audit_claim(
        output_dir=tmp_path,
        winner_sha256="winner",
        commitment_sha256="commitment",
        manifest_sha256="manifest",
    )

    assert first["claim_before_model_load"] is True
    assert first["claim_before_final_seed_iteration"] is True
    with pytest.raises(FileExistsError, match="one-use"):
        recovery.create_final_audit_claim(
            output_dir=tmp_path,
            winner_sha256="winner",
            commitment_sha256="commitment",
            manifest_sha256="manifest",
        )


def test_cli_help_exposes_train_and_audit(capsys):
    with pytest.raises(SystemExit) as error:
        recovery._main(["--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert "train" in help_text
    assert "audit" in help_text
