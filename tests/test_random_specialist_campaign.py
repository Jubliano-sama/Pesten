from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

import random_specialist_campaign as campaign
from random_specialist_campaign import (
    BOOTSTRAP_SAMPLES,
    LEGACY_V2_COMMITMENT_FORMAT,
    SMOKE_FORMAT,
    _assert_checkpoint_admission,
    _checkpoint_admission_record,
    _checkpoint_lineage,
    _evaluate_checkpoint_development,
    _projection_before_training,
    _stage_checkpoint,
    _source_hashes,
    audit_claim_order,
    assert_canonical_final_audit_paths,
    bootstrap_algorithm,
    bootstrap_pair_scores,
    build_implementation_commitment,
    calibration_admission_evidence,
    canonical_final_audit_paths,
    current_policy_rollout_seed,
    evaluate_seat_balanced_pairs,
    _json_sha256,
    validate_pretraining_admission,
)
from random_specialist_workflow import (
    PhaseZeroTracePolicy,
    file_sha256,
    validate_frozen_campaign,
)


ROOT = Path("benchmarks/random_specialist_20260802")


def _calibration_manifest_copy(tmp_path):
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_bytes((ROOT / "calibration.json").read_bytes())
    manifest = json.loads((ROOT / "frozen_campaign.json").read_text(encoding="utf-8"))
    manifest["calibration"] = dict(manifest["calibration"])
    manifest["calibration"]["path"] = str(calibration_path)
    return manifest, calibration_path


def _checkpoint_commitment(manifest):
    evidence = calibration_admission_evidence(manifest)
    return {
        "commitment_sha256": "test-commitment",
        "configuration_sha256": "test-configuration",
        "source_sha256": {"test": "source"},
        "opponent_contract_sha256": "test-opponent",
        "configuration": {
            "checkpoint_admission": deepcopy(manifest["checkpoint_admission"]),
            "calibration_admission": evidence,
            "training": {"checkpoint_thresholds": [2_169]},
        },
    }, evidence


def test_scalar_evaluator_pairs_same_seed_in_both_seats_and_resets():
    policy = PhaseZeroTracePolicy().eval()
    result = evaluate_seat_balanced_pairs(
        policy,
        deal_seeds=(91_000_001, 91_000_002),
        opponent_seeds=(91_100_001, 91_100_002),
        phase="test",
    )

    assert result["pair_count"] == 2
    assert result["game_count"] == 4
    assert result["all_legal_actions_verified"]
    assert result["all_recurrent_resets_verified"]
    assert result["all_opponent_resets_verified"]
    for pair in result["pairs"]:
        assert pair["candidate_seats"] == [0, 1]
        assert pair["same_deal_and_opponent_rng_each_seat"]
        first, second = pair["games"]
        assert first["deal_seed"] == second["deal_seed"] == pair["deal_seed"]
        assert (
            first["opponent_rng_seed"]
            == second["opponent_rng_seed"]
            == pair["opponent_rng_seed"]
        )
        for game_result in pair["games"]:
            assert game_result["candidate"]["reset_verified"]
            assert game_result["opponent"]["rng_reset_verified"]
            assert game_result["legal_actions_verified"]
            assert game_result["candidate_policy_rng_seed"] == current_policy_rollout_seed(
                pair["deal_seed"]
            )


def test_scalar_evaluator_repeat_is_bitwise_trace_stable_for_fixture():
    policy = PhaseZeroTracePolicy().eval()
    kwargs = {
        "deal_seeds": (91_200_001,),
        "opponent_seeds": (91_300_001,),
        "phase": "repeat",
    }
    first = evaluate_seat_balanced_pairs(policy, **kwargs)
    second = evaluate_seat_balanced_pairs(policy, **kwargs)

    assert first["evaluation_sha256"] == second["evaluation_sha256"]
    assert first["pairs"][0]["pair_trace_sha256"] == second["pairs"][0]["pair_trace_sha256"]


def test_pair_bootstrap_is_frozen_and_reproducible():
    first = bootstrap_pair_scores((1.0, 0.5, 0.0, 1.0), seed=91_400_001)
    second = bootstrap_pair_scores((1.0, 0.5, 0.0, 1.0), seed=91_400_001)

    assert first == second
    assert first["algorithm"] == bootstrap_algorithm()
    assert first["algorithm"]["samples"] == BOOTSTRAP_SAMPLES == 100_000
    assert first["algorithm"]["with_replacement"] is True
    assert first["algorithm"]["rng_algorithm"] == "MT19937"
    assert first["algorithm"]["quantile_method"] == "numpy.quantile(method='linear')"


def test_precommit_payload_binds_new_runner_and_exact_registry_records(tmp_path):
    manifest_path = ROOT / "frozen_campaign.json"
    registry_path = ROOT / "seed_registry.jsonl"
    manifest = validate_frozen_campaign(manifest_path)
    failed_smoke_path = tmp_path / "runner_calibration_smoke.json"
    failed_smoke_path.write_text(
        json.dumps(
            {
                "format": SMOKE_FORMAT,
                "passed": False,
                "candidate_training_started": False,
                "candidate_checkpoint_created": False,
                "evaluation_sha256": "failed-first",
                "repeat_evaluation_sha256": "failed-repeat",
                "mapping_reset_repeat": {
                    "repeat_evaluation_sha256_equal": False,
                },
            }
        ),
        encoding="utf-8",
    )
    smoke_path = tmp_path / "runner_calibration_smoke_recheck.json"
    smoke_path.write_text(
        json.dumps(
            {
                "format": SMOKE_FORMAT,
                "passed": True,
                "candidate_training_started": False,
                "candidate_checkpoint_created": False,
                "frozen_manifest_sha256": __import__(
                    "random_specialist_workflow"
                ).file_sha256(manifest_path),
                "registry_sha256": __import__(
                    "random_specialist_workflow"
                ).file_sha256(registry_path),
                "source_sha256": _source_hashes(),
                "evaluation_sha256": "fixture-smoke-evidence",
                "gates": {
                    "calibration_only": True,
                    "pair_mapping": True,
                    "legal_actions": True,
                    "recurrent_resets": True,
                    "opponent_resets": True,
                    "exact_repeat_digest": True,
                    "native_puffer_policy": True,
                    "claim_before_model_and_final_seed_iteration": True,
                },
            }
        ),
        encoding="utf-8",
    )
    superseded_path = tmp_path / "implementation_commitment_v2.json"
    superseded = {
        "format": LEGACY_V2_COMMITMENT_FORMAT,
        "candidate_training": {"started": False},
    }
    superseded["commitment_sha256"] = _json_sha256(superseded)
    superseded_path.write_text(json.dumps(superseded), encoding="utf-8")

    commitment = build_implementation_commitment(
        manifest_path=manifest_path,
        registry_path=registry_path,
        smoke_path=smoke_path,
        failed_smoke_path=failed_smoke_path,
        superseded_v2_path=superseded_path,
    )

    assert commitment["candidate_training"] == {
        "started": False,
        "checkpoint_created": False,
        "precommit_only": True,
    }
    assert commitment["source_sha256"]["random_specialist_campaign.py"] == _source_hashes()[
        "random_specialist_campaign.py"
    ]
    assert commitment["configuration"]["training"]["checkpoint_thresholds"][-1] == 26_028
    assert commitment["configuration"]["bootstrap"] == bootstrap_algorithm()
    assert commitment["configuration"]["checkpoint_admission"] == manifest[
        "checkpoint_admission"
    ]
    assert commitment["configuration"]["calibration_admission"]["privacy"][
        "fingerprint"
    ] == manifest["execution"]["privacy"]["fingerprint"]
    assert commitment["calibration_smoke"]["successful_recheck"]["acceptance_authority"]
    assert not commitment["calibration_smoke"]["retained_failed_smoke"][
        "acceptance_authority"
    ]
    assert commitment["supersedes"]["commitment_sha256"] == superseded[
        "commitment_sha256"
    ]
    assert set(commitment["seed_registry"]["namespace_records"]) == set(
        manifest["namespace_allocations"]["calibration"]
    ) | set(manifest["namespace_allocations"]["future_campaign"])


def test_pretraining_admission_rejects_tampered_calibration_sha(tmp_path):
    manifest, calibration_path = _calibration_manifest_copy(tmp_path)
    commitment, _ = _checkpoint_commitment(manifest)
    calibration_path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA"):
        validate_pretraining_admission(commitment, manifest)


def test_pretraining_admission_rejects_tampered_privacy_fingerprint(
    tmp_path, monkeypatch
):
    manifest, calibration_path = _calibration_manifest_copy(tmp_path)
    commitment, _ = _checkpoint_commitment(manifest)
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    payload["privacy_boundary"]["safe_schema_sha256"] = "0" * 64
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest["calibration"]["sha256"] = file_sha256(calibration_path)
    monkeypatch.setattr(
        campaign, "FROZEN_CALIBRATION_SHA256", manifest["calibration"]["sha256"]
    )

    with pytest.raises(RuntimeError, match="privacy fingerprint"):
        validate_pretraining_admission(commitment, manifest)


def test_pretraining_admission_rejects_tampered_fixed_random_parity(
    tmp_path, monkeypatch
):
    manifest, calibration_path = _calibration_manifest_copy(tmp_path)
    commitment, _ = _checkpoint_commitment(manifest)
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    payload["fixed_random_parity"]["cases"][0]["passed"] = False
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    manifest["calibration"]["sha256"] = file_sha256(calibration_path)
    monkeypatch.setattr(
        campaign, "FROZEN_CALIBRATION_SHA256", manifest["calibration"]["sha256"]
    )

    with pytest.raises(RuntimeError, match="parity"):
        validate_pretraining_admission(commitment, manifest)


def test_checkpoint_manifest_records_all_true_admission_gates(tmp_path):
    manifest, _ = _calibration_manifest_copy(tmp_path)
    commitment, evidence = _checkpoint_commitment(manifest)
    policy = PhaseZeroTracePolicy().eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    metrics = {"loss": 0.25, "nested": [0.0, 1.0]}
    lineage = _checkpoint_lineage(
        commitment=commitment,
        cumulative_games=2_169,
        metrics=metrics,
        training_seconds=1.0,
    )
    admission = _checkpoint_admission_record(
        policy=policy,
        optimizer=optimizer,
        commitment=commitment,
        cumulative_games=2_169,
        metrics=metrics,
        lineage=lineage,
        all_legal_actions=True,
        pretraining_evidence=evidence,
    )
    output_dir = tmp_path / "checkpoint_stage"
    output_dir.mkdir()
    _, pending_manifest, checkpoint_manifest = _stage_checkpoint(
        output_dir=output_dir,
        policy=policy,
        optimizer=optimizer,
        commitment=commitment,
        cumulative_games=2_169,
        metrics=metrics,
        training_seconds=1.0,
        admission=admission,
    )

    stored = json.loads(pending_manifest.read_text(encoding="utf-8"))
    _assert_checkpoint_admission(stored, commitment)
    assert stored["admission"] == admission
    assert stored["admission"]["eligible"]
    assert all(stored["admission"]["gates"].values())
    assert checkpoint_manifest["admission"] == admission


def test_ineligible_checkpoint_is_refused_before_development_screening(tmp_path):
    manifest, _ = _calibration_manifest_copy(tmp_path)
    commitment, evidence = _checkpoint_commitment(manifest)
    policy = PhaseZeroTracePolicy().eval()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    metrics = {"loss": 0.25}
    lineage = _checkpoint_lineage(
        commitment=commitment,
        cumulative_games=2_169,
        metrics=metrics,
        training_seconds=1.0,
    )
    admission = _checkpoint_admission_record(
        policy=policy,
        optimizer=optimizer,
        commitment=commitment,
        cumulative_games=2_169,
        metrics=metrics,
        lineage=lineage,
        all_legal_actions=True,
        pretraining_evidence=evidence,
    )
    admission["gates"]["100 percent legal actions"] = False
    admission["eligible"] = False
    checkpoint_path = tmp_path / "ineligible.pt"
    checkpoint_manifest = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": "not-loaded",
        "cumulative_training_games": 2_169,
        "complete_model_state": True,
        "complete_optimizer_state": True,
        "lineage": lineage,
        "admission": admission,
    }

    with pytest.raises(RuntimeError, match="ineligible"):
        _evaluate_checkpoint_development(
            checkpoint_path=checkpoint_path,
            checkpoint_manifest=checkpoint_manifest,
            commitment=commitment,
            manifest=manifest,
            checkpoint_index=1,
            device=torch.device("cpu"),
            output_dir=tmp_path,
        )
    assert not (tmp_path / "development_screen_01.json").exists()


def test_training_projection_uses_exact_partial_checkpoint_microbatches():
    manifest = validate_frozen_campaign(ROOT / "frozen_campaign.json")
    commitment_like = {"capacity": manifest["lanes"]["recurrent_temporal"]["capacity"], "checkpoint_thresholds": manifest["checkpoint_schedule"]["recurrent_temporal"]["evenly_spaced_cumulative_game_thresholds"]}
    projection = _projection_before_training(commitment_like)

    assert projection["completed_games"] == 26_028
    assert projection["within_cap"]
    assert projection["microbatch_sizes"][16] == 121
    assert projection["projected_training_seconds"] <= 45 * 60


def test_audit_claim_order_precedes_model_load_and_final_seed_iteration():
    order = audit_claim_order()
    assert order.index("exclusive-create one-use claim") < order.index(
        "load candidate checkpoint"
    )
    assert order.index("load candidate checkpoint") < order.index(
        "construct final pair iterator and evaluate outcomes"
    )


def test_audit_paths_must_resolve_to_the_committed_ledger_and_result(tmp_path):
    commitment = {
        "configuration": {
            "final_audit": {
                "ledger": {"path": str(tmp_path / "final_audit_claim.json")},
                "result": {"path": str(tmp_path / "final_audit_result.json")},
            }
        }
    }
    claim, result = canonical_final_audit_paths(commitment)
    assert_canonical_final_audit_paths(
        commitment=commitment,
        claim_path=claim,
        output_path=result,
    )
    with pytest.raises(ValueError, match="--claim"):
        assert_canonical_final_audit_paths(
            commitment=commitment,
            claim_path=tmp_path / "alternate_claim.json",
            output_path=result,
        )
    with pytest.raises(ValueError, match="--output"):
        assert_canonical_final_audit_paths(
            commitment=commitment,
            claim_path=claim,
            output_path=tmp_path / "alternate_result.json",
        )
def test_native_puffer_scalar_evaluator_smoke_when_native_environment_available():
    pytest.importorskip("pufferlib")
    from temporal_puffer_v4 import NativePufferTemporalPolicy

    torch.manual_seed(91_500_001)
    policy = NativePufferTemporalPolicy(
        hidden_size=96,
        num_layers=1,
        mode_conditioned_heads=False,
    ).eval()
    result = evaluate_seat_balanced_pairs(
        policy,
        deal_seeds=(91_500_001,),
        opponent_seeds=(91_600_001,),
        phase="native-test",
        require_native_policy=True,
    )

    assert result["all_legal_actions_verified"]
    assert result["all_recurrent_resets_verified"]
    assert result["pairs"][0]["candidate_seats"] == [0, 1]
