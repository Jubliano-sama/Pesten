import json

import pytest

from random_specialist_workflow import (
    PhaseZeroTracePolicy,
    campaign_preflight,
    run_fixed_random_parity_gate,
    uniform_action_kernel,
)
from specialist_seed_registry import AllocationRequest, SeedRegistry


def test_phase_zero_scalar_and_batched_audited_uniform_parity_is_exact():
    result = run_fixed_random_parity_gate(
        PhaseZeroTracePolicy().eval(),
        deal_seeds=tuple(range(81_100_001, 81_100_005)),
        opponent_seeds=tuple(range(81_200_001, 81_200_005)),
        max_turns=160,
    )

    assert result["passed"]
    assert result["opponent"]["legacy_agent_random"] is False
    assert result["metadata"]["games"] == {"random": 4}
    for case in result["cases"]:
        assert case["passed"], case["mismatches"]
        assert case["recurrent_reset_verified"]
        assert case["scalar_trace_sha256"] == case["batched_trace_sha256"]


def test_uniform_kernel_is_exactly_uniform_over_only_legal_actions():
    legal = tuple(index in (1, 13, 54) for index in range(55))
    kernel = uniform_action_kernel(legal)

    assert kernel == {1: pytest.approx(1 / 3), 13: pytest.approx(1 / 3), 54: pytest.approx(1 / 3)}
    assert sum(kernel.values()) == pytest.approx(1.0)


def test_registry_blocks_derived_ranges_not_only_enumerated_base_seed(tmp_path):
    benchmark_root = tmp_path / "benchmarks"
    experiment_root = tmp_path / "experiments"
    live_root = tmp_path / "live_training"
    benchmark_root.mkdir()
    experiment_root.mkdir()
    live_root.mkdir()
    (benchmark_root / "prior.json").write_text(
        json.dumps({"base_seed": 20_000, "pairs": 50}),
        encoding="utf-8",
    )
    registry = SeedRegistry(
        benchmark_root / "seed_registry.jsonl",
        artifact_roots=(benchmark_root, experiment_root, live_root),
    )

    with pytest.raises(ValueError, match="overlaps"):
        registry.allocate_many(
            [
                AllocationRequest(
                    "calibration",
                    1,
                    "must collide with base_seed + pairs range",
                    20_049,
                )
            ]
        )

    allocated = registry.allocate_many(
        [AllocationRequest("calibration", 4, "safe test range", 21_000)]
    )
    assert allocated["calibration"].as_dict() == {
        "start_inclusive": 21_000,
        "end_exclusive": 21_004,
        "count": 4,
    }
    with pytest.raises(FileExistsError, match="already allocates"):
        registry.allocate_many(
            [AllocationRequest("calibration", 1, "cannot reuse namespace", 22_000)]
        )
    with pytest.raises(ValueError, match="overlaps"):
        registry.allocate_many(
            [
                AllocationRequest(
                    "final_audit",
                    1,
                    "must never reuse consumed old sealed interval",
                    4_294_000_000,
                )
            ]
        )


def test_later_campaign_interface_fails_closed_without_frozen_manifest(tmp_path):
    with pytest.raises(FileNotFoundError, match="frozen campaign manifest"):
        campaign_preflight(
            tmp_path / "frozen_campaign.json",
            lane="recurrent_temporal",
        )
