import json
from pathlib import Path

import pytest

from benchmark_guided_search import derive_seed
from benchmark_hybrid import derive_deal_seed
from seed_integrity import (
    DEVELOPMENT_SEED_LIMIT,
    RETIRED_SEALED_SEED_END,
    RETIRED_SEALED_SEED_START,
    SEALED_SEED_END,
    SEALED_SEED_START,
    claim_sealed_evaluation,
    scan_recorded_seed_interval,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_active_namespace_has_no_recorded_development_seed_contamination():
    uses = scan_recorded_seed_interval(
        (PROJECT_ROOT / "benchmarks", PROJECT_ROOT / "experiments"),
    )
    assert uses == []


def test_scanner_finds_retired_interval_contamination(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    contaminated = root / "development.json"
    contaminated.write_text(
        json.dumps(
            {
                "deal_seed": RETIRED_SEALED_SEED_START + 123,
                "bootstrap_seed_not_used_for_games": (
                    RETIRED_SEALED_SEED_START + 456
                ),
            }
        ),
        encoding="utf-8",
    )

    uses = scan_recorded_seed_interval(
        (root,),
        start=RETIRED_SEALED_SEED_START,
        end=RETIRED_SEALED_SEED_END,
    )
    assert [(use.key_path, use.seed) for use in uses] == [
        ("deal_seed", RETIRED_SEALED_SEED_START + 123)
    ]


def test_scanner_omits_only_explicit_sealed_summary_section(tmp_path):
    root = tmp_path / "benchmarks"
    root.mkdir()
    summary = root / "autoresearch_summary.json"
    summary.write_text(
        json.dumps(
            {
                "sealed_evaluation": {
                    "base_seed": SEALED_SEED_START,
                    "deal_seed": SEALED_SEED_START + 1,
                },
                "development_evidence": {
                    "deal_seed": SEALED_SEED_START + 2,
                },
            }
        ),
        encoding="utf-8",
    )

    allowed = scan_recorded_seed_interval((root,))
    assert [(use.key_path, use.seed) for use in allowed] == [
        ("development_evidence.deal_seed", SEALED_SEED_START + 2)
    ]

    included = scan_recorded_seed_interval(
        (root,),
        allow_sealed_results=False,
    )
    assert [(use.key_path, use.seed) for use in included] == [
        ("sealed_evaluation.base_seed", SEALED_SEED_START),
        ("sealed_evaluation.deal_seed", SEALED_SEED_START + 1),
        ("development_evidence.deal_seed", SEALED_SEED_START + 2),
    ]


def test_development_hash_generators_cannot_enter_active_namespace():
    seeds = {
        derive_deal_seed(20_260_731, namespace, pair_index)
        for namespace in ("direct", "pool:random", "pool:smart", "pool:smarter")
        for pair_index in range(2_000)
    }
    seeds.update(
        derive_seed(20_260_731, pair_index)
        for pair_index in range(2_000)
    )
    assert all(0 <= seed < DEVELOPMENT_SEED_LIMIT for seed in seeds)
    assert all(
        not SEALED_SEED_START <= seed < SEALED_SEED_END
        for seed in seeds
    )


def test_global_claim_is_atomic_and_refuses_alternate_output_rerun(tmp_path):
    ledger = tmp_path / "claims" / "sealed.json"
    first = claim_sealed_evaluation(
        evaluator="test_projection",
        candidate_path=tmp_path / "candidate.pt",
        candidate_sha256="a" * 64,
        base_seed=SEALED_SEED_START,
        pairs=1_000,
        output_path=tmp_path / "first-result.json",
        configuration={"alpha": 1.0, "pool": ["random", "smart"]},
        ledger_path=ledger,
    )

    persisted = json.loads(ledger.read_text(encoding="utf-8"))
    assert persisted == first
    assert persisted["status"] == "holdout_burned_before_evaluation"
    assert persisted["candidate"]["sha256"] == "a" * 64
    assert persisted["base_seed"] == SEALED_SEED_START
    assert persisted["pairs"] == 1_000
    assert persisted["configuration"]["alpha"] == 1.0

    with pytest.raises(FileExistsError, match="already claimed"):
        claim_sealed_evaluation(
            evaluator="different_evaluator",
            candidate_path=tmp_path / "different.pt",
            candidate_sha256="b" * 64,
            base_seed=SEALED_SEED_START + 50_000,
            pairs=10,
            output_path=tmp_path / "alternate-result.json",
            configuration={"alpha": 0.5},
            ledger_path=ledger,
        )

    assert json.loads(ledger.read_text(encoding="utf-8")) == first
