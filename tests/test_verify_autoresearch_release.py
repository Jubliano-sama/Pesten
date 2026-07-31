import ast
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import verify_autoresearch_release as verifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_checked_in_release_recomputes_all_headline_means_without_games():
    result = verifier.verify_release(PROJECT_ROOT, require_git=False)

    assert result["status"] == "verified"
    assert result["mode"] == "recorded_artifacts_only_no_game_execution"
    assert result["deal_seeds"] == 20_000
    assert result["games"] == 70_000
    assert result["headline"] == {
        "direct_candidate_score": 0.5203,
        "direct_advantage": 0.0406,
        "combined_pool_delta": 0.016783333333333334,
        "opponent_deltas": {
            "random": 0.0227,
            "smart": 0.0217,
            "smarter": 0.00595,
        },
    }


def test_verifier_has_no_game_or_evaluator_imports():
    source = (PROJECT_ROOT / "verify_autoresearch_release.py").read_text(
        encoding="utf-8"
    )
    imported_roots = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])

    assert imported_roots.isdisjoint(
        {
            "game",
            "batched_pesten",
            "best_player",
            "benchmark_checkpoint_batched",
            "benchmark_sealed_suit_symmetry",
        }
    )


def test_corrupt_pair_score_is_detected(monkeypatch):
    original_load = verifier._load_json

    def corrupt_raw(path, *, max_bytes):
        payload = original_load(path, max_bytes=max_bytes)
        if path.name == "sealed_suit_symmetry_final_20260731.json":
            payload = copy.deepcopy(payload)
            payload["direct"]["per_pair"][0]["score"] = 0.25
        return payload

    monkeypatch.setattr(verifier, "_load_json", corrupt_raw)
    with pytest.raises(verifier.VerificationError, match="pair score mismatch"):
        verifier.verify_release(PROJECT_ROOT, require_git=False)


def test_corrupt_summary_hash_is_detected(monkeypatch):
    original_load = verifier._load_json

    def corrupt_summary(path, *, max_bytes):
        payload = original_load(path, max_bytes=max_bytes)
        if path.name == "autoresearch_summary.json":
            payload = copy.deepcopy(payload)
            payload["sealed_evaluation"]["raw_result_sha256"] = "0" * 64
        return payload

    monkeypatch.setattr(verifier, "_load_json", corrupt_summary)
    with pytest.raises(
        verifier.VerificationError,
        match="raw result SHA-256 does not match summary",
    ):
        verifier.verify_release(PROJECT_ROOT, require_git=False)


def test_json_loader_refuses_input_above_explicit_bound(tmp_path):
    artifact = tmp_path / "oversized.json"
    artifact.write_text('{"value": 1}', encoding="utf-8")
    with pytest.raises(verifier.VerificationError, match="oversized JSON"):
        verifier._load_json(artifact, max_bytes=artifact.stat().st_size - 1)


def test_chunked_bootstrap_matches_one_shot_and_bounds_index_array(monkeypatch):
    values = [float(index % 5) for index in range(1_000)]
    seed = int.from_bytes(hashlib.sha256(b"audit").digest()[:4], "big")
    samples = 101
    memory_bound = 8 * len(values)

    reference_rng = np.random.default_rng(seed)
    reference_indices = reference_rng.integers(
        0, len(values), size=(samples, len(values))
    )
    array = np.asarray(values, dtype=np.float64)
    tail = (1.0 - 0.95) / 2.0
    expected = [
        float(value)
        for value in np.quantile(
            array[reference_indices].mean(axis=1),
            (tail, 1.0 - tail),
        )
    ]

    original_default_rng = np.random.default_rng
    observed_bytes = []

    class RecordingGenerator:
        def __init__(self, inner):
            self.inner = inner

        def integers(self, low, high, *, size):
            observed_bytes.append(int(np.prod(size)) * 8)
            return self.inner.integers(low, high, size=size)

    monkeypatch.setattr(
        np.random,
        "default_rng",
        lambda chosen_seed: RecordingGenerator(
            original_default_rng(chosen_seed)
        ),
    )
    actual = verifier._bootstrap_mean_interval(
        np,
        values,
        seed=seed,
        samples=samples,
        max_index_bytes=memory_bound,
    )

    assert actual == expected
    assert observed_bytes
    assert max(observed_bytes) <= memory_bound
