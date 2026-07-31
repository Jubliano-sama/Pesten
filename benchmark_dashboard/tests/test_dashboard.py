from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from benchmark_dashboard.artifact import build_artifact
from benchmark_dashboard.normalize import (
    BenchmarkInputError,
    MetricReconciliationError,
    load_and_normalize,
    normalize_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_FIXTURE = REPO_ROOT / "benchmarks" / "training_performance_20260730.json"
TOURNAMENT_FIXTURE = REPO_ROOT / "benchmarks" / "bayesian_tree_vs_live_ppo_20260731.json"


class NormalizationTests(unittest.TestCase):
    def test_training_performance_normalizes_and_reconciles_rates(self) -> None:
        result = load_and_normalize([TRAINING_FIXTURE], REPO_ROOT)
        self.assertEqual(len(result.runs), 12)
        final = [run for run in result.runs if run.get("stage") == "final"]
        self.assertEqual(len(final), 3)
        first = next(run for run in final if run["provenance"]["seed"] == 20260730)
        self.assertAlmostEqual(
            first["performance"]["games_per_second"],
            100 / 3.8356398250034545,
            places=10,
        )
        self.assertEqual(first["performance"]["policy_decisions"], 4252)
        self.assertIsNone(first["performance"]["environment_turns"])
        self.assertTrue(any("legacy 'samples'" in warning for warning in result.warnings))

    def test_tournament_reconciles_confidence_interval_and_seats(self) -> None:
        result = load_and_normalize([TOURNAMENT_FIXTURE], REPO_ROOT)
        self.assertEqual(len(result.runs), 1)
        run = result.runs[0]
        self.assertEqual(run["game"]["player_count"], 2)
        self.assertEqual(run["outcome"]["games"], 2000)
        self.assertAlmostEqual(run["outcome"]["win_rate"], 838 / 2000)
        self.assertAlmostEqual(
            run["outcome"]["win_rate_ci95"][0],
            0.39755185088828765,
            places=12,
        )
        self.assertEqual(run["performance"]["environment_turns"], 88640)
        self.assertEqual(len(run["outcome"]["seats"]), 2)

    def test_metric_mismatch_is_rejected(self) -> None:
        payload = json.loads(TRAINING_FIXTURE.read_text(encoding="utf-8"))
        payload["final"]["runs"][0]["samples_per_second"] = 1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "training_performance_20260730.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                MetricReconciliationError, "samples_per_second"
            ):
                normalize_file(path, Path(temp))

    def test_malformed_json_is_rejected_with_location(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "broken_20260731.json"
            path.write_text('{"result": ', encoding="utf-8")
            with self.assertRaisesRegex(BenchmarkInputError, "malformed JSON"):
                normalize_file(path, Path(temp))

    def test_tournament_count_mismatch_is_rejected(self) -> None:
        payload = json.loads(TOURNAMENT_FIXTURE.read_text(encoding="utf-8"))
        payload["result"]["wins"] += 1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad_match_20260731.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(
                MetricReconciliationError, "wins \\+ losses \\+ draws"
            ):
                normalize_file(path, Path(temp))

    def test_incomplete_schema_run_is_preserved_as_no_data(self) -> None:
        run = {
            "schema_version": 1,
            "run_id": "future-incomplete-run",
            "run_kind": "training",
            "status": "incomplete",
            "recorded_at": "2026-07-31",
            "game": {
                "player_count": 4,
                "observation_version": "obs-v-next",
                "rules_version": None,
            },
            "models": [{"id": "future_model", "role": "candidate"}],
            "performance": {},
            "outcome": None,
            "timing": [],
            "resources": {},
            "guardrails": [],
            "learning_curve": [],
            "provenance": {
                "source_path": "benchmarks/future_incomplete_20260731.json",
                "source_sha256": "0" * 64,
                "seed": 7,
            },
            "notes": ["Run stopped before metrics were emitted."],
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "future_incomplete_20260731.json"
            path.write_text(json.dumps(run), encoding="utf-8")
            result = load_and_normalize([path], Path(temp))
            self.assertEqual(result.runs[0]["status"], "incomplete")
            self.assertEqual(result.runs[0]["game"]["player_count"], 4)
            artifact = build_artifact(
                result.runs,
                result.source_files,
                normalized_path="benchmarks/dashboard/normalized_runs.json",
            )
            training_rows = artifact["snapshot"]["datasets"]["training_runs"]
            self.assertEqual(training_rows[0]["status"], "incomplete")
            self.assertIsNone(training_rows[0]["total_seconds"])
            coverage = {
                row["view"]: row["status"]
                for row in artifact["snapshot"]["datasets"]["coverage"]
            }
            self.assertEqual(coverage["Learning curves"], "No data")
            self.assertEqual(artifact["snapshot"]["status"], "ready")


class ArtifactTests(unittest.TestCase):
    def test_current_artifact_has_honest_champion_and_requested_datasets(self) -> None:
        inputs = sorted((REPO_ROOT / "benchmarks").glob("*.json"))
        result = load_and_normalize(inputs, REPO_ROOT)
        artifact = build_artifact(
            result.runs,
            result.source_files,
            normalized_path="benchmarks/dashboard/normalized_runs.json",
        )
        datasets = artifact["snapshot"]["datasets"]
        self.assertEqual(datasets["headline"][0]["champion_status"], "Not declared")
        self.assertEqual(
            datasets["headline"][0]["challenge_status"],
            "No eligible declared matchup",
        )
        self.assertEqual(len(datasets["matchups"]), 4)
        self.assertEqual(len(datasets["crossplay"]), 6)
        self.assertEqual(len(datasets["ratings"]), 4)
        self.assertEqual(len(datasets["seed_variance"]), 6)
        self.assertTrue(datasets["time_breakdown"])
        self.assertTrue(
            any(row["status"] == "No data" for row in datasets["coverage"])
        )
        self.assertNotIn("accessIssues", artifact["snapshot"])


if __name__ == "__main__":
    unittest.main()
