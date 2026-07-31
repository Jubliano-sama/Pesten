"""CLI for normalized benchmark data and canonical dashboard artifact output."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from .artifact import build_artifact
from .normalize import BenchmarkInputError, discover_inputs, load_and_normalize


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def build(
    *,
    root: Path,
    inputs: list[Path],
    normalized_output: Path,
    sqlite_output: Path,
    artifact_output: Path,
) -> dict[str, object]:
    result = load_and_normalize(inputs, root)
    normalized_payload = {
        "schema_version": 1,
        "generated_from": result.source_files,
        "warnings": result.warnings,
        "runs": result.runs,
    }
    normalized_output.parent.mkdir(parents=True, exist_ok=True)
    normalized_output.write_text(
        json.dumps(normalized_payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    sqlite_output.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(sqlite_output)
    try:
        connection.executescript(
            """
            DROP TABLE IF EXISTS benchmark_runs;
            DROP TABLE IF EXISTS benchmark_sources;
            CREATE TABLE benchmark_runs (
                run_id TEXT PRIMARY KEY,
                recorded_at TEXT NOT NULL,
                run_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                player_count INTEGER,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE benchmark_sources (
                path TEXT PRIMARY KEY,
                sha256 TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO benchmark_runs (
                run_id, recorded_at, run_kind, status, player_count, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run["run_id"],
                    run["recorded_at"],
                    run["run_kind"],
                    run["status"],
                    run.get("game", {}).get("player_count"),
                    json.dumps(run, sort_keys=True, separators=(",", ":")),
                )
                for run in result.runs
            ],
        )
        connection.executemany(
            "INSERT INTO benchmark_sources (path, sha256, recorded_at) VALUES (?, ?, ?)",
            [
                (source["path"], source["sha256"], source["recorded_at"])
                for source in result.source_files
            ],
        )
        connection.commit()
        artifact_runs = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT payload_json FROM benchmark_runs ORDER BY run_id"
            )
        ]
    finally:
        connection.close()
    artifact = build_artifact(
        artifact_runs,
        result.source_files,
        normalized_path=_relative(sqlite_output, root),
    )
    artifact_output.parent.mkdir(parents=True, exist_ok=True)
    artifact_output.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return {
        "inputs": [_relative(path, root) for path in inputs],
        "run_count": len(result.runs),
        "warnings": result.warnings,
        "normalized_output": _relative(normalized_output, root),
        "sqlite_output": _relative(sqlite_output, root),
        "artifact_output": _relative(artifact_output, root),
        "snapshot_status": artifact["snapshot"]["status"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the Pesten benchmark dashboard artifact."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for safe relative source paths.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("benchmarks"),
        help="Directory containing top-level benchmark JSON inputs.",
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Explicit input JSON file; repeat to provide multiple inputs.",
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        default=Path("benchmarks/dashboard/normalized_runs.json"),
    )
    parser.add_argument(
        "--sqlite-output",
        type=Path,
        default=Path("benchmarks/dashboard/benchmark_runs.sqlite"),
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=Path("benchmarks/dashboard/artifact.json"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    input_dir = args.input_dir
    if not input_dir.is_absolute():
        input_dir = root / input_dir
    inputs = args.input or discover_inputs(input_dir)
    inputs = [path if path.is_absolute() else root / path for path in inputs]
    normalized_output = args.normalized_output
    if not normalized_output.is_absolute():
        normalized_output = root / normalized_output
    artifact_output = args.artifact_output
    if not artifact_output.is_absolute():
        artifact_output = root / artifact_output
    sqlite_output = args.sqlite_output
    if not sqlite_output.is_absolute():
        sqlite_output = root / sqlite_output
    try:
        receipt = build(
            root=root,
            inputs=inputs,
            normalized_output=normalized_output,
            sqlite_output=sqlite_output,
            artifact_output=artifact_output,
        )
    except BenchmarkInputError as exc:
        print(f"benchmark dashboard generation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
