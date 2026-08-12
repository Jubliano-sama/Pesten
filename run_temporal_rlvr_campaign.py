#!/usr/bin/env python3
"""Run matched RLVR and high-confidence pool evaluation for temporal models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from sota_distill_temporal import PLAYER_COUNT_SCHEDULE


OPPONENT_SCHEDULE = (
    "current_selfplay",
    "ppo_champion",
    "canonical_random",
)


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--alternative", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--games-per-epoch", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--evaluation-pairs", type=int, default=500)
    parser.add_argument("--blocks-per-stratum", type=int, default=300)
    parser.add_argument("--rollout-workers", type=int, default=10)
    parser.add_argument("--benchmark-workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20_260_850)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = (
        args.epochs,
        args.games_per_epoch,
        args.update_epochs,
        args.evaluation_pairs,
        args.blocks_per_stratum,
        args.rollout_workers,
        args.benchmark_workers,
    )
    if any(value < 1 for value in counts):
        raise ValueError("campaign counts must be positive")
    inputs = {
        "teacher": args.teacher.resolve(),
        "large_min_gru": args.large.resolve(),
        "hawk_residual": args.alternative.resolve(),
    }
    mlp = args.mlp.resolve()
    for path in (*inputs.values(), mlp):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "campaign_manifest.json"
    manifest = {
        "schema_version": 1,
        "kind": "temporal_rlvr_population_campaign",
        "status": "running",
        "started_at_unix": time.time(),
        "inputs": {name: str(path) for name, path in inputs.items()},
        "mlp": str(mlp),
        "protocol": {
            "epochs": args.epochs,
            "games_per_epoch": args.games_per_epoch,
            "opponent_schedule": list(OPPONENT_SCHEDULE),
            "player_count_schedule": list(PLAYER_COUNT_SCHEDULE),
            "bulk_2_3_4_fraction": 2 / 3,
            "tail_5_8_fraction": 1 / 3,
            "blocks_per_evaluation_stratum": args.blocks_per_stratum,
        },
        "models": {},
    }
    atomic_json(manifest, manifest_path)
    player_counts = ",".join(str(value) for value in PLAYER_COUNT_SCHEDULE)
    schedule = ",".join(OPPONENT_SCHEDULE)
    learning_rates = {
        "teacher": 5e-5,
        "large_min_gru": 7.5e-5,
        "hawk_residual": 7.5e-5,
    }

    try:
        for index, (name, before) in enumerate(inputs.items()):
            model_dir = output_dir / name
            training_dir = model_dir / "training"
            model_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "train_puffer_fast.py",
                "--resume",
                str(before),
                "--output-dir",
                str(training_dir),
                "--champion",
                str(mlp),
                "--epochs",
                str(args.epochs),
                "--games-per-epoch",
                str(args.games_per_epoch),
                "--rollout-workers",
                str(args.rollout_workers),
                "--rollout-engine",
                "vector",
                "--learning-rate",
                str(learning_rates[name]),
                "--update-epochs",
                str(args.update_epochs),
                "--minibatch-trajectories",
                "128",
                "--diagnostic-minibatch-trajectories",
                "256",
                "--target-kl",
                "0.015",
                "--early-stop-diagnostic-trajectories",
                "512",
                "--checkpoint-every",
                "10",
                "--evaluate-every",
                str(args.epochs),
                "--evaluation-pairs",
                str(args.evaluation_pairs),
                "--schedule",
                schedule,
                "--player-count-schedule",
                player_counts,
                "--seed",
                str(args.seed),
                "--persistent-replay",
                "--device",
                "cuda",
            ]
            if name == "hawk_residual":
                command.append("--no-compile-updates")
            training_log = model_dir / "training.log"
            model_row = {
                "before": str(before),
                "learning_rate": learning_rates[name],
                "training_command": command,
                "training_log": str(training_log),
                "status": "training",
            }
            manifest["models"][name] = model_row
            atomic_json(manifest, manifest_path)
            with training_log.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    command,
                    check=True,
                    cwd=Path(__file__).resolve().parent,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            after = training_dir / "latest.pt"
            if not after.is_file():
                raise RuntimeError(f"{name} RLVR did not produce latest.pt")
            model_row["after"] = str(after)
            model_row["status"] = "evaluating"
            evaluation_path = model_dir / "matched_pool_evaluation.json"
            evaluation_log = model_dir / "evaluation.log"
            evaluation_command = [
                sys.executable,
                "benchmark_rlvr_pool.py",
                "--before",
                str(before),
                "--after",
                str(after),
                "--mlp",
                str(mlp),
                "--output",
                str(evaluation_path),
                "--blocks-per-stratum",
                str(args.blocks_per_stratum),
                "--workers",
                str(args.benchmark_workers),
                "--chunks-per-stratum",
                "3",
                "--seed",
                str(args.seed + 100_000_000 * (index + 1)),
            ]
            model_row["evaluation_command"] = evaluation_command
            atomic_json(manifest, manifest_path)
            with evaluation_log.open("w", encoding="utf-8") as handle:
                subprocess.run(
                    evaluation_command,
                    check=True,
                    cwd=Path(__file__).resolve().parent,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            model_row["weighted_result"] = evaluation["weighted_result"]
            model_row["status"] = "complete"
            atomic_json(manifest, manifest_path)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["finished_at_unix"] = time.time()
        atomic_json(manifest, manifest_path)
        raise

    ranked = sorted(
        (
            {
                "model": name,
                **row["weighted_result"],
            }
            for name, row in manifest["models"].items()
        ),
        key=lambda row: row["delta_normalized_win_share"],
        reverse=True,
    )
    manifest["ranking_by_matched_normalized_delta"] = ranked
    manifest["status"] = "complete"
    manifest["finished_at_unix"] = time.time()
    manifest["elapsed_seconds"] = (
        manifest["finished_at_unix"] - manifest["started_at_unix"]
    )
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
