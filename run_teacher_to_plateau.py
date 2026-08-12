#!/usr/bin/env python3
"""Continue small-teacher RLVR until a confirmed material-gain plateau."""

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
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-phases", type=int, required=True)
    parser.add_argument("--block-epochs", type=int, default=60)
    parser.add_argument("--games-per-epoch", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--monitor-blocks-per-stratum", type=int, default=300)
    parser.add_argument(
        "--confirmation-blocks-per-stratum", type=int, default=600
    )
    parser.add_argument("--plateau-materiality", type=float, default=0.02)
    parser.add_argument("--plateau-consecutive-blocks", type=int, default=2)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=0,
        help="zero continues until confirmed plateau",
    )
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rollout-workers", type=int, default=10)
    parser.add_argument("--benchmark-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_900)
    return parser.parse_args()


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            cwd=Path(__file__).resolve().parent,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def evaluate(
    before: Path,
    after: Path,
    *,
    baseline: Path,
    mlp: Path,
    output: Path,
    blocks: int,
    workers: int,
    seed: int,
) -> dict:
    command = [
        sys.executable,
        "benchmark_rlvr_pool_batched.py",
        "--before",
        str(before),
        "--after",
        str(after),
        "--self-anchor",
        str(baseline),
        "--mlp",
        str(mlp),
        "--output",
        str(output),
        "--blocks-per-stratum",
        str(blocks),
        "--workers",
        str(workers),
        "--chunks-per-stratum",
        "3",
        "--seed",
        str(seed),
    ]
    run_logged(command, output.with_suffix(".log"))
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    positive = (
        args.initial_phases,
        args.block_epochs,
        args.games_per_epoch,
        args.update_epochs,
        args.monitor_blocks_per_stratum,
        args.confirmation_blocks_per_stratum,
        args.plateau_consecutive_blocks,
        args.rollout_workers,
        args.benchmark_workers,
    )
    if any(value < 1 for value in positive):
        raise ValueError(
            "phase, game, update, evaluation, and worker counts must be positive"
        )
    if not 0 < args.plateau_materiality < 1:
        raise ValueError("plateau materiality must be between zero and one")
    if args.max_rounds < 0:
        raise ValueError("max rounds must be nonnegative")
    baseline = args.baseline.resolve()
    current = args.current.resolve()
    mlp = args.mlp.resolve()
    for path in (baseline, current, mlp):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "plateau_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing plateau campaign {manifest_path}"
        )
    manifest = {
        "schema_version": 1,
        "kind": "small_teacher_rlvr_until_confirmed_plateau",
        "status": "running",
        "started_at_unix": time.time(),
        "baseline": str(baseline),
        "initial_checkpoint": str(current),
        "mlp": str(mlp),
        "protocol": {
            "initial_completed_phases": args.initial_phases,
            "block_epochs": args.block_epochs,
            "games_per_epoch": args.games_per_epoch,
            "opponent_schedule": list(OPPONENT_SCHEDULE),
            "player_count_schedule": list(PLAYER_COUNT_SCHEDULE),
            "bulk_2_3_4_fraction": 2 / 3,
            "tail_5_8_fraction": 1 / 3,
            "monitor_blocks_per_stratum": (
                args.monitor_blocks_per_stratum
            ),
            "confirmation_blocks_per_stratum": (
                args.confirmation_blocks_per_stratum
            ),
            "plateau_metric": "weighted normalized win-share delta",
            "plateau_confidence": "paired normal 99.9 percent",
            "plateau_materiality_per_block": args.plateau_materiality,
            "consecutive_monitor_blocks": (
                args.plateau_consecutive_blocks
            ),
            "confirmation_seed_is_disjoint": True,
            "larger_model_training": False,
        },
        "rounds": [],
    }
    atomic_json(manifest, manifest_path)
    phases = args.initial_phases
    consecutive = 0
    plateau_window_start = None
    plateau = False

    try:
        initial_evaluation_path = output_dir / "initial_vs_baseline.json"
        initial_evaluation = evaluate(
            baseline,
            current,
            baseline=baseline,
            mlp=mlp,
            output=initial_evaluation_path,
            blocks=args.monitor_blocks_per_stratum,
            workers=args.benchmark_workers,
            seed=args.seed + 100_000_000,
        )
        manifest["initial_weighted_result"] = initial_evaluation[
            "weighted_result"
        ]
        atomic_json(manifest, manifest_path)

        round_index = 0
        while not plateau:
            round_index += 1
            if args.max_rounds and round_index > args.max_rounds:
                manifest["status"] = "max_rounds_without_plateau"
                break
            round_dir = output_dir / f"round_{round_index:03d}"
            training_dir = round_dir / "training"
            previous = current
            command = [
                sys.executable,
                "train_puffer_fast.py",
                "--resume",
                str(previous),
                "--output-dir",
                str(training_dir),
                "--champion",
                str(mlp),
                "--epochs",
                str(args.block_epochs),
                "--games-per-epoch",
                str(args.games_per_epoch),
                "--rollout-workers",
                str(args.rollout_workers),
                "--rollout-engine",
                "vector",
                "--learning-rate",
                str(args.learning_rate),
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
                str(args.block_epochs),
                "--evaluate-every",
                str(args.block_epochs + 1),
                "--evaluation-pairs",
                "1",
                "--schedule",
                ",".join(OPPONENT_SCHEDULE),
                "--player-count-schedule",
                ",".join(str(value) for value in PLAYER_COUNT_SCHEDULE),
                "--seed",
                str(args.seed),
                "--campaign-offset",
                str(phases),
                "--persistent-replay",
                "--device",
                "cuda",
            ]
            row = {
                "round": round_index,
                "phase_start": phases,
                "phase_end": phases + args.block_epochs - 1,
                "before": str(previous),
                "training_command": command,
                "status": "training",
            }
            manifest["rounds"].append(row)
            atomic_json(manifest, manifest_path)
            run_logged(command, round_dir / "training.log")
            current = training_dir / "latest.pt"
            if not current.is_file():
                raise RuntimeError("teacher plateau block produced no checkpoint")
            phases += args.block_epochs
            row["after"] = str(current)
            row["status"] = "monitoring"
            atomic_json(manifest, manifest_path)

            evaluation_path = round_dir / "incremental_evaluation.json"
            evaluation = evaluate(
                previous,
                current,
                baseline=baseline,
                mlp=mlp,
                output=evaluation_path,
                blocks=args.monitor_blocks_per_stratum,
                workers=args.benchmark_workers,
                seed=args.seed + 200_000_000,
            )
            weighted = evaluation["weighted_result"]
            upper = weighted["delta_normalized_ci99_9"][1]
            signal = upper <= args.plateau_materiality
            row["weighted_result"] = weighted
            row["plateau_signal"] = signal
            row["status"] = "complete"
            if signal:
                if consecutive == 0:
                    plateau_window_start = previous
                consecutive += 1
            else:
                consecutive = 0
                plateau_window_start = None
            row["consecutive_plateau_signals"] = consecutive
            manifest["current_checkpoint"] = str(current)
            manifest["completed_phases"] = phases
            atomic_json(manifest, manifest_path)

            if consecutive < args.plateau_consecutive_blocks:
                continue
            assert plateau_window_start is not None
            confirmation_path = round_dir / "plateau_confirmation.json"
            confirmation = evaluate(
                plateau_window_start,
                current,
                baseline=baseline,
                mlp=mlp,
                output=confirmation_path,
                blocks=args.confirmation_blocks_per_stratum,
                workers=args.benchmark_workers,
                seed=args.seed + 900_000_000 + round_index * 10_000_000,
            )
            confirmation_weighted = confirmation["weighted_result"]
            confirmation_limit = (
                args.plateau_materiality
                * args.plateau_consecutive_blocks
            )
            plateau = (
                confirmation_weighted["delta_normalized_ci99_9"][1]
                <= confirmation_limit
            )
            row["confirmation"] = {
                "path": str(confirmation_path),
                "weighted_result": confirmation_weighted,
                "maximum_material_gain_over_window": confirmation_limit,
                "passed": plateau,
            }
            if not plateau:
                consecutive = 0
                plateau_window_start = None
            atomic_json(manifest, manifest_path)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["finished_at_unix"] = time.time()
        atomic_json(manifest, manifest_path)
        raise

    if plateau:
        manifest["status"] = "confirmed_plateau"
        manifest["plateau_checkpoint"] = str(current)
    manifest["finished_at_unix"] = time.time()
    manifest["elapsed_seconds"] = (
        manifest["finished_at_unix"] - manifest["started_at_unix"]
    )
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
