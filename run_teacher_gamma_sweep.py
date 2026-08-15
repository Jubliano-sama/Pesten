#!/usr/bin/env python3
"""Train matched teacher-GRU gamma arms and confirm playing-strength effects."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


DEFAULT_GAMMAS = (0.90, 0.25 ** (1.0 / 20.0), 0.97, 0.99)
HISTORICAL_GAMMA = 0.97
OPPONENT_SCHEDULE = (
    "current_selfplay",
    "ppo_champion",
    "canonical_random",
)
# This is the same 63-phase, 2/3/4-heavy schedule used to train the promoted
# epoch-4387 teacher. Keeping it local makes the orchestration script importable
# in the repository's lightweight test environment, which lacks PufferLib.
PLAYER_COUNT_SCHEDULE = (
    (2,) * 14
    + (3,) * 14
    + (4,) * 14
    + (5,) * 6
    + (6,) * 5
    + (7,) * 5
    + (8,) * 5
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gammas(value: str) -> tuple[float, ...]:
    gammas = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if len(gammas) < 2:
        raise argparse.ArgumentTypeError("gamma sweep needs at least two values")
    if any(not math.isfinite(gamma) or not 0 < gamma <= 1 for gamma in gammas):
        raise argparse.ArgumentTypeError("every gamma must be finite and in (0, 1]")
    if len(set(gammas)) != len(gammas):
        raise argparse.ArgumentTypeError("gamma values must be unique")
    return gammas


def gamma_slug(gamma: float) -> str:
    return f"gamma_{gamma:.12g}".replace(".", "p").replace("-", "m")


def retention_summary(gamma: float) -> dict:
    return {
        "gamma": gamma,
        "reward_decay_per_decision": 1.0 - gamma,
        "terminal_reward_retention_after_20_decisions": gamma**20,
    }


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


def training_command(
    args: argparse.Namespace,
    gamma: float,
    output_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "train_puffer_fast.py",
        "--resume",
        str(args.teacher),
        "--output-dir",
        str(output_dir),
        "--champion",
        str(args.champion),
        "--epochs",
        str(args.epochs),
        "--games-per-epoch",
        str(args.games_per_epoch),
        "--rollout-workers",
        str(args.rollout_workers),
        "--rollout-engine",
        "vector",
        "--rollout-precision",
        "float32",
        "--rollout-max-public-events",
        "0",
        "--learning-rate",
        str(args.learning_rate),
        "--fresh-optimizer",
        "--update-epochs",
        str(args.update_epochs),
        "--minibatch-trajectories",
        "128",
        "--diagnostic-minibatch-trajectories",
        "256",
        "--gamma",
        repr(gamma),
        "--gae-lambda",
        str(args.gae_lambda),
        "--target-kl",
        str(args.target_kl),
        "--early-stop-diagnostic-trajectories",
        "512",
        "--checkpoint-every",
        str(args.epochs),
        "--evaluate-every",
        str(args.epochs + 1),
        "--evaluation-pairs",
        "1",
        "--schedule",
        ",".join(OPPONENT_SCHEDULE),
        "--player-count-schedule",
        ",".join(str(value) for value in PLAYER_COUNT_SCHEDULE),
        "--seed",
        str(args.training_seed),
        "--campaign-offset",
        str(args.campaign_offset),
        "--persistent-replay",
        "--compile-updates",
        "--update-precision",
        "bfloat16",
        "--no-pipelined-rollouts",
        "--no-privileged-critic",
        "--device",
        "cuda",
    ]


def benchmark_command(
    args: argparse.Namespace,
    *,
    before: Path,
    after: Path,
    output: Path,
    blocks: int,
    seed: int,
) -> list[str]:
    return [
        sys.executable,
        "benchmark_rlvr_pool_batched.py",
        "--before",
        str(before),
        "--after",
        str(after),
        "--self-anchor",
        str(args.teacher),
        "--mlp",
        str(args.champion),
        "--output",
        str(output),
        "--blocks-per-stratum",
        str(blocks),
        "--workers",
        str(args.benchmark_workers),
        "--chunks-per-stratum",
        "3",
        "--seed",
        str(seed),
    ]


def compact_training_summary(result_path: Path) -> dict:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    curve = payload["curve"]
    return {
        "start_epoch": payload["start_epoch"],
        "last_completed_epoch": payload["last_completed_epoch"],
        "planned_epochs": payload["planned_epochs"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "total_games": sum(sum(row["contribution"]["games"].values()) for row in curve),
        "mean_games_per_second": sum(row["games_per_second"] for row in curve)
        / len(curve),
        "mean_update_seconds": sum(row["update_seconds"] for row in curve) / len(curve),
        "maximum_approx_kl": max(row["update"]["approx_kl"] for row in curve),
        "final_strategic_pass_rate": curve[-1]["strategic_pass"]["rate"],
        "runtime": payload["runtime"],
    }


def ranked_arms(arms: list[dict]) -> list[dict]:
    return sorted(
        arms,
        key=lambda row: row["screen_weighted_result"][
            "delta_normalized_win_share"
        ],
        reverse=True,
    )


def confirmation_reference(ranking: list[dict]) -> dict:
    winner = ranking[0]
    historical = next(
        (
            row
            for row in ranking
            if math.isclose(
                row["gamma"], HISTORICAL_GAMMA, rel_tol=0.0, abs_tol=1e-12
            )
        ),
        None,
    )
    if historical is not None and historical is not winner:
        return historical
    return ranking[1]


def classify_decision(
    *,
    winner_gamma: float,
    reference_gamma: float,
    direct_result: dict,
    baseline_result: dict,
) -> dict:
    direct_lower = direct_result["delta_normalized_ci99_9"][0]
    baseline_lower = baseline_result["delta_normalized_ci99_9"][0]
    preferred_over_reference = direct_lower > 0
    improved_over_start = baseline_lower > 0
    changed_gamma = not math.isclose(
        winner_gamma, HISTORICAL_GAMMA, rel_tol=0.0, abs_tol=1e-12
    )
    gamma_helped = changed_gamma and preferred_over_reference and improved_over_start
    if gamma_helped:
        outcome = "alternate_gamma_confirmed_better"
    elif preferred_over_reference and improved_over_start:
        outcome = "historical_gamma_and_more_training_confirmed_better"
    elif preferred_over_reference:
        outcome = "screen_winner_beats_reference_but_not_starting_teacher"
    elif improved_over_start:
        outcome = "continued_training_helped_but_gamma_preference_unconfirmed"
    else:
        outcome = "no_confirmed_performance_gain"
    return {
        "outcome": outcome,
        "screen_winner_gamma": winner_gamma,
        "confirmation_reference_gamma": reference_gamma,
        "screen_winner_preferred_over_reference_at_99_9_percent": (
            preferred_over_reference
        ),
        "screen_winner_improved_over_start_at_99_9_percent": improved_over_start,
        "alternate_gamma_confirmed_helpful": gamma_helped,
        "promotion_authorized": False,
        "promotion_note": (
            "This experiment selects evidence; promotion remains a separate "
            "explicit action with the checkpoint and evidence recorded together."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt"),
    )
    parser.add_argument("--champion", type=Path, default=Path("ppo_actor_100run3.pth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--gammas",
        type=parse_gammas,
        default=DEFAULT_GAMMAS,
        help="comma-separated gamma values",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--games-per-epoch", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--rollout-workers", type=int, default=10)
    parser.add_argument("--benchmark-workers", type=int, default=10)
    parser.add_argument("--screen-blocks-per-stratum", type=int, default=300)
    parser.add_argument("--confirmation-blocks-per-stratum", type=int, default=1200)
    parser.add_argument("--training-seed", type=int, default=202_608_140)
    parser.add_argument("--campaign-offset", type=int, default=5000)
    parser.add_argument("--screen-seed", type=int, default=1_202_608_140)
    parser.add_argument("--direct-confirmation-seed", type=int, default=2_202_608_140)
    parser.add_argument("--baseline-confirmation-seed", type=int, default=3_202_608_140)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = (
        args.epochs,
        args.games_per_epoch,
        args.update_epochs,
        args.rollout_workers,
        args.benchmark_workers,
        args.screen_blocks_per_stratum,
        args.confirmation_blocks_per_stratum,
    )
    if any(value < 1 for value in counts):
        raise ValueError("epoch, game, worker, and benchmark counts must be positive")
    if args.learning_rate <= 0 or not 0 < args.gae_lambda <= 1 or args.target_kl <= 0:
        raise ValueError("learning rate, GAE lambda, and target KL must be positive")
    if args.campaign_offset < 0:
        raise ValueError("campaign offset must be nonnegative")
    args.teacher = args.teacher.resolve()
    args.champion = args.champion.resolve()
    for path in (args.teacher, args.champion):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "gamma_sweep_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing sweep {manifest_path}")

    manifest = {
        "schema_version": 1,
        "kind": "teacher_min_gru_matched_gamma_sweep",
        "status": "running",
        "started_at": utc_now(),
        "teacher": str(args.teacher),
        "teacher_sha256": file_sha256(args.teacher),
        "champion": str(args.champion),
        "champion_sha256": file_sha256(args.champion),
        "protocol": {
            "gamma_arms": [retention_summary(gamma) for gamma in args.gammas],
            "historical_teacher_gamma": HISTORICAL_GAMMA,
            "epochs_per_arm": args.epochs,
            "games_per_epoch": args.games_per_epoch,
            "training_seed_shared_across_arms": args.training_seed,
            "campaign_offset_shared_across_arms": args.campaign_offset,
            "fresh_optimizer_per_arm": True,
            "opponent_schedule": list(OPPONENT_SCHEDULE),
            "player_count_schedule": list(PLAYER_COUNT_SCHEDULE),
            "update_epochs": args.update_epochs,
            "learning_rate": args.learning_rate,
            "gae_lambda": args.gae_lambda,
            "target_kl": args.target_kl,
            "critic": "actor public value head; external privileged critic disabled",
            "critic_rationale": (
                "Only gamma=0.9330329915368074 has a promoted matching external "
                "critic, so disabling it for every arm avoids a critic-quality confound."
            ),
            "known_bias": (
                "The starting public value head was historically fitted at gamma=0.97; "
                "this conservatively favors the historical arm until other heads adapt."
            ),
            "screen_blocks_per_opponent_player_stratum": (
                args.screen_blocks_per_stratum
            ),
            "confirmation_blocks_per_opponent_player_stratum": (
                args.confirmation_blocks_per_stratum
            ),
            "screen_seed": args.screen_seed,
            "direct_confirmation_seed": args.direct_confirmation_seed,
            "baseline_confirmation_seed": args.baseline_confirmation_seed,
            "evaluation": (
                "fresh deterministic matched deals, all seats, 2-8 players, "
                "fixed starting-teacher self anchor, MLP, and canonical random"
            ),
            "selection_confidence": "paired normal 99.9 percent",
        },
        "arms": [],
    }
    atomic_json(manifest, manifest_path)
    started = time.perf_counter()
    try:
        for gamma in args.gammas:
            arm_dir = output_dir / gamma_slug(gamma)
            train_dir = arm_dir / "training"
            command = training_command(args, gamma, train_dir)
            row = {
                **retention_summary(gamma),
                "status": "training",
                "training_command": command,
            }
            manifest["arms"].append(row)
            atomic_json(manifest, manifest_path)
            run_logged(command, arm_dir / "training.log")
            checkpoint = train_dir / "latest.pt"
            result_path = train_dir / "training_result.json"
            if not checkpoint.is_file() or not result_path.is_file():
                raise RuntimeError(f"gamma {gamma} produced incomplete training outputs")
            row["checkpoint"] = str(checkpoint)
            row["checkpoint_sha256"] = file_sha256(checkpoint)
            row["training_summary"] = compact_training_summary(result_path)
            row["status"] = "screening"
            atomic_json(manifest, manifest_path)

            evaluation_path = arm_dir / "screen_vs_start.json"
            command = benchmark_command(
                args,
                before=args.teacher,
                after=checkpoint,
                output=evaluation_path,
                blocks=args.screen_blocks_per_stratum,
                seed=args.screen_seed,
            )
            row["screen_command"] = command
            run_logged(command, arm_dir / "screen_vs_start.log")
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            row["screen_evidence"] = str(evaluation_path)
            row["screen_weighted_result"] = evaluation["weighted_result"]
            row["screen_opponent_results"] = evaluation["opponent_results"]
            row["status"] = "complete"
            atomic_json(manifest, manifest_path)

        ranking = ranked_arms(manifest["arms"])
        manifest["screen_ranking"] = [
            {
                "rank": index + 1,
                "gamma": row["gamma"],
                "delta_normalized_win_share": row["screen_weighted_result"][
                    "delta_normalized_win_share"
                ],
                "delta_normalized_ci99_9": row["screen_weighted_result"][
                    "delta_normalized_ci99_9"
                ],
            }
            for index, row in enumerate(ranking)
        ]
        winner = ranking[0]
        reference = confirmation_reference(ranking)
        manifest["status"] = "confirming"
        atomic_json(manifest, manifest_path)

        confirmation_dir = output_dir / "confirmation"
        direct_path = confirmation_dir / "screen_winner_vs_reference.json"
        direct_command = benchmark_command(
            args,
            before=Path(reference["checkpoint"]),
            after=Path(winner["checkpoint"]),
            output=direct_path,
            blocks=args.confirmation_blocks_per_stratum,
            seed=args.direct_confirmation_seed,
        )
        run_logged(direct_command, confirmation_dir / "screen_winner_vs_reference.log")
        direct = json.loads(direct_path.read_text(encoding="utf-8"))

        baseline_path = confirmation_dir / "screen_winner_vs_start.json"
        baseline_command = benchmark_command(
            args,
            before=args.teacher,
            after=Path(winner["checkpoint"]),
            output=baseline_path,
            blocks=args.confirmation_blocks_per_stratum,
            seed=args.baseline_confirmation_seed,
        )
        run_logged(baseline_command, confirmation_dir / "screen_winner_vs_start.log")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        manifest["confirmation"] = {
            "screen_winner_gamma": winner["gamma"],
            "reference_gamma": reference["gamma"],
            "direct_command": direct_command,
            "direct_evidence": str(direct_path),
            "direct_weighted_result": direct["weighted_result"],
            "baseline_command": baseline_command,
            "baseline_evidence": str(baseline_path),
            "baseline_weighted_result": baseline["weighted_result"],
        }
        manifest["decision"] = classify_decision(
            winner_gamma=winner["gamma"],
            reference_gamma=reference["gamma"],
            direct_result=direct["weighted_result"],
            baseline_result=baseline["weighted_result"],
        )
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        manifest["finished_at"] = utc_now()
        manifest["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(manifest, manifest_path)
    print(json.dumps(manifest["decision"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
