#!/usr/bin/env python3
"""Screen teacher PPO reward contracts for loss avoidance and win rate."""

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


OPPONENT_SCHEDULE = (
    "current_selfplay",
    "ppo_champion",
    "canonical_random",
)
PLAYER_COUNT_SCHEDULE = (
    (2,) * 14
    + (3,) * 14
    + (4,) * 14
    + (5,) * 6
    + (6,) * 5
    + (7,) * 5
    + (8,) * 5
)
DEFAULT_ARMS = (
    # Orthogonal L9 screen: every level of each factor appears three times and
    # every pair of factor levels appears once.
    {"name": "l9_01", "gamma": 0.99, "decision_reward": -0.02, "draw_reward": -1.0},
    {"name": "l9_02", "gamma": 0.99, "decision_reward": -0.01, "draw_reward": -0.25},
    {"name": "l9_03", "gamma": 0.99, "decision_reward": 0.0, "draw_reward": 0.0},
    {"name": "l9_04", "gamma": 0.995, "decision_reward": -0.02, "draw_reward": -0.25},
    {"name": "l9_05", "gamma": 0.995, "decision_reward": -0.01, "draw_reward": 0.0},
    {"name": "l9_06", "gamma": 0.995, "decision_reward": 0.0, "draw_reward": -1.0},
    {"name": "l9_07", "gamma": 0.9975, "decision_reward": -0.02, "draw_reward": 0.0},
    {"name": "l9_08", "gamma": 0.9975, "decision_reward": -0.01, "draw_reward": -1.0},
    {"name": "l9_09", "gamma": 0.9975, "decision_reward": 0.0, "draw_reward": -0.25},
    # Operational control selected on main before this joint sweep.
    {"name": "control", "gamma": 0.995, "decision_reward": -0.01, "draw_reward": -1.0},
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_arms(arms: list[dict] | tuple[dict, ...]) -> tuple[dict, ...]:
    normalized = []
    names = set()
    contracts = set()
    for raw in arms:
        arm = {
            "name": str(raw["name"]),
            "gamma": float(raw["gamma"]),
            "decision_reward": float(raw["decision_reward"]),
            "draw_reward": float(raw["draw_reward"]),
        }
        if not arm["name"] or arm["name"] in names:
            raise ValueError("reward arms need unique non-empty names")
        contract = (
            arm["gamma"],
            arm["decision_reward"],
            arm["draw_reward"],
        )
        if contract in contracts:
            raise ValueError("reward arms need unique parameter contracts")
        if not 0 < arm["gamma"] <= 1 or not math.isfinite(arm["gamma"]):
            raise ValueError("arm gamma must be finite and in (0, 1]")
        if (
            not math.isfinite(arm["decision_reward"])
            or arm["decision_reward"] > 0
        ):
            raise ValueError("arm decision reward must be finite and nonpositive")
        if (
            not math.isfinite(arm["draw_reward"])
            or not -1 <= arm["draw_reward"] <= 1
        ):
            raise ValueError("arm draw reward must be finite and in [-1, 1]")
        names.add(arm["name"])
        contracts.add(contract)
        normalized.append(arm)
    if len(normalized) < 2:
        raise ValueError("reward sweep needs at least two arms")
    return tuple(normalized)


def load_arms(path: Path | None) -> tuple[dict, ...]:
    if path is None:
        return validate_arms(DEFAULT_ARMS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_arms(payload["arms"] if isinstance(payload, dict) else payload)


def training_command(args: argparse.Namespace, arm: dict, output_dir: Path) -> list[str]:
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
        repr(arm["gamma"]),
        "--decision-reward",
        repr(arm["decision_reward"]),
        "--draw-reward",
        repr(arm["draw_reward"]),
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


def compact_training_summary(result_path: Path) -> dict:
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    curve = payload["curve"]
    return {
        "start_epoch": payload["start_epoch"],
        "last_completed_epoch": payload["last_completed_epoch"],
        "planned_epochs": payload["planned_epochs"],
        "elapsed_seconds": payload["elapsed_seconds"],
        "total_games": sum(
            sum(row["contribution"]["games"].values()) for row in curve
        ),
        "mean_games_per_second": sum(row["games_per_second"] for row in curve)
        / len(curve),
        "maximum_approx_kl": max(row["update"]["approx_kl"] for row in curve),
        "runtime": payload["runtime"],
    }


def ranking_key(row: dict) -> tuple[float, float, float]:
    outcomes = row["screen_weighted_outcomes"]
    # The explicit product objective: avoid losses first, then win, then draw.
    return (
        outcomes["loss"]["after_rate"],
        -outcomes["win"]["after_rate"],
        -outcomes["draw"]["after_rate"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=Path("models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt"),
    )
    parser.add_argument("--champion", type=Path, default=Path("ppo_actor_100run3.pth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arms-json", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--games-per-epoch", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--rollout-workers", type=int, default=10)
    parser.add_argument("--benchmark-workers", type=int, default=10)
    parser.add_argument("--screen-blocks-per-stratum", type=int, default=100)
    parser.add_argument("--confirmation-blocks-per-stratum", type=int, default=600)
    parser.add_argument("--training-seed", type=int, default=202_608_150)
    parser.add_argument("--campaign-offset", type=int, default=9000)
    parser.add_argument("--screen-seed", type=int, default=1_202_608_150)
    parser.add_argument("--direct-confirmation-seed", type=int, default=2_202_608_150)
    parser.add_argument("--baseline-confirmation-seed", type=int, default=3_202_608_150)
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
    if args.campaign_offset < 0:
        raise ValueError("campaign offset must be nonnegative")
    if args.learning_rate <= 0 or not 0 < args.gae_lambda <= 1 or args.target_kl <= 0:
        raise ValueError("learning rate, GAE lambda, and target KL must be positive")
    args.teacher = args.teacher.resolve()
    args.champion = args.champion.resolve()
    for path in (args.teacher, args.champion):
        if not path.is_file():
            raise FileNotFoundError(path)
    arms = load_arms(args.arms_json)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "reward_sweep_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing sweep {manifest_path}")

    manifest = {
        "schema_version": 1,
        "kind": "teacher_min_gru_joint_reward_sweep",
        "status": "running",
        "started_at": utc_now(),
        "teacher": str(args.teacher),
        "teacher_sha256": file_sha256(args.teacher),
        "champion": str(args.champion),
        "champion_sha256": file_sha256(args.champion),
        "protocol": {
            "design": "orthogonal L9 main-effects screen plus operational control",
            "arms": list(arms),
            "epochs_per_arm": args.epochs,
            "games_per_epoch": args.games_per_epoch,
            "identical_teacher_start": True,
            "fresh_optimizer_per_arm": True,
            "shared_training_seed": args.training_seed,
            "shared_campaign_offset": args.campaign_offset,
            "win_reward": 1.0,
            "loss_reward": -1.0,
            "draw_reward_is_distinct_from_loss": True,
            "external_privileged_critic": False,
            "critic_rationale": (
                "No available external critic matches every joint gamma, decision, "
                "and draw reward arm; all arms use the same teacher public value head."
            ),
            "selection_objective": (
                "lexicographic: lowest weighted loss rate, then highest weighted "
                "win rate, then highest weighted draw rate"
            ),
            "evaluation": (
                "fresh matched deals, all seats, 2-8 players, fixed teacher self "
                "anchor, MLP champion, and canonical random"
            ),
            "screen_blocks_per_stratum": args.screen_blocks_per_stratum,
            "confirmation_blocks_per_stratum": args.confirmation_blocks_per_stratum,
        },
        "arms": [],
    }
    atomic_json(manifest, manifest_path)
    started = time.perf_counter()
    try:
        for index, arm in enumerate(arms, start=1):
            print(
                json.dumps(
                    {"kind": "reward_sweep_arm_start", "arm": arm, "index": index, "total": len(arms)},
                    sort_keys=True,
                ),
                flush=True,
            )
            arm_dir = output_dir / arm["name"]
            train_dir = arm_dir / "training"
            command = training_command(args, arm, train_dir)
            row = {**arm, "status": "training", "training_command": command}
            manifest["arms"].append(row)
            atomic_json(manifest, manifest_path)
            run_logged(command, arm_dir / "training.log")
            checkpoint = train_dir / "latest.pt"
            result_path = train_dir / "training_result.json"
            if not checkpoint.is_file() or not result_path.is_file():
                raise RuntimeError(f"arm {arm['name']} produced incomplete outputs")
            row["checkpoint"] = str(checkpoint)
            row["checkpoint_sha256"] = file_sha256(checkpoint)
            row["training_summary"] = compact_training_summary(result_path)
            row["status"] = "screening"
            atomic_json(manifest, manifest_path)

            screen_path = arm_dir / "screen_vs_teacher.json"
            command = benchmark_command(
                args,
                before=args.teacher,
                after=checkpoint,
                output=screen_path,
                blocks=args.screen_blocks_per_stratum,
                seed=args.screen_seed,
            )
            row["screen_command"] = command
            run_logged(command, arm_dir / "screen_vs_teacher.log")
            screen = json.loads(screen_path.read_text(encoding="utf-8"))
            row["screen_evidence"] = str(screen_path)
            row["screen_weighted_result"] = screen["weighted_result"]
            row["screen_weighted_outcomes"] = screen["weighted_outcomes"]
            row["status"] = "complete"
            atomic_json(manifest, manifest_path)
            print(
                json.dumps(
                    {
                        "kind": "reward_sweep_arm_complete",
                        "name": arm["name"],
                        "loss_rate": screen["weighted_outcomes"]["loss"]["after_rate"],
                        "win_rate": screen["weighted_outcomes"]["win"]["after_rate"],
                        "draw_rate": screen["weighted_outcomes"]["draw"]["after_rate"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        ranking = sorted(manifest["arms"], key=ranking_key)
        manifest["screen_ranking"] = [
            {
                "rank": index,
                "name": row["name"],
                "gamma": row["gamma"],
                "decision_reward": row["decision_reward"],
                "draw_reward": row["draw_reward"],
                "loss_rate": row["screen_weighted_outcomes"]["loss"]["after_rate"],
                "win_rate": row["screen_weighted_outcomes"]["win"]["after_rate"],
                "draw_rate": row["screen_weighted_outcomes"]["draw"]["after_rate"],
            }
            for index, row in enumerate(ranking, start=1)
        ]
        winner = ranking[0]
        control = next(row for row in ranking if row["name"] == "control")
        manifest["status"] = "confirming"
        atomic_json(manifest, manifest_path)

        confirmation_dir = output_dir / "confirmation"
        direct_path = confirmation_dir / "screen_winner_vs_control.json"
        direct_command = benchmark_command(
            args,
            before=Path(control["checkpoint"]),
            after=Path(winner["checkpoint"]),
            output=direct_path,
            blocks=args.confirmation_blocks_per_stratum,
            seed=args.direct_confirmation_seed,
        )
        run_logged(direct_command, confirmation_dir / "screen_winner_vs_control.log")
        direct = json.loads(direct_path.read_text(encoding="utf-8"))

        baseline_path = confirmation_dir / "screen_winner_vs_teacher.json"
        baseline_command = benchmark_command(
            args,
            before=args.teacher,
            after=Path(winner["checkpoint"]),
            output=baseline_path,
            blocks=args.confirmation_blocks_per_stratum,
            seed=args.baseline_confirmation_seed,
        )
        run_logged(baseline_command, confirmation_dir / "screen_winner_vs_teacher.log")
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

        direct_loss = direct["weighted_outcomes"]["loss_avoidance"]
        baseline_loss = baseline["weighted_outcomes"]["loss_avoidance"]
        manifest["confirmation"] = {
            "screen_winner": winner["name"],
            "control": control["name"],
            "direct_command": direct_command,
            "direct_evidence": str(direct_path),
            "direct_weighted_result": direct["weighted_result"],
            "direct_weighted_outcomes": direct["weighted_outcomes"],
            "baseline_command": baseline_command,
            "baseline_evidence": str(baseline_path),
            "baseline_weighted_result": baseline["weighted_result"],
            "baseline_weighted_outcomes": baseline["weighted_outcomes"],
        }
        manifest["decision"] = {
            "screen_winner": {key: winner[key] for key in ("name", "gamma", "decision_reward", "draw_reward", "checkpoint", "checkpoint_sha256")},
            "loss_reduction_vs_control_confirmed_at_99_9_percent": direct_loss[
                "delta_normal_ci99_9"
            ][0]
            > 0,
            "loss_reduction_vs_teacher_confirmed_at_99_9_percent": baseline_loss[
                "delta_normal_ci99_9"
            ][0]
            > 0,
            "promotion_authorized": False,
            "note": (
                "This is a reward-contract development screen from a shared "
                "teacher start. Promotion requires a separately authorized, "
                "longer confirmation run."
            ),
        }
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
