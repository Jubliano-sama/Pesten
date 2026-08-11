#!/usr/bin/env python3
"""Run long warm training segments while keeping every benchmark checkpoint.

The original live supervisor restarted after each short four-iteration/four-
epoch cycle.  This supervisor treats those units as micro-cycles and restarts
only after ``--restart-cycles`` of them.  It resumes the latest actor,
critic, optimizer, and native-Puffer optimizer state from the previous live
run, so changing the cadence does not reset learning.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import time

import torch


REPO = Path(__file__).resolve().parent
LIVE_ROOT = REPO / "live_training"
LONG_ROOT = LIVE_ROOT / "long_segments"
MEASURE_ROOT = LIVE_ROOT / "measurements"
SOURCE_FILES = (
    "ppo2.py",
    "temporal_training.py",
    "temporal_puffer_v4.py",
    "temporal_league.py",
    "live_mlp_segment.py",
    "live_puffer_segment.py",
)
DEFAULT_MLP_AGENT_SETS = (
    "smart,rl;smart,rl;random,rl;self,self,rl;self,self,rl;"
    "smart,smart,rl,random;random,random,rl,smart;"
    "smart,random,random,smart,rl,random;"
    "random,smart,random,smart,random,rl,random;"
    "random,smart,random,smart,random,random,rl,random"
)
DEFAULT_PUFFER_PLAYER_COUNT_SCHEDULE = "2,2,2,2,2,2,3,3,4,4,5,6,7,8"
DEFAULT_PUFFER_SCHEDULE = (
    "current_selfplay,current_selfplay,current_selfplay,"
    "historical_temporal,historical_temporal,ppo_champion,ppo_champion,"
    "ppo_champion,ppo_champion,bayesian_tree,smart_rule,smarter_rule,"
    "random_rule"
)
DEFAULT_PUFFER_SCHEDULE_ALT = (
    "current_selfplay,current_selfplay,current_selfplay,current_selfplay,"
    "historical_temporal,historical_temporal,ppo_champion,ppo_champion,"
    "bayesian_tree,smart_rule,smarter_rule,random_rule,random_rule"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint() -> dict[str, str]:
    return {name: _sha256(REPO / name) for name in SOURCE_FILES}


def _deadline(value: str) -> datetime:
    hour, minute = (int(part) for part in value.split(":", 1))
    now = datetime.now().astimezone()
    deadline = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    return deadline


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    print(
        json.dumps(
            {
                "command": command[0],
                "log": str(log_path),
                "returncode": completed.returncode,
                "elapsed_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return completed.returncode


def _load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _measurement_summary(path: Path) -> dict:
    result = _load_json(path)
    if "results" in result:
        return {
            opponent: {
                "score": payload.get("score"),
                "ci95": payload.get("paired_bootstrap_ci95"),
                "games": payload.get("games"),
            }
            for opponent, payload in result["results"].items()
        }
    return {
        opponent["opponent"]: {
            "score": opponent["candidate_score"].get("mean"),
            "ci95": opponent["candidate_score"].get("bootstrap_ci95"),
            "delta_vs_old": opponent["paired_delta"].get("mean"),
        }
        for opponent in result.get("pool", [])
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", default="16:00")
    parser.add_argument(
        "--no-deadline",
        action="store_true",
        help="train indefinitely until the supervisor process is stopped",
    )
    parser.add_argument("--restart-cycles", type=int, default=16)
    parser.add_argument("--base-mlp-iterations", type=int, default=4)
    parser.add_argument("--mlp-episodes", type=int, default=512)
    parser.add_argument("--base-puffer-epochs", type=int, default=4)
    parser.add_argument("--puffer-games-per-epoch", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument(
        "--benchmark-every",
        type=int,
        default=1,
        help="run external measurements once every N segments; other "
        "segments skip measurements entirely (default 1)",
    )
    parser.add_argument(
        "--benchmark-pairs",
        type=int,
        default=0,
        help="seat-balanced pairs per measurement run when one is scheduled; "
        "0 falls back to --pairs",
    )
    parser.add_argument("--mlp-seed", type=int, default=2026080101)
    parser.add_argument("--puffer-seed", type=int, default=54010001)
    parser.add_argument(
        "--resume-state",
        type=Path,
        default=LIVE_ROOT / "supervisor_state.json",
    )
    parser.add_argument("--resume-mlp-actor", type=Path)
    parser.add_argument("--resume-mlp-critic", type=Path)
    parser.add_argument("--resume-mlp-actor-optimizer", type=Path)
    parser.add_argument("--resume-mlp-critic-optimizer", type=Path)
    parser.add_argument("--resume-puffer", type=Path)
    parser.add_argument("--resume-puffer-epoch", type=int)
    parser.add_argument("--start-segment", type=int, default=1)
    parser.add_argument("--mlp-agent-sets", default=DEFAULT_MLP_AGENT_SETS)
    parser.add_argument("--mlp-temperature", type=float, default=0.8)
    parser.add_argument("--mlp-past-policy-probability", type=float, default=0.35)
    parser.add_argument("--puffer-schedule", default=DEFAULT_PUFFER_SCHEDULE)
    parser.add_argument("--puffer-schedule-alt", default=DEFAULT_PUFFER_SCHEDULE_ALT)
    parser.add_argument(
        "--puffer-player-count-schedule",
        default=DEFAULT_PUFFER_PLAYER_COUNT_SCHEDULE,
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.restart_cycles < 1:
        raise ValueError("restart-cycles must be positive")
    if args.benchmark_every < 1:
        raise ValueError("benchmark-every must be positive")
    if args.benchmark_pairs < 0:
        raise ValueError("benchmark-pairs must be non-negative")
    benchmark_pairs = args.benchmark_pairs or args.pairs
    training_device = "cuda" if torch.cuda.is_available() else "cpu"
    deadline = None if args.no_deadline else _deadline(args.deadline)
    old_state = _load_json(args.resume_state.resolve())
    warm = old_state.get("current_warm_resume")
    if not warm or not warm.get("mlp") or not warm.get("puffer"):
        raise ValueError("resume state has no complete warm checkpoint set")

    override_mlp = {
        "actor": args.resume_mlp_actor,
        "critic": args.resume_mlp_critic,
        "actor_optimizer": args.resume_mlp_actor_optimizer,
        "critic_optimizer": args.resume_mlp_critic_optimizer,
    }
    if any(value is not None for value in override_mlp.values()):
        if any(value is None for value in override_mlp.values()):
            raise ValueError(
                "all four MLP resume paths are required when overriding resume"
            )
        mlp_resume = {
            name: Path(value).resolve()
            for name, value in override_mlp.items()
        }
    else:
        mlp_resume = {
            name: Path(value).resolve() for name, value in warm["mlp"].items()
        }
    puffer_resume = (
        args.resume_puffer.resolve()
        if args.resume_puffer is not None
        else Path(warm["puffer"]).resolve()
    )
    puffer_epoch = (
        args.resume_puffer_epoch
        if args.resume_puffer_epoch is not None
        else int(warm.get("puffer_epoch", 8))
    )
    if args.start_segment < 1:
        raise ValueError("start-segment must be positive")
    for name, path in mlp_resume.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing warm MLP {name}: {path}")
    if not puffer_resume.is_file():
        raise FileNotFoundError(f"missing warm Puffer checkpoint: {puffer_resume}")

    mlp_iterations = args.base_mlp_iterations * args.restart_cycles
    puffer_epochs = args.base_puffer_epochs * args.restart_cycles
    mlp_python = REPO / "venv/bin/python"
    puffer_python = REPO / "puffer4-native/.venv/bin/python"
    if not mlp_python.is_file() or not puffer_python.is_file():
        raise FileNotFoundError("one of the required project Python runtimes is missing")

    LONG_ROOT.mkdir(parents=True, exist_ok=True)
    MEASURE_ROOT.mkdir(parents=True, exist_ok=True)
    segment = args.start_segment
    prior_segments = list(old_state.get("segments", []))
    state = {
        "kind": "long_warm_training_supervisor",
        "deadline": deadline.isoformat() if deadline is not None else None,
        "started_at": datetime.now().astimezone().isoformat(),
        "resumed_from": str(args.resume_state.resolve()),
        "previous_cycles": (
            len(old_state.get("cycles", [])) + len(prior_segments)
        ),
        "restart_cycles": args.restart_cycles,
        "mlp_iterations_per_segment": mlp_iterations,
        "puffer_epochs_per_segment": puffer_epochs,
        "benchmark": {
            "every_segments": args.benchmark_every,
            "pairs": benchmark_pairs,
        },
        "recipe": {
            "mlp_agent_sets": args.mlp_agent_sets,
            "mlp_temperature": args.mlp_temperature,
            "mlp_past_policy_probability": args.mlp_past_policy_probability,
            "puffer_schedule": args.puffer_schedule,
            "puffer_schedule_alt": args.puffer_schedule_alt,
            "puffer_player_count_schedule": args.puffer_player_count_schedule,
            "device": training_device,
        },
        "source_fingerprint": source_fingerprint(),
        "warm_initialization": {
            "mlp": {name: str(path) for name, path in mlp_resume.items()},
            "puffer": str(puffer_resume),
            "puffer_epoch": puffer_epoch,
        },
        "segments": prior_segments,
    }
    state_path = LIVE_ROOT / "long_supervisor_state.json"

    def write_state() -> None:
        state["updated_at"] = datetime.now().astimezone().isoformat()
        state_path.write_text(
            json.dumps(state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    write_state()
    while (
        deadline is None
        or datetime.now().astimezone() < deadline
    ):
        segment_started = datetime.now().astimezone()
        run_benchmarks = (
            (segment - args.start_segment) % args.benchmark_every == 0
        )
        record = {
            "segment": segment,
            "started_at": segment_started.isoformat(),
            "micro_cycles": args.restart_cycles,
            "benchmark_scheduled": run_benchmarks,
            "source_fingerprint_before": source_fingerprint(),
        }
        segment_root = LONG_ROOT / f"segment_{segment:04d}"
        mlp_dir = segment_root / "mlp"
        mlp_command = [
            str(mlp_python), str(REPO / "live_mlp_segment.py"),
            "--resume-actor", str(mlp_resume["actor"]),
            "--resume-critic", str(mlp_resume["critic"]),
            "--resume-actor-optimizer", str(mlp_resume["actor_optimizer"]),
            "--resume-critic-optimizer", str(mlp_resume["critic_optimizer"]),
            "--output-dir", str(mlp_dir),
            "--iterations", str(mlp_iterations),
            "--episodes", str(args.mlp_episodes),
            "--seed", str(args.mlp_seed + segment),
            "--temperature", str(args.mlp_temperature),
            "--past-policy-probability", str(args.mlp_past_policy_probability),
            "--agent-sets", args.mlp_agent_sets,
        ]
        record["mlp_returncode"] = _run(mlp_command, mlp_dir / "training.log")
        if record["mlp_returncode"] == 0:
            record["mlp_segment"] = _load_json(mlp_dir / "segment_result.json")
        else:
            record["mlp_segment"] = {
                "error": "missing_segment_result",
                "path": str(mlp_dir / "segment_result.json"),
            }
        if record["mlp_returncode"] != 0:
            state["segments"].append(record)
            write_state()
            break

        mlp_resume = {
            "actor": (mlp_dir / f"ppo_actor_{mlp_iterations}.pth").resolve(),
            "critic": (mlp_dir / f"ppo_critic_{mlp_iterations}.pth").resolve(),
            "actor_optimizer": (mlp_dir / f"ppo_actor_opt_{mlp_iterations}.pth").resolve(),
            "critic_optimizer": (mlp_dir / f"ppo_critic_opt_{mlp_iterations}.pth").resolve(),
        }
        mlp_measurement = MEASURE_ROOT / f"mlp_long_segment_{segment:04d}_vs_old_random.json"
        record["mlp_benchmark_returncode"] = None
        record["mlp_measurement"] = {"path": str(mlp_measurement)}
        if run_benchmarks:
            record["mlp_benchmark_returncode"] = _run(
                [
                    str(mlp_python), str(REPO / "benchmark_checkpoint_batched.py"),
                    "--candidate", str(mlp_resume["actor"]),
                    "--parent", str(REPO / "ppo_actor_100run3.pth"),
                    "--pairs", str(benchmark_pairs),
                    "--seed", str(82806001 + segment * 10_000),
                    "--pool", "random", "--device", training_device,
                    "--canonical-throughput-pairs", "0",
                    "--output", str(mlp_measurement),
                ],
                MEASURE_ROOT / f"mlp_long_segment_{segment:04d}_benchmark.log",
            )
            if record["mlp_benchmark_returncode"] == 0:
                record["mlp_measurement"]["summary"] = _measurement_summary(
                    mlp_measurement
                )
            else:
                record["mlp_measurement"]["error"] = (
                    "benchmark_returncode_"
                    + str(record["mlp_benchmark_returncode"])
                )

        # Make the completed MLP side resumable even if the Puffer child or
        # its benchmark later fails.
        state["current_warm_resume"] = {
            "mlp": {name: str(path) for name, path in mlp_resume.items()},
            "puffer": str(puffer_resume),
            "puffer_epoch": puffer_epoch,
        }
        write_state()

        puffer_dir = segment_root / "puffer"
        puffer_schedule = (
            args.puffer_schedule
            if segment % 2
            else args.puffer_schedule_alt
        )
        puffer_command = [
            str(puffer_python), str(REPO / "live_puffer_segment.py"),
            "--resume", str(puffer_resume),
            "--output-dir", str(puffer_dir),
            "--epochs", str(puffer_epochs),
            "--global-epoch", str(puffer_epoch + 1),
            "--seed", str(args.puffer_seed + segment),
            "--games-per-epoch", str(args.puffer_games_per_epoch),
            "--development-pairs", str(max(10, min(50, args.pairs // 5))),
            "--device", training_device,
            "--schedule", puffer_schedule,
            "--player-count-schedule", args.puffer_player_count_schedule,
        ]
        record["puffer_returncode"] = _run(puffer_command, puffer_dir / "training.log")
        if record["puffer_returncode"] == 0:
            record["puffer_segment"] = _load_json(
                puffer_dir / "segment_result.json"
            )
        else:
            record["puffer_segment"] = {
                "error": "missing_segment_result",
                "path": str(puffer_dir / "segment_result.json"),
            }
        if record["puffer_returncode"] != 0:
            state["segments"].append(record)
            write_state()
            break

        puffer_epoch += puffer_epochs
        puffer_resume = (puffer_dir / "latest.pt").resolve()
        puffer_measurement = MEASURE_ROOT / f"puffer_long_segment_{segment:04d}_vs_old_random.json"
        record["puffer_benchmark_returncode"] = None
        record["puffer_measurement"] = {"path": str(puffer_measurement)}
        if run_benchmarks:
            record["puffer_benchmark_returncode"] = _run(
                [
                    str(puffer_python), str(REPO / "live_puffer_benchmark.py"),
                    "--checkpoint", str(puffer_resume),
                    "--pairs", str(benchmark_pairs),
                    "--seed", str(args.puffer_seed + segment * 10_000),
                    "--output", str(puffer_measurement),
                ],
                MEASURE_ROOT / f"puffer_long_segment_{segment:04d}_benchmark.log",
            )
            if record["puffer_benchmark_returncode"] == 0:
                record["puffer_measurement"]["summary"] = _measurement_summary(
                    puffer_measurement
                )
            else:
                record["puffer_measurement"]["error"] = (
                    "benchmark_returncode_"
                    + str(record["puffer_benchmark_returncode"])
                )
        record["recipe"] = {"puffer_schedule": puffer_schedule}
        record["source_fingerprint_after"] = source_fingerprint()
        record["finished_at"] = datetime.now().astimezone().isoformat()
        state["segments"].append(record)
        state["current_warm_resume"] = {
            "mlp": {name: str(path) for name, path in mlp_resume.items()},
            "puffer": str(puffer_resume),
            "puffer_epoch": puffer_epoch,
        }
        state["source_fingerprint"] = source_fingerprint()
        write_state()
        print(
            json.dumps(
                {
                    "segment": segment,
                    "micro_cycles": args.restart_cycles,
                    "benchmark_scheduled": run_benchmarks,
                    "mlp": record["mlp_measurement"].get("summary"),
                    "puffer": record["puffer_measurement"].get("summary"),
                    "source_changed": (
                        record["source_fingerprint_before"]
                        != record["source_fingerprint_after"]
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        segment += 1

    state["finished_at"] = datetime.now().astimezone().isoformat()
    state["finished_reason"] = (
        "no_deadline_child_failure"
        if deadline is None
        else "deadline_or_child_failure"
    )
    write_state()
    print(
        json.dumps(
            {
                "finished_at": state["finished_at"],
                "deadline": state["deadline"],
                "segments_completed": len(state["segments"]),
                "state": str(state_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
