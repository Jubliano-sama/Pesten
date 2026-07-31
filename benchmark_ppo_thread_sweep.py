#!/usr/bin/env python3
"""Repeat-controlled CPU thread sweep for one fixed PPO iteration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from seed_integrity import DEVELOPMENT_SEED_LIMIT


def worker_command(args, threads: int) -> list[str]:
    return [
        sys.executable,
        "benchmark_training_speed.py",
        "--worker",
        "--implementation",
        "vectorized",
        "--thermal-state",
        "cold",
        "--seed",
        str(args.seed),
        "--episodes",
        str(args.games),
        "--warmup-episodes",
        "16",
        "--batch-size",
        str(args.rollout_batch_size),
        "--threads",
        str(threads),
        "--rollout-device",
        "cpu",
        "--no-compile-updates",
    ]


def run_worker(args, threads: int) -> dict:
    completed = subprocess.run(
        worker_command(args, threads),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def summarize(runs: list[dict]) -> dict:
    timed_fields = (
        "rollout_seconds",
        "prep_seconds",
        "update_seconds",
        "total_seconds",
        "rss_peak_mib",
    )
    return {
        "repeats": len(runs),
        "median": {
            field: statistics.median(run[field] for run in runs)
            for field in timed_fields
        },
        "runs": runs,
        "repeat_rollout_hash_exact": (
            len({run["rollout_sha256"] for run in runs}) == 1
        ),
        "repeat_actor_state_exact": (
            len({run["actor_state_sha256"] for run in runs}) == 1
        ),
        "repeat_critic_state_exact": (
            len({run["critic_state_sha256"] for run in runs}) == 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, nargs="+", default=[4, 6, 8, 10, 12])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--seed", type=int, default=84_350_001)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_thread_sweep_after_lookups.json"),
    )
    args = parser.parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if any(threads < 1 for threads in args.threads):
        raise SystemExit("thread counts must be positive")
    if 8 not in args.threads:
        raise SystemExit("--threads must include the current default of 8")
    if args.seed < 0 or args.seed + args.games > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("sweep seeds must remain in development range")

    # Rotate the starting thread count by repeat so host drift does not always
    # favor the same candidate while retaining sequential, non-contending runs.
    raw = {str(threads): [] for threads in args.threads}
    for repeat in range(args.repeats):
        order = args.threads[repeat:] + args.threads[:repeat]
        for threads in order:
            raw[str(threads)].append(run_worker(args, threads))
    summaries = {
        threads: summarize(runs) for threads, runs in raw.items()
    }
    rollout_hashes = {
        run["rollout_sha256"]
        for runs in raw.values()
        for run in runs
    }
    actor_hashes = {
        run["actor_state_sha256"]
        for runs in raw.values()
        for run in runs
    }
    critic_hashes = {
        run["critic_state_sha256"]
        for runs in raw.values()
        for run in runs
    }
    winner = min(
        args.threads,
        key=lambda threads: summaries[str(threads)]["median"]["total_seconds"],
    )
    default_total = summaries["8"]["median"]["total_seconds"]
    winner_total = summaries[str(winner)]["median"]["total_seconds"]
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_thread_sweep_after_lookup_tables",
        "sweep_workload": {
            "device": "cpu",
            "fixed_seed": args.seed,
            "games": args.games,
            "rollout_batch_size": args.rollout_batch_size,
            "threads": args.threads,
            "repeats": args.repeats,
        },
        "seed_audit": {
            "development_only": True,
            "sealed_outcomes_used": False,
        },
        "cross_thread_parity": {
            "rollout_hash_exact": len(rollout_hashes) == 1,
            "actor_state_hash_exact": len(actor_hashes) == 1,
            "critic_state_hash_exact": len(critic_hashes) == 1,
            "rollout_hashes": sorted(rollout_hashes),
            "actor_state_hashes": sorted(actor_hashes),
            "critic_state_hashes": sorted(critic_hashes),
        },
        "thread_results": summaries,
        "fastest_threads": winner,
        "fastest_vs_default_eight_speedup": default_total / winner_total,
        "decision": (
            "retain default 8"
            if winner == 8
            else "candidate requires materiality and exact-state review"
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
