#!/usr/bin/env python3
"""Paired end-to-end benchmark for segmented native GAE."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from seed_integrity import DEVELOPMENT_SEED_LIMIT


def worker_command(args, seed: int, *, native: bool) -> list[str]:
    return [
        sys.executable,
        "benchmark_training_speed.py",
        "--worker",
        "--implementation",
        "vectorized",
        "--thermal-state",
        "warm",
        "--seed",
        str(seed),
        "--episodes",
        str(args.games),
        "--warmup-episodes",
        str(args.warmup_games),
        "--temperature",
        "0.8",
        "--batch-size",
        str(args.batch_size),
        "--threads",
        str(args.threads),
        "--rollout-device",
        args.rollout_device,
        (
            "--compile-updates"
            if args.compile_updates
            else "--no-compile-updates"
        ),
        "--no-cuda-rollout-autocast",
        "--reuse-cpu-critic-forward",
        "--trace-cpu-rollout-actor",
        "--native-smart-policy",
        "--direct-rollout-sampling",
        "--native-rollout-engine",
        "--no-record-rollout-public-events",
        "--numpy-rollout-masks",
        "--reuse-rollout-workspaces",
        "--coalesce-rollout-agent-sets",
        "--group-compatible-rollout-agent-sets",
        "--coordinate-rollout-engines",
        "--drain-non-neural-rollout",
        "--past-policy-pool-size",
        "2",
        "--native-gae" if native else "--no-native-gae",
    ]


def run_worker(args, seed: int, *, native: bool) -> dict:
    completed = subprocess.run(
        worker_command(args, seed, native=native),
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONHASHSEED": str(seed)},
    )
    return json.loads(completed.stdout.splitlines()[-1])


def exact_contract(tensor: dict, native: dict) -> bool:
    return all(
        tensor[field] == native[field]
        for field in (
            "rollout_sha256",
            "actor_state_sha256",
            "critic_state_sha256",
            "collected_policy_decisions",
            "environment_turns",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--warmup-games", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=85_120_021)
    parser.add_argument(
        "--rollout-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--compile-updates",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_native_gae_20260801.json"),
    )
    args = parser.parse_args()
    if min(
        args.pairs,
        args.games,
        args.warmup_games,
        args.batch_size,
        args.threads,
    ) < 1:
        raise SystemExit("all counts must be positive")
    if (
        args.seed < 0
        or args.seed + args.pairs > DEVELOPMENT_SEED_LIMIT
    ):
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    runs = {"tensor": [], "native": []}
    for pair_index in range(args.pairs):
        seed = args.seed + pair_index
        order = (
            (False, True)
            if pair_index % 2 == 0
            else (True, False)
        )
        for native in order:
            runs["native" if native else "tensor"].append(
                run_worker(args, seed, native=native)
            )

    exact = all(
        exact_contract(tensor, native)
        for tensor, native in zip(runs["tensor"], runs["native"])
    )
    paired_speedups = [
        tensor["total_seconds"] / native["total_seconds"]
        for tensor, native in zip(runs["tensor"], runs["native"])
    ]
    medians = {
        field: {
            name: statistics.median(
                run[field] for run in selected
            )
            for name, selected in runs.items()
        }
        for field in (
            "prep_seconds",
            "update_seconds",
            "rollout_seconds",
            "total_seconds",
        )
    }
    prep_speedup = (
        medians["prep_seconds"]["tensor"]
        / medians["prep_seconds"]["native"]
    )
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_native_gae",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "decision": (
            "accept exact native GAE"
            if exact and prep_speedup >= 1.25
            else "retain tensor GAE"
        ),
        "exact_state_contract": exact,
        "workload": {
            "pairs": args.pairs,
            "games": args.games,
            "warmup_games": args.warmup_games,
            "batch_size": args.batch_size,
            "threads": args.threads,
            "seeds": [
                args.seed + offset for offset in range(args.pairs)
            ],
            "rollout_device": args.rollout_device,
            "compile_updates": args.compile_updates,
        },
        "medians": medians,
        "preparation_speedup": prep_speedup,
        "paired_total_speedups": paired_speedups,
        "paired_total_speedup_median": statistics.median(
            paired_speedups
        ),
        "timing_caveat": (
            "Rollout time was perturbed by a concurrent trainer; "
            "preparation and update components isolate the GAE change."
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
