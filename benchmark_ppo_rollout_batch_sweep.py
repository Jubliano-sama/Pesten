#!/usr/bin/env python3
"""Repeat-controlled exact PPO rollout-batch-size throughput sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from seed_integrity import DEVELOPMENT_SEED_LIMIT


DEFAULT_BATCH_SIZES = (128, 256, 512, 1024)
CURRENT_DEFAULT = 256


def worker_command(args, batch_size: int) -> list[str]:
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
        str(batch_size),
        "--threads",
        str(args.threads),
        "--rollout-device",
        "cpu",
        "--trace-cpu-rollout-actor",
        "--no-compile-updates",
    ]


def run_worker(args, batch_size: int) -> dict:
    completed = subprocess.run(
        worker_command(args, batch_size),
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
        "rollout_buffer_peak_mib",
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


def exact_hash_contract(raw: dict[str, list[dict]]) -> dict:
    rollout_hashes = {
        run["rollout_sha256"] for runs in raw.values() for run in runs
    }
    actor_hashes = {
        run["actor_state_sha256"] for runs in raw.values() for run in runs
    }
    critic_hashes = {
        run["critic_state_sha256"] for runs in raw.values() for run in runs
    }
    return {
        "rollout_hash_exact": len(rollout_hashes) == 1,
        "actor_state_hash_exact": len(actor_hashes) == 1,
        "critic_state_hash_exact": len(critic_hashes) == 1,
        "rollout_hashes": sorted(rollout_hashes),
        "actor_state_hashes": sorted(actor_hashes),
        "critic_state_hashes": sorted(critic_hashes),
    }


def select_batch_size(
    summaries: dict[str, dict],
    parity: dict,
    *,
    current_default: int = CURRENT_DEFAULT,
    material_speedup: float = 1.02,
) -> dict:
    fastest = min(
        (int(value) for value in summaries),
        key=lambda value: summaries[str(value)]["median"]["total_seconds"],
    )
    default_seconds = summaries[str(current_default)]["median"][
        "total_seconds"
    ]
    fastest_seconds = summaries[str(fastest)]["median"]["total_seconds"]
    speedup = default_seconds / fastest_seconds
    exact = all(
        parity[field]
        for field in (
            "rollout_hash_exact",
            "actor_state_hash_exact",
            "critic_state_hash_exact",
        )
    )
    selected = (
        fastest
        if exact and speedup >= material_speedup
        else current_default
    )
    return {
        "fastest_batch_size": fastest,
        "fastest_vs_current_default_speedup": speedup,
        "material_speedup_threshold": material_speedup,
        "selected_batch_size": selected,
        "decision": (
            f"select {fastest}"
            if selected == fastest and fastest != current_default
            else f"retain current default {current_default}"
        ),
        "exactness_required": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=84_920_001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_rollout_batch_sweep_20260731.json"),
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.games < 1 or args.threads < 1:
        raise SystemExit("repeats, games, and threads must be positive")
    if len(set(args.batch_sizes)) != len(args.batch_sizes):
        raise SystemExit("batch sizes must be unique")
    if any(batch_size < 1 for batch_size in args.batch_sizes):
        raise SystemExit("batch sizes must be positive")
    if CURRENT_DEFAULT not in args.batch_sizes:
        raise SystemExit(
            f"batch sizes must include current default {CURRENT_DEFAULT}"
        )
    if args.seed < 0 or args.seed + args.games > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("sweep seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    raw = {str(batch_size): [] for batch_size in args.batch_sizes}
    for repeat in range(args.repeats):
        offset = repeat % len(args.batch_sizes)
        order = args.batch_sizes[offset:] + args.batch_sizes[:offset]
        for batch_size in order:
            raw[str(batch_size)].append(run_worker(args, batch_size))
    summaries = {
        batch_size: summarize(runs) for batch_size, runs in raw.items()
    }
    parity = exact_hash_contract(raw)
    decision = select_batch_size(summaries, parity)
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_rollout_batch_size_sweep",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "workload": {
            "fixed_seed": args.seed,
            "games": args.games,
            "batch_sizes": args.batch_sizes,
            "repeats": args.repeats,
            "threads": args.threads,
            "thermal_state": "cold",
            "device": "cpu",
        },
        "seed_audit": {
            "development_only": True,
            "sealed_inputs_used": False,
            "development_seed_limit": DEVELOPMENT_SEED_LIMIT,
        },
        "cross_batch_parity": parity,
        "batch_results": summaries,
        **decision,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
