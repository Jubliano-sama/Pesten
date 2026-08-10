#!/usr/bin/env python3
"""Repeat-controlled exact benchmark for physical-core CPU affinity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from benchmark_ppo_rollout_trace import (
    exact_hash_contract,
    summarize,
    worker_command,
)
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def select_affinity(
    summaries: dict[str, dict],
    parity: dict,
    *,
    material_speedup: float = 1.02,
) -> dict:
    unpinned_seconds = summaries["unpinned"]["median"]["total_seconds"]
    pinned_seconds = summaries["pinned"]["median"]["total_seconds"]
    speedup = unpinned_seconds / pinned_seconds
    accepted = all(parity.values()) and speedup >= material_speedup
    return {
        "unpinned_to_physical_cores_speedup": speedup,
        "material_speedup_threshold": material_speedup,
        "selected_physical_core_affinity": accepted,
        "decision": (
            "accept physical-core affinity"
            if accepted
            else "retain unrestricted CPU affinity"
        ),
    }


def run_worker(args, *, pinned: bool) -> dict:
    command = worker_command(args, traced=True)
    if pinned:
        command = ["taskset", "-c", args.cpu_list, *command]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=85_080_001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpu-list", default="0-7")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_cpu_affinity_20260801.json"),
    )
    args = parser.parse_args()
    if min(args.repeats, args.games, args.batch_size, args.threads) < 1:
        raise SystemExit("repeats, games, batch size, and threads must be positive")
    if args.seed < 0 or args.seed + args.games > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    args.native_smart_policy = True
    args.native_rollout_engine = True
    args.direct_rollout_sampling = False
    raw = {"unpinned": [], "pinned": []}
    for repeat in range(args.repeats):
        order = (
            ("unpinned", "pinned")
            if repeat % 2 == 0
            else ("pinned", "unpinned")
        )
        for name in order:
            raw[name].append(run_worker(args, pinned=name == "pinned"))
    summaries = {name: summarize(runs) for name, runs in raw.items()}
    parity = exact_hash_contract(raw)
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_physical_core_affinity",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "workload": {
            "fixed_seed": args.seed,
            "games": args.games,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "threads": args.threads,
            "cpu_list": args.cpu_list,
            "rollout_device": "cpu",
            "traced_cpu_rollout_actor": True,
            "native_smart_policy": True,
            "native_rollout_engine": True,
        },
        "exact_parity": parity,
        "results": summaries,
        **select_affinity(summaries, parity),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
