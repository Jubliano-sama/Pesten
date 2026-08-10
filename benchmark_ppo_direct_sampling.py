#!/usr/bin/env python3
"""Repeat-controlled exact benchmark for direct rollout action sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark_ppo_rollout_trace import (
    exact_hash_contract,
    run_worker,
    summarize,
)
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def select_direct_sampling(
    summaries: dict[str, dict],
    parity: dict,
    *,
    material_speedup: float = 1.01,
) -> dict:
    categorical_seconds = summaries["categorical"]["median"]["total_seconds"]
    direct_seconds = summaries["direct"]["median"]["total_seconds"]
    speedup = categorical_seconds / direct_seconds
    accepted = all(parity.values()) and speedup >= material_speedup
    return {
        "categorical_to_direct_speedup": speedup,
        "material_speedup_threshold": material_speedup,
        "selected_direct_rollout_sampling": accepted,
        "decision": (
            "accept direct rollout sampling"
            if accepted
            else "retain Categorical rollout sampling"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=85_010_001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_direct_sampling_20260801.json"),
    )
    args = parser.parse_args()
    if min(args.repeats, args.games, args.batch_size, args.threads) < 1:
        raise SystemExit("repeats, games, batch size, and threads must be positive")
    if args.seed < 0 or args.seed + args.games > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    args.native_smart_policy = True
    raw = {"categorical": [], "direct": []}
    for repeat in range(args.repeats):
        order = (
            ("categorical", "direct")
            if repeat % 2 == 0
            else ("direct", "categorical")
        )
        for name in order:
            args.direct_rollout_sampling = name == "direct"
            raw[name].append(run_worker(args, traced=True))
    summaries = {name: summarize(runs) for name, runs in raw.items()}
    parity = exact_hash_contract(raw)
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_direct_rollout_sampling",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "workload": {
            "fixed_seed": args.seed,
            "games": args.games,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "threads": args.threads,
            "rollout_device": "cpu",
            "traced_cpu_rollout_actor": True,
            "native_smart_policy": True,
        },
        "exact_parity": parity,
        "results": summaries,
        **select_direct_sampling(summaries, parity),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
