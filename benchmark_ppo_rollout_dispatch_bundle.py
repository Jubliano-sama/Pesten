#!/usr/bin/env python3
"""Exact benchmark for combined NumPy masks and direct action sampling."""

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


def select_bundle(
    summaries: dict[str, dict],
    parity: dict,
    *,
    material_speedup: float = 1.01,
) -> dict:
    baseline_seconds = summaries["baseline"]["median"]["total_seconds"]
    bundled_seconds = summaries["bundle"]["median"]["total_seconds"]
    speedup = baseline_seconds / bundled_seconds
    accepted = all(parity.values()) and speedup >= material_speedup
    return {
        "baseline_to_dispatch_bundle_speedup": speedup,
        "material_speedup_threshold": material_speedup,
        "selected_rollout_dispatch_bundle": accepted,
        "decision": (
            "accept rollout dispatch bundle"
            if accepted
            else "retain baseline rollout dispatch"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=85_110_001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/ppo_rollout_dispatch_bundle_20260801.json"
        ),
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
    raw = {"baseline": [], "bundle": []}
    for repeat in range(args.repeats):
        order = (
            ("baseline", "bundle")
            if repeat % 2 == 0
            else ("bundle", "baseline")
        )
        for name in order:
            enabled = name == "bundle"
            args.direct_rollout_sampling = enabled
            args.numpy_rollout_masks = enabled
            raw[name].append(run_worker(args, traced=True))
    summaries = {name: summarize(runs) for name, runs in raw.items()}
    parity = exact_hash_contract(raw)
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_rollout_dispatch_bundle",
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
            "native_rollout_engine": True,
        },
        "exact_parity": parity,
        "results": summaries,
        **select_bundle(summaries, parity),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
