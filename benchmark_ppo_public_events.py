#!/usr/bin/env python3
"""Exact rollout benchmark for disabling unused public-event recording."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from benchmark_ppo_rollout_workspaces_only import (
    _rollout_hash,
    _seed_everything,
)
from ppo2 import PPO_Supervised
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def select_public_event_recording(
    medians: dict[str, float],
    paired_median: float,
    exact: bool,
    *,
    material_speedup: float = 1.01,
) -> dict:
    speedup = medians["recorded"] / medians["disabled"]
    accepted = (
        exact
        and speedup >= material_speedup
        and paired_median >= material_speedup
    )
    return {
        "median_speedup": speedup,
        "paired_median_speedup": paired_median,
        "material_speedup_threshold": material_speedup,
        "selected_disable_rollout_public_events": accepted,
        "decision": (
            "disable rollout public-event recording"
            if accepted
            else "retain rollout public-event recording"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=85_125_001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_public_events_20260801.json"),
    )
    args = parser.parse_args()
    if min(args.repeats, args.games, args.batch_size, args.threads) < 1:
        raise SystemExit("all count arguments must be positive")
    if args.seed < 0 or args.seed + args.games > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    _seed_everything(args.seed)
    trainer = PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device="cpu",
        rollout_batch_size=args.batch_size,
        training_actor_device="cpu",
        compile_updates=False,
        trace_cpu_rollout_actor=True,
        native_smart_policy=True,
        native_rollout_engine=True,
        record_rollout_public_events=True,
        direct_rollout_sampling=True,
        numpy_rollout_masks=True,
        reuse_rollout_workspaces=True,
        past_policy_probability=0.0,
        torch_threads=args.threads,
    )
    _seed_everything(args.seed ^ 0xA5A5A5)
    trainer.generate_data(16, 0.8, [["smart", "rl"]])

    raw = {"recorded": [], "disabled": []}
    for repeat in range(args.repeats):
        order = (
            ("recorded", "disabled")
            if repeat % 2 == 0
            else ("disabled", "recorded")
        )
        for name in order:
            trainer.record_rollout_public_events = name == "recorded"
            _seed_everything(args.seed)
            started = time.perf_counter()
            batch = trainer.generate_data(
                args.games,
                0.8,
                [["smart", "rl"]],
            )
            raw[name].append(
                {
                    "seconds": time.perf_counter() - started,
                    "rollout_sha256": _rollout_hash(batch),
                    "decisions": len(batch[1]),
                    "environment_turns": trainer.last_rollout_metrics[
                        "environment_turns"
                    ],
                }
            )
    medians = {
        name: statistics.median(run["seconds"] for run in runs)
        for name, runs in raw.items()
    }
    paired = [
        recorded["seconds"] / disabled["seconds"]
        for recorded, disabled in zip(raw["recorded"], raw["disabled"])
    ]
    paired_median = statistics.median(paired)
    exact = (
        len(
            {
                run["rollout_sha256"]
                for runs in raw.values()
                for run in runs
            }
        )
        == 1
    )
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_public_events",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "workload": {
            "fixed_seed": args.seed,
            "games": args.games,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "threads": args.threads,
        },
        "exact_rollout_hash": exact,
        "median_seconds": medians,
        "paired_speedups": paired,
        "raw": raw,
        **select_public_event_recording(
            medians,
            paired_median,
            exact,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
