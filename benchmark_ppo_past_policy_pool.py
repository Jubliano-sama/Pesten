#!/usr/bin/env python3
"""Benchmark a per-iteration uniform historical-policy pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from benchmark_ppo_agent_set_coalescing import _seed_everything
from long_training_supervisor import DEFAULT_MLP_AGENT_SETS
from ppo2 import PPO_Supervised, parse_agent_sets
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def select_past_policy_pool(
    medians: dict[str, float],
    paired_median: float,
    *,
    material_speedup: float = 1.05,
) -> dict:
    speedup = medians["full"]["seconds"] / medians["pool8"]["seconds"]
    accepted = (
        speedup >= material_speedup
        and paired_median >= material_speedup
    )
    return {
        "median_speedup": speedup,
        "paired_median_speedup": paired_median,
        "material_speedup_threshold": material_speedup,
        "selected_past_policy_pool_size": 8 if accepted else None,
        "decision": (
            "use a uniform pool of eight historical policies"
            if accepted
            else "use every historical policy each iteration"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--historical-policies", type=int, default=32)
    parser.add_argument("--seed", type=int, default=85_130_101)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_past_policy_pool_20260801.json"),
    )
    args = parser.parse_args()
    if min(
        args.repeats,
        args.games,
        args.historical_policies,
        args.threads,
    ) < 1:
        raise SystemExit("all count arguments must be positive")
    if args.seed < 0 or args.seed + args.repeats > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    agent_sets = parse_agent_sets(DEFAULT_MLP_AGENT_SETS)
    _seed_everything(args.seed)
    trainer = PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device="cpu",
        rollout_batch_size=256,
        training_actor_device="cpu",
        compile_updates=False,
        past_policy_probability=0.35,
        past_policy_pool_size=None,
        torch_threads=args.threads,
    )
    # Identical actor weights isolate scheduling overhead while retaining the
    # real current-versus-greedy-historical action contract.
    trainer._available_actor_checkpoints = lambda: list(
        range(args.historical_policies)
    )
    trainer._checkpoint_actor = lambda checkpoint: trainer.rollout_actor
    for pool_size in (None, 8):
        trainer.past_policy_pool_size = pool_size
        _seed_everything(args.seed ^ 999)
        trainer.generate_data(32, 0.8, agent_sets)

    raw = {"full": [], "pool8": []}
    for repeat in range(args.repeats):
        order = (
            ("full", "pool8")
            if repeat % 2 == 0
            else ("pool8", "full")
        )
        for name in order:
            trainer.past_policy_pool_size = None if name == "full" else 8
            _seed_everything(args.seed + repeat)
            started = time.perf_counter()
            trainer.generate_data(args.games, 0.8, agent_sets)
            elapsed = time.perf_counter() - started
            metrics = trainer.last_rollout_metrics
            raw[name].append(
                {
                    "seconds": elapsed,
                    "environment_turns": metrics["environment_turns"],
                    "policy_decisions": metrics["policy_decisions"],
                    "inference_calls": metrics["inference_calls"],
                    "past_policy_games": metrics["past_policy_games"],
                    "actor_keys": len(metrics["actor_key_counts"]),
                }
            )
    median_metrics = {
        name: {
            field: statistics.median(run[field] for run in runs)
            for field in (
                "seconds",
                "environment_turns",
                "policy_decisions",
                "inference_calls",
                "past_policy_games",
                "actor_keys",
            )
        }
        for name, runs in raw.items()
    }
    paired = [
        full["seconds"] / pool["seconds"]
        for full, pool in zip(raw["full"], raw["pool8"])
    ]
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_past_policy_pool",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "semantic_contract": (
            "uniform subset sampling preserves each checkpoint's marginal "
            "selection probability at 1/N; within-iteration correlations and "
            "fixed-seed trajectories intentionally change"
        ),
        "workload": {
            "seed_start": args.seed,
            "games": args.games,
            "repeats": args.repeats,
            "threads": args.threads,
            "historical_policies": args.historical_policies,
            "past_policy_probability": 0.35,
            "agent_sets": agent_sets,
        },
        "median_metrics": median_metrics,
        "paired_speedups": paired,
        "raw": raw,
        **select_past_policy_pool(
            median_metrics,
            statistics.median(paired),
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
