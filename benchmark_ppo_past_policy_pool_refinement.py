#!/usr/bin/env python3
"""Refine the historical-policy pool cap for coordinated PPO rollouts."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import statistics
import time

from benchmark_ppo_agent_set_coalescing import _seed_everything
from long_training_supervisor import DEFAULT_MLP_AGENT_SETS
from ppo2 import PPO_Supervised, parse_agent_sets
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def select_refined_pool(
    medians: dict[str, float],
    paired_median: float,
    *,
    material_speedup: float = 1.10,
) -> dict:
    speedup = medians["pool8"]["seconds"] / medians["pool2"]["seconds"]
    accepted = speedup >= material_speedup and paired_median >= material_speedup
    return {
        "median_speedup": speedup,
        "paired_median_speedup": paired_median,
        "material_speedup_threshold": material_speedup,
        "selected_past_policy_pool_size": 2 if accepted else 8,
        "decision": (
            "use a uniform pool of two historical policies"
            if accepted
            else "retain the uniform pool of eight historical policies"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--historical-policies", type=int, default=32)
    parser.add_argument("--seed", type=int, default=85_148_101)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/ppo_past_policy_pool_refinement_20260801.json"
        ),
    )
    args = parser.parse_args()
    if min(
        args.repeats,
        args.games,
        args.historical_policies,
        args.threads,
    ) < 1:
        raise SystemExit("count arguments must be positive")
    if args.historical_policies < 8:
        raise SystemExit("at least eight historical policies are required")
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
        rollout_batch_size=args.games,
        training_actor_device="cpu",
        compile_updates=False,
        past_policy_probability=0.35,
        past_policy_pool_size=8,
        torch_threads=args.threads,
    )
    trainer._available_actor_checkpoints = lambda: list(
        range(args.historical_policies)
    )
    trainer._checkpoint_actor = lambda checkpoint: trainer.rollout_actor
    silent = io.StringIO()
    for pool_size in (8, 2):
        trainer.past_policy_pool_size = pool_size
        _seed_everything(args.seed ^ 999)
        with redirect_stdout(silent):
            trainer.generate_data(32, 0.8, agent_sets)

    raw = {"pool8": [], "pool2": []}
    for repeat in range(args.repeats):
        order = (
            ("pool8", "pool2")
            if repeat % 2 == 0
            else ("pool2", "pool8")
        )
        for name in order:
            trainer.past_policy_pool_size = 8 if name == "pool8" else 2
            _seed_everything(args.seed + repeat)
            started = time.perf_counter()
            with redirect_stdout(silent):
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
        pool8["seconds"] / pool2["seconds"]
        for pool8, pool2 in zip(raw["pool8"], raw["pool2"])
    ]
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_past_policy_pool_refinement",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "semantic_contract": (
            "uniform subset sampling preserves every checkpoint's marginal "
            "selection probability at 1/N; pool two increases within-iteration "
            "correlation relative to pool eight while retaining diversity"
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
        **select_refined_pool(
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
