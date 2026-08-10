#!/usr/bin/env python3
"""Benchmark cross-engine actor coordination for mixed PPO rollouts."""

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


def select_coordinated_engines(
    medians: dict[str, float],
    paired_median: float,
    *,
    material_speedup: float = 1.05,
) -> dict:
    speedup = (
        medians["sequential_engines"]["seconds"]
        / medians["coordinated_engines"]["seconds"]
    )
    accepted = speedup >= material_speedup and paired_median >= material_speedup
    return {
        "median_speedup": speedup,
        "paired_median_speedup": paired_median,
        "material_speedup_threshold": material_speedup,
        "selected_coordinate_rollout_engines": accepted,
        "decision": (
            "coordinate independent rollout engines"
            if accepted
            else "run rollout engines sequentially"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--seed", type=int, default=85_141_101)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_coordinated_engines_20260801.json"),
    )
    args = parser.parse_args()
    if min(args.repeats, args.games, args.threads) < 1:
        raise SystemExit("count arguments must be positive")
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
        past_policy_probability=0.0,
        torch_threads=args.threads,
    )
    silent = io.StringIO()
    for coordinated in (False, True):
        trainer.coordinate_rollout_engines = coordinated
        _seed_everything(args.seed ^ 999)
        with redirect_stdout(silent):
            trainer.generate_data(32, 0.8, agent_sets)

    raw = {"sequential_engines": [], "coordinated_engines": []}
    for repeat in range(args.repeats):
        order = (
            ("sequential_engines", "coordinated_engines")
            if repeat % 2 == 0
            else ("coordinated_engines", "sequential_engines")
        )
        for name in order:
            trainer.coordinate_rollout_engines = (
                name == "coordinated_engines"
            )
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
            )
        }
        for name, runs in raw.items()
    }
    paired = [
        sequential["seconds"] / coordinated["seconds"]
        for sequential, coordinated in zip(
            raw["sequential_engines"],
            raw["coordinated_engines"],
        )
    ]
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_coordinated_engines",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "semantic_contract": (
            "each game and agent-set weight is unchanged; actor requests from "
            "independent player-count engines share calls, intentionally "
            "changing the fixed-seed action-sampling stream"
        ),
        "workload": {
            "seed_start": args.seed,
            "games": args.games,
            "repeats": args.repeats,
            "threads": args.threads,
            "temperature": 0.8,
            "agent_sets": agent_sets,
        },
        "median_metrics": median_metrics,
        "paired_speedups": paired,
        "raw": raw,
        **select_coordinated_engines(
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
