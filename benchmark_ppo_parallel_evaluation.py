#!/usr/bin/env python3
"""Benchmark exact canonical-game evaluation across spawned CPU workers."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import statistics
import time

import torch

from ppo2 import (
    StudentNet,
    _init_evaluation_worker,
    _run_evaluation_game,
)
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def select_parallel_evaluation(
    *,
    exact: bool,
    paired_speedup: float,
    material_speedup: float = 1.25,
) -> dict:
    accepted = exact and paired_speedup >= material_speedup
    return {
        "material_speedup_threshold": material_speedup,
        "selected_parallel_evaluation": accepted,
        "decision": (
            "spawn canonical evaluation games across CPU workers"
            if accepted
            else "retain sequential canonical evaluation"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=85_220_001)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/ppo_parallel_evaluation_20260801.json"
        ),
    )
    args = parser.parse_args()
    if min(args.pairs, args.games, args.workers) < 1:
        raise SystemExit("count arguments must be positive")
    if args.seed < 0 or args.seed + args.games > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    torch.manual_seed(args.seed)
    actor = StudentNet().eval()
    actor_state = {
        name: tensor.detach().cpu()
        for name, tensor in actor.state_dict().items()
    }
    tasks = [
        (0, args.seed + game_index, ("smart", "rl"))
        for game_index in range(args.games)
    ]
    _init_evaluation_worker(actor_state)
    context = multiprocessing.get_context("spawn")
    chunksize = max(1, len(tasks) // (args.workers * 4))

    runs = {"sequential": [], "parallel": []}
    exact = True
    for pair_index in range(args.pairs):
        order = (
            ("sequential", "parallel")
            if pair_index % 2 == 0
            else ("parallel", "sequential")
        )
        pair_results = {}
        for implementation in order:
            started = time.perf_counter()
            if implementation == "parallel":
                with context.Pool(
                    args.workers,
                    initializer=_init_evaluation_worker,
                    initargs=(actor_state,),
                ) as pool:
                    results = pool.map(
                        _run_evaluation_game,
                        tasks,
                        chunksize,
                    )
            else:
                results = [
                    _run_evaluation_game(task)
                    for task in tasks
                ]
            elapsed = time.perf_counter() - started
            runs[implementation].append(
                {
                    "seconds": elapsed,
                    "turns": sum(result[1] for result in results),
                    "wins": sum(result[2] for result in results),
                }
            )
            pair_results[implementation] = results
        exact = exact and (
            pair_results["sequential"] == pair_results["parallel"]
        )

    paired_speedups = [
        sequential["seconds"] / parallel["seconds"]
        for sequential, parallel in zip(
            runs["sequential"],
            runs["parallel"],
        )
    ]
    median_seconds = {
        implementation: statistics.median(
            run["seconds"] for run in selected_runs
        )
        for implementation, selected_runs in runs.items()
    }
    median_speedup = (
        median_seconds["sequential"]
        / median_seconds["parallel"]
    )
    paired_median = statistics.median(paired_speedups)
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_parallel_evaluation",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "semantic_contract": (
            "each independently seeded match uses the canonical Game and "
            "agent implementations; per-game turns and winner are exact"
        ),
        "workload": {
            "pairs": args.pairs,
            "games": args.games,
            "workers": args.workers,
            "seed_start": args.seed,
            "agent_sets": [["smart", "rl"]],
            "process_start_method": "spawn",
        },
        "exact_results": exact,
        "median_seconds": median_seconds,
        "median_speedup": median_speedup,
        "paired_speedups": paired_speedups,
        "paired_median_speedup": paired_median,
        "includes_process_startup": True,
        "runs": runs,
        **select_parallel_evaluation(
            exact=exact,
            paired_speedup=paired_median,
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
