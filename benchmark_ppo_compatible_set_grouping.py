#!/usr/bin/env python3
"""Benchmark same-player-count vector grouping on the live rollout recipe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

from benchmark_ppo_agent_set_coalescing import (
    _seed_everything,
    select_coalescing,
)
from long_training_supervisor import DEFAULT_MLP_AGENT_SETS
from ppo2 import PPO_Supervised, parse_agent_sets
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=85_129_001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/ppo_compatible_set_grouping_20260801.json"
        ),
    )
    args = parser.parse_args()
    if min(args.repeats, args.games, args.batch_size, args.threads) < 1:
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
        rollout_batch_size=args.batch_size,
        training_actor_device="cpu",
        compile_updates=False,
        coalesce_rollout_agent_sets=True,
        group_compatible_rollout_agent_sets=False,
        past_policy_probability=0.0,
        torch_threads=args.threads,
    )
    for enabled in (False, True):
        trainer.group_compatible_rollout_agent_sets = enabled
        _seed_everything(args.seed ^ 999)
        trainer.generate_data(32, 0.8, agent_sets)

    raw = {"baseline": [], "coalesced": []}
    for repeat in range(args.repeats):
        order = (
            ("baseline", "coalesced")
            if repeat % 2 == 0
            else ("coalesced", "baseline")
        )
        for name in order:
            trainer.group_compatible_rollout_agent_sets = (
                name == "coalesced"
            )
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
                }
            )
    medians = {
        name: statistics.median(run["seconds"] for run in runs)
        for name, runs in raw.items()
    }
    paired = [
        baseline["seconds"] / grouped["seconds"]
        for baseline, grouped in zip(
            raw["baseline"],
            raw["coalesced"],
        )
    ]
    means = {
        name: {
            field: statistics.mean(run[field] for run in runs)
            for field in (
                "environment_turns",
                "policy_decisions",
                "inference_calls",
            )
        }
        for name, runs in raw.items()
    }
    selection = select_coalescing(
        medians,
        statistics.median(paired),
    )
    selection["selected_group_compatible_rollout_agent_sets"] = (
        selection.pop("selected_coalesce_rollout_agent_sets")
    )
    selection["decision"] = (
        "group compatible consecutive agent sets"
        if selection["selected_group_compatible_rollout_agent_sets"]
        else "retain one vector engine per distinct agent set"
    )
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_compatible_set_grouping",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "semantic_contract": (
            "preserves episode weights and per-game policy configurations; "
            "vector grouping intentionally changes fixed-seed trajectories"
        ),
        "workload": {
            "seed_start": args.seed,
            "games": args.games,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "threads": args.threads,
            "agent_sets": agent_sets,
        },
        "median_seconds": medians,
        "paired_speedups": paired,
        "mean_metrics": means,
        "raw": raw,
        **selection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
