#!/usr/bin/env python3
"""Benchmark draining two-player games to their next neural-policy boundary."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import statistics
import time

from benchmark_ppo_agent_set_coalescing import _seed_everything
from ppo2 import PPO_Supervised
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def select_non_neural_drain(
    medians: dict[str, float],
    paired_median: float,
    *,
    material_speedup: float = 1.02,
) -> dict:
    speedup = (
        medians["one_step"]["seconds"]
        / medians["drain_to_policy"]["seconds"]
    )
    accepted = speedup >= material_speedup and paired_median >= material_speedup
    return {
        "median_speedup": speedup,
        "paired_median_speedup": paired_median,
        "material_speedup_threshold": material_speedup,
        "selected_drain_non_neural_rollout": accepted,
        "decision": (
            "drain a lone two-player engine to each policy boundary"
            if accepted
            else "advance one non-neural turn per rollout step"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--seed", type=int, default=85_145_101)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_drain_non_neural_20260801.json"),
    )
    args = parser.parse_args()
    if min(args.repeats, args.games, args.threads) < 1:
        raise SystemExit("count arguments must be positive")
    if args.seed < 0 or args.seed + args.repeats > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

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
    for drain in (False, True):
        trainer.drain_non_neural_rollout = drain
        _seed_everything(args.seed ^ 999)
        with redirect_stdout(silent):
            trainer.generate_data(32, 0.8, [["smart", "rl"]])

    raw = {"one_step": [], "drain_to_policy": []}
    for repeat in range(args.repeats):
        order = (
            ("one_step", "drain_to_policy")
            if repeat % 2 == 0
            else ("drain_to_policy", "one_step")
        )
        for name in order:
            trainer.drain_non_neural_rollout = name == "drain_to_policy"
            _seed_everything(args.seed + repeat)
            started = time.perf_counter()
            with redirect_stdout(silent):
                trainer.generate_data(
                    args.games,
                    0.8,
                    [["smart", "rl"]],
                )
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
        one_step["seconds"] / drain["seconds"]
        for one_step, drain in zip(
            raw["one_step"],
            raw["drain_to_policy"],
        )
    ]
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_drain_non_neural",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "semantic_contract": (
            "independent game rules and weights are unchanged; Smart turns "
            "advance to the next neural-policy boundary before actor batching, "
            "intentionally changing the fixed-seed sampling stream"
        ),
        "workload": {
            "seed_start": args.seed,
            "games": args.games,
            "repeats": args.repeats,
            "threads": args.threads,
            "temperature": 0.8,
            "agent_sets": [["smart", "rl"]],
        },
        "median_metrics": median_metrics,
        "paired_speedups": paired,
        "raw": raw,
        **select_non_neural_drain(
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
