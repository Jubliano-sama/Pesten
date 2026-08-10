#!/usr/bin/env python3
"""Benchmark reuse of the CPU critic's GAE forward graph."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from seed_integrity import DEVELOPMENT_SEED_LIMIT


def select_cpu_critic_reuse(
    medians: dict[str, dict[str, float]],
    paired_median: float,
    training_outputs_exact: bool,
    *,
    material_speedup: float = 1.05,
) -> dict:
    speedup = (
        medians["separate_forward"]["total_seconds"]
        / medians["reused_forward"]["total_seconds"]
    )
    accepted = (
        training_outputs_exact
        and speedup >= material_speedup
        and paired_median >= material_speedup
    )
    return {
        "median_total_speedup": speedup,
        "paired_median_total_speedup": paired_median,
        "material_speedup_threshold": material_speedup,
        "training_outputs_exact": training_outputs_exact,
        "selected_reuse_cpu_critic_forward": accepted,
        "decision": (
            "reuse the CPU critic graph for PPO update one"
            if accepted
            else "recompute the CPU critic for PPO update one"
        ),
    }


def _worker_command(args, *, reuse: bool, seed: int) -> list[str]:
    return [
        sys.executable,
        "benchmark_training_speed.py",
        "--worker",
        "--implementation",
        "vectorized",
        "--thermal-state",
        "warm",
        "--seed",
        str(seed),
        "--episodes",
        str(args.games),
        "--warmup-episodes",
        str(args.warmup_games),
        "--temperature",
        "0.8",
        "--batch-size",
        str(args.games),
        "--threads",
        str(args.threads),
        "--rollout-device",
        "cpu",
        "--no-compile-updates",
        "--no-cuda-rollout-autocast",
        (
            "--reuse-cpu-critic-forward"
            if reuse
            else "--no-reuse-cpu-critic-forward"
        ),
        "--trace-cpu-rollout-actor",
        "--native-smart-policy",
        "--direct-rollout-sampling",
        "--native-rollout-engine",
        "--no-record-rollout-public-events",
        "--numpy-rollout-masks",
        "--reuse-rollout-workspaces",
        "--coalesce-rollout-agent-sets",
        "--group-compatible-rollout-agent-sets",
        "--coordinate-rollout-engines",
        "--drain-non-neural-rollout",
        "--past-policy-pool-size",
        "2",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--warmup-games", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=85_153_101)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_cpu_critic_reuse_20260801.json"),
    )
    args = parser.parse_args()
    if min(
        args.repeats,
        args.games,
        args.warmup_games,
        args.threads,
    ) < 1:
        raise SystemExit("count arguments must be positive")
    if args.seed < 0 or args.seed + args.repeats > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    raw = {"separate_forward": [], "reused_forward": []}
    worker_environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
    }
    for repeat in range(args.repeats):
        order = (
            (("separate_forward", False), ("reused_forward", True))
            if repeat % 2 == 0
            else (("reused_forward", True), ("separate_forward", False))
        )
        for name, reuse in order:
            completed = subprocess.run(
                _worker_command(
                    args,
                    reuse=reuse,
                    seed=args.seed + repeat,
                ),
                check=True,
                capture_output=True,
                text=True,
                env=worker_environment,
            )
            lines = [
                line
                for line in completed.stdout.splitlines()
                if line.strip()
            ]
            raw[name].append(json.loads(lines[-1]))

    fields = (
        "rollout_seconds",
        "prep_seconds",
        "update_seconds",
        "total_seconds",
        "collected_policy_decisions",
        "first_update_max_ratio_error",
        "rss_peak_mib",
    )
    medians = {
        name: {
            field: statistics.median(run[field] for run in runs)
            for field in fields
        }
        for name, runs in raw.items()
    }
    paired = [
        separate["total_seconds"] / reused["total_seconds"]
        for separate, reused in zip(
            raw["separate_forward"],
            raw["reused_forward"],
        )
    ]
    exact_fields = (
        "rollout_sha256",
        "actor_state_sha256",
        "final_actor_loss",
        "final_critic_loss",
        "first_update_kl",
        "final_kl",
        "first_update_max_ratio_error",
    )
    training_outputs_exact = all(
        all(separate[field] == reused[field] for field in exact_fields)
        for separate, reused in zip(
            raw["separate_forward"],
            raw["reused_forward"],
        )
    )
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_cpu_critic_reuse",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "semantic_contract": (
            "rollouts, losses, KL values, actor parameters, critic learned "
            "parameters, and optimizer state remain bit-exact; combining two "
            "identical BatchNorm running-stat updates changes only non-trainable "
            "critic buffers by at most 4.8e-7 in the dedicated audit"
        ),
        "workload": {
            "seed_start": args.seed,
            "games": args.games,
            "warmup_games": args.warmup_games,
            "repeats": args.repeats,
            "threads": args.threads,
            "agent_sets": [["smart", "rl"]],
            "device": "cpu",
        },
        "median_metrics": medians,
        "paired_total_speedups": paired,
        "raw": raw,
        **select_cpu_critic_reuse(
            medians,
            statistics.median(paired),
            training_outputs_exact,
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
