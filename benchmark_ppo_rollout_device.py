#!/usr/bin/env python3
"""Compare traced CPU and FP32 CUDA actor rollouts end to end."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

import torch

from seed_integrity import DEVELOPMENT_SEED_LIMIT


def select_rollout_device(
    medians: dict[str, dict[str, float]],
    paired_median: float,
    max_initial_ratio_error: float,
    *,
    material_speedup: float = 1.25,
    ratio_error_limit: float = 0.02,
) -> dict:
    total_speedup = (
        medians["cpu"]["total_seconds"]
        / medians["cuda_fp32"]["total_seconds"]
    )
    rollout_speedup = (
        medians["cpu"]["rollout_seconds"]
        / medians["cuda_fp32"]["rollout_seconds"]
    )
    accepted = (
        total_speedup >= material_speedup
        and rollout_speedup >= material_speedup
        and paired_median >= material_speedup
        and max_initial_ratio_error < ratio_error_limit
    )
    return {
        "median_total_speedup": total_speedup,
        "median_rollout_speedup": rollout_speedup,
        "paired_median_total_speedup": paired_median,
        "max_cuda_initial_ratio_error": max_initial_ratio_error,
        "material_speedup_threshold": material_speedup,
        "ratio_error_limit": ratio_error_limit,
        "selected_rollout_device": "cuda" if accepted else "cpu",
        "selected_cuda_rollout_autocast": False,
        "decision": (
            "use FP32 CUDA rollout when CUDA is available"
            if accepted
            else "retain traced CPU rollout"
        ),
    }


def _worker_command(args, *, device: str, seed: int) -> list[str]:
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
        device,
        "--compile-updates",
        "--no-cuda-rollout-autocast",
        (
            "--trace-cpu-rollout-actor"
            if device == "cpu"
            else "--no-trace-cpu-rollout-actor"
        ),
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
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--warmup-games", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=85_151_101)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_rollout_device_20260801.json"),
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
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    raw = {"cpu": [], "cuda_fp32": []}
    for repeat in range(args.repeats):
        order = (
            (("cpu", "cpu"), ("cuda_fp32", "cuda"))
            if repeat % 2 == 0
            else (("cuda_fp32", "cuda"), ("cpu", "cpu"))
        )
        for name, device in order:
            completed = subprocess.run(
                _worker_command(
                    args,
                    device=device,
                    seed=args.seed + repeat,
                ),
                check=True,
                capture_output=True,
                text=True,
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
        "inference_calls",
        "first_update_max_ratio_error",
        "cuda_peak_allocated_mib",
    )
    medians = {
        name: {
            field: statistics.median(run[field] for run in runs)
            for field in fields
        }
        for name, runs in raw.items()
    }
    paired = [
        cpu["total_seconds"] / cuda["total_seconds"]
        for cpu, cuda in zip(raw["cpu"], raw["cuda_fp32"])
    ]
    max_ratio_error = max(
        run["first_update_max_ratio_error"]
        for run in raw["cuda_fp32"]
    )
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_rollout_device",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "semantic_contract": (
            "both devices evaluate the same FP32 policy and legal-action "
            "distribution; device-specific floating-point kernels and action "
            "sampling intentionally change fixed-seed trajectories"
        ),
        "workload": {
            "seed_start": args.seed,
            "games": args.games,
            "warmup_games": args.warmup_games,
            "repeats": args.repeats,
            "threads": args.threads,
            "agent_sets": [["smart", "rl"]],
            "cuda_rollout_autocast": False,
        },
        "median_metrics": medians,
        "paired_total_speedups": paired,
        "raw": raw,
        **select_rollout_device(
            medians,
            statistics.median(paired),
            max_ratio_error,
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
