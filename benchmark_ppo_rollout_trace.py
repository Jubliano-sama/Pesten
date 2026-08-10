#!/usr/bin/env python3
"""Repeat-controlled exact PPO CPU rollout-actor trace benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys

from seed_integrity import DEVELOPMENT_SEED_LIMIT


def worker_command(args, *, traced: bool) -> list[str]:
    return [
        sys.executable,
        "benchmark_training_speed.py",
        "--worker",
        "--implementation",
        "vectorized",
        "--thermal-state",
        "cold",
        "--seed",
        str(args.seed),
        "--episodes",
        str(args.games),
        "--warmup-episodes",
        "16",
        "--batch-size",
        str(args.batch_size),
        "--threads",
        str(args.threads),
        "--rollout-device",
        "cpu",
        (
            "--trace-cpu-rollout-actor"
            if traced
            else "--no-trace-cpu-rollout-actor"
        ),
        (
            "--cuda-rollout-autocast"
            if getattr(args, "cuda_rollout_autocast", False)
            else "--no-cuda-rollout-autocast"
        ),
        (
            "--reuse-cpu-critic-forward"
            if getattr(args, "reuse_cpu_critic_forward", True)
            else "--no-reuse-cpu-critic-forward"
        ),
        (
            "--native-smart-policy"
            if args.native_smart_policy
            else "--no-native-smart-policy"
        ),
        (
            "--direct-rollout-sampling"
            if getattr(args, "direct_rollout_sampling", True)
            else "--no-direct-rollout-sampling"
        ),
        (
            "--native-rollout-engine"
            if getattr(args, "native_rollout_engine", True)
            else "--no-native-rollout-engine"
        ),
        (
            "--record-rollout-public-events"
            if getattr(args, "record_rollout_public_events", False)
            else "--no-record-rollout-public-events"
        ),
        (
            "--numpy-rollout-masks"
            if getattr(args, "numpy_rollout_masks", True)
            else "--no-numpy-rollout-masks"
        ),
        (
            "--reuse-rollout-workspaces"
            if getattr(args, "reuse_rollout_workspaces", True)
            else "--no-reuse-rollout-workspaces"
        ),
        (
            "--coalesce-rollout-agent-sets"
            if getattr(args, "coalesce_rollout_agent_sets", True)
            else "--no-coalesce-rollout-agent-sets"
        ),
        (
            "--group-compatible-rollout-agent-sets"
            if getattr(args, "group_compatible_rollout_agent_sets", True)
            else "--no-group-compatible-rollout-agent-sets"
        ),
        (
            "--coordinate-rollout-engines"
            if getattr(args, "coordinate_rollout_engines", True)
            else "--no-coordinate-rollout-engines"
        ),
        (
            "--drain-non-neural-rollout"
            if getattr(args, "drain_non_neural_rollout", True)
            else "--no-drain-non-neural-rollout"
        ),
        "--past-policy-pool-size",
        str(getattr(args, "past_policy_pool_size", 2)),
        "--no-compile-updates",
    ]


def run_worker(args, *, traced: bool) -> dict:
    completed = subprocess.run(
        worker_command(args, traced=traced),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def summarize(runs: list[dict]) -> dict:
    fields = (
        "rollout_seconds",
        "prep_seconds",
        "update_seconds",
        "total_seconds",
        "rss_peak_mib",
        "rollout_buffer_peak_mib",
    )
    return {
        "repeats": len(runs),
        "median": {
            field: statistics.median(run[field] for run in runs)
            for field in fields
        },
        "runs": runs,
    }


def exact_hash_contract(runs: dict[str, list[dict]]) -> dict:
    return {
        name: len(
            {
                run[field]
                for variant_runs in runs.values()
                for run in variant_runs
            }
        )
        == 1
        for name, field in (
            ("rollout_hash_exact", "rollout_sha256"),
            ("actor_state_hash_exact", "actor_state_sha256"),
            ("critic_state_hash_exact", "critic_state_sha256"),
        )
    }


def select_trace(
    summaries: dict[str, dict],
    parity: dict,
    *,
    material_speedup: float = 1.02,
) -> dict:
    eager_seconds = summaries["eager"]["median"]["total_seconds"]
    traced_seconds = summaries["traced"]["median"]["total_seconds"]
    speedup = eager_seconds / traced_seconds
    exact = all(parity.values())
    accepted = exact and speedup >= material_speedup
    return {
        "eager_to_traced_speedup": speedup,
        "material_speedup_threshold": material_speedup,
        "selected_trace_cpu_rollout_actor": accepted,
        "decision": "accept traced CPU rollout actor" if accepted else "retain eager actor",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--games", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=84_950_001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--native-smart-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/ppo_rollout_trace_20260801.json"),
    )
    args = parser.parse_args()
    if min(args.repeats, args.games, args.batch_size, args.threads) < 1:
        raise SystemExit("repeats, games, batch size, and threads must be positive")
    if args.seed < 0 or args.seed + args.games > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    raw = {"eager": [], "traced": []}
    for repeat in range(args.repeats):
        order = ("eager", "traced") if repeat % 2 == 0 else ("traced", "eager")
        for name in order:
            raw[name].append(run_worker(args, traced=name == "traced"))
    summaries = {name: summarize(runs) for name, runs in raw.items()}
    parity = exact_hash_contract(raw)
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_cpu_rollout_actor_trace",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "workload": {
            "fixed_seed": args.seed,
            "games": args.games,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "threads": args.threads,
            "rollout_device": "cpu",
            "native_smart_policy": args.native_smart_policy,
        },
        "exact_parity": parity,
        "results": summaries,
        **select_trace(summaries, parity),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
