#!/usr/bin/env python3
"""Exact paired benchmark for same-shape rollout-engine reuse."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import statistics
import time

from benchmark_ppo_agent_set_coalescing import _seed_everything
from long_training_supervisor import DEFAULT_MLP_AGENT_SETS
from ppo2 import PPO_Supervised, parse_agent_sets
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def batch_hash(batch) -> str:
    digest = hashlib.sha256()
    for tensor in batch[:4]:
        digest.update(
            tensor.detach().cpu().contiguous().numpy().tobytes()
        )
    for length in batch[4]:
        digest.update(int(length).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=85_210_101)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/ppo_rollout_engine_reuse_20260801.json"
        ),
    )
    args = parser.parse_args()
    if min(
        args.pairs,
        args.games,
        args.batch_size,
        args.threads,
    ) < 1:
        raise SystemExit("all counts must be positive")
    if (
        args.seed < 0
        or args.seed + args.pairs > DEVELOPMENT_SEED_LIMIT
    ):
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    agent_sets = parse_agent_sets(DEFAULT_MLP_AGENT_SETS)
    trainer = PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device="cpu",
        rollout_batch_size=args.batch_size,
        training_actor_device="cpu",
        compile_updates=False,
        past_policy_probability=0.35,
        past_policy_pool_size=2,
        reuse_rollout_engines=False,
        torch_threads=args.threads,
    )
    silent = io.StringIO()
    for reuse in (False, True):
        trainer.reuse_rollout_engines = reuse
        _seed_everything(args.seed ^ 999)
        with redirect_stdout(silent):
            trainer.generate_data(args.games, 0.8, agent_sets)

    runs = {"allocate": [], "reuse": []}
    exact = True
    for repeat in range(args.pairs):
        seed = args.seed + repeat
        order = (
            ("allocate", "reuse")
            if repeat % 2 == 0
            else ("reuse", "allocate")
        )
        pair = {}
        for name in order:
            trainer.reuse_rollout_engines = name == "reuse"
            _seed_everything(seed)
            started = time.perf_counter()
            with redirect_stdout(silent):
                batch = trainer.generate_data(
                    args.games,
                    0.8,
                    agent_sets,
                )
            metrics = trainer.last_rollout_metrics
            record = {
                "seed": seed,
                "seconds": time.perf_counter() - started,
                "rollout_sha256": batch_hash(batch),
                "environment_turns": metrics["environment_turns"],
                "policy_decisions": metrics["policy_decisions"],
                "inference_calls": metrics["inference_calls"],
            }
            runs[name].append(record)
            pair[name] = record
        exact &= all(
            pair["allocate"][field] == pair["reuse"][field]
            for field in (
                "rollout_sha256",
                "environment_turns",
                "policy_decisions",
                "inference_calls",
            )
        )

    speedups = [
        allocate["seconds"] / reuse["seconds"]
        for allocate, reuse in zip(runs["allocate"], runs["reuse"])
    ]
    paired_median = statistics.median(speedups)
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_rollout_engine_reuse",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "decision": (
            "accept exact same-shape rollout engine reuse"
            if exact and paired_median >= 1.02
            else "retain fresh rollout engine allocation"
        ),
        "exact_rollout_hashes": exact,
        "workload": {
            "pairs": args.pairs,
            "games": args.games,
            "batch_size": args.batch_size,
            "threads": args.threads,
            "rollout_device": "cpu",
            "agent_sets": agent_sets,
        },
        "paired_speedups": speedups,
        "paired_speedup_median": paired_median,
        "grouped_speedup": (
            statistics.median(
                run["seconds"] for run in runs["allocate"]
            )
            / statistics.median(
                run["seconds"] for run in runs["reuse"]
            )
        ),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
