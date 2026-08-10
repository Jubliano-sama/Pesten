#!/usr/bin/env python3
"""Persistent-process rollout-only benchmark for reusable workspaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import numpy as np
import torch

from ppo2 import PPO_Supervised
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rollout_hash(batch) -> str:
    digest = hashlib.sha256()
    for tensor in batch[:4]:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    digest.update(np.asarray(batch[4], dtype=np.int64).tobytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=85_121_001)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/ppo_rollout_workspaces_only_20260801.json"
        ),
    )
    args = parser.parse_args()
    if min(args.repeats, args.games, args.batch_size, args.threads) < 1:
        raise SystemExit("all count arguments must be positive")
    if args.seed < 0 or args.seed + args.games > DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    _seed_everything(args.seed)
    trainer = PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device="cpu",
        rollout_batch_size=args.batch_size,
        training_actor_device="cpu",
        compile_updates=False,
        trace_cpu_rollout_actor=True,
        native_smart_policy=True,
        native_rollout_engine=True,
        direct_rollout_sampling=True,
        numpy_rollout_masks=True,
        reuse_rollout_workspaces=False,
        past_policy_probability=0.0,
        torch_threads=args.threads,
    )
    _seed_everything(args.seed ^ 0xA5A5A5)
    trainer.generate_data(
        num_episodes=16,
        temperature=0.8,
        agent_sets=[["smart", "rl"]],
    )

    raw = {"baseline": [], "workspaces": []}
    for repeat in range(args.repeats):
        order = (
            ("baseline", "workspaces")
            if repeat % 2 == 0
            else ("workspaces", "baseline")
        )
        for name in order:
            trainer.reuse_rollout_workspaces = name == "workspaces"
            _seed_everything(args.seed)
            started = time.perf_counter()
            batch = trainer.generate_data(
                num_episodes=args.games,
                temperature=0.8,
                agent_sets=[["smart", "rl"]],
            )
            elapsed = time.perf_counter() - started
            raw[name].append(
                {
                    "seconds": elapsed,
                    "rollout_sha256": _rollout_hash(batch),
                    "decisions": len(batch[1]),
                    "environment_turns": trainer.last_rollout_metrics[
                        "environment_turns"
                    ],
                }
            )
    medians = {
        name: statistics.median(run["seconds"] for run in runs)
        for name, runs in raw.items()
    }
    paired_speedups = [
        baseline["seconds"] / workspace["seconds"]
        for baseline, workspace in zip(raw["baseline"], raw["workspaces"])
    ]
    exact = (
        len(
            {
                run["rollout_sha256"]
                for runs in raw.values()
                for run in runs
            }
        )
        == 1
    )
    speedup = medians["baseline"] / medians["workspaces"]
    paired_median = statistics.median(paired_speedups)
    accepted = exact and min(speedup, paired_median) >= 1.01
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_rollout_workspaces_only",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "workload": {
            "fixed_seed": args.seed,
            "games": args.games,
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "threads": args.threads,
        },
        "exact_rollout_hash": exact,
        "raw": raw,
        "median_seconds": medians,
        "median_speedup": speedup,
        "paired_speedups": paired_speedups,
        "paired_median_speedup": paired_median,
        "material_speedup_threshold": 1.01,
        "selected_reuse_rollout_workspaces": accepted,
        "decision": (
            "accept reusable rollout workspaces"
            if accepted
            else "retain per-call rollout arrays"
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
