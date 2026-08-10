#!/usr/bin/env python3
"""Exact paired benchmark for batched native Smart-policy dispatch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    POLICY_RL,
    POLICY_SMART,
    collect_batched_rollout,
)
from ppo2 import PPO_Supervised
from seed_integrity import DEVELOPMENT_SEED_LIMIT


def rollout_hash(result) -> str:
    digest = hashlib.sha256()
    for tensor in (
        result.observations,
        result.actions,
        result.log_probabilities,
        result.rewards,
    ):
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    digest.update(
        np.asarray(result.episode_lengths, dtype=np.int64).tobytes()
    )
    return digest.hexdigest()


def build_engine(seed: int, games: int, *, batched: bool) -> BatchedPesten:
    kinds = np.empty((games, 2), dtype=np.int8)
    kinds[::2] = (POLICY_SMART, POLICY_RL)
    kinds[1::2] = (POLICY_RL, POLICY_SMART)
    actor_keys = np.full((games, 2), None, dtype=object)
    actor_keys[kinds == POLICY_RL] = "current"
    engine = BatchedPesten(
        kinds,
        actor_keys,
        np.arange(seed, seed + games, dtype=np.uint32),
        max_turns=500,
        native_smart_policy=True,
        native_rollout_engine=True,
        record_public_events=False,
    )
    if not batched:
        # The scalar native kernel remains enabled; only its Python dispatch
        # frequency differs between the paired variants.
        engine._smart_actions_native = None
    return engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=7)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=85_130_011)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/ppo_batched_smart_dispatch_20260801.json"
        ),
    )
    args = parser.parse_args()
    if min(args.pairs, args.games, args.threads) < 1:
        raise SystemExit("pairs, games, and threads must be positive")
    if (
        args.seed < 0
        or args.seed + args.pairs + args.games > DEVELOPMENT_SEED_LIMIT
    ):
        raise SystemExit("benchmark seeds must remain in development range")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    torch.set_num_threads(args.threads)
    ppo = PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device="cpu",
        rollout_batch_size=args.games,
        compile_updates=False,
        trace_cpu_rollout_actor=True,
        native_smart_policy=True,
        native_rollout_engine=True,
        torch_threads=args.threads,
    )
    policies = {
        "current": ActorPolicy(
            actor=ppo.rollout_actor,
            device=torch.device("cpu"),
            temperature=0.8,
            collect_data=True,
            autocast=False,
        )
    }
    runs = {"scalar": [], "batched": []}
    exact = True
    for pair_index in range(args.pairs):
        seed = args.seed + pair_index
        pair = {}
        order = (
            (False, True)
            if pair_index % 2 == 0
            else (True, False)
        )
        for batched in order:
            engine = build_engine(seed, args.games, batched=batched)
            torch.manual_seed(seed ^ 0xA5A5A5A5)
            started = time.perf_counter()
            result = collect_batched_rollout(
                engine,
                policies,
                output_device=torch.device("cpu"),
                direct_action_sampling=True,
                numpy_action_masks=True,
                reuse_rollout_workspaces=True,
                drain_non_neural=True,
            )
            name = "batched" if batched else "scalar"
            record = {
                "seed": seed,
                "seconds": time.perf_counter() - started,
                "rollout_sha256": rollout_hash(result),
                "inference_calls": result.inference_calls,
                "policy_decisions": result.policy_decisions,
                "environment_turns": result.environment_turns,
            }
            runs[name].append(record)
            pair[name] = record
        exact &= all(
            pair["scalar"][field] == pair["batched"][field]
            for field in (
                "rollout_sha256",
                "inference_calls",
                "policy_decisions",
                "environment_turns",
            )
        )

    speedups = [
        scalar["seconds"] / batched["seconds"]
        for scalar, batched in zip(runs["scalar"], runs["batched"])
    ]
    paired_median = statistics.median(speedups)
    payload = {
        "schema_version": 1,
        "benchmark": "ppo_batched_smart_dispatch",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "decision": (
            "accept exact batched native Smart dispatch"
            if exact and paired_median >= 1.02
            else "retain scalar native Smart dispatch"
        ),
        "exact_rollout_hashes": exact,
        "workload": {
            "pairs": args.pairs,
            "games": args.games,
            "players": 2,
            "policies": ["smart", "rl"],
            "threads": args.threads,
            "rollout_device": "cpu",
        },
        "paired_speedups": speedups,
        "paired_speedup_median": paired_median,
        "grouped_speedup": (
            statistics.median(
                run["seconds"] for run in runs["scalar"]
            )
            / statistics.median(
                run["seconds"] for run in runs["batched"]
            )
        ),
        "timing_caveat": (
            "Concurrent training can perturb pair-to-pair load; "
            "the alternating paired median is the selection statistic."
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
