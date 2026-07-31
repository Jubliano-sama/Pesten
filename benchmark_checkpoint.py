#!/usr/bin/env python3
"""Fresh matched-deal comparison of two feed-forward PPO checkpoints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from benchmark_hybrid import (
    bootstrap_mean_ci,
    derive_deal_seed,
    file_sha256,
    run_seat_pair,
)
from hybrid_agent import load_frozen_ppo_actor, make_ppo_agent_factory
import randomAgent
import smartAgent


POOL = {
    "random": randomAgent.AgentRandom,
    "smart": smartAgent.Agent,
    "smarter": smartAgent.SmarterAgent,
}


def summarize(values, *, seed):
    return {
        "mean": float(np.mean(values)),
        "bootstrap_ci95": list(
            bootstrap_mean_ci(values, seed=seed)
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pool", default="random,smart,smarter")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    candidate_actor, device = load_frozen_ppo_actor(
        args.candidate, args.device
    )
    parent_actor, _ = load_frozen_ppo_actor(args.parent, args.device)
    candidate = make_ppo_agent_factory(
        candidate_actor, device, shield=False
    )
    parent = make_ppo_agent_factory(parent_actor, device, shield=False)

    direct = []
    for pair_index in range(args.pairs):
        deal_seed = derive_deal_seed(
            args.seed, "candidate_vs_parent", pair_index
        )
        pair = run_seat_pair(
            candidate,
            parent,
            pair_index=pair_index,
            deal_seed=deal_seed,
        )
        direct.append(pair)
    direct_advantages = [2.0 * pair.score - 1.0 for pair in direct]

    pool_results = []
    combined_deltas = []
    for name in tuple(filter(None, args.pool.split(","))):
        opponent = POOL[name]
        records = []
        for pair_index in range(args.pairs):
            deal_seed = derive_deal_seed(
                args.seed, f"pool:{name}", pair_index
            )
            candidate_pair = run_seat_pair(
                candidate,
                opponent,
                pair_index=pair_index,
                deal_seed=deal_seed,
            )
            parent_pair = run_seat_pair(
                parent,
                opponent,
                pair_index=pair_index,
                deal_seed=deal_seed,
            )
            delta = candidate_pair.score - parent_pair.score
            combined_deltas.append(delta)
            records.append(
                {
                    "pair_index": pair_index,
                    "deal_seed": deal_seed,
                    "candidate": asdict(candidate_pair),
                    "parent": asdict(parent_pair),
                    "delta": delta,
                }
            )
        deltas = [record["delta"] for record in records]
        pool_results.append(
            {
                "opponent": name,
                "candidate_score": summarize(
                    [record["candidate"]["score"] for record in records],
                    seed=derive_deal_seed(
                        args.seed, f"candidate:{name}", 0
                    ),
                ),
                "parent_score": summarize(
                    [record["parent"]["score"] for record in records],
                    seed=derive_deal_seed(
                        args.seed, f"parent:{name}", 0
                    ),
                ),
                "paired_delta": summarize(
                    deltas,
                    seed=derive_deal_seed(
                        args.seed, f"delta:{name}", 0
                    ),
                ),
                "per_pair": records,
            }
        )

    payload = {
        "schema_version": 1,
        "candidate": {
            "path": str(args.candidate),
            "sha256": file_sha256(args.candidate),
        },
        "parent": {
            "path": str(args.parent),
            "sha256": file_sha256(args.parent),
        },
        "pairs_per_matchup": args.pairs,
        "base_seed": args.seed,
        "direct": {
            "candidate_score": summarize(
                [pair.score for pair in direct],
                seed=derive_deal_seed(args.seed, "direct_score", 0),
            ),
            "candidate_advantage": summarize(
                direct_advantages,
                seed=derive_deal_seed(args.seed, "direct_advantage", 0),
            ),
            "per_pair": [asdict(pair) for pair in direct],
        },
        "pool": pool_results,
        "combined_pool_delta": summarize(
            combined_deltas,
            seed=derive_deal_seed(args.seed, "combined_pool_delta", 0),
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "direct": payload["direct"]["candidate_score"],
                "direct_advantage": payload["direct"][
                    "candidate_advantage"
                ],
                "combined_pool_delta": payload["combined_pool_delta"],
            },
            indent=2,
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
