#!/usr/bin/env python3
"""Same-deal dev benchmark for the draw-retry-corrected frozen PPO."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import numpy as np

from benchmark_hybrid import (
    bootstrap_mean_ci,
    derive_deal_seed,
    file_sha256,
    run_seat_pair,
)
from corrected_ppo_agent import (
    CorrectedPPOStats,
    make_corrected_ppo_agent_factory,
)
from hybrid_agent import load_frozen_ppo_actor, make_ppo_agent_factory
import randomAgent
import smartAgent


UNSEALED_SEED_LIMIT = 92_000_000
POOL: dict[str, Callable] = {
    "random": randomAgent.AgentRandom,
    "smart": smartAgent.Agent,
    "smarter": smartAgent.SmarterAgent,
}


def derive_unsealed_seed(
    base_seed: int,
    namespace: str,
    pair_index: int,
) -> int:
    """Derive deterministic evaluation seeds strictly below 92 million."""
    if not 0 <= base_seed < UNSEALED_SEED_LIMIT:
        raise ValueError(
            f"base seed must be in [0, {UNSEALED_SEED_LIMIT})"
        )
    return (
        derive_deal_seed(base_seed, namespace, pair_index)
        % UNSEALED_SEED_LIMIT
    )


def summarize(values: Sequence[float], *, seed: int) -> dict:
    return {
        "mean": float(np.mean(values)),
        "bootstrap_ci95": list(
            bootstrap_mean_ci(values, seed=seed)
        ),
    }


def run_benchmark(
    corrected_factory: Callable,
    legacy_factory: Callable,
    *,
    pairs: int,
    base_seed: int,
    pool_names: Sequence[str],
    max_turns: int,
) -> dict:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if not pool_names:
        raise ValueError("at least one pool policy is required")
    unknown = sorted(set(pool_names) - set(POOL))
    if unknown:
        raise ValueError(f"unknown pool policies: {', '.join(unknown)}")
    used_deal_seeds: set[int] = set()

    def deal_seed(namespace: str, pair_index: int) -> int:
        seed = derive_unsealed_seed(base_seed, namespace, pair_index)
        if seed in used_deal_seeds:
            raise RuntimeError("derived evaluation seed collision")
        used_deal_seeds.add(seed)
        return seed

    direct = []
    for pair_index in range(pairs):
        seed = deal_seed("corrected_vs_legacy", pair_index)
        direct.append(
            run_seat_pair(
                corrected_factory,
                legacy_factory,
                pair_index=pair_index,
                deal_seed=seed,
                max_turns=max_turns,
            )
        )
    direct_advantages = [2.0 * pair.score - 1.0 for pair in direct]

    pool_results = []
    combined_deltas = []
    for opponent_name in pool_names:
        records = []
        opponent_factory = POOL[opponent_name]
        for pair_index in range(pairs):
            seed = deal_seed(f"pool:{opponent_name}", pair_index)
            corrected_pair = run_seat_pair(
                corrected_factory,
                opponent_factory,
                pair_index=pair_index,
                deal_seed=seed,
                max_turns=max_turns,
            )
            legacy_pair = run_seat_pair(
                legacy_factory,
                opponent_factory,
                pair_index=pair_index,
                deal_seed=seed,
                max_turns=max_turns,
            )
            delta = corrected_pair.score - legacy_pair.score
            combined_deltas.append(delta)
            records.append(
                {
                    "pair_index": pair_index,
                    "deal_seed": seed,
                    "corrected": asdict(corrected_pair),
                    "legacy": asdict(legacy_pair),
                    "paired_score_delta": delta,
                }
            )
        deltas = [record["paired_score_delta"] for record in records]
        pool_results.append(
            {
                "opponent": opponent_name,
                "corrected_score": summarize(
                    [record["corrected"]["score"] for record in records],
                    seed=derive_unsealed_seed(
                        base_seed, f"bootstrap:corrected:{opponent_name}", 0
                    ),
                ),
                "legacy_score": summarize(
                    [record["legacy"]["score"] for record in records],
                    seed=derive_unsealed_seed(
                        base_seed, f"bootstrap:legacy:{opponent_name}", 0
                    ),
                ),
                "paired_score_delta": summarize(
                    deltas,
                    seed=derive_unsealed_seed(
                        base_seed, f"bootstrap:delta:{opponent_name}", 0
                    ),
                ),
                "per_pair": records,
            }
        )

    stats = corrected_factory.stats.to_dict()
    if stats["outside_retry_action_mismatches"]:
        raise RuntimeError("corrected wrapper changed a non-retry action")
    return {
        "direct": {
            "corrected_score": summarize(
                [pair.score for pair in direct],
                seed=derive_unsealed_seed(
                    base_seed, "bootstrap:direct_score", 0
                ),
            ),
            "corrected_advantage": summarize(
                direct_advantages,
                seed=derive_unsealed_seed(
                    base_seed, "bootstrap:direct_advantage", 0
                ),
            ),
            "per_pair": [asdict(pair) for pair in direct],
        },
        "pool": pool_results,
        "combined_pool_delta": summarize(
            combined_deltas,
            seed=derive_unsealed_seed(
                base_seed, "bootstrap:combined_pool_delta", 0
            ),
        ),
        "action_audit": {
            **stats,
            "exact_action_parity_outside_retry": True,
        },
        "deal_seed_audit": {
            "count": len(used_deal_seeds),
            "minimum": min(used_deal_seeds),
            "maximum": max(used_deal_seeds),
            "all_below_92m": max(used_deal_seeds) < UNSEALED_SEED_LIMIT,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=82_607_031)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--pool", default="random,smart,smarter")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pool_names = tuple(
        name.strip() for name in args.pool.split(",") if name.strip()
    )
    actor, device = load_frozen_ppo_actor(args.model, args.device)
    stats = CorrectedPPOStats()
    corrected_factory = make_corrected_ppo_agent_factory(
        actor, device, stats=stats
    )
    legacy_factory = make_ppo_agent_factory(actor, device, shield=False)
    started = time.perf_counter()
    results = run_benchmark(
        corrected_factory,
        legacy_factory,
        pairs=args.pairs,
        base_seed=args.seed,
        pool_names=pool_names,
        max_turns=args.max_turns,
    )
    payload = {
        "schema_version": 1,
        "benchmark": "draw_retry_corrected_frozen_ppo",
        "model": {
            "path": str(args.model),
            "sha256": file_sha256(args.model),
            "frozen": True,
            "greedy": True,
            "device": str(device),
        },
        "protocol": {
            "development_seed_space": "all deal seeds < 92000000",
            "base_seed": args.seed,
            "pairs_per_matchup": args.pairs,
            "games_per_pair": 2,
            "same_deal_candidate_and_legacy": True,
            "seat_balanced": True,
            "max_turns": args.max_turns,
        },
        "privacy": {
            "observation": "unchanged production 121-feature observation",
            "additional_tracked_state": [
                "just-drawn own card number",
                "public Game.turn_count",
            ],
            "opponent_card_identities": False,
            "draw_pile_inspection": False,
        },
        **results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
