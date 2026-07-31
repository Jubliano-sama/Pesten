#!/usr/bin/env python3
"""100-pair common-deal screen for frozen-PPO suit group averaging."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
from typing import Callable, Sequence

import torch

from benchmark_corrected_ppo import (
    POOL,
    UNSEALED_SEED_LIMIT,
    derive_unsealed_seed,
    summarize,
)
from benchmark_hybrid import file_sha256, run_seat_pair
from hybrid_agent import load_frozen_ppo_actor, make_ppo_agent_factory
from suit_symmetry_ppo_agent import (
    SuitSymmetryStats,
    make_suit_symmetry_ppo_factory,
)


def run_symmetry_screen(
    symmetry_factory: Callable,
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
        seed = deal_seed("suit_symmetry_vs_legacy", pair_index)
        direct.append(
            run_seat_pair(
                symmetry_factory,
                legacy_factory,
                pair_index=pair_index,
                deal_seed=seed,
                max_turns=max_turns,
            )
        )
    advantages = [2.0 * pair.score - 1.0 for pair in direct]

    pool_results = []
    combined_deltas = []
    for opponent_name in pool_names:
        opponent_factory = POOL[opponent_name]
        records = []
        for pair_index in range(pairs):
            seed = deal_seed(f"pool:{opponent_name}", pair_index)
            symmetry_pair = run_seat_pair(
                symmetry_factory,
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
            delta = symmetry_pair.score - legacy_pair.score
            combined_deltas.append(delta)
            records.append(
                {
                    "pair_index": pair_index,
                    "deal_seed": seed,
                    "symmetry": asdict(symmetry_pair),
                    "legacy": asdict(legacy_pair),
                    "paired_score_delta": delta,
                }
            )
        deltas = [record["paired_score_delta"] for record in records]
        pool_results.append(
            {
                "opponent": opponent_name,
                "symmetry_score": summarize(
                    [record["symmetry"]["score"] for record in records],
                    seed=derive_unsealed_seed(
                        base_seed, f"bootstrap:symmetry:{opponent_name}", 0
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

    return {
        "direct": {
            "symmetry_score": summarize(
                [pair.score for pair in direct],
                seed=derive_unsealed_seed(
                    base_seed, "bootstrap:direct_score", 0
                ),
            ),
            "symmetry_advantage": summarize(
                advantages,
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
        "decision_audit": symmetry_factory.stats.to_dict(),
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
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=83_607_031)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="CPU inference threads; one avoids severe tiny-batch oversubscription",
    )
    parser.add_argument("--pool", default="random,smart,smarter")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pool_names = tuple(
        name.strip() for name in args.pool.split(",") if name.strip()
    )
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    actor, device = load_frozen_ppo_actor(args.model, args.device)
    stats = SuitSymmetryStats()
    symmetry_factory = make_suit_symmetry_ppo_factory(
        actor, device, stats=stats
    )
    legacy_factory = make_ppo_agent_factory(actor, device, shield=False)
    started = time.perf_counter()
    results = run_symmetry_screen(
        symmetry_factory,
        legacy_factory,
        pairs=args.pairs,
        base_seed=args.seed,
        pool_names=pool_names,
        max_turns=args.max_turns,
    )
    payload = {
        "schema_version": 1,
        "benchmark": "full_s4_suit_symmetry_test_time_ensemble",
        "model": {
            "path": str(args.model),
            "sha256": file_sha256(args.model),
            "frozen": True,
            "greedy": True,
            "device": str(device),
        },
        "approach": {
            "ordinary_suit_permutations": 24,
            "aggregation": "mean mapped-back logits",
            "legal_mask": "unchanged production mask after aggregation",
            "draw_retry": "just-drawn card plus PASS correction",
        },
        "protocol": {
            "scope": "common-development screen",
            "development_seed_space": "all deal seeds < 92000000",
            "base_seed": args.seed,
            "pairs_per_matchup": args.pairs,
            "games_per_pair": 2,
            "same_deal_candidate_and_legacy": True,
            "seat_balanced": True,
            "max_turns": args.max_turns,
            "torch_threads": args.torch_threads,
        },
        "privacy": {
            "inputs": (
                "unchanged own-hand/public 121-feature observation; "
                "ordinary suit labels are relabeled synthetically"
            ),
            "additional_runtime_state": [
                "just-drawn own card number",
                "public turn number",
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
