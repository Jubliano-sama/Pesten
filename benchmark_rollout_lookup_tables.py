#!/usr/bin/env python3
"""Isolate immutable card-lookup-table rollout throughput and parity."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
import time

import numpy as np
import torch

import batched_pesten as batched
from benchmark_rollout_smart_fastpath import (
    assert_rollout_parity,
    run_once,
)
from hybrid_agent import load_frozen_ppo_actor
from seed_integrity import DEVELOPMENT_SEED_LIMIT


class _RecomputedRankMasks:
    def __getitem__(self, rank):
        return batched.CARD_RANK == rank


class _RecomputedCompatibilityMasks:
    def __getitem__(self, key):
        suit, rank = key
        return (
            (batched.CARD_RANK == rank)
            | (batched.CARD_RANK == 0)
            | (batched.CARD_RANK == 13)
            | (batched.CARD_SORT == suit)
        )


class _RecomputedPenaltyMask:
    def __getitem__(self, indices):
        return np.isin(
            batched.CARD_RANK[indices],
            tuple(batched.PENALTY_RANKS),
        )


class LegacyLookupBatchedPesten(batched.BatchedPesten):
    """Current engine with pre-lookup Smart suit scoring."""

    def choose_suit(self, game_index: int, policy_kind: int) -> int:
        if policy_kind != batched.POLICY_SMART:
            return super().choose_suit(game_index, policy_kind)
        player_index = int(self.current_player[game_index])
        hand = self._ordered_hand(game_index, player_index)
        best_suit = 0
        best_score = -float("inf")
        ranks = batched.CARD_RANK[hand]
        suits = batched.CARD_SORT[hand]
        for suit in range(4):
            same_suit = suits == suit
            total = int(np.count_nonzero(same_suit))
            penalty = int(
                np.count_nonzero(
                    same_suit
                    & np.isin(ranks, tuple(batched.PENALTY_RANKS))
                )
            )
            score = (
                0.43244878288347055 * total
                - (-1.8463831782446556) * penalty
            )
            if score > best_score:
                best_score = score
                best_suit = suit
        return best_suit


@contextmanager
def recomputed_tables():
    saved = (
        batched.CARD_RANK_MASKS,
        batched.CARD_COMPATIBILITY_MASKS,
        batched.CARD_IS_PENALTY,
    )
    batched.CARD_RANK_MASKS = _RecomputedRankMasks()
    batched.CARD_COMPATIBILITY_MASKS = _RecomputedCompatibilityMasks()
    batched.CARD_IS_PENALTY = _RecomputedPenaltyMask()
    try:
        yield
    finally:
        (
            batched.CARD_RANK_MASKS,
            batched.CARD_COMPATIBILITY_MASKS,
            batched.CARD_IS_PENALTY,
        ) = saved


def measured_run(
    engine_type,
    actor,
    *,
    games,
    seed,
    device,
    max_turns,
):
    context = (
        recomputed_tables()
        if engine_type is LegacyLookupBatchedPesten
        else nullcontext()
    )
    with context:
        return run_once(
            engine_type,
            actor,
            games=games,
            seed=seed,
            device=device,
            max_turns=max_turns,
        )


def benchmark(args) -> dict:
    if args.seed < 0 or args.seed + args.games > DEVELOPMENT_SEED_LIMIT:
        raise ValueError("benchmark seeds must remain in development range")
    device = torch.device(args.device)
    actor, _ = load_frozen_ppo_actor(args.actor, device)
    before_times = []
    after_times = []
    parity = None
    for repeat in range(args.repeats):
        order = (
            (LegacyLookupBatchedPesten, batched.BatchedPesten)
            if repeat % 2 == 0
            else (batched.BatchedPesten, LegacyLookupBatchedPesten)
        )
        measured = {}
        for engine_type in order:
            measured[engine_type] = measured_run(
                engine_type,
                actor,
                games=args.games,
                seed=args.seed,
                device=device,
                max_turns=args.max_turns,
            )
        before, before_seconds = measured[LegacyLookupBatchedPesten]
        after, after_seconds = measured[batched.BatchedPesten]
        parity = assert_rollout_parity(before, after)
        before_times.append(before_seconds)
        after_times.append(after_seconds)
    before_median = float(np.median(before_times))
    after_median = float(np.median(after_times))
    return {
        "schema_version": 1,
        "benchmark": "vectorized_rollout_immutable_card_tables",
        "workload": {
            "games": args.games,
            "repeats": args.repeats,
            "seed_start": args.seed,
            "max_turns": args.max_turns,
            "torch_threads": args.torch_threads,
        },
        "seed_audit": {
            "development_only": True,
            "sealed_outcomes_used": False,
        },
        "before": {
            "seconds": before_times,
            "median_seconds": before_median,
        },
        "after": {
            "seconds": after_times,
            "median_seconds": after_median,
        },
        "speedup": before_median / after_median,
        "parity": parity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor", type=Path, default=Path("ppo_actor_100run3.pth")
    )
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=84_340_001)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    payload = benchmark(args)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
