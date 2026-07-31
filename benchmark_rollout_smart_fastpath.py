#!/usr/bin/env python3
"""Parity and throughput benchmark for the vectorized smart-opponent fast path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    CARD_RANK,
    PASS_ACTION,
    PENALTY_RANKS,
    POLICY_RL,
    POLICY_SMART,
    collect_batched_rollout,
    _compatible,
)
from hybrid_agent import load_frozen_ppo_actor
from seed_integrity import DEVELOPMENT_SEED_LIMIT


DEFAULT_SEED = 84_300_001


class LegacySmartBatchedPesten(BatchedPesten):
    """Pre-fast-path smart scorer retained only for parity measurement."""

    def smart_action(self, game_index: int) -> int:
        player_index = int(self.current_player[game_index])
        hand = self._ordered_hand(game_index, player_index)
        compatible = np.asarray(
            [
                _compatible(
                    int(number),
                    int(self.current_sort[game_index]),
                    int(self.current_rank[game_index]),
                )
                for number in hand
            ],
            dtype=np.bool_,
        )
        playable = hand[compatible]
        if len(playable) == 0:
            return PASS_ACTION

        current_rank = int(self.current_rank[game_index])
        candidates = playable
        if current_rank in PENALTY_RANKS and current_rank != 7:
            matching = playable[CARD_RANK[playable] == current_rank]
            if len(matching):
                candidates = matching

        penalty_bonuses = {
            8: -1.5328572573442552,
            1: 0.04122020478114498,
            0: -0.03556864491834867,
            13: -3.0043124890687474,
            2: 0.04665718005466267,
        }
        sevens = candidates[CARD_RANK[candidates] == 7]
        if not (
            current_rank in PENALTY_RANKS
            and current_rank != 7
            and np.any(CARD_RANK[playable] == current_rank)
        ):
            candidates = np.concatenate(
                (sevens, candidates[CARD_RANK[candidates] != 7])
            )
        seven_end_values: dict[int, float] = {}
        if len(sevens):
            for number in sevens:
                remaining = hand.copy()
                offset = int(np.flatnonzero(remaining == number)[0])
                remaining = np.delete(remaining, offset)
                seven_end_values[int(number)] = self._smart_end_value(
                    int(number), remaining
                )
            max_end = max(seven_end_values.values())
        else:
            max_end = 0.0

        best_number = int(candidates[0])
        best_score = -float("inf")
        penalty_branch = (
            current_rank in PENALTY_RANKS
            and current_rank != 7
            and np.any(CARD_RANK[playable] == current_rank)
        )
        for number_value in candidates:
            number = int(number_value)
            remaining = hand.copy()
            offset = int(np.flatnonzero(remaining == number)[0])
            remaining = np.delete(remaining, offset)
            rank = int(CARD_RANK[number])
            if rank == 7:
                if penalty_branch:
                    score = 1.3119838792608858 + (
                        max_end - seven_end_values[number]
                    )
                else:
                    score = (
                        1.3119838792608858
                        * len(sevens)
                        * (max_end - seven_end_values[number])
                    )
            else:
                score = self._smart_hand_value(remaining)
                score += penalty_bonuses.get(rank, 0.0)
            if score > best_score:
                best_score = score
                best_number = number
        return best_number


def _layout(games: int) -> tuple[np.ndarray, np.ndarray]:
    kinds = np.full((games, 2), POLICY_SMART, dtype=np.int8)
    keys = np.full((games, 2), None, dtype=object)
    for game_index in range(games):
        learner_seat = game_index % 2
        kinds[game_index, learner_seat] = POLICY_RL
        keys[game_index, learner_seat] = "learner"
    return kinds, keys


def run_once(
    engine_type,
    actor: torch.nn.Module,
    *,
    games: int,
    seed: int,
    device: torch.device,
    max_turns: int,
    smart_agents_have_game_reference: bool = False,
):
    if seed < 0 or seed + games - 1 >= DEVELOPMENT_SEED_LIMIT:
        raise ValueError(
            "explicit deal range must stay below the shared development limit"
        )
    kinds, keys = _layout(games)
    engine = engine_type(
        kinds,
        keys,
        np.arange(seed, seed + games, dtype=np.uint32),
        max_turns=max_turns,
        smart_agents_have_game_reference=smart_agents_have_game_reference,
    )
    torch.manual_seed(seed + 10_000)
    started = time.perf_counter()
    result = collect_batched_rollout(
        engine,
        {
            "learner": ActorPolicy(
                actor=actor,
                device=device,
                temperature=0.8,
                collect_data=True,
                greedy=False,
            )
        },
        output_device=torch.device("cpu"),
    )
    return result, time.perf_counter() - started


def assert_rollout_parity(reference, optimized) -> dict:
    tensor_fields = (
        "observations",
        "actions",
        "log_probabilities",
        "rewards",
    )
    for field in tensor_fields:
        if not torch.equal(getattr(reference, field), getattr(optimized, field)):
            raise AssertionError(f"rollout tensor differs: {field}")
    if reference.episode_lengths != optimized.episode_lengths:
        raise AssertionError("episode lengths differ")
    scalar_fields = (
        "environment_turns",
        "completed_games",
        "policy_decisions",
        "inference_calls",
        "peak_buffer_bytes",
        "wins",
        "draws",
    )
    for field in scalar_fields:
        if getattr(reference, field) != getattr(optimized, field):
            raise AssertionError(f"rollout scalar differs: {field}")
    return {
        "tensor_fields_bit_exact": list(tensor_fields),
        "episode_lengths_exact": True,
        "scalar_fields_exact": list(scalar_fields),
        "samples": len(reference.observations),
    }


def benchmark(
    *,
    actor_path: Path,
    games: int,
    seed: int,
    repeats: int,
    device: torch.device,
    max_turns: int = 300,
) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    actor, _ = load_frozen_ppo_actor(actor_path, device)
    reference_times = []
    optimized_times = []
    parity = None
    for repeat in range(repeats):
        order = (
            (LegacySmartBatchedPesten, BatchedPesten)
            if repeat % 2 == 0
            else (BatchedPesten, LegacySmartBatchedPesten)
        )
        measured = {}
        for engine_type in order:
            result, elapsed = run_once(
                engine_type,
                actor,
                games=games,
                seed=seed,
                device=device,
                max_turns=max_turns,
            )
            measured[engine_type] = (result, elapsed)
        reference, reference_elapsed = measured[
            LegacySmartBatchedPesten
        ]
        optimized, optimized_elapsed = measured[BatchedPesten]
        parity = assert_rollout_parity(reference, optimized)
        reference_times.append(reference_elapsed)
        optimized_times.append(optimized_elapsed)
    reference_median = float(np.median(reference_times))
    optimized_median = float(np.median(optimized_times))
    return {
        "schema_version": 1,
        "benchmark": "vectorized_ppo_smart_fastpath",
        "workload": {
            "games": games,
            "max_turns": max_turns,
            "actor": str(actor_path),
            "temperature": 0.8,
            "opponent": "canonical smart",
            "repeats": repeats,
        },
        "seed_audit": {
            "strategy": "explicit consecutive development seeds",
            "minimum": seed,
            "maximum": seed + games - 1,
            "all_below_development_limit": (
                seed + games - 1 < DEVELOPMENT_SEED_LIMIT
            ),
            "development_seed_limit": DEVELOPMENT_SEED_LIMIT,
            "sealed_outcomes_used": False,
        },
        "parity": parity,
        "before": {
            "seconds": reference_times,
            "median_seconds": reference_median,
            "games_per_second": games / reference_median,
            "peak_buffer_bytes": reference.peak_buffer_bytes,
        },
        "after": {
            "seconds": optimized_times,
            "median_seconds": optimized_median,
            "games_per_second": games / optimized_median,
            "peak_buffer_bytes": optimized.peak_buffer_bytes,
        },
        "speedup": reference_median / optimized_median,
        "memory_ratio": (
            optimized.peak_buffer_bytes / reference.peak_buffer_bytes
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--actor", type=Path, default=Path("ppo_actor_100run3.pth")
    )
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    payload = benchmark(
        actor_path=args.actor,
        games=args.games,
        seed=args.seed,
        repeats=args.repeats,
        device=torch.device(args.device),
        max_turns=args.max_turns,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
