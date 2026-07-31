#!/usr/bin/env python3
"""Profile the exact public-context fastpath in complete best-player games."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time

import numpy as np
import torch

import game
import ppo2
from ai_observation import (
    DIRECTION_INDEX,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PLAYER_HAND_SIZES_END,
    PLAYER_HAND_SIZES_START,
    PLAYER_HAND_SIZE_SLOTS,
    PLAYER_INDEX,
    SORT_COUNT,
    SORTS_PLAYED_START,
    populate_game_context,
    relative_player_indices,
)
from batched_pesten import POLICY_RL
from benchmark_checkpoint_batched import (
    BatchGameSpec,
    run_batched_specs,
)
from best_player import BestPlayerConfig, load_best_player


DEVELOPMENT_BASE_SEED = 89_607_031


def scalar_reference(
    observation,
    *,
    sorts_played,
    hand_sizes,
    player_index,
    direction,
    is_change_sort=False,
):
    """The pre-optimization context writer used as an exact reference."""
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(
            f"observation must contain {OBSERVATION_SIZE} values, "
            f"got {len(observation)}"
        )
    if len(sorts_played) < SORT_COUNT:
        raise ValueError(
            f"sorts_played must contain at least {SORT_COUNT} values, "
            f"got {len(sorts_played)}"
        )
    for offset in range(SORT_COUNT):
        observation[SORTS_PLAYED_START + offset] = sorts_played[offset]
    observation[MODE_INDEX] = 1 if is_change_sort else 0
    for feature_index in range(
        PLAYER_HAND_SIZES_START,
        PLAYER_HAND_SIZES_END,
    ):
        observation[feature_index] = 0
    order = relative_player_indices(
        len(hand_sizes),
        player_index,
        direction,
    )
    for slot, ordered_player_index in enumerate(
        order[:PLAYER_HAND_SIZE_SLOTS]
    ):
        observation[PLAYER_HAND_SIZES_START + slot] = (
            hand_sizes[ordered_player_index]
        )
    observation[PLAYER_INDEX] = player_index
    observation[DIRECTION_INDEX] = direction
    return observation


def _run_canonical(factory, *, games: int, context_writer) -> tuple:
    original = ppo2.populate_game_context
    ppo2.populate_game_context = context_writer
    outcomes = []
    try:
        for game_index in range(games):
            seed = DEVELOPMENT_BASE_SEED + game_index
            random.seed(seed)
            np.random.seed(seed)
            instance = game.Game([factory, factory])
            turns, winner = instance.auto_simulate(max_turns=1_000)
            outcomes.append((winner, turns))
    finally:
        ppo2.populate_game_context = original
    return tuple(outcomes)


def _compare_canonical(
    factory,
    *,
    games: int,
    rounds: int,
) -> tuple[float, float, tuple]:
    timings = {"scalar": [], "optimized": []}
    expected_outcomes = None
    writers = {
        "scalar": scalar_reference,
        "optimized": populate_game_context,
    }
    for round_index in range(rounds):
        order = (
            ("scalar", "optimized")
            if round_index % 2 == 0
            else ("optimized", "scalar")
        )
        for name in order:
            started = time.perf_counter()
            outcomes = _run_canonical(
                factory,
                games=games,
                context_writer=writers[name],
            )
            timings[name].append(time.perf_counter() - started)
            if expected_outcomes is None:
                expected_outcomes = outcomes
            elif outcomes != expected_outcomes:
                raise RuntimeError(
                    "context writer changed winners or turn counts"
                )
    return (
        statistics.median(timings["scalar"]),
        statistics.median(timings["optimized"]),
        expected_outcomes,
    )


def _batched_profile(loaded, *, games: int) -> dict:
    specs = []
    for game_index in range(games):
        seed = DEVELOPMENT_BASE_SEED + 100_000 + game_index
        specs.append(
            BatchGameSpec(
                section="development_hotpath_profile",
                opponent=None,
                pair_index=game_index,
                evaluated_key="best",
                evaluated_seat=0,
                deal_seed=seed,
                policy_kinds=(POLICY_RL, POLICY_RL),
                actor_keys=("best", "best"),
            )
        )
    outcomes, statistics_record = run_batched_specs(
        specs,
        {"best": loaded.batched_actor_policy()},
        max_turns=1_000,
    )
    outcome_records = [
        {"winner": outcome.winner, "turns": outcome.turns}
        for outcome in outcomes
    ]
    outcome_sha256 = hashlib.sha256(
        json.dumps(
            outcome_records,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return {
        "winner_turn_pairs_sha256": outcome_sha256,
        "statistics": statistics_record,
    }


def run_benchmark(
    *,
    canonical_games: int,
    batched_games: int,
    rounds: int,
) -> dict:
    if DEVELOPMENT_BASE_SEED + 100_000 + batched_games >= 92_000_000:
        raise ValueError("development seeds must stay below 92m")
    loaded = load_best_player(BestPlayerConfig(device="cpu"))
    factory = loaded.canonical_game_factory()
    (
        scalar_seconds,
        optimized_seconds,
        _canonical_outcomes,
    ) = _compare_canonical(
        factory,
        games=canonical_games,
        rounds=rounds,
    )
    return {
        "scope": "development-only non-sealed performance profile",
        "checkpoint_sha256": loaded.metadata["checkpoint"]["sha256"],
        "production_projection": loaded.metadata["projection"],
        "canonical": {
            "games": canonical_games,
            "rounds": rounds,
            "winner_turn_pairs_exact": True,
            "scalar_reference_seconds": scalar_seconds,
            "optimized_seconds": optimized_seconds,
            "speedup": scalar_seconds / optimized_seconds,
        },
        "batched": {
            "games": batched_games,
            **_batched_profile(loaded, games=batched_games),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-games", type=int, default=30)
    parser.add_argument("--batched-games", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if min(
        args.canonical_games,
        args.batched_games,
        args.rounds,
        args.torch_threads,
    ) < 1:
        parser.error("game counts, rounds, and threads must be positive")
    torch.set_num_threads(args.torch_threads)
    print(
        json.dumps(
            run_benchmark(
                canonical_games=args.canonical_games,
                batched_games=args.batched_games,
                rounds=args.rounds,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
