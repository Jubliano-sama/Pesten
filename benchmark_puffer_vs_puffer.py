#!/usr/bin/env python3
"""Parallel, seat-balanced comparison of two native Puffer checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import time

import numpy as np


_CANDIDATE_FACTORY = None
_OPPONENT_FACTORY = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    from temporal_puffer_v4 import load_temporal_policy_checkpoint

    policy, payload = load_temporal_policy_checkpoint(path)
    policy.eval()
    return policy, payload


def _initialize_worker(candidate_path: str, opponent_path: str) -> None:
    global _CANDIDATE_FACTORY, _OPPONENT_FACTORY
    import torch

    torch.set_num_threads(1)
    from temporal_puffer_agent import temporal_agent_factory

    candidate, _ = _load(Path(candidate_path))
    opponent, _ = _load(Path(opponent_path))
    _CANDIDATE_FACTORY = temporal_agent_factory(candidate, deterministic=True)
    _OPPONENT_FACTORY = temporal_agent_factory(opponent, deterministic=True)


def _evaluate_task(task: tuple[int, int]):
    if _CANDIDATE_FACTORY is None or _OPPONENT_FACTORY is None:
        raise RuntimeError("benchmark worker was not initialized")
    import game

    seed, pairs = task
    wins = losses = draws = turns = 0
    seat_wins = [0, 0]
    pair_scores = []
    for pair in range(pairs):
        deal_seed = seed + pair
        scores = []
        for candidate_seat in (0, 1):
            np.random.seed(deal_seed)
            random.seed(deal_seed)
            factories = (
                [_CANDIDATE_FACTORY, _OPPONENT_FACTORY]
                if candidate_seat == 0
                else [_OPPONENT_FACTORY, _CANDIDATE_FACTORY]
            )
            match = game.Game(factories)
            game_turns, winner = match.auto_simulate()
            turns += game_turns
            if winner is None:
                draws += 1
                scores.append(0.5)
            elif winner == candidate_seat:
                wins += 1
                seat_wins[candidate_seat] += 1
                scores.append(1.0)
            else:
                losses += 1
                scores.append(0.0)
        pair_scores.append(sum(scores) / 2.0)
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "turns": turns,
        "seat_wins": seat_wins,
        "pair_scores": pair_scores,
    }


def _summarize(rows: list[dict], elapsed_seconds: float) -> dict:
    wins = sum(row["wins"] for row in rows)
    losses = sum(row["losses"] for row in rows)
    draws = sum(row["draws"] for row in rows)
    turns = sum(row["turns"] for row in rows)
    seat_wins = [
        sum(row["seat_wins"][seat] for row in rows) for seat in (0, 1)
    ]
    pair_scores = np.asarray(
        [score for row in rows for score in row["pair_scores"]],
        dtype=np.float64,
    )
    pairs = len(pair_scores)
    games = wins + losses + draws
    score = float(pair_scores.mean())
    standard_error = float(pair_scores.std(ddof=1) / math.sqrt(pairs))
    alpha = 0.001
    hoeffding_margin = math.sqrt(math.log(2.0 / alpha) / (2.0 * pairs))
    return {
        "pairs": pairs,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "paired_normal_ci95": [
            max(0.0, score - 1.959963984540054 * standard_error),
            min(1.0, score + 1.959963984540054 * standard_error),
        ],
        "paired_standard_error": standard_error,
        "hoeffding_ci99_9": [
            max(0.0, score - hoeffding_margin),
            min(1.0, score + hoeffding_margin),
        ],
        "hoeffding_margin99_9": hoeffding_margin,
        "candidate_wins_as_seat_0": seat_wins[0],
        "candidate_wins_as_seat_1": seat_wins[1],
        "average_turns": turns / games,
        "elapsed_seconds": elapsed_seconds,
        "gate_passed_99_9": score - hoeffding_margin > 0.5,
    }


def run_parallel_benchmark(
    candidate_path: Path,
    opponent_path: Path,
    *,
    pairs: int,
    seed: int,
    workers: int,
    chunks_per_worker: int,
) -> dict:
    if pairs < 1 or workers < 1 or chunks_per_worker < 1:
        raise ValueError("pairs, workers, and chunks per worker must be positive")
    if seed < 0 or seed + pairs - 1 >= 2**32:
        raise ValueError("seed range must fit NumPy's unsigned 32-bit seed")
    worker_count = min(workers, pairs)
    task_count = min(pairs, worker_count * chunks_per_worker)
    chunk_size = math.ceil(pairs / task_count)
    tasks = [
        (seed + start, min(chunk_size, pairs - start))
        for start in range(0, pairs, chunk_size)
    ]
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with context.Pool(
        processes=worker_count,
        initializer=_initialize_worker,
        initargs=(str(candidate_path), str(opponent_path)),
    ) as pool:
        rows = pool.map(_evaluate_task, tasks)
    return _summarize(rows, time.perf_counter() - started)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunks-per-worker", type=int, default=3)
    args = parser.parse_args()
    candidate_path = args.candidate.resolve()
    opponent_path = args.opponent.resolve()
    _, candidate_payload = _load(candidate_path)
    _, opponent_payload = _load(opponent_path)
    result = run_parallel_benchmark(
        candidate_path,
        opponent_path,
        pairs=args.pairs,
        seed=args.seed,
        workers=args.workers,
        chunks_per_worker=args.chunks_per_worker,
    )
    output = {
        "schema_version": 2,
        "kind": "puffer_vs_puffer_parallel_benchmark",
        "candidate": {
            "path": str(candidate_path),
            "sha256": _sha256(candidate_path),
            "model": candidate_payload.get("model"),
        },
        "opponent": {
            "path": str(opponent_path),
            "sha256": _sha256(opponent_path),
            "model": opponent_payload.get("model"),
        },
        "protocol": {
            "pairs": args.pairs,
            "games_per_pair": 2,
            "seat_balanced": True,
            "fresh_development_seeds": True,
            "seed": args.seed,
            "workers": args.workers,
            "confidence": "distribution-free Hoeffding bound over paired scores",
        },
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
