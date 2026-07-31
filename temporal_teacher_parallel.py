"""Deterministic multi-process collection of privacy-safe PPO imitation data.

Workers receive only a checkpoint path and public game index/seed metadata.
They return ``ImitationSample`` episodes containing the existing immutable
``DecisionInput`` boundary.  Hidden opponent hands and draw-pile order never
cross the process boundary or enter benchmark reports.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch

import game
from benchmark_model_zoo import (
    DEFAULT_TEACHER,
    ImitationSample,
    _teacher_recorder_factory,
    load_frozen_teacher,
)
from randomAgent import AgentRandom
from temporal_model_zoo import LargeRecurrentTemporalPolicy
from temporal_observation import (
    DecisionInput,
    assert_policy_schema_is_hidden_state_free,
    serialize_safe_history,
)


_WORKER_TEACHER = None
_WORKER_RECORDER = None


def _initialize_worker(
    teacher_path: str,
    torch_threads: int,
) -> None:
    """Load one frozen teacher per process and pin CPU thread usage."""
    global _WORKER_TEACHER, _WORKER_RECORDER
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # A reused interpreter may already have initialized its interop pool.
        pass
    torch.use_deterministic_algorithms(True)
    _WORKER_TEACHER, _ = load_frozen_teacher(Path(teacher_path))
    # The recorder overrides _act with the frozen PPO.  This small policy is
    # used only to provide the existing zero/reset state contract.
    boundary_policy = LargeRecurrentTemporalPolicy(
        hidden_size=8,
        num_layers=1,
        cell="gru",
    )
    _WORKER_RECORDER = _teacher_recorder_factory(
        boundary_policy, _WORKER_TEACHER
    )


def _collect_game_worker(
    task: tuple[int, int, int],
) -> tuple[int, tuple[ImitationSample, ...]]:
    game_index, base_seed, max_turns = task
    if _WORKER_RECORDER is None:
        raise RuntimeError("teacher worker was not initialized")
    deal_seed = base_seed + game_index
    random.seed(deal_seed)
    np.random.seed(deal_seed)
    torch.manual_seed(deal_seed)
    seat = game_index % 2
    factories = (
        [_WORKER_RECORDER, AgentRandom]
        if seat == 0
        else [AgentRandom, _WORKER_RECORDER]
    )
    match = game.Game(factories)
    match.auto_simulate(max_turns=max_turns)
    episode = tuple(match.players[seat].imitation)
    _validate_safe_episode(episode)
    return game_index, episode


def collect_teacher_episodes_parallel(
    *,
    teacher_path: Path = DEFAULT_TEACHER,
    games: int,
    seed: int,
    max_turns: int = 160,
    workers: int | None = None,
    torch_threads_per_worker: int = 1,
    start_method: str = "spawn",
) -> tuple[tuple[ImitationSample, ...], ...]:
    """Collect frozen-PPO episodes and merge them by exact game index.

    The output has exactly ``games`` entries, including an empty tuple if the
    teacher made no decision in a game.  This preserves the index-to-seed and
    alternating-seat mapping for reproducible chronological splits.
    """
    if games < 1:
        raise ValueError("games must be positive")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if workers is None:
        workers = min(4, os.cpu_count() or 1)
    if workers < 1:
        raise ValueError("workers must be positive")
    if torch_threads_per_worker < 1:
        raise ValueError("torch_threads_per_worker must be positive")
    if start_method not in multiprocessing.get_all_start_methods():
        raise ValueError(f"unsupported multiprocessing start method: {start_method}")

    assert_policy_schema_is_hidden_state_free()
    checkpoint = teacher_path.resolve(strict=True)
    tasks = [(game_index, seed, max_turns) for game_index in range(games)]
    context = multiprocessing.get_context(start_method)
    with ProcessPoolExecutor(
        max_workers=min(workers, games),
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(str(checkpoint), torch_threads_per_worker),
    ) as executor:
        unordered_results = list(
            executor.map(_collect_game_worker, tasks, chunksize=1)
        )

    # executor.map already preserves input order, but sorting and checking the
    # index makes the merge contract explicit and detects worker corruption.
    ordered_results = sorted(unordered_results, key=lambda item: item[0])
    actual_indices = [game_index for game_index, _ in ordered_results]
    expected_indices = list(range(games))
    if actual_indices != expected_indices:
        raise RuntimeError(
            "parallel collector returned missing or duplicate game indices"
        )
    return tuple(episode for _, episode in ordered_results)


def canonical_imitation_bytes(
    episodes: Sequence[Sequence[ImitationSample]],
) -> bytes:
    """Serialize only safe decisions and legal PPO labels, canonically."""
    payload = []
    for episode in episodes:
        _validate_safe_episode(episode)
        decisions = [sample.decision for sample in episode]
        payload.append(
            {
                "safe_decisions": json.loads(
                    serialize_safe_history(decisions)
                ),
                "actions": [int(sample.action) for sample in episode],
            }
        )
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def imitation_digest(
    episodes: Sequence[Sequence[ImitationSample]],
) -> str:
    return hashlib.sha256(canonical_imitation_bytes(episodes)).hexdigest()


def benchmark_worker_counts(
    *,
    teacher_path: Path = DEFAULT_TEACHER,
    games: int = 24,
    seed: int = 7301,
    max_turns: int = 80,
    worker_counts: Sequence[int] = (1, 2, 4),
    torch_threads_per_worker: int = 1,
    start_method: str = "spawn",
) -> dict:
    """Compare collectors without exposing any episode payload in the report."""
    runs = []
    reference_bytes = None
    reference_digest = None
    for workers in worker_counts:
        started = time.perf_counter()
        episodes = collect_teacher_episodes_parallel(
            teacher_path=teacher_path,
            games=games,
            seed=seed,
            max_turns=max_turns,
            workers=workers,
            torch_threads_per_worker=torch_threads_per_worker,
            start_method=start_method,
        )
        elapsed = time.perf_counter() - started
        encoded = canonical_imitation_bytes(episodes)
        digest = hashlib.sha256(encoded).hexdigest()
        if reference_bytes is None:
            reference_bytes = encoded
            reference_digest = digest
        byte_equal = encoded == reference_bytes
        semantic_equal = digest == reference_digest and all(
            left == right
            for left, right in zip(
                json.loads(encoded), json.loads(reference_bytes)
            )
        )
        decision_count = sum(len(episode) for episode in episodes)
        runs.append(
            {
                "workers": workers,
                "seconds": elapsed,
                "games_per_second": games / elapsed,
                "decisions_per_second": decision_count / elapsed,
                "games": len(episodes),
                "decisions": decision_count,
                "safe_sha256": digest,
                "byte_equal_to_one_worker": byte_equal,
                "semantic_equal_to_one_worker": semantic_equal,
            }
        )
    baseline_seconds = runs[0]["seconds"]
    for run in runs:
        run["speedup_vs_one_worker"] = baseline_seconds / run["seconds"]
    if not all(
        run["byte_equal_to_one_worker"]
        and run["semantic_equal_to_one_worker"]
        for run in runs
    ):
        raise AssertionError("worker-count invariance failed")
    return {
        "schema_version": 1,
        "scope": "privacy-safe frozen-PPO teacher collection throughput",
        "seed": seed,
        "max_turns": max_turns,
        "start_method": start_method,
        "torch_threads_per_worker": torch_threads_per_worker,
        "privacy": {
            "episode_payload_in_report": False,
            "policy_input": "DecisionInput",
            "public_draw_representation": "actor plus count only",
            "hidden_opponent_identities": False,
            "draw_pile_order": False,
        },
        "runs": runs,
    }


def _validate_safe_episode(
    episode: Sequence[ImitationSample],
) -> None:
    for sample in episode:
        if not isinstance(sample, ImitationSample):
            raise TypeError("episode contains a non-ImitationSample value")
        if not isinstance(sample.decision, DecisionInput):
            raise TypeError("imitation sample must contain DecisionInput")
        if not 0 <= sample.action < len(sample.decision.legal_actions):
            raise ValueError("teacher action is outside the action space")
        if not sample.decision.legal_actions[sample.action]:
            raise ValueError("teacher action is illegal for its DecisionInput")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--games", type=int, default=24)
    parser.add_argument("--seed", type=int, default=7301)
    parser.add_argument("--max-turns", type=int, default=80)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--torch-threads-per-worker", type=int, default=1)
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = benchmark_worker_counts(
        teacher_path=args.teacher,
        games=args.games,
        seed=args.seed,
        max_turns=args.max_turns,
        worker_counts=args.workers,
        torch_threads_per_worker=args.torch_threads_per_worker,
        start_method=args.start_method,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
