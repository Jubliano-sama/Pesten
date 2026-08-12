"""Process-parallel, exact-semantics rollout collection for Puffer training."""

from __future__ import annotations

from collections import Counter
import copy
import math
import multiprocessing
from pathlib import Path
from typing import Sequence

import torch


_WORKER_POLICY = None
_WORKER_CHAMPION = None


def _initialize_worker(
    state_dict: dict,
    hidden_size: int,
    num_layers: int,
    mode_conditioned_heads: bool,
    champion_path: str,
) -> None:
    global _WORKER_POLICY, _WORKER_CHAMPION
    torch.set_num_threads(1)
    from benchmark_agents import load_ppo_agent
    from temporal_puffer_v4 import NativePufferTemporalPolicy

    policy = NativePufferTemporalPolicy(
        hidden_size=hidden_size,
        num_layers=num_layers,
        mode_conditioned_heads=mode_conditioned_heads,
    )
    policy.load_state_dict(state_dict)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    _WORKER_POLICY = policy
    _WORKER_CHAMPION = load_ppo_agent(Path(champion_path), "cpu")


def _collect_chunk(task: dict):
    if _WORKER_POLICY is None or _WORKER_CHAMPION is None:
        raise RuntimeError("parallel rollout worker was not initialized")
    from temporal_league import collect_league_batch

    return collect_league_batch(
        _WORKER_POLICY,
        _WORKER_CHAMPION,
        [_WORKER_POLICY],
        games=task["games"],
        seed=task["seed"],
        schedule_offset=task["schedule_offset"],
        mixture=task["mixture"],
        player_counts=task["player_counts"],
        game_index_offset=task["game_index_offset"],
    )


def _merge_metadata(rows: Sequence[dict]) -> dict:
    counter_fields = (
        "games",
        "decisions",
        "player_counts",
        "league_fallbacks",
    )
    counters = {name: Counter() for name in counter_fields}
    deck_counts = {}
    total_turns = trajectories = 0
    for row in rows:
        total_turns += int(row.get("total_turns", 0))
        trajectories += int(row.get("trajectories", 0))
        for name in counter_fields:
            counters[name].update(row.get(name, {}))
        for players, deck_count in row.get(
            "deck_count_by_player_count", {}
        ).items():
            previous = deck_counts.setdefault(players, deck_count)
            if previous != deck_count:
                raise RuntimeError(
                    "parallel shards disagree on deck count: "
                    f"players={players}, first={previous}, current={deck_count}"
                )
    result = {
        "games": dict(counters["games"]),
        "decisions": dict(counters["decisions"]),
        "total_turns": total_turns,
        "trajectories": trajectories,
    }
    if counters["player_counts"]:
        result["player_counts"] = dict(counters["player_counts"])
        result["deck_count_by_player_count"] = deck_counts
    if counters["league_fallbacks"]:
        result["league_fallbacks"] = dict(counters["league_fallbacks"])
    return result


def collect_league_batch_parallel(
    policy,
    *,
    games: int,
    seed: int,
    schedule_offset: int = 0,
    mixture,
    player_counts,
    workers: int,
    chunks_per_worker: int = 2,
    champion_path: Path = Path("ppo_actor_100run3.pth"),
):
    """Collect the same seeded games in ordered, load-balanced processes.

    Each task receives a contiguous game-index interval. Seeds, opponent
    schedule, player-count schedule, evaluated seat, and returned episode order
    therefore match the sequential collector exactly.
    """
    if games < 1:
        raise ValueError("games must be positive")
    if workers < 1:
        raise ValueError("workers must be positive")
    if chunks_per_worker < 1:
        raise ValueError("chunks_per_worker must be positive")
    if workers == 1:
        from benchmark_agents import load_ppo_agent
        from temporal_league import collect_league_batch

        champion = load_ppo_agent(champion_path, "cpu")
        return collect_league_batch(
            policy,
            champion,
            [policy],
            games=games,
            seed=seed,
            schedule_offset=schedule_offset,
            mixture=mixture,
            player_counts=player_counts,
        )

    worker_count = min(workers, games)
    task_count = min(games, worker_count * chunks_per_worker)
    chunk_size = math.ceil(games / task_count)
    tasks = []
    for start in range(0, games, chunk_size):
        count = min(chunk_size, games - start)
        tasks.append(
            {
                "games": count,
                "seed": seed + start,
                "schedule_offset": schedule_offset + start,
                "game_index_offset": start,
                "mixture": tuple(mixture),
                "player_counts": tuple(player_counts),
            }
        )

    state_dict = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=worker_count,
        initializer=_initialize_worker,
        initargs=(
            state_dict,
            int(policy.hidden_size),
            int(policy.num_layers),
            bool(getattr(policy, "mode_conditioned_heads", False)),
            str(champion_path.resolve()),
        ),
    ) as pool:
        results = pool.map(_collect_chunk, tasks)

    episodes = [
        episode
        for chunk_episodes, _ in results
        for episode in chunk_episodes
    ]
    metadata = _merge_metadata([row for _, row in results])
    return episodes, metadata
