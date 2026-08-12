"""Process-parallel vector rollouts for the mixed-player Puffer league."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import time
from typing import Sequence

import torch

from batched_pesten import deck_count_for_players


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
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from benchmark_agents import load_ppo_agent
    from temporal_puffer_v4 import NativePufferTemporalPolicy

    policy = NativePufferTemporalPolicy(
        hidden_size=hidden_size,
        num_layers=num_layers,
        mode_conditioned_heads=mode_conditioned_heads,
    )
    policy.load_state_dict(state_dict, assign=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    _WORKER_POLICY = policy
    _WORKER_CHAMPION = load_ppo_agent(Path(champion_path), "cpu")


def _collect_player_count(task: dict):
    if _WORKER_POLICY is None or _WORKER_CHAMPION is None:
        raise RuntimeError("vector rollout worker was not initialized")
    from temporal_batched_speed import BatchRolloutStats, collect_league_batch_batched

    started = time.perf_counter()
    stats = BatchRolloutStats()
    episodes, metadata = collect_league_batch_batched(
        _WORKER_POLICY,
        _WORKER_CHAMPION,
        [_WORKER_POLICY],
        games=task["games"],
        seed=task["seed"],
        schedule_offset=task["schedule_offset"],
        schedule=task["schedule"],
        player_count=task["player_count"],
        game_indices=task["game_indices"],
        stats=stats,
    )
    if task["prepack_training"]:
        from packed_puffer_rollout import pack_puffer_episode

        episodes = [
            pack_puffer_episode(_WORKER_POLICY, episode)
            for episode in episodes
        ]
    return {
        "kind": "vector",
        "player_count": task["player_count"],
        "game_indices": task["game_indices"],
        "episodes": episodes,
        "metadata": metadata,
        "stats": asdict(stats),
        "task_seconds": time.perf_counter() - started,
    }


def _collect_tree_chunk(task: dict):
    if _WORKER_POLICY is None:
        raise RuntimeError("tree rollout worker was not initialized")
    from temporal_batched_speed import _collect_scalar_tree_game

    started = time.perf_counter()
    rows = []
    total_turns = decisions = 0
    for game_index, player_count in task["games"]:
        episodes, turns = _collect_scalar_tree_game(
            _WORKER_POLICY,
            absolute_game_index=game_index,
            deal_seed=task["seed"] + game_index,
            player_count=player_count,
            max_turns=1000,
        )
        if task["prepack_training"]:
            from packed_puffer_rollout import pack_puffer_episode

            episodes = [
                pack_puffer_episode(_WORKER_POLICY, episode)
                for episode in episodes
            ]
        rows.append((game_index, episodes))
        total_turns += turns
        decisions += sum(len(trajectory) for trajectory, _ in episodes)
    return {
        "kind": "tree",
        "rows": rows,
        "metadata": {
            "games": {"bayesian_tree": len(rows)},
            "decisions": {"bayesian_tree": decisions},
            "total_turns": total_turns,
            "trajectories": len(rows),
            "task_seconds": time.perf_counter() - started,
        },
    }


def _collect_task(task: dict):
    return (
        _collect_player_count(task)
        if task["kind"] == "vector"
        else _collect_tree_chunk(task)
    )


def _episode_count_for_game(
    game_index: int,
    player_count: int,
    schedule_offset: int,
    schedule: Sequence[str],
) -> int:
    opponent = schedule[(schedule_offset + game_index) % len(schedule)]
    return player_count if opponent == "current_selfplay" else 1


def _merge_results(
    results: Sequence[dict],
    *,
    games: int,
    schedule_offset: int,
    schedule: Sequence[str],
    player_count_schedule: Sequence[int],
) -> tuple[list, dict]:
    episodes_by_game = {}
    game_counts = Counter()
    decision_counts = Counter()
    player_counts = Counter()
    deck_counts = {}
    fallbacks = Counter()
    total_turns = trajectories = 0
    cohort_stats = {}
    tree_chunk_seconds = []
    for result in results:
        if result["kind"] == "tree":
            for game_index, episodes in result["rows"]:
                episodes_by_game[game_index] = episodes
            metadata = result["metadata"]
            game_counts.update(metadata["games"])
            decision_counts.update(metadata["decisions"])
            total_turns += int(metadata["total_turns"])
            trajectories += int(metadata["trajectories"])
            tree_chunk_seconds.append(float(metadata["task_seconds"]))
            continue
        count = int(result["player_count"])
        indices = result["game_indices"]
        episodes = result["episodes"]
        cursor = 0
        for game_index in indices:
            episode_count = _episode_count_for_game(
                game_index, count, schedule_offset, schedule
            )
            episodes_by_game[game_index] = episodes[
                cursor : cursor + episode_count
            ]
            cursor += episode_count
        if cursor != len(episodes):
            raise RuntimeError("vector cohort episode count does not match schedule")

        metadata = result["metadata"]
        game_counts.update(metadata["games"])
        decision_counts.update(metadata["decisions"])
        total_turns += int(metadata["total_turns"])
        trajectories += int(metadata["trajectories"])
        cohort_stats.setdefault(str(count), []).append(
            {
                **result["stats"],
                "task_seconds": result["task_seconds"],
            }
        )

    for game_index in range(games):
        count = player_count_schedule[
            (schedule_offset + game_index) % len(player_count_schedule)
        ]
        player_counts[str(count)] += 1
        deck_counts[str(count)] = deck_count_for_players(count)
        opponent = schedule[(schedule_offset + game_index) % len(schedule)]
        if opponent == "bayesian_tree" and count >= 7:
            fallbacks["bayesian_tree_multideck_to_smarter_rule"] += 1

    if len(episodes_by_game) != games:
        missing = sorted(set(range(games)) - set(episodes_by_game))
        raise RuntimeError(f"vector rollout omitted games: {missing[:10]}")
    episodes = [
        episode
        for game_index in range(games)
        for episode in episodes_by_game[game_index]
    ]
    metadata = {
        "games": dict(game_counts),
        "decisions": dict(decision_counts),
        "total_turns": total_turns,
        "trajectories": trajectories,
        "player_counts": dict(player_counts),
        "deck_count_by_player_count": deck_counts,
        "vector_cohort_stats": cohort_stats,
        "tree_chunk_seconds": tree_chunk_seconds,
    }
    if fallbacks:
        metadata["league_fallbacks"] = dict(fallbacks)
    return episodes, metadata


def collect_league_batch_vector_parallel(
    policy,
    *,
    games: int,
    seed: int,
    schedule_offset: int,
    schedule: Sequence[str],
    player_counts: Sequence[int],
    workers: int,
    champion_path: Path = Path("ppo_actor_100run3.pth"),
    prepack_training: bool = False,
):
    """Run one batch through a temporary vector worker pool."""
    with VectorPufferRolloutPool(
        policy,
        workers=workers,
        champion_path=champion_path,
    ) as pool:
        return pool.collect(
            policy,
            games=games,
            seed=seed,
            schedule_offset=schedule_offset,
            schedule=schedule,
            player_counts=player_counts,
            prepack_training=prepack_training,
        )


class VectorPufferRolloutPool:
    """Persistent workers reading an atomically refreshed shared policy."""

    def __init__(
        self,
        policy,
        *,
        workers: int,
        champion_path: Path = Path("ppo_actor_100run3.pth"),
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        self.workers = workers
        self.closed = False
        self.shared_state = {
            name: value.detach().cpu().clone().share_memory_()
            for name, value in policy.state_dict().items()
        }
        context = multiprocessing.get_context("spawn")
        self.pool = context.Pool(
            processes=workers,
            initializer=_initialize_worker,
            initargs=(
                self.shared_state,
                int(policy.hidden_size),
                int(policy.num_layers),
                bool(getattr(policy, "mode_conditioned_heads", False)),
                str(champion_path.resolve()),
            ),
        )

    def _refresh_policy(self, policy) -> None:
        if self.closed:
            raise RuntimeError("vector rollout pool is closed")
        state = policy.state_dict()
        if state.keys() != self.shared_state.keys():
            raise ValueError("rollout policy structure changed")
        with torch.no_grad():
            for name, target in self.shared_state.items():
                target.copy_(state[name].detach(), non_blocking=False)

    def collect(
        self,
        policy,
        *,
        games: int,
        seed: int,
        schedule_offset: int,
        schedule: Sequence[str],
        player_counts: Sequence[int],
        prepack_training: bool = False,
    ):
        """Refresh shared weights, run cohorts, and restore game ordering."""
        self._refresh_policy(policy)
        tasks, schedule, player_counts = _build_tasks(
            games=games,
            seed=seed,
            schedule_offset=schedule_offset,
            schedule=schedule,
            player_counts=player_counts,
            workers=self.workers,
            prepack_training=prepack_training,
        )
        results = self.pool.map(_collect_task, tasks, chunksize=1)
        return _merge_results(
            results,
            games=games,
            schedule_offset=schedule_offset,
            schedule=schedule,
            player_count_schedule=player_counts,
        )

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.pool.close()
            self.pool.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _build_tasks(
    *,
    games: int,
    seed: int,
    schedule_offset: int,
    schedule: Sequence[str],
    player_counts: Sequence[int],
    workers: int,
    prepack_training: bool,
):
    if games < 1 or workers < 1:
        raise ValueError("games and workers must be positive")
    schedule = tuple(schedule)
    player_counts = tuple(player_counts)
    if not schedule or not player_counts or any(count < 2 for count in player_counts):
        raise ValueError("schedule and valid player counts are required")
    grouped_indices: dict[int, list[int]] = {}
    tree_games = []
    for game_index in range(games):
        count = player_counts[
            (schedule_offset + game_index) % len(player_counts)
        ]
        opponent = schedule[(schedule_offset + game_index) % len(schedule)]
        if opponent == "bayesian_tree" and count < 7:
            tree_games.append((game_index, count))
        else:
            grouped_indices.setdefault(count, []).append(game_index)
    tasks = [
        {
            "kind": "vector",
            "games": games,
            "seed": seed,
            "schedule_offset": schedule_offset,
            "schedule": schedule,
            "player_count": count,
            "game_indices": tuple(indices),
            "prepack_training": prepack_training,
        }
        for count, indices in grouped_indices.items()
    ]
    if tree_games:
        tree_chunk_size = 1
        tasks.extend(
            {
                "kind": "tree",
                "seed": seed,
                "games": tuple(tree_games[start : start + tree_chunk_size]),
                "prepack_training": prepack_training,
            }
            for start in range(0, len(tree_games), tree_chunk_size)
        )
    return tasks, schedule, player_counts
