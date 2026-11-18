"""Process-parallel vector rollouts for the mixed-player Puffer league."""

from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import time
from typing import Sequence
import warnings

import torch

from batched_pesten import deck_count_for_players


_WORKER_POLICY = None
_WORKER_POLICY_SOURCE = None
_WORKER_CHAMPION = None
_WORKER_HISTORICAL = None
_WORKER_FIXED_HISTORICAL = None
_WORKER_POLICY_GENERATION = -1
_WORKER_ROLLOUT_PRECISION = "float32"
_WORKER_TRACE_MIN_GRU = True

ROLLOUT_PRECISION_FLOAT32 = "float32"
ROLLOUT_PRECISION_INT8_RECURRENT = "int8_recurrent"
ROLLOUT_PRECISION_INT8_RESIDUAL_BASE = "int8_residual_base"
ROLLOUT_PRECISIONS = (
    ROLLOUT_PRECISION_FLOAT32,
    ROLLOUT_PRECISION_INT8_RECURRENT,
    ROLLOUT_PRECISION_INT8_RESIDUAL_BASE,
)


class _MinGRUDecisionTrace(torch.nn.Module):
    """Tensor-only wrapper so TorchScript can erase recurrent Python calls."""

    def __init__(self, core: torch.nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(
        self, tokens: torch.Tensor, recurrent_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, next_state = self.core.forward_eval(
            tokens, (recurrent_state,)
        )
        return logits, values, next_state[0]


class _MinGRUEventTrace(torch.nn.Module):
    """Trace event-state advancement without the unused decoder heads."""

    def __init__(self, core: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = core.encoder
        self.network = core.network

    def forward(
        self, tokens: torch.Tensor, recurrent_state: torch.Tensor
    ) -> torch.Tensor:
        hidden = self.encoder(tokens)
        _, next_state = self.network.forward_eval(
            hidden, (recurrent_state,)
        )
        return next_state[0]


class _TracedMinGRURolloutCore(torch.nn.Module):
    """Shape-polymorphic decision and event graphs for a CPU MinGRU."""

    def __init__(self, core: torch.nn.Module) -> None:
        super().__init__()
        network = core.network
        self.hidden_size = int(network.hidden_size)
        self.num_layers = int(network.num_layers)
        input_size = int(core.encoder.encoder.in_features)
        example_tokens = torch.zeros(1, input_size)
        example_state = torch.zeros(1, self.num_layers, 1, self.hidden_size)
        # Trace with a batch of one but keep all dimensions dynamic. MinGRU's
        # eval graph contains no data-dependent branch; the Python assertion
        # about matching batch dimensions is validated by the caller's state
        # construction and intentionally omitted from the resulting graph.
        with torch.inference_mode(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=torch.jit.TracerWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="`torch.jit.trace.*deprecated.*",
                category=DeprecationWarning,
            )
            self.decision_graph = torch.jit.trace(
                _MinGRUDecisionTrace(core),
                (example_tokens, example_state[0]),
                check_trace=False,
            )
            self.event_graph = torch.jit.trace(
                _MinGRUEventTrace(core),
                (example_tokens, example_state[0]),
                check_trace=False,
            )

    def initial_state(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(
                self.num_layers,
                batch_size,
                self.hidden_size,
                device=device,
            ),
        )

    def forward_eval(
        self,
        tokens: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        logits, values, next_state = self.decision_graph(tokens, state[0])
        return logits, values, (next_state,)

    def forward_events_eval(
        self,
        tokens: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        return (self.event_graph(tokens, state[0]),)


def _trace_worker_min_gru(policy, *, shared_float_weights=False) -> None:
    """Trace a worker-owned MinGRU, keeping its live parameter storage."""
    core = policy.puffer_policy
    if (
        core.__class__.__name__ == "Policy"
        and getattr(core, "network", None).__class__.__name__ == "MinGRU"
    ):
        traced = _TracedMinGRURolloutCore(core)
        if shared_float_weights:
            # Keep the source module and its canonical state_dict intact.
            # The traced methods reference those same live parameters.
            core.forward_eval = traced.forward_eval
            core.forward_events_eval = traced.forward_events_eval
        else:
            policy.puffer_policy = traced
        policy._rollout_device = torch.device("cpu")
        policy._rollout_dtype = torch.float32


def _recurrent_networks(policy) -> tuple[torch.nn.Module, ...]:
    """Return recurrent cores without quantizing encoders or action heads."""
    core = policy.puffer_policy
    if hasattr(core, "network"):
        return (core.network,)
    networks = tuple(
        branch.network
        for name in ("base", "adapter")
        if (branch := getattr(core, name, None)) is not None
        and hasattr(branch, "network")
    )
    if not networks:
        raise ValueError("rollout policy has no supported recurrent network")
    return networks


def _build_rollout_policy(
    policy,
    rollout_precision: str,
    *,
    trace_min_gru: bool = True,
):
    """Build a CPU behavior policy, optionally with weight-only int8 GRUs."""
    if rollout_precision not in ROLLOUT_PRECISIONS:
        raise ValueError(f"unsupported rollout precision: {rollout_precision}")
    if rollout_precision == ROLLOUT_PRECISION_FLOAT32:
        return policy

    quantized = copy.deepcopy(policy).cpu().eval()
    if rollout_precision == ROLLOUT_PRECISION_INT8_RESIDUAL_BASE:
        base = getattr(quantized.puffer_policy, "base", None)
        if base is not None:
            # Residual students preserve their teacher as a complete base
            # policy.  Using that much cheaper actor for behavior collection
            # is deliberately approximate; the full residual learner still
            # evaluates PPO ratios and receives every optimizer update.
            quantized.puffer_policy = base
    # Per-channel dynamic quantization keeps recurrent activations and state in
    # float32 but packs each Linear weight row as int8. This targets the large
    # MinGRU matrix multiplies while leaving feature encoding and policy/value
    # heads in float32. No calibration corpus is required.
    qconfig = {
        torch.nn.Linear: torch.ao.quantization.per_channel_dynamic_qconfig
    }
    for network in _recurrent_networks(quantized):
        # PyTorch 2.13 ships this optimized CPU kernel in core but warns about
        # its future migration to the optional torchao package, which is not a
        # dependency of this pinned training environment.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="torch.ao.quantization is deprecated.*",
                category=DeprecationWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="torch.quantize_per_tensor.*deprecated.*",
                category=UserWarning,
            )
            torch.ao.quantization.quantize_dynamic(
                network,
                qconfig,
                dtype=torch.qint8,
                inplace=True,
            )
    if trace_min_gru:
        _trace_worker_min_gru(quantized)
    for parameter in quantized.parameters():
        parameter.requires_grad_(False)
    # The traced graph exposes duplicated module trees, making a fresh
    # ``next(policy.parameters())`` traversal surprisingly expensive at every
    # vector turn. Rollout policies are immutable CPU float32 actors, so cache
    # this stable execution metadata outside the state dict.
    quantized._rollout_device = torch.device("cpu")
    quantized._rollout_dtype = torch.float32
    return quantized


def _refresh_worker_policy(policy_generation: int) -> None:
    """Repack int8 weights once after each shared float-policy refresh."""
    global _WORKER_POLICY, _WORKER_HISTORICAL, _WORKER_POLICY_GENERATION
    if _WORKER_POLICY_SOURCE is None or _WORKER_FIXED_HISTORICAL is None:
        raise RuntimeError("vector rollout worker was not initialized")
    if policy_generation == _WORKER_POLICY_GENERATION:
        return
    if policy_generation < _WORKER_POLICY_GENERATION:
        raise RuntimeError("rollout policy generation moved backwards")
    _WORKER_POLICY = _build_rollout_policy(
        _WORKER_POLICY_SOURCE,
        _WORKER_ROLLOUT_PRECISION,
        trace_min_gru=_WORKER_TRACE_MIN_GRU,
    )
    _WORKER_HISTORICAL = (
        _WORKER_FIXED_HISTORICAL
        if _WORKER_FIXED_HISTORICAL
        else [_WORKER_POLICY]
    )
    _WORKER_POLICY_GENERATION = policy_generation


def _initialize_worker(
    state_dict: dict,
    model_config: dict,
    champion_path: str,
    historical_paths: tuple[str, ...],
    rollout_precision: str = ROLLOUT_PRECISION_FLOAT32,
    trace_min_gru: bool = True,
) -> None:
    global _WORKER_POLICY, _WORKER_POLICY_SOURCE, _WORKER_CHAMPION
    global _WORKER_HISTORICAL, _WORKER_FIXED_HISTORICAL
    global _WORKER_POLICY_GENERATION, _WORKER_ROLLOUT_PRECISION
    global _WORKER_TRACE_MIN_GRU
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from benchmark_agents import load_ppo_agent
    from temporal_puffer_v4 import (
        build_temporal_policy,
        load_temporal_policy_checkpoint,
    )

    policy = build_temporal_policy(model_config)
    policy.load_state_dict(state_dict, assign=True)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    _WORKER_POLICY_SOURCE = policy
    _WORKER_POLICY = (
        policy if rollout_precision == ROLLOUT_PRECISION_FLOAT32 else None
    )
    _WORKER_POLICY_GENERATION = -1
    _WORKER_ROLLOUT_PRECISION = rollout_precision
    _WORKER_TRACE_MIN_GRU = trace_min_gru
    _WORKER_CHAMPION = load_ppo_agent(Path(champion_path), "cpu")
    _WORKER_FIXED_HISTORICAL = []
    for raw_path in historical_paths:
        historical, _ = load_temporal_policy_checkpoint(
            Path(raw_path), device="cpu"
        )
        historical.eval()
        for parameter in historical.parameters():
            parameter.requires_grad_(False)
        _WORKER_FIXED_HISTORICAL.append(
            _build_rollout_policy(
                historical,
                rollout_precision,
                trace_min_gru=trace_min_gru,
            )
        )
    if rollout_precision == ROLLOUT_PRECISION_FLOAT32 and trace_min_gru:
        # Tracing (without freezing) preserves the parameters installed with
        # assign=True. The current graph sees every in-place shared refresh.
        for worker_policy in (policy, *_WORKER_FIXED_HISTORICAL):
            _trace_worker_min_gru(worker_policy, shared_float_weights=True)
    _WORKER_HISTORICAL = list(_WORKER_FIXED_HISTORICAL)


def _collect_player_count(task: dict):
    if (
        _WORKER_POLICY is None
        or _WORKER_CHAMPION is None
        or _WORKER_HISTORICAL is None
    ):
        raise RuntimeError("vector rollout worker was not initialized")
    from temporal_batched_speed import BatchRolloutStats, collect_league_batch_batched

    started = time.perf_counter()
    stats = BatchRolloutStats()
    with torch.inference_mode():
        episodes, metadata = collect_league_batch_batched(
            _WORKER_POLICY,
            _WORKER_CHAMPION,
            _WORKER_HISTORICAL,
            games=task["games"],
            seed=task["seed"],
            schedule_offset=task["schedule_offset"],
            schedule=task["schedule"],
            player_count=task["player_count"],
            game_indices=task["game_indices"],
            max_turns=task["max_turns"],
            max_public_events=task["max_public_events"],
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
    total_turns = decisions = turn_limit_draws = 0
    for game_index, player_count in task["games"]:
        episodes, turns, winner = _collect_scalar_tree_game(
            _WORKER_POLICY,
            absolute_game_index=game_index,
            deal_seed=task["seed"] + game_index,
            player_count=player_count,
            max_turns=task["max_turns"],
        )
        if task["prepack_training"]:
            from packed_puffer_rollout import pack_puffer_episode

            episodes = [
                pack_puffer_episode(_WORKER_POLICY, episode)
                for episode in episodes
            ]
        rows.append((game_index, episodes))
        total_turns += turns
        if winner is None and turns >= task["max_turns"]:
            turn_limit_draws += 1
        decisions += sum(len(trajectory) for trajectory, _ in episodes)
    return {
        "kind": "tree",
        "rows": rows,
        "metadata": {
            "games": {"bayesian_tree": len(rows)},
            "decisions": {"bayesian_tree": decisions},
            "total_turns": total_turns,
            "trajectories": len(rows),
            "max_turns": task["max_turns"],
            "turn_limit_draws": turn_limit_draws,
            "task_seconds": time.perf_counter() - started,
        },
    }


def _collect_task(task: dict):
    _refresh_worker_policy(int(task["policy_generation"]))
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
    total_turns = trajectories = turn_limit_draws = 0
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
            turn_limit_draws += int(metadata.get("turn_limit_draws", 0))
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
        turn_limit_draws += int(result["stats"].get("turn_limit_draws", 0))
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
        "turn_limit_draws": turn_limit_draws,
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
    historical_paths: Sequence[Path] = (),
    rollout_precision: str = ROLLOUT_PRECISION_FLOAT32,
    trace_min_gru: bool = True,
    max_public_events: int | None = None,
    prepack_training: bool = False,
    max_turns: int = 1000,
):
    """Run one batch through a temporary vector worker pool."""
    with VectorPufferRolloutPool(
        policy,
        workers=workers,
        champion_path=champion_path,
        historical_paths=historical_paths,
        rollout_precision=rollout_precision,
        trace_min_gru=trace_min_gru,
        max_public_events=max_public_events,
    ) as pool:
        return pool.collect(
            policy,
            games=games,
            seed=seed,
            schedule_offset=schedule_offset,
            schedule=schedule,
            player_counts=player_counts,
            prepack_training=prepack_training,
            max_turns=max_turns,
        )


class VectorPufferRolloutPool:
    """Persistent workers reading an atomically refreshed shared policy."""

    def __init__(
        self,
        policy,
        *,
        workers: int,
        champion_path: Path = Path("ppo_actor_100run3.pth"),
        historical_paths: Sequence[Path] = (),
        rollout_precision: str = ROLLOUT_PRECISION_FLOAT32,
        trace_min_gru: bool = True,
        max_public_events: int | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be positive")
        if rollout_precision not in ROLLOUT_PRECISIONS:
            raise ValueError(
                f"unsupported rollout precision: {rollout_precision}"
            )
        if max_public_events is not None and max_public_events < 1:
            raise ValueError(
                "max_public_events must be positive when provided"
            )
        self.workers = workers
        self.closed = False
        self.rollout_precision = rollout_precision
        self.trace_min_gru = trace_min_gru
        self.max_public_events = max_public_events
        self.policy_generation = -1
        self.historical_paths = tuple(
            Path(path).resolve() for path in historical_paths
        )
        for path in self.historical_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.shared_state = {
            name: value.detach().cpu().clone().share_memory_()
            for name, value in policy.state_dict().items()
        }
        context = multiprocessing.get_context("spawn")
        from temporal_puffer_v4 import policy_model_config

        self.pool = context.Pool(
            processes=workers,
            initializer=_initialize_worker,
            initargs=(
                self.shared_state,
                policy_model_config(policy),
                str(champion_path.resolve()),
                tuple(str(path) for path in self.historical_paths),
                rollout_precision,
                trace_min_gru,
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
        max_turns: int = 1000,
    ):
        """Refresh shared weights, run cohorts, and restore game ordering."""
        self._refresh_policy(policy)
        self.policy_generation += 1
        tasks, schedule, player_counts = _build_tasks(
            games=games,
            seed=seed,
            schedule_offset=schedule_offset,
            schedule=schedule,
            player_counts=player_counts,
            workers=self.workers,
            prepack_training=prepack_training,
            max_turns=max_turns,
        )
        for task in tasks:
            task["policy_generation"] = self.policy_generation
            task["max_public_events"] = self.max_public_events
        results = self.pool.map(_collect_task, tasks, chunksize=1)
        episodes, metadata = _merge_results(
            results,
            games=games,
            schedule_offset=schedule_offset,
            schedule=schedule,
            player_count_schedule=player_counts,
        )
        metadata["max_turns"] = max_turns
        metadata["rollout_precision"] = self.rollout_precision
        metadata["trace_min_gru"] = self.trace_min_gru
        metadata["max_public_events"] = self.max_public_events
        return episodes, metadata

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
    max_turns: int = 1000,
):
    if games < 1 or workers < 1 or max_turns < 1:
        raise ValueError("games, workers, and max_turns must be positive")
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
    # One task per player count caps useful parallelism at seven, even when a
    # large recurrent policy makes each cohort expensive enough to split. Each
    # game's deal, opponent, seat, and sampling RNG are keyed by its absolute
    # game index, so sharding a cohort does not change training semantics.
    # Reserve capacity for scalar tree tasks, then split the heaviest remaining
    # cohorts until the requested worker pool can stay busy. Two-deck games get
    # a modest premium because seven/eight-player cohorts cost more per game.
    vector_games = sum(len(indices) for indices in grouped_indices.values())
    vector_task_target = min(
        vector_games,
        max(len(grouped_indices), workers - len(tree_games)),
    )
    shard_counts = {count: 1 for count in grouped_indices}
    while sum(shard_counts.values()) < vector_task_target:
        candidates = [
            count
            for count, indices in grouped_indices.items()
            if shard_counts[count] < len(indices)
        ]
        if not candidates:
            break
        count = max(
            candidates,
            key=lambda value: (
                len(grouped_indices[value])
                * value
                * (3 if deck_count_for_players(value) > 1 else 2)
                / shard_counts[value],
                value,
            ),
        )
        shard_counts[count] += 1

    tasks = []
    for count, indices in grouped_indices.items():
        shards = shard_counts[count]
        base, remainder = divmod(len(indices), shards)
        cursor = 0
        for shard in range(shards):
            shard_size = base + (shard < remainder)
            shard_indices = tuple(indices[cursor : cursor + shard_size])
            cursor += shard_size
            tasks.append(
                {
                    "kind": "vector",
                    "games": games,
                    "seed": seed,
                    "schedule_offset": schedule_offset,
                    "schedule": schedule,
                    "player_count": count,
                    "game_indices": shard_indices,
                    "prepack_training": prepack_training,
                    "max_turns": max_turns,
                }
            )
    if tree_games:
        tree_chunk_size = 1
        tasks.extend(
            {
                "kind": "tree",
                "seed": seed,
                "games": tuple(tree_games[start : start + tree_chunk_size]),
                "prepack_training": prepack_training,
                "max_turns": max_turns,
            }
            for start in range(0, len(tree_games), tree_chunk_size)
        )
    return tasks, schedule, player_counts
