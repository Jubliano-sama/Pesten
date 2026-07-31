"""Batched temporal-policy league rollouts with canonical Pesten semantics.

The expensive recurrent policy and frozen PPO champion are evaluated in
batches.  Bayesian-tree games deliberately remain on the canonical scalar
``Game`` implementation because their search depends on a live game object.
Episodes are merged back into scalar league order before training.

Only information already admitted by :mod:`temporal_observation` crosses the
policy boundary.  In particular, draws become public ``DRAW_COUNT`` events;
drawn card identities and draw-pile state are never included.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import random
import time
from typing import Hashable, Sequence

import numpy as np
import torch

import game
from batched_pesten import (
    BatchedPesten,
    PHASE_SUIT,
    POLICY_RL,
    POLICY_SMART,
    SUIT_NAMES,
)
from hardcodedAgent import HardcodedStrategyAgent
from ppo2 import build_action_masks
from temporal_observation import (
    DecisionInput,
    EventKind,
    PublicEventToken,
    build_legal_actions,
    build_private_observation,
    relative_index,
)
from temporal_puffer_agent import TrajectoryStep, temporal_agent_factory


LEAGUE_SCHEDULE = (
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "historical_temporal",
    "historical_temporal",
    "ppo_champion",
    "ppo_champion",
    "bayesian_tree",
    "smart_rule",
)
VALID_OPPONENT_TYPES = frozenset(LEAGUE_SCHEDULE)

ROLE_CURRENT = "current"
ROLE_HISTORICAL = "historical"
ROLE_PPO = "ppo_champion"
ROLE_SMART = "smart_rule"


@dataclass(slots=True)
class SafeEventCursor:
    """Per-seat public-event cursor used by the recurrent safe adapter."""

    last_hand_sizes: tuple[int, ...] = ()
    public_cursor: int = 0
    decision_count: int = 0


@dataclass(slots=True)
class TemporalSeatRuntime:
    """Policy-local state for one temporal seat in one vector game."""

    game_index: int
    absolute_game_index: int
    seat: int
    policy: torch.nn.Module
    generator: torch.Generator
    state: tuple[torch.Tensor, ...]
    cursor: SafeEventCursor
    trajectory: list[TrajectoryStep] = field(default_factory=list)
    safe_history: list[DecisionInput] = field(default_factory=list)


@dataclass(slots=True)
class BatchRolloutStats:
    """Optional profiler counters; not part of the scalar league result."""

    vector_games: int = 0
    tree_games: int = 0
    temporal_decisions: int = 0
    temporal_forward_calls: int = 0
    champion_decisions: int = 0
    champion_forward_calls: int = 0
    engine_bytes: int = 0
    vector_seconds: float = 0.0
    tree_seconds: float = 0.0
    safe_observation_event_seconds: float = 0.0
    temporal_feature_encoding_seconds: float = 0.0
    temporal_recurrent_inference_transfer_seconds: float = 0.0
    temporal_sampling_state_scatter_seconds: float = 0.0
    champion_inference_transfer_seconds: float = 0.0
    rules_stepping_seconds: float = 0.0


def league_schedule() -> tuple[str, ...]:
    """Return the fixed 40/20/20/10/10 opponent mixture."""

    return LEAGUE_SCHEDULE


def _hand_sizes(engine: BatchedPesten, game_index: int) -> tuple[int, ...]:
    return tuple(int(value) for value in engine.hand_sizes[game_index])


def _own_cards(
    engine: BatchedPesten,
    game_index: int,
    seat: int,
) -> tuple[int, ...]:
    size = int(engine.hand_sizes[game_index, seat])
    return tuple(
        int(number) for number in engine.hands[game_index, seat, :size]
    )


def build_safe_decision(
    engine: BatchedPesten,
    game_index: int,
    seat: int,
    cursor: SafeEventCursor,
) -> DecisionInput:
    """Build the same safe decision/event stream as ``TemporalPestenAgent``."""

    player_count = engine.num_players
    direction = int(engine.direction[game_index])
    current_sizes = _hand_sizes(engine, game_index)
    # The scalar adapter first establishes its episode marker at the player's
    # first decision, after any earlier seats may already have acted.
    if cursor.decision_count == 0:
        cursor.last_hand_sizes = current_sizes
    events: list[PublicEventToken] = []
    if cursor.public_cursor == 0 and cursor.decision_count == 0:
        events.append(PublicEventToken(EventKind.EPISODE_START))

    predicted_sizes = list(cursor.last_hand_sizes)
    history = engine.public_events[game_index]
    for item in history[cursor.public_cursor :]:
        action = item[0]
        actor = int(item[1])
        actor_relative = relative_index(
            actor, seat, player_count, direction
        )
        if action == "play":
            number = int(item[2])
            events.append(
                PublicEventToken(
                    EventKind.PLAY,
                    actor_relative=actor_relative,
                    card_id=number,
                    suit_id=int(number // 12 if number < 48 else 4),
                    count=1,
                )
            )
            if 0 <= actor < len(predicted_sizes):
                predicted_sizes[actor] -= 1
        elif action == "choose_sort":
            events.append(
                PublicEventToken(
                    EventKind.CHOOSE_SUIT,
                    actor_relative=actor_relative,
                    suit_id=int(item[2]),
                )
            )

    for actor, (predicted, current) in enumerate(
        zip(predicted_sizes, current_sizes)
    ):
        draw_count = current - predicted
        if draw_count > 0:
            events.append(
                PublicEventToken(
                    EventKind.DRAW_COUNT,
                    actor_relative=relative_index(
                        actor, seat, player_count, direction
                    ),
                    count=draw_count,
                )
            )

    cursor.public_cursor = len(history)
    cursor.last_hand_sizes = current_sizes
    cursor.decision_count += 1

    own_cards = _own_cards(engine, game_index, seat)
    suit_mode = int(engine.phase[game_index]) == PHASE_SUIT
    current_suit = SUIT_NAMES[int(engine.current_sort[game_index])]
    current_rank = int(engine.current_rank[game_index])
    penalty_amount = int(engine.penalty_amount[game_index])
    observation = build_private_observation(
        own_cards=own_cards,
        current_suit=current_suit,
        current_rank=current_rank,
        public_hand_sizes=current_sizes,
        observer_index=seat,
        current_player_index=int(engine.current_player[game_index]),
        direction=direction,
        penalty_amount=penalty_amount,
        public_sorts_played=tuple(
            int(value) for value in engine.sorts_played[game_index]
        ),
        change_suit_mode=suit_mode,
    )
    legal = build_legal_actions(
        own_cards=own_cards,
        current_suit=current_suit,
        current_rank=current_rank,
        penalty_amount=penalty_amount,
        change_suit_mode=suit_mode,
    )
    return DecisionInput(observation, legal, tuple(events))


def _stack_states(
    runtimes: Sequence[TemporalSeatRuntime],
) -> tuple[torch.Tensor, ...]:
    if not runtimes:
        raise ValueError("cannot stack an empty recurrent-state batch")
    part_count = len(runtimes[0].state)
    if any(len(runtime.state) != part_count for runtime in runtimes):
        raise ValueError("recurrent state structures differ within a policy")
    return tuple(
        torch.cat([runtime.state[part] for runtime in runtimes], dim=1)
        for part in range(part_count)
    )


def _scatter_states(
    runtimes: Sequence[TemporalSeatRuntime],
    states: tuple[torch.Tensor, ...],
) -> None:
    for batch_index, runtime in enumerate(runtimes):
        runtime.state = tuple(
            part[:, batch_index : batch_index + 1].detach()
            for part in states
        )


def _batch_temporal_actions(
    policy: torch.nn.Module,
    entries: Sequence[tuple[TemporalSeatRuntime, DecisionInput]],
    stats: BatchRolloutStats | None,
) -> list[int]:
    """Advance variable-length event streams, then sample one action per seat."""

    if not entries:
        return []
    parameter = next(policy.parameters())
    device, dtype = parameter.device, parameter.dtype
    runtimes = [entry[0] for entry in entries]
    decisions = [entry[1] for entry in entries]
    feature_started = time.perf_counter()
    event_rows = [policy._event_features(decision) for decision in decisions]
    if stats is not None:
        stats.temporal_feature_encoding_seconds += (
            time.perf_counter() - feature_started
        )

    inference_started = time.perf_counter()
    with torch.inference_mode():
        max_events = max((len(rows) for rows in event_rows), default=0)
        for event_index in range(max_events):
            selected = [
                index
                for index, rows in enumerate(event_rows)
                if event_index < len(rows)
            ]
            selected_runtimes = [runtimes[index] for index in selected]
            tokens = torch.tensor(
                [event_rows[index][event_index] for index in selected],
                device=device,
                dtype=dtype,
            )
            _, _, next_states = policy.puffer_policy.forward_eval(
                tokens, _stack_states(selected_runtimes)
            )
            _scatter_states(selected_runtimes, next_states)
            if stats is not None:
                stats.temporal_forward_calls += 1

        tokens = torch.tensor(
            [policy._decision_features(decision) for decision in decisions],
            device=device,
            dtype=dtype,
        )
        logits, values, next_states = policy.puffer_policy.forward_eval(
            tokens, _stack_states(runtimes)
        )
        legal = torch.tensor(
            [decision.legal_actions for decision in decisions],
            device=device,
            dtype=torch.bool,
        )
        masked_logits = logits.masked_fill(
            ~legal, torch.finfo(logits.dtype).min
        )
        probabilities = torch.softmax(masked_logits, dim=-1).cpu()
        if stats is not None:
            stats.temporal_recurrent_inference_transfer_seconds += (
                time.perf_counter() - inference_started
            )
        scatter_started = time.perf_counter()
        actions = [
            int(
                torch.multinomial(
                    probabilities[index],
                    1,
                    generator=runtime.generator,
                ).item()
            )
            for index, runtime in enumerate(runtimes)
        ]
        log_probabilities = torch.log_softmax(masked_logits, dim=-1)
        _scatter_states(runtimes, next_states)

    for index, (runtime, decision) in enumerate(entries):
        action = actions[index]
        runtime.safe_history.append(decision)
        runtime.trajectory.append(
            TrajectoryStep(
                decision=decision,
                action=action,
                old_log_prob=float(
                    log_probabilities[index, action].item()
                ),
                value=float(values[index].reshape(-1)[0].item()),
            )
        )
    if stats is not None:
        stats.temporal_decisions += len(entries)
        stats.temporal_forward_calls += 1
        stats.temporal_sampling_state_scatter_seconds += (
            time.perf_counter() - scatter_started
        )
    return actions


def _champion_actor_and_device(champion_factory):
    actor = getattr(champion_factory, "shared_actor", None)
    device = getattr(champion_factory, "shared_device", None)
    if actor is None and isinstance(champion_factory, torch.nn.Module):
        actor = champion_factory
    if actor is None:
        raise ValueError(
            "champion_factory must expose shared_actor or be a torch module"
        )
    if device is None:
        device = next(actor.parameters()).device
    return actor, torch.device(device)


def _batch_champion_actions(
    engine: BatchedPesten,
    game_indices: Sequence[int],
    champion_factory,
    stats: BatchRolloutStats | None,
) -> list[int]:
    """Run the frozen legacy champion as one masked greedy batch."""

    if not game_indices:
        return []
    started = time.perf_counter()
    actor, device = _champion_actor_and_device(champion_factory)
    observations = torch.from_numpy(
        np.stack(
            [
                engine.observation(
                    game_index,
                    suit_mode=(
                        int(engine.phase[game_index]) == PHASE_SUIT
                    ),
                )
                for game_index in game_indices
            ]
        )
    ).to(device=device)
    with torch.inference_mode():
        logits = actor(observations)
        legal = build_action_masks(observations)
        actions = (
            logits.masked_fill(
                ~legal, torch.finfo(logits.dtype).min
            )
            .argmax(dim=-1)
            .cpu()
            .tolist()
        )
    if stats is not None:
        stats.champion_decisions += len(game_indices)
        stats.champion_forward_calls += 1
        stats.champion_inference_transfer_seconds += (
            time.perf_counter() - started
        )
    return [int(action) for action in actions]


def _champion_has_card_decision(
    engine: BatchedPesten,
    game_index: int,
) -> bool:
    """Match legacy PPO's compatible-card check, including duplicate cards."""

    observation = engine.observation(game_index)
    return bool(np.any(observation[50:100] >= 1))


def _tree_factory(game_instance):
    return HardcodedStrategyAgent(
        game_instance,
        belief_samples=8,
        search_depth=3,
        beam_width=3,
    )


def _collect_scalar_tree_game(
    policy: torch.nn.Module,
    *,
    absolute_game_index: int,
    deal_seed: int,
    max_turns: int,
) -> tuple[list[tuple[Sequence[TrajectoryStep], bool]], int]:
    np.random.seed(deal_seed)
    random.seed(deal_seed)
    current_factory = temporal_agent_factory(
        policy, deterministic=False, seed=deal_seed * 2
    )
    current_seat = absolute_game_index % 2
    factories = (
        [current_factory, _tree_factory]
        if current_seat == 0
        else [_tree_factory, current_factory]
    )
    match = game.Game(factories)
    turns, winner = match.auto_simulate(max_turns=max_turns)
    trajectory = tuple(match.players[current_seat].trajectory)
    return [(trajectory, winner == current_seat)], turns


def _make_vector_layout(
    *,
    games: int,
    seed: int,
    schedule_offset: int,
    schedule: Sequence[str],
    historical: Sequence[torch.nn.Module],
) -> tuple[
    list[int],
    np.ndarray,
    np.ndarray,
    dict[tuple[int, int], tuple[str, torch.nn.Module | None, int]],
]:
    vector_absolute_indices = [
        index
        for index in range(games)
        if schedule[
            (schedule_offset + index) % len(schedule)
        ]
        != "bayesian_tree"
    ]
    policy_kinds = np.full((len(vector_absolute_indices), 2), POLICY_RL, np.int8)
    actor_keys = np.empty((len(vector_absolute_indices), 2), dtype=object)
    roles: dict[
        tuple[int, int], tuple[str, torch.nn.Module | None, int]
    ] = {}

    for vector_index, absolute_index in enumerate(vector_absolute_indices):
        opponent_type = schedule[
            (schedule_offset + absolute_index) % len(schedule)
        ]
        deal_seed = seed + absolute_index
        if opponent_type == "current_selfplay":
            seat_roles = (
                (ROLE_CURRENT, None, deal_seed * 2),
                (ROLE_CURRENT, None, deal_seed * 2 + 1),
            )
        else:
            current_seat = absolute_index % 2
            if opponent_type == "historical_temporal":
                if not historical:
                    raise ValueError(
                        "historical snapshots are required by the league mix"
                    )
                opponent_policy = historical[absolute_index % len(historical)]
                opponent_role = (
                    ROLE_HISTORICAL,
                    opponent_policy,
                    deal_seed + 99,
                )
            elif opponent_type == "ppo_champion":
                opponent_role = (ROLE_PPO, None, 0)
            elif opponent_type == "smart_rule":
                opponent_role = (ROLE_SMART, None, 0)
            else:
                raise AssertionError(f"unexpected vector role {opponent_type}")
            current_role = (ROLE_CURRENT, None, deal_seed * 2)
            seat_roles = (
                (current_role, opponent_role)
                if current_seat == 0
                else (opponent_role, current_role)
            )

        for seat, role in enumerate(seat_roles):
            role_name, _, _ = role
            roles[(vector_index, seat)] = role
            actor_keys[vector_index, seat] = (
                role_name,
                id(role[1]) if role[1] is not None else 0,
            )
            if role_name == ROLE_SMART:
                policy_kinds[vector_index, seat] = POLICY_SMART

    seeds = np.asarray(
        [seed + index for index in vector_absolute_indices],
        dtype=np.uint32,
    )
    return vector_absolute_indices, policy_kinds, actor_keys, roles


def collect_league_batch_batched(
    policy: torch.nn.Module,
    champion_factory,
    historical: Sequence[torch.nn.Module],
    *,
    games: int,
    seed: int,
    schedule_offset: int = 0,
    schedule: Sequence[str] = LEAGUE_SCHEDULE,
    max_turns: int = 1000,
    stats: BatchRolloutStats | None = None,
) -> tuple[list[tuple[Sequence[TrajectoryStep], bool]], dict]:
    """Collect an explicit league schedule with batched neural decisions.

    The default retains the frozen 40/20/20/10/10 equivalence benchmark.
    Training recipes should pass ``schedule`` explicitly; omitted opponent
    types are never constructed or evaluated.  The return value intentionally
    matches ``temporal_league.collect_league_batch``.
    """

    if games < 0:
        raise ValueError("games must be non-negative")
    schedule = tuple(schedule)
    if not schedule:
        raise ValueError("schedule must contain at least one opponent type")
    unknown_opponents = set(schedule) - VALID_OPPONENT_TYPES
    if unknown_opponents:
        raise ValueError(
            "unknown opponent types: "
            + ", ".join(sorted(unknown_opponents))
        )
    if stats is None:
        stats = BatchRolloutStats()
    (
        vector_absolute_indices,
        policy_kinds,
        actor_keys,
        roles,
    ) = _make_vector_layout(
        games=games,
        seed=seed,
        schedule_offset=schedule_offset,
        schedule=schedule,
        historical=historical,
    )
    absolute_to_vector = {
        absolute: vector
        for vector, absolute in enumerate(vector_absolute_indices)
    }
    episodes_by_game: dict[
        int, list[tuple[Sequence[TrajectoryStep], bool]]
    ] = {}
    contributions = Counter()
    game_counts = Counter()
    total_turns = 0

    stats.vector_games = len(vector_absolute_indices)
    stats.tree_games = games - len(vector_absolute_indices)
    if vector_absolute_indices:
        engine = BatchedPesten(
            policy_kinds,
            actor_keys,
            np.asarray(
                [seed + index for index in vector_absolute_indices],
                dtype=np.uint32,
            ),
            max_turns=max_turns,
            smart_agents_have_game_reference=True,
        )
        stats.engine_bytes = engine.allocated_bytes
        runtimes: dict[tuple[int, int], TemporalSeatRuntime] = {}
        for (vector_index, seat), role in roles.items():
            role_name, opponent_policy, agent_seed = role
            if role_name not in (ROLE_CURRENT, ROLE_HISTORICAL):
                continue
            seat_policy = (
                policy if role_name == ROLE_CURRENT else opponent_policy
            )
            assert seat_policy is not None
            runtimes[(vector_index, seat)] = TemporalSeatRuntime(
                game_index=vector_index,
                absolute_game_index=vector_absolute_indices[vector_index],
                seat=seat,
                policy=seat_policy,
                generator=torch.Generator(device="cpu").manual_seed(
                    agent_seed
                ),
                state=seat_policy.initial_state(),
                cursor=SafeEventCursor(),
            )

        vector_started = time.perf_counter()
        while True:
            active = engine.active_games()
            if len(active) == 0:
                break
            actions: dict[int, int] = {}
            temporal_groups: dict[
                int,
                tuple[
                    torch.nn.Module,
                    list[tuple[TemporalSeatRuntime, DecisionInput]],
                ],
            ] = {}
            champion_games: list[int] = []

            for vector_value in active:
                vector_index = int(vector_value)
                seat = int(engine.current_player[vector_index])
                role_name, _, _ = roles[(vector_index, seat)]
                if role_name in (ROLE_CURRENT, ROLE_HISTORICAL):
                    runtime = runtimes[(vector_index, seat)]
                    safe_started = time.perf_counter()
                    decision = build_safe_decision(
                        engine, vector_index, seat, runtime.cursor
                    )
                    stats.safe_observation_event_seconds += (
                        time.perf_counter() - safe_started
                    )
                    group = temporal_groups.setdefault(
                        id(runtime.policy), (runtime.policy, [])
                    )
                    group[1].append((runtime, decision))
                elif role_name == ROLE_PPO:
                    if (
                        int(engine.phase[vector_index]) != PHASE_SUIT
                        and not _champion_has_card_decision(
                            engine, vector_index
                        )
                    ):
                        actions[vector_index] = 54
                    else:
                        champion_games.append(vector_index)
                else:
                    rules_started = time.perf_counter()
                    actions[vector_index] = engine.non_neural_action(
                        vector_index
                    )
                    stats.rules_stepping_seconds += (
                        time.perf_counter() - rules_started
                    )

            for _, (seat_policy, entries) in temporal_groups.items():
                chosen = _batch_temporal_actions(
                    seat_policy, entries, stats
                )
                for (runtime, _), action in zip(entries, chosen):
                    actions[runtime.game_index] = action

            if champion_games:
                champion_actions = _batch_champion_actions(
                    engine, champion_games, champion_factory, stats
                )
                actions.update(zip(champion_games, champion_actions))
            if len(actions) != len(active):
                raise RuntimeError(
                    "each active game must produce exactly one action"
                )
            rules_started = time.perf_counter()
            for vector_value in active:
                vector_index = int(vector_value)
                engine.apply_action(vector_index, actions[vector_index])
            stats.rules_stepping_seconds += (
                time.perf_counter() - rules_started
            )

        stats.vector_seconds = time.perf_counter() - vector_started
        total_turns += int(engine.turn_counts.sum())
        for vector_index, absolute_index in enumerate(
            vector_absolute_indices
        ):
            opponent_type = schedule[
                (schedule_offset + absolute_index) % len(schedule)
            ]
            game_counts[opponent_type] += 1
            winner = int(engine.winners[vector_index])
            current_seats = [
                seat
                for seat in range(2)
                if roles[(vector_index, seat)][0] == ROLE_CURRENT
            ]
            game_episodes = []
            for seat in current_seats:
                trajectory = tuple(runtimes[(vector_index, seat)].trajectory)
                game_episodes.append((trajectory, winner == seat))
                contributions[opponent_type] += len(trajectory)
            episodes_by_game[absolute_index] = game_episodes

    tree_started = time.perf_counter()
    for absolute_index in range(games):
        opponent_type = schedule[
            (schedule_offset + absolute_index) % len(schedule)
        ]
        if opponent_type != "bayesian_tree":
            continue
        tree_episodes, turns = _collect_scalar_tree_game(
            policy,
            absolute_game_index=absolute_index,
            deal_seed=seed + absolute_index,
            max_turns=max_turns,
        )
        episodes_by_game[absolute_index] = tree_episodes
        total_turns += turns
        game_counts[opponent_type] += 1
        contributions[opponent_type] += sum(
            len(trajectory) for trajectory, _ in tree_episodes
        )
    stats.tree_seconds = time.perf_counter() - tree_started

    episodes = [
        episode
        for absolute_index in range(games)
        for episode in episodes_by_game.get(absolute_index, ())
    ]
    return episodes, {
        "games": dict(game_counts),
        "decisions": dict(contributions),
        "total_turns": total_turns,
        "trajectories": len(episodes),
    }
