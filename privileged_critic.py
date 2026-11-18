"""Causal privileged-state value models for temporal Pesten PPO.

The actor-facing contract in :mod:`temporal_observation` deliberately excludes
hidden cards.  This module lives on the environment/training side of that
boundary.  It captures only state that already exists immediately *before* an
action, then trains a separate value estimator.  No value-model tensor is ever
fed back into the actor.

The representation is deliberately structured rather than one enormous flat
vector:

* player hands retain counts and engine order in relative-seat coordinates;
* the draw and discard piles retain their current order;
* cumulative public plays retain both counts and recent order;
* the acting policy contributes probabilities and its pre-action memory; and
* a causal temporal backbone may remember arbitrarily many earlier decisions.

Random-number-generator state and deal seeds are metadata only.  They are not
model inputs: exposing them would let a critic predict future random samples
rather than estimate their conditional expectation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import math
from pathlib import Path
import random
from typing import Hashable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from batched_pesten import (
    PHASE_CARD,
    PHASE_DRAW_RETRY,
    PHASE_SUIT,
    POLICY_RANDOM,
    POLICY_RL,
    POLICY_SMART,
    POLICY_SMARTER,
    BatchedPesten,
)
from temporal_batched_speed import SafeEventCursor, build_safe_decision
from temporal_observation import (
    ACTION_COUNT as TEMPORAL_ACTION_COUNT,
    CARD_TYPES,
    MAX_PLAYERS,
    OBSERVATION_SIZE,
)


# Two physical decks are enough for every currently supported 2-13 player
# game.  ``BatchedPesten`` uses the same 54-card canonical deck definition.
MAX_PILE_CARDS = 108
MAX_RECENT_PLAYS = 64

# Selected teacher-training default. A future terminal reward retains about
# 90.46% of its value after twenty of the controlled seat's own decisions.
DEFAULT_GAMMA = 0.995

HAND_ENCODERS = ("counts_moments", "token_gru", "token_bigru")
CARD_TOKENIZATIONS = ("exact", "rank_suit")

ROLE_PADDING = 0
ROLE_CURRENT = 1
ROLE_HISTORICAL = 2
ROLE_PPO = 3
ROLE_SMART = 4
ROLE_SMARTER = 5
ROLE_RANDOM = 6
ROLE_TREE = 7
ROLE_UNKNOWN = 8
ROLE_COUNT = 9

ROLE_IDS = {
    "current": ROLE_CURRENT,
    "historical": ROLE_HISTORICAL,
    "ppo_champion": ROLE_PPO,
    "smart_rule": ROLE_SMART,
    "smarter_rule": ROLE_SMARTER,
    "audited_uniform_random": ROLE_RANDOM,
    "canonical_random": ROLE_RANDOM,
    "random": ROLE_RANDOM,
    "random_rule": ROLE_RANDOM,
    "bayesian_tree": ROLE_TREE,
}
ROLE_NAMES = {
    ROLE_CURRENT: "current_selfplay",
    ROLE_HISTORICAL: "historical_temporal",
    ROLE_PPO: "ppo_champion",
    ROLE_SMART: "smart_rule",
    ROLE_SMARTER: "smarter_rule",
    ROLE_RANDOM: "random",
    ROLE_TREE: "bayesian_tree",
    ROLE_UNKNOWN: "unknown",
}


SCALAR_FEATURE_NAMES = (
    "phase_card",
    "phase_draw_retry",
    "phase_suit",
    "current_suit_harten",
    "current_suit_ruiten",
    "current_suit_schoppen",
    "current_suit_klaveren",
    "current_suit_special",
    "current_rank_0",
    "current_rank_1",
    "current_rank_2",
    "current_rank_3",
    "current_rank_4",
    "current_rank_5",
    "current_rank_6",
    "current_rank_7",
    "current_rank_8",
    "current_rank_9",
    "current_rank_10",
    "current_rank_11",
    "current_rank_12",
    "current_rank_13",
    "direction",
    "penalty_amount",
    "draw_fraction",
    "discard_fraction",
    "turn_fraction",
    "cards_played_fraction",
    "player_count_fraction",
    "absolute_seat_fraction",
    "deck_count_fraction",
    "turn_started_with_one_card",
    "own_hand_fraction",
    "closest_opponent_hand_fraction",
    "total_hand_fraction",
)
SCALAR_FEATURE_COUNT = len(SCALAR_FEATURE_NAMES)


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


@lru_cache(maxsize=256)
def _relative_seats(
    observer: int,
    player_count: int,
    direction: int,
) -> np.ndarray:
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    return _readonly((
        observer + direction * np.arange(player_count, dtype=np.int64)
    ) % player_count)


def flatten_recurrent_state(state: object) -> np.ndarray:
    """Flatten a policy state without retaining an autograd graph."""

    parts = state if isinstance(state, tuple) else (state,)
    flattened = []
    for part in parts:
        if not isinstance(part, torch.Tensor):
            raise TypeError("recurrent state parts must be tensors")
        flattened.append(part.detach().float().cpu().numpy())
    if len(flattened) == 1:
        return _readonly(np.ascontiguousarray(flattened[0], dtype=np.float16).reshape(-1))
    return _readonly(np.concatenate([part.reshape(-1) for part in flattened]).astype(np.float16))


@dataclass(slots=True)
class _EncodedSeatMemory:
    source: object
    flat: np.ndarray
    quantized: np.ndarray
    scale: float


def _encode_seat_memory(state, key, cache) -> _EncodedSeatMemory:
    if cache is not None:
        cached = cache.get(key)
        if cached is not None and cached.source is state:
            return cached
    flat = flatten_recurrent_state(state)
    values = flat.astype(np.float32)
    maximum = float(np.max(np.abs(values), initial=0.0))
    scale = maximum / 127.0 if maximum > 0 else 1.0
    quantized = np.clip(np.rint(values / scale), -127, 127).astype(np.int8)
    encoded = _EncodedSeatMemory(state, flat, quantized, scale)
    if cache is not None:
        cache[key] = encoded
    return encoded


@dataclass(frozen=True, slots=True)
class PrivilegedSnapshot:
    """One compact, pre-action critic input.

    Every array is read-only.  The snapshot intentionally has no action,
    reward, winner, future event, seed, or RNG-state field.
    """

    safe_observation: np.ndarray
    hands: np.ndarray
    hand_order: np.ndarray
    hand_lengths: np.ndarray
    player_actor_memory: np.ndarray
    player_actor_memory_scale: np.ndarray
    player_actor_memory_mask: np.ndarray
    player_mask: np.ndarray
    role_ids: np.ndarray
    draw_counts: np.ndarray
    draw_order: np.ndarray
    draw_length: int
    discard_counts: np.ndarray
    discard_order: np.ndarray
    discard_length: int
    played_counts: np.ndarray
    recent_plays: np.ndarray
    recent_play_length: int
    scalars: np.ndarray
    actor_probabilities: np.ndarray
    actor_memory: np.ndarray
    public_value: float


def _padded_top_first(
    cards: np.ndarray,
    count: int,
    *,
    limit: int,
) -> tuple[np.ndarray, int]:
    count = min(int(count), len(cards), limit)
    result = np.full(limit, -1, dtype=np.int8)
    if count:
        result[:count] = np.asarray(cards[:count], dtype=np.int8)[::-1]
    return _readonly(result), count


def _card_counts(cards: np.ndarray, count: int, dtype) -> np.ndarray:
    valid = cards[: int(count)]
    if valid.dtype == np.int8:
        # Engine piles use signed bytes. Negative padding lands above every
        # valid card ID when viewed unsigned and is excluded by the slice.
        valid = valid.view(np.uint8)
    else:
        valid = np.asarray(valid, dtype=np.int64)
        valid = valid[(valid >= 0) & (valid < CARD_TYPES)]
    counts = np.bincount(valid, minlength=CARD_TYPES)[:CARD_TYPES]
    return _readonly(counts.astype(dtype, copy=False))


def capture_privileged_snapshot(
    engine: BatchedPesten,
    game_index: int,
    observer: int,
    *,
    safe_observation: Sequence[float],
    absolute_roles: Sequence[str],
    actor_probabilities: Sequence[float],
    actor_memory: object,
    public_value: float,
    all_actor_memories: Sequence[object | None] | None = None,
    rollout_memory_cache: dict[tuple[int, int], _EncodedSeatMemory] | None = None,
) -> PrivilegedSnapshot:
    """Capture the exact environment state before the current action.

    ``actor_probabilities`` and ``actor_memory`` must already include the
    current decision observation, but must be captured before sampling or
    applying its action.  They therefore describe the behavior policy without
    becoming an action-dependent baseline.

    The vector collector installs a fresh state tuple whenever a seat acts.
    It may pass a per-collection cache to reuse other seats' immutable state
    encodings until their next decision. Standalone callers need no cache.
    """

    game_index = int(game_index)
    observer = int(observer)
    player_count = int(engine.num_players)
    if int(engine.current_player[game_index]) != observer:
        raise ValueError("privileged snapshot observer must be the acting seat")
    if len(absolute_roles) != player_count:
        raise ValueError("one role is required for every player")
    safe = np.asarray(safe_observation, dtype=np.float32)
    if safe.shape != (OBSERVATION_SIZE,):
        raise ValueError(
            f"safe observation must have shape ({OBSERVATION_SIZE},)"
        )
    probabilities = np.asarray(actor_probabilities, dtype=np.float32)
    if probabilities.shape != (TEMPORAL_ACTION_COUNT,):
        raise ValueError(
            "actor probabilities must have shape "
            f"({TEMPORAL_ACTION_COUNT},)"
        )
    # A nonfinite entry makes the sum nonfinite. This scalar comparison
    # preserves np.isclose(sum, 1, atol=2e-4) including its default rtol.
    if not abs(float(probabilities.sum()) - 1.0) <= 2e-4 + 1e-5:
        raise ValueError("actor probabilities must be finite and sum to one")

    direction = int(engine.direction[game_index])
    relative = _relative_seats(observer, player_count, direction)
    hands = np.zeros((MAX_PLAYERS, CARD_TYPES), dtype=np.uint8)
    hands[:player_count] = engine.hand_counts[game_index, relative]
    hand_order = np.full(
        (MAX_PLAYERS, MAX_PILE_CARDS), -1, dtype=np.int8
    )
    hand_lengths = np.zeros(MAX_PLAYERS, dtype=np.uint8)
    for relative_index, absolute_index in enumerate(relative):
        length = int(engine.hand_sizes[game_index, absolute_index])
        hand_lengths[relative_index] = length
        hand_order[relative_index, :length] = engine.hands[
            game_index, absolute_index, :length
        ]
    player_mask = np.zeros(MAX_PLAYERS, dtype=np.bool_)
    player_mask[:player_count] = True
    roles = np.full(MAX_PLAYERS, ROLE_PADDING, dtype=np.uint8)
    roles[:player_count] = [
        ROLE_IDS.get(str(absolute_roles[index]), ROLE_UNKNOWN)
        for index in relative
    ]

    own_memory = _encode_seat_memory(actor_memory, (game_index, observer), rollout_memory_cache)
    flat_actor_memory = own_memory.flat
    if all_actor_memories is None:
        all_actor_memories = tuple(
            actor_memory if seat == observer else None
            for seat in range(player_count)
        )
    if len(all_actor_memories) != player_count:
        raise ValueError("one actor-memory slot is required for every player")
    player_memories = np.zeros(
        (MAX_PLAYERS, len(flat_actor_memory)), dtype=np.int8
    )
    player_memory_scales = np.zeros(MAX_PLAYERS, dtype=np.float16)
    player_memory_mask = np.zeros(MAX_PLAYERS, dtype=np.bool_)
    for relative_index, absolute_index in enumerate(relative):
        state = all_actor_memories[int(absolute_index)]
        if state is None:
            continue
        encoded = own_memory if state is actor_memory else _encode_seat_memory(
            state, (game_index, int(absolute_index)), rollout_memory_cache
        )
        if encoded.flat.shape != flat_actor_memory.shape:
            # A historical opponent may legitimately use another recurrent
            # width.  The fixed-width current critic cannot encode that state;
            # leave its explicit mask false rather than corrupting dimensions.
            continue
        player_memories[relative_index] = encoded.quantized
        player_memory_scales[relative_index] = encoded.scale
        player_memory_mask[relative_index] = True

    draw_length = int(engine.draw_counts[game_index])
    draw_cards = engine.draw_piles[game_index]
    draw_order, encoded_draw_length = _padded_top_first(
        draw_cards, draw_length, limit=MAX_PILE_CARDS
    )
    discard_length = int(engine.discard_counts[game_index])
    discard_cards = engine.discards[game_index]
    discard_order, encoded_discard_length = _padded_top_first(
        discard_cards, discard_length, limit=MAX_PILE_CARDS
    )

    played = [
        int(item[2])
        for item in engine.public_events[game_index]
        if item and item[0] == "play"
    ]
    played_array = np.asarray(played, dtype=np.int16)
    played_counts = _readonly(
        np.bincount(played_array, minlength=CARD_TYPES).astype(
            np.uint16, copy=False
        )
        if len(played_array)
        else np.zeros(CARD_TYPES, dtype=np.uint16)
    )
    recent_length = min(len(played), MAX_RECENT_PLAYS)
    recent = np.full(MAX_RECENT_PLAYS, -1, dtype=np.int8)
    if recent_length:
        recent[:recent_length] = np.asarray(
            played[-recent_length:][::-1], dtype=np.int8
        )

    phase = int(engine.phase[game_index])
    current_suit = int(engine.current_sort[game_index])
    current_rank = int(engine.current_rank[game_index])
    scalar_values = np.zeros(SCALAR_FEATURE_COUNT, dtype=np.float32)
    scalar_values[phase] = 1.0
    scalar_values[3 + current_suit] = 1.0
    scalar_values[8 + current_rank] = 1.0
    scalar_values[22] = float(direction)
    scalar_values[23] = float(engine.penalty_amount[game_index]) / 10.0
    scalar_values[24] = draw_length / MAX_PILE_CARDS
    scalar_values[25] = discard_length / MAX_PILE_CARDS
    scalar_values[26] = min(int(engine.turn_counts[game_index]), 500) / 500.0
    scalar_values[27] = min(int(engine.cards_played[game_index]), 500) / 500.0
    scalar_values[28] = player_count / MAX_PLAYERS
    scalar_values[29] = observer / max(1, MAX_PLAYERS - 1)
    scalar_values[30] = float(engine.deck_count) / 2.0
    scalar_values[31] = float(
        engine.turn_started_with_one_card[game_index]
    )
    sizes = hand_lengths[:player_count]
    scalar_values[32] = float(sizes[0]) / 54.0
    scalar_values[33] = float(sizes[1:].min()) / 54.0
    scalar_values[34] = float(sizes.sum()) / MAX_PILE_CARDS

    return PrivilegedSnapshot(
        safe_observation=_readonly(safe.astype(np.float16)),
        hands=_readonly(hands),
        hand_order=_readonly(hand_order),
        hand_lengths=_readonly(hand_lengths),
        player_actor_memory=_readonly(player_memories),
        player_actor_memory_scale=_readonly(player_memory_scales),
        player_actor_memory_mask=_readonly(player_memory_mask),
        player_mask=_readonly(player_mask),
        role_ids=_readonly(roles),
        draw_counts=_card_counts(draw_cards, draw_length, np.uint8),
        draw_order=draw_order,
        draw_length=encoded_draw_length,
        discard_counts=_card_counts(
            discard_cards, discard_length, np.uint8
        ),
        discard_order=discard_order,
        discard_length=encoded_discard_length,
        played_counts=played_counts,
        recent_plays=_readonly(recent),
        recent_play_length=recent_length,
        scalars=_readonly(scalar_values.astype(np.float16)),
        actor_probabilities=_readonly(probabilities.astype(np.float16)),
        actor_memory=flat_actor_memory,
        public_value=float(public_value),
    )


def discounted_returns(
    decision_count: int,
    *,
    won: bool | None,
    gamma: float = DEFAULT_GAMMA,
    decision_reward: float = -0.01,
    draw_reward: float = -0.75,
) -> np.ndarray:
    """Match the temporal PPO terminal and per-decision reward contract."""

    if decision_count < 1:
        return np.empty(0, dtype=np.float32)
    rewards = np.full(decision_count, decision_reward, dtype=np.float32)
    rewards[-1] += (
        draw_reward if won is None else (1.0 if won else -1.0)
    )
    result = np.empty_like(rewards)
    running = 0.0
    for index in range(decision_count - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def discounted_return_targets(
    remaining_decisions: torch.Tensor,
    won: torch.Tensor,
    *,
    gamma: float = DEFAULT_GAMMA,
) -> torch.Tensor:
    """Vectorize the exact terminal-plus-step return for stored trajectories."""

    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    remaining = remaining_decisions.to(dtype=torch.float32)
    valid = remaining > 0
    safe_remaining = remaining.clamp_min(1)
    if gamma == 1.0:
        step_cost = -0.01 * safe_remaining
    else:
        gamma_tensor = remaining.new_tensor(gamma)
        step_cost = -0.01 * (
            1.0 - torch.pow(gamma_tensor, safe_remaining)
        ) / (1.0 - gamma_tensor)
    terminal_sign = torch.where(
        won.bool(),
        torch.ones_like(remaining),
        -torch.ones_like(remaining),
    )
    terminal = terminal_sign * torch.pow(
        remaining.new_tensor(gamma), safe_remaining - 1.0
    )
    return torch.where(valid, step_cost + terminal, torch.zeros_like(remaining))


def episode_from_snapshots(
    snapshots: Sequence[PrivilegedSnapshot],
    *,
    won: bool | None,
    deal_seed: int,
    seat: int,
    player_count: int,
    gamma: float = DEFAULT_GAMMA,
    decision_reward: float = -0.01,
    draw_reward: float = -0.75,
) -> dict[str, np.ndarray | int | bool | None]:
    """Stack one player's causal snapshots and attach post-game targets."""

    if not snapshots:
        raise ValueError("cannot build an empty critic episode")
    fields = (
        "safe_observation",
        "hands",
        "hand_order",
        "hand_lengths",
        "player_actor_memory",
        "player_actor_memory_scale",
        "player_actor_memory_mask",
        "player_mask",
        "role_ids",
        "draw_counts",
        "draw_order",
        "discard_counts",
        "discard_order",
        "played_counts",
        "recent_plays",
        "scalars",
        "actor_probabilities",
        "actor_memory",
    )
    result: dict[str, np.ndarray | int | bool | None] = {
        name: np.stack([getattr(item, name) for item in snapshots])
        for name in fields
    }
    result.update(
        {
            "draw_length": np.asarray(
                [item.draw_length for item in snapshots], dtype=np.uint8
            ),
            "discard_length": np.asarray(
                [item.discard_length for item in snapshots], dtype=np.uint8
            ),
            "recent_play_length": np.asarray(
                [item.recent_play_length for item in snapshots],
                dtype=np.uint8,
            ),
            "public_value": np.asarray(
                [item.public_value for item in snapshots], dtype=np.float32
            ),
            "returns": discounted_returns(
                len(snapshots),
                won=won,
                gamma=gamma,
                decision_reward=decision_reward,
                draw_reward=draw_reward,
            ),
            "win_target": np.full(len(snapshots), won is True, dtype=np.bool_),
            "remaining_decisions": np.arange(
                len(snapshots), 0, -1, dtype=np.uint16
            ),
            "deal_seed": int(deal_seed),
            "seat": int(seat),
            "player_count": int(player_count),
            "won": won,
        }
    )
    return result


def absolute_roles_for_game(
    opponent_type: str,
    *,
    absolute_game_index: int,
    player_count: int,
) -> tuple[str, ...]:
    """Reconstruct the vector league's role layout without private RNG data."""

    if opponent_type == "current_selfplay":
        return ("current",) * player_count
    opponent_role = {
        "historical_temporal": "historical",
        "ppo_champion": "ppo_champion",
        "smart_rule": "smart_rule",
        "smarter_rule": "smarter_rule",
        "random": "audited_uniform_random",
        "canonical_random": "canonical_random",
        "random_rule": "canonical_random",
    }.get(opponent_type)
    if opponent_role is None:
        raise ValueError(
            f"privileged trace replay does not support {opponent_type!r}"
        )
    current_seat = absolute_game_index % player_count
    return tuple(
        "current" if seat == current_seat else opponent_role
        for seat in range(player_count)
    )


def replay_privileged_trace(
    policy: nn.Module,
    trace: Mapping[str, object],
    *,
    opponent_type: str,
    player_count: int,
    gamma: float = DEFAULT_GAMMA,
) -> list[dict[str, object]]:
    """Replay a recorded action stream and capture causal critic snapshots.

    The trace provides actions only.  Hidden state is reconstructed from the
    deal seed through the canonical vector engine, then sampled actions are
    applied after each snapshot.  This makes the pre-action boundary directly
    auditable and avoids modifying the actor rollout contract.
    """

    deal_seed = int(trace["deal_seed"])
    absolute_game_index = int(trace["absolute_game_index"])
    roles = absolute_roles_for_game(
        opponent_type,
        absolute_game_index=absolute_game_index,
        player_count=player_count,
    )
    role_policy_kind = {
        "current": POLICY_RL,
        "historical": POLICY_RL,
        "ppo_champion": POLICY_RL,
        "smart_rule": POLICY_SMART,
        "smarter_rule": POLICY_SMARTER,
        "audited_uniform_random": POLICY_RL,
        "canonical_random": POLICY_RANDOM,
    }
    kinds = np.asarray(
        [[role_policy_kind[role] for role in roles]], dtype=np.int8
    )
    keys = np.full((1, player_count), None, dtype=object)
    engine = BatchedPesten(
        kinds,
        keys,
        np.asarray([deal_seed], dtype=np.uint32),
        max_turns=max(1, int(trace["turn_count"])),
        smart_agents_have_game_reference=True,
        record_public_events=True,
    )

    initial = trace.get("initial_deal")
    if isinstance(initial, Mapping):
        actual_hands = [
            engine.hands[0, seat, : int(engine.hand_sizes[0, seat])]
            .astype(int)
            .tolist()
            for seat in range(player_count)
        ]
        expected_hands = [list(row) for row in initial["hands"]]  # type: ignore[index]
        opening_count = int(engine.discard_counts[0])
        actual_opening = int(engine.discards[0, opening_count - 1])
        if actual_hands != expected_hands or actual_opening != int(
            initial["opening_card"]  # type: ignore[index]
        ):
            raise RuntimeError("trace replay initial deal does not match")

    current_seats = [
        seat for seat, role in enumerate(roles) if role == "current"
    ]
    neural_seats = [
        seat
        for seat, role in enumerate(roles)
        if role in ("current", "historical")
    ]
    states = {seat: policy.initial_state() for seat in neural_seats}
    cursors = {seat: SafeEventCursor() for seat in neural_seats}
    snapshots: dict[int, list[PrivilegedSnapshot]] = {
        seat: [] for seat in current_seats
    }

    with torch.inference_mode():
        for action_index, raw_action in enumerate(trace["actions"]):  # type: ignore[index]
            if not isinstance(raw_action, Mapping):
                raise TypeError("trace actions must be mappings")
            seat = int(engine.current_player[0])
            phase = int(engine.phase[0])
            if seat != int(raw_action["seat"]) or phase != int(
                raw_action["phase"]
            ):
                raise RuntimeError(
                    "trace replay diverged before action "
                    f"{action_index}: state seat/phase=({seat}, {phase}), "
                    f"trace=({raw_action['seat']}, {raw_action['phase']})"
                )
            if roles[seat] in ("current", "historical"):
                decision = build_safe_decision(
                    engine, 0, seat, cursors[seat]
                )
                expected_legal = raw_action.get("legal_actions")
                if expected_legal is not None and tuple(expected_legal) != (
                    decision.legal_actions
                ):
                    raise RuntimeError(
                        f"trace replay legal mask diverged at action {action_index}"
                    )
                result = policy.step(decision, states[seat])
                probabilities = torch.softmax(
                    result.logits.float(), dim=-1
                ).cpu().numpy()
                next_state = tuple(
                    part.detach() for part in result.state
                )
                if roles[seat] == "current":
                    turn_memories = [
                        next_state
                        if player == seat
                        else states.get(player)
                        for player in range(player_count)
                    ]
                    snapshots[seat].append(
                        capture_privileged_snapshot(
                            engine,
                            0,
                            seat,
                            safe_observation=decision.observation.values,
                            absolute_roles=roles,
                            actor_probabilities=probabilities,
                            actor_memory=next_state,
                            public_value=float(result.value.item()),
                            all_actor_memories=turn_memories,
                        )
                    )
                states[seat] = next_state
            engine.apply_action(0, int(raw_action["action"]))

    expected_winner = int(trace["winner"])
    if int(engine.winners[0]) != expected_winner:
        raise RuntimeError(
            "trace replay winner diverged: "
            f"engine={int(engine.winners[0])}, trace={expected_winner}"
        )
    if int(engine.turn_counts[0]) != int(trace["turn_count"]):
        raise RuntimeError(
            "trace replay turn count diverged: "
            f"engine={int(engine.turn_counts[0])}, trace={trace['turn_count']}"
        )
    expected_events = tuple(tuple(item) for item in trace["public_events"])  # type: ignore[index]
    actual_events = tuple(tuple(item) for item in engine.public_events[0])
    if actual_events != expected_events:
        raise RuntimeError("trace replay public history diverged")

    return [
        episode_from_snapshots(
            snapshots[seat],
            won=expected_winner == seat,
            deal_seed=deal_seed,
            seat=seat,
            player_count=player_count,
            gamma=gamma,
        )
        for seat in current_seats
        if snapshots[seat]
    ]


DATA_FIELDS = (
    "safe_observation",
    "hands",
    "hand_order",
    "hand_lengths",
    "player_actor_memory",
    "player_actor_memory_scale",
    "player_actor_memory_mask",
    "player_mask",
    "role_ids",
    "draw_counts",
    "draw_order",
    "draw_length",
    "discard_counts",
    "discard_order",
    "discard_length",
    "played_counts",
    "recent_plays",
    "recent_play_length",
    "scalars",
    "actor_probabilities",
    "actor_memory",
    "public_value",
    "returns",
    "win_target",
    "remaining_decisions",
)

SNAPSHOT_MODEL_FIELDS = DATA_FIELDS[: DATA_FIELDS.index("returns")]


def snapshots_to_model_batch(
    snapshots: Sequence[PrivilegedSnapshot],
    *,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Tensorize simultaneous current-turn snapshots for ``model.step``."""

    if not snapshots:
        raise ValueError("at least one privileged snapshot is required")
    arrays = {}
    for name in SNAPSHOT_MODEL_FIELDS:
        values = [getattr(snapshot, name) for snapshot in snapshots]
        arrays[name] = torch.as_tensor(np.stack(values), device=device)
    return arrays


def pack_episodes(
    episodes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Pack variable trajectories into compact concatenated CPU tensors."""

    if not episodes:
        raise ValueError("at least one episode is required")
    offsets = [0]
    for episode in episodes:
        offsets.append(offsets[-1] + len(episode["returns"]))  # type: ignore[arg-type]
    packed = {
        name: torch.from_numpy(
            np.concatenate(
                [np.asarray(episode[name]) for episode in episodes], axis=0
            )
        )
        for name in DATA_FIELDS
    }
    packed.update(
        {
            "format": "pesten-privileged-critic-dataset-v1",
            "offsets": torch.tensor(offsets, dtype=torch.int64),
            "deal_seed": torch.tensor(
                [int(episode["deal_seed"]) for episode in episodes],
                dtype=torch.int64,
            ),
            "seat": torch.tensor(
                [int(episode["seat"]) for episode in episodes],
                dtype=torch.uint8,
            ),
            "player_count": torch.tensor(
                [int(episode["player_count"]) for episode in episodes],
                dtype=torch.uint8,
            ),
            "won": torch.tensor(
                [bool(episode["won"]) for episode in episodes],
                dtype=torch.bool,
            ),
            "actor_memory_size": int(
                np.asarray(episodes[0]["actor_memory"]).shape[-1]
            ),
            "scalar_feature_names": SCALAR_FEATURE_NAMES,
        }
    )
    return packed


@dataclass(frozen=True)
class CriticFeatureConfig:
    """Information groups admitted to a critic candidate."""

    safe_observation: bool = True
    policy_context: bool = True
    public_value: bool = True
    opponent_hands: bool = True
    hand_order: bool = True
    opponent_memories: bool = True
    player_roles: bool = True
    draw_composition: bool = True
    draw_order: bool = True
    draw_top_n: int = MAX_PILE_CARDS
    discard_composition: bool = True
    discard_order: bool = True
    discard_top_n: int = MAX_PILE_CARDS
    played_history: bool = True
    played_top_n: int = MAX_RECENT_PLAYS
    engine_controls: bool = True

    def __post_init__(self) -> None:
        limits = (
            ("draw_top_n", self.draw_top_n, MAX_PILE_CARDS),
            ("discard_top_n", self.discard_top_n, MAX_PILE_CARDS),
            ("played_top_n", self.played_top_n, MAX_RECENT_PLAYS),
        )
        for name, value, maximum in limits:
            if not 0 <= value <= maximum:
                raise ValueError(f"{name} must be in [0, {maximum}]")


def public_feature_config() -> CriticFeatureConfig:
    """A strong actor-safe recurrent baseline with identical model capacity."""

    return CriticFeatureConfig(
        opponent_hands=False,
        hand_order=False,
        opponent_memories=False,
        player_roles=False,
        draw_composition=False,
        draw_order=False,
        draw_top_n=0,
        discard_composition=False,
        discard_order=False,
        discard_top_n=0,
        played_history=False,
        played_top_n=0,
        engine_controls=False,
    )


class FactorizedCardEmbedding(nn.Module):
    """Losslessly embed a card occurrence as its paired suit and rank.

    Ordinary ids 0-47 are the Cartesian product of four suits and twelve
    ranks.  Id 48 is the special rank-13 card and id 49 is the special rank-0
    card.  Unlike suit/rank *marginal counts*, the paired token therefore
    retains the exact identity of every card occurrence.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        if width < 2:
            raise ValueError("factorized card embedding width must be at least 2")
        suit_width = width // 2
        rank_width = width - suit_width
        # Four ordinary suits, one special suit, and padding.
        self.suit_embedding = nn.Embedding(6, suit_width, padding_idx=5)
        # Ranks 0-13 plus padding.
        self.rank_embedding = nn.Embedding(15, rank_width, padding_idx=14)

        card_ids = torch.arange(CARD_TYPES + 1, dtype=torch.long)
        suits = torch.where(card_ids < 48, card_ids // 12, 4)
        ranks = torch.where(
            card_ids < 48,
            card_ids % 12 + 1,
            torch.where(card_ids == 48, 13, 0),
        )
        suits[-1] = 5
        ranks[-1] = 14
        self.register_buffer("card_suits", suits, persistent=False)
        self.register_buffer("card_ranks", ranks, persistent=False)

    def forward(self, card_ids: torch.Tensor) -> torch.Tensor:
        safe_ids = card_ids.long().clamp(min=0, max=CARD_TYPES)
        return torch.cat(
            (
                self.suit_embedding(self.card_suits[safe_ids]),
                self.rank_embedding(self.card_ranks[safe_ids]),
            ),
            dim=-1,
        )


def _card_embedding(
    tokenization: str,
    width: int,
) -> nn.Module:
    if tokenization == "exact":
        # Keep this as a direct Embedding so v1 exact-token checkpoint keys
        # remain byte-for-byte compatible.
        return nn.Embedding(CARD_TYPES + 1, width, padding_idx=CARD_TYPES)
    if tokenization == "rank_suit":
        return FactorizedCardEmbedding(width)
    raise ValueError(f"unknown card tokenization: {tokenization}")


class CardSequenceEncoder(nn.Module):
    """Encode an ordered current pile without looking past its valid length."""

    def __init__(self, width: int, *, tokenization: str = "exact") -> None:
        super().__init__()
        embedding = max(16, width // 2)
        self.card_embedding = _card_embedding(tokenization, embedding)
        self.depth_embedding = nn.Embedding(MAX_PILE_CARDS, embedding)
        self.gru = nn.GRU(embedding, width, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        card_ids: torch.Tensor,
        lengths: torch.Tensor,
        *,
        top_n: int,
    ) -> torch.Tensor:
        leading = card_ids.shape[:-1]
        sequence_length = min(card_ids.shape[-1], max(1, int(top_n)))
        flat_ids = card_ids[..., :sequence_length].reshape(
            -1, sequence_length
        ).long()
        flat_lengths = lengths.reshape(-1).long().clamp(
            min=0, max=min(top_n, sequence_length)
        )
        positions = torch.arange(
            sequence_length, device=flat_ids.device
        ).unsqueeze(0)
        valid = positions < flat_lengths.unsqueeze(1)
        # Store piles top-first for auditability, but run the recurrence from
        # deeper cards toward the top.  The final state therefore has the next
        # draw/discard at the shortest recurrent distance.
        source_depth = (
            flat_lengths.unsqueeze(1) - 1 - positions
        ).clamp(min=0)
        ordered_ids = flat_ids.gather(1, source_depth)
        safe_ids = torch.where(
            valid,
            ordered_ids.clamp(min=0, max=CARD_TYPES - 1),
            torch.full_like(flat_ids, CARD_TYPES),
        )
        embedded = self.card_embedding(safe_ids)
        embedded = embedded + self.depth_embedding(
            source_depth
        ) * valid.unsqueeze(-1)
        outputs, _ = self.gru(embedded)
        gather_index = (flat_lengths - 1).clamp(min=0)
        encoded = outputs[
            torch.arange(len(outputs), device=outputs.device), gather_index
        ]
        encoded = torch.where(
            (flat_lengths > 0).unsqueeze(1),
            encoded,
            torch.zeros_like(encoded),
        )
        return self.norm(encoded).reshape(*leading, -1)


class RecentPlayEncoder(nn.Module):
    def __init__(self, width: int, *, tokenization: str = "exact") -> None:
        super().__init__()
        embedding = max(16, width // 2)
        self.card_embedding = _card_embedding(tokenization, embedding)
        self.depth_embedding = nn.Embedding(MAX_RECENT_PLAYS, embedding)
        self.gru = nn.GRU(embedding, width, batch_first=True)
        self.norm = nn.LayerNorm(width)

    def forward(
        self,
        card_ids: torch.Tensor,
        lengths: torch.Tensor,
        *,
        top_n: int,
    ) -> torch.Tensor:
        leading = card_ids.shape[:-1]
        sequence_length = min(card_ids.shape[-1], max(1, int(top_n)))
        flat_ids = card_ids[..., :sequence_length].reshape(
            -1, sequence_length
        ).long()
        flat_lengths = lengths.reshape(-1).long().clamp(
            min=0, max=min(top_n, sequence_length)
        )
        positions = torch.arange(
            sequence_length, device=flat_ids.device
        ).unsqueeze(0)
        valid = positions < flat_lengths.unsqueeze(1)
        source_depth = (
            flat_lengths.unsqueeze(1) - 1 - positions
        ).clamp(min=0)
        ordered_ids = flat_ids.gather(1, source_depth)
        safe_ids = torch.where(
            valid,
            ordered_ids.clamp(min=0, max=CARD_TYPES - 1),
            torch.full_like(flat_ids, CARD_TYPES),
        )
        embedded = self.card_embedding(safe_ids)
        embedded = embedded + self.depth_embedding(
            source_depth
        ) * valid.unsqueeze(-1)
        outputs, _ = self.gru(embedded)
        gather_index = (flat_lengths - 1).clamp(min=0)
        encoded = outputs[
            torch.arange(len(outputs), device=outputs.device), gather_index
        ]
        encoded = torch.where(
            (flat_lengths > 0).unsqueeze(1),
            encoded,
            torch.zeros_like(encoded),
        )
        return self.norm(encoded).reshape(*leading, -1)


class HandOrderEncoder(nn.Module):
    """Compactly encode full hand order without a GRU per player and turn.

    Learned positional moments consume every card occurrence.  Exact first
    positions separately retain the engine's deterministic tie-break order.
    Duplicate occurrences remain represented by the moment sums.
    """

    def __init__(self, width: int, *, moments: int = 8) -> None:
        super().__init__()
        self.moments = int(moments)
        self.position = nn.Embedding(MAX_PILE_CARDS, self.moments)
        self.project = nn.Sequential(
            nn.Linear(CARD_TYPES * (self.moments + 1), width),
            nn.GELU(),
            nn.LayerNorm(width),
        )

    def forward(
        self,
        card_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        leading = card_ids.shape[:-1]
        sequence_length = card_ids.shape[-1]
        flat_ids = card_ids.reshape(-1, sequence_length).long()
        flat_lengths = lengths.reshape(-1).long().clamp(
            min=0, max=sequence_length
        )
        positions = torch.arange(sequence_length, device=flat_ids.device)
        valid = positions.unsqueeze(0) < flat_lengths.unsqueeze(1)
        safe_ids = flat_ids.clamp(min=0, max=CARD_TYPES - 1)

        positional = self.position(positions)
        sources = (
            positional.unsqueeze(0)
            * valid.unsqueeze(-1).to(positional.dtype)
        )
        moments = positional.new_zeros(
            len(flat_ids), CARD_TYPES, self.moments
        )
        moments.scatter_add_(
            1,
            safe_ids.unsqueeze(-1).expand(-1, -1, self.moments),
            sources,
        )
        moments = moments / 4.0

        normalized_positions = positions.to(positional.dtype) / max(
            1, sequence_length
        )
        first_sources = torch.where(
            valid,
            normalized_positions.unsqueeze(0),
            torch.ones_like(flat_ids, dtype=positional.dtype),
        )
        first = positional.new_ones(len(flat_ids), CARD_TYPES)
        first.scatter_reduce_(
            1,
            safe_ids,
            first_sources,
            reduce="amin",
            include_self=True,
        )
        present = first < 1.0
        first_priority = torch.where(present, 1.0 - first, 0.0)
        encoded = torch.cat(
            (moments.reshape(len(flat_ids), -1), first_priority), dim=-1
        )
        return self.project(encoded).reshape(*leading, -1)


class HandSequenceGRUEncoder(nn.Module):
    """Encode each exact hand as a sequence of card-occurrence tokens.

    ``token_bigru`` uses two explicit recurrent passes so padded cards never
    contaminate the reverse direction.  Both directions inspect only the
    current hand; neither can see a later turn.
    """

    def __init__(
        self,
        width: int,
        *,
        tokenization: str,
        bidirectional: bool,
    ) -> None:
        super().__init__()
        embedding = max(16, width // 2)
        self.width = int(width)
        self.bidirectional = bool(bidirectional)
        self.card_embedding = _card_embedding(tokenization, embedding)
        self.position_embedding = nn.Embedding(MAX_PILE_CARDS, embedding)
        direction_width = width // 2 if bidirectional else width
        if bidirectional and width % 2:
            raise ValueError("bidirectional hand GRU width must be even")
        self.forward_gru = nn.GRU(
            embedding, direction_width, batch_first=True
        )
        self.reverse_gru = (
            nn.GRU(embedding, direction_width, batch_first=True)
            if bidirectional
            else None
        )
        self.norm = nn.LayerNorm(width)

    def _encode_direction(
        self,
        card_ids: torch.Tensor,
        lengths: torch.Tensor,
        gru: nn.GRU,
    ) -> torch.Tensor:
        sequence_length = card_ids.shape[-1]
        positions = torch.arange(sequence_length, device=card_ids.device)
        valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
        safe_ids = torch.where(
            valid,
            card_ids.clamp(min=0, max=CARD_TYPES - 1),
            torch.full_like(card_ids, CARD_TYPES),
        )
        embedded = self.card_embedding(safe_ids)
        embedded = embedded + self.position_embedding(positions).unsqueeze(
            0
        ) * valid.unsqueeze(-1)
        outputs, _ = gru(embedded)
        gather_index = (lengths - 1).clamp(min=0)
        encoded = outputs[
            torch.arange(len(outputs), device=outputs.device), gather_index
        ]
        return torch.where(
            (lengths > 0).unsqueeze(1), encoded, torch.zeros_like(encoded)
        )

    def forward(
        self,
        card_ids: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        leading = card_ids.shape[:-1]
        full_length = card_ids.shape[-1]
        flat_lengths = lengths.reshape(-1).long().clamp(
            min=0, max=full_length
        )
        # Unlike pile depth, the maximum current hand length is data-dependent.
        # Trim the globally padded 108 slots while retaining every valid card.
        used_length = max(1, int(flat_lengths.max().detach().cpu().item()))
        flat_ids = card_ids[..., :used_length].reshape(
            -1, used_length
        ).long()
        forward = self._encode_direction(
            flat_ids, flat_lengths, self.forward_gru
        )
        if self.reverse_gru is None:
            encoded = forward
        else:
            positions = torch.arange(used_length, device=flat_ids.device)
            source = (
                flat_lengths.unsqueeze(1) - 1 - positions.unsqueeze(0)
            ).clamp(min=0)
            reverse_ids = flat_ids.gather(1, source)
            reverse = self._encode_direction(
                reverse_ids, flat_lengths, self.reverse_gru
            )
            encoded = torch.cat((forward, reverse), dim=-1)
        return self.norm(encoded).reshape(*leading, self.width)


class PlayerStateEncoder(nn.Module):
    """Shared per-player encoder plus attention over relative seats."""

    def __init__(
        self,
        width: int,
        actor_memory_size: int,
        *,
        heads: int = 4,
        dropout: float = 0.05,
        hand_encoder: str = "counts_moments",
        card_tokenization: str = "exact",
    ) -> None:
        super().__init__()
        if hand_encoder not in HAND_ENCODERS:
            raise ValueError(f"unknown hand encoder: {hand_encoder}")
        self.width = width
        self.hand_encoder = hand_encoder
        self.hand = nn.Sequential(
            nn.Linear(CARD_TYPES, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.hand_sequence = (
            HandOrderEncoder(width)
            if hand_encoder == "counts_moments"
            else HandSequenceGRUEncoder(
                width,
                tokenization=card_tokenization,
                bidirectional=hand_encoder == "token_bigru",
            )
        )
        self.memory = nn.Sequential(
            nn.Linear(actor_memory_size, width),
            nn.GELU(),
            nn.LayerNorm(width),
        )
        self.role = nn.Embedding(ROLE_COUNT, width)
        self.position = nn.Embedding(MAX_PLAYERS, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(
        self,
        hands: torch.Tensor,
        hand_order: torch.Tensor,
        hand_lengths: torch.Tensor,
        player_actor_memory: torch.Tensor,
        player_actor_memory_scale: torch.Tensor,
        player_actor_memory_mask: torch.Tensor,
        player_mask: torch.Tensor,
        role_ids: torch.Tensor,
        *,
        opponent_hands: bool,
        use_hand_order: bool,
        opponent_memories: bool,
        player_roles: bool,
    ) -> torch.Tensor:
        leading = hands.shape[:-2]
        flat_hands = hands.reshape(-1, MAX_PLAYERS, CARD_TYPES).float()
        flat_mask = player_mask.reshape(-1, MAX_PLAYERS).bool()
        if not opponent_hands:
            own_only = torch.zeros_like(flat_hands)
            own_only[:, 0] = flat_hands[:, 0]
            flat_hands = own_only
        count_tokens = self.hand(flat_hands / 4.0)
        tokens = count_tokens
        if use_hand_order:
            ordered = self.hand_sequence(
                hand_order,
                hand_lengths,
            ).reshape(-1, MAX_PLAYERS, self.width)
            if not opponent_hands:
                own_ordered = torch.zeros_like(ordered)
                own_ordered[:, 0] = ordered[:, 0]
                ordered = own_ordered
            # Token GRUs replace the one-hot/count vector.  The legacy path
            # keeps its additive positional moments for checkpoint parity.
            tokens = (
                count_tokens + ordered
                if self.hand_encoder == "counts_moments"
                else ordered
            )
        if opponent_memories:
            memories = (
                player_actor_memory.float()
                * player_actor_memory_scale.float().unsqueeze(-1)
            ).reshape(-1, MAX_PLAYERS, self.memory[0].in_features)
            memory_mask = player_actor_memory_mask.reshape(
                -1, MAX_PLAYERS
            ).clone()
            # Relative seat zero is the acting player and is already encoded
            # losslessly by the policy-context group.  This group isolates
            # memories of neural opponents.
            memory_mask[:, 0] = False
            memory_mask = memory_mask.unsqueeze(-1)
            memory_tokens = self.memory(memories)
            tokens = tokens + memory_tokens * memory_mask
        positions = torch.arange(MAX_PLAYERS, device=hands.device)
        tokens = tokens + self.position(positions).unsqueeze(0)
        if player_roles:
            tokens = tokens + self.role(
                role_ids.reshape(-1, MAX_PLAYERS).long()
            )
        tokens = self.attention(tokens, src_key_padding_mask=~flat_mask)
        weights = flat_mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return self.output_norm(pooled).reshape(*leading, self.width)


class PrivilegedSnapshotEncoder(nn.Module):
    """Encode independently maskable turn-time information groups."""

    def __init__(
        self,
        *,
        actor_memory_size: int,
        output_width: int,
        group_width: int = 96,
        dropout: float = 0.05,
        hand_encoder: str = "counts_moments",
        card_tokenization: str = "exact",
    ) -> None:
        super().__init__()
        self.actor_memory_size = int(actor_memory_size)
        self.output_width = int(output_width)
        self.group_width = int(group_width)
        self.hand_encoder = hand_encoder
        self.card_tokenization = card_tokenization
        self.safe = self._mlp(OBSERVATION_SIZE, group_width)
        self.policy = self._mlp(
            TEMPORAL_ACTION_COUNT + actor_memory_size + 1, group_width
        )
        self.players = PlayerStateEncoder(
            group_width,
            actor_memory_size,
            dropout=dropout,
            hand_encoder=hand_encoder,
            card_tokenization=card_tokenization,
        )
        self.draw_composition = self._mlp(CARD_TYPES, group_width)
        self.draw_order = CardSequenceEncoder(
            group_width, tokenization=card_tokenization
        )
        self.discard_composition = self._mlp(CARD_TYPES, group_width)
        self.discard_order = CardSequenceEncoder(
            group_width, tokenization=card_tokenization
        )
        self.played_composition = self._mlp(CARD_TYPES, group_width)
        self.played_order = RecentPlayEncoder(
            group_width, tokenization=card_tokenization
        )
        self.controls = self._mlp(SCALAR_FEATURE_COUNT, group_width)
        self.project = nn.Sequential(
            nn.Linear(group_width * 10, output_width),
            nn.GELU(),
            nn.LayerNorm(output_width),
        )

    @staticmethod
    def _mlp(inputs: int, outputs: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(inputs, outputs),
            nn.GELU(),
            nn.LayerNorm(outputs),
        )

    @staticmethod
    def _admit(
        value: torch.Tensor,
        enabled: bool,
    ) -> torch.Tensor:
        return value if enabled else torch.zeros_like(value)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        features: CriticFeatureConfig,
    ) -> torch.Tensor:
        safe = self.safe(batch["safe_observation"].float())
        safe = self._admit(safe, features.safe_observation)
        zero_group = torch.zeros_like(safe)
        policy_input = torch.cat(
            (
                batch["actor_probabilities"].float(),
                batch["actor_memory"].float(),
                (
                    batch["public_value"].float().unsqueeze(-1)
                    if features.public_value
                    else torch.zeros_like(
                        batch["public_value"].float().unsqueeze(-1)
                    )
                ),
            ),
            dim=-1,
        )
        policy = self._admit(
            self.policy(policy_input), features.policy_context
        )
        players = self.players(
            batch["hands"],
            batch["hand_order"],
            batch["hand_lengths"],
            batch["player_actor_memory"],
            batch["player_actor_memory_scale"],
            batch["player_actor_memory_mask"],
            batch["player_mask"],
            batch["role_ids"],
            opponent_hands=features.opponent_hands,
            use_hand_order=features.hand_order,
            opponent_memories=features.opponent_memories,
            player_roles=features.player_roles,
        )
        draw_composition = (
            self.draw_composition(batch["draw_counts"].float() / 4.0)
            if features.draw_composition
            else zero_group
        )
        draw_order = (
            self.draw_order(
                batch["draw_order"],
                batch["draw_length"],
                top_n=features.draw_top_n,
            )
            if features.draw_order and features.draw_top_n > 0
            else zero_group
        )
        discard_composition = (
            self.discard_composition(
                batch["discard_counts"].float() / 4.0
            )
            if features.discard_composition
            else zero_group
        )
        discard_order = (
            self.discard_order(
                batch["discard_order"],
                batch["discard_length"],
                top_n=features.discard_top_n,
            )
            if features.discard_order and features.discard_top_n > 0
            else zero_group
        )
        played_composition = (
            self.played_composition(
                torch.log1p(batch["played_counts"].float())
                / math.log1p(500.0)
            )
            if features.played_history
            else zero_group
        )
        played_order = (
            self.played_order(
                batch["recent_plays"],
                batch["recent_play_length"],
                top_n=features.played_top_n,
            )
            if features.played_history and features.played_top_n > 0
            else zero_group
        )
        controls = (
            self.controls(batch["scalars"].float())
            if features.engine_controls
            else zero_group
        )
        return self.project(
            torch.cat(
                (
                    safe,
                    policy,
                    players,
                    draw_composition,
                    draw_order,
                    discard_composition,
                    discard_order,
                    played_composition,
                    played_order,
                    controls,
                ),
                dim=-1,
            )
        )


class CausalTemporalValueModel(nn.Module):
    """Large asymmetric critic with interchangeable causal backbones."""

    def __init__(
        self,
        *,
        actor_memory_size: int,
        features: CriticFeatureConfig | None = None,
        backbone: str = "mlp",
        hidden_size: int = 384,
        num_layers: int = 3,
        group_width: int = 96,
        transformer_heads: int = 8,
        dropout: float = 0.1,
        max_decisions: int = 1024,
        hand_encoder: str = "counts_moments",
        card_tokenization: str = "exact",
    ) -> None:
        super().__init__()
        if hidden_size < 32 or num_layers < 1:
            raise ValueError("critic hidden size and layer count are too small")
        if backbone not in ("mlp", "gru", "lstm", "transformer"):
            raise ValueError(f"unknown critic backbone: {backbone}")
        if backbone == "transformer" and hidden_size % transformer_heads:
            raise ValueError("transformer hidden size must divide by head count")
        if hand_encoder not in HAND_ENCODERS:
            raise ValueError(f"unknown hand encoder: {hand_encoder}")
        if card_tokenization not in CARD_TOKENIZATIONS:
            raise ValueError(
                f"unknown card tokenization: {card_tokenization}"
            )
        self.features = features or CriticFeatureConfig()
        self.backbone_name = backbone
        self.hand_encoder = hand_encoder
        self.card_tokenization = card_tokenization
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_decisions = max_decisions
        self.transformer_heads = transformer_heads
        self.dropout = float(dropout)
        self.snapshot_encoder = PrivilegedSnapshotEncoder(
            actor_memory_size=actor_memory_size,
            output_width=hidden_size,
            group_width=group_width,
            dropout=dropout,
            hand_encoder=hand_encoder,
            card_tokenization=card_tokenization,
        )
        if backbone == "gru":
            self.temporal = nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif backbone == "lstm":
            self.temporal = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        elif backbone == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.temporal = nn.TransformerEncoder(
                layer, num_layers=num_layers, enable_nested_tensor=False
            )
            self.position = nn.Embedding(max_decisions, hidden_size)
        else:
            self.temporal = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size),
            )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.value_head[-1].weight)
        nn.init.zeros_(self.value_head[-1].bias)
        self.win_head = nn.Linear(hidden_size, 1)
        self.remaining_head = nn.Linear(hidden_size, 1)
        # Validation-fitted affine calibration is deliberately non-trainable.
        # One pair per player count handles the large change in return scale
        # from two- to thirteen-player games without fragmenting the encoder.
        self.register_buffer(
            "value_calibration_scale", torch.ones(MAX_PLAYERS + 1)
        )
        self.register_buffer(
            "value_calibration_bias", torch.zeros(MAX_PLAYERS + 1)
        )

    def reset_value_calibration(self) -> None:
        self.value_calibration_scale.fill_(1.0)
        self.value_calibration_bias.zero_()

    def set_value_calibration(
        self,
        scale: Sequence[float],
        bias: Sequence[float],
    ) -> None:
        scale_tensor = torch.as_tensor(
            scale,
            device=self.value_calibration_scale.device,
            dtype=self.value_calibration_scale.dtype,
        )
        bias_tensor = torch.as_tensor(
            bias,
            device=self.value_calibration_bias.device,
            dtype=self.value_calibration_bias.dtype,
        )
        if scale_tensor.shape != self.value_calibration_scale.shape:
            raise ValueError("critic calibration scale has the wrong shape")
        if bias_tensor.shape != self.value_calibration_bias.shape:
            raise ValueError("critic calibration bias has the wrong shape")
        if not torch.isfinite(scale_tensor).all() or not torch.isfinite(
            bias_tensor
        ).all():
            raise ValueError("critic calibration must be finite")
        self.value_calibration_scale.copy_(scale_tensor)
        self.value_calibration_bias.copy_(bias_tensor)

    def _calibrate_value(
        self,
        value: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        player_count = batch["player_mask"].sum(dim=-1).long().clamp(
            min=0, max=MAX_PLAYERS
        )
        return (
            value * self.value_calibration_scale[player_count]
            + self.value_calibration_bias[player_count]
        )

    def model_config(self) -> dict[str, object]:
        return {
            "architecture": "causal_privileged_value_v1",
            "backbone": self.backbone_name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "max_decisions": self.max_decisions,
            "transformer_heads": self.transformer_heads,
            "dropout": self.dropout,
            "actor_memory_size": self.snapshot_encoder.actor_memory_size,
            "group_width": self.snapshot_encoder.group_width,
            "hand_encoder": self.hand_encoder,
            "card_tokenization": self.card_tokenization,
            "features": asdict(self.features),
        }

    def initial_state(
        self,
        batch_size: int,
        *,
        device: str | torch.device | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """Create streaming state for an online GRU/LSTM critic."""

        if batch_size < 1:
            raise ValueError("critic state batch size must be positive")
        parameter = next(self.parameters())
        state_device = parameter.device if device is None else torch.device(device)
        shape = (self.num_layers, batch_size, self.hidden_size)
        if self.backbone_name == "gru":
            return torch.zeros(
                shape, device=state_device, dtype=parameter.dtype
            )
        if self.backbone_name == "lstm":
            return (
                torch.zeros(shape, device=state_device, dtype=parameter.dtype),
                torch.zeros(shape, device=state_device, dtype=parameter.dtype),
            )
        if self.backbone_name == "mlp":
            return None
        raise ValueError(
            "streaming transformer inference requires retained token history"
        )

    def step(
        self,
        batch: Mapping[str, torch.Tensor],
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[
        dict[str, torch.Tensor],
        torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        """Evaluate one current-turn snapshot without revisiting prior turns."""

        if self.backbone_name == "transformer":
            raise ValueError(
                "use forward with causal token history for transformer critics"
            )
        temporal_batch = {
            name: value.unsqueeze(1) for name, value in batch.items()
        }
        encoded = self.snapshot_encoder(temporal_batch, self.features)
        if self.backbone_name in ("gru", "lstm"):
            hidden, next_state = self.temporal(  # type: ignore[operator]
                encoded, state
            )
        else:
            hidden = self.temporal(encoded)  # type: ignore[operator]
            next_state = None
        hidden = hidden[:, 0]
        value = self.value_head(hidden).squeeze(-1) + (
            batch["public_value"].float()
            if self.features.public_value
            else 0.0
        )
        return (
            {
                "value": self._calibrate_value(value, batch),
                "win_logit": self.win_head(hidden).squeeze(-1),
                "remaining_log": self.remaining_head(hidden).squeeze(-1),
            },
            next_state,
        )

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.snapshot_encoder(batch, self.features)
        if mask is None:
            mask = torch.ones(
                encoded.shape[:2], device=encoded.device, dtype=torch.bool
            )
        if encoded.shape[1] > self.max_decisions:
            raise ValueError(
                f"sequence has {encoded.shape[1]} decisions, above the "
                f"configured {self.max_decisions}"
            )
        if self.backbone_name in ("gru", "lstm"):
            hidden, _ = self.temporal(encoded)  # type: ignore[operator]
        elif self.backbone_name == "transformer":
            positions = torch.arange(
                encoded.shape[1], device=encoded.device
            )
            temporal_input = encoded + self.position(positions).unsqueeze(0)
            causal = torch.triu(
                torch.ones(
                    encoded.shape[1],
                    encoded.shape[1],
                    device=encoded.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            hidden = self.temporal(  # type: ignore[operator]
                temporal_input,
                mask=causal,
                src_key_padding_mask=~mask,
            )
        else:
            hidden = self.temporal(encoded)  # type: ignore[operator]
        value = self.value_head(hidden).squeeze(-1)
        if self.features.public_value:
            value = value + batch["public_value"].float()
        value = self._calibrate_value(value, batch)
        return {
            "value": value,
            "win_logit": self.win_head(hidden).squeeze(-1),
            "remaining_log": self.remaining_head(hidden).squeeze(-1),
        }


def _model_from_checkpoint_payload(
    payload: Mapping[str, object],
    *,
    device: str | torch.device,
) -> CausalTemporalValueModel:
    """Construct one critic member from a validated checkpoint payload."""

    config = dict(payload["model"])
    if config.pop("architecture", None) != "causal_privileged_value_v1":
        raise ValueError("unsupported privileged critic architecture")
    config["features"] = CriticFeatureConfig(**config["features"])
    model = CausalTemporalValueModel(**config)
    incompatible = model.load_state_dict(payload["state_dict"], strict=False)
    allowed_missing = {
        "value_calibration_scale",
        "value_calibration_bias",
    }
    if set(incompatible.missing_keys) - allowed_missing:
        raise RuntimeError(
            "critic checkpoint is missing model tensors: "
            + ", ".join(incompatible.missing_keys)
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "critic checkpoint has unexpected model tensors: "
            + ", ".join(incompatible.unexpected_keys)
        )
    model.to(device).eval()
    return model


def load_privileged_critic_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CausalTemporalValueModel, dict[str, object]]:
    """Load an audited critic without constructing or mutating an actor."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "pesten-causal-privileged-critic-v1":
        raise ValueError("unsupported privileged critic checkpoint format")
    model = _model_from_checkpoint_payload(payload, device=device)
    return model, payload


class PrivilegedCriticRuntime:
    """Batched online state manager keyed by game/seat trajectory id."""

    def __init__(self, model: CausalTemporalValueModel) -> None:
        if model.backbone_name == "transformer":
            raise ValueError(
                "online runtime currently supports MLP, GRU, and LSTM critics"
            )
        self.model = model.eval()
        self.states: dict[
            Hashable,
            torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
        ] = {}

    def reset(self, stream_ids: Sequence[Hashable] | None = None) -> None:
        if stream_ids is None:
            self.states.clear()
            return
        for stream_id in stream_ids:
            self.states.pop(stream_id, None)

    def evaluate(
        self,
        snapshots: Sequence[PrivilegedSnapshot],
        stream_ids: Sequence[Hashable],
    ) -> dict[str, torch.Tensor]:
        if len(snapshots) != len(stream_ids) or not snapshots:
            raise ValueError(
                "one non-empty critic stream id is required per snapshot"
            )
        if len(set(stream_ids)) != len(stream_ids):
            raise ValueError("a critic stream may advance at most once per batch")
        device = next(self.model.parameters()).device
        batch = snapshots_to_model_batch(snapshots, device=device)
        if self.model.backbone_name in ("gru", "lstm"):
            states = [
                self.states[stream_id] if stream_id in self.states
                else self.model.initial_state(1, device=device)
                for stream_id in stream_ids
            ]
        if self.model.backbone_name == "gru":
            stacked_state = torch.cat(states, dim=1)  # type: ignore[arg-type]
        elif self.model.backbone_name == "lstm":
            stacked_state = tuple(
                torch.cat(
                    [state[part] for state in states], dim=1  # type: ignore[index]
                )
                for part in range(2)
            )
        else:
            stacked_state = None
        with torch.inference_mode():
            outputs, next_state = self.model.step(batch, stacked_state)
        if self.model.backbone_name == "gru":
            assert isinstance(next_state, torch.Tensor)
            for index, stream_id in enumerate(stream_ids):
                self.states[stream_id] = next_state[
                    :, index : index + 1
                ].detach()
        elif self.model.backbone_name == "lstm":
            assert isinstance(next_state, tuple)
            for index, stream_id in enumerate(stream_ids):
                self.states[stream_id] = tuple(
                    part[:, index : index + 1].detach()
                    for part in next_state
                )
        else:
            for stream_id in stream_ids:
                self.states[stream_id] = None
        return {name: value.detach().cpu() for name, value in outputs.items()}


class AveragedPrivilegedCritic(nn.Module):
    """Validation-calibrated arithmetic ensemble of independent critics."""

    def __init__(
        self, models: Sequence[CausalTemporalValueModel]
    ) -> None:
        super().__init__()
        if not models:
            raise ValueError("critic ensemble requires at least one member")
        self.models = nn.ModuleList(models)
        self.register_buffer(
            "value_calibration_scale", torch.ones(MAX_PLAYERS + 1)
        )
        self.register_buffer(
            "value_calibration_bias", torch.zeros(MAX_PLAYERS + 1)
        )

    def reset_value_calibration(self) -> None:
        self.value_calibration_scale.fill_(1.0)
        self.value_calibration_bias.zero_()

    def set_value_calibration(
        self,
        scale: Sequence[float],
        bias: Sequence[float],
    ) -> None:
        scale_tensor = torch.as_tensor(
            scale,
            device=self.value_calibration_scale.device,
            dtype=self.value_calibration_scale.dtype,
        )
        bias_tensor = torch.as_tensor(
            bias,
            device=self.value_calibration_bias.device,
            dtype=self.value_calibration_bias.dtype,
        )
        if scale_tensor.shape != self.value_calibration_scale.shape:
            raise ValueError("ensemble calibration scale has the wrong shape")
        if bias_tensor.shape != self.value_calibration_bias.shape:
            raise ValueError("ensemble calibration bias has the wrong shape")
        self.value_calibration_scale.copy_(scale_tensor)
        self.value_calibration_bias.copy_(bias_tensor)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        rows = [model(batch, mask=mask) for model in self.models]
        result = {
            name: torch.stack([row[name] for row in rows]).mean(dim=0)
            for name in rows[0]
        }
        player_count = batch["player_mask"].sum(dim=-1).long().clamp(
            min=0, max=MAX_PLAYERS
        )
        result["value"] = (
            result["value"] * self.value_calibration_scale[player_count]
            + self.value_calibration_bias[player_count]
        )
        return result


class AveragedPrivilegedCriticRuntime:
    """Streaming counterpart of :class:`AveragedPrivilegedCritic`."""

    def __init__(self, ensemble: AveragedPrivilegedCritic) -> None:
        self.ensemble = ensemble.eval()
        self.runtimes = [
            PrivilegedCriticRuntime(model) for model in ensemble.models
        ]

    def reset(self, stream_ids: Sequence[Hashable] | None = None) -> None:
        for runtime in self.runtimes:
            runtime.reset(stream_ids)

    def evaluate(
        self,
        snapshots: Sequence[PrivilegedSnapshot],
        stream_ids: Sequence[Hashable],
    ) -> dict[str, torch.Tensor]:
        rows = [
            runtime.evaluate(snapshots, stream_ids)
            for runtime in self.runtimes
        ]
        result = {
            name: torch.stack([row[name] for row in rows]).mean(dim=0)
            for name in rows[0]
        }
        player_count = torch.tensor(
            [int(snapshot.player_mask.sum()) for snapshot in snapshots],
            dtype=torch.long,
        )
        scale = self.ensemble.value_calibration_scale.detach().cpu()
        bias = self.ensemble.value_calibration_bias.detach().cpu()
        result["value"] = (
            result["value"] * scale[player_count] + bias[player_count]
        )
        return result


def load_privileged_critic_ensemble_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[AveragedPrivilegedCritic, dict[str, object]]:
    """Load a self-contained arithmetic ensemble and shared calibration."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format") != "pesten-causal-privileged-critic-ensemble-v1":
        raise ValueError("unsupported privileged critic ensemble format")
    members = payload.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("privileged critic ensemble has no members")
    models = [
        _model_from_checkpoint_payload(member, device=device)
        for member in members
    ]
    ensemble = AveragedPrivilegedCritic(models).to(device)
    scale = payload.get("value_calibration_scale")
    bias = payload.get("value_calibration_bias")
    if not isinstance(scale, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError("privileged critic ensemble has no shared calibration")
    ensemble.set_value_calibration(scale.tolist(), bias.tolist())
    ensemble.eval()
    return ensemble, payload


def load_privileged_value_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[CausalTemporalValueModel | AveragedPrivilegedCritic, dict[str, object]]:
    """Load either a single privileged critic or a self-contained ensemble."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    checkpoint_format = payload.get("format")
    if checkpoint_format == "pesten-causal-privileged-critic-v1":
        return _model_from_checkpoint_payload(payload, device=device), payload
    if checkpoint_format == "pesten-causal-privileged-critic-ensemble-v1":
        members = payload.get("members")
        if not isinstance(members, list) or not members:
            raise ValueError("privileged critic ensemble has no members")
        ensemble = AveragedPrivilegedCritic(
            [
                _model_from_checkpoint_payload(member, device=device)
                for member in members
            ]
        ).to(device)
        scale = payload.get("value_calibration_scale")
        bias = payload.get("value_calibration_bias")
        if not isinstance(scale, torch.Tensor) or not isinstance(
            bias, torch.Tensor
        ):
            raise ValueError("privileged critic ensemble has no shared calibration")
        ensemble.set_value_calibration(scale.tolist(), bias.tolist())
        ensemble.eval()
        return ensemble, payload
    raise ValueError("unsupported privileged value checkpoint format")


def critic_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    mask: torch.Tensor,
    *,
    win_coefficient: float = 0.15,
    remaining_coefficient: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Monte Carlo value loss with causal representation auxiliaries."""

    valid = mask.bool()
    target = batch["returns"].float()[valid]
    predicted = outputs["value"][valid]
    value_loss = F.mse_loss(predicted, target)
    win_loss = F.binary_cross_entropy_with_logits(
        outputs["win_logit"][valid], batch["win_target"].float()[valid]
    )
    remaining_target = torch.log1p(
        batch["remaining_decisions"].float()[valid]
    ) / math.log1p(512.0)
    remaining_loss = F.smooth_l1_loss(
        outputs["remaining_log"][valid], remaining_target
    )
    total = (
        value_loss
        + win_coefficient * win_loss
        + remaining_coefficient * remaining_loss
    )
    return total, {
        "loss": float(total.detach()),
        "value_mse": float(value_loss.detach()),
        "win_bce": float(win_loss.detach()),
        "remaining_smooth_l1": float(remaining_loss.detach()),
    }


def collate_privileged_ppo_episodes(
    episodes: Sequence[tuple[Sequence[object], bool | None]],
    *,
    gamma: float = DEFAULT_GAMMA,
    decision_reward: float = -0.01,
    draw_reward: float = -0.75,
    device: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Materialize snapshots retained by the opt-in PPO rollout collector."""

    rows = []
    for episode_index, (trajectory, won) in enumerate(episodes):
        if not trajectory:
            continue
        snapshots = [
            getattr(step, "privileged_snapshot", None)
            for step in trajectory
        ]
        if any(snapshot is None for snapshot in snapshots):
            raise ValueError(
                "every PPO step must retain a privileged pre-action snapshot"
            )
        player_count = int(snapshots[0].player_mask.sum())
        rows.append(
            episode_from_snapshots(
                snapshots,
                won=won,
                deal_seed=episode_index,
                seat=0,
                player_count=player_count,
                gamma=gamma,
                decision_reward=decision_reward,
                draw_reward=draw_reward,
            )
        )
    if not rows:
        raise ValueError("privileged critic update requires non-empty episodes")
    max_length = max(len(row["returns"]) for row in rows)
    lengths = np.asarray([len(row["returns"]) for row in rows])
    mask = torch.from_numpy(
        np.arange(max_length)[None, :] < lengths[:, None]
    ).to(device=device)
    batch = {}
    for name in DATA_FIELDS:
        sample = np.asarray(rows[0][name])
        fill = (
            -1
            if name
            in ("hand_order", "draw_order", "discard_order", "recent_plays")
            else 0
        )
        # Pad on the host before transferring. Per-trajectory CUDA assignment
        # launches hundreds of tiny transfers/copies per critic minibatch.
        target = np.full(
            (len(rows), max_length, *sample.shape[1:]),
            fill_value=fill,
            dtype=sample.dtype,
        )
        for row_index, row in enumerate(rows):
            values = np.asarray(row[name])
            target[row_index, : len(values)] = values
        batch[name] = torch.from_numpy(target).to(device=device)
    return batch, mask


def update_privileged_critic_from_ppo(
    model: nn.Module,
    episodes: Sequence[tuple[Sequence[object], bool | None]],
    optimizer: torch.optim.Optimizer,
    *,
    update_epochs: int = 4,
    minibatch_trajectories: int = 32,
    seed: int = 0,
    gamma: float = DEFAULT_GAMMA,
    decision_reward: float = -0.01,
    draw_reward: float = -0.75,
    win_coefficient: float = 0.15,
    remaining_coefficient: float = 0.05,
) -> dict[str, object]:
    """Update only the external critic on newly collected on-policy games."""

    if update_epochs < 1 or minibatch_trajectories < 1:
        raise ValueError("critic update epochs and minibatch size must be positive")
    usable = [episode for episode in episodes if episode[0]]
    if not usable:
        raise ValueError("critic update requires at least one decision")
    device = next(model.parameters()).device
    rng = random.Random(seed)
    epoch_rows = []
    model.train()
    for epoch in range(update_epochs):
        order = list(range(len(usable)))
        rng.shuffle(order)
        totals = {
            "loss": 0.0,
            "value_mse": 0.0,
            "win_bce": 0.0,
            "remaining_smooth_l1": 0.0,
            "samples": 0,
        }
        for start in range(0, len(order), minibatch_trajectories):
            selected = [
                usable[index]
                for index in order[start : start + minibatch_trajectories]
            ]
            batch, mask = collate_privileged_ppo_episodes(
                selected,
                gamma=gamma,
                decision_reward=decision_reward,
                draw_reward=draw_reward,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch, mask=mask)
            loss, row = critic_loss(
                outputs,
                batch,
                mask,
                win_coefficient=win_coefficient,
                remaining_coefficient=remaining_coefficient,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            samples = int(mask.sum().item())
            for name in (
                "loss",
                "value_mse",
                "win_bce",
                "remaining_smooth_l1",
            ):
                totals[name] += row[name] * samples
            totals["samples"] += samples
        epoch_rows.append(
            {
                "epoch": epoch + 1,
                **{
                    name: totals[name] / totals["samples"]
                    for name in (
                        "loss",
                        "value_mse",
                        "win_bce",
                        "remaining_smooth_l1",
                    )
                },
                "samples": totals["samples"],
            }
        )
    model.eval()
    return {
        "epochs": epoch_rows,
        "trajectories": len(usable),
        "decisions": sum(len(trajectory) for trajectory, _ in usable),
    }


def value_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.size == 0:
        raise ValueError("non-empty matching value vectors are required")
    errors = predicted - target
    variance = float(np.var(target))
    correlation = (
        float(np.corrcoef(predicted, target)[0, 1])
        if predicted.size > 1
        and float(np.std(predicted)) > 0
        and float(np.std(target)) > 0
        else 0.0
    )
    prediction_variance = float(np.var(predicted))
    calibration_slope = (
        float(
            np.mean(
                (predicted - predicted.mean())
                * (target - target.mean())
            )
            / prediction_variance
        )
        if prediction_variance > 0
        else 0.0
    )
    calibration_intercept = float(
        target.mean() - calibration_slope * predicted.mean()
    )
    return {
        "samples": int(predicted.size),
        "mse": float(np.mean(errors * errors)),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "mae": float(np.mean(np.abs(errors))),
        "bias": float(np.mean(errors)),
        "explained_variance": (
            1.0 - float(np.var(errors)) / variance if variance > 0 else 0.0
        ),
        "correlation": correlation,
        "calibration_slope": calibration_slope,
        "calibration_intercept": calibration_intercept,
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
