"""Information-safe intermediate states for a future DAGS training pilot.

Data-Augmented Game Starts (DAGS; arXiv:2605.14379) restores a *full*
environment state but gives each policy only its ordinary player observation.
This module keeps those two interfaces explicit:

* :class:`DAGSState` is simulator-private and contains hidden cards and RNG state.
* :func:`dags_observation` returns only the existing 121 policy features plus the
  paper's binary root/dataset task flag.

It deliberately does not alter the current PPO or NashPG training paths.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ai_observation import OBSERVATION_SIZE
from batched_pesten import (
    PHASE_CARD,
    PHASE_DRAW_RETRY,
    PHASE_SUIT,
    STANDARD_DECK,
    BatchedPesten,
)


DAGS_OBSERVATION_SIZE = OBSERVATION_SIZE + 1
DAGS_SOURCE_FLAG_INDEX = OBSERVATION_SIZE

# These are the per-game arrays that fully determine future game mechanics.
# Policy assignment intentionally remains owned by the destination engine.
_STATE_ARRAY_FIELDS = (
    "hands",
    "hand_counts",
    "hand_sizes",
    "draw_piles",
    "draw_counts",
    "discards",
    "discard_counts",
    "current_sort",
    "current_rank",
    "current_player",
    "turn_player",
    "direction",
    "penalty_amount",
    "sorts_played",
    "turn_counts",
    "cards_played",
    "game_over",
    "winners",
    "phase",
    "turn_started_with_one_card",
    "drawn_card",
)


@dataclass(frozen=True)
class DAGSState:
    """One immutable, simulator-private intermediate game state."""

    num_players: int
    arrays: tuple[tuple[str, np.ndarray], ...]
    rng_state: tuple
    public_events: tuple[tuple, ...]

    def array(self, name: str) -> np.ndarray:
        for field_name, value in self.arrays:
            if field_name == name:
                return value
        raise KeyError(name)

    @property
    def source_turn(self) -> int:
        return int(self.array("turn_counts"))


def _game_index(engine: BatchedPesten, game_index: int) -> int:
    index = int(game_index)
    if not 0 <= index < engine.num_games:
        raise IndexError(f"game index {index} is out of range")
    return index


def _readonly_copy(value: np.ndarray) -> np.ndarray:
    result = np.array(value, copy=True)
    result.setflags(write=False)
    return result


def _readonly_rng_state(rng: np.random.RandomState) -> tuple:
    state = list(copy.deepcopy(rng.get_state()))
    state[1] = _readonly_copy(state[1])
    return tuple(state)


def _validate_state(state: DAGSState) -> None:
    if state.num_players != int(state.array("hand_sizes").shape[0]):
        raise ValueError("DAGS state player count is inconsistent")
    if bool(state.array("game_over")):
        raise ValueError("DAGS state must be nonterminal")
    phase = int(state.array("phase"))
    if phase not in (PHASE_CARD, PHASE_DRAW_RETRY, PHASE_SUIT):
        raise ValueError("DAGS state has an invalid phase")

    hand_sizes = state.array("hand_sizes").astype(np.int64, copy=False)
    hands = state.array("hands")
    hand_counts = state.array("hand_counts")
    cards: list[int] = []
    for player_index, size_value in enumerate(hand_sizes):
        size = int(size_value)
        if size < 0 or size > hands.shape[1]:
            raise ValueError("DAGS state has an invalid hand size")
        hand = hands[player_index, :size].astype(np.int64, copy=False)
        expected_counts = np.bincount(hand, minlength=hand_counts.shape[1])
        if not np.array_equal(expected_counts, hand_counts[player_index]):
            raise ValueError("DAGS state hand counts do not match its hands")
        cards.extend(hand.tolist())

    draw_count = int(state.array("draw_counts"))
    discard_count = int(state.array("discard_counts"))
    draw_pile = state.array("draw_piles")
    discards = state.array("discards")
    if not 0 <= draw_count <= len(draw_pile):
        raise ValueError("DAGS state has an invalid draw count")
    if not 1 <= discard_count <= len(discards):
        raise ValueError("DAGS state has an invalid discard count")
    cards.extend(draw_pile[:draw_count].astype(np.int64, copy=False).tolist())
    cards.extend(discards[:discard_count].astype(np.int64, copy=False).tolist())

    actual_deck = np.sort(np.asarray(cards, dtype=np.int64))
    expected_deck = np.sort(np.asarray(STANDARD_DECK, dtype=np.int64))
    if not np.array_equal(actual_deck, expected_deck):
        raise ValueError("DAGS state does not conserve the complete deck")

    current_player = int(state.array("current_player"))
    turn_player = int(state.array("turn_player"))
    if not 0 <= current_player < state.num_players:
        raise ValueError("DAGS state has an invalid current player")
    if not 0 <= turn_player < state.num_players:
        raise ValueError("DAGS state has an invalid turn player")


def capture_dags_state(
    engine: BatchedPesten,
    game_index: int,
) -> DAGSState:
    """Capture one valid decision-boundary state from a live simulator."""
    index = _game_index(engine, game_index)
    if bool(engine.game_over[index]) or int(engine.turn_counts[index]) >= (
        engine.max_turns
    ):
        raise ValueError("DAGS state must come from an active game")
    state = DAGSState(
        num_players=engine.num_players,
        arrays=tuple(
            (name, _readonly_copy(getattr(engine, name)[index]))
            for name in _STATE_ARRAY_FIELDS
        ),
        rng_state=_readonly_rng_state(engine.rngs[index]),
        public_events=tuple(
            tuple(event) for event in engine.public_events[index]
        ),
    )
    _validate_state(state)
    return state


def restore_dags_state(
    engine: BatchedPesten,
    game_index: int,
    state: DAGSState,
) -> None:
    """Restore simulator-private state without changing policy assignments."""
    index = _game_index(engine, game_index)
    _validate_state(state)
    if engine.num_players != state.num_players:
        raise ValueError("DAGS state and destination player counts differ")
    if state.source_turn >= engine.max_turns:
        raise ValueError("DAGS state exceeds the destination turn horizon")

    expected_names = set(_STATE_ARRAY_FIELDS)
    actual_names = {name for name, _ in state.arrays}
    if actual_names != expected_names:
        raise ValueError("DAGS state fields are incomplete")
    for name in _STATE_ARRAY_FIELDS:
        target = getattr(engine, name)
        destination = target[index]
        source = state.array(name)
        if np.shape(destination) != np.shape(source):
            raise ValueError(f"DAGS state field {name!r} has the wrong shape")
        if source.ndim == 0:
            target[index] = source
        else:
            np.copyto(destination, source)
    engine.rngs[index].set_state(copy.deepcopy(state.rng_state))
    engine.public_events[index] = [
        tuple(event) for event in state.public_events
    ]


def dags_observation(
    engine: BatchedPesten,
    game_index: int,
    *,
    dataset_start: bool,
) -> np.ndarray:
    """Return the normal information set plus the paper's task-source flag."""
    index = _game_index(engine, game_index)
    base = engine.observation(
        index,
        suit_mode=int(engine.phase[index]) == PHASE_SUIT,
    )
    flagged = np.empty(DAGS_OBSERVATION_SIZE, dtype=np.float32)
    flagged[:OBSERVATION_SIZE] = base
    flagged[DAGS_SOURCE_FLAG_INDEX] = float(bool(dataset_start))
    return flagged
