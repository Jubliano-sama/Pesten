"""Batched, hidden-information-safe Pesten rollouts for PPO training.

The canonical :mod:`game` module remains the rules authority for interactive
play.  This module stores many independent games in compact integer arrays and
advances them at common policy-decision boundaries.  That makes the expensive
actor call genuinely batched while retaining the legacy 121-feature/55-action
training contract.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Hashable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_END,
    COMPATIBLE_START,
    DIRECTION_INDEX,
    HAND_END,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PASS_ACTION,
    PLAYER_HAND_SIZE_SLOTS,
    PLAYER_HAND_SIZES_END,
    PLAYER_HAND_SIZES_START,
    PLAYER_INDEX,
    SORTS_PLAYED_START,
    SUIT_ACTION_END,
    SUIT_ACTION_START,
)


POLICY_RL = 0
POLICY_SMART = 1
POLICY_RANDOM = 2
POLICY_FIRST_LEGAL = 3  # Deterministic canonical differential-test policy.
POLICY_SMARTER = 4

PHASE_CARD = 0
PHASE_DRAW_RETRY = 1
PHASE_SUIT = 2

SUIT_NAMES = ("harten", "ruiten", "schoppen", "klaveren", "special")
PENALTY_RANKS = frozenset((7, 8, 1, 0, 13, 2))
LAST_CARD_PENALTY_RANKS = PENALTY_RANKS
INVALID_START_RANKS = PENALTY_RANKS

CARD_SORT = np.asarray(
    [number // 12 if number < 48 else 4 for number in range(50)],
    dtype=np.int8,
)
CARD_RANK = np.asarray(
    [
        number % 12 + 1 if number < 48 else (13 if number == 48 else 0)
        for number in range(50)
    ],
    dtype=np.int8,
)
STANDARD_DECK = np.asarray(
    list(range(50)) + [49, 48, 48, 48],
    dtype=np.int8,
)


def deck_count_for_players(num_players: int) -> int:
    """Return enough 54-card decks for seven cards and an opening reserve."""
    if num_players < 2:
        raise ValueError("Pesten requires at least two players")
    minimum_for_capacity = (
        num_players * 7 + 3 + len(STANDARD_DECK) - 1
    ) // len(STANDARD_DECK)
    # Seven players can leave only five cards in one deck; two decks keep a
    # reliable valid-opening reserve as well as the seven-card invariant.
    return max(2 if num_players >= 7 else 1, minimum_for_capacity)
CARD_RANK_MASKS = (
    CARD_RANK[np.newaxis, :]
    == np.arange(14, dtype=np.int8)[:, np.newaxis]
)
CARD_COMPATIBILITY_MASKS = (
    CARD_RANK_MASKS[np.newaxis, :, :]
    | (CARD_RANK[np.newaxis, np.newaxis, :] == 0)
    | (CARD_RANK[np.newaxis, np.newaxis, :] == 13)
    | (
        CARD_SORT[np.newaxis, np.newaxis, :]
        == np.arange(5, dtype=np.int8)[:, np.newaxis, np.newaxis]
    )
)
CARD_IS_PENALTY = np.isin(CARD_RANK, tuple(PENALTY_RANKS))
SMART_PENALTY_BONUS = (
    -0.03556864491834867,
    0.04122020478114498,
    0.04665718005466267,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    -1.5328572573442552,
    0.0,
    0.0,
    0.0,
    0.0,
    -3.0043124890687474,
)
SMARTER_SPECIAL_BONUS = (
    1.5,
    1.0,
    1.0,
    0.0,
    0.0,
    0.0,
    0.0,
    2.0,
    1.5,
    0.0,
    0.0,
    0.0,
    0.0,
    1.0,
)
CARD_RANK_MASKS.setflags(write=False)
CARD_COMPATIBILITY_MASKS.setflags(write=False)
CARD_IS_PENALTY.setflags(write=False)


def _compatible(card_number: int, current_sort: int, current_rank: int) -> bool:
    rank = int(CARD_RANK[card_number])
    return (
        rank == current_rank
        or rank in (0, 13)
        or int(CARD_SORT[card_number]) == current_sort
    )


@dataclass(frozen=True)
class ActorPolicy:
    """One actor shared by any number of seats and games."""

    actor: torch.nn.Module
    device: torch.device
    temperature: float
    collect_data: bool
    greedy: bool = False
    autocast: bool = True


@dataclass
class RolloutResult:
    observations: torch.Tensor
    actions: torch.Tensor
    log_probabilities: torch.Tensor
    rewards: torch.Tensor
    episode_lengths: list[int]
    environment_turns: int
    completed_games: int
    policy_decisions: int
    inference_calls: int
    peak_buffer_bytes: int
    wins: int
    draws: int


class VectorRolloutBuffer:
    """A growable, preallocated buffer with episode-contiguous finalization."""

    def __init__(self, episode_slots: int, initial_capacity: int) -> None:
        self.capacity = max(1, int(initial_capacity))
        self.size = 0
        self.observations = np.empty(
            (self.capacity, OBSERVATION_SIZE),
            dtype=np.float32,
        )
        self.actions = np.empty(self.capacity, dtype=np.int64)
        self.log_probabilities = np.empty(self.capacity, dtype=np.float32)
        self.rewards = np.empty(self.capacity, dtype=np.float32)
        self.episode_ids = np.empty(self.capacity, dtype=np.int32)
        self.episode_counts = np.zeros(episode_slots, dtype=np.int32)
        self.last_sample = np.full(episode_slots, -1, dtype=np.int64)
        self.peak_bytes = self._allocated_bytes()

    def _allocated_bytes(self) -> int:
        return sum(
            array.nbytes
            for array in (
                self.observations,
                self.actions,
                self.log_probabilities,
                self.rewards,
                self.episode_ids,
                self.episode_counts,
                self.last_sample,
            )
        )

    def _grow(self, needed: int) -> None:
        if needed <= self.capacity:
            return
        new_capacity = max(needed, self.capacity * 2)
        for name in (
            "observations",
            "actions",
            "log_probabilities",
            "rewards",
            "episode_ids",
        ):
            old = getattr(self, name)
            new = np.empty((new_capacity, *old.shape[1:]), dtype=old.dtype)
            new[: self.size] = old[: self.size]
            setattr(self, name, new)
        self.capacity = new_capacity
        self.peak_bytes = max(self.peak_bytes, self._allocated_bytes())

    def append(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        log_probabilities: np.ndarray,
        rewards: np.ndarray,
        episode_ids: np.ndarray,
    ) -> None:
        count = len(actions)
        if count == 0:
            return
        start = self.size
        end = start + count
        self._grow(end)
        self.observations[start:end] = observations
        self.actions[start:end] = actions
        self.log_probabilities[start:end] = log_probabilities
        self.rewards[start:end] = rewards
        self.episode_ids[start:end] = episode_ids
        sample_indices = np.arange(start, end, dtype=np.int64)
        np.add.at(self.episode_counts, episode_ids, 1)
        self.last_sample[episode_ids] = sample_indices
        self.size = end

    def add_terminal_rewards(
        self,
        episode_ids: np.ndarray,
        terminal_rewards: np.ndarray,
    ) -> None:
        if len(episode_ids) == 0:
            return
        last = self.last_sample[episode_ids]
        populated = last >= 0
        self.rewards[last[populated]] += terminal_rewards[populated]

    def finalize(
        self,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        if self.size == 0:
            raise RuntimeError("data generation produced no policy samples")
        order = np.argsort(self.episode_ids[: self.size], kind="stable")
        populated_ids = np.flatnonzero(self.episode_counts)
        episode_lengths = self.episode_counts[populated_ids].tolist()
        observations = torch.from_numpy(
            np.ascontiguousarray(self.observations[: self.size][order])
        ).to(device=device)
        actions = torch.from_numpy(
            np.ascontiguousarray(self.actions[: self.size][order])
        ).to(device=device)
        log_probabilities = torch.from_numpy(
            np.ascontiguousarray(self.log_probabilities[: self.size][order])
        ).to(device=device)
        rewards = torch.from_numpy(
            np.ascontiguousarray(self.rewards[: self.size][order])
        ).to(device=device)
        return (
            observations,
            actions,
            log_probabilities,
            rewards,
            episode_lengths,
        )


class BatchedPesten:
    """Integer-array state machine for independent games with equal seat counts."""

    def __init__(
        self,
        policy_kinds: np.ndarray,
        actor_keys: np.ndarray,
        seeds: np.ndarray,
        *,
        max_turns: int = 500,
        smart_agents_have_game_reference: bool = False,
        native_smart_policy: bool = False,
        native_rollout_engine: bool = False,
        record_public_events: bool = True,
    ) -> None:
        policy_kinds = np.asarray(policy_kinds, dtype=np.int8)
        actor_keys = np.asarray(actor_keys, dtype=object)
        seeds = np.asarray(seeds, dtype=np.uint32)
        if policy_kinds.ndim != 2:
            raise ValueError("policy_kinds must have shape (games, players)")
        if actor_keys.shape != policy_kinds.shape:
            raise ValueError("actor_keys must match policy_kinds")
        if len(seeds) != len(policy_kinds):
            raise ValueError("one deterministic seed is required per game")
        if policy_kinds.shape[1] < 2:
            raise ValueError("Pesten requires at least two players")

        self.policy_kinds = policy_kinds
        self.actor_keys = actor_keys
        self.num_games, self.num_players = policy_kinds.shape
        self.max_turns = int(max_turns)
        self.record_public_events = bool(record_public_events)
        self.smart_agents_have_game_reference = smart_agents_have_game_reference
        self.deck_count = deck_count_for_players(self.num_players)
        self.deck_template = np.tile(STANDARD_DECK, self.deck_count)
        self._smart_action_native = None
        self._smart_actions_native = None
        if native_smart_policy:
            try:
                from numba_smart_policy import (
                    smart_action_native,
                    smart_actions_native,
                )
            except ImportError:
                smart_action_native = None
                smart_actions_native = None
            self._smart_action_native = smart_action_native
            self._smart_actions_native = smart_actions_native
        self._apply_simple_actions_native = None
        self._rl_has_card_decisions_native = None
        self._observations_into_native = None
        self._native_event_deltas = np.zeros(8, dtype=np.int64)
        if native_rollout_engine:
            try:
                from numba_rollout_engine import (
                    apply_simple_card_actions_native,
                    observations_into_native,
                    rl_has_card_decisions_native,
                )
            except ImportError:
                apply_simple_card_actions_native = None
                observations_into_native = None
                rl_has_card_decisions_native = None
            self._apply_simple_actions_native = (
                apply_simple_card_actions_native
            )
            self._rl_has_card_decisions_native = (
                rl_has_card_decisions_native
            )
            self._observations_into_native = observations_into_native
        self.rngs = [np.random.RandomState(int(seed)) for seed in seeds]

        shape = (self.num_games,)
        player_shape = (self.num_games, self.num_players)
        self.hands = np.full((*player_shape, len(self.deck_template)), -1, dtype=np.int8)
        self.hand_counts = np.zeros((*player_shape, 50), dtype=np.uint8)
        self.hand_sizes = np.zeros(player_shape, dtype=np.uint8)
        self.draw_piles = np.full(
            (self.num_games, len(self.deck_template)),
            -1,
            dtype=np.int8,
        )
        self.draw_counts = np.zeros(shape, dtype=np.uint8)
        self.discards = np.full_like(self.draw_piles, -1)
        self.discard_counts = np.zeros(shape, dtype=np.uint8)
        self.current_sort = np.zeros(shape, dtype=np.int8)
        self.current_rank = np.zeros(shape, dtype=np.int8)
        self.current_player = np.zeros(shape, dtype=np.int8)
        self.turn_player = np.zeros(shape, dtype=np.int8)
        self.direction = np.ones(shape, dtype=np.int8)
        self.penalty_amount = np.zeros(shape, dtype=np.int16)
        self.sorts_played = np.zeros((self.num_games, 5), dtype=np.int32)
        self.turn_counts = np.zeros(shape, dtype=np.int32)
        self.cards_played = np.zeros(shape, dtype=np.int32)
        self.game_over = np.zeros(shape, dtype=np.bool_)
        self.winners = np.full(shape, -1, dtype=np.int8)
        self.phase = np.full(shape, PHASE_CARD, dtype=np.int8)
        self.turn_started_with_one_card = np.zeros(shape, dtype=np.bool_)
        self.drawn_card = np.full(shape, -1, dtype=np.int8)
        self.public_events: list[list[tuple]] = [
            [] for _ in range(self.num_games)
        ]
        self.event_counts = {
            "draw": 0,
            "draw_retry": 0,
            "penalty_draw": 0,
            "reverse": 0,
            "skip": 0,
            "extra_turn": 0,
            "suit_change": 0,
            "duplicate_play": 0,
        }
        self._reset()

    def reset_games(
        self,
        policy_kinds: np.ndarray,
        actor_keys: np.ndarray,
        seeds: np.ndarray,
    ) -> None:
        """Reset an allocated engine for another same-shaped rollout."""
        policy_kinds = np.asarray(policy_kinds, dtype=np.int8)
        actor_keys = np.asarray(actor_keys, dtype=object)
        seeds = np.asarray(seeds, dtype=np.uint32)
        if policy_kinds.shape != self.policy_kinds.shape:
            raise ValueError("reused policy_kinds must preserve engine shape")
        if actor_keys.shape != self.actor_keys.shape:
            raise ValueError("reused actor_keys must preserve engine shape")
        if len(seeds) != self.num_games:
            raise ValueError("one deterministic seed is required per game")

        self.policy_kinds[...] = policy_kinds
        self.actor_keys[...] = actor_keys
        for rng, seed in zip(self.rngs, seeds):
            rng.seed(int(seed))
        self.hands.fill(-1)
        self.hand_counts.fill(0)
        self.hand_sizes.fill(0)
        self.draw_piles.fill(-1)
        self.draw_counts.fill(0)
        self.discards.fill(-1)
        self.discard_counts.fill(0)
        self.current_sort.fill(0)
        self.current_rank.fill(0)
        self.current_player.fill(0)
        self.turn_player.fill(0)
        self.direction.fill(1)
        self.penalty_amount.fill(0)
        self.sorts_played.fill(0)
        self.turn_counts.fill(0)
        self.cards_played.fill(0)
        self.game_over.fill(False)
        self.winners.fill(-1)
        self.phase.fill(PHASE_CARD)
        self.turn_started_with_one_card.fill(False)
        self.drawn_card.fill(-1)
        self.public_events = [[] for _ in range(self.num_games)]
        for key in self.event_counts:
            self.event_counts[key] = 0
        self._native_event_deltas.fill(0)
        self._reset()

    @property
    def allocated_bytes(self) -> int:
        arrays = (
            self.policy_kinds,
            self.hands,
            self.hand_counts,
            self.hand_sizes,
            self.draw_piles,
            self.draw_counts,
            self.discards,
            self.discard_counts,
            self.current_sort,
            self.current_rank,
            self.current_player,
            self.turn_player,
            self.direction,
            self.penalty_amount,
            self.sorts_played,
            self.turn_counts,
            self.cards_played,
            self.game_over,
            self.winners,
            self.phase,
            self.turn_started_with_one_card,
            self.drawn_card,
        )
        return sum(array.nbytes for array in arrays)

    def _reset(self) -> None:
        for game_index, rng in enumerate(self.rngs):
            shuffled = self.deck_template.copy()
            rng.shuffle(shuffled)
            self.draw_piles[game_index] = shuffled
            self.draw_counts[game_index] = len(shuffled)
            for player_index in range(self.num_players):
                for _ in range(7):
                    self._draw_card(game_index, player_index)
            start = self._take_draw_top(game_index)
            self._append_discard(game_index, start)
            while int(CARD_RANK[start]) in INVALID_START_RANKS:
                if self.draw_counts[game_index] == 0:
                    self._recycle_discard_into_draw(game_index)
                if self.draw_counts[game_index] == 0:
                    self.game_over[game_index] = True
                    break
                start = self._take_draw_top(game_index)
                self._append_discard(game_index, start)
            if self.game_over[game_index]:
                continue
            self.current_sort[game_index] = CARD_SORT[start]
            self.current_rank[game_index] = CARD_RANK[start]
            current = int(self.current_player[game_index])
            self.turn_started_with_one_card[game_index] = (
                self.hand_sizes[game_index, current] == 1
            )

    def active_games(self) -> np.ndarray:
        return np.flatnonzero(
            ~self.game_over & (self.turn_counts < self.max_turns)
        )

    def _append_hand(self, game_index: int, player_index: int, number: int) -> None:
        size = int(self.hand_sizes[game_index, player_index])
        self.hands[game_index, player_index, size] = number
        self.hand_sizes[game_index, player_index] = size + 1
        self.hand_counts[game_index, player_index, number] += 1

    def _remove_hand(self, game_index: int, player_index: int, number: int) -> None:
        size = int(self.hand_sizes[game_index, player_index])
        hand = self.hands[game_index, player_index]
        try:
            offset = hand[:size].tolist().index(number)
        except ValueError:
            return
        if offset + 1 < size:
            hand[offset : size - 1] = hand[offset + 1 : size]
        hand[size - 1] = -1
        self.hand_sizes[game_index, player_index] = size - 1
        self.hand_counts[game_index, player_index, number] -= 1

    def _append_discard(self, game_index: int, number: int) -> None:
        count = int(self.discard_counts[game_index])
        self.discards[game_index, count] = number
        self.discard_counts[game_index] = count + 1

    def _take_draw_top(self, game_index: int) -> int:
        count = int(self.draw_counts[game_index])
        if count <= 0:
            raise RuntimeError("cannot draw from an empty pile")
        count -= 1
        number = int(self.draw_piles[game_index, count])
        self.draw_piles[game_index, count] = -1
        self.draw_counts[game_index] = count
        return number

    def _recycle_discard_into_draw(self, game_index: int) -> bool:
        """Rebuild the draw pile from the discard stack, preserving its top."""
        discard_count = int(self.discard_counts[game_index])
        if discard_count <= 1:
            return False
        recyclable = self.discards[game_index, : discard_count - 1].copy()
        self.rngs[game_index].shuffle(recyclable)
        recycle_count = len(recyclable)
        self.draw_piles[game_index].fill(-1)
        self.draw_piles[game_index, :recycle_count] = recyclable
        self.draw_counts[game_index] = recycle_count
        top = int(self.discards[game_index, discard_count - 1])
        self.discards[game_index].fill(-1)
        self.discards[game_index, 0] = top
        self.discard_counts[game_index] = 1
        return True

    def _draw_card(self, game_index: int, player_index: int) -> Optional[int]:
        if self.draw_counts[game_index] == 0:
            if not self._recycle_discard_into_draw(game_index):
                self.game_over[game_index] = True
                return None
        number = self._take_draw_top(game_index)
        self._append_hand(game_index, player_index, number)
        self.event_counts["draw"] += 1
        return number

    def observations(
        self,
        game_indices: np.ndarray,
        *,
        suit_modes: np.ndarray | bool = False,
    ) -> np.ndarray:
        """Build public views for many current players in one array pass."""
        game_indices = np.asarray(game_indices, dtype=np.int64)
        if game_indices.ndim != 1:
            raise ValueError("game_indices must be one-dimensional")
        if len(game_indices) == 0:
            return np.empty((0, OBSERVATION_SIZE), dtype=np.float32)
        if np.any((game_indices < 0) | (game_indices >= self.num_games)):
            raise IndexError("game index is out of range")

        observations = np.zeros(
            (len(game_indices), OBSERVATION_SIZE),
            dtype=np.float32,
        )
        return self.observations_into(
            game_indices,
            observations,
            suit_modes=suit_modes,
        )

    def observations_into(
        self,
        game_indices: np.ndarray,
        observations: np.ndarray,
        *,
        suit_modes: np.ndarray | bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Populate caller-owned observation and optional action-mask storage."""
        game_indices = np.asarray(game_indices, dtype=np.int64)
        if game_indices.ndim != 1:
            raise ValueError("game_indices must be one-dimensional")
        if np.any((game_indices < 0) | (game_indices >= self.num_games)):
            raise IndexError("game index is out of range")
        if (
            observations.shape != (len(game_indices), OBSERVATION_SIZE)
            or observations.dtype != np.float32
        ):
            raise ValueError(
                "observations must be a float32 array with shape "
                f"({len(game_indices)}, {OBSERVATION_SIZE})"
            )
        if action_masks is not None and (
            action_masks.shape != (len(game_indices), ACTION_COUNT)
            or action_masks.dtype != np.bool_
        ):
            raise ValueError(
                "action_masks must be a bool array with shape "
                f"({len(game_indices)}, {ACTION_COUNT})"
            )
        if len(game_indices) == 0:
            return observations
        if (
            self._observations_into_native is not None
            and action_masks is not None
        ):
            native_suit_modes = np.asarray(suit_modes, dtype=np.bool_)
            if native_suit_modes.ndim == 0:
                native_suit_modes = np.full(
                    len(game_indices),
                    bool(native_suit_modes),
                    dtype=np.bool_,
                )
            if native_suit_modes.shape == (len(game_indices),):
                self._observations_into_native(
                    game_indices,
                    native_suit_modes,
                    self.current_player,
                    self.hand_counts,
                    self.current_sort,
                    self.current_rank,
                    self.penalty_amount,
                    self.phase,
                    self.drawn_card,
                    self.sorts_played,
                    self.direction,
                    self.hand_sizes,
                    self.num_players,
                    observations,
                    action_masks,
                )
                return observations

        player_indices = self.current_player[game_indices].astype(
            np.int64,
            copy=False,
        )
        counts = self.hand_counts[game_indices, player_indices]
        observations[:, :HAND_END] = counts
        playable = CARD_COMPATIBILITY_MASKS[
            self.current_sort[game_indices],
            self.current_rank[game_indices],
        ]
        penalty_rows = self.penalty_amount[game_indices] > 0
        if np.any(penalty_rows):
            playable[penalty_rows] = CARD_RANK_MASKS[
                self.current_rank[game_indices[penalty_rows]]
            ]
        retry_rows = self.phase[game_indices] == PHASE_DRAW_RETRY
        if np.any(retry_rows):
            # Legacy Game permits only the card that was just drawn to be
            # played during the immediate retry.  Other compatible cards
            # already in hand are not executable in this phase.
            playable[retry_rows] = False
            retry_offsets = np.flatnonzero(retry_rows)
            drawn = self.drawn_card[game_indices[retry_rows]].astype(
                np.int64,
                copy=False,
            )
            valid = drawn >= 0
            playable[retry_offsets[valid], drawn[valid]] = True
        compatible_counts = counts * playable
        observations[:, COMPATIBLE_START:COMPATIBLE_END] = compatible_counts
        observations[
            :, SORTS_PLAYED_START : SORTS_PLAYED_START + 5
        ] = self.sorts_played[game_indices]
        observations[:, MODE_INDEX] = np.asarray(
            suit_modes,
            dtype=np.float32,
        )
        observations[
            :, PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_END
        ] = 0
        directions = self.direction[game_indices].astype(np.int64, copy=False)
        slots = min(self.num_players, PLAYER_HAND_SIZE_SLOTS)
        slot_offsets = np.arange(slots, dtype=np.int64)
        ordered_players = (
            player_indices[:, None]
            + directions[:, None] * slot_offsets[None, :]
        ) % self.num_players
        observations[
            :,
            PLAYER_HAND_SIZES_START : PLAYER_HAND_SIZES_START + slots,
        ] = self.hand_sizes[game_indices[:, None], ordered_players]
        observations[:, PLAYER_INDEX] = player_indices
        observations[:, DIRECTION_INDEX] = directions
        if action_masks is not None:
            action_masks.fill(False)
            action_masks[:, :HAND_END] = compatible_counts >= 1
            action_masks[:, PASS_ACTION] = True
            suit_rows = observations[:, MODE_INDEX] != 0
            if np.any(suit_rows):
                action_masks[suit_rows] = False
                action_masks[
                    suit_rows,
                    SUIT_ACTION_START:SUIT_ACTION_END,
                ] = True
        return observations

    def observation(
        self,
        game_index: int,
        *,
        suit_mode: bool = False,
    ) -> np.ndarray:
        """Return exactly the public 121-feature view of the current player."""
        return self.observations(
            np.asarray([game_index], dtype=np.int64),
            suit_modes=suit_mode,
        )[0]

    def rl_has_card_decisions(
        self,
        game_indices: np.ndarray,
    ) -> np.ndarray:
        """Return executable-card availability for many current RL seats."""
        game_indices = np.asarray(game_indices, dtype=np.int64)
        result = np.zeros(len(game_indices), dtype=np.bool_)
        if len(game_indices) == 0:
            return result
        if self._rl_has_card_decisions_native is not None:
            return self._rl_has_card_decisions_native(
                game_indices,
                self.current_player,
                self.hand_counts,
                self.current_sort,
                self.current_rank,
                self.penalty_amount,
                self.phase,
                self.drawn_card,
            )
        retry_rows = self.phase[game_indices] == PHASE_DRAW_RETRY
        result[retry_rows] = self.drawn_card[game_indices[retry_rows]] >= 0
        card_rows = ~retry_rows
        if not np.any(card_rows):
            return result
        card_games = game_indices[card_rows]
        players = self.current_player[card_games].astype(
            np.int64,
            copy=False,
        )
        playable = CARD_COMPATIBILITY_MASKS[
            self.current_sort[card_games],
            self.current_rank[card_games],
        ].copy()
        penalty_rows = self.penalty_amount[card_games] > 0
        if np.any(penalty_rows):
            playable[penalty_rows] = CARD_RANK_MASKS[
                self.current_rank[card_games[penalty_rows]]
            ]
        result[card_rows] = np.any(
            self.hand_counts[card_games, players] & playable,
            axis=1,
        )
        return result

    def rl_has_card_decision(self, game_index: int) -> bool:
        return bool(
            self.rl_has_card_decisions(
                np.asarray([game_index], dtype=np.int64)
            )[0]
        )

    def _ordered_hand(self, game_index: int, player_index: int) -> np.ndarray:
        size = int(self.hand_sizes[game_index, player_index])
        return self.hands[game_index, player_index, :size]

    def first_legal_action(self, game_index: int) -> int:
        player_index = int(self.current_player[game_index])
        compatible = CARD_COMPATIBILITY_MASKS[
            int(self.current_sort[game_index]),
            int(self.current_rank[game_index]),
        ]
        for number in self._ordered_hand(game_index, player_index):
            if compatible[int(number)]:
                return int(number)
        return PASS_ACTION

    def random_action(self, game_index: int) -> int:
        player_index = int(self.current_player[game_index])
        size = int(self.hand_sizes[game_index, player_index])
        hand = self.hands[game_index, player_index]
        shuffled = hand[:size]
        self.rngs[game_index].shuffle(shuffled)
        compatible = CARD_COMPATIBILITY_MASKS[
            int(self.current_sort[game_index]),
            int(self.current_rank[game_index]),
        ]
        for number in shuffled:
            if compatible[int(number)]:
                return int(number)
        return PASS_ACTION

    @staticmethod
    def _smart_stop_score(
        hand: np.ndarray,
        *,
        two_player_game_reference: bool,
    ) -> float:
        if two_player_game_reference and np.any(CARD_RANK[hand] == 8):
            return -1.9467897139799917
        suit_counts = np.bincount(CARD_SORT[hand], minlength=5)
        return float(np.max(suit_counts[:4]))

    def _smart_hand_value(self, hand: np.ndarray) -> float:
        return self._smart_stop_score(
            hand,
            two_player_game_reference=(
                self.smart_agents_have_game_reference and self.num_players == 2
            ),
        ) - 1.0290830868190015 * len(hand)

    @staticmethod
    def _smart_end_value(number: int, hand: np.ndarray) -> float:
        suit = CARD_SORT[number]
        same_suit = CARD_SORT[hand] == suit
        total = int(np.count_nonzero(same_suit))
        penalty = int(
            np.count_nonzero(
                same_suit
                & CARD_IS_PENALTY[hand]
            )
        )
        return 0.43244878288347055 * total + penalty

    def smart_action(self, game_index: int) -> int:
        player_index = int(self.current_player[game_index])
        hand = self._ordered_hand(game_index, player_index)
        if self._smart_action_native is not None:
            return int(
                self._smart_action_native(
                    hand,
                    int(self.current_sort[game_index]),
                    int(self.current_rank[game_index]),
                    (
                        self.smart_agents_have_game_reference
                        and self.num_players == 2
                    ),
                )
            )
        hand_ranks = CARD_RANK[hand]
        hand_suits = CARD_SORT[hand]
        current_sort = int(self.current_sort[game_index])
        current_rank = int(self.current_rank[game_index])
        compatible = CARD_COMPATIBILITY_MASKS[
            current_sort, current_rank
        ][hand]
        playable = hand[compatible]
        if len(playable) == 0:
            return PASS_ACTION

        playable_ranks = CARD_RANK[playable]
        candidates = playable
        if current_rank in PENALTY_RANKS and current_rank != 7:
            matching = playable[playable_ranks == current_rank]
            if len(matching):
                candidates = matching

        penalty_branch = (
            current_rank in PENALTY_RANKS
            and current_rank != 7
            and np.any(playable_ranks == current_rank)
        )
        sevens = candidates[CARD_RANK[candidates] == 7]
        if not penalty_branch:
            candidates = np.concatenate(
                (sevens, candidates[CARD_RANK[candidates] != 7])
            )

        # Every candidate evaluation removes exactly one card.  Compute the
        # sufficient statistics once instead of copying and deleting the hand
        # for each candidate.
        suit_counts = np.bincount(hand_suits, minlength=5)
        two_player_game_reference = (
            self.smart_agents_have_game_reference
            and self.num_players == 2
        )
        rank_eight_count = (
            int(np.count_nonzero(hand_ranks == 8))
            if two_player_game_reference
            else 0
        )
        remaining_length = len(hand) - 1

        seven_end_values: dict[int, float] = {}
        if len(sevens):
            penalty_mask = CARD_IS_PENALTY[hand]
            penalty_suit_counts = np.bincount(
                hand_suits[penalty_mask], minlength=5
            )
            for number in sevens:
                number_int = int(number)
                suit = int(CARD_SORT[number_int])
                total = int(suit_counts[suit] - 1)
                penalty = int(penalty_suit_counts[suit] - 1)
                seven_end_values[number_int] = (
                    0.43244878288347055 * total + penalty
                )
            max_end = max(seven_end_values.values())
        else:
            max_end = 0.0

        standard_suit_counts = suit_counts[:4]
        maximum_suit_count = int(np.max(standard_suit_counts))
        maximum_suit_ties = int(
            np.count_nonzero(standard_suit_counts == maximum_suit_count)
        )
        if maximum_suit_ties == 1:
            unique_maximum_suit = int(np.argmax(standard_suit_counts))
            second_suit_count = max(
                int(standard_suit_counts[(unique_maximum_suit + 1) % 4]),
                int(standard_suit_counts[(unique_maximum_suit + 2) % 4]),
                int(standard_suit_counts[(unique_maximum_suit + 3) % 4]),
            )
            decremented_unique_maximum = max(
                maximum_suit_count - 1, second_suit_count
            )
        else:
            unique_maximum_suit = -1
            decremented_unique_maximum = maximum_suit_count

        best_number = int(candidates[0])
        best_score = -float("inf")
        for number_value in candidates:
            number = int(number_value)
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
                if (
                    two_player_game_reference
                    and rank_eight_count - int(rank == 8) > 0
                ):
                    stop_score = -1.9467897139799917
                else:
                    removed_suit = int(CARD_SORT[number])
                    stop_score = float(
                        decremented_unique_maximum
                        if removed_suit == unique_maximum_suit
                        else maximum_suit_count
                    )
                score = (
                    stop_score
                    - 1.0290830868190015 * remaining_length
                )
                score += SMART_PENALTY_BONUS[rank]
            if score > best_score:
                best_score = score
                best_number = number
        return best_number

    def smart_actions(
        self,
        game_indices: np.ndarray,
    ) -> np.ndarray:
        """Choose exact Smart actions with one native dispatch per group."""
        game_indices = np.asarray(game_indices, dtype=np.int64)
        if self._smart_actions_native is not None:
            return self._smart_actions_native(
                self.hands,
                self.hand_sizes,
                self.current_player,
                self.current_sort,
                self.current_rank,
                game_indices,
                (
                    self.smart_agents_have_game_reference
                    and self.num_players == 2
                ),
            )
        return np.fromiter(
            (
                self.smart_action(int(game_index))
                for game_index in game_indices
            ),
            dtype=np.int64,
            count=len(game_indices),
        )

    @staticmethod
    def _smarter_hand_value(hand: np.ndarray) -> float:
        """Exact array equivalent of ``SmarterAgent.hand_value``."""
        if len(hand) == 0:
            return 0.0
        standard_suits = CARD_SORT[hand]
        suit_counts = np.bincount(standard_suits, minlength=5)[:4]
        # Python ``max(dict, key=dict.get)`` and NumPy argmax both resolve a
        # tie to the first standard suit.
        max_suit = int(np.argmax(suit_counts))
        ranks = CARD_RANK[hand]
        value = 0.5 * int(np.count_nonzero(standard_suits == max_suit))
        value += sum(SMARTER_SPECIAL_BONUS[int(rank)] for rank in ranks)
        return value - len(hand)

    def smarter_action(self, game_index: int) -> int:
        """Match ``smartAgent.SmarterAgent`` with a live public Game reference."""
        player_index = int(self.current_player[game_index])
        hand = self._ordered_hand(game_index, player_index)
        compatible = CARD_COMPATIBILITY_MASKS[
            int(self.current_sort[game_index]),
            int(self.current_rank[game_index]),
        ]
        playable = hand[compatible[hand]]
        if len(playable) == 0:
            return PASS_ACTION

        best_number = int(playable[0])
        best_score = -float("inf")
        direction = int(self.direction[game_index])
        next_player = (player_index + direction) % self.num_players
        next_count = int(self.hand_sizes[game_index, next_player])
        public_sorts = self.sorts_played[game_index]
        for number_value in playable:
            number = int(number_value)
            # Canonical ``Card.__eq__`` makes the list comprehension remove
            # every duplicate of the candidate number, not just one instance.
            remaining = hand[hand != number]
            score = self._smarter_hand_value(remaining)
            rank = int(CARD_RANK[number])
            if rank == 7:
                score += 2.0
            elif rank == 8:
                score += 5.0 / (next_count + 1)
            elif rank == 1:
                score += 0.5
            elif rank == 0:
                score += 1.0
            elif rank == 2:
                score += 0.4
            elif rank == 13:
                best_suit = int(np.argmax(public_sorts[:4]))
                # Preserve the canonical left-associative expression:
                # ``base_score + suit_change_bonus + depletion * weight``.
                # Grouping the final two terms can change a last-bit tie and
                # therefore select a different first-max candidate.
                score = (
                    score
                    + 1.0
                    + public_sorts[best_suit] * 0.1
                )
            else:
                score = (
                    score
                    + public_sorts[int(CARD_SORT[number])] * 0.1
                )
            if score > best_score:
                best_score = score
                best_number = number
        return best_number

    def choose_suit(self, game_index: int, policy_kind: int) -> int:
        if policy_kind == POLICY_FIRST_LEGAL:
            return 0
        if policy_kind == POLICY_RANDOM:
            return int(self.rngs[game_index].randint(0, 4))
        if policy_kind == POLICY_SMARTER:
            return int(np.argmax(self.sorts_played[game_index, :4]))
        if policy_kind != POLICY_SMART:
            raise ValueError("RL suit choices must come from the actor")
        player_index = int(self.current_player[game_index])
        hand = self._ordered_hand(game_index, player_index)
        best_suit = 0
        best_score = -float("inf")
        suits = CARD_SORT[hand]
        suit_counts = np.bincount(suits, minlength=5)
        penalty_suit_counts = np.bincount(
            suits[CARD_IS_PENALTY[hand]], minlength=5
        )
        for suit in range(4):
            total = int(suit_counts[suit])
            penalty = int(penalty_suit_counts[suit])
            score = (
                0.43244878288347055 * total
                - (-1.8463831782446556) * penalty
            )
            if score > best_score:
                best_score = score
                best_suit = suit
        return best_suit

    def apply_action(self, game_index: int, action: int) -> None:
        if self.game_over[game_index]:
            return
        phase = int(self.phase[game_index])
        if phase == PHASE_SUIT:
            suit = int(action) - SUIT_ACTION_START
            if not 0 <= suit < 4:
                raise ValueError(f"invalid suit action {action}")
            self.current_sort[game_index] = suit
            if self.record_public_events:
                self.public_events[game_index].append(
                    ("choose_sort", int(self.current_player[game_index]), suit)
                )
            self.event_counts["suit_change"] += 1
            self._finish_turn(game_index)
            return
        if phase == PHASE_DRAW_RETRY:
            drawn = int(self.drawn_card[game_index])
            player_index = int(self.current_player[game_index])
            if (
                self.policy_kinds[game_index, player_index] == POLICY_RL
                and action not in (drawn, PASS_ACTION)
            ):
                raise ValueError(
                    "draw-retry action must be the drawn card or pass"
                )
            if action == drawn:
                self._process_played_card(
                    game_index,
                    drawn,
                    is_last_card=False,
                )
            else:
                self._finish_turn(game_index)
            return

        if action != PASS_ACTION:
            if not 0 <= action < len(CARD_RANK):
                raise ValueError(f"invalid card action {action}")
            player_index = int(self.current_player[game_index])
            if self.hand_counts[game_index, player_index, action] <= 0:
                raise ValueError(
                    "player cannot play a card that is not in hand"
                )

        if self.penalty_amount[game_index] > 0:
            if (
                action != PASS_ACTION
                and int(CARD_RANK[action]) == self.current_rank[game_index]
            ):
                self._process_played_card(
                    game_index,
                    action,
                    is_last_card=bool(
                        self.turn_started_with_one_card[game_index]
                    ),
                )
                return
            previous_rank = int(self.current_rank[game_index])
            self._apply_penalty(game_index)
            if previous_rank == 0 and not self.game_over[game_index]:
                self.phase[game_index] = PHASE_SUIT
            else:
                self._finish_turn(game_index)
            return

        self._process_played_card(
            game_index,
            action,
            is_last_card=bool(self.turn_started_with_one_card[game_index]),
        )

    def apply_actions(
        self,
        game_indices: np.ndarray,
        actions: np.ndarray,
    ) -> None:
        """Apply one action per independent game, batching the common case."""
        game_indices = np.asarray(game_indices, dtype=np.int64)
        actions = np.asarray(actions, dtype=np.int64)
        if game_indices.shape != actions.shape or game_indices.ndim != 1:
            raise ValueError("game indices and actions must be matching vectors")
        if len(game_indices) == 0:
            return
        if self._apply_simple_actions_native is None:
            for game_index, action in zip(game_indices, actions):
                self.apply_action(int(game_index), int(action))
            return

        handled = np.empty(len(game_indices), dtype=np.bool_)
        acting_players = np.empty(len(game_indices), dtype=np.int8)
        public_event_kinds = np.empty(len(game_indices), dtype=np.int8)
        self._native_event_deltas.fill(0)
        self._apply_simple_actions_native(
            game_indices,
            actions,
            self.num_players,
            CARD_SORT,
            CARD_RANK,
            CARD_IS_PENALTY,
            CARD_COMPATIBILITY_MASKS,
            self.hands,
            self.hand_counts,
            self.hand_sizes,
            self.draw_piles,
            self.draw_counts,
            self.discards,
            self.discard_counts,
            self.current_sort,
            self.current_rank,
            self.current_player,
            self.turn_player,
            self.direction,
            self.penalty_amount,
            self.sorts_played,
            self.turn_counts,
            self.cards_played,
            self.game_over,
            self.winners,
            self.phase,
            self.turn_started_with_one_card,
            self.drawn_card,
            self._native_event_deltas,
            handled,
            acting_players,
            public_event_kinds,
        )
        for name, delta in zip(
            (
                "duplicate_play",
                "extra_turn",
                "skip",
                "reverse",
                "draw",
                "draw_retry",
                "suit_change",
                "penalty_draw",
            ),
            self._native_event_deltas,
        ):
            self.event_counts[name] += int(delta)
        if self.record_public_events:
            python_offsets = np.flatnonzero(
                (public_event_kinds != 0) | ~handled
            )
        else:
            python_offsets = np.flatnonzero(~handled)
        for offset in python_offsets:
            game_index = game_indices[offset]
            player_index = acting_players[offset]
            action = actions[offset]
            event_kind = public_event_kinds[offset]
            if not self.record_public_events:
                self.apply_action(int(game_index), int(action))
                continue
            if event_kind == 1:
                self.public_events[int(game_index)].append(
                    (
                        "play",
                        int(player_index),
                        int(action),
                    )
                )
            elif event_kind == 2:
                self.public_events[int(game_index)].append(
                    (
                        "choose_sort",
                        int(player_index),
                        int(action) - SUIT_ACTION_START,
                    )
                )
            else:
                self.apply_action(int(game_index), int(action))

    def _apply_penalty(self, game_index: int) -> None:
        amount = int(self.penalty_amount[game_index])
        player_index = int(self.current_player[game_index])
        for _ in range(amount):
            if self.game_over[game_index]:
                break
            self._draw_card(game_index, player_index)
            self.event_counts["penalty_draw"] += 1
        self.penalty_amount[game_index] = 0

    def _process_played_card(
        self,
        game_index: int,
        action: int,
        *,
        is_last_card: bool,
    ) -> None:
        self.cards_played[game_index] += 1
        if self.cards_played[game_index] > 500:
            self.game_over[game_index] = True
            self._finish_turn(game_index)
            return
        if action == PASS_ACTION:
            player_index = int(self.current_player[game_index])
            drawn = self._draw_card(game_index, player_index)
            if self.game_over[game_index] or drawn is None:
                self._finish_turn(game_index)
                return
            if _compatible(
                drawn,
                int(self.current_sort[game_index]),
                int(self.current_rank[game_index]),
            ):
                self.drawn_card[game_index] = drawn
                self.phase[game_index] = PHASE_DRAW_RETRY
                self.event_counts["draw_retry"] += 1
            else:
                self._finish_turn(game_index)
            return
        if not 0 <= action < len(CARD_RANK):
            raise ValueError(f"invalid card action {action}")
        player_index = int(self.current_player[game_index])
        if self.hand_counts[game_index, player_index, action] <= 0:
            raise ValueError("player cannot play a card that is not in hand")
        if not _compatible(
            action,
            int(self.current_sort[game_index]),
            int(self.current_rank[game_index]),
        ):
            self._finish_turn(game_index)
            return
        rank = int(CARD_RANK[action])
        if is_last_card and rank in LAST_CARD_PENALTY_RANKS:
            player_index = int(self.current_player[game_index])
            self._draw_card(game_index, player_index)
            if not self.game_over[game_index]:
                self._draw_card(game_index, player_index)
            self._finish_turn(game_index)
            return

        if self.hand_counts[game_index, player_index, action] > 1:
            self.event_counts["duplicate_play"] += 1
        self._append_discard(game_index, action)
        self.current_sort[game_index] = CARD_SORT[action]
        self.current_rank[game_index] = rank
        self.sorts_played[game_index, CARD_SORT[action]] += 1
        self._remove_hand(game_index, player_index, action)
        if self.record_public_events:
            self.public_events[game_index].append(
                ("play", player_index, action)
            )

        if rank == 7:
            self.current_player[game_index] = (
                self.current_player[game_index] - self.direction[game_index]
            ) % self.num_players
            self.event_counts["extra_turn"] += 1
        elif rank == 8:
            self.current_player[game_index] = (
                self.current_player[game_index] + self.direction[game_index]
            ) % self.num_players
            self.event_counts["skip"] += 1
        elif rank == 1:
            self.direction[game_index] *= -1
            self.event_counts["reverse"] += 1
        elif rank == 0:
            self.penalty_amount[game_index] += 5
        elif rank == 13:
            self.phase[game_index] = PHASE_SUIT
            return
        elif rank == 2:
            self.penalty_amount[game_index] += 2
        self._finish_turn(game_index)

    def _finish_turn(self, game_index: int) -> None:
        self.turn_counts[game_index] += 1
        acting_player = int(self.turn_player[game_index])
        if self.hand_sizes[game_index, acting_player] == 0:
            # For legal terminal plays current_player is unchanged because every
            # card that mutates it is also prohibited as a final card.
            self.winners[game_index] = int(self.current_player[game_index])
            self.game_over[game_index] = True
        if not self.game_over[game_index]:
            self.current_player[game_index] = (
                self.current_player[game_index] + self.direction[game_index]
            ) % self.num_players
        self.phase[game_index] = PHASE_CARD
        self.drawn_card[game_index] = -1
        if not self.game_over[game_index]:
            current = int(self.current_player[game_index])
            self.turn_player[game_index] = current
            self.turn_started_with_one_card[game_index] = (
                self.hand_sizes[game_index, current] == 1
            )

    def non_neural_action(self, game_index: int) -> int:
        player_index = int(self.current_player[game_index])
        kind = int(self.policy_kinds[game_index, player_index])
        if self.phase[game_index] == PHASE_SUIT:
            return SUIT_ACTION_START + self.choose_suit(game_index, kind)
        if kind == POLICY_FIRST_LEGAL:
            return self.first_legal_action(game_index)
        if kind == POLICY_RANDOM:
            return self.random_action(game_index)
        if kind == POLICY_SMART:
            return self.smart_action(game_index)
        if kind == POLICY_SMARTER:
            return self.smarter_action(game_index)
        raise ValueError("an RL action requires batched actor inference")

    def non_neural_actions(
        self,
        game_indices: np.ndarray,
    ) -> np.ndarray:
        """Choose non-neural actions while batching the native Smart policy."""
        game_indices = np.asarray(game_indices, dtype=np.int64)
        actions = np.empty(len(game_indices), dtype=np.int64)
        if len(game_indices) == 0:
            return actions
        players = self.current_player[game_indices].astype(
            np.int64,
            copy=False,
        )
        kinds = self.policy_kinds[game_indices, players]
        all_card_phase = np.all(
            self.phase[game_indices] != PHASE_SUIT
        )
        if all_card_phase and np.all(kinds == POLICY_SMART):
            return self.smart_actions(game_indices)
        suit_offsets = np.flatnonzero(
            self.phase[game_indices] == PHASE_SUIT
        )
        for offset in suit_offsets:
            actions[offset] = (
                SUIT_ACTION_START
                + self.choose_suit(
                    int(game_indices[offset]),
                    int(kinds[offset]),
                )
            )

        card_offsets = np.flatnonzero(
            self.phase[game_indices] != PHASE_SUIT
        )
        for kind in (
            POLICY_FIRST_LEGAL,
            POLICY_RANDOM,
            POLICY_SMART,
            POLICY_SMARTER,
        ):
            offsets = card_offsets[kinds[card_offsets] == kind]
            if not len(offsets):
                continue
            selected = game_indices[offsets]
            if kind == POLICY_SMART:
                actions[offsets] = self.smart_actions(selected)
                continue
            chooser = {
                POLICY_FIRST_LEGAL: self.first_legal_action,
                POLICY_RANDOM: self.random_action,
                POLICY_SMARTER: self.smarter_action,
            }[int(kind)]
            actions[offsets] = np.fromiter(
                (
                    chooser(int(game_index))
                    for game_index in selected
                ),
                dtype=np.int64,
                count=len(selected),
            )
        return actions

    def run_non_neural(self) -> None:
        while True:
            active = self.active_games()
            if len(active) == 0:
                return
            for game_index in active:
                self.apply_action(
                    int(game_index),
                    self.non_neural_action(int(game_index)),
                )


def _autocast_for(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def _action_masks(observations: torch.Tensor) -> torch.Tensor:
    shape = (*observations.shape[:-1], ACTION_COUNT)
    masks = torch.zeros(shape, dtype=torch.bool, device=observations.device)
    masks[..., : COMPATIBLE_END - COMPATIBLE_START] = (
        observations[..., COMPATIBLE_START:COMPATIBLE_END] >= 1
    )
    masks[..., PASS_ACTION] = True
    suit_mode = observations[..., MODE_INDEX] != 0
    if torch.any(suit_mode):
        masks[suit_mode] = False
        masks[suit_mode, SUIT_ACTION_START:SUIT_ACTION_END] = True
    return masks


def _action_masks_numpy(observations: np.ndarray) -> np.ndarray:
    """Build the same rollout masks without small-tensor Torch dispatch."""
    shape = (*observations.shape[:-1], ACTION_COUNT)
    masks = np.zeros(shape, dtype=np.bool_)
    masks[..., : COMPATIBLE_END - COMPATIBLE_START] = (
        observations[..., COMPATIBLE_START:COMPATIBLE_END] >= 1
    )
    masks[..., PASS_ACTION] = True
    suit_mode = observations[..., MODE_INDEX] != 0
    if np.any(suit_mode):
        masks[suit_mode] = False
        masks[suit_mode, SUIT_ACTION_START:SUIT_ACTION_END] = True
    return masks


def _sample_actions_and_log_probabilities(
    scaled_logits: torch.Tensor,
    *,
    greedy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match ``Categorical(logits=...).sample/log_prob`` without its wrappers."""
    normalized = scaled_logits - scaled_logits.logsumexp(
        dim=-1,
        keepdim=True,
    )
    if greedy:
        actions = torch.argmax(scaled_logits, dim=-1)
    else:
        probabilities = normalized.softmax(dim=-1)
        actions = torch.multinomial(
            probabilities,
            num_samples=1,
        ).squeeze(-1)
    log_probabilities = normalized.gather(
        dim=-1,
        index=actions.unsqueeze(-1),
    ).squeeze(-1)
    return actions, log_probabilities


def collect_batched_rollout(
    engine: BatchedPesten,
    actor_policies: Mapping[Hashable, ActorPolicy],
    *,
    output_device: torch.device,
    direct_action_sampling: bool = False,
    numpy_action_masks: bool = False,
    reuse_rollout_workspaces: bool = False,
    drain_non_neural: bool = False,
) -> RolloutResult:
    """Run all games, batching every simultaneously pending actor decision."""
    return collect_coordinated_batched_rollouts(
        [engine],
        actor_policies,
        output_device=output_device,
        direct_action_sampling=direct_action_sampling,
        numpy_action_masks=numpy_action_masks,
        reuse_rollout_workspaces=reuse_rollout_workspaces,
        drain_non_neural=drain_non_neural,
    )[0]


@dataclass
class _CoordinatedRolloutState:
    engine: BatchedPesten
    buffer: VectorRolloutBuffer
    finalized: np.ndarray
    inference_calls: int = 0


def collect_coordinated_batched_rollouts(
    engines: Sequence[BatchedPesten],
    actor_policies: Mapping[Hashable, ActorPolicy],
    *,
    output_device: torch.device,
    direct_action_sampling: bool = False,
    numpy_action_masks: bool = False,
    reuse_rollout_workspaces: bool = False,
    drain_non_neural: bool = False,
) -> list[RolloutResult]:
    """Advance independent engines together and share their actor calls."""
    if not engines:
        return []
    states = []
    for engine in engines:
        episode_slots = engine.num_games * engine.num_players
        buffer = VectorRolloutBuffer(
            episode_slots,
            initial_capacity=max(1024, engine.num_games * 64),
        )
        states.append(
            _CoordinatedRolloutState(
                engine=engine,
                buffer=buffer,
                finalized=np.zeros(engine.num_games, dtype=np.bool_),
            )
        )
    workspace_rows = sum(engine.num_games for engine in engines)
    observation_workspace = (
        np.empty(
            (workspace_rows, OBSERVATION_SIZE),
            dtype=np.float32,
        )
        if reuse_rollout_workspaces
        else None
    )
    mask_workspace = (
        np.empty((workspace_rows, ACTION_COUNT), dtype=np.bool_)
        if reuse_rollout_workspaces and numpy_action_masks
        else None
    )

    while True:
        grouped: dict[Hashable, list[tuple[int, np.ndarray]]] = {}
        any_active = False
        for state_index, state in enumerate(states):
            engine = state.engine
            unsettled = engine.active_games()
            if len(unsettled) == 0:
                continue
            any_active = True
            while len(unsettled):
                immediate_indices_list: list[int] = []
                immediate_actions_list: list[int | None] = []
                non_neural_indices: list[int] = []
                non_neural_offsets: list[int] = []
                requests: dict[Hashable, list[int]] = {}
                active_players = engine.current_player[unsettled].astype(
                    np.int64,
                    copy=False,
                )
                active_kinds = engine.policy_kinds[
                    unsettled,
                    active_players,
                ]
                has_card_decision = np.ones(
                    len(unsettled),
                    dtype=np.bool_,
                )
                rl_card_offsets = np.flatnonzero(
                    (active_kinds == POLICY_RL)
                    & (engine.phase[unsettled] != PHASE_SUIT)
                )
                if len(rl_card_offsets):
                    has_card_decision[rl_card_offsets] = (
                        engine.rl_has_card_decisions(
                            unsettled[rl_card_offsets]
                        )
                    )
                for offset, (game_value, kind_value) in enumerate(
                    zip(unsettled, active_kinds)
                ):
                    game_index = int(game_value)
                    player_index = int(
                        engine.current_player[game_index]
                    )
                    kind = int(kind_value)
                    if kind != POLICY_RL:
                        non_neural_offsets.append(
                            len(immediate_actions_list)
                        )
                        non_neural_indices.append(game_index)
                        immediate_indices_list.append(game_index)
                        immediate_actions_list.append(None)
                        continue
                    if not has_card_decision[offset]:
                        immediate_indices_list.append(game_index)
                        immediate_actions_list.append(PASS_ACTION)
                        continue
                    key = engine.actor_keys[game_index, player_index]
                    requests.setdefault(key, []).append(game_index)

                for key, game_indices in requests.items():
                    grouped.setdefault(key, []).append(
                        (
                            state_index,
                            np.asarray(game_indices, dtype=np.int64),
                        )
                    )
                if non_neural_indices:
                    selected_actions = engine.non_neural_actions(
                        np.asarray(non_neural_indices, dtype=np.int64)
                    )
                    for action_offset, action in zip(
                        non_neural_offsets,
                        selected_actions,
                    ):
                        immediate_actions_list[action_offset] = int(action)
                if not immediate_indices_list:
                    break
                immediate_indices = np.asarray(
                    immediate_indices_list,
                    dtype=np.int64,
                )
                engine.apply_actions(
                    immediate_indices,
                    np.asarray(immediate_actions_list, dtype=np.int64),
                )
                if not drain_non_neural or engine.num_players != 2:
                    break
                still_active = ~(
                    engine.game_over[immediate_indices]
                    | (
                        engine.turn_counts[immediate_indices]
                        >= engine.max_turns
                    )
                )
                unsettled = immediate_indices[still_active]

        if not any_active:
            break

        for key, segments in grouped.items():
            policy = actor_policies[key]
            request_count = sum(
                len(game_indices) for _, game_indices in segments
            )
            if observation_workspace is None:
                observation_parts = [
                    states[state_index].engine.observations(
                        game_indices,
                        suit_modes=(
                            states[state_index].engine.phase[game_indices]
                            == PHASE_SUIT
                        ),
                    )
                    for state_index, game_indices in segments
                ]
                observations_np = (
                    observation_parts[0]
                    if len(observation_parts) == 1
                    else np.concatenate(observation_parts)
                )
            else:
                observations_np = observation_workspace[:request_count]
                offset = 0
                for state_index, game_indices in segments:
                    engine = states[state_index].engine
                    end = offset + len(game_indices)
                    engine.observations_into(
                        game_indices,
                        observations_np[offset:end],
                        suit_modes=(
                            engine.phase[game_indices] == PHASE_SUIT
                        ),
                        action_masks=(
                            mask_workspace[offset:end]
                            if (
                                mask_workspace is not None
                                and policy.device.type == "cpu"
                            )
                            else None
                        ),
                    )
                    offset = end
            observations = torch.from_numpy(observations_np).to(
                device=policy.device
            )
            with (
                torch.inference_mode(),
                (
                    _autocast_for(policy.device)
                    if policy.autocast
                    else nullcontext()
                ),
            ):
                logits = policy.actor(observations)
                if numpy_action_masks and policy.device.type == "cpu":
                    masks = torch.from_numpy(
                        mask_workspace[:request_count]
                        if mask_workspace is not None
                        else _action_masks_numpy(observations_np)
                    )
                else:
                    masks = _action_masks(observations)
                masked = logits.masked_fill(
                    ~masks,
                    torch.finfo(logits.dtype).min,
                )
                scaled = masked / policy.temperature
                if direct_action_sampling:
                    actions, log_probabilities = (
                        _sample_actions_and_log_probabilities(
                            scaled,
                            greedy=policy.greedy,
                        )
                    )
                else:
                    distribution = Categorical(
                        logits=scaled,
                        validate_args=False,
                    )
                    actions = (
                        torch.argmax(scaled, dim=-1)
                        if policy.greedy
                        else distribution.sample()
                    )
                    log_probabilities = distribution.log_prob(actions)
            actions_np = actions.to(
                device="cpu",
                dtype=torch.int64,
            ).numpy()
            log_probabilities_np = log_probabilities.to(
                device="cpu",
                dtype=torch.float32,
            ).numpy()
            states[segments[0][0]].inference_calls += 1

            offset = 0
            for state_index, game_indices in segments:
                state = states[state_index]
                engine = state.engine
                end = offset + len(game_indices)
                segment_actions = actions_np[offset:end]
                if policy.collect_data:
                    player_indices = engine.current_player[
                        game_indices
                    ].astype(np.int64, copy=True)
                    episode_ids = (
                        game_indices * engine.num_players + player_indices
                    ).astype(np.int32)
                    rewards = np.where(
                        segment_actions == PASS_ACTION,
                        -0.1,
                        0.0,
                    ).astype(np.float32)
                    # A suit choice has zero shaping reward even though action
                    # 54 is impossible in this mode.
                    rewards[
                        observations_np[offset:end, MODE_INDEX] != 0
                    ] = 0.0
                    state.buffer.append(
                        observations_np[offset:end],
                        segment_actions,
                        log_probabilities_np[offset:end],
                        rewards,
                        episode_ids,
                    )
                engine.apply_actions(game_indices, segment_actions)
                offset = end

        for state in states:
            engine = state.engine
            newly_finished = np.flatnonzero(
                ~state.finalized
                & (
                    engine.game_over
                    | (engine.turn_counts >= engine.max_turns)
                )
            )
            for game_index in newly_finished:
                collect_seats = []
                terminal_rewards = []
                winner = int(engine.winners[game_index])
                for player_index in range(engine.num_players):
                    if (
                        engine.policy_kinds[game_index, player_index]
                        != POLICY_RL
                    ):
                        continue
                    key = engine.actor_keys[game_index, player_index]
                    if not actor_policies[key].collect_data:
                        continue
                    collect_seats.append(
                        int(game_index) * engine.num_players + player_index
                    )
                    terminal_rewards.append(
                        0.0
                        if winner < 0
                        else (
                            1.0 if winner == player_index else -1.0
                        )
                    )
                state.buffer.add_terminal_rewards(
                    np.asarray(collect_seats, dtype=np.int32),
                    np.asarray(terminal_rewards, dtype=np.float32),
                )
            state.finalized[newly_finished] = True

    results = []
    coordinated_peak_bytes = sum(
        state.buffer.peak_bytes + state.engine.allocated_bytes
        for state in states
    )
    for state_index, state in enumerate(states):
        engine = state.engine
        (
            observations,
            actions,
            log_probabilities,
            rewards,
            episode_lengths,
        ) = state.buffer.finalize(device=output_device)
        completed_games = int(np.count_nonzero(state.finalized))
        wins = int(np.count_nonzero(engine.winners >= 0))
        results.append(
            RolloutResult(
                observations=observations,
                actions=actions,
                log_probabilities=log_probabilities,
                rewards=rewards,
                episode_lengths=episode_lengths,
                environment_turns=int(engine.turn_counts.sum()),
                completed_games=completed_games,
                policy_decisions=state.buffer.size,
                inference_calls=state.inference_calls,
                peak_buffer_bytes=(
                    coordinated_peak_bytes if state_index == 0 else 0
                ),
                wins=wins,
                draws=completed_games - wins,
            )
        )
    return results
