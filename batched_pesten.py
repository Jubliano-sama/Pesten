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
from typing import Hashable, Mapping, Optional

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
        self.smart_agents_have_game_reference = smart_agents_have_game_reference
        self.rngs = [np.random.RandomState(int(seed)) for seed in seeds]

        shape = (self.num_games,)
        player_shape = (self.num_games, self.num_players)
        self.hands = np.full((*player_shape, len(STANDARD_DECK)), -1, dtype=np.int8)
        self.hand_counts = np.zeros((*player_shape, 50), dtype=np.uint8)
        self.hand_sizes = np.zeros(player_shape, dtype=np.uint8)
        self.draw_piles = np.full(
            (self.num_games, len(STANDARD_DECK)),
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
            shuffled = STANDARD_DECK.copy()
            rng.shuffle(shuffled)
            self.draw_piles[game_index] = shuffled
            self.draw_counts[game_index] = len(shuffled)
            for player_index in range(self.num_players):
                for _ in range(7):
                    self._draw_card(game_index, player_index)
            start = self._take_draw_top(game_index)
            self._append_discard(game_index, start)
            while int(CARD_RANK[start]) in INVALID_START_RANKS:
                start = self._take_draw_top(game_index)
                self._append_discard(game_index, start)
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
        matches = np.flatnonzero(hand[:size] == number)
        if len(matches) == 0:
            return
        offset = int(matches[0])
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

    def _draw_card(self, game_index: int, player_index: int) -> Optional[int]:
        if self.draw_counts[game_index] == 0:
            discard_count = int(self.discard_counts[game_index])
            if discard_count <= 1:
                self.game_over[game_index] = True
                return None
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
        number = self._take_draw_top(game_index)
        self._append_hand(game_index, player_index, number)
        self.event_counts["draw"] += 1
        return number

    def observation(self, game_index: int, *, suit_mode: bool = False) -> np.ndarray:
        """Return exactly the public 121-feature view of the current player."""
        player_index = int(self.current_player[game_index])
        observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        counts = self.hand_counts[game_index, player_index]
        observation[:HAND_END] = counts
        if self.phase[game_index] == PHASE_DRAW_RETRY:
            # Legacy Game permits only the card that was just drawn to be
            # played during the immediate retry.  Other compatible cards
            # already in hand are not executable in this phase.
            playable = np.zeros(len(CARD_RANK), dtype=np.bool_)
            drawn = int(self.drawn_card[game_index])
            if drawn >= 0:
                playable[drawn] = True
        elif self.penalty_amount[game_index] > 0:
            playable = CARD_RANK == self.current_rank[game_index]
        else:
            playable = (
                (CARD_RANK == self.current_rank[game_index])
                | (CARD_RANK == 0)
                | (CARD_RANK == 13)
                | (CARD_SORT == self.current_sort[game_index])
            )
        observation[COMPATIBLE_START:COMPATIBLE_END] = counts * playable
        observation[SORTS_PLAYED_START : SORTS_PLAYED_START + 5] = (
            self.sorts_played[game_index]
        )
        observation[MODE_INDEX] = 1.0 if suit_mode else 0.0
        direction = int(self.direction[game_index])
        for slot in range(min(self.num_players, PLAYER_HAND_SIZE_SLOTS)):
            ordered = (player_index + slot * direction) % self.num_players
            observation[PLAYER_HAND_SIZES_START + slot] = self.hand_sizes[
                game_index, ordered
            ]
        observation[PLAYER_INDEX] = player_index
        observation[DIRECTION_INDEX] = direction
        return observation

    def rl_has_card_decision(self, game_index: int) -> bool:
        if self.phase[game_index] == PHASE_DRAW_RETRY:
            return int(self.drawn_card[game_index]) >= 0
        player_index = int(self.current_player[game_index])
        counts = self.hand_counts[game_index, player_index]
        if self.penalty_amount[game_index] > 0:
            playable = CARD_RANK == self.current_rank[game_index]
        else:
            playable = (
                (CARD_RANK == self.current_rank[game_index])
                | (CARD_RANK == 0)
                | (CARD_RANK == 13)
                | (CARD_SORT == self.current_sort[game_index])
            )
        return bool(np.any(counts & playable))

    def _ordered_hand(self, game_index: int, player_index: int) -> np.ndarray:
        size = int(self.hand_sizes[game_index, player_index])
        return self.hands[game_index, player_index, :size]

    def first_legal_action(self, game_index: int) -> int:
        player_index = int(self.current_player[game_index])
        for number in self._ordered_hand(game_index, player_index):
            if _compatible(
                int(number),
                int(self.current_sort[game_index]),
                int(self.current_rank[game_index]),
            ):
                return int(number)
        return PASS_ACTION

    def random_action(self, game_index: int) -> int:
        player_index = int(self.current_player[game_index])
        size = int(self.hand_sizes[game_index, player_index])
        hand = self.hands[game_index, player_index]
        shuffled = hand[:size].copy()
        self.rngs[game_index].shuffle(shuffled)
        hand[:size] = shuffled
        for number in shuffled:
            if _compatible(
                int(number),
                int(self.current_sort[game_index]),
                int(self.current_rank[game_index]),
            ):
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
                & np.isin(CARD_RANK[hand], tuple(PENALTY_RANKS))
            )
        )
        return 0.43244878288347055 * total + penalty

    def smart_action(self, game_index: int) -> int:
        player_index = int(self.current_player[game_index])
        hand = self._ordered_hand(game_index, player_index)
        compatible = np.asarray(
            [
                _compatible(
                    int(number),
                    int(self.current_sort[game_index]),
                    int(self.current_rank[game_index]),
                )
                for number in hand
            ],
            dtype=np.bool_,
        )
        playable = hand[compatible]
        if len(playable) == 0:
            return PASS_ACTION

        current_rank = int(self.current_rank[game_index])
        candidates = playable
        if current_rank in PENALTY_RANKS and current_rank != 7:
            matching = playable[CARD_RANK[playable] == current_rank]
            if len(matching):
                candidates = matching

        penalty_bonuses = {
            8: -1.5328572573442552,
            1: 0.04122020478114498,
            0: -0.03556864491834867,
            13: -3.0043124890687474,
            2: 0.04665718005466267,
        }
        sevens = candidates[CARD_RANK[candidates] == 7]
        if not (
            current_rank in PENALTY_RANKS
            and current_rank != 7
            and np.any(CARD_RANK[playable] == current_rank)
        ):
            candidates = np.concatenate(
                (sevens, candidates[CARD_RANK[candidates] != 7])
            )
        seven_end_values: dict[int, float] = {}
        if len(sevens):
            for number in sevens:
                remaining = hand.copy()
                offset = int(np.flatnonzero(remaining == number)[0])
                remaining = np.delete(remaining, offset)
                seven_end_values[int(number)] = self._smart_end_value(
                    int(number), remaining
                )
            max_end = max(seven_end_values.values())
        else:
            max_end = 0.0

        best_number = int(candidates[0])
        best_score = -float("inf")
        penalty_branch = (
            current_rank in PENALTY_RANKS
            and current_rank != 7
            and np.any(CARD_RANK[playable] == current_rank)
        )
        for number_value in candidates:
            number = int(number_value)
            remaining = hand.copy()
            offset = int(np.flatnonzero(remaining == number)[0])
            remaining = np.delete(remaining, offset)
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
                score = self._smart_hand_value(remaining)
                score += penalty_bonuses.get(rank, 0.0)
            if score > best_score:
                best_score = score
                best_number = number
        return best_number

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
        special_bonus = {
            7: 2.0,
            8: 1.5,
            1: 1.0,
            0: 1.5,
            13: 1.0,
            2: 1.0,
        }
        value = 0.5 * int(np.count_nonzero(standard_suits == max_suit))
        value += sum(special_bonus.get(int(rank), 0.0) for rank in ranks)
        return value - len(hand)

    def smarter_action(self, game_index: int) -> int:
        """Match ``smartAgent.SmarterAgent`` with a live public Game reference."""
        player_index = int(self.current_player[game_index])
        hand = self._ordered_hand(game_index, player_index)
        playable = np.asarray(
            [
                number
                for number in hand
                if _compatible(
                    int(number),
                    int(self.current_sort[game_index]),
                    int(self.current_rank[game_index]),
                )
            ],
            dtype=np.int8,
        )
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
        ranks = CARD_RANK[hand]
        suits = CARD_SORT[hand]
        for suit in range(4):
            same_suit = suits == suit
            total = int(np.count_nonzero(same_suit))
            penalty = int(
                np.count_nonzero(
                    same_suit & np.isin(ranks, tuple(PENALTY_RANKS))
                )
            )
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


def collect_batched_rollout(
    engine: BatchedPesten,
    actor_policies: Mapping[Hashable, ActorPolicy],
    *,
    output_device: torch.device,
) -> RolloutResult:
    """Run all games, batching every simultaneously pending actor decision."""
    episode_slots = engine.num_games * engine.num_players
    buffer = VectorRolloutBuffer(
        episode_slots,
        initial_capacity=max(1024, engine.num_games * 64),
    )
    finalized = np.zeros(engine.num_games, dtype=np.bool_)
    inference_calls = 0

    while True:
        active = engine.active_games()
        if len(active) == 0:
            break
        grouped: dict[Hashable, list[tuple[int, np.ndarray]]] = {}
        immediate: list[tuple[int, int]] = []
        for game_value in active:
            game_index = int(game_value)
            player_index = int(engine.current_player[game_index])
            kind = int(engine.policy_kinds[game_index, player_index])
            if kind != POLICY_RL:
                immediate.append(
                    (game_index, engine.non_neural_action(game_index))
                )
                continue
            if (
                engine.phase[game_index] != PHASE_SUIT
                and not engine.rl_has_card_decision(game_index)
            ):
                immediate.append((game_index, PASS_ACTION))
                continue
            key = engine.actor_keys[game_index, player_index]
            observation = engine.observation(
                game_index,
                suit_mode=engine.phase[game_index] == PHASE_SUIT,
            )
            grouped.setdefault(key, []).append((game_index, observation))

        for game_index, action in immediate:
            engine.apply_action(game_index, action)

        for key, requests in grouped.items():
            policy = actor_policies[key]
            observations_np = np.stack(
                [request[1] for request in requests],
                axis=0,
            )
            observations = torch.from_numpy(observations_np).to(
                device=policy.device
            )
            with torch.inference_mode(), _autocast_for(policy.device):
                logits = policy.actor(observations)
                masks = _action_masks(observations)
                masked = logits.masked_fill(
                    ~masks,
                    torch.finfo(logits.dtype).min,
                )
                scaled = masked / policy.temperature
                distribution = Categorical(
                    logits=scaled,
                    validate_args=False,
                )
                if policy.greedy:
                    actions = torch.argmax(scaled, dim=-1)
                else:
                    actions = distribution.sample()
                log_probabilities = distribution.log_prob(actions)
            actions_np = actions.to(
                device="cpu",
                dtype=torch.int64,
            ).numpy()
            log_probabilities_np = log_probabilities.to(
                device="cpu",
                dtype=torch.float32,
            ).numpy()
            inference_calls += 1

            if policy.collect_data:
                game_indices = np.asarray(
                    [request[0] for request in requests],
                    dtype=np.int64,
                )
                player_indices = engine.current_player[game_indices].astype(
                    np.int64,
                    copy=True,
                )
                episode_ids = (
                    game_indices * engine.num_players + player_indices
                ).astype(np.int32)
                rewards = np.where(
                    actions_np == PASS_ACTION,
                    -0.1,
                    0.0,
                ).astype(np.float32)
                # A suit choice has zero shaping reward even though action 54 is
                # impossible in this mode.
                rewards[observations_np[:, MODE_INDEX] != 0] = 0.0
                buffer.append(
                    observations_np,
                    actions_np,
                    log_probabilities_np,
                    rewards,
                    episode_ids,
                )

            for (game_index, _), action in zip(requests, actions_np):
                engine.apply_action(game_index, int(action))

        newly_finished = np.flatnonzero(
            ~finalized
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
                if engine.policy_kinds[game_index, player_index] != POLICY_RL:
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
                    else (1.0 if winner == player_index else -1.0)
                )
            buffer.add_terminal_rewards(
                np.asarray(collect_seats, dtype=np.int32),
                np.asarray(terminal_rewards, dtype=np.float32),
            )
        finalized[newly_finished] = True

    observations, actions, log_probabilities, rewards, episode_lengths = (
        buffer.finalize(device=output_device)
    )
    completed_games = int(np.count_nonzero(finalized))
    wins = int(np.count_nonzero(engine.winners >= 0))
    return RolloutResult(
        observations=observations,
        actions=actions,
        log_probabilities=log_probabilities,
        rewards=rewards,
        episode_lengths=episode_lengths,
        environment_turns=int(engine.turn_counts.sum()),
        completed_games=completed_games,
        policy_decisions=buffer.size,
        inference_calls=inference_calls,
        peak_buffer_bytes=buffer.peak_bytes + engine.allocated_bytes,
        wins=wins,
        draws=completed_games - wins,
    )
