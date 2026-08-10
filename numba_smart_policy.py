"""Optional native kernel for the exact batched Smart-agent card policy."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised on minimal installations.
    njit = None


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
CARD_IS_PENALTY = np.isin(
    CARD_RANK,
    np.asarray((7, 8, 1, 0, 13, 2), dtype=np.int8),
)
SMART_PENALTY_BONUS = np.asarray(
    (
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
    ),
    dtype=np.float64,
)
PASS_ACTION = 54


def _smart_action_kernel(
    hand: np.ndarray,
    current_sort: int,
    current_rank: int,
    two_player_game_reference: bool,
) -> int:
    """Return the exact first-max action used by ``smartAgent.Agent``."""
    hand_size = len(hand)
    playable = np.empty(hand_size, dtype=np.int8)
    playable_count = 0
    for offset in range(hand_size):
        number = int(hand[offset])
        rank = int(CARD_RANK[number])
        if (
            rank == current_rank
            or rank == 0
            or rank == 13
            or int(CARD_SORT[number]) == current_sort
        ):
            playable[playable_count] = number
            playable_count += 1
    if playable_count == 0:
        return PASS_ACTION

    penalty_branch = False
    if (
        current_rank in (7, 8, 1, 0, 13, 2)
        and current_rank != 7
    ):
        for offset in range(playable_count):
            if int(CARD_RANK[int(playable[offset])]) == current_rank:
                penalty_branch = True
                break

    candidates = np.empty(playable_count, dtype=np.int8)
    candidate_count = 0
    if penalty_branch:
        for offset in range(playable_count):
            number = int(playable[offset])
            if int(CARD_RANK[number]) == current_rank:
                candidates[candidate_count] = number
                candidate_count += 1
    else:
        for offset in range(playable_count):
            number = int(playable[offset])
            if int(CARD_RANK[number]) == 7:
                candidates[candidate_count] = number
                candidate_count += 1
        for offset in range(playable_count):
            number = int(playable[offset])
            if int(CARD_RANK[number]) != 7:
                candidates[candidate_count] = number
                candidate_count += 1

    suit_counts = np.zeros(5, dtype=np.int64)
    penalty_suit_counts = np.zeros(5, dtype=np.int64)
    rank_eight_count = 0
    for offset in range(hand_size):
        number = int(hand[offset])
        suit = int(CARD_SORT[number])
        rank = int(CARD_RANK[number])
        suit_counts[suit] += 1
        if bool(CARD_IS_PENALTY[number]):
            penalty_suit_counts[suit] += 1
        if two_player_game_reference and rank == 8:
            rank_eight_count += 1

    seven_count = 0
    seven_end_values = np.zeros(50, dtype=np.float64)
    max_end = -np.inf
    for offset in range(candidate_count):
        number = int(candidates[offset])
        if int(CARD_RANK[number]) != 7:
            continue
        seven_count += 1
        suit = int(CARD_SORT[number])
        end_value = (
            0.43244878288347055 * int(suit_counts[suit] - 1)
            + int(penalty_suit_counts[suit] - 1)
        )
        seven_end_values[number] = end_value
        if end_value > max_end:
            max_end = end_value
    if seven_count == 0:
        max_end = 0.0

    maximum_suit_count = int(suit_counts[0])
    unique_maximum_suit = 0
    maximum_suit_ties = 1
    for suit in range(1, 4):
        count = int(suit_counts[suit])
        if count > maximum_suit_count:
            maximum_suit_count = count
            unique_maximum_suit = suit
            maximum_suit_ties = 1
        elif count == maximum_suit_count:
            maximum_suit_ties += 1
    if maximum_suit_ties == 1:
        second_suit_count = 0
        for suit in range(4):
            if suit != unique_maximum_suit:
                second_suit_count = max(
                    second_suit_count,
                    int(suit_counts[suit]),
                )
        decremented_unique_maximum = max(
            maximum_suit_count - 1,
            second_suit_count,
        )
    else:
        unique_maximum_suit = -1
        decremented_unique_maximum = maximum_suit_count

    remaining_length = hand_size - 1
    best_number = int(candidates[0])
    best_score = -np.inf
    for offset in range(candidate_count):
        number = int(candidates[offset])
        rank = int(CARD_RANK[number])
        if rank == 7:
            if penalty_branch:
                score = 1.3119838792608858 + (
                    max_end - seven_end_values[number]
                )
            else:
                score = (
                    1.3119838792608858
                    * seven_count
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


_smart_action_compiled = (
    njit(cache=True)(_smart_action_kernel)
    if njit is not None
    else None
)


def _smart_actions_kernel(
    hands: np.ndarray,
    hand_sizes: np.ndarray,
    current_players: np.ndarray,
    current_sorts: np.ndarray,
    current_ranks: np.ndarray,
    game_indices: np.ndarray,
    two_player_game_reference: bool,
) -> np.ndarray:
    """Apply the exact Smart policy to many independent current seats."""
    actions = np.empty(len(game_indices), dtype=np.int64)
    for offset in range(len(game_indices)):
        game_index = int(game_indices[offset])
        player_index = int(current_players[game_index])
        hand_size = int(hand_sizes[game_index, player_index])
        actions[offset] = _smart_action_compiled(
            hands[game_index, player_index, :hand_size],
            int(current_sorts[game_index]),
            int(current_ranks[game_index]),
            two_player_game_reference,
        )
    return actions


if njit is None:
    smart_action_native = None
    smart_actions_native = None
else:
    smart_action_native = _smart_action_compiled
    smart_actions_native = njit(cache=True)(_smart_actions_kernel)


def warm_native_smart_policy() -> bool:
    """Compile or load both scalar and batched policy signatures."""
    if smart_action_native is None or smart_actions_native is None:
        return False
    hands = np.zeros((1, 2, 8), dtype=np.int8)
    hand_sizes = np.ones((1, 2), dtype=np.uint8)
    current_players = np.zeros(1, dtype=np.int8)
    current_sorts = np.zeros(1, dtype=np.int8)
    current_ranks = np.ones(1, dtype=np.int8)
    game_indices = np.zeros(1, dtype=np.int64)
    smart_action_native(hands[0, 0, :1], 0, 1, False)
    smart_actions_native(
        hands,
        hand_sizes,
        current_players,
        current_sorts,
        current_ranks,
        game_indices,
        False,
    )
    return True
