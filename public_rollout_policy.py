"""Small action ranker distilled from the frozen recurrent teacher for simulated continuations."""

import numpy as np
from batched_pesten import CARD_SORT, CARD_RANK, CARD_IS_PENALTY

try:
    from numba import njit
except ImportError:

    def njit(*, cache):
        return lambda function: function


F = 24
C = 8


@njit(cache=True)
def features(
    feature,
    card,
    suit_counts,
    ordinary_counts,
    tempo,
    ranks,
    size,
    rank,
    suit,
    penalty,
    next_size,
    players,
    ordinary,
    wildcards,
):
    category_by_rank = (6, 1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 5)
    category = 7 if card == 54 else category_by_rank[CARD_RANK[card]]
    suited = card < 48
    card_rank = CARD_RANK[card] if card < 50 else -1
    card_suit = CARD_SORT[card] if suited else 0
    suit_after = suit_counts[card_suit] - 1 if suited else 0
    ordinary_left = ordinary - (category == 0)
    feature[0] = 1.0
    feature[1] = min(size, 24) / 8.0
    feature[2] = 1.0 / size
    feature[3] = size == 1
    feature[4] = size == 2
    feature[5] = size == 3
    feature[6] = 1.0 / max(1, next_size)
    feature[7] = next_size == 1
    feature[8] = min(4.0, penalty / 2)
    feature[9] = ordinary / 7
    feature[10] = ordinary_left == 0
    feature[11] = ordinary_left == 1
    feature[12] = wildcards / 3
    feature[13] = suit_after / 5.0
    feature[14] = suited and suit_after == 0
    feature[15] = suited and suit_after == 1
    feature[16] = (
        (ordinary_counts[card_suit] - (category == 0)) / 4.0 if suited else 0.0
    )
    feature[17] = suited and ordinary_counts[card_suit] > (category == 0)
    feature[18] = (
        (tempo[card_suit] - (category == 3 or (category == 4 and players == 2)))
        if suited
        else 0.0
    )
    feature[19] = ranks[card_rank] - 1 if suited else 0.0
    feature[20] = suited and card_suit == suit
    feature[21] = card_rank == rank
    feature[22] = suit_counts.max() / 7
    feature[23] = (size - ordinary) / size
    return category


@njit(cache=True)
def summarize(hand, size, players):
    """Count suits, finishing cards, repeat-turn cards, and ranks in one hand."""
    suit_counts = np.zeros(4, np.int8)
    ordinary_counts = np.zeros(4, np.int8)
    tempo = np.zeros(4, np.int8)
    ranks = np.zeros(14, np.int8)
    ordinary = wildcards = 0
    for index in range(size):
        card = hand[index]
        card_rank = CARD_RANK[card]
        card_suit = CARD_SORT[card]
        if card < 48:
            suit_counts[card_suit] += 1
            tempo[card_suit] += card_rank == 7 or (card_rank == 8 and players == 2)
            ordinary_counts[card_suit] += not CARD_IS_PENALTY[card]
        ranks[card_rank] += 1
        ordinary += not CARD_IS_PENALTY[card]
        wildcards += card >= 48
    return suit_counts, ordinary_counts, tempo, ranks, ordinary, wildcards


@njit(cache=True)
def choose_proxy(hand, size, suit, rank, penalty, next_size, players, seed, weights):
    """Return a hand index or -1 to draw, using the fitted action logit gaps."""
    legal_count = 0
    for index in range(size):
        card = hand[index]
        legal_count += (
            CARD_RANK[card] == rank
            if penalty
            else (card >= 48 or CARD_SORT[card] == suit or CARD_RANK[card] == rank)
        )
    if legal_count == 0:
        return -1
    suit_counts, ordinary_counts, tempo, ranks, ordinary, wildcards = summarize(
        hand, size, players
    )
    feature = np.empty(F, np.float64)
    best = -1
    best_value = -1.0e20
    for index in range(-1, size):
        card = 54 if index < 0 else hand[index]
        if card < 50 and not (
            CARD_RANK[card] == rank
            if penalty
            else (card >= 48 or CARD_SORT[card] == suit or CARD_RANK[card] == rank)
        ):
            continue
        category = features(
            feature,
            card,
            suit_counts,
            ordinary_counts,
            tempo,
            ranks,
            size,
            rank,
            suit,
            penalty,
            next_size,
            players,
            ordinary,
            wildcards,
        )
        value = 0.0
        for column in range(F):
            value += feature[column] * weights[players - 2, category, column]
        seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
        value += 0.45 * ((seed >> 8) / 16777216.0)
        if value > best_value:
            best_value = value
            best = index
    return best
