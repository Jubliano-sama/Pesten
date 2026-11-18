"""Suit-equivariant card probabilities conditioned on the observer's hand and public board."""

import numpy as np

from batched_pesten import CARD_RANK, CARD_SORT

try:
    from numba import njit
except ImportError:

    def njit(*, cache):
        return lambda function: function


# ace, two, seven, eight, free choice, joker; all other ranks are interchangeable.
RANK_CATEGORY = np.array((6, 1, 2, 0, 0, 0, 0, 3, 4, 0, 0, 0, 0, 5), dtype=np.int8)
CATEGORY = RANK_CATEGORY[CARD_RANK]
TOP_TYPE = np.minimum(RANK_CATEGORY, 5)
SIZE_LIMITS = np.array((1, 2, 4, 7))
FEATURES = 96  # Width of the serialized coefficient table.
FAMILIES = 7


@njit(cache=True)
def size_bin(size):
    return np.searchsorted(SIZE_LIMITS, size)


@njit(cache=True)
def card_features(own, sizes, rel, suit, rank, penalty):
    """Seven categorical factors, all preserved by ordinary-rank and suit relabelings."""
    idx = np.zeros((50, FAMILIES), np.uint8)
    by_suit = np.zeros(4, np.int16)
    by_rank = np.zeros(12, np.int16)
    for c in range(48):
        by_suit[CARD_SORT[c]] += int(own[c])
        by_rank[CARD_RANK[c] - 1] += int(own[c])
    previous = int(rel == len(sizes) - 1)
    top_type = TOP_TYPE[rank]
    small = 0 if sizes[0] <= 2 else 1 if sizes[0] <= 5 else 2
    near = int(sizes[(rel + 1) % len(sizes)] <= 2)
    # Index zero contributes no log odds. Other indices partition the seven
    # categorical factors into disjoint ranges of the coefficient table.
    for c in range(50):
        cat = CATEGORY[c]
        idx[c, 0] = 1 + cat
        if c < 48:
            cs, cr = CARD_SORT[c], CARD_RANK[c]
            if cs == suit:
                idx[c, 1] = 8 + 6 * previous + top_type
            if cr == rank:
                idx[c, 2] = 20 + 6 * previous + top_type
            idx[c, 3] = 32 + 5 * int(cat != 0) + min(4, by_suit[cs])
            idx[c, 4] = 42 + 4 * int(cat != 0) + min(3, by_rank[cr - 1])
        idx[c, 5] = 50 + 7 * small + cat
        idx[c, 6] = 71 + 7 * near + cat
    return idx


@njit(cache=True)
def apply_card_belief(weights, own, sizes, rel, suit, rank, penalty, coefficients):
    """Update the two hypothesized styles' odds for each remaining physical card."""
    feature = card_features(own, sizes, rel, suit, rank, penalty)
    bucket = size_bin(sizes[rel])
    for role in range(2):
        params = coefficients[role, len(sizes) - 2, bucket]
        for card in range(50):
            log_ratio = 0.0
            for j in range(FAMILIES):
                log_ratio += params[feature[card, j]]
            weights[role, rel, card] *= np.exp(min(2.5, max(-2.5, log_ratio)))
