"""Outcome-trained leaf evaluation for possible worlds in public Pesten search."""

import math

import numpy as np

from batched_pesten import CARD_IS_PENALTY, CARD_RANK, CARD_SORT

try:
    from numba import njit
except ImportError:
    def njit(*, cache):
        return lambda function: function


FEATURES = 50
HIDDEN = 24
# Linear skip, hidden matrix (feature-major), hidden bias, hidden-to-score,
# shared score bias, shared draw logit.
WEIGHT_COUNT = FEATURES + FEATURES * HIDDEN + HIDDEN + HIDDEN + 2


@njit(cache=True)
def _position_features(hands, sizes, seat, direction, suit, rank, penalty, random_others):
    """Suit/rank-symmetric summaries for each seat in a sampled world."""
    players = len(sizes)
    f = np.empty((players, FEATURES), np.float32)
    suits = np.zeros((players, 5), np.int16)
    ranks = np.zeros((players, 14), np.int16)
    tempo_suits = np.zeros(players, np.int8)
    for i in range(players):
        for j in range(int(sizes[i])):
            c = int(hands[i, j])
            s, r = CARD_SORT[c], CARD_RANK[c]
            suits[i, s] += 1
            ranks[i, r] += 1
            if r == 7 or (players == 2 and r == 8):
                tempo_suits[i] |= 1 << s
    for i in range(players):
        size = int(sizes[i])
        turn = (i - seat) * direction % players
        prev_seat, next_seat = (i - direction) % players, (i + direction) % players
        ordinary = 0
        distinct_ranks = 0
        playable = 0
        playable_ordinary = 0
        playable_tempo = 0
        tempo = 0
        tempo_links = 0
        ordinary_from_tempo = 0
        only_ordinary = -1
        for j in range(size):
            c = int(hands[i, j])
            r, s = CARD_RANK[c], CARD_SORT[c]
            is_plain = not CARD_IS_PENALTY[c]
            is_tempo = r == 7 or (players == 2 and r == 8)
            legal = r == rank if penalty > 0 else (s == suit or r == rank or c >= 48)
            ordinary += is_plain
            playable += legal
            playable_ordinary += legal and is_plain
            playable_tempo += legal and is_tempo
            tempo += is_tempo
            if is_tempo:
                tempo_links += suits[i, s] > 1
            elif is_plain:
                only_ordinary = c
                ordinary_from_tempo += (tempo_suits[i] & (1 << s)) != 0
        for r in (3, 4, 5, 6, 9, 10, 11, 12):
            distinct_ranks += ranks[i, r] > 0
        n_suits, max_suit, suit_pairs, only_suit = 0, 0, 0, 0
        for s in range(4):
            count = int(suits[i, s])
            n_suits += count > 0
            max_suit = max(max_suit, count)
            suit_pairs += count * (count - 1) // 2
            if suits[next_seat, s] == 0:
                only_suit += count
        next_size = int(sizes[next_seat])
        prev_size = int(sizes[prev_seat])
        low = 99
        for p in range(players):
            if p != i:
                low = min(low, int(sizes[p]))
        row = f[i]
        row[0] = min(size, 30) / 8.
        row[1] = math.log1p(size) / 2.
        row[2] = 1. / max(size, 1)
        row[3] = size == 1
        row[4] = size == 2
        row[5] = size == 3
        row[6] = ordinary / 6.
        row[7] = ordinary == 0
        row[8] = ordinary == 1
        row[9] = ranks[i, 0] / 2.
        row[10] = ranks[i, 13] / 2.
        row[11] = ranks[i, 2] / 2.
        row[12] = ranks[i, 1] / 2.
        row[13] = ranks[i, 7] / 2.
        row[14] = ranks[i, 8] / 2.
        row[15] = n_suits / 4.
        row[16] = max_suit / 5.
        row[17] = distinct_ranks / 5.
        row[18] = (ordinary - distinct_ranks) / 2.
        row[19] = suit_pairs / min(50, max(3, size))
        row[20] = playable / 5.
        row[21] = playable_ordinary / 4.
        row[22] = playable > 0
        row[23] = playable_tempo > 0
        row[24] = playable_ordinary > 0 and size == 1
        row[25] = tempo / 3.
        row[26] = tempo_links / 3.
        row[27] = ordinary_from_tempo / 3.
        row[28] = ordinary == 1 and size == 2 and ordinary_from_tempo > 0 and playable_tempo > 0
        row[29] = tempo == size - 1 and ordinary_from_tempo > 0
        row[30] = turn / float(players)
        row[31] = turn == 0
        row[32] = turn == 1
        row[33] = min(8, penalty) / 5. if turn == 0 else 0.
        row[34] = ranks[i, rank] > 0 and penalty > 0 and turn == 0
        row[35] = players == 2
        row[36] = players == 3
        row[37] = min(next_size, 15) / 6.
        row[38] = 1. / max(next_size, 1)
        row[39] = min(prev_size, 15) / 6.
        row[40] = 1. / max(low, 1)
        row[41] = (random_others and i != 0)
        row[42] = random_others
        row[43] = only_suit / max(size, 1)
        row[44] = ranks[i, 0] > 0 and ranks[next_seat, 0] == 0
        row[45] = ranks[i, 2] > 0 and ranks[next_seat, 2] == 0
        row[46] = suits[i, suit] / max(1, size) if suit < 4 else 0.
        if only_ordinary >= 0 and ordinary == 1:
            s = CARD_SORT[only_ordinary]
            r = CARD_RANK[only_ordinary]
            row[47] = (suits[prev_seat, s] + ranks[prev_seat, r]) / max(1, prev_size)
            row[48] = suits[i, s] - 1
            row[49] = ranks[i, 13] > 0
        else:
            row[47] = row[48] = row[49] = 0.
    return f


@njit(cache=True)
def _leaf_value(hands, sizes, seat, direction, suit, rank, penalty, random_opponents, weights):
    """Probability that seat zero wins from this simulated (never actual) world."""
    feat = _position_features(hands, sizes, seat, direction, suit, rank, penalty, random_opponents)
    # Accumulate contiguous hidden channels so LLVM can SIMD this small network.
    logits = np.empty(len(sizes) + 1)
    off = FEATURES + FEATURES * HIDDEN
    for p in range(len(sizes)):
        hidden = weights[off:off+HIDDEN].copy()
        score = weights[-2]
        for i in range(FEATURES):
            value = feat[p, i]
            if value != 0:
                score += value * weights[i]
                for j in range(HIDDEN):
                    hidden[j] += value * weights[FEATURES + i * HIDDEN + j]
        for j in range(HIDDEN):
            if hidden[j] > 0:
                score += hidden[j] * weights[off + HIDDEN + j]
        logits[p] = score
    logits[len(sizes)] = weights[-1]
    largest = logits.max()
    denom = 0.
    for logit in logits:
        denom += math.exp(logit - largest)
    return math.exp(logits[0] - largest) / denom


@njit(cache=True)
def batch_features(hands, sizes, seat, direction, suit, rank, penalty, random_others):
    output = np.zeros((len(sizes), 4, FEATURES), np.float32)
    for row in range(len(sizes)):
        output[row, :len(sizes[row])] = _position_features(
            hands[row], sizes[row], seat[row], direction[row], suit[row], rank[row],
            penalty[row], random_others[row])
    return output
