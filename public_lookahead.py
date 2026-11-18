"""Model-based policy improvement from a player's own cards and public tokens.

For every candidate card (or suit), sample the same possible deals and play
each forward. Blend the resulting win estimates with the recurrent policy's
logits. The real game's legal mask, including the draw retry restriction, is
applied by the ordinary temporal player. Each possible deal uses a random or
strong continuation. The latter can use a fast action policy fitted to the
frozen RL agent. A private deterministic generator is seeded from public
observations and the player's hand.
"""
from __future__ import annotations

import json
import math
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import nn

from batched_pesten import CARD_IS_PENALTY, CARD_RANK, CARD_SORT
from public_card_belief import FEATURES as CARD_BELIEF_FEATURES, apply_card_belief
from public_lookahead_leaf import WEIGHT_COUNT, _leaf_value
from public_rollout_policy import C as ROLLOUT_CATEGORIES, F as ROLLOUT_FEATURES, choose_proxy

try:
    from numba import njit
except ImportError:  # Same implementation also runs with plain NumPy.
    def njit(*, cache):
        return lambda function: function


RECENT_PLAYS = 24
MAX_PLAYERS = 4
# One clockwise-relative opponent row holds four suit log weights, a wild
# weight, and twelve rank weights. A short per-decision buffer describes the
# chronological public plays, declarations and inferred draw counts.
BELIEF_WIDTH = 17
EVIDENCE = RECENT_PLAYS
PENDING = EVIDENCE + MAX_PLAYERS * BELIEF_WIDTH
LAST_SUIT = PENDING + MAX_PLAYERS * 3
LAST_RANK = LAST_SUIT + 1
LAST_ACTOR = LAST_SUIT + 2
AMBIGUOUS = LAST_SUIT + 3
LAST_CARD = LAST_SUIT + 4
DECLARATION = LAST_CARD + MAX_PLAYERS
LONG_RECENT = DECLARATION + MAX_PLAYERS
PILE_SIZE = LONG_RECENT + 54
LAST_CAPACITY = PILE_SIZE + 1
LAST_PLAY_COUNT = PILE_SIZE + 2
HISTORY_SIZE = PILE_SIZE + 3
EMPTY_WEIGHTS = np.empty(0, dtype=np.float64)
EMPTY_ROLLOUT = np.empty((0, 0, 0), dtype=np.float64)
EMPTY_CARD_BELIEF = np.empty((0, 0, 0, 0), dtype=np.float64)
CATEGORY = np.asarray([1 if r == 1 else 2 if r == 2 else 3 if r == 7 else
                       4 if r == 8 else 5 if r == 13 else 6 if r == 0 else 0
                       for r in CARD_RANK], dtype=np.int8)



@njit(cache=True)
def _next(state):
    return (1664525 * state + 1013904223) & 0xffffffff


@njit(cache=True)
def _shuffle(cards, size, state):
    for index in range(size - 1, 0, -1):
        state = _next(state)
        other = (state * (index + 1)) >> 32
        cards[index], cards[other] = cards[other], cards[index]
    return state


@njit(cache=True)
def _compatible(card_id, suit, rank):
    return card_id >= 48 or CARD_SORT[card_id] == suit or CARD_RANK[card_id] == rank


@njit(cache=True)
def _best_suit(hand, size):
    suit_values = np.zeros(4)
    for index in range(size):
        card_id = hand[index]
        if card_id < 48:
            suit_values[CARD_SORT[card_id]] += 1.0 + .25 * (not CARD_IS_PENALTY[card_id])
    return np.argmax(suit_values)


@njit(cache=True)
def _choose(hand, size, suit, rank, penalty, next_size, players, random_player, seed):
    """Choose a card index, or -1 to draw. Reads only one hand."""
    best, best_value, matches = -1, -1.e20, 0
    ordinary = 0
    suit_counts = np.zeros(4, np.int8)
    if not random_player:
        for i in range(size):
            card_id = hand[i]
            ordinary += not CARD_IS_PENALTY[card_id]
            if card_id < 48:
                suit_counts[CARD_SORT[card_id]] += 1
    for i in range(size):
        card_id = hand[i]
        card_rank = CARD_RANK[card_id]
        if not _compatible(card_id, suit, rank):
            continue
        seed = _next(seed)
        if random_player:
            matches += 1
            if (seed * matches) >> 32 == 0:
                best = i
            continue
        if penalty and card_rank != rank:
            continue
        if size == 1 and CARD_IS_PENALTY[card_id] and penalty <= 2:
            continue
        card_suit = CARD_SORT[card_id]
        value = .7 * ((seed >> 8) / 16777216.0)
        if card_suit < 4:
            value += .5 * (suit_counts[card_suit] - 1)
        if card_rank == 7 or (players == 2 and card_rank == 8):
            value += 2.3 if suit_counts[card_suit] > 1 else -.8
        if card_rank == 8 and players > 2:
            value += .7 + 1.5 * (next_size <= 2)
        if card_rank == 2 or card_rank == 0:
            value += 1.4 + (1.5 if players == 2 else .2) + 2.0 * (next_size <= 2)
        if card_rank == 13:
            value -= .5
        if ordinary == 1 and not CARD_IS_PENALTY[card_id] and size > 1:
            value -= 1. + 8. / (size - 1)
        if value > best_value:
            best = i
            best_value = value
    return best


@njit(cache=True)
def _draw(hand, size, stock, stock_size, discard, discard_size, seed):
    if stock_size == 0 and discard_size > 1:
        stock_size = discard_size - 1
        stock[:stock_size] = discard[:stock_size]
        discard[0] = discard[discard_size - 1]
        discard_size = 1
        seed = _shuffle(stock, stock_size, seed)
    if stock_size == 0:
        return -1, stock_size, discard_size, seed
    stock_size -= 1
    hand[size] = stock[stock_size]
    return int(hand[size]), stock_size, discard_size, seed


@njit(cache=True)
def _rollout(initial_hands, initial_sizes, initial_stock, initial_discard,
             initial_stock_size, initial_discard_size, suit, rank, penalty, action,
             random_opponents, seed, horizon, leaf_weights=EMPTY_WEIGHTS,
             rollout_weights=EMPTY_ROLLOUT):
    """Score one action in one sampled possible world, relative seat zero."""
    hands, sizes = initial_hands.copy(), initial_sizes.copy()
    stock, discard = initial_stock.copy(), initial_discard.copy()
    stock_size, discard_size = initial_stock_size, initial_discard_size
    players, seat, direction = len(sizes), 0, 1
    for turn in range(horizon):
        seed = _next(seed)
        before = sizes[seat]
        hand = hands[seat]
        next_seat = (seat + direction) % players
        if turn == 0 and 50 <= action < 54:
            suit = action - 50
            seat = next_seat
            continue
        random_player = random_opponents and seat != 0
        if turn == 0:
            choice = -1
            for i in range(before):
                if hand[i] == action:
                    choice = i
                    break
        elif rollout_weights.size and not random_player:
            choice = choose_proxy(hand, before, suit, rank, penalty, sizes[next_seat],
                                  players, seed, rollout_weights)
        else:
            choice = _choose(hand, before, suit, rank, penalty, sizes[next_seat],
                             players, random_player, seed)
        if penalty and (choice < 0 or CARD_RANK[hand[choice]] != rank):
            for i in range(penalty):
                drawn, stock_size, discard_size, seed = _draw(
                    hand, sizes[seat], stock, stock_size, discard, discard_size, seed)
                if drawn < 0:
                    return 0.
                sizes[seat] += 1
            if rank == 0:
                suit = (seed >> 30) if random_player else _best_suit(hand, sizes[seat])
            penalty = 0
            seat = next_seat
            continue
        after_draw = False
        if choice < 0:
            drawn, stock_size, discard_size, seed = _draw(
                hand, before, stock, stock_size, discard, discard_size, seed)
            if drawn < 0:
                return 0.
            sizes[seat] += 1
            if not _compatible(drawn, suit, rank):
                seat = next_seat
                continue
            if random_player:
                retry = _choose(hand, sizes[seat], suit, rank, 0, sizes[next_seat],
                                players, True, _next(seed))
                if retry < 0 or hand[retry] != drawn:
                    seat = next_seat
                    continue
            elif before == 1 and CARD_IS_PENALTY[hand[0]] and not CARD_IS_PENALTY[drawn]:
                # Keep an ordinary card available to finish.
                seat = next_seat
                continue
            choice = before
            after_draw = True
        played = hand[choice]
        if before == 1 and not after_draw and CARD_IS_PENALTY[played]:
            for i in range(2):
                drawn, stock_size, discard_size, seed = _draw(
                    hand, sizes[seat], stock, stock_size, discard, discard_size, seed)
                if drawn < 0:
                    return 0.
                sizes[seat] += 1
            seat = next_seat
            continue
        sizes[seat] -= 1
        hand[choice] = hand[sizes[seat]]
        if sizes[seat] == 0:
            return 1. if seat == 0 else 0.
        discard[discard_size] = played
        discard_size += 1
        rank, suit = CARD_RANK[played], CARD_SORT[played]
        if rank == 1:
            direction = -direction
        elif rank == 2:
            penalty += 2
        elif rank == 0:
            penalty += 5
        elif rank == 13:
            suit = (seed >> 30) if random_player else _best_suit(hand, sizes[seat])
        if rank != 7:
            seat = (seat + direction * (2 if rank == 8 else 1)) % players

    if leaf_weights.size:
        return _leaf_value(hands, sizes, seat, direction, suit, rank, penalty,
                           random_opponents, leaf_weights)
    weights = np.empty(players)
    for i in range(players):
        counts = 0
        for index in range(sizes[i]):
            counts += not CARD_IS_PENALTY[hands[i, index]]
        effective_size = sizes[i] + 2.5 * (counts == 0) - .6 * (i == seat)
        weights[i] = math.exp(-.4 * effective_size)
    return weights[0] / weights.sum()


@njit(cache=True)
def _advance_tracking(tokens, previous, infer_voids, track_pile=False):
    """Track public discard identities and soft evidence from reveals and draws."""
    result = previous.copy()
    for i in range(len(tokens)):
        token, history = tokens[i], result[i]
        if token[99] == 0:
            kind = int(round(token[94] * 3))
            seat = int(round(token[95] * 12))
            if kind == 1:
                card_id = int(round(token[96] * 50)) - 1
                for k in range(RECENT_PLAYS - 1):
                    history[k] = history[k + 1]
                history[RECENT_PLAYS - 1] = card_id + 1
                if track_pile:
                    for k in range(LONG_RECENT, PILE_SIZE - 1):
                        history[k] = history[k + 1]
                    history[PILE_SIZE - 1] = card_id + 1
            if not infer_voids or seat < 0 or seat >= MAX_PLAYERS:
                continue
            pending = PENDING + 3 * seat
            if kind == 1 or kind == 3:
                if seat + 1 < history[LAST_ACTOR]:
                    history[AMBIGUOUS] = 1
                history[LAST_ACTOR] = seat + 1
            if kind == 1:
                history[pending] = card_id + 1
                if CARD_RANK[card_id] in (1, 7, 8):
                    history[AMBIGUOUS] = 1
            elif kind == 3:
                history[pending + 1] = round(token[97] * 5)
            elif kind == 2:
                history[pending + 2] += round(token[98] * 54)
            continue

        n = int(token[92])
        if track_pile and 2 <= n <= MAX_PLAYERS:
            capacity = 54 - int(token[69:69+n].sum())
            public_plays = int(token[82:87].sum())
            size = int(history[PILE_SIZE])
            if size == 0:
                # Opening burns are hidden. Two cards approximate the ordinary
                # opener plus a possible special before it.
                size = min(capacity, 2 + public_plays)
                if public_plays == 0:
                    s, r = np.argmax(token[50:55]), np.argmax(token[55:69])
                    if s < 4 and 1 <= r <= 12:
                        history[PILE_SIZE - 1] = 12 * s + r
            else:
                played = public_plays - int(history[LAST_PLAY_COUNT])
                stock = int(history[LAST_CAPACITY]) - size
                draws = int(history[LAST_CAPACITY]) - capacity + played
                if draws > stock:
                    # A recycle preserves the top card. Approximate how many
                    # of this interval's visible plays happened afterwards.
                    fraction = (draws - stock - 1) / max(1, draws)
                    size = max(1, min(capacity, 1 + round(fraction * played)))
                else:
                    size = min(capacity, size + played)
            history[PILE_SIZE] = size
            history[LAST_CAPACITY] = capacity
            history[LAST_PLAY_COUNT] = public_plays
        if not infer_voids or not 2 <= n <= MAX_PLAYERS:
            continue
        # Events in this batch use the direction at the current decision.
        # Persistent rows use clockwise order around the observing player.
        direction = int(token[89])
        top_suit, top_rank = int(history[LAST_SUIT]) - 1, int(history[LAST_RANK]) - 1
        for rel in range(n):
            pending = PENDING + 3 * rel
            played = int(history[pending]) - 1
            declared = int(history[pending + 1]) - 1
            draws = int(history[pending + 2])
            slot = EVIDENCE + ((rel * direction) % n) * BELIEF_WIDTH
            size = max(1, int(token[69 + rel]))
            if played >= 0:
                history[LAST_CARD + ((rel * direction) % n)] = played + 1
                history[DECLARATION + ((rel * direction) % n)] = 0
            if declared >= 0:
                history[DECLARATION + ((rel * direction) % n)] = declared + 1
            if rel > 0:
                dilution = max(.3, 1. - draws / size) * (.91 if played >= 0 else 1.)
                history[slot:slot + BELIEF_WIDTH] *= dilution
                if declared >= 0:
                    history[slot + declared] = min(2., history[slot + declared] + 1.1)
                if draws == 1 and played < 0:
                    # In two-player games the unchanged pile identifies the
                    # opponent's pass even after a reverse. At larger tables
                    # infer only when visible plays followed ordinary seat order.
                    if n == 2:
                        top_suit = np.argmax(token[50:55])
                        top_rank = np.argmax(token[55:69])
                    if ((n == 2 or not history[AMBIGUOUS]) and 0 <= top_suit < 4
                            and 1 <= top_rank <= 12):
                        # A new draw can match, and a player can voluntarily pass.
                        strength = .5 + 1.2 * (1. - 1. / size)
                        history[slot + top_suit] = -strength
                        history[slot + 4] = -strength
                        history[slot + 4 + top_rank] = -strength
            if played >= 0:
                top_suit, top_rank = CARD_SORT[played], CARD_RANK[played]
            if declared >= 0:
                top_suit = declared
        history[PENDING:LAST_SUIT] = 0
        history[LAST_SUIT] = np.argmax(token[50:55]) + 1
        history[LAST_RANK] = np.argmax(token[55:69]) + 1
        history[LAST_ACTOR] = 0
        history[AMBIGUOUS] = 0
    return result


@njit(cache=True)
def _sample_informed(pool, pool_size, sizes, hands, stock, weights, seed, first):
    """Sample without replacement, rotating allocation order across opponents."""
    available = pool_size
    for offset in range(len(sizes) - 1):
        seat = 1 + (first + offset) % (len(sizes) - 1)
        for j in range(sizes[seat]):
            total = 0.
            for k in range(available):
                total += weights[seat, pool[k]]
            seed = _next(seed)
            threshold = total * (seed / 4294967296.)
            index = available - 1
            for k in range(available):
                threshold -= weights[seat, pool[k]]
                if threshold < 0.:
                    index = k
                    break
            hands[seat, j] = pool[index]
            available -= 1
            pool[index], pool[available] = pool[available], pool[index]
    stock[:available] = pool[:available]
    return available, _shuffle(stock, available, seed)


@njit(cache=True)
def _action_values(token, history, samples, horizon, infer_voids,
                   leaf_weights=EMPTY_WEIGHTS, posterior=EMPTY_WEIGHTS, random_probability=.5,
                   rollout_weights=EMPTY_ROLLOUT, card_belief=EMPTY_CARD_BELIEF):
    values = np.zeros(55)
    players = int(token[92])
    if token[99] != 1 or token[98] != 0 or players < 2 or players > MAX_PLAYERS:
        return values
    suit = np.argmax(token[50:55])
    rank = np.argmax(token[55:69])
    penalty = int(token[90])
    sizes = token[69:69 + players].astype(np.int64)
    hands = np.zeros((players, 54), np.int8)
    stock = np.zeros(54, np.int8)
    discard = np.zeros(54, np.int8)
    pool = np.zeros(54, np.int8)
    remaining = np.ones(50, np.int8)
    remaining[48], remaining[49] = 4, 2
    choices = np.empty(54, np.int8)
    candidates, hand_size, seed = 0, 0, 10299847
    for card_id in range(50):
        held = int(token[card_id])
        if held:
            seed = _next(seed ^ ((held * 55 + card_id) * 129))
            hands[0, hand_size:hand_size + held] = card_id
            hand_size += held
            remaining[card_id] -= held
            if (CARD_RANK[card_id] == rank if penalty else _compatible(card_id, suit, rank)):
                choices[candidates] = card_id
                candidates += 1
    if token[93] == 1:
        candidates = 4
        for i in range(4):
            choices[i] = 50 + i
    if candidates < 2:
        return values
    # Latest observed discards survive most reshuffles. Draw identities, the
    # opening reserve and precise recycle times are unavailable to the actor.
    # Fit the retained window to the physical deck and public hand sizes.
    capacity = 54 - sizes.sum()
    discard_size = 0
    observed_size = int(history[PILE_SIZE]) if len(history) > PILE_SIZE else 0
    start = PILE_SIZE - 1 if observed_size else RECENT_PLAYS - 1
    end = start - observed_size if observed_size else -1
    for i in range(start, end, -1):
        card_id = int(history[i]) - 1
        if card_id >= 0 and remaining[card_id] > 0 and discard_size < capacity:
            discard[discard_size] = card_id
            discard_size += 1
            remaining[card_id] -= 1
    if discard_size == 0 and capacity > 0 and 0 <= suit < 4 and 1 <= rank <= 12:
        top = 12 * suit + rank - 1
        if remaining[top] > 0:
            discard[discard_size] = top
            discard_size += 1
            remaining[top] -= 1
    discard[:discard_size] = discard[:discard_size][::-1].copy()
    pool_size = 0
    for card_id in range(50):
        for i in range(remaining[card_id]):
            pool[pool_size] = card_id
            pool_size += 1
    weights = np.ones((2, players, 50))
    evidence = False
    for rel in range(1, players):
        absolute = (rel * int(token[89])) % players
        slot = EVIDENCE + absolute * BELIEF_WIDTH
        last = int(history[LAST_CARD + absolute]) - 1
        declared = int(history[DECLARATION + absolute]) - 1
        for card_id in range(50):
            base_factor = 0.
            if infer_voids:
                base_factor = history[slot + CARD_SORT[card_id]]
                if card_id < 48:
                    base_factor += history[slot + 4 + CARD_RANK[card_id]]
            for role in range(2):
                log_factor = base_factor
                if posterior.size:
                    # Physical-deck prior reweighted by own-hand frequencies
                    # measured from separate strong and random training games.
                    start = ((role * 3 + players - 2) * 9 + min(9, sizes[rel]) - 1) * 10
                    log_factor += posterior[start + CATEGORY[card_id]]
                    if 0 <= last < 48 and card_id < 48:
                        if CARD_SORT[last] == CARD_SORT[card_id]:
                            log_factor += posterior[start + 7]
                        if CARD_RANK[last] == CARD_RANK[card_id]:
                            log_factor += posterior[start + 8]
                    if card_id < 48 and declared == CARD_SORT[card_id]:
                        # The learned declaration supersedes the fixed bonus.
                        log_factor += posterior[start + 9]
                        if base_factor > 0.:
                            log_factor -= base_factor
                weights[role, rel, card_id] = math.exp(log_factor)
                evidence |= log_factor != 0.
        if card_belief.size:
            apply_card_belief(weights, token[:50], sizes, rel, suit, rank, penalty, card_belief)
            evidence = True
    for sample in range(samples):
        # Stratify both simulated styles across the same possible worlds.
        # At the neutral prior, even indices are random and odd indices are strong.
        rank_in_stratum = sample // 2
        stratum_size = (samples + 1 - sample % 2) // 2
        quantile = .5 * (sample % 2 + (rank_in_stratum + .5) / stratum_size)
        random_opponents = quantile < random_probability
        if evidence:
            stock_size, seed = _sample_informed(
                pool, pool_size, sizes, hands, stock, weights[int(random_opponents)], _next(seed), sample)
        else:
            seed = _shuffle(pool, pool_size, _next(seed))
            index = 0
            for seat in range(1, players):
                hands[seat, :sizes[seat]] = pool[index:index + sizes[seat]]
                index += sizes[seat]
            stock_size = pool_size - index
            stock[:stock_size] = pool[index:pool_size]
        for i in range(candidates):
            action = choices[i]
            values[action] += _rollout(
                hands, sizes, stock, discard, stock_size, discard_size,
                suit, rank, penalty, action, random_opponents, seed, horizon, leaf_weights,
                rollout_weights)
    center = 0.
    for i in range(candidates):
        center += values[choices[i]] / candidates
    for i in range(candidates):
        action = choices[i]
        values[action] = (values[action] - center) / samples
    return values


@njit(cache=True)
def _batch_values(tokens, recent, samples, horizon, infer_voids,
                  leaf_weights=EMPTY_WEIGHTS, posterior=EMPTY_WEIGHTS, style_probabilities=EMPTY_WEIGHTS,
                  rollout_weights=EMPTY_ROLLOUT, card_belief=EMPTY_CARD_BELIEF):
    values = np.zeros((len(tokens), 55), np.float32)
    for index in range(len(tokens)):
        p_random = style_probabilities[index] if style_probabilities.size else .5
        values[index] = _action_values(tokens[index], recent[index], samples, horizon, infer_voids,
                                      leaf_weights, posterior, p_random, rollout_weights, card_belief)
    return values


class PublicLookaheadCore(nn.Module):
    """Keep the teacher's recurrent state and a short public discard window."""

    def __init__(self, core, samples, horizon, trust, infer_voids=False,
                 leaf_weights=EMPTY_WEIGHTS, posterior=EMPTY_WEIGHTS, style=None, calibration=None,
                 track_pile=False, rollout_weights=EMPTY_ROLLOUT, card_belief=EMPTY_CARD_BELIEF):
        super().__init__()
        if (type(samples) is not int or samples < 1 or type(horizon) is not int
                or horizon < 1 or not math.isfinite(trust) or trust < 0):
            raise ValueError("expected positive samples/horizon and finite nonnegative trust")
        self.core = core
        self.samples = samples
        self.horizon = horizon
        self.trust = trust
        self.infer_voids = infer_voids
        self.track_pile = track_pile
        self.style = style
        table = None
        if calibration is not None:
            table = torch.as_tensor(calibration, dtype=torch.float32)
            if table.shape != (2, 3, 2) or not table.isfinite().all() or (table < 0).any():
                raise ValueError("invalid learned search-trust table")
        self.register_buffer("trust_table", table)
        self.leaf_weights = np.ascontiguousarray(leaf_weights, dtype=np.float64).ravel()
        self.posterior = np.ascontiguousarray(posterior, dtype=np.float64).ravel()
        self.card_belief = np.ascontiguousarray(card_belief, dtype=np.float64)
        if not self.card_belief.size:
            self.card_belief = EMPTY_CARD_BELIEF
        elif (self.card_belief.shape != (2, 3, 5, CARD_BELIEF_FEATURES)
              or not np.isfinite(self.card_belief).all()):
            raise ValueError("invalid conditional card belief")
        self.rollout_weights = np.ascontiguousarray(rollout_weights, dtype=np.float64)
        if not self.rollout_weights.size:
            self.rollout_weights = EMPTY_ROLLOUT
        elif (self.rollout_weights.shape != (3, ROLLOUT_CATEGORIES, ROLLOUT_FEATURES)
              or not np.isfinite(self.rollout_weights).all()):
            raise ValueError("invalid rollout policy weights")
        if self.leaf_weights.size not in (0, WEIGHT_COUNT) or not np.isfinite(self.leaf_weights).all():
            raise ValueError("invalid search leaf weights")
        if self.posterior.size not in (0, 2 * 3 * 9 * 10) or not np.isfinite(self.posterior).all():
            raise ValueError("invalid opponent posterior weights")

    def initial_state(self, batch_size, device):
        state = self.core.initial_state(batch_size, device)
        if self.style is not None:
            state = (*state, self.style.initial_state(batch_size, device))
        return (*state,
                torch.zeros((1, batch_size, HISTORY_SIZE), device=device))

    def _recent(self, tokens, history):
        result = _advance_tracking(tokens.detach().cpu().numpy(),
                                   history[0].cpu().numpy(), self.infer_voids, self.track_pile)
        return torch.from_numpy(result).unsqueeze(0).to(tokens.device)

    def forward_events_eval(self, tokens, states):
        stop = -2 if self.style is not None else -1
        return (*self.core.forward_events_eval(tokens, states[:stop]), *states[stop:-1],
                self._recent(tokens, states[-1]))

    def forward_eval(self, tokens, states):
        # Event columns are spare on a decision token. Search uses 98 to
        # skip forced choices; an action-scoring prior can also need the
        # exact playable card on a draw retry. Restore its marker from 97.
        actor_tokens = tokens.clone()
        decisions = tokens[:, 99] == 1
        actor_tokens[decisions, 98] = tokens[decisions, 97]
        actor_tokens[decisions, 97] = 0
        stop = -2 if self.style is not None else -1
        logits, values, next_states = self.core.forward_eval(actor_tokens, states[:stop])
        recent = self._recent(tokens, states[-1])
        probabilities = EMPTY_WEIGHTS
        if self.style is not None:
            probabilities, next_style = self.style.observe(actor_tokens, states[-2])
            next_states = (*next_states, next_style)
        delta = _batch_values(tokens.detach().cpu().numpy(), recent[0].cpu().numpy(),
                              self.samples, self.horizon, self.infer_voids,
                              self.leaf_weights, self.posterior, probabilities, self.rollout_weights, self.card_belief)
        trust = self.trust
        if self.trust_table is not None:
            n = (tokens[:, 92].long() - 2).clamp(0, 2)
            phase = tokens[:, 93].long().clamp(0, 1)
            p = torch.as_tensor(probabilities, device=tokens.device) if self.style is not None else .5
            trust = trust * torch.lerp(self.trust_table[0, n, phase], self.trust_table[1, n, phase], p)[:, None]
        return (logits + trust * torch.from_numpy(delta).to(logits),
                values, (*next_states, recent))


def _decision_features(decision, prior_features=None):
    # The adapter already knows which card can be played on its immediate retry.
    # Keep the teacher's exact drawn-card/pass choice when there is no card ranking.
    skip = not decision.observation.values[93] and sum(decision.legal_actions[:50]) < 2
    prior_marker = 0. if prior_features is None else prior_features(decision)[98]
    return (*decision.observation.values, 0., 0., 0., prior_marker, float(skip), 1.)


def load_candidate(path, device="cpu"):
    """Load the hash-pinned prior and apply the selected rollout recipe."""
    from production_temporal_policy import TEACHER_CHECKPOINT_SHA256, file_sha256
    from strongest_policy_search import NativeSuitOrbitCore, TEACHER
    from temporal_puffer_v4 import NativePufferTemporalPolicy, load_temporal_policy_checkpoint

    config = json.loads(Path(path).read_text())
    if config.get("kind") != "pesten_public_lookahead":
        raise ValueError("expected a pesten_public_lookahead profile")
    root = Path(__file__).resolve().parent
    prior = config.get("prior", {"kind": "native", "path": str(TEACHER),
                                 "sha256": TEACHER_CHECKPOINT_SHA256})
    prior_path = root / prior["path"]
    if file_sha256(prior_path) != prior["sha256"]:
        raise ValueError("lookahead prior hash mismatch")
    if prior["kind"] == "native":
        policy, _ = load_temporal_policy_checkpoint(prior_path, device=device)
        policy.puffer_policy = NativeSuitOrbitCore(policy.puffer_policy, orbit_size=24)
    elif prior["kind"] == "outcome_residual":
        from train_outcome_residual import load_candidate as load_prior
        policy = load_prior(prior_path, device=device)
    elif prior["kind"] == "rank_suit":
        from rank_suit_orbit import load_candidate as load_prior
        policy = load_prior(prior_path, device=device)
    else:
        raise ValueError("unrecognized lookahead prior")
    def verified_path(key):
        artifact = config[key]
        path = root / artifact["path"]
        if file_sha256(path) != artifact["sha256"]:
            raise ValueError(f"lookahead {key} hash mismatch")
        return path

    leaf_weights = (np.load(verified_path("leaf"), allow_pickle=False)
                    if "leaf" in config else EMPTY_WEIGHTS)
    rollout_weights = (np.load(verified_path("rollout"), allow_pickle=False)
                       if "rollout" in config else EMPTY_ROLLOUT)
    card_belief = (np.load(verified_path("card_belief"), allow_pickle=False)
                   if "card_belief" in config else EMPTY_CARD_BELIEF)
    posterior = (np.asarray(json.loads(verified_path("posterior").read_text())["weights"])
                 if "posterior" in config else EMPTY_WEIGHTS)
    style = None
    if "style" in config:
        from opponent_style import load_style
        style = load_style(verified_path("style"), device)
    calibration = None
    if "calibration" in config:
        fitted = json.loads(verified_path("calibration").read_text())
        if fitted["kind"] != "pesten_empirical_search_trust":
            raise ValueError("unexpected search calibration")
        calibration = fitted["trust"]
    policy.puffer_policy = PublicLookaheadCore(policy.puffer_policy, config["samples"],
                                              config["horizon"], config["trust"],
                                              config.get("infer_voids", False),
                                              leaf_weights, posterior, style, calibration,
                                              config.get("track_pile", False), rollout_weights, card_belief)
    prior_features = policy._decision_features
    policy._decision_features = (_decision_features
                                 if prior_features is NativePufferTemporalPolicy._decision_features
                                 else partial(_decision_features, prior_features=prior_features))
    return policy.to(device).eval().requires_grad_(False)
