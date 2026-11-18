"""Regression checks for public rollout search, physical rules, and adapter state."""

import numpy as np
import pytest
import torch

from public_lookahead import (
    BELIEF_WIDTH,
    EVIDENCE,
    HISTORY_SIZE,
    RECENT_PLAYS,
    PublicLookaheadCore,
    _advance_tracking,
    _batch_values,
    _decision_features,
    _draw,
    _rollout,
)
from temporal_observation import (
    DecisionInput,
    PlayerPrivateObservation,
    build_draw_retry_legal_actions,
)


def token(hand, others=(1,), suit=0, rank=4, direction=1):
    row = np.zeros(100, np.float32)
    for card in hand:
        row[card] += 1
    row[50 + suit] = row[55 + rank] = 1
    row[69 : 70 + len(others)] = (len(hand), *others)
    row[89], row[91], row[92], row[99] = direction, len(hand), 1 + len(others), 1
    return row


def event(kind, actor=0, card=-1, suit=-1, count=0):
    row = np.zeros(100, np.float32)
    row[94:99] = (kind / 3, actor / 12, (1 + card) / 50, (1 + suit) / 5, count / 54)
    return row


@pytest.mark.parametrize(
    ("card", "players", "won"), [(6, 2, 1), (6, 3, 1), (7, 2, 1), (7, 3, 0)]
)
def test_tempo_then_ordinary_finish(card, players, won):
    hands = np.zeros((players, 54), np.int8)
    sizes = np.ones(players, np.int64)
    sizes[0] = 2
    hands[0, :2] = [card, 2]
    for seat in range(1, players):
        hands[seat, 0] = 3 + seat
    stock, discard = np.zeros(54, np.int8), np.zeros(54, np.int8)
    assert (
        _rollout(hands, sizes, stock, discard, 0, 1, 0, 4, 0, card, False, 42, 3) == won
    )


def test_playing_last_special_draws_two_and_passes_the_turn():
    hands = np.zeros((2, 54), np.int8)
    hands[0, 0], hands[1, 0] = 6, 2
    stock, discard = np.zeros(54, np.int8), np.zeros(54, np.int8)
    stock[:2] = [26, 39]
    sizes = np.ones(2, np.int64)
    assert _rollout(hands, sizes, stock, discard, 2, 1, 0, 4, 0, 6, False, 42, 2) == 0
    assert sizes.tolist() == [1, 1]


def test_recycle_keeps_top_discard_out_of_draws():
    physical = np.zeros(54, np.int8)
    physical[:3] = [16, 25, 37]
    stock = np.zeros_like(physical)
    hand = np.zeros_like(physical)
    chosen, left, discard_size, _ = _draw(hand, 0, stock, 0, physical, 3, 17)
    assert chosen in (16, 25)
    assert {chosen, int(stock[0])} == {16, 25}
    assert left == discard_size == 1
    assert physical[0] == 37
    assert _draw(hand, 1, stock, 0, physical, 1, 17)[0] == -1


def test_public_discards_and_soft_draw_inference_are_seat_relative():
    history = np.zeros((1, HISTORY_SIZE), np.float32)
    for row in [
        token([2, 6], others=(3, 5)),
        event(1, card=2, suit=0, count=1),
        event(2, actor=1, count=1),
        token([6], others=(4, 5), rank=3, direction=-1),
    ]:
        history = _advance_tracking(row[None], history, True)
    assert history[0, RECENT_PLAYS - 1] == 3
    assert history[0, EVIDENCE + 2 * BELIEF_WIDTH] < 0
    assert history[0, EVIDENCE + BELIEF_WIDTH] == 0


def test_rollout_search_is_reproducible_when_batch_order_changes():
    rows = np.stack([token([2, 6, 18]), token([15, 27, 39, 48], others=(2, 2))])
    history = np.zeros((2, HISTORY_SIZE), np.float32)
    values = _batch_values(rows, history, 3, 12, True)
    replay = _batch_values(rows[::-1].copy(), history, 3, 12, True)
    np.testing.assert_array_equal(values, replay[::-1])
    assert np.isfinite(values).all()
    assert (abs(values) <= 1).all()


class Prior(torch.nn.Module):
    def initial_state(self, batch_size, device):
        return (torch.zeros(1, batch_size, 1, device=device),)

    def forward_eval(self, tokens, states):
        self.last = tokens
        logits = torch.arange(55).float().expand(len(tokens), -1)
        return logits, torch.ones(len(tokens), 1), (states[0] + 1,)


def test_retry_pass_and_card_logits_are_exact_even_with_other_playable_cards():
    row = token([2, 6, 48])
    restricted = DecisionInput(
        PlayerPrivateObservation(tuple(row[:94])), build_draw_retry_legal_actions(2)
    )
    marked = torch.tensor([_decision_features(restricted)])
    prior = Prior()
    core = PublicLookaheadCore(prior, samples=3, horizon=12, trust=500.0)
    logits, _, state = core.forward_eval(marked, core.initial_state(1, "cpu"))
    assert logits[0, 2] == 2 and logits[0, 54] == 54
    assert prior.last[0, 98] == 0
    assert state[0].item() == 1


def test_search_preserves_action_scoring_priors_exact_draw_retry():
    from branch_outcome_policy import _decision_features as branch_features

    row = token([2, 6, 48])
    restricted = DecisionInput(
        PlayerPrivateObservation(tuple(row[:94])), build_draw_retry_legal_actions(48)
    )
    search_token = torch.tensor([_decision_features(restricted, branch_features)])
    prior = Prior()
    core = PublicLookaheadCore(prior, samples=3, horizon=12, trust=500.0)
    logits, _, _ = core.forward_eval(search_token, core.initial_state(1, "cpu"))
    torch.testing.assert_close(prior.last, torch.tensor([branch_features(restricted)]))
    assert logits[0, 48] == 48 and logits[0, 54] == 54


def test_public_card_conservation_recognizes_recycling():
    from public_lookahead import LAST_CAPACITY, LAST_PLAY_COUNT, PILE_SIZE

    history = np.zeros((1, HISTORY_SIZE), np.float32)
    history[0, PILE_SIZE] = 10
    history[0, LAST_CAPACITY] = 10  # Estimated draw stock already empty.
    history[0, LAST_PLAY_COUNT] = 20
    row = token([2], others=(44,))
    row[82] = 20
    result = _advance_tracking(row[None], history, True, True)
    assert result[0, PILE_SIZE] == 1
    assert history[0, PILE_SIZE] == 10


@pytest.mark.parametrize(
    "kw",
    [
        {"leaf_weights": np.zeros(3)},
        {"posterior": [float("nan")] * 540},
        {"rollout_weights": np.zeros((2, 8, 24))},
        {"rollout_weights": np.full((3, 8, 24), np.nan)},
        {"samples": 1.5},
        {"horizon": 0},
    ],
)
def test_invalid_search_profiles_fail_before_compiled_simulation(kw):
    params = {"samples": 1, "horizon": 2, "trust": 1, **kw}
    with pytest.raises(ValueError):
        PublicLookaheadCore(Prior(), **params)


def test_empty_continuation_weights_use_the_default_compiled_shape():
    core = PublicLookaheadCore(Prior(), samples=1, horizon=1, trust=1, rollout_weights=[])
    assert core.rollout_weights.shape == (0, 0, 0)
