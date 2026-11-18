import numpy as np
import pytest
import torch
from types import SimpleNamespace

from public_lookahead_leaf import FEATURES, HIDDEN, WEIGHT_COUNT, _leaf_value, _position_features


def _world():
    hands = np.full((3, 54), -1, dtype=np.int8)
    hands[0, :5] = [0, 7, 12, 33, 49]
    hands[1, :2] = [3, 16]
    hands[2, :3] = [24, 41, 48]
    return hands, np.array([5, 2, 3], dtype=np.int64)


def test_leaf_scores_match_exported_torch_network():
    pytest.importorskip("pufferlib")
    from train_public_leaf import LeafNet
    torch.manual_seed(284)
    net = LeafNet(FEATURES, HIDDEN)
    weights = net.pack()
    assert weights.shape == (WEIGHT_COUNT,)
    for n in (2, 3):
        hands, sizes = _world()
        for direction in (-1, 1):
            feat = _position_features(hands[:n], sizes[:n], 1, direction, 2, 8, 0, True)
            padded = np.zeros((1, 4, FEATURES), np.float32)
            padded[0, :n] = feat
            mask = torch.arange(4)[None, :] < n
            with torch.no_grad():
                expected = float(net(torch.from_numpy(padded), mask).softmax(1)[0, 0])
            value = _leaf_value(hands[:n], sizes[:n], 1, direction, 2, 8, 0, True, weights)
            assert abs(expected - value) < 1e-6


def test_features_follow_turn_order_and_ignore_unused_hand_slots():
    hands, sizes = _world()
    x = _position_features(hands, sizes, 2, -1, 0, 2, 2, False)
    changed = hands.copy()
    for i in range(len(sizes)):
        changed[i, sizes[i]:] = 48
    assert np.array_equal(x, _position_features(changed,sizes,2,-1,0,2,2,False))
    rotated = np.roll(hands, 1, axis=0)
    f = _position_features(rotated, np.roll(sizes,1), 0,-1,0,2,2,False)
    assert np.allclose(np.roll(x, 1, axis=0), f)
    assert x[2,31] == 1
    assert x[1,32] == 1


def test_paired_afterstates_wait_for_resolved_actions_and_keep_timeout_outcomes():
    pytest.importorskip("pufferlib")
    from batched_pesten import PHASE_CARD, PHASE_SUIT
    from refine_leaf_contrasts import PairedAfterstates

    cards, sizes = _world()
    game = SimpleNamespace(
        num_players=2, game_over=np.array([False, False]), max_turns=1000,
        turn_counts=np.array([15, 15]), winners=np.array([-1, -1]),
        phase=np.array([PHASE_SUIT, PHASE_CARD]), current_player=np.array([1, 0]),
        current_sort=np.array([2, 2]), current_rank=np.array([8, 8]),
        penalty_amount=np.array([0, 0]), direction=np.array([1, 1]),
        hands=np.repeat(cards[None, :2], 2, axis=0),
        hand_sizes=np.repeat(sizes[None, :2], 2, axis=0),
    )
    receiver = PairedAfterstates(1)
    focal = np.array([1])
    random_others = np.array([False])
    receiver(game, np.array([False]), focal, random_others)
    assert not receiver.recorded.any()
    receiver(game, np.array([True]), focal, random_others)
    assert receiver.recorded.tolist() == [False, True]
    stored = receiver.features[1].copy()

    game.turn_counts[0] = game.max_turns
    game.game_over[1] = True
    game.winners[1] = 1
    receiver(game, np.array([True]), focal, random_others)
    assert receiver.recorded.all()
    assert receiver.terminal.tolist() == [True, False]
    assert receiver.focal_wins.tolist() == [False, True]
    assert np.array_equal(receiver.features[1], stored)
