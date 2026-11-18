import numpy as np

from public_card_belief import FEATURES, apply_card_belief
from train_card_belief import encode


def test_equivalent_suit_and_ordinary_rank_names_give_the_same_odds():
    new_suit = np.array([2, 0, 3, 1])
    new_rank = np.arange(12)
    new_rank[[2, 9]] = [9, 2]
    cards = np.concatenate(((12 * new_suit[:, None] + new_rank).ravel(), [48, 49]))
    own = np.zeros(50, np.uint8)
    own[[2, 4, 12, 39, 48, 49]] = 1
    permuted = np.zeros_like(own)
    permuted[cards] = own
    sizes = np.array([6, 2, 3])
    rng = np.random.default_rng(121)
    coeff = rng.normal(size=(2, 3, 5, FEATURES)) * 0.15
    w1, w2 = np.ones((2, 3, 50)), np.ones((2, 3, 50))
    apply_card_belief(w1, own, sizes, 2, 0, 3, 0, coeff)
    apply_card_belief(w2, permuted, sizes, 2, new_suit[0], new_rank[2] + 1, 0, coeff)
    np.testing.assert_allclose(w1, w2[:, :, cards], atol=0, rtol=0)
    assert np.all(w1[:, :2] == 1)


def test_predictions_use_only_observer_cards_and_public_information():
    # Cases share their public turn and observer hand while the two hidden hands change.
    hands = np.full((2, 3, 54), -1, dtype=np.int8)
    hands[:, 0, :2] = [4, 48]
    hands[0, 1, :1] = [14]
    hands[0, 2, :2] = [20, 36]
    hands[1, 1, :1] = [36]
    hands[1, 2, :2] = [14, 20]
    sizes = np.array([[2, 1, 2], [2, 1, 2]], dtype=np.uint8)
    src, y, x, groups = encode(
        hands,
        sizes,
        np.array([-1, -1]),
        np.array([2, 2]),
        np.array([5, 5]),
        np.array([0, 0]),
        np.array([False, False]),
        np.zeros((2, 3, 9, 10)),
    )
    np.testing.assert_array_equal(x[:2], x[2:])
    np.testing.assert_array_equal(groups[:2], groups[2:])
    np.testing.assert_array_equal(src[:2], src[2:])
    assert np.any(y[:2] != y[2:])
    assert src[0, 4] == src[0, 28] == 0 and src[0, 48] == 3
    # Reversed direction presents physical seat two as the next opponent.
    np.testing.assert_array_equal(np.flatnonzero(y[0]), [20, 36])
    np.testing.assert_array_equal(np.flatnonzero(y[1]), [14])
