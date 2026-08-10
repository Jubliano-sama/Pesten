import numpy as np
import pytest

from batched_pesten import (
    BatchedPesten,
    PHASE_SUIT,
    POLICY_FIRST_LEGAL,
    POLICY_RANDOM,
    POLICY_SMART,
    POLICY_SMARTER,
)
from numba_smart_policy import (
    smart_action_native,
    smart_actions_native,
)


@pytest.mark.skipif(smart_action_native is None, reason="Numba is not installed")
def test_native_smart_policy_matches_randomized_python_reference() -> None:
    engine = BatchedPesten(
        np.full((1, 2), POLICY_SMART, dtype=np.int8),
        np.full((1, 2), None, dtype=object),
        np.asarray([84_970_001], dtype=np.uint32),
        smart_agents_have_game_reference=False,
    )
    rng = np.random.RandomState(84_970_002)
    for with_reference in (False, True):
        engine.smart_agents_have_game_reference = with_reference
        for _ in range(50_000):
            size = int(rng.randint(1, 40))
            hand = rng.randint(0, 50, size=size, dtype=np.int8)
            engine.hands[0, 0].fill(-1)
            engine.hands[0, 0, :size] = hand
            engine.hand_sizes[0, 0] = size
            engine.current_player[0] = 0
            engine.current_sort[0] = rng.randint(0, 5)
            engine.current_rank[0] = rng.randint(0, 14)
            expected = engine.smart_action(0)
            actual = smart_action_native(
                hand,
                int(engine.current_sort[0]),
                int(engine.current_rank[0]),
                with_reference,
            )
            assert actual == expected


@pytest.mark.skipif(smart_actions_native is None, reason="Numba is not installed")
def test_batched_native_smart_policy_matches_scalar_reference() -> None:
    game_count = 32
    reference = BatchedPesten(
        np.full((game_count, 2), POLICY_SMART, dtype=np.int8),
        np.full((game_count, 2), None, dtype=object),
        np.arange(84_970_010, 84_970_010 + game_count, dtype=np.uint32),
        smart_agents_have_game_reference=False,
    )
    native = BatchedPesten(
        reference.policy_kinds.copy(),
        reference.actor_keys.copy(),
        np.arange(84_970_010, 84_970_010 + game_count, dtype=np.uint32),
        smart_agents_have_game_reference=False,
        native_smart_policy=True,
    )
    indices = np.arange(game_count, dtype=np.int64)

    expected = np.asarray(
        [reference.smart_action(int(index)) for index in indices],
        dtype=np.int64,
    )
    actual = native.smart_actions(indices)

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(smart_actions_native is None, reason="Numba is not installed")
def test_batched_non_neural_dispatch_preserves_actions_and_random_hands() -> None:
    kinds = np.asarray(
        [
            [POLICY_SMART, POLICY_SMART],
            [POLICY_RANDOM, POLICY_RANDOM],
            [POLICY_FIRST_LEGAL, POLICY_FIRST_LEGAL],
            [POLICY_SMARTER, POLICY_SMARTER],
        ]
        * 4,
        dtype=np.int8,
    )
    seeds = np.arange(84_970_100, 84_970_100 + len(kinds), dtype=np.uint32)
    reference = BatchedPesten(
        kinds,
        np.full(kinds.shape, None, dtype=object),
        seeds,
        native_smart_policy=False,
    )
    native = BatchedPesten(
        kinds,
        np.full(kinds.shape, None, dtype=object),
        seeds,
        native_smart_policy=True,
    )
    suit_games = np.asarray([0, 5, 10, 15], dtype=np.int64)
    reference.phase[suit_games] = PHASE_SUIT
    native.phase[suit_games] = PHASE_SUIT
    indices = np.arange(len(kinds), dtype=np.int64)

    expected = np.asarray(
        [
            reference.non_neural_action(int(index))
            for index in indices
        ],
        dtype=np.int64,
    )
    actual = native.non_neural_actions(indices)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(native.hands, reference.hands)
