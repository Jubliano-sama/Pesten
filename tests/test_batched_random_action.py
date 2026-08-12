import numpy as np

from batched_pesten import (
    BatchedPesten,
    CARD_COMPATIBILITY_MASKS,
    PHASE_DRAW_RETRY,
    PASS_ACTION,
    POLICY_RANDOM,
)


def _engine() -> BatchedPesten:
    return BatchedPesten(
        np.full((8, 4), POLICY_RANDOM, dtype=np.int8),
        np.full((8, 4), None, dtype=object),
        np.arange(85_144_001, 85_144_009, dtype=np.uint32),
        max_turns=500,
    )


def _copied_random_action(engine: BatchedPesten, game_index: int) -> int:
    player_index = int(engine.current_player[game_index])
    size = int(engine.hand_sizes[game_index, player_index])
    hand = engine.hands[game_index, player_index]
    shuffled = hand[:size].copy()
    engine.rngs[game_index].shuffle(shuffled)
    hand[:size] = shuffled
    compatible = CARD_COMPATIBILITY_MASKS[
        int(engine.current_sort[game_index]),
        int(engine.current_rank[game_index]),
    ]
    for number in shuffled:
        if compatible[int(number)]:
            return int(number)
    return PASS_ACTION


def test_in_place_random_shuffle_preserves_actions_state_and_rng() -> None:
    reference = _engine()
    actual = _engine()
    for _ in range(32):
        for game_index in range(reference.num_games):
            assert actual.random_action(game_index) == (
                _copied_random_action(reference, game_index)
            )
    assert np.array_equal(actual.hands, reference.hands)
    for actual_rng, expected_rng in zip(actual.rngs, reference.rngs):
        actual_state = actual_rng.get_state()
        expected_state = expected_rng.get_state()
        assert actual_state[0] == expected_state[0]
        assert np.array_equal(actual_state[1], expected_state[1])
        assert actual_state[2:] == expected_state[2:]


def test_random_retry_plays_the_only_newly_compatible_drawn_card() -> None:
    engine = _engine()
    game_index = 0
    player_index = int(engine.current_player[game_index])
    hand = engine.hands[game_index, player_index]
    hand.fill(-1)
    hand[:2] = (12, 30)  # Neither hearts nor rank five.
    engine.hand_sizes[game_index, player_index] = 2
    engine.hand_counts[game_index, player_index].fill(0)
    engine.hand_counts[game_index, player_index, 12] = 1
    engine.hand_counts[game_index, player_index, 30] = 1
    engine.current_sort[game_index] = 0
    engine.current_rank[game_index] = 5

    assert engine.random_action(game_index) == PASS_ACTION

    drawn_card = 16  # Ruiten five: compatible by rank.
    size = int(engine.hand_sizes[game_index, player_index])
    engine.hands[game_index, player_index, size] = drawn_card
    engine.hand_sizes[game_index, player_index] = size + 1
    engine.hand_counts[game_index, player_index, drawn_card] = 1
    engine.phase[game_index] = PHASE_DRAW_RETRY
    engine.drawn_card[game_index] = drawn_card

    assert engine.random_action(game_index) == drawn_card
