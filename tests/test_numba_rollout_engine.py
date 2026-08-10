import numpy as np
import pytest

from batched_pesten import (
    BatchedPesten,
    POLICY_FIRST_LEGAL,
)


_STATE_ARRAYS = (
    "hands",
    "hand_counts",
    "hand_sizes",
    "draw_piles",
    "draw_counts",
    "discards",
    "discard_counts",
    "current_sort",
    "current_rank",
    "current_player",
    "turn_player",
    "direction",
    "penalty_amount",
    "sorts_played",
    "turn_counts",
    "cards_played",
    "game_over",
    "winners",
    "phase",
    "turn_started_with_one_card",
    "drawn_card",
)


def _engine(
    *,
    native: bool,
    games: int = 64,
    players: int = 2,
) -> BatchedPesten:
    policy_kinds = np.full(
        (games, players),
        POLICY_FIRST_LEGAL,
        dtype=np.int8,
    )
    actor_keys = np.full((games, players), None, dtype=object)
    seeds = np.arange(85_030_001, 85_030_001 + games, dtype=np.uint32)
    return BatchedPesten(
        policy_kinds,
        actor_keys,
        seeds,
        max_turns=500,
        native_rollout_engine=native,
    )


def test_native_simple_action_batches_preserve_complete_game_state() -> None:
    reference = _engine(native=False)
    native = _engine(native=True)
    native_fast_actions = 0

    while True:
        active = reference.active_games()
        if len(active) == 0:
            break
        suit_modes = reference.phase[active] == 2
        expected_observations = np.empty(
            (len(active), 121),
            dtype=np.float32,
        )
        actual_observations = np.empty_like(expected_observations)
        expected_masks = np.empty((len(active), 55), dtype=np.bool_)
        actual_masks = np.empty_like(expected_masks)
        reference.observations_into(
            active,
            expected_observations,
            suit_modes=suit_modes,
            action_masks=expected_masks,
        )
        native.observations_into(
            active,
            actual_observations,
            suit_modes=suit_modes,
            action_masks=actual_masks,
        )
        np.testing.assert_array_equal(
            actual_observations,
            expected_observations,
        )
        np.testing.assert_array_equal(actual_masks, expected_masks)
        np.testing.assert_array_equal(
            native.rl_has_card_decisions(active),
            reference.rl_has_card_decisions(active),
        )
        actions = np.asarray(
            [
                reference.non_neural_action(int(game_index))
                for game_index in active
            ],
            dtype=np.int64,
        )
        for game_index, action in zip(active, actions):
            reference.apply_action(int(game_index), int(action))

        before_events = sum(map(len, native.public_events))
        native.apply_actions(active, actions)
        after_events = sum(map(len, native.public_events))
        native_fast_actions += after_events - before_events

    assert native_fast_actions > 1_000
    for name in _STATE_ARRAYS:
        assert np.array_equal(
            getattr(native, name),
            getattr(reference, name),
        ), name
    assert native.public_events == reference.public_events
    assert native.event_counts == reference.event_counts
    for actual_rng, expected_rng in zip(native.rngs, reference.rngs):
        actual_state = actual_rng.get_state()
        expected_state = expected_rng.get_state()
        assert actual_state[0] == expected_state[0]
        assert np.array_equal(actual_state[1], expected_state[1])
        assert actual_state[2:] == expected_state[2:]


def test_native_rl_decision_precheck_preserves_duplicate_count_behavior() -> None:
    reference = _engine(native=False, games=1)
    native = _engine(native=True, games=1)
    for engine in (reference, native):
        engine.current_player[0] = 0
        engine.current_sort[0] = 0
        engine.current_rank[0] = 1
        engine.hand_counts[0, 0].fill(0)
        engine.hand_counts[0, 0, 0] = 2

    expected = reference.rl_has_card_decisions(
        np.asarray([0], dtype=np.int64)
    )
    actual = native.rl_has_card_decisions(
        np.asarray([0], dtype=np.int64)
    )

    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("players", [3, 8])
def test_native_observations_preserve_multi_player_layout_and_masks(
    players: int,
) -> None:
    reference = _engine(native=False, games=4, players=players)
    native = _engine(native=True, games=4, players=players)
    current_players = np.asarray(
        [0, players - 1, players // 2, 1],
        dtype=np.int8,
    )
    directions = np.asarray([1, -1, 1, -1], dtype=np.int8)
    phases = np.asarray([0, 1, 2, 0], dtype=np.int8)
    for engine in (reference, native):
        engine.current_player[:] = current_players
        engine.direction[:] = directions
        engine.phase[:] = phases
        engine.drawn_card[1] = engine.hands[1, current_players[1], 0]

    game_indices = np.arange(4, dtype=np.int64)
    suit_modes = phases == 2
    expected_observations = np.empty((4, 121), dtype=np.float32)
    actual_observations = np.empty_like(expected_observations)
    expected_masks = np.empty((4, 55), dtype=np.bool_)
    actual_masks = np.empty_like(expected_masks)

    reference.observations_into(
        game_indices,
        expected_observations,
        suit_modes=suit_modes,
        action_masks=expected_masks,
    )
    native.observations_into(
        game_indices,
        actual_observations,
        suit_modes=suit_modes,
        action_masks=actual_masks,
    )

    np.testing.assert_array_equal(actual_observations, expected_observations)
    np.testing.assert_array_equal(actual_masks, expected_masks)


def test_same_shape_engine_reset_matches_fresh_allocation() -> None:
    games = 16
    players = 3
    reused = _engine(native=True, games=games, players=players)
    dirty_games = reused.active_games()[:8]
    dirty_actions = np.asarray(
        [
            reused.non_neural_action(int(game_index))
            for game_index in dirty_games
        ],
        dtype=np.int64,
    )
    reused.apply_actions(dirty_games, dirty_actions)

    policy_kinds = np.full(
        (games, players),
        POLICY_FIRST_LEGAL,
        dtype=np.int8,
    )
    actor_keys = np.full((games, players), None, dtype=object)
    seeds = np.arange(85_031_001, 85_031_001 + games, dtype=np.uint32)
    reused.reset_games(policy_kinds, actor_keys, seeds)
    fresh = BatchedPesten(
        policy_kinds,
        actor_keys,
        seeds,
        max_turns=500,
        native_rollout_engine=True,
    )

    while True:
        active = fresh.active_games()
        if len(active) == 0:
            break
        actions = np.asarray(
            [
                fresh.non_neural_action(int(game_index))
                for game_index in active
            ],
            dtype=np.int64,
        )
        fresh.apply_actions(active, actions)
        reused.apply_actions(active, actions)

    for name in _STATE_ARRAYS:
        np.testing.assert_array_equal(
            getattr(reused, name),
            getattr(fresh, name),
        )
    assert reused.public_events == fresh.public_events
    assert reused.event_counts == fresh.event_counts
    for actual_rng, expected_rng in zip(reused.rngs, fresh.rngs):
        actual_state = actual_rng.get_state()
        expected_state = expected_rng.get_state()
        assert actual_state[0] == expected_state[0]
        np.testing.assert_array_equal(actual_state[1], expected_state[1])
        assert actual_state[2:] == expected_state[2:]


@pytest.mark.parametrize("players", [3, 4, 6, 8])
def test_native_engine_preserves_multi_player_multi_deck_games(
    players: int,
) -> None:
    reference = _engine(native=False, games=8, players=players)
    native = _engine(native=True, games=8, players=players)

    while True:
        active = reference.active_games()
        if len(active) == 0:
            break
        actions = np.asarray(
            [
                reference.non_neural_action(int(game_index))
                for game_index in active
            ],
            dtype=np.int64,
        )
        reference.apply_actions(active, actions)
        native.apply_actions(active, actions)

    for name in _STATE_ARRAYS:
        assert np.array_equal(
            getattr(native, name),
            getattr(reference, name),
        ), (players, name)
    assert native.public_events == reference.public_events
    assert native.event_counts == reference.event_counts
