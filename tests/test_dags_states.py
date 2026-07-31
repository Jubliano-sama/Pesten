import numpy as np
import pytest

from ai_observation import OBSERVATION_SIZE
from batched_pesten import (
    PHASE_DRAW_RETRY,
    PHASE_SUIT,
    POLICY_RL,
    SUIT_ACTION_START,
    BatchedPesten,
)
from dags_states import (
    DAGS_OBSERVATION_SIZE,
    DAGS_SOURCE_FLAG_INDEX,
    capture_dags_state,
    dags_observation,
    restore_dags_state,
)


def _engine(seed: int) -> BatchedPesten:
    return BatchedPesten(
        np.full((1, 2), POLICY_RL, dtype=np.int8),
        np.full((1, 2), "self", dtype=object),
        np.asarray([seed], dtype=np.uint32),
    )


def _deterministic_action(engine: BatchedPesten) -> int:
    if int(engine.phase[0]) == PHASE_SUIT:
        return SUIT_ACTION_START
    if int(engine.phase[0]) == PHASE_DRAW_RETRY:
        return int(engine.drawn_card[0])
    return engine.first_legal_action(0)


def _advance(engine: BatchedPesten, decisions: int) -> None:
    for _ in range(decisions):
        if engine.game_over[0]:
            break
        engine.apply_action(0, _deterministic_action(engine))


def test_capture_restore_reproduces_observation_and_continuation():
    source = _engine(84_500_101)
    _advance(source, 40)
    state = capture_dags_state(source, 0)
    before = dags_observation(source, 0, dataset_start=True)

    restored = _engine(84_500_102)
    restore_dags_state(restored, 0, state)
    assert np.array_equal(
        dags_observation(restored, 0, dataset_start=True),
        before,
    )
    assert restored.actor_keys[0].tolist() == ["self", "self"]
    source_rng = source.rngs[0].get_state()
    restored_rng = restored.rngs[0].get_state()
    assert source_rng[0] == restored_rng[0]
    assert np.array_equal(source_rng[1], restored_rng[1])
    assert source_rng[2:] == restored_rng[2:]

    for _ in range(30):
        assert _deterministic_action(restored) == _deterministic_action(source)
        restored.apply_action(0, _deterministic_action(restored))
        source.apply_action(0, _deterministic_action(source))
        if source.game_over[0]:
            break
        assert np.array_equal(
            dags_observation(restored, 0, dataset_start=True),
            dags_observation(source, 0, dataset_start=True),
        )
    assert restored.turn_counts[0] == source.turn_counts[0]
    assert restored.game_over[0] == source.game_over[0]
    assert restored.winners[0] == source.winners[0]


def test_policy_observation_does_not_expose_opponent_or_draw_identities():
    engine = _engine(84_500_103)
    _advance(engine, 20)
    acting = int(engine.current_player[0])
    opponent = 1 - acting
    assert int(engine.hand_sizes[0, opponent]) > 0
    assert int(engine.draw_counts[0]) > 0
    before = dags_observation(engine, 0, dataset_start=True)

    opponent_card = int(engine.hands[0, opponent, 0])
    draw_offset = int(engine.draw_counts[0]) - 1
    draw_card = int(engine.draw_piles[0, draw_offset])
    if opponent_card == draw_card:
        draw_offset -= 1
        draw_card = int(engine.draw_piles[0, draw_offset])
    engine.hands[0, opponent, 0] = draw_card
    engine.hand_counts[0, opponent, opponent_card] -= 1
    engine.hand_counts[0, opponent, draw_card] += 1
    engine.draw_piles[0, draw_offset] = opponent_card

    after = dags_observation(engine, 0, dataset_start=True)
    assert np.array_equal(after, before)
    # The mutated hidden allocation remains a valid, deck-conserving snapshot.
    capture_dags_state(engine, 0)


def test_source_flag_is_the_only_observation_difference():
    engine = _engine(84_500_104)
    state = capture_dags_state(engine, 0)
    root = dags_observation(engine, 0, dataset_start=False)
    dataset = dags_observation(engine, 0, dataset_start=True)

    assert root.shape == dataset.shape == (DAGS_OBSERVATION_SIZE,)
    assert np.array_equal(root[:OBSERVATION_SIZE], dataset[:OBSERVATION_SIZE])
    assert root[DAGS_SOURCE_FLAG_INDEX] == 0.0
    assert dataset[DAGS_SOURCE_FLAG_INDEX] == 1.0
    with pytest.raises(ValueError, match="read-only"):
        state.array("hands")[0, 0] = 0
    with pytest.raises(ValueError, match="read-only"):
        state.rng_state[1][0] = 0


def test_restore_rejects_player_count_mismatch():
    state = capture_dags_state(_engine(84_500_105), 0)
    three_player = BatchedPesten(
        np.full((1, 3), POLICY_RL, dtype=np.int8),
        np.full((1, 3), "self", dtype=object),
        np.asarray([84_500_106], dtype=np.uint32),
    )
    with pytest.raises(ValueError, match="player counts differ"):
        restore_dags_state(three_player, 0, state)


def test_capture_rejects_inactive_game():
    engine = _engine(84_500_107)
    engine.turn_counts[0] = engine.max_turns
    with pytest.raises(ValueError, match="active game"):
        capture_dags_state(engine, 0)
