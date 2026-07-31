"""Exactness checks for vectorized public-context observation writes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ai_observation import (
    COMPATIBLE_END,
    COMPATIBLE_START,
    DIRECTION_INDEX,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PLAYER_HAND_SIZES_END,
    PLAYER_HAND_SIZES_START,
    PLAYER_HAND_SIZE_SLOTS,
    PLAYER_INDEX,
    SORT_COUNT,
    SORTS_PLAYED_START,
    populate_game_context,
    relative_player_indices,
)
from best_player import BestPlayerConfig, load_best_player
from ppo2 import build_action_masks


def scalar_reference(
    observation,
    *,
    sorts_played,
    hand_sizes,
    player_index,
    direction,
    is_change_sort=False,
):
    """The scalar assignment implementation frozen before this optimization."""
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(
            f"observation must contain {OBSERVATION_SIZE} values, "
            f"got {len(observation)}"
        )
    if len(sorts_played) < SORT_COUNT:
        raise ValueError(
            f"sorts_played must contain at least {SORT_COUNT} values, "
            f"got {len(sorts_played)}"
        )
    for offset in range(SORT_COUNT):
        observation[SORTS_PLAYED_START + offset] = sorts_played[offset]
    observation[MODE_INDEX] = 1 if is_change_sort else 0
    for feature_index in range(
        PLAYER_HAND_SIZES_START,
        PLAYER_HAND_SIZES_END,
    ):
        observation[feature_index] = 0
    order = relative_player_indices(
        len(hand_sizes),
        player_index,
        direction,
    )
    for slot, ordered_player_index in enumerate(
        order[:PLAYER_HAND_SIZE_SLOTS]
    ):
        observation[PLAYER_HAND_SIZES_START + slot] = (
            hand_sizes[ordered_player_index]
        )
    observation[PLAYER_INDEX] = player_index
    observation[DIRECTION_INDEX] = direction
    return observation


def _container(kind, values):
    if kind == "list":
        return list(values)
    if kind == "numpy":
        return np.asarray(values, dtype=np.float32)
    return torch.asarray(values, dtype=torch.float32)


def _exact(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    return left == right


class IntegerOnlySequence:
    """Mutable sequence implementing the scalar contract but no slices."""

    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError("slice access is unsupported")
        return self.values[index]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise TypeError("slice assignment is unsupported")
        self.values[index] = value


@pytest.mark.parametrize("kind", ("list", "numpy", "torch"))
def test_vectorized_context_is_exact_over_fuzzed_valid_inputs(kind):
    rng = np.random.RandomState(89_607_031)
    for player_count in range(0, 14):
        player_indices = (
            (-1, 0)
            if player_count == 0
            else (-1, *range(player_count))
        )
        for player_index in player_indices:
            for direction in (-1, 1):
                initial = rng.standard_normal(OBSERVATION_SIZE).tolist()
                sorts_played = rng.randint(-5, 50, size=8).tolist()
                hand_sizes = rng.randint(
                    0,
                    30,
                    size=player_count,
                ).tolist()
                for is_change_sort in (False, True):
                    expected = _container(kind, initial)
                    actual = _container(kind, initial)
                    scalar_reference(
                        expected,
                        sorts_played=sorts_played,
                        hand_sizes=hand_sizes,
                        player_index=player_index,
                        direction=direction,
                        is_change_sort=is_change_sort,
                    )
                    returned = populate_game_context(
                        actual,
                        sorts_played=sorts_played,
                        hand_sizes=hand_sizes,
                        player_index=player_index,
                        direction=direction,
                        is_change_sort=is_change_sort,
                    )
                    assert returned is actual
                    assert _exact(actual, expected)


@pytest.mark.parametrize("kind", ("list", "numpy", "torch"))
def test_invalid_direction_preserves_reference_partial_mutation(kind):
    initial = [99.0] * OBSERVATION_SIZE
    expected = _container(kind, initial)
    actual = _container(kind, initial)
    arguments = {
        "sorts_played": [1, 2, 3, 4, 5],
        "hand_sizes": [7, 6],
        "player_index": 0,
        "direction": 0,
    }
    with pytest.raises(ValueError, match="direction"):
        scalar_reference(expected, **arguments)
    with pytest.raises(ValueError, match="direction"):
        populate_game_context(actual, **arguments)
    assert _exact(actual, expected)


def test_frozen_best_logits_and_actions_are_bit_exact():
    rng = np.random.RandomState(89_607_131)
    expected_observations = []
    actual_observations = []
    for case_index in range(32):
        initial = rng.standard_normal(OBSERVATION_SIZE).astype(np.float32)
        initial[COMPATIBLE_START:COMPATIBLE_END] = (
            initial[COMPATIBLE_START:COMPATIBLE_END] > 0
        )
        arguments = {
            "sorts_played": rng.randint(0, 30, size=5).tolist(),
            "hand_sizes": rng.randint(
                1,
                20,
                size=2 + case_index % 6,
            ).tolist(),
            "player_index": case_index % (2 + case_index % 6),
            "direction": -1 if case_index % 2 else 1,
            "is_change_sort": bool(case_index % 3 == 0),
        }
        expected = torch.from_numpy(initial.copy())
        actual = torch.from_numpy(initial.copy())
        scalar_reference(expected, **arguments)
        populate_game_context(actual, **arguments)
        expected_observations.append(expected)
        actual_observations.append(actual)

    expected_batch = torch.stack(expected_observations)
    actual_batch = torch.stack(actual_observations)
    assert torch.equal(actual_batch, expected_batch)
    loaded = load_best_player(BestPlayerConfig(device="cpu"))
    with torch.inference_mode():
        expected_logits = loaded.actor(expected_batch)
        actual_logits = loaded.actor(actual_batch)
    assert torch.equal(actual_logits, expected_logits)
    expected_actions = expected_logits.masked_fill(
        ~build_action_masks(expected_batch),
        -torch.inf,
    ).argmax(-1)
    actual_actions = actual_logits.masked_fill(
        ~build_action_masks(actual_batch),
        -torch.inf,
    ).argmax(-1)
    assert torch.equal(actual_actions, expected_actions)


@pytest.mark.parametrize("kind", ("numpy", "torch"))
@pytest.mark.parametrize("trailing_width", (1, 3))
def test_vectorized_context_preserves_scalar_row_broadcasting(
    kind,
    trailing_width,
):
    shape = (OBSERVATION_SIZE, trailing_width)
    initial = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    expected = initial.copy()
    actual = initial.copy()
    if kind == "torch":
        expected = torch.from_numpy(expected)
        actual = torch.from_numpy(actual)
    arguments = {
        "sorts_played": [1, 2, 3, 4, 5],
        "hand_sizes": [8, 6, 4],
        "player_index": 1,
        "direction": -1,
        "is_change_sort": True,
    }

    scalar_reference(expected, **arguments)
    populate_game_context(actual, **arguments)

    assert _exact(actual, expected)


def test_generic_integer_indexed_sequences_preserve_scalar_contract():
    initial = np.arange(OBSERVATION_SIZE, dtype=np.float32).tolist()
    expected = IntegerOnlySequence(initial)
    actual = IntegerOnlySequence(initial)
    sorts_played = IntegerOnlySequence([9, 7, 5, 3, 1, 99])
    arguments = {
        "sorts_played": sorts_played,
        "hand_sizes": IntegerOnlySequence([4, 8, 6]),
        "player_index": 2,
        "direction": -1,
        "is_change_sort": False,
    }

    scalar_reference(expected, **arguments)
    returned = populate_game_context(actual, **arguments)

    assert returned is actual
    assert actual.values == expected.values
