import numpy as np
import pytest
import torch

from ai_observation import (
    COMPATIBLE_END,
    COMPATIBLE_START,
    MODE_INDEX,
    OBSERVATION_SIZE,
)
from batched_pesten import _action_masks, _action_masks_numpy
from batched_pesten import BatchedPesten, POLICY_RL


@pytest.mark.parametrize("batch_size", [1, 2, 7, 24, 64, 256])
def test_numpy_action_masks_match_torch_exactly(batch_size: int) -> None:
    generator = np.random.default_rng(85_090_000 + batch_size)
    observations = generator.normal(
        size=(batch_size, OBSERVATION_SIZE)
    ).astype(np.float32)
    observations[:, COMPATIBLE_START:COMPATIBLE_END] = (
        generator.random((batch_size, 50)) > 0.7
    )
    observations[:, MODE_INDEX] = (
        generator.random(batch_size) > 0.9
    )

    expected = _action_masks(torch.from_numpy(observations))
    actual = torch.from_numpy(_action_masks_numpy(observations))
    assert torch.equal(actual, expected)


def test_observation_workspace_and_fused_masks_are_exact() -> None:
    games = 64
    engine = BatchedPesten(
        np.full((games, 3), POLICY_RL, dtype=np.int8),
        np.full((games, 3), "current", dtype=object),
        np.arange(85_091_000, 85_091_000 + games, dtype=np.uint32),
    )
    game_indices = np.arange(games, dtype=np.int64)
    suit_modes = game_indices % 7 == 0
    expected_observations = engine.observations(
        game_indices,
        suit_modes=suit_modes,
    )
    expected_masks = _action_masks_numpy(expected_observations)
    observations = np.empty_like(expected_observations)
    masks = np.empty((games, 55), dtype=np.bool_)

    returned = engine.observations_into(
        game_indices,
        observations,
        suit_modes=suit_modes,
        action_masks=masks,
    )

    assert returned is observations
    assert observations.tobytes() == expected_observations.tobytes()
    assert masks.tobytes() == expected_masks.tobytes()
