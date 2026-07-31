import inspect

import pytest
import torch

from ai_observation import ACTION_COUNT, OBSERVATION_SIZE
from batched_suit_symmetry import BatchedSuitAveragedActor
from ppo2 import StudentNet
from symmetry_distill import (
    DEVELOPMENT_SEED_LIMIT,
    _augment_by_suit,
    collect_public_observations,
    compute_orbit_teacher_logits,
    validate_development_seed_range,
)


def test_seed_range_is_consecutive_and_fails_closed_at_90m():
    validate_development_seed_range(87_000_001, 100)
    with pytest.raises(ValueError, match="below"):
        validate_development_seed_range(DEVELOPMENT_SEED_LIMIT - 1, 2)


def test_collection_api_cannot_receive_hidden_world_fields():
    parameters = set(inspect.signature(collect_public_observations).parameters)
    assert parameters == {"actor", "games", "seed", "device", "max_turns"}
    assert not parameters & {
        "opponent_cards",
        "draw_pile",
        "deck_order",
        "game",
    }


def test_suit_augmentation_relabels_observation_and_targets_exactly():
    helper = BatchedSuitAveragedActor(StudentNet(), alpha=0.0)
    observations = torch.arange(
        2 * OBSERVATION_SIZE, dtype=torch.float32
    ).reshape(2, OBSERVATION_SIZE)
    targets = torch.arange(
        2 * ACTION_COUNT, dtype=torch.float32
    ).reshape(2, ACTION_COUNT)
    rows = torch.tensor([0, 17])
    transformed_observations, transformed_targets = _augment_by_suit(
        observations, targets, rows
    )
    assert torch.equal(
        transformed_observations[1],
        observations[1, helper.observation_indices[17]],
    )
    for original_action, transformed_action in enumerate(
        helper.action_indices[17]
    ):
        assert transformed_targets[1, transformed_action] == targets[
            1, original_action
        ]
    assert torch.equal(transformed_observations[0], observations[0])
    assert torch.equal(transformed_targets[0], targets[0])


def test_orbit_targets_are_suit_equivariant():
    torch.manual_seed(9)
    champion = StudentNet().eval()
    helper = BatchedSuitAveragedActor(champion, alpha=0.0)
    observations = torch.randn(3, OBSERVATION_SIZE)
    canonical = compute_orbit_teacher_logits(
        observations,
        champion,
        device=torch.device("cpu"),
        chunk_size=3,
    )
    rows = torch.tensor([5, 11, 19])
    transformed_observations, transformed_targets = _augment_by_suit(
        observations, canonical, rows
    )
    recomputed = compute_orbit_teacher_logits(
        transformed_observations,
        champion,
        device=torch.device("cpu"),
        chunk_size=3,
    )
    torch.testing.assert_close(
        recomputed, transformed_targets, atol=2e-6, rtol=2e-6
    )
    assert helper.observation_indices.shape == (24, OBSERVATION_SIZE)
