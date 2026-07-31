import inspect

import torch

from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_START,
    OBSERVATION_SIZE,
)
from robust_suit_aggregation import (
    RobustSuitAggregationActor,
    aggregate_mapped_logits,
)
from suit_symmetry_ppo_agent import (
    map_permuted_logits_back,
    permute_suit_observation,
)


def legal_mask() -> torch.Tensor:
    mask = torch.zeros(1, ACTION_COUNT, dtype=torch.bool)
    mask[0, :3] = True
    return mask


def test_majority_ignores_one_unbounded_logit_outlier():
    mapped = torch.zeros(1, 24, ACTION_COUNT)
    mapped[:, :23, 0] = 1.0
    mapped[:, :23, 1] = 0.9
    mapped[:, 23, 1] = 1_000_000.0
    majority = aggregate_mapped_logits(
        mapped, legal_mask(), aggregation="majority_vote"
    )
    assert majority.argmax(dim=-1).item() == 0
    assert mapped.mean(dim=1).argmax(dim=-1).item() == 1


def test_borda_is_invariant_to_positive_per_view_scale_and_shift():
    generator = torch.Generator().manual_seed(17)
    mapped = torch.randn(2, 24, ACTION_COUNT, generator=generator)
    legal = torch.zeros(2, ACTION_COUNT, dtype=torch.bool)
    legal[:, :7] = True
    original = aggregate_mapped_logits(
        mapped, legal, aggregation="mean_borda"
    )
    scales = torch.rand(2, 24, 1, generator=generator) * 20.0 + 0.1
    shifts = torch.randn(2, 24, 1, generator=generator) * 100.0
    transformed = aggregate_mapped_logits(
        mapped * scales + shifts,
        legal,
        aggregation="mean_borda",
    )
    assert torch.equal(
        original.argmax(dim=-1), transformed.argmax(dim=-1)
    )


def test_robust_actor_is_full_suit_equivariant():
    torch.manual_seed(23)
    base = torch.nn.Linear(OBSERVATION_SIZE, ACTION_COUNT)
    observation = torch.randn(OBSERVATION_SIZE)
    observation[COMPATIBLE_START : COMPATIBLE_START + 48] = 1.0
    permutation = (2, 0, 3, 1)
    for aggregation in ("majority_vote", "mean_borda"):
        actor = RobustSuitAggregationActor(
            base, aggregation=aggregation
        )
        canonical = actor(observation.unsqueeze(0)).squeeze(0)
        transformed = actor(
            permute_suit_observation(
                observation, permutation
            ).unsqueeze(0)
        ).squeeze(0)
        torch.testing.assert_close(
            map_permuted_logits_back(transformed, permutation),
            canonical,
            atol=2e-6,
            rtol=2e-6,
        )


def test_actor_api_has_no_seed_game_or_hidden_inputs():
    parameters = set(
        inspect.signature(RobustSuitAggregationActor.forward).parameters
    )
    assert parameters == {"self", "observations"}
    assert not parameters & {
        "seed",
        "game",
        "opponent_cards",
        "draw_pile",
    }
