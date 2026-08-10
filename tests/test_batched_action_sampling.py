import pytest
import torch
from torch.distributions import Categorical

from batched_pesten import _sample_actions_and_log_probabilities
from ppo2 import action_log_probabilities


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 2, 7, 64, 256])
@pytest.mark.parametrize("greedy", [False, True])
def test_direct_action_sampling_matches_categorical_exactly(
    dtype: torch.dtype,
    batch_size: int,
    greedy: bool,
) -> None:
    generator = torch.Generator().manual_seed(17 + batch_size)
    logits = torch.randn(
        batch_size,
        55,
        generator=generator,
        dtype=dtype,
    )
    masks = torch.rand(
        batch_size,
        55,
        generator=generator,
    ) > 0.3
    masks[:, -1] = True
    scaled = logits.masked_fill(
        ~masks,
        torch.finfo(dtype).min,
    ) / 0.9

    torch.manual_seed(9000 + batch_size)
    distribution = Categorical(logits=scaled, validate_args=False)
    expected_actions = (
        torch.argmax(scaled, dim=-1)
        if greedy
        else distribution.sample()
    )
    expected_log_probabilities = distribution.log_prob(expected_actions)
    expected_rng_state = torch.random.get_rng_state()

    torch.manual_seed(9000 + batch_size)
    actual_actions, actual_log_probabilities = (
        _sample_actions_and_log_probabilities(
            scaled,
            greedy=greedy,
        )
    )
    actual_rng_state = torch.random.get_rng_state()

    assert torch.equal(actual_actions, expected_actions)
    assert torch.equal(
        actual_log_probabilities,
        expected_log_probabilities,
    )
    assert torch.equal(actual_rng_state, expected_rng_state)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch_size", [1, 7, 256])
def test_direct_update_log_probabilities_match_categorical_exactly(
    dtype: torch.dtype,
    batch_size: int,
) -> None:
    generator = torch.Generator().manual_seed(85_132_000 + batch_size)
    logits = torch.randn(
        batch_size,
        55,
        generator=generator,
        dtype=dtype,
    )
    actions = torch.randint(
        0,
        55,
        (batch_size,),
        generator=generator,
    )
    expected = Categorical(
        logits=logits,
        validate_args=False,
    ).log_prob(actions)
    actual = action_log_probabilities(logits, actions)
    assert torch.equal(actual, expected)

