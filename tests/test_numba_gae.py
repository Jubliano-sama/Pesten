import numpy as np
import pytest
import torch

from numba_gae import (
    compute_gae_native,
    native_gae_available,
    warm_native_gae,
)
from ppo2 import PPO_Supervised, compute_gae


pytestmark = pytest.mark.skipif(
    not native_gae_available(),
    reason="numba is unavailable",
)


@pytest.mark.parametrize(
    "lengths",
    (
        [1],
        [0, 3, 1, 17, 0, 9],
        [127, 2, 63, 4],
    ),
)
def test_native_gae_is_bit_exact_on_cpu(lengths) -> None:
    sample_count = sum(lengths)
    generator = torch.Generator().manual_seed(85_120_003)
    rewards = torch.randn(sample_count, generator=generator)
    values = torch.randn(sample_count, generator=generator)

    expected = compute_gae(
        rewards,
        values,
        episode_lengths=lengths,
        gamma_=0.97,
        lam=0.95,
    )
    actual = compute_gae_native(
        rewards,
        values,
        lengths,
        gamma=0.97,
        lam=0.95,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_native_gae_validates_segment_coverage_and_dtype() -> None:
    values = torch.zeros(3)
    with pytest.raises(ValueError, match="cover every reward"):
        compute_gae_native(
            values,
            values,
            [2],
            gamma=0.97,
            lam=0.95,
        )
    with pytest.raises(ValueError, match="float32"):
        compute_gae_native(
            values.double(),
            values.double(),
            [3],
            gamma=0.97,
            lam=0.95,
        )


def test_ppo_warms_native_gae_before_training() -> None:
    assert warm_native_gae()
    ppo = PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        native_gae=True,
        native_smart_policy=False,
        native_rollout_engine=False,
        trace_cpu_rollout_actor=False,
        torch_threads=1,
    )
    assert ppo.native_gae


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_native_gae_cuda_round_trip_matches_tensor_path() -> None:
    rng = np.random.default_rng(85_120_004)
    lengths = [3, 19, 1, 71]
    rewards = torch.from_numpy(
        rng.normal(size=sum(lengths)).astype(np.float32)
    ).cuda()
    values = torch.from_numpy(
        rng.normal(size=sum(lengths)).astype(np.float32)
    ).cuda()

    expected = compute_gae(
        rewards,
        values,
        episode_lengths=lengths,
        gamma_=0.97,
        lam=0.95,
    )
    actual = compute_gae_native(
        rewards,
        values,
        lengths,
        gamma=0.97,
        lam=0.95,
    )

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
