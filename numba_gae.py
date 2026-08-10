"""Native segmented generalized-advantage estimation for PPO.

The recurrence is inherently sequential within each episode.  Running it in a
small Numba CPU kernel avoids one Python-dispatched tensor operation per time
step.  CUDA inputs make one bounded host round trip; for the small rollout
vectors used here that is still much cheaper than hundreds of CUDA launches.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised by the PPO fallback path.
    njit = None


if njit is not None:

    @njit(cache=True)
    def _segmented_gae(
        rewards: np.ndarray,
        values: np.ndarray,
        episode_lengths: np.ndarray,
        gamma: np.float32,
        lam: np.float32,
    ) -> np.ndarray:
        advantages = np.empty_like(values)
        episode_start = 0
        for episode_index in range(episode_lengths.shape[0]):
            episode_stop = (
                episode_start + episode_lengths[episode_index]
            )
            next_value = np.float32(0.0)
            last_advantage = np.float32(0.0)
            for sample_index in range(
                episode_stop - 1,
                episode_start - 1,
                -1,
            ):
                delta = (
                    rewards[sample_index]
                    + gamma * next_value
                    - values[sample_index]
                )
                last_advantage = (
                    delta + gamma * lam * last_advantage
                )
                advantages[sample_index] = last_advantage
                next_value = values[sample_index]
            episode_start = episode_stop
        return advantages

else:
    _segmented_gae = None


def native_gae_available() -> bool:
    """Return whether the optional native kernel can be used."""
    return _segmented_gae is not None


def warm_native_gae() -> bool:
    """Compile or load the one concrete signature used during training."""
    if _segmented_gae is None:
        return False
    zeros = np.zeros(1, dtype=np.float32)
    _segmented_gae(
        zeros,
        zeros,
        np.ones(1, dtype=np.int64),
        np.float32(0.97),
        np.float32(0.95),
    )
    return True


def compute_gae_native(
    rewards: torch.Tensor,
    values: torch.Tensor,
    episode_lengths,
    *,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE with the same float32 recurrence as the tensor path."""
    if _segmented_gae is None:
        raise RuntimeError("native GAE requires numba")
    if rewards.ndim != 1 or values.ndim != 1:
        raise ValueError("rewards and values must be one-dimensional")
    if len(rewards) != len(values):
        raise ValueError("rewards and values must have the same length")
    lengths = np.asarray(episode_lengths, dtype=np.int64)
    if lengths.ndim != 1:
        raise ValueError("episode lengths must be one-dimensional")
    if np.any(lengths < 0):
        raise ValueError("episode lengths cannot be negative")
    if int(lengths.sum()) != len(rewards):
        raise ValueError("episode lengths must cover every reward exactly once")
    if rewards.dtype != torch.float32 or values.dtype != torch.float32:
        raise ValueError("native GAE requires float32 rewards and values")
    if rewards.device != values.device:
        raise ValueError("rewards and values must share one device")
    if len(rewards) == 0:
        return torch.zeros_like(rewards), values.clone()

    rewards_np = rewards.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()
    advantages_np = _segmented_gae(
        rewards_np,
        values_np,
        lengths,
        np.float32(gamma),
        np.float32(lam),
    )
    advantages = torch.from_numpy(advantages_np).to(
        device=values.device,
    )
    return advantages, advantages + values
