"""Compact, process-friendly temporal trajectories for persistent PPO replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from temporal_observation import EVENT_TOKEN_SIZE, OBSERVATION_SIZE


NATIVE_INPUT_SIZE = OBSERVATION_SIZE + EVENT_TOKEN_SIZE + 1


@dataclass(frozen=True)
class PackedPufferTrajectory:
    """All immutable training fields with safe tokens already encoded."""

    tokens: np.ndarray
    decision_indices: np.ndarray
    legal_masks: np.ndarray
    actions: np.ndarray
    old_log_probabilities: np.ndarray
    behavior_values: np.ndarray

    def __len__(self) -> int:
        return int(self.actions.size)

    @property
    def safe_token_count(self) -> int:
        return int(self.tokens.shape[0])


def pack_puffer_trajectory(
    policy,
    trajectory: Sequence,
) -> PackedPufferTrajectory:
    """Encode one safe trajectory once, preferably inside its rollout worker."""
    token_count = sum(
        len(step.decision.public_events) + 1 for step in trajectory
    )
    event_values = getattr(policy, "_event_values", None)
    tokens = np.zeros((token_count, NATIVE_INPUT_SIZE), dtype=np.float32)
    decision_indices = np.empty(len(trajectory), dtype=np.int32)
    legal_masks = np.empty((len(trajectory), 55), dtype=np.bool_)
    actions = np.empty(len(trajectory), dtype=np.int64)
    old_log_probabilities = np.empty(len(trajectory), dtype=np.float32)
    behavior_values = np.empty(len(trajectory), dtype=np.float32)
    token_cursor = 0
    for decision_index, step in enumerate(trajectory):
        events = step.decision.public_events
        event_end = token_cursor + len(events)
        if events:
            if event_values is not None:
                tokens[token_cursor:event_end, OBSERVATION_SIZE:-1] = [
                    event_values(event) for event in events
                ]
            else:
                tokens[token_cursor:event_end] = policy._event_features(step.decision)
        token_cursor = event_end
        tokens[token_cursor] = policy._decision_features(step.decision)
        decision_indices[decision_index] = token_cursor
        legal_masks[decision_index] = step.decision.legal_actions
        actions[decision_index] = step.action
        old_log_probabilities[decision_index] = step.old_log_prob
        behavior_values[decision_index] = step.value
        token_cursor += 1
    if token_cursor != token_count:
        raise RuntimeError("packed Puffer token count mismatch")
    return PackedPufferTrajectory(
        tokens=tokens,
        decision_indices=decision_indices,
        legal_masks=legal_masks,
        actions=actions,
        old_log_probabilities=old_log_probabilities,
        behavior_values=behavior_values,
    )


def pack_puffer_episode(policy, episode):
    trajectory, won = episode
    outcome = None if won is None else bool(won)
    return pack_puffer_trajectory(policy, trajectory), outcome
