"""Scale-robust aggregation over all 24 ordinary-suit views."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ai_observation import ACTION_COUNT, OBSERVATION_SIZE
from batched_suit_symmetry import (
    SUIT_PERMUTATIONS,
    BatchedSuitAveragedActor,
)
from ppo2 import build_action_masks


ROBUST_AGGREGATIONS = ("majority_vote", "mean_borda")


def aggregate_mapped_logits(
    mapped_logits: torch.Tensor,
    legal_actions: torch.Tensor,
    *,
    aggregation: str,
) -> torch.Tensor:
    """Aggregate mapped orbit logits with legal, scale-robust scores."""
    if aggregation not in ROBUST_AGGREGATIONS:
        raise ValueError(f"unknown robust aggregation: {aggregation}")
    if (
        mapped_logits.ndim != 3
        or mapped_logits.shape[1] != len(SUIT_PERMUTATIONS)
        or mapped_logits.shape[2] != ACTION_COUNT
    ):
        raise ValueError("expected mapped logits shaped (batch, 24, 55)")
    if legal_actions.shape != (
        mapped_logits.shape[0],
        ACTION_COUNT,
    ):
        raise ValueError("legal mask shape does not match mapped logits")
    floor = torch.finfo(mapped_logits.dtype).min
    masked = mapped_logits.masked_fill(
        ~legal_actions.unsqueeze(1), floor
    )
    mean_logits = mapped_logits.mean(dim=1).masked_fill(
        ~legal_actions, floor
    )
    # Bounded tie-break: a one-vote or one-Borda-point lead always dominates.
    tie_break = F.softmax(mean_logits, dim=-1) * 0.25

    if aggregation == "majority_vote":
        votes = masked.argmax(dim=-1)
        counts = F.one_hot(votes, num_classes=ACTION_COUNT).sum(dim=1)
        scores = counts.to(mapped_logits.dtype) + tie_break
        return scores.masked_fill(~legal_actions, floor)

    # Descending rank per view.  Stable sorting makes exact ties deterministic;
    # the final bounded mean-logit term resolves any aggregate Borda tie.
    order = torch.argsort(masked, dim=-1, descending=True, stable=True)
    ranks = torch.empty_like(order)
    rank_values = torch.arange(
        ACTION_COUNT,
        device=mapped_logits.device,
        dtype=order.dtype,
    ).view(1, 1, -1).expand_as(order)
    ranks.scatter_(2, order, rank_values)
    legal_counts = legal_actions.sum(dim=-1).view(-1, 1, 1)
    points = (legal_counts - ranks).clamp_min(0).to(mapped_logits.dtype)
    points = points.masked_fill(~legal_actions.unsqueeze(1), 0.0)
    scores = points.mean(dim=1) + tie_break
    return scores.masked_fill(~legal_actions, floor)


class RobustSuitAggregationActor(nn.Module):
    """Evaluate all suit views and aggregate legal actions robustly."""

    def __init__(
        self,
        actor: nn.Module,
        *,
        aggregation: str,
        base_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        if aggregation not in ROBUST_AGGREGATIONS:
            raise ValueError(f"unknown robust aggregation: {aggregation}")
        if base_chunk_size < 1:
            raise ValueError("base_chunk_size must be positive")
        self.actor = actor
        self.aggregation = aggregation
        self.base_chunk_size = int(base_chunk_size)
        helper = BatchedSuitAveragedActor(nn.Identity(), alpha=0.0)
        self.register_buffer(
            "observation_indices",
            helper.observation_indices,
            persistent=False,
        )
        self.register_buffer(
            "action_indices",
            helper.action_indices,
            persistent=False,
        )
        self.decisions = 0

    def _mapped_orbit(self, observations: torch.Tensor) -> torch.Tensor:
        transformed = torch.gather(
            observations.unsqueeze(1).expand(
                -1, len(SUIT_PERMUTATIONS), -1
            ),
            dim=2,
            index=self.observation_indices.unsqueeze(0).expand(
                len(observations), -1, -1
            ),
        )
        transformed_logits = self.actor(
            transformed.reshape(-1, OBSERVATION_SIZE)
        ).reshape(
            len(observations),
            len(SUIT_PERMUTATIONS),
            ACTION_COUNT,
        )
        return torch.gather(
            transformed_logits,
            dim=2,
            index=self.action_indices.unsqueeze(0).expand(
                len(observations), -1, -1
            ),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if (
            observations.ndim != 2
            or observations.shape[1] != OBSERVATION_SIZE
        ):
            raise ValueError(
                f"expected observations shaped (batch, {OBSERVATION_SIZE})"
            )
        parts = []
        for start in range(0, len(observations), self.base_chunk_size):
            chunk = observations[start : start + self.base_chunk_size]
            mapped = self._mapped_orbit(chunk)
            parts.append(
                aggregate_mapped_logits(
                    mapped,
                    build_action_masks(chunk),
                    aggregation=self.aggregation,
                )
            )
            self.decisions += len(chunk)
        if not parts:
            return observations.new_empty((0, ACTION_COUNT))
        return torch.cat(parts)

    def statistics(self) -> dict:
        return {
            "aggregation": self.aggregation,
            "ordinary_suit_views": len(SUIT_PERMUTATIONS),
            "decisions": self.decisions,
            "information": "current 121-feature observation and logits only",
        }
