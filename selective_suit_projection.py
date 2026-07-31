"""Information-safe selective inference over the champion's suit orbit.

The selector sees only the existing 121-feature observation and diagnostics
computed from the frozen champion's current logits.  It has no game, seed,
opponent, recurrent state, or hidden-card input.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from ai_observation import ACTION_COUNT, OBSERVATION_SIZE
from batched_suit_symmetry import (
    SUIT_PERMUTATIONS,
    BatchedSuitAveragedActor,
)
from ppo2 import build_action_masks


@dataclass(frozen=True)
class SelectiveSuitRule:
    """A small deployable gate for changed orbit decisions."""

    name: str
    minimum_vote_agreement: float
    maximum_legacy_margin: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_vote_agreement <= 1.0:
            raise ValueError("minimum_vote_agreement must be in [0, 1]")
        if self.maximum_legacy_margin < 0:
            raise ValueError("maximum_legacy_margin cannot be negative")

    def to_dict(self) -> dict:
        result = asdict(self)
        if math.isinf(self.maximum_legacy_margin):
            result["maximum_legacy_margin"] = "infinity"
        return result


# Fixed before development evaluation.  Six rules span confidence-only and
# confidence-plus-low-legacy-margin gates without a per-seed search surface.
PREDECLARED_RULE_GRID = (
    SelectiveSuitRule("vote50_any_margin", 0.50, math.inf),
    SelectiveSuitRule("vote67_any_margin", 2.0 / 3.0, math.inf),
    SelectiveSuitRule("vote75_any_margin", 0.75, math.inf),
    SelectiveSuitRule("vote50_margin2", 0.50, 2.0),
    SelectiveSuitRule("vote67_margin2", 2.0 / 3.0, 2.0),
    SelectiveSuitRule("vote67_margin5", 2.0 / 3.0, 5.0),
)


@dataclass(frozen=True)
class SuitProjectionDiagnostics:
    legacy_logits: torch.Tensor
    orbit_logits: torch.Tensor
    legacy_action: torch.Tensor
    orbit_action: torch.Tensor
    action_changed: torch.Tensor
    legacy_margin: torch.Tensor
    orbit_vote_agreement: torch.Tensor
    orbit_delta_variance: torch.Tensor


class SelectiveSuitProjectionActor(nn.Module):
    """Choose legacy or full-orbit logits using only current safe diagnostics."""

    def __init__(
        self,
        actor: nn.Module,
        rule: SelectiveSuitRule,
        *,
        base_chunk_size: int = 512,
    ) -> None:
        super().__init__()
        if base_chunk_size < 1:
            raise ValueError("base_chunk_size must be positive")
        self.actor = actor
        self.rule = rule
        self.base_chunk_size = int(base_chunk_size)
        helper = BatchedSuitAveragedActor(
            nn.Identity(),
            alpha=0.0,
        )
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
        self.reset_statistics()

    def reset_statistics(self) -> None:
        self.decisions = 0
        self.changed_decisions = 0
        self.projected_decisions = 0
        self.vote_sum = 0.0
        self.margin_sum = 0.0
        self.delta_variance_sum = 0.0

    def statistics(self) -> dict:
        return {
            "rule": self.rule.to_dict(),
            "decisions": self.decisions,
            "changed_decisions": self.changed_decisions,
            "projected_decisions": self.projected_decisions,
            "changed_fraction": (
                self.changed_decisions / self.decisions
                if self.decisions
                else 0.0
            ),
            "projected_fraction": (
                self.projected_decisions / self.decisions
                if self.decisions
                else 0.0
            ),
            "mean_vote_agreement": (
                self.vote_sum / self.decisions if self.decisions else 0.0
            ),
            "mean_legacy_margin": (
                self.margin_sum / self.decisions if self.decisions else 0.0
            ),
            "mean_orbit_delta_variance": (
                self.delta_variance_sum / self.decisions
                if self.decisions
                else 0.0
            ),
        }

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

    def diagnostics(
        self, observations: torch.Tensor
    ) -> SuitProjectionDiagnostics:
        if (
            observations.ndim != 2
            or observations.shape[1] != OBSERVATION_SIZE
        ):
            raise ValueError(
                f"expected observations shaped (batch, {OBSERVATION_SIZE})"
            )
        legacy_logits = self.actor(observations)
        mapped_orbit = self._mapped_orbit(observations)
        orbit_logits = mapped_orbit.mean(dim=1)
        legal = build_action_masks(observations)
        floor = torch.finfo(legacy_logits.dtype).min
        masked_legacy = legacy_logits.masked_fill(~legal, floor)
        masked_orbit = orbit_logits.masked_fill(~legal, floor)
        legacy_action = masked_legacy.argmax(dim=-1)
        orbit_action = masked_orbit.argmax(dim=-1)
        action_changed = legacy_action != orbit_action

        top_two = torch.topk(masked_legacy, k=2, dim=-1).values
        legacy_margin = top_two[:, 0] - top_two[:, 1]
        legal_counts = legal.sum(dim=-1)
        legacy_margin = torch.where(
            legal_counts >= 2,
            legacy_margin,
            torch.full_like(legacy_margin, math.inf),
        )
        orbit_votes = mapped_orbit.masked_fill(
            ~legal.unsqueeze(1), floor
        ).argmax(dim=-1)
        orbit_vote_agreement = (
            orbit_votes == orbit_action.unsqueeze(1)
        ).float().mean(dim=1)

        gather_orbit = orbit_action.view(-1, 1, 1).expand(-1, 24, 1)
        gather_legacy = legacy_action.view(-1, 1, 1).expand(-1, 24, 1)
        orbit_delta = (
            torch.gather(mapped_orbit, 2, gather_orbit).squeeze(-1)
            - torch.gather(mapped_orbit, 2, gather_legacy).squeeze(-1)
        )
        orbit_delta_variance = orbit_delta.var(dim=1, unbiased=False)
        return SuitProjectionDiagnostics(
            legacy_logits=legacy_logits,
            orbit_logits=orbit_logits,
            legacy_action=legacy_action,
            orbit_action=orbit_action,
            action_changed=action_changed,
            legacy_margin=legacy_margin,
            orbit_vote_agreement=orbit_vote_agreement,
            orbit_delta_variance=orbit_delta_variance,
        )

    def selection_mask(
        self, diagnostics: SuitProjectionDiagnostics
    ) -> torch.Tensor:
        return (
            diagnostics.action_changed
            & (
                diagnostics.orbit_vote_agreement
                >= self.rule.minimum_vote_agreement
            )
            & (
                diagnostics.legacy_margin
                <= self.rule.maximum_legacy_margin
            )
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        parts = []
        for start in range(0, len(observations), self.base_chunk_size):
            chunk = observations[start : start + self.base_chunk_size]
            diagnostics = self.diagnostics(chunk)
            selected = self.selection_mask(diagnostics)
            parts.append(
                torch.where(
                    selected.unsqueeze(1),
                    diagnostics.orbit_logits,
                    diagnostics.legacy_logits,
                )
            )
            self.decisions += len(chunk)
            self.changed_decisions += int(
                diagnostics.action_changed.sum().item()
            )
            self.projected_decisions += int(selected.sum().item())
            self.vote_sum += float(
                diagnostics.orbit_vote_agreement.sum().item()
            )
            finite_margins = diagnostics.legacy_margin.masked_fill(
                ~torch.isfinite(diagnostics.legacy_margin), 0.0
            )
            self.margin_sum += float(finite_margins.sum().item())
            self.delta_variance_sum += float(
                diagnostics.orbit_delta_variance.sum().item()
            )
        if not parts:
            return observations.new_empty((0, ACTION_COUNT))
        return torch.cat(parts)
