"""Privacy-safe confidence gate for recurrent residual checkpoints.

The recurrent residual policy is always stepped first, so public temporal
memory advances even when its proposed action is rejected.  The gate then
reconstructs the frozen PPO's legacy observation solely from ``DecisionInput``
and permits a disagreeing residual action only when the frozen PPO's legal
top-two logit margin is below the configured threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import torch
from torch import nn

from benchmark_model_zoo import safe_to_legacy_observation
from residual_temporal_policy import ResidualRecurrentTemporalPolicy
from temporal_observation import DecisionInput
from temporal_puffer_agent import PolicyStep


@dataclass(frozen=True, slots=True)
class GateDecision:
    frozen_action: int
    residual_action: int
    selected_action: int
    frozen_top2_margin: float
    disagreed: bool
    residual_override_allowed: bool


class ConfidenceGatedResidualPolicy(nn.Module):
    """Inference wrapper around a frozen-PPO recurrent residual policy."""

    architecture = "confidence_gated_recurrent_residual"
    critic_privileged = False

    def __init__(
        self,
        residual_policy: ResidualRecurrentTemporalPolicy,
        *,
        margin_threshold: float,
    ) -> None:
        super().__init__()
        threshold = float(margin_threshold)
        if math.isnan(threshold) or threshold < 0.0:
            raise ValueError("margin_threshold must be non-negative")
        if not hasattr(residual_policy, "teacher"):
            raise TypeError("residual_policy must expose its frozen PPO teacher")
        self.residual_policy = residual_policy
        self.margin_threshold = threshold
        self.last_gate_decision: GateDecision | None = None
        self.reset_diagnostics()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: Path,
        *,
        champion_path: Path,
        margin_threshold: float,
        map_location: torch.device | str = "cpu",
    ) -> "ConfidenceGatedResidualPolicy":
        residual = ResidualRecurrentTemporalPolicy.load_checkpoint(
            checkpoint,
            champion_path=champion_path,
            map_location=map_location,
        ).eval()
        return cls(
            residual,
            margin_threshold=margin_threshold,
        ).eval()

    def initial_state(self, *args, **kwargs):
        return self.residual_policy.initial_state(*args, **kwargs)

    def reset_diagnostics(self) -> None:
        self.decisions = 0
        self.disagreements = 0
        self.allowed_overrides = 0
        self.blocked_overrides = 0
        self.margin_sum = 0.0
        self.finite_margin_count = 0
        self.last_gate_decision = None

    def diagnostics(self) -> dict:
        return {
            "margin_threshold": self.margin_threshold,
            "decisions": self.decisions,
            "disagreements": self.disagreements,
            "allowed_overrides": self.allowed_overrides,
            "blocked_overrides": self.blocked_overrides,
            "disagreement_rate": (
                self.disagreements / self.decisions if self.decisions else 0.0
            ),
            "override_rate": (
                self.allowed_overrides / self.decisions
                if self.decisions
                else 0.0
            ),
            "mean_finite_frozen_margin": (
                self.margin_sum / self.finite_margin_count
                if self.finite_margin_count
                else None
            ),
        }

    def _frozen_legal_logits(
        self,
        decision: DecisionInput,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        legacy = safe_to_legacy_observation(decision)
        teacher = self.residual_policy.teacher
        teacher_parameter = next(teacher.parameters())
        legacy = legacy.to(
            device=teacher_parameter.device,
            dtype=teacher_parameter.dtype,
        )
        with torch.inference_mode():
            logits = teacher(legacy.unsqueeze(0)).squeeze(0)
        logits = logits.to(device=device)
        legal = torch.tensor(
            decision.legal_actions,
            device=device,
            dtype=torch.bool,
        )
        return (
            logits.masked_fill(~legal, torch.finfo(logits.dtype).min),
            legal,
        )

    @staticmethod
    def _top2_margin(
        legal_logits: torch.Tensor,
        legal: torch.Tensor,
    ) -> float:
        legal_count = int(legal.sum().item())
        if legal_count < 2:
            return math.inf
        top_two = torch.topk(legal_logits[legal], k=2).values
        return float((top_two[0] - top_two[1]).item())

    def step(self, decision: DecisionInput, state) -> PolicyStep:
        if not isinstance(decision, DecisionInput):
            raise TypeError("confidence gate accepts only DecisionInput")

        # This must happen before any gate decision. Even a rejected residual
        # proposal advances its private recurrent state over the safe tokens.
        residual_result = self.residual_policy.step(decision, state)
        residual_logits = residual_result.logits
        frozen_logits, legal = self._frozen_legal_logits(
            decision,
            device=residual_logits.device,
        )
        frozen_logits = frozen_logits.to(dtype=residual_logits.dtype)

        frozen_action = int(torch.argmax(frozen_logits).item())
        residual_action = int(torch.argmax(residual_logits).item())
        disagreed = frozen_action != residual_action
        margin = self._top2_margin(frozen_logits, legal)
        allow_override = disagreed and margin < self.margin_threshold
        selected_logits = residual_logits if allow_override else frozen_logits
        selected_action = residual_action if allow_override else frozen_action

        self.decisions += 1
        if math.isfinite(margin):
            self.margin_sum += margin
            self.finite_margin_count += 1
        if disagreed:
            self.disagreements += 1
            if allow_override:
                self.allowed_overrides += 1
            else:
                self.blocked_overrides += 1
        self.last_gate_decision = GateDecision(
            frozen_action=frozen_action,
            residual_action=residual_action,
            selected_action=selected_action,
            frozen_top2_margin=margin,
            disagreed=disagreed,
            residual_override_allowed=allow_override,
        )
        return PolicyStep(
            logits=selected_logits,
            value=residual_result.value,
            state=residual_result.state,
        )

    def metadata(self) -> dict:
        decision = (
            asdict(self.last_gate_decision)
            if self.last_gate_decision is not None
            else None
        )
        return {
            "architecture": self.architecture,
            "information_boundary": (
                "DecisionInput plus typed public history and own hand only"
            ),
            "margin_rule": (
                "allow disagreeing residual argmax only when frozen PPO "
                "legal top-2 logit margin is strictly below threshold"
            ),
            "last_gate_decision": decision,
            **self.diagnostics(),
        }
