"""Public-observation action-value residuals for a frozen Pesten actor."""

from __future__ import annotations

import torch
from torch import nn

from ai_observation import ACTION_COUNT, OBSERVATION_SIZE
from ppo2 import build_action_masks


class ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.layers = nn.Sequential(
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.layers(self.norm(inputs))


class ActionValueNet(nn.Module):
    """Estimate terminal return for every legal action from public state."""

    def __init__(self, *, width: int = 512, blocks: int = 4) -> None:
        super().__init__()
        if width < 1 or blocks < 1:
            raise ValueError("width and blocks must be positive")
        self.width = int(width)
        self.blocks = int(blocks)
        self.input = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, self.width),
            nn.LayerNorm(self.width),
            nn.SiLU(),
        )
        self.residual = nn.Sequential(
            *(ResidualBlock(self.width) for _ in range(self.blocks))
        )
        self.output = nn.Sequential(
            nn.LayerNorm(self.width),
            nn.SiLU(),
            nn.Linear(self.width, ACTION_COUNT),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.output(self.residual(self.input(observations)))


class FittedQResidualPolicy(nn.Module):
    """Add a fitted-Q preference correction to a frozen base actor."""

    def __init__(
        self,
        base_actor: nn.Module,
        action_value: ActionValueNet,
        *,
        scale: float,
        normalize: bool = True,
        confidence_gate: float | None = None,
    ) -> None:
        super().__init__()
        if scale < 0:
            raise ValueError("scale cannot be negative")
        if confidence_gate is not None and confidence_gate < 0:
            raise ValueError("confidence_gate cannot be negative")
        self.base_actor = base_actor
        self.action_value = action_value
        self.scale = float(scale)
        self.normalize = bool(normalize)
        self.confidence_gate = confidence_gate

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        base_logits = self.base_actor(observations)
        if self.scale == 0:
            return base_logits
        q_values = self.action_value(observations)
        masks = build_action_masks(observations)
        if self.normalize:
            legal_count = masks.sum(dim=-1, keepdim=True).clamp_min(1)
            legal_q = torch.where(masks, q_values, torch.zeros_like(q_values))
            mean = legal_q.sum(dim=-1, keepdim=True) / legal_count
            centered = torch.where(
                masks,
                q_values - mean,
                torch.zeros_like(q_values),
            )
            variance = centered.square().sum(dim=-1, keepdim=True) / legal_count
            q_values = centered / variance.sqrt().clamp_min(1e-3)
        if self.confidence_gate is not None:
            legal_base = base_logits.masked_fill(
                ~masks,
                torch.finfo(base_logits.dtype).min,
            )
            top_two = legal_base.topk(k=2, dim=-1).values
            margin = top_two[:, 0] - top_two[:, 1]
            enabled = (margin <= self.confidence_gate).unsqueeze(-1)
            q_values = torch.where(enabled, q_values, torch.zeros_like(q_values))
        return base_logits + self.scale * q_values

