"""Small public-state tactical preference layer for a frozen Pesten actor."""

from __future__ import annotations

import torch
from torch import nn

from ai_observation import PASS_ACTION, SUIT_ACTION_END, SUIT_ACTION_START
from batched_pesten import CARD_RANK


BIAS_DIMENSIONS = 19


class RankBiasPolicy(nn.Module):
    """Add global card-rank, suit-choice, and pass preferences to logits."""

    def __init__(self, base_actor: nn.Module, biases: torch.Tensor) -> None:
        super().__init__()
        if biases.shape != (BIAS_DIMENSIONS,):
            raise ValueError(
                f"expected {BIAS_DIMENSIONS} biases, got {tuple(biases.shape)}"
            )
        self.base_actor = base_actor
        self.register_buffer(
            "rank_indices",
            torch.as_tensor(CARD_RANK, dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "biases",
            biases.detach().clone().to(dtype=torch.float32),
            persistent=True,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        logits = self.base_actor(observations)
        additions = torch.zeros_like(logits)
        additions[:, : len(self.rank_indices)] = self.biases[
            self.rank_indices
        ]
        additions[:, SUIT_ACTION_START:SUIT_ACTION_END] = self.biases[14:18]
        additions[:, PASS_ACTION] = self.biases[18]
        return logits + additions

