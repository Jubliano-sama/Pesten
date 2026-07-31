"""Native PufferLib 4 MinGRU policy for information-safe Pesten inputs.

This module intentionally imports the pinned v4 API and therefore belongs in
the isolated ``puffer4-native`` runtime.  The separate fallback implementation
in ``temporal_puffer_agent.py`` remains usable by the legacy environment.
"""

from __future__ import annotations

import torch
from torch import nn

import pufferlib
import pufferlib.models

from temporal_observation import (
    DecisionInput,
    EVENT_TOKEN_SIZE,
    OBSERVATION_SIZE,
)
from temporal_puffer_agent import PolicyStep


PUFFERLIB_REVISION = "c5d3c637446047a6efbcaa74c039c5295d201ab0"
NATIVE_INPUT_SIZE = OBSERVATION_SIZE + EVENT_TOKEN_SIZE + 1


def require_pufferlib_v4() -> None:
    version = str(getattr(pufferlib, "__version__", ""))
    if not version.startswith("4"):
        raise RuntimeError(
            f"native temporal policy requires PufferLib 4, found {version!r}"
        )


class NativePufferTemporalPolicy(nn.Module):
    """Real ``Policy`` + ``MinGRU`` v4 model behind the safe schema boundary."""

    native_pufferlib = True
    pufferlib_revision = PUFFERLIB_REVISION
    integration_status = (
        "Native PufferLib 4.0 Policy/MinGRU at "
        f"{PUFFERLIB_REVISION}"
    )

    def __init__(
        self,
        hidden_size: int = 96,
        num_layers: int = 1,
        *,
        mode_conditioned_heads: bool = False,
    ) -> None:
        super().__init__()
        require_pufferlib_v4()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.mode_conditioned_heads = mode_conditioned_heads
        encoder = pufferlib.models.DefaultEncoder(
            NATIVE_INPUT_SIZE, hidden_size
        )
        decoder = (
            ModeConditionedDecoder(hidden_size)
            if mode_conditioned_heads
            else pufferlib.models.DefaultDecoder((55,), hidden_size)
        )
        network = pufferlib.models.MinGRU(
            hidden_size=hidden_size, num_layers=num_layers
        )
        self.puffer_policy = pufferlib.models.Policy(
            encoder, decoder, network
        )

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, ...]:
        parameter = next(self.parameters())
        return self.puffer_policy.initial_state(
            batch_size, device or parameter.device
        )

    @staticmethod
    def _event_features(decision: DecisionInput) -> list[tuple[float, ...]]:
        rows = []
        for event in decision.public_events:
            event_values = (
                int(event.kind) / 3.0,
                event.actor_relative / 12.0,
                (event.card_id + 1) / 50.0,
                (event.suit_id + 1) / 5.0,
                event.count / 54.0,
            )
            rows.append(
                (0.0,) * OBSERVATION_SIZE + event_values + (0.0,)
            )
        return rows

    @staticmethod
    def _decision_features(decision: DecisionInput) -> tuple[float, ...]:
        return (
            decision.observation.values
            + (0.0,) * EVENT_TOKEN_SIZE
            + (1.0,)
        )

    def step(
        self,
        decision: DecisionInput,
        state: tuple[torch.Tensor, ...],
    ) -> PolicyStep:
        """Advance native MinGRU over public events, then the private decision."""
        if not isinstance(decision, DecisionInput):
            raise TypeError("native policy accepts only DecisionInput")
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        next_state = state
        for row in self._event_features(decision):
            token = torch.tensor(row, device=device, dtype=dtype).unsqueeze(0)
            _, _, next_state = self.puffer_policy.forward_eval(
                token, next_state
            )

        token = torch.tensor(
            self._decision_features(decision),
            device=device,
            dtype=dtype,
        ).unsqueeze(0)
        logits, values, next_state = self.puffer_policy.forward_eval(
            token, next_state
        )
        legal = torch.tensor(
            decision.legal_actions, device=device, dtype=torch.bool
        ).unsqueeze(0)
        masked_logits = logits.masked_fill(
            ~legal, torch.finfo(logits.dtype).min
        )
        return PolicyStep(
            logits=masked_logits.squeeze(0),
            value=values.reshape(-1)[0],
            state=next_state,
        )

    def forward_native_sequence(
        self, safe_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expose the genuine v4 batched training path for safe token tensors."""
        if safe_tokens.ndim != 3 or safe_tokens.shape[-1] != NATIVE_INPUT_SIZE:
            raise ValueError(
                "safe_tokens must have shape [batch, time, "
                f"{NATIVE_INPUT_SIZE}]"
            )
        return self.puffer_policy(safe_tokens)


class ModeConditionedDecoder(nn.Module):
    """Separate card/pass and four-way suit heads in the 55-action contract."""

    nvec = (55,)
    is_continuous = False

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.card_and_pass = nn.Linear(hidden_size, 51)
        self.suit = nn.Linear(hidden_size, 4)
        self.value_function = nn.Linear(hidden_size, 1)

    def forward(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        card_and_pass = self.card_and_pass(hidden)
        logits = torch.cat(
            (
                card_and_pass[..., :50],
                self.suit(hidden),
                card_and_pass[..., 50:51],
            ),
            dim=-1,
        )
        return logits, self.value_function(hidden)
