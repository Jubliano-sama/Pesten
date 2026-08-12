"""Native PufferLib 4 MinGRU policy for information-safe Pesten inputs.

This module intentionally imports the pinned v4 API and therefore belongs in
the isolated ``puffer4-native`` runtime.  The separate fallback implementation
in ``temporal_puffer_agent.py`` remains usable by the legacy environment.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
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
        event_rows = self._event_features(decision)
        if event_rows:
            event_tokens = torch.tensor(
                event_rows,
                device=device,
                dtype=dtype,
            )
            for token in event_tokens:
                _, _, next_state = self.puffer_policy.forward_eval(
                    token.unsqueeze(0), next_state
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

    def model_config(self) -> dict[str, object]:
        return {
            "architecture": "min_gru",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "mode_conditioned_heads": self.mode_conditioned_heads,
        }


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


class RealGatedLinearRecurrentUnit(nn.Module):
    """Hawk-style stable diagonal recurrence with input-dependent forgetting.

    The recurrence follows the RG-LRU design from De et al. (2024): a
    learnable stable diagonal coefficient is modulated by an input gate, and
    the injected value is variance-normalized by ``sqrt(1 - a**2)``.  Its
    state is linear in width, which keeps online inference cheap.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        if width < 1:
            raise ValueError("RG-LRU width must be positive")
        self.width = width
        self.input_gate = nn.Linear(width, width)
        self.recurrence_gate = nn.Linear(width, width)
        # Spread initial memory constants from roughly 8 to 512 steps.  The
        # paper uses a stable sigmoid parameterization and c=8.
        target = torch.exp(
            -1.0
            / torch.logspace(
                math.log10(8.0), math.log10(512.0), width
            )
        ).clamp(1e-4, 1.0 - 1e-4)
        self.recurrence_logits = nn.Parameter(torch.logit(target))

    def _terms(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        recurrence_gate = torch.sigmoid(self.recurrence_gate(x))
        input_gate = torch.sigmoid(self.input_gate(x))
        log_base = F.logsigmoid(self.recurrence_logits)
        log_a = 8.0 * recurrence_gate * log_base
        a = torch.exp(log_a)
        injected = (
            torch.sqrt(torch.clamp(1.0 - a.square(), min=1e-6))
            * input_gate
            * x
        )
        return a, injected

    @staticmethod
    def _signed_parallel_scan(
        a: torch.Tensor, injected: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate ``h[t] = a[t] h[t-1] + injected[t]`` in parallel.

        Positive coefficients let us use two stable log-cumulative-exp scans
        for the positive and negative injected values.  This avoids a Python
        time loop and remains differentiable on ordinary PyTorch kernels.
        """
        log_a_prefix = torch.log(a).cumsum(dim=1)
        # A finite log-zero keeps backward stable when an entire prefix has
        # no values of one sign.  Identical floors cancel between the two
        # signed scans and are far below float32 policy precision.
        log_zero = math.log(torch.finfo(injected.dtype).tiny)
        negative_infinity = torch.full_like(injected, log_zero)
        positive_mask = injected > 0
        negative_mask = injected < 0
        positive_magnitude = torch.where(
            positive_mask, injected, torch.ones_like(injected)
        )
        negative_magnitude = torch.where(
            negative_mask, -injected, torch.ones_like(injected)
        )
        positive_logs = torch.where(
            positive_mask,
            torch.log(positive_magnitude),
            negative_infinity,
        )
        negative_logs = torch.where(
            negative_mask,
            torch.log(negative_magnitude),
            negative_infinity,
        )
        positive = torch.exp(
            log_a_prefix
            + torch.logcumsumexp(positive_logs - log_a_prefix, dim=1)
        )
        negative = torch.exp(
            log_a_prefix
            + torch.logcumsumexp(negative_logs - log_a_prefix, dim=1)
        )
        return positive - negative

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError("RG-LRU training input must be [batch,time,width]")
        a, injected = self._terms(x)
        return self._signed_parallel_scan(a, injected)

    def forward_eval(
        self, x: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.width:
            raise ValueError("RG-LRU eval input must be [batch,width]")
        if state.shape != x.shape:
            raise ValueError("RG-LRU state must match the eval input shape")
        a, injected = self._terms(x)
        next_state = a * state + injected
        return next_state, next_state


class HawkRGLRUBlock(nn.Module):
    """Inference-efficient recurrent residual block inspired by Hawk."""

    def __init__(
        self,
        hidden_size: int,
        recurrent_size: int,
        mlp_expansion: int,
    ) -> None:
        super().__init__()
        self.recurrent_norm = nn.RMSNorm(hidden_size)
        self.recurrent_projection = nn.Linear(
            hidden_size, 2 * recurrent_size, bias=False
        )
        self.rg_lru = RealGatedLinearRecurrentUnit(recurrent_size)
        self.recurrent_output = nn.Linear(
            recurrent_size, hidden_size, bias=False
        )
        self.mlp_norm = nn.RMSNorm(hidden_size)
        expanded = hidden_size * mlp_expansion
        self.mlp_projection = nn.Linear(
            hidden_size, 2 * expanded, bias=False
        )
        self.mlp_output = nn.Linear(expanded, hidden_size, bias=False)

    def _recurrent_input(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value, gate = self.recurrent_projection(
            self.recurrent_norm(hidden)
        ).chunk(2, dim=-1)
        return value, F.silu(gate)

    def _mlp(self, hidden: torch.Tensor) -> torch.Tensor:
        gate, value = self.mlp_projection(
            self.mlp_norm(hidden)
        ).chunk(2, dim=-1)
        return self.mlp_output(F.silu(gate) * value)

    def forward_train(self, hidden: torch.Tensor) -> torch.Tensor:
        value, gate = self._recurrent_input(hidden)
        recurrent = self.rg_lru.forward_train(value)
        hidden = hidden + self.recurrent_output(recurrent * gate)
        return hidden + self._mlp(hidden)

    def forward_eval(
        self, hidden: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value, gate = self._recurrent_input(hidden)
        recurrent, next_state = self.rg_lru.forward_eval(value, state)
        hidden = hidden + self.recurrent_output(recurrent * gate)
        return hidden + self._mlp(hidden), next_state


class HawkRGLRUNetwork(nn.Module):
    """Stacked Hawk-style blocks behind PufferLib's Policy interface."""

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        *,
        recurrent_size: int,
        mlp_expansion: int,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_size = recurrent_size
        self.layers = nn.ModuleList(
            HawkRGLRUBlock(
                hidden_size,
                recurrent_size,
                mlp_expansion,
            )
            for _ in range(num_layers)
        )
        self.final_norm = nn.RMSNorm(hidden_size)
        residual_scale = 1.0 / math.sqrt(max(1, 2 * num_layers))
        with torch.no_grad():
            for block in self.layers:
                block.recurrent_output.weight.mul_(residual_scale)
                block.mlp_output.weight.mul_(residual_scale)

    def initial_state(
        self, batch_size: int, device: torch.device | str
    ) -> tuple[torch.Tensor, ...]:
        return (
            torch.zeros(
                self.num_layers,
                batch_size,
                self.recurrent_size,
                device=device,
            ),
        )

    def forward_train(self, hidden: torch.Tensor) -> torch.Tensor:
        for block in self.layers:
            hidden = block.forward_train(hidden)
        return self.final_norm(hidden)

    def forward_eval(
        self,
        hidden: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        recurrent_state = state[0]
        if recurrent_state.shape[:2] != (self.num_layers, hidden.shape[0]):
            raise ValueError("Hawk recurrent state shape does not match batch")
        next_rows = []
        for index, block in enumerate(self.layers):
            hidden, next_state = block.forward_eval(
                hidden, recurrent_state[index]
            )
            next_rows.append(next_state)
        return self.final_norm(hidden), (torch.stack(next_rows, dim=0),)


class HawkRGLRUTemporalPolicy(NativePufferTemporalPolicy):
    """Hawk-inspired fixed-state alternative to the Puffer MinGRU policy."""

    integration_status = "Hawk-style RG-LRU policy on PufferLib 4 Policy API"

    def __init__(
        self,
        hidden_size: int = 256,
        num_layers: int = 4,
        *,
        recurrent_size: int | None = None,
        mlp_expansion: int = 2,
        mode_conditioned_heads: bool = True,
    ) -> None:
        nn.Module.__init__(self)
        require_pufferlib_v4()
        if hidden_size < 1 or num_layers < 1 or mlp_expansion < 1:
            raise ValueError("Hawk dimensions must be positive")
        recurrent_size = recurrent_size or math.ceil(4 * hidden_size / 3)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.recurrent_size = recurrent_size
        self.mlp_expansion = mlp_expansion
        self.mode_conditioned_heads = mode_conditioned_heads
        encoder = pufferlib.models.DefaultEncoder(
            NATIVE_INPUT_SIZE, hidden_size
        )
        decoder = (
            ModeConditionedDecoder(hidden_size)
            if mode_conditioned_heads
            else pufferlib.models.DefaultDecoder((55,), hidden_size)
        )
        network = HawkRGLRUNetwork(
            hidden_size,
            num_layers,
            recurrent_size=recurrent_size,
            mlp_expansion=mlp_expansion,
        )
        self.puffer_policy = pufferlib.models.Policy(
            encoder, decoder, network
        )

    def model_config(self) -> dict[str, object]:
        return {
            "architecture": "hawk_rglru",
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "recurrent_size": self.recurrent_size,
            "mlp_expansion": self.mlp_expansion,
            "mode_conditioned_heads": self.mode_conditioned_heads,
        }


def build_temporal_policy(model: Mapping[str, object]) -> nn.Module:
    """Construct a supported native safe-temporal policy from metadata."""
    architecture = str(model.get("architecture", "min_gru"))
    common = {
        "hidden_size": int(model.get("hidden_size", 96)),
        "num_layers": int(model.get("num_layers", 1)),
        "mode_conditioned_heads": bool(
            model.get("mode_conditioned_heads", False)
        ),
    }
    if architecture in {"min_gru", "native_min_gru"}:
        return NativePufferTemporalPolicy(**common)
    if architecture == "hawk_rglru":
        return HawkRGLRUTemporalPolicy(
            **common,
            recurrent_size=int(
                model.get(
                    "recurrent_size",
                    math.ceil(4 * common["hidden_size"] / 3),
                )
            ),
            mlp_expansion=int(model.get("mlp_expansion", 2)),
        )
    raise ValueError(f"unsupported temporal architecture: {architecture}")


def policy_model_config(policy: nn.Module) -> dict[str, object]:
    getter = getattr(policy, "model_config", None)
    if not callable(getter):
        raise TypeError("temporal policy must expose model_config()")
    return dict(getter())


def load_temporal_policy_checkpoint(
    path: Path | str,
    *,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict]:
    """Load old MinGRU and new architecture-tagged checkpoints uniformly."""
    checkpoint = Path(path)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("temporal checkpoint payload must be a mapping")
    state = payload.get("state_dict", payload)
    if not isinstance(state, Mapping):
        raise TypeError("temporal checkpoint state_dict must be a mapping")
    model = dict(payload.get("model", {}))
    if "hidden_size" not in model:
        model["hidden_size"] = int(
            state["puffer_policy.encoder.encoder.weight"].shape[0]
        )
    if "num_layers" not in model:
        model["num_layers"] = sum(
            key.startswith("puffer_policy.network.layers.")
            and key.endswith(".weight")
            for key in state
        )
    if "mode_conditioned_heads" not in model:
        model["mode_conditioned_heads"] = any(
            "card_and_pass" in key for key in state
        )
    model.setdefault("architecture", "min_gru")
    policy = build_temporal_policy(model).to(device)
    policy.load_state_dict(state, strict=True)
    return policy, dict(payload)
