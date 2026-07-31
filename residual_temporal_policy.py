"""Information-safe causal residual policy anchored to the frozen PPO champion.

The verified production PPO remains the exact base actor.  A causal Transformer
sees only the existing 100-value typed safe-token stream and predicts an
additive 55-action residual.  Its actor head is initialized to zero, so a new
model has bit-exact legal-masked logits and actions from the champion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from benchmark_model_zoo import (
    DEFAULT_TEACHER,
    KNOWN_TEACHER_SHA256,
    file_sha256,
    load_frozen_teacher,
    safe_batch_to_legacy_observations,
    safe_to_legacy_observation,
)
from temporal_model_zoo import (
    SAFE_TOKEN_SIZE,
    SafeSequenceBatch,
    SafeTemporalPolicy,
    decision_safe_tokens,
)
from temporal_observation import ACTION_COUNT, DecisionInput, OBSERVATION_SIZE
from temporal_puffer_agent import PolicyStep


CHECKPOINT_FORMAT = "pesten-residual-temporal-v1"
RECURRENT_CHECKPOINT_FORMAT = "pesten-residual-recurrent-v1"
LEGACY_OBSERVATION_SIZE = 121


class ResidualCausalTemporalPolicy(SafeTemporalPolicy):
    """Frozen exact PPO logits plus a trainable safe-history residual."""

    architecture = "frozen_ppo_plus_causal_transformer_residual"
    critic_privileged = False

    def __init__(
        self,
        *,
        champion_path: Path = DEFAULT_TEACHER,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward_size: int = 384,
        dropout: float = 0.0,
        max_context: int = 384,
        residual_scale: float = 1.0,
        logit_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if num_layers < 1 or max_context < 1:
            raise ValueError("num_layers and max_context must be positive")
        if residual_scale <= 0 or logit_temperature <= 0:
            raise ValueError(
                "residual_scale and logit_temperature must be positive"
            )
        champion_path = Path(champion_path)
        digest = file_sha256(champion_path)
        if digest != KNOWN_TEACHER_SHA256:
            raise RuntimeError(
                "residual policy requires the verified PPO champion: "
                f"expected {KNOWN_TEACHER_SHA256}, got {digest}"
            )
        teacher, _ = load_frozen_teacher(champion_path)
        self.teacher = teacher
        self.champion_path = str(champion_path)
        self.champion_sha256 = digest
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.feedforward_size = feedforward_size
        self.dropout = dropout
        self.max_context = max_context
        self.residual_scale = float(residual_scale)
        self.logit_temperature = float(logit_temperature)

        self.token_projection = nn.Sequential(
            nn.Linear(SAFE_TOKEN_SIZE, d_model),
            nn.LayerNorm(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.residual_transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.residual_actor = nn.Linear(d_model, ACTION_COUNT)
        self.critic = nn.Linear(d_model, 1)
        nn.init.zeros_(self.residual_actor.weight)
        nn.init.zeros_(self.residual_actor.bias)
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "ResidualCausalTemporalPolicy":
        """Train residual modules while permanently keeping PPO in eval mode."""
        super().train(mode)
        self._freeze_teacher()
        return self

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if batch_size != 1:
            raise ValueError("online residual state supports batch_size=1")
        parameter = next(self.residual_actor.parameters())
        return torch.empty(
            (0, SAFE_TOKEN_SIZE),
            device=device or parameter.device,
            dtype=parameter.dtype,
        )

    def _residual_sequence(
        self,
        tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_safe_tokens(tokens, valid_tokens)
        if tokens.shape[1] > self.max_context:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds max_context "
                f"{self.max_context}"
            )
        hidden = self.token_projection(tokens)
        hidden = hidden + _sinusoidal_positions(
            tokens.shape[1],
            hidden.shape[-1],
            device=hidden.device,
            dtype=hidden.dtype,
        )
        causal_mask = torch.triu(
            torch.ones(
                tokens.shape[1],
                tokens.shape[1],
                device=tokens.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        hidden = self.residual_transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=~valid_tokens,
        )
        return (
            self.residual_actor(hidden),
            self.critic(hidden).squeeze(-1),
        )

    def forward_packed(
        self,
        batch: SafeSequenceBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate packed episodes through exact PPO plus causal residual."""
        residuals, all_values = self._residual_sequence(
            batch.tokens, batch.valid_tokens
        )
        if batch.decision_count == 0:
            return (
                residuals.new_empty((0, ACTION_COUNT)),
                all_values.new_empty((0,)),
            )
        episode_rows = batch.decision_rows[:, 0]
        time_rows = batch.decision_rows[:, 1]
        safe_observations = batch.tokens[
            episode_rows, time_rows, :OBSERVATION_SIZE
        ]
        base_logits = self._teacher_logits_from_safe_batch(
            safe_observations, batch.legal_actions
        )
        residual_logits = residuals[episode_rows, time_rows]
        combined = (
            base_logits + self.residual_scale * residual_logits
        ) / self.logit_temperature
        masked = combined.masked_fill(
            ~batch.legal_actions, torch.finfo(combined.dtype).min
        )
        return masked, all_values[episode_rows, time_rows]

    def step(
        self,
        decision: DecisionInput,
        state: torch.Tensor,
    ) -> PolicyStep:
        """Advance public history and return champion plus causal residual."""
        if not isinstance(decision, DecisionInput):
            raise TypeError("residual policy accepts only DecisionInput")
        if not isinstance(state, torch.Tensor) or (
            state.ndim != 2 or state.shape[1] != SAFE_TOKEN_SIZE
        ):
            raise TypeError("state must be a [history, 100] safe-token tensor")
        new_rows = torch.tensor(
            decision_safe_tokens(decision),
            device=state.device,
            dtype=state.dtype,
        )
        history = torch.cat((state, new_rows), dim=0)[-self.max_context :]
        valid = torch.ones(
            (1, history.shape[0]),
            device=history.device,
            dtype=torch.bool,
        )
        residuals, values = self._residual_sequence(
            history.unsqueeze(0), valid
        )
        legacy = safe_to_legacy_observation(decision).to(
            device=history.device,
            dtype=next(self.teacher.parameters()).dtype,
        )
        with torch.no_grad():
            base_logits = self.teacher(legacy.unsqueeze(0)).squeeze(0)
        combined = (
            base_logits + self.residual_scale * residuals[0, -1]
        ) / self.logit_temperature
        legal = torch.tensor(
            decision.legal_actions,
            device=combined.device,
            dtype=torch.bool,
        )
        return PolicyStep(
            logits=combined.masked_fill(
                ~legal, torch.finfo(combined.dtype).min
            ),
            value=values[0, -1],
            state=history,
        )

    def _teacher_logits_from_safe_batch(
        self,
        safe_observations: torch.Tensor,
        legal_actions: torch.Tensor,
    ) -> torch.Tensor:
        legacy = _safe_batch_to_legacy(safe_observations, legal_actions)
        teacher_parameter = next(self.teacher.parameters())
        legacy = legacy.to(
            device=teacher_parameter.device,
            dtype=teacher_parameter.dtype,
        )
        with torch.no_grad():
            return self.teacher(legacy)

    def checkpoint_payload(self) -> dict[str, Any]:
        residual_state = {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith("teacher.")
        }
        return {
            "format": CHECKPOINT_FORMAT,
            "champion": {
                "path": self.champion_path,
                "sha256": self.champion_sha256,
            },
            "model": {
                "d_model": self.d_model,
                "num_layers": self.num_layers,
                "num_heads": self.num_heads,
                "feedforward_size": self.feedforward_size,
                "dropout": self.dropout,
                "max_context": self.max_context,
                "residual_scale": self.residual_scale,
                "logit_temperature": self.logit_temperature,
                "critic_privileged": False,
            },
            "residual_state_dict": residual_state,
        }

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        champion_path: Path | None = None,
        map_location: torch.device | str = "cpu",
    ) -> "ResidualCausalTemporalPolicy":
        payload = torch.load(
            path, map_location=map_location, weights_only=True
        )
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("unsupported residual temporal checkpoint")
        champion = payload["champion"]
        if champion["sha256"] != KNOWN_TEACHER_SHA256:
            raise RuntimeError("checkpoint does not reference verified champion")
        model_config = dict(payload["model"])
        if model_config.pop("critic_privileged", None) is not False:
            raise RuntimeError("checkpoint critic privacy contract changed")
        model = cls(
            champion_path=(
                Path(champion_path)
                if champion_path is not None
                else Path(champion["path"])
            ),
            **model_config,
        )
        incompatible = model.load_state_dict(
            payload["residual_state_dict"], strict=False
        )
        if (
            any(
                not name.startswith("teacher.")
                for name in incompatible.missing_keys
            )
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "residual checkpoint parameter contract changed: "
                f"{incompatible}"
            )
        model.to(map_location)
        model._freeze_teacher()
        return model


class ResidualRecurrentTemporalPolicy(SafeTemporalPolicy):
    """Frozen PPO plus a fast fixed-state GRU/LSTM residual.

    ``logit_temperature`` applies to the residual only.  The base PPO logits
    are never rescaled, so a zero residual actor is bit-exact for every valid
    residual scale and temperature.
    """

    architecture = "frozen_ppo_plus_recurrent_residual"
    critic_privileged = False

    def __init__(
        self,
        *,
        champion_path: Path = DEFAULT_TEACHER,
        hidden_size: int = 128,
        num_layers: int = 2,
        cell: str = "gru",
        dropout: float = 0.0,
        residual_scale: float = 1.0,
        logit_temperature: float = 1.0,
        policy_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        normalized_cell = cell.lower()
        if normalized_cell not in {"gru", "lstm"}:
            raise ValueError("cell must be 'gru' or 'lstm'")
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        if residual_scale < 0:
            raise ValueError("residual_scale cannot be negative")
        if logit_temperature <= 0 or policy_temperature <= 0:
            raise ValueError(
                "logit_temperature and policy_temperature must be positive"
            )
        champion_path = Path(champion_path)
        digest = file_sha256(champion_path)
        if digest != KNOWN_TEACHER_SHA256:
            raise RuntimeError(
                "residual policy requires the verified PPO champion: "
                f"expected {KNOWN_TEACHER_SHA256}, got {digest}"
            )
        teacher, _ = load_frozen_teacher(champion_path)
        self.teacher = teacher
        self.champion_path = str(champion_path)
        self.champion_sha256 = digest
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = normalized_cell
        self.dropout = dropout
        self.residual_scale = float(residual_scale)
        self.logit_temperature = float(logit_temperature)
        self.policy_temperature = float(policy_temperature)

        self.token_projection = nn.Sequential(
            nn.Linear(SAFE_TOKEN_SIZE, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        recurrent_type = nn.GRU if normalized_cell == "gru" else nn.LSTM
        self.residual_recurrent = recurrent_type(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.residual_actor = nn.Linear(hidden_size, ACTION_COUNT)
        self.critic = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.residual_actor.weight)
        nn.init.zeros_(self.residual_actor.bias)
        self._freeze_teacher()

    def _freeze_teacher(self) -> None:
        self.teacher.eval()
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "ResidualRecurrentTemporalPolicy":
        super().train(mode)
        self._freeze_teacher()
        return self

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        parameter = next(self.residual_actor.parameters())
        zeros = torch.zeros(
            self.num_layers,
            batch_size,
            self.hidden_size,
            device=device or parameter.device,
            dtype=parameter.dtype,
        )
        if self.cell == "lstm":
            return zeros, zeros.clone()
        return zeros

    def _combine(
        self,
        base_logits: torch.Tensor,
        residual_logits: torch.Tensor,
    ) -> torch.Tensor:
        return (
            base_logits
            + self.residual_scale
            * residual_logits
            / self.logit_temperature
        ) / self.policy_temperature

    def forward_packed(
        self,
        batch: SafeSequenceBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_safe_tokens(batch.tokens, batch.valid_tokens)
        hidden, _ = self.residual_recurrent(
            self.token_projection(batch.tokens)
        )
        if batch.decision_count == 0:
            return (
                hidden.new_empty((0, ACTION_COUNT)),
                hidden.new_empty((0,)),
            )
        episode_rows = batch.decision_rows[:, 0]
        time_rows = batch.decision_rows[:, 1]
        selected_hidden = hidden[episode_rows, time_rows]
        safe_observations = batch.tokens[
            episode_rows, time_rows, :OBSERVATION_SIZE
        ]
        base_logits = self._teacher_logits_from_safe_batch(
            safe_observations, batch.legal_actions
        )
        combined = self._combine(
            base_logits, self.residual_actor(selected_hidden)
        )
        return (
            combined.masked_fill(
                ~batch.legal_actions, torch.finfo(combined.dtype).min
            ),
            self.critic(selected_hidden).squeeze(-1),
        )

    def step(
        self,
        decision: DecisionInput,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> PolicyStep:
        if not isinstance(decision, DecisionInput):
            raise TypeError("residual policy accepts only DecisionInput")
        self._validate_state(state)
        parameter = next(self.residual_actor.parameters())
        tokens = torch.tensor(
            decision_safe_tokens(decision),
            device=parameter.device,
            dtype=parameter.dtype,
        ).unsqueeze(0)
        hidden, next_state = self.residual_recurrent(
            self.token_projection(tokens), state
        )
        final_hidden = hidden[0, -1]
        legacy = safe_to_legacy_observation(decision).to(
            device=parameter.device,
            dtype=next(self.teacher.parameters()).dtype,
        )
        with torch.no_grad():
            base_logits = self.teacher(legacy.unsqueeze(0)).squeeze(0)
        combined = self._combine(
            base_logits, self.residual_actor(final_hidden)
        )
        legal = torch.tensor(
            decision.legal_actions,
            device=combined.device,
            dtype=torch.bool,
        )
        return PolicyStep(
            logits=combined.masked_fill(
                ~legal, torch.finfo(combined.dtype).min
            ),
            value=self.critic(final_hidden).reshape(-1)[0],
            state=next_state,
        )

    def _validate_state(
        self,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        expected_shape = (self.num_layers, 1, self.hidden_size)
        if self.cell == "gru":
            if not isinstance(state, torch.Tensor) or (
                tuple(state.shape) != expected_shape
            ):
                raise TypeError(
                    f"GRU state must have shape {expected_shape}"
                )
            return
        if (
            not isinstance(state, tuple)
            or len(state) != 2
            or any(
                not isinstance(part, torch.Tensor)
                or tuple(part.shape) != expected_shape
                for part in state
            )
        ):
            raise TypeError(
                f"LSTM state must be a pair with shape {expected_shape}"
            )

    def _teacher_logits_from_safe_batch(
        self,
        safe_observations: torch.Tensor,
        legal_actions: torch.Tensor,
    ) -> torch.Tensor:
        legacy = _safe_batch_to_legacy(safe_observations, legal_actions)
        teacher_parameter = next(self.teacher.parameters())
        legacy = legacy.to(
            device=teacher_parameter.device,
            dtype=teacher_parameter.dtype,
        )
        with torch.no_grad():
            return self.teacher(legacy)

    def checkpoint_payload(self) -> dict[str, Any]:
        residual_state = {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if not name.startswith("teacher.")
        }
        return {
            "format": RECURRENT_CHECKPOINT_FORMAT,
            "champion": {
                "path": self.champion_path,
                "sha256": self.champion_sha256,
            },
            "model": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "cell": self.cell,
                "dropout": self.dropout,
                "residual_scale": self.residual_scale,
                "logit_temperature": self.logit_temperature,
                "policy_temperature": self.policy_temperature,
                "critic_privileged": False,
            },
            "residual_state_dict": residual_state,
        }

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)

    @classmethod
    def load_checkpoint(
        cls,
        path: Path,
        *,
        champion_path: Path | None = None,
        map_location: torch.device | str = "cpu",
    ) -> "ResidualRecurrentTemporalPolicy":
        payload = torch.load(
            path, map_location=map_location, weights_only=True
        )
        if payload.get("format") != RECURRENT_CHECKPOINT_FORMAT:
            raise ValueError("unsupported recurrent residual checkpoint")
        champion = payload["champion"]
        if champion["sha256"] != KNOWN_TEACHER_SHA256:
            raise RuntimeError("checkpoint does not reference verified champion")
        model_config = dict(payload["model"])
        if model_config.pop("critic_privileged", None) is not False:
            raise RuntimeError("checkpoint critic privacy contract changed")
        model = cls(
            champion_path=(
                Path(champion_path)
                if champion_path is not None
                else Path(champion["path"])
            ),
            **model_config,
        )
        incompatible = model.load_state_dict(
            payload["residual_state_dict"], strict=False
        )
        if (
            any(
                not name.startswith("teacher.")
                for name in incompatible.missing_keys
            )
            or incompatible.unexpected_keys
        ):
            raise RuntimeError(
                "recurrent residual checkpoint contract changed: "
                f"{incompatible}"
            )
        model.to(map_location)
        model._freeze_teacher()
        return model


def _safe_batch_to_legacy(
    safe_observations: torch.Tensor,
    legal_actions: torch.Tensor,
) -> torch.Tensor:
    """Vectorized safe-to-121 conversion equivalent to the online adapter."""
    return safe_batch_to_legacy_observations(
        safe_observations,
        legal_actions,
    )


def _validate_safe_tokens(
    tokens: torch.Tensor,
    valid_tokens: torch.Tensor,
) -> None:
    if tokens.ndim != 3 or tokens.shape[-1] != SAFE_TOKEN_SIZE:
        raise ValueError("tokens must have shape [batch, time, 100]")
    if valid_tokens.shape != tokens.shape[:2]:
        raise ValueError("valid_tokens must match tokens [batch, time]")
    if valid_tokens.dtype is not torch.bool:
        raise TypeError("valid_tokens must be boolean")


def _sinusoidal_positions(
    length: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    positions = torch.arange(
        length, device=device, dtype=torch.float32
    ).unsqueeze(1)
    half_width = (width + 1) // 2
    scales = torch.exp(
        torch.arange(
            half_width, device=device, dtype=torch.float32
        )
        * (
            -torch.log(torch.tensor(10_000.0, device=device))
            / max(1, half_width)
        )
    )
    angles = positions * scales.unsqueeze(0)
    encoding = torch.zeros(
        (length, width), device=device, dtype=torch.float32
    )
    encoding[:, 0::2] = torch.sin(
        angles[:, : encoding[:, 0::2].shape[1]]
    )
    encoding[:, 1::2] = torch.cos(
        angles[:, : encoding[:, 1::2].shape[1]]
    )
    return encoding.to(dtype=dtype).unsqueeze(0)
