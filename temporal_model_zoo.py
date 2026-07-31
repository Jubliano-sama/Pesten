"""Information-safe temporal policy architectures for broad pilot studies.

Every model in this module accepts the existing :class:`DecisionInput`
boundary.  Consequently, none of the architectures can receive opponent card
identities or draw-pile contents.  Public draws remain count-only
``PublicEventToken`` values.

The module intentionally has no Transformers dependency.  The decoder adapter
can wrap a locally available Hugging Face decoder, while
``TinyRandomCausalDecoder`` provides a small, randomly initialized test and
benchmark substitute when no pretrained weights are installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Sequence

import torch
from torch import nn

from temporal_observation import (
    ACTION_COUNT,
    DecisionInput,
    EVENT_TOKEN_SIZE,
    OBSERVATION_SIZE,
    PublicEventToken,
)
from temporal_puffer_agent import PolicyStep


SAFE_TOKEN_SIZE = OBSERVATION_SIZE + EVENT_TOKEN_SIZE + 1


def public_event_features(event: PublicEventToken) -> tuple[float, ...]:
    """Encode one already-redacted public event into bounded coordinates."""
    if not isinstance(event, PublicEventToken):
        raise TypeError("event history must contain PublicEventToken values")
    return (
        int(event.kind) / 3.0,
        event.actor_relative / 12.0,
        (event.card_id + 1) / 50.0,
        (event.suit_id + 1) / 5.0,
        event.count / 54.0,
    )


def event_token(event: PublicEventToken) -> tuple[float, ...]:
    return (
        (0.0,) * OBSERVATION_SIZE
        + public_event_features(event)
        + (0.0,)
    )


def decision_token(decision: DecisionInput) -> tuple[float, ...]:
    if not isinstance(decision, DecisionInput):
        raise TypeError("temporal policies accept only DecisionInput")
    return (
        decision.observation.values
        + (0.0,) * EVENT_TOKEN_SIZE
        + (1.0,)
    )


def decision_safe_tokens(decision: DecisionInput) -> tuple[tuple[float, ...], ...]:
    """Return new public-event tokens followed by the private decision token."""
    if not isinstance(decision, DecisionInput):
        raise TypeError("temporal policies accept only DecisionInput")
    return tuple(event_token(event) for event in decision.public_events) + (
        decision_token(decision),
    )


@dataclass(frozen=True, slots=True)
class SafeSequenceBatch:
    """Padded safe-token episodes and decision-level supervision."""

    tokens: torch.Tensor
    valid_tokens: torch.Tensor
    decision_rows: torch.Tensor
    legal_actions: torch.Tensor
    targets: torch.Tensor

    @property
    def decision_count(self) -> int:
        return int(self.decision_rows.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.valid_tokens.sum().item())

    def to(self, device: torch.device | str) -> "SafeSequenceBatch":
        return SafeSequenceBatch(
            tokens=self.tokens.to(device),
            valid_tokens=self.valid_tokens.to(device),
            decision_rows=self.decision_rows.to(device),
            legal_actions=self.legal_actions.to(device),
            targets=self.targets.to(device),
        )


def pack_safe_episodes(
    episodes: Sequence[Sequence[Any]],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> SafeSequenceBatch:
    """Pack episodes whose samples expose ``decision`` and ``action``.

    The function deliberately reads only those two attributes.  A target is
    rejected if the frozen teacher selected an action outside the
    ``DecisionInput`` legal-action mask.
    """
    rows_by_episode: list[list[tuple[float, ...]]] = []
    decision_rows: list[tuple[int, int]] = []
    legal_actions: list[tuple[bool, ...]] = []
    targets: list[int] = []
    for episode_index, episode in enumerate(episodes):
        rows: list[tuple[float, ...]] = []
        for sample in episode:
            decision = getattr(sample, "decision", None)
            if not isinstance(decision, DecisionInput):
                raise TypeError("every sample must expose a DecisionInput")
            action = int(getattr(sample, "action"))
            if not 0 <= action < ACTION_COUNT:
                raise ValueError(f"target action is out of range: {action}")
            if not decision.legal_actions[action]:
                raise ValueError(f"target action {action} is not legal")
            rows.extend(event_token(event) for event in decision.public_events)
            rows.append(decision_token(decision))
            decision_rows.append((episode_index, len(rows) - 1))
            legal_actions.append(decision.legal_actions)
            targets.append(action)
        rows_by_episode.append(rows)

    batch_size = len(rows_by_episode)
    max_length = max((len(rows) for rows in rows_by_episode), default=0)
    tokens = torch.zeros(
        (batch_size, max_length, SAFE_TOKEN_SIZE),
        device=device,
        dtype=dtype,
    )
    valid = torch.zeros(
        (batch_size, max_length), device=device, dtype=torch.bool
    )
    for episode_index, rows in enumerate(rows_by_episode):
        if rows:
            tokens[episode_index, : len(rows)] = torch.tensor(
                rows, device=device, dtype=dtype
            )
            valid[episode_index, : len(rows)] = True

    return SafeSequenceBatch(
        tokens=tokens,
        valid_tokens=valid,
        decision_rows=torch.tensor(
            decision_rows, device=device, dtype=torch.long
        ).reshape(-1, 2),
        legal_actions=torch.tensor(
            legal_actions, device=device, dtype=torch.bool
        ).reshape(-1, ACTION_COUNT),
        targets=torch.tensor(targets, device=device, dtype=torch.long),
    )


class SafeTemporalPolicy(nn.Module):
    """Common batched path and legal-action masking for model-zoo policies."""

    architecture: str = "abstract"
    critic_privileged = False

    def forward_safe_sequence(
        self,
        tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def forward_packed(
        self, batch: SafeSequenceBatch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_logits, all_values = self.forward_safe_sequence(
            batch.tokens, batch.valid_tokens
        )
        if batch.decision_count == 0:
            return (
                all_logits.new_empty((0, ACTION_COUNT)),
                all_values.new_empty((0,)),
            )
        episode_rows = batch.decision_rows[:, 0]
        time_rows = batch.decision_rows[:, 1]
        logits = all_logits[episode_rows, time_rows]
        values = all_values[episode_rows, time_rows]
        masked = logits.masked_fill(
            ~batch.legal_actions, torch.finfo(logits.dtype).min
        )
        return masked, values


class LargeRecurrentTemporalPolicy(SafeTemporalPolicy):
    """Multi-layer GRU or LSTM over public events and private decisions."""

    def __init__(
        self,
        *,
        hidden_size: int = 192,
        num_layers: int = 2,
        cell: str = "gru",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        normalized = cell.lower()
        if normalized not in {"gru", "lstm"}:
            raise ValueError("cell must be 'gru' or 'lstm'")
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("hidden_size and num_layers must be positive")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cell = normalized
        self.architecture = f"{normalized}_{num_layers}x{hidden_size}"
        self.token_encoder = nn.Sequential(
            nn.Linear(SAFE_TOKEN_SIZE, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        recurrent_type = nn.GRU if normalized == "gru" else nn.LSTM
        self.recurrent = recurrent_type(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.actor = nn.Linear(hidden_size, ACTION_COUNT)
        self.critic = nn.Linear(hidden_size, 1)

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        parameter = next(self.parameters())
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

    def forward_safe_sequence(
        self,
        tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_safe_tensor_pair(tokens, valid_tokens)
        encoded = self.token_encoder(tokens)
        hidden, _ = self.recurrent(encoded)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def step(
        self,
        decision: DecisionInput,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> PolicyStep:
        rows = decision_safe_tokens(decision)
        parameter = next(self.parameters())
        tokens = torch.tensor(
            rows, device=parameter.device, dtype=parameter.dtype
        ).unsqueeze(0)
        encoded = self.token_encoder(tokens)
        hidden, next_state = self.recurrent(encoded, state)
        final = hidden[:, -1]
        logits = self.actor(final).squeeze(0)
        value = self.critic(final).reshape(-1)[0]
        return PolicyStep(
            logits=_mask_one(logits, decision),
            value=value,
            state=next_state,
        )


class CausalTransformerTemporalPolicy(SafeTemporalPolicy):
    """Compact causal Transformer over the information-safe token history."""

    architecture = "causal_transformer"

    def __init__(
        self,
        *,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        feedforward_size: int = 384,
        dropout: float = 0.0,
        max_context: int = 384,
    ) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.hidden_size = d_model
        self.num_layers = num_layers
        self.max_context = max_context
        self.token_encoder = nn.Sequential(
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
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, norm=nn.LayerNorm(d_model)
        )
        self.actor = nn.Linear(d_model, ACTION_COUNT)
        self.critic = nn.Linear(d_model, 1)

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if batch_size != 1:
            raise ValueError("online causal state currently supports batch_size=1")
        parameter = next(self.parameters())
        return torch.empty(
            (0, SAFE_TOKEN_SIZE),
            device=device or parameter.device,
            dtype=parameter.dtype,
        )

    def forward_safe_sequence(
        self,
        tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_safe_tensor_pair(tokens, valid_tokens)
        if tokens.shape[1] > self.max_context:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds max_context "
                f"{self.max_context}"
            )
        encoded = self.token_encoder(tokens)
        encoded = encoded + _sinusoidal_positions(
            tokens.shape[1],
            encoded.shape[-1],
            device=encoded.device,
            dtype=encoded.dtype,
        )
        causal = torch.triu(
            torch.ones(
                tokens.shape[1],
                tokens.shape[1],
                device=tokens.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        hidden = self.transformer(
            encoded,
            mask=causal,
            src_key_padding_mask=~valid_tokens,
        )
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def step(
        self,
        decision: DecisionInput,
        state: torch.Tensor,
    ) -> PolicyStep:
        if not isinstance(state, torch.Tensor) or (
            state.ndim != 2 or state.shape[1] != SAFE_TOKEN_SIZE
        ):
            raise TypeError(
                "transformer state must be a [history, 100] tensor"
            )
        new_rows = torch.tensor(
            decision_safe_tokens(decision),
            device=state.device,
            dtype=state.dtype,
        )
        history = torch.cat((state, new_rows), dim=0)[
            -self.max_context :
        ]
        valid = torch.ones(
            (1, history.shape[0]), device=history.device, dtype=torch.bool
        )
        logits, values = self.forward_safe_sequence(
            history.unsqueeze(0), valid
        )
        return PolicyStep(
            logits=_mask_one(logits[0, -1], decision),
            value=values[0, -1],
            state=history,
        )


class TinyRandomCausalDecoder(nn.Module):
    """Small decoder-shaped backbone used when pretrained weights are absent."""

    def __init__(
        self,
        *,
        hidden_size: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        feedforward_size: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(hidden_size),
        )

    def forward(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Any:
        del output_hidden_states
        length = inputs_embeds.shape[1]
        positioned = inputs_embeds + _sinusoidal_positions(
            length,
            inputs_embeds.shape[-1],
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        causal = torch.triu(
            torch.ones(
                length,
                length,
                device=inputs_embeds.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        padding = None if attention_mask is None else ~attention_mask.bool()
        hidden = self.decoder(
            positioned,
            mask=causal,
            src_key_padding_mask=padding,
        )
        output = SimpleNamespace(
            last_hidden_state=hidden,
            hidden_states=(hidden,),
        )
        return output if return_dict else (hidden,)


class DecoderLMTemporalPolicy(SafeTemporalPolicy):
    """Projection/head adapter for a causal pretrained decoder backbone.

    A Hugging Face ``AutoModel``-style backbone must accept ``inputs_embeds``
    and ``attention_mask``.  Loading defaults to ``local_files_only=True``;
    network access requires the caller to explicitly set ``allow_download``.
    Remote model code is never trusted.
    """

    architecture = "decoder_lm_adapter"

    def __init__(
        self,
        backbone: nn.Module,
        *,
        hidden_size: int | None = None,
        max_context: int = 384,
        freeze_backbone: bool = False,
        source: str = "random-small-config",
    ) -> None:
        super().__init__()
        inferred = hidden_size or _backbone_hidden_size(backbone)
        self.backbone = backbone
        self.hidden_size = inferred
        self.max_context = max_context
        self.backbone_source = source
        self.safe_projection = nn.Sequential(
            nn.Linear(SAFE_TOKEN_SIZE, inferred),
            nn.LayerNorm(inferred),
        )
        self.actor = nn.Linear(inferred, ACTION_COUNT)
        self.critic = nn.Linear(inferred, 1)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        allow_download: bool = False,
        freeze_backbone: bool = False,
        max_context: int = 384,
        dtype: torch.dtype | str | None = "auto",
    ) -> "DecoderLMTemporalPolicy":
        """Load an installed/cache-resident Transformers decoder.

        ``transformers`` is an optional dependency.  To use this path, install
        it separately and either pre-populate its cache or explicitly opt into
        the named download with ``allow_download=True``.
        """
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise RuntimeError(
                "pretrained decoder support requires the optional "
                "'transformers' package; no model was downloaded"
            ) from error
        try:
            load_arguments: dict[str, Any] = {
                "local_files_only": not allow_download,
                "trust_remote_code": False,
            }
            if dtype is not None:
                load_arguments["dtype"] = dtype
            backbone = AutoModel.from_pretrained(
                model_name_or_path, **load_arguments
            )
        except Exception as error:
            mode = (
                "explicit download was allowed"
                if allow_download
                else "local/cache-only loading was enforced"
            )
            raise RuntimeError(
                f"could not load decoder {model_name_or_path!r}; {mode}. "
                "Install transformers and pre-download reviewed weights, or "
                "pass allow_download=True for this exact model."
            ) from error
        return cls(
            backbone,
            max_context=max_context,
            freeze_backbone=freeze_backbone,
            source=f"pretrained:{model_name_or_path}",
        )

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if batch_size != 1:
            raise ValueError("online decoder state currently supports batch_size=1")
        parameter = next(self.parameters())
        return torch.empty(
            (0, SAFE_TOKEN_SIZE),
            device=device or parameter.device,
            dtype=parameter.dtype,
        )

    def _hidden(
        self, tokens: torch.Tensor, valid_tokens: torch.Tensor
    ) -> torch.Tensor:
        embedded = self.safe_projection(tokens)
        backbone_parameter = next(self.backbone.parameters(), None)
        if backbone_parameter is not None:
            embedded = embedded.to(dtype=backbone_parameter.dtype)
        output = self.backbone(
            inputs_embeds=embedded,
            attention_mask=valid_tokens.to(dtype=torch.long),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            states = getattr(output, "hidden_states", None)
            if not states:
                raise TypeError(
                    "decoder backbone must return last_hidden_state or "
                    "hidden_states"
                )
            hidden = states[-1]
        return hidden.to(dtype=self.actor.weight.dtype)

    def forward_safe_sequence(
        self,
        tokens: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_safe_tensor_pair(tokens, valid_tokens)
        if tokens.shape[1] > self.max_context:
            raise ValueError(
                f"sequence length {tokens.shape[1]} exceeds max_context "
                f"{self.max_context}"
            )
        hidden = self._hidden(tokens, valid_tokens)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def step(
        self,
        decision: DecisionInput,
        state: torch.Tensor,
    ) -> PolicyStep:
        if not isinstance(state, torch.Tensor) or (
            state.ndim != 2 or state.shape[1] != SAFE_TOKEN_SIZE
        ):
            raise TypeError("decoder state must be a [history, 100] tensor")
        new_rows = torch.tensor(
            decision_safe_tokens(decision),
            device=state.device,
            dtype=state.dtype,
        )
        history = torch.cat((state, new_rows), dim=0)[
            -self.max_context :
        ]
        valid = torch.ones(
            (1, history.shape[0]), device=history.device, dtype=torch.bool
        )
        logits, values = self.forward_safe_sequence(
            history.unsqueeze(0), valid
        )
        return PolicyStep(
            logits=_mask_one(logits[0, -1], decision),
            value=values[0, -1],
            state=history,
        )


def parameter_report(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    bytes_total = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    trainable_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "parameters": total,
        "trainable_parameters": trainable,
        "parameter_bytes": bytes_total,
        "estimated_training_state_bytes": trainable_bytes * 4,
    }


def _validate_safe_tensor_pair(
    tokens: torch.Tensor, valid_tokens: torch.Tensor
) -> None:
    if tokens.ndim != 3 or tokens.shape[-1] != SAFE_TOKEN_SIZE:
        raise ValueError(
            "tokens must have shape [batch, time, "
            f"{SAFE_TOKEN_SIZE}]"
        )
    if valid_tokens.shape != tokens.shape[:2]:
        raise ValueError("valid_tokens must match tokens [batch, time]")
    if valid_tokens.dtype is not torch.bool:
        raise TypeError("valid_tokens must be a boolean tensor")


def _mask_one(logits: torch.Tensor, decision: DecisionInput) -> torch.Tensor:
    legal = torch.tensor(
        decision.legal_actions, device=logits.device, dtype=torch.bool
    )
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)


def _backbone_hidden_size(backbone: nn.Module) -> int:
    config = getattr(backbone, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(config, name, None)
        if isinstance(value, int) and value > 0:
            return value
    raise ValueError(
        "hidden_size was not supplied and could not be inferred from "
        "backbone.config"
    )


def _sinusoidal_positions(
    length: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length == 0:
        return torch.empty((1, 0, width), device=device, dtype=dtype)
    positions = torch.arange(
        length, device=device, dtype=torch.float32
    ).unsqueeze(1)
    even_width = (width + 1) // 2
    scales = torch.exp(
        torch.arange(even_width, device=device, dtype=torch.float32)
        * (-torch.log(torch.tensor(10_000.0, device=device)) / max(1, even_width))
    )
    angles = positions * scales.unsqueeze(0)
    encoding = torch.zeros(
        (length, width), device=device, dtype=torch.float32
    )
    encoding[:, 0::2] = torch.sin(angles[:, : encoding[:, 0::2].shape[1]])
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.to(dtype=dtype).unsqueeze(0)
