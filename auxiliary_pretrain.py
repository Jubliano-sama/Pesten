"""Self-supervised pretraining for privacy-safe recurrent temporal encoders.

Targets are derived only from what every human player can observe at the next
decision: the first public event category and the aggregate opponent hand-size
delta.  Inputs stop at the current typed ``DecisionInput`` token, so the GRU or
LSTM cannot inspect the target transition or later history.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import random
from typing import Any, Sequence

import torch
from torch import nn

from temporal_model_zoo import (
    SAFE_TOKEN_SIZE,
    SafeSequenceBatch,
    pack_safe_episodes,
)
from temporal_observation import DecisionInput, EventKind


SEALED_SEED_START = 92_000_000
NEXT_EVENT_NAMES = (
    "episode_start",
    "play",
    "draw_count",
    "choose_suit",
    "no_public_event",
)
NO_PUBLIC_EVENT = 4
AUXILIARY_CHECKPOINT_FORMAT = "pesten-public-history-auxiliary-v1"


@dataclass(frozen=True, slots=True)
class AuxiliaryBatch:
    safe: SafeSequenceBatch
    prediction_rows: torch.Tensor
    next_event_targets: torch.Tensor
    opponent_hand_delta_targets: torch.Tensor

    @property
    def sample_count(self) -> int:
        return int(self.prediction_rows.shape[0])


class PublicHistoryAuxiliaryModel(nn.Module):
    """Encoder-compatible GRU/LSTM with two public-transition heads."""

    def __init__(
        self,
        *,
        hidden_size: int = 64,
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
        self.dropout = dropout
        self.token_projection = nn.Sequential(
            nn.Linear(SAFE_TOKEN_SIZE, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
        )
        recurrent_type = nn.GRU if normalized == "gru" else nn.LSTM
        self.residual_recurrent = recurrent_type(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.next_event_head = nn.Linear(
            hidden_size, len(NEXT_EVENT_NAMES)
        )
        self.opponent_delta_head = nn.Linear(hidden_size, 1)

    def forward_batch(
        self,
        batch: AuxiliaryBatch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, _ = self.residual_recurrent(
            self.token_projection(batch.safe.tokens)
        )
        episode_rows = batch.prediction_rows[:, 0]
        time_rows = batch.prediction_rows[:, 1]
        selected = hidden[episode_rows, time_rows]
        return (
            self.next_event_head(selected),
            self.opponent_delta_head(selected).squeeze(-1),
        )

    def checkpoint_payload(self) -> dict[str, Any]:
        return {
            "format": AUXILIARY_CHECKPOINT_FORMAT,
            "model": {
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "cell": self.cell,
                "dropout": self.dropout,
            },
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in self.state_dict().items()
            },
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
        map_location: torch.device | str = "cpu",
    ) -> "PublicHistoryAuxiliaryModel":
        payload = torch.load(
            path, map_location=map_location, weights_only=True
        )
        if payload.get("format") != AUXILIARY_CHECKPOINT_FORMAT:
            raise ValueError("unsupported auxiliary checkpoint")
        model = cls(**payload["model"])
        model.load_state_dict(payload["state_dict"], strict=True)
        return model.to(map_location)


def pack_auxiliary_episodes(
    episodes: Sequence[Sequence[Any]],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> AuxiliaryBatch:
    """Align current safe histories with next visible transition targets."""
    safe = pack_safe_episodes(episodes, device=device, dtype=dtype)
    selected_flat_rows = []
    event_targets = []
    delta_targets = []
    flat_offset = 0
    for episode in episodes:
        for index in range(max(0, len(episode) - 1)):
            current = _sample_decision(episode[index])
            following = _sample_decision(episode[index + 1])
            selected_flat_rows.append(flat_offset + index)
            event_targets.append(_next_event_category(following))
            delta_targets.append(
                _opponent_hand_total(following)
                - _opponent_hand_total(current)
            )
        flat_offset += len(episode)
    if not selected_flat_rows:
        raise ValueError(
            "auxiliary pretraining requires a consecutive decision pair"
        )
    indices = torch.tensor(
        selected_flat_rows, device=device, dtype=torch.long
    )
    return AuxiliaryBatch(
        safe=safe,
        prediction_rows=safe.decision_rows[indices],
        next_event_targets=torch.tensor(
            event_targets, device=device, dtype=torch.long
        ),
        opponent_hand_delta_targets=torch.tensor(
            delta_targets, device=device, dtype=dtype
        ),
    )


def auxiliary_metrics(
    model: PublicHistoryAuxiliaryModel,
    episodes: Sequence[Sequence[Any]],
    *,
    device: torch.device | str,
    batch_episodes: int = 16,
) -> dict[str, Any]:
    if batch_episodes < 1:
        raise ValueError("batch_episodes must be positive")
    usable = [episode for episode in episodes if len(episode) >= 2]
    if not usable:
        raise ValueError("no auxiliary decision pairs")
    model.eval()
    event_loss_sum = delta_absolute_sum = delta_square_sum = 0.0
    target_delta_absolute_sum = target_delta_square_sum = 0.0
    event_counts = [0 for _ in NEXT_EVENT_NAMES]
    correct = sign_correct = samples = 0
    zero_delta_targets = 0
    with torch.inference_mode():
        for start in range(0, len(usable), batch_episodes):
            batch = pack_auxiliary_episodes(
                usable[start : start + batch_episodes],
                device=device,
            )
            event_logits, predicted_delta = model.forward_batch(batch)
            event_losses = torch.nn.functional.cross_entropy(
                event_logits,
                batch.next_event_targets,
                reduction="none",
            )
            delta_errors = (
                predicted_delta - batch.opponent_hand_delta_targets
            )
            event_loss_sum += float(event_losses.sum())
            delta_absolute_sum += float(delta_errors.abs().sum())
            delta_square_sum += float(delta_errors.square().sum())
            target_delta_absolute_sum += float(
                batch.opponent_hand_delta_targets.abs().sum()
            )
            target_delta_square_sum += float(
                batch.opponent_hand_delta_targets.square().sum()
            )
            target_counts = torch.bincount(
                batch.next_event_targets,
                minlength=len(NEXT_EVENT_NAMES),
            ).cpu()
            for index, count in enumerate(target_counts.tolist()):
                event_counts[index] += int(count)
            correct += int(
                (
                    event_logits.argmax(dim=-1)
                    == batch.next_event_targets
                ).sum()
            )
            sign_correct += int(
                (
                    torch.sign(predicted_delta)
                    == torch.sign(batch.opponent_hand_delta_targets)
                ).sum()
            )
            zero_delta_targets += int(
                (batch.opponent_hand_delta_targets == 0).sum()
            )
            samples += batch.sample_count
    return {
        "samples": samples,
        "next_event_target_counts": {
            name: event_counts[index]
            for index, name in enumerate(NEXT_EVENT_NAMES)
        },
        "next_event_majority_baseline_accuracy": (
            max(event_counts) / samples
        ),
        "next_event_cross_entropy": event_loss_sum / samples,
        "next_event_accuracy": correct / samples,
        "opponent_delta_zero_baseline_mae": (
            target_delta_absolute_sum / samples
        ),
        "opponent_delta_zero_baseline_rmse": (
            target_delta_square_sum / samples
        ) ** 0.5,
        "opponent_delta_zero_baseline_sign_accuracy": (
            zero_delta_targets / samples
        ),
        "opponent_delta_mae": delta_absolute_sum / samples,
        "opponent_delta_rmse": (
            delta_square_sum / samples
        ) ** 0.5,
        "opponent_delta_sign_accuracy": sign_correct / samples,
    }


def pretrain_auxiliary_encoder(
    model: PublicHistoryAuxiliaryModel,
    training_episodes: Sequence[Sequence[Any]],
    validation_episodes: Sequence[Sequence[Any]],
    *,
    device: torch.device | str,
    epochs: int = 4,
    batch_episodes: int = 8,
    learning_rate: float = 1e-3,
    delta_loss_weight: float = 0.25,
    seed: int = 0,
) -> dict:
    validate_development_seed(seed, span=max(1, epochs))
    if epochs < 1 or batch_episodes < 1:
        raise ValueError("epochs and batch_episodes must be positive")
    if learning_rate <= 0 or delta_loss_weight < 0:
        raise ValueError("invalid auxiliary optimizer configuration")
    usable = [episode for episode in training_episodes if len(episode) >= 2]
    if not usable:
        raise ValueError("no auxiliary training decision pairs")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    before = auxiliary_metrics(
        model,
        validation_episodes,
        device=device,
        batch_episodes=batch_episodes,
    )
    rng = random.Random(seed)
    epochs_log = []
    for epoch in range(epochs):
        order = list(range(len(usable)))
        rng.shuffle(order)
        weighted_loss = 0.0
        samples = 0
        model.train()
        for start in range(0, len(order), batch_episodes):
            selected = [
                usable[index]
                for index in order[start : start + batch_episodes]
            ]
            batch = pack_auxiliary_episodes(
                selected, device=device
            )
            event_logits, predicted_delta = model.forward_batch(batch)
            event_loss = torch.nn.functional.cross_entropy(
                event_logits, batch.next_event_targets
            )
            delta_loss = torch.nn.functional.smooth_l1_loss(
                predicted_delta,
                batch.opponent_hand_delta_targets,
            )
            loss = event_loss + delta_loss_weight * delta_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            weighted_loss += float(loss.detach()) * batch.sample_count
            samples += batch.sample_count
        heldout = auxiliary_metrics(
            model,
            validation_episodes,
            device=device,
            batch_episodes=batch_episodes,
        )
        epochs_log.append(
            {
                "epoch": epoch + 1,
                "training_objective": weighted_loss / samples,
                "heldout": heldout,
            }
        )
    return {
        "before": before,
        "after": epochs_log[-1]["heldout"],
        "epochs": epochs_log,
    }


def initialize_residual_encoder(
    residual_policy,
    auxiliary_model: PublicHistoryAuxiliaryModel,
) -> dict[str, str | int]:
    """Copy only projection/recurrent weights into a compatible policy."""
    required = (
        "token_projection",
        "residual_recurrent",
        "hidden_size",
        "num_layers",
        "cell",
    )
    if any(not hasattr(residual_policy, name) for name in required):
        raise TypeError("policy is not a compatible recurrent residual")
    for name in ("hidden_size", "num_layers", "cell"):
        if getattr(residual_policy, name) != getattr(auxiliary_model, name):
            raise ValueError(f"encoder configuration mismatch: {name}")
    before = _encoder_digest(residual_policy)
    residual_policy.token_projection.load_state_dict(
        auxiliary_model.token_projection.state_dict(), strict=True
    )
    residual_policy.residual_recurrent.load_state_dict(
        auxiliary_model.residual_recurrent.state_dict(), strict=True
    )
    after = _encoder_digest(residual_policy)
    return {
        "parameters_copied": sum(
            parameter.numel()
            for module in (
                residual_policy.token_projection,
                residual_policy.residual_recurrent,
            )
            for parameter in module.parameters()
        ),
        "encoder_sha256_before": before,
        "encoder_sha256_after": after,
    }


def validate_development_seed(seed: int, *, span: int = 1) -> None:
    if seed < 0 or span < 1 or seed + span - 1 >= SEALED_SEED_START:
        raise ValueError("auxiliary work cannot use sealed final seeds")


def _sample_decision(sample: Any) -> DecisionInput:
    decision = getattr(sample, "decision", None)
    if not isinstance(decision, DecisionInput):
        raise TypeError("auxiliary samples must expose DecisionInput")
    return decision


def _next_event_category(decision: DecisionInput) -> int:
    if not decision.public_events:
        return NO_PUBLIC_EVENT
    return int(decision.public_events[0].kind)


def _opponent_hand_total(decision: DecisionInput) -> float:
    values = decision.observation.values
    player_count = int(round(values[92]))
    if not 2 <= player_count <= 13:
        raise ValueError("safe observation has invalid player count")
    # Slot zero is always the observer; all remaining public sizes are human
    # visible and their sum is stable across direction reversals.
    return float(sum(values[70 : 69 + player_count]))


def _encoder_digest(model) -> str:
    digest = hashlib.sha256()
    for module_name in ("token_projection", "residual_recurrent"):
        module = getattr(model, module_name)
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(f"{module_name}.{name}".encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
