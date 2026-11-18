"""Portable CPU inference for the already-promoted native teacher4387.

Serving depends on PyTorch and the public observation schema; PufferLib and
the training stack are not imported. Each room/seat owns its recurrent state.
"""

from __future__ import annotations

import hashlib
from itertools import permutations
import json
from pathlib import Path

import torch
from torch import Tensor, nn

from temporal_observation import DecisionInput, PublicEventToken


TEACHER_CHECKPOINT_SHA256 = "f8f0aece2d759eadc60b3cbccca30c321c01da13322f7cae511a73c8da3a9f93"
PORTABLE_FORMAT = "pesten-portable-teacher4387-v1"
DEFAULT_PORTABLE_WEIGHTS = Path(__file__).resolve().parent / "models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387_portable.pt"
MODEL_CONFIG = {
    "architecture": "min_gru", "hidden_size": 192, "num_layers": 2,
    "mode_conditioned_heads": True, "action_head": "mode_conditioned_55",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encode_event_token(event: PublicEventToken) -> tuple[float, ...]:
    """Encode one already-redacted public event in the native 100-value schema."""
    if not isinstance(event, PublicEventToken):
        raise TypeError("event must be a PublicEventToken")
    return (0.0,) * 94 + (
        int(event.kind) / 3.0,
        event.actor_relative / 12.0,
        (event.card_id + 1) / 50.0,
        (event.suit_id + 1) / 5.0,
        event.count / 54.0,
        0.0,
    )


def encode_decision_token(decision: DecisionInput) -> tuple[float, ...]:
    if not isinstance(decision, DecisionInput):
        raise TypeError("decision must be a DecisionInput")
    return decision.observation.values + (0.0,) * 5 + (1.0,)


class ProductionTemporalPolicy(nn.Module):
    """Native 192x2 MinGRU eval, with tensor state [2,batch,192]."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Linear(100, 192)
        self.layers = nn.ModuleList([
            nn.Linear(192, 576, bias=False),
            nn.Linear(192, 576, bias=False),
        ])
        self.card_and_pass = nn.Linear(192, 51)
        self.suit = nn.Linear(192, 4)
        self.value_function = nn.Linear(192, 1)

    def initial_state(self, batch_size: int = 1) -> Tensor:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        return self.encoder.weight.new_zeros((2, batch_size, 192))

    def forward(self, token: Tensor, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if token.ndim != 2 or token.shape[1] != 100:
            raise ValueError("token must have shape [batch,100]")
        if state.shape != (2, token.shape[0], 192):
            raise ValueError("state must have shape [2,batch,192]")
        if token.dtype != torch.float32 or state.dtype != torch.float32:
            raise ValueError("production tokens and state must be float32")
        if token.device.type != "cpu" or state.device.type != "cpu":
            raise ValueError("production inference requires CPU tensors")

        # Keep native forward_eval's arithmetic and rank-3 linear inputs exact.
        hidden = self.encoder(token.view(token.shape[0], -1).float()).unsqueeze(1)
        state_out = []
        for index, layer in enumerate(self.layers):
            candidate, gate, projection = layer(hidden).chunk(3, dim=-1)
            positive = torch.where(candidate >= 0, candidate + 0.5, candidate.sigmoid())
            out = torch.lerp(state[index:index + 1].transpose(0, 1), positive, gate.sigmoid())
            highway_gate = projection.sigmoid()
            hidden = highway_gate * out + (1.0 - highway_gate) * hidden
            state_out.append(out[:, -1:])
        hidden = hidden.squeeze(1)
        cards = self.card_and_pass(hidden)
        logits = torch.cat((cards[..., :50], self.suit(hidden), cards[..., 50:51]), dim=-1)
        values = self.value_function(hidden)
        next_state = torch.stack(state_out, 0).squeeze(2)
        return logits, values, next_state


class ProductionSuitOrbitPolicy(nn.Module):
    """The evaluated 24-suit mean-logit policy, using only portable inference.

    Each permutation keeps its own recurrent memory. Packing that memory as
    [2,batch,24*192] preserves the room/seat adapter's ordinary tensor-state API.
    Legal-action masking belongs to the caller, after logits are averaged.
    """

    orbit_size = 24

    def __init__(self, core: ProductionTemporalPolicy) -> None:
        super().__init__()
        self.frozen_core = core
        self.suit_permutations = tuple(permutations(range(4)))
        observation_indices = []
        action_indices = []
        card_maps = []
        suit_maps = []
        for permutation in self.suit_permutations:
            indices = list(range(100))
            actions = list(range(55))
            for old_suit, new_suit in enumerate(permutation):
                for rank in range(12):
                    indices[new_suit * 12 + rank] = old_suit * 12 + rank
                    actions[old_suit * 12 + rank] = new_suit * 12 + rank
                indices[50 + new_suit] = 50 + old_suit
                indices[82 + new_suit] = 82 + old_suit
                actions[50 + old_suit] = 50 + new_suit
            observation_indices.append(indices)
            action_indices.append(actions)
            card_maps.append(actions[:50])
            suit_maps.append([*permutation, 4])
        self.register_buffer("observation_indices", torch.tensor(observation_indices))
        self.register_buffer("action_indices", torch.tensor(action_indices))
        self.register_buffer("card_maps", torch.tensor(card_maps))
        self.register_buffer("suit_maps", torch.tensor(suit_maps))

    def initial_state(self, batch_size: int = 1) -> Tensor:
        return self.frozen_core.initial_state(batch_size * self.orbit_size).reshape(
            2, batch_size, self.orbit_size * 192,
        )

    def relabel_tokens(self, tokens: Tensor) -> Tensor:
        if tokens.ndim != 2 or tokens.shape[1] != 100:
            raise ValueError("token must have shape [batch,100]")
        if tokens.dtype != torch.float32 or tokens.device.type != "cpu":
            raise ValueError("production tokens must be CPU float32 tensors")
        # These maps and validation match the frozen NativeSuitOrbitCore.
        transformed = tokens[:, self.observation_indices]
        event_rows = tokens[:, 99] == 0
        kinds = torch.round(tokens[:, 94] * 3).long()
        play = event_rows & (kinds == 1)
        choose_suit = event_rows & (kinds == 3)
        if play.any():
            card_ids = torch.round(tokens[play, 96] * 50).long() - 1
            if ((card_ids < 0) | (card_ids >= 50)).any():
                raise ValueError("invalid public PLAY card identity")
            play_suit_ids = torch.round(tokens[play, 97] * 5).long() - 1
            expected_suit_ids = torch.where(card_ids < 48, card_ids // 12, 4)
            if (play_suit_ids != expected_suit_ids).any():
                raise ValueError("public PLAY card/suit mismatch")
            transformed[play, :, 96] = (self.card_maps[:, card_ids].T + 1) / 50
        if choose_suit.any():
            announced = torch.round(tokens[choose_suit, 97] * 5).long() - 1
            if ((announced < 0) | (announced >= 4)).any():
                raise ValueError("CHOOSE_SUIT must name an ordinary suit")
        suit_rows = play | choose_suit
        if suit_rows.any():
            suit_ids = torch.round(tokens[suit_rows, 97] * 5).long() - 1
            if ((suit_ids < 0) | (suit_ids >= 5)).any():
                raise ValueError("invalid public event suit")
            transformed[suit_rows, :, 97] = (self.suit_maps[:, suit_ids].T + 1) / 5
        return transformed.flatten(0, 1)

    def forward(self, token: Tensor, state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        expanded = self.relabel_tokens(token)
        batch_size = token.shape[0]
        if state.shape != (2, batch_size, self.orbit_size * 192):
            raise ValueError("state must have shape [2,batch,24*192]")
        logits, values, next_state = self.frozen_core(
            expanded, state.reshape(2, batch_size * self.orbit_size, 192),
        )
        logits = logits.reshape(batch_size, self.orbit_size, 55)
        canonical = logits.gather(2, self.action_indices.expand(batch_size, -1, -1))
        # Preserve the evaluated wrapper's alpha=1 arithmetic exactly.
        combined = canonical[:, 0] * 0.0 + canonical.mean(1) * 1.0
        values = values.reshape(batch_size, self.orbit_size, -1).mean(1)
        return combined, values, next_state.reshape(2, batch_size, self.orbit_size * 192)


def load_production_policy(
    weights_path: str | Path = DEFAULT_PORTABLE_WEIGHTS,
    manifest_path: str | Path | None = None,
) -> ProductionTemporalPolicy:
    """Validate the promoted-source manifest and load frozen tensor-only weights."""
    path = Path(weights_path)
    manifest_path = Path(manifest_path) if manifest_path is not None else path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != PORTABLE_FORMAT:
        raise ValueError("unexpected portable policy format")
    if manifest.get("source_checkpoint_sha256") != TEACHER_CHECKPOINT_SHA256:
        raise ValueError("portable policy does not name the promoted teacher4387")
    if manifest.get("model") != MODEL_CONFIG:
        raise ValueError("portable policy model configuration differs from teacher4387")
    if file_sha256(path) != manifest.get("weights_sha256"):
        raise ValueError("portable policy weights hash does not match its manifest")
    weights = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or any(
        not isinstance(name, str) or not isinstance(value, Tensor)
        or value.dtype != torch.float32 or not torch.isfinite(value).all()
        for name, value in weights.items()
    ):
        raise ValueError("portable weights must contain only finite float32 tensors")
    policy = ProductionTemporalPolicy()
    try:
        policy.load_state_dict(weights, strict=True)
    except RuntimeError as error:
        raise ValueError("portable weight names or dimensions differ from teacher4387") from error
    policy.requires_grad_(False)
    policy.eval()
    policy.metadata = manifest
    return policy
