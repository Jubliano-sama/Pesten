"""Production loader for the currently selected privacy-safe Pesten player.

The selection is intentionally centralized here so interactive and web entry
points can adopt it later without duplicating checkpoint or projection choices.
The current default is the verified production PPO champion with full
ordinary-suit logit averaging and no player-label orbit averaging.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Callable
import warnings

import torch

from ai_observation import OBSERVATION_SIZE
from batched_pesten import ActorPolicy
from batched_suit_symmetry import BatchedSuitAveragedActor
from corrected_ppo_agent import make_corrected_ppo_agent_factory
from hybrid_agent import load_frozen_ppo_actor


CHAMPION_SHA256 = (
    "aac589b3c7300895604e6c77755631ab4a31c8bf9810768448ab2f3ff5bbe186"
)
# Explicit CPU throughput option; the sealed production default remains 512.
OPT_IN_CPU_BASE_CHUNK_SIZE = 64
DEFAULT_CHAMPION_PATH = Path(__file__).resolve().with_name(
    "ppo_actor_100run3.pth"
)
INFORMATION_BOUNDARY = {
    "policy_observation": (
        "unchanged 121 features: own hand and public game state"
    ),
    "additional_runtime_state": (
        "just-drawn own card number and public Game.turn_count for the "
        "immediate draw-retry correction"
    ),
    "synthetic_transforms": (
        "ordinary suit relabelings; optional absolute player-label "
        "rotations/reflections"
    ),
    "opponent_card_identities": False,
    "draw_pile_inspection": False,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class BestPlayerConfig:
    model_path: Path = DEFAULT_CHAMPION_PATH
    device: str = "auto"
    alpha: float = 1.0
    aggregation: str = "logits"
    player_orbit_average: bool = False
    base_chunk_size: int = 512
    trace_cpu_actor: bool = True

    def __post_init__(self):
        object.__setattr__(self, "model_path", Path(self.model_path).resolve())
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be between zero and one")
        if self.aggregation not in ("logits", "probabilities"):
            raise ValueError(
                "aggregation must be 'logits' or 'probabilities'"
            )
        if self.base_chunk_size < 1:
            raise ValueError("base_chunk_size must be positive")


@dataclass(frozen=True)
class LoadedBestPlayer:
    """One verified projected actor with canonical and batched adapters."""

    actor: BatchedSuitAveragedActor
    device: torch.device
    config: BestPlayerConfig
    metadata: dict

    def canonical_game_factory(self) -> Callable:
        factory = make_corrected_ppo_agent_factory(
            self.actor,
            self.device,
        )
        factory.best_player_metadata = self.metadata
        return factory

    def batched_actor_policy(self) -> ActorPolicy:
        return ActorPolicy(
            actor=self.actor,
            device=self.device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        )


def verify_champion(path: str | Path) -> str:
    """Fail closed unless ``path`` is the exact selected champion artifact."""
    resolved = Path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"champion checkpoint not found: {resolved}")
    actual = _sha256(resolved)
    if actual != CHAMPION_SHA256:
        raise RuntimeError(
            "champion checkpoint SHA-256 mismatch: "
            f"expected {CHAMPION_SHA256}, got {actual}"
        )
    return actual


def load_best_player(
    config: BestPlayerConfig | None = None,
) -> LoadedBestPlayer:
    """Verify, freeze, and project the currently selected production actor."""
    config = BestPlayerConfig() if config is None else config
    champion_sha = verify_champion(config.model_path)
    base_actor, device = load_frozen_ppo_actor(
        config.model_path,
        config.device,
    )
    base_actor_execution = "eager"
    if config.trace_cpu_actor and device.type == "cpu":
        # Every suit-projected call reaches the base actor with a dynamic
        # batch dimension and at least 24 rows.  Tracing only this frozen,
        # inference-only module removes Python dispatch without changing any
        # tensor operation or checkpoint value.  Do not freeze the graph:
        # constant folding can change floating-point association.
        trace_example = torch.zeros(
            (24, OBSERVATION_SIZE),
            dtype=torch.float32,
            device=device,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            base_actor = torch.jit.trace(
                base_actor,
                trace_example,
                check_trace=False,
            ).eval()
        base_actor_execution = "torchscript_trace"
    actor = BatchedSuitAveragedActor(
        base_actor,
        alpha=config.alpha,
        aggregation=config.aggregation,
        player_orbit_average=config.player_orbit_average,
        base_chunk_size=config.base_chunk_size,
    ).to(device)
    actor.eval()
    actor.requires_grad_(False)
    metadata = {
        "selection": "verified_champion_with_symmetry_projection",
        "checkpoint": {
            "path": str(config.model_path),
            "sha256": champion_sha,
            "frozen": True,
        },
        "projection": {
            "alpha": config.alpha,
            "aggregation": config.aggregation,
            "player_orbit_average": config.player_orbit_average,
            "ordinary_suit_permutations": 24,
            "base_chunk_size": config.base_chunk_size,
            "base_actor_execution": base_actor_execution,
            "draw_retry": "just-drawn card plus PASS",
        },
        "information_boundary": dict(INFORMATION_BOUNDARY),
        "config": {
            **asdict(config),
            "model_path": str(config.model_path),
        },
    }
    actor.best_player_metadata = metadata
    return LoadedBestPlayer(
        actor=actor,
        device=device,
        config=config,
        metadata=metadata,
    )


def load_canonical_game_factory(
    config: BestPlayerConfig | None = None,
) -> Callable:
    """Load the verified player and return a factory accepted by ``Game``."""
    return load_best_player(config).canonical_game_factory()


def load_batched_actor(
    config: BestPlayerConfig | None = None,
) -> tuple[BatchedSuitAveragedActor, torch.device]:
    """Load the ActorPolicy-compatible projected actor and its device."""
    loaded = load_best_player(config)
    return loaded.actor, loaded.device
