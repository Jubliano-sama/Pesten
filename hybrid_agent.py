"""A conservative public-information tactical shield for the production PPO.

The shield deliberately has a very small authority surface.  The frozen PPO
still chooses every action except for two endgame cases that follow directly
from the engine rules:

* on an unpenalized turn, pass instead of trying to discard a lone special
  card (the engine keeps that card and draws two cards); and
* when the PPO would discard an ordinary card and leave a special as its last
  card, play the special first only when it provably retains the turn and the
  ordinary card is then a legal finisher.

No opponent card identity or draw-pile state is read.  The inherited PPO
observation uses the agent's own hand and public ``Game`` fields only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import torch

import card
from ai_observation import PASS_ACTION
from ppo2 import RL_SupervisedAgent, StudentNet, build_action_masks


SPECIAL_RANKS = frozenset({0, 1, 2, 7, 8, 13})


def _resolve_device(device_name: str | torch.device = "auto") -> torch.device:
    if isinstance(device_name, torch.device):
        device = device_name
    else:
        requested = str(device_name)
        if requested == "auto":
            requested = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _extract_state_dict(loaded: object) -> Mapping[str, torch.Tensor]:
    """Accept a raw actor state dict or a narrowly recognized checkpoint."""
    if not isinstance(loaded, Mapping):
        raise TypeError("PPO checkpoint must contain a state dictionary")
    for key in ("actor_state_dict", "state_dict"):
        candidate = loaded.get(key)
        if isinstance(candidate, Mapping):
            loaded = candidate
            break
    if not loaded or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor)
        for key, value in loaded.items()
    ):
        raise TypeError("PPO checkpoint does not contain tensor parameters")
    return loaded


def load_frozen_ppo_actor(
    model_path: str | Path,
    device_name: str | torch.device = "auto",
) -> tuple[StudentNet, torch.device]:
    """Load the production actor once, strictly, and freeze it for inference."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"PPO checkpoint not found: {path}")
    device = _resolve_device(device_name)
    try:
        loaded = torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # Compatibility with older PyTorch releases.
        loaded = torch.load(path, map_location=device)
    actor = StudentNet().to(device)
    actor.load_state_dict(_extract_state_dict(loaded), strict=True)
    actor.eval()
    actor.requires_grad_(False)
    return actor, device


class FrozenGreedyPPOAgent(RL_SupervisedAgent):
    """Deterministic, inference-only wrapper around the production PPO actor."""

    def __init__(self, game_instance, actor, device):
        if actor is None:
            raise ValueError("actor is required")
        super().__init__(
            game_instance,
            actor,
            _resolve_device(device),
            temperature=1.0,
        )
        self.eval_mode = True
        self.collect_data = False
        self.type = "frozen_greedy_ppo"

    def act_rl(self, observation, temperature=None):
        device_observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            logits = self.actor(device_observation.unsqueeze(0)).squeeze(0)
            mask = build_action_masks(device_observation)
            masked_logits = logits.masked_fill(
                ~mask,
                torch.finfo(logits.dtype).min,
            )
            action = int(torch.argmax(masked_logits).item())
        return (None if action == PASS_ACTION else action), 0.0


class HybridPPOAgent(FrozenGreedyPPOAgent):
    """Frozen PPO with two conservative, public-information endgame rules."""

    def __init__(self, game_instance, actor, device):
        super().__init__(game_instance, actor, device)
        self.type = "hybrid_ppo_public_tactical_shield"
        self.last_override_reason: str | None = None

    def _is_legal_play(self, held_card, current_sort, current_true_number):
        if self.game.penalty_amount > 0:
            return held_card.truenumber == current_true_number
        return held_card.compatible(current_sort, current_true_number)

    def _retains_turn(self, candidate) -> bool:
        # A seven grants another turn for every player count.  In a two-player
        # game an eight skips the sole opponent, also returning play to us.
        return candidate.truenumber == 7 or (
            candidate.truenumber == 8 and self.game.num_players == 2
        )

    @staticmethod
    def _ordinary_follows_special(ordinary, special) -> bool:
        return ordinary.compatible(special.sort, special.truenumber)

    def _shield_action(
        self,
        proposed,
        current_sort,
        current_true_number,
    ):
        hand = tuple(self.mydeck.cards)
        self.last_override_reason = None

        if (
            len(hand) == 1
            and self.game.penalty_amount == 0
            and hand[0].truenumber in SPECIAL_RANKS
            and self._is_legal_play(
                hand[0],
                current_sort,
                current_true_number,
            )
        ):
            self.last_override_reason = "pass_lone_special"
            return None

        if (
            len(hand) == 2
            and proposed is not None
            and proposed.truenumber not in SPECIAL_RANKS
        ):
            ordinary = proposed
            safe_specials = [
                held_card
                for held_card in hand
                if held_card is not ordinary
                and held_card.truenumber in SPECIAL_RANKS
                and self._is_legal_play(
                    held_card,
                    current_sort,
                    current_true_number,
                )
                and self._retains_turn(held_card)
                and self._ordinary_follows_special(ordinary, held_card)
            ]
            if safe_specials:
                self.last_override_reason = "preserve_ordinary_finisher"
                return min(safe_specials, key=lambda held_card: held_card.number)

        return proposed

    def playCard(self, current_sort, current_true_number, temperature=None):
        proposed = super().playCard(
            current_sort,
            current_true_number,
            temperature,
        )
        return self._shield_action(
            proposed,
            current_sort,
            current_true_number,
        )


def make_ppo_agent_factory(
    actor,
    device,
    *,
    shield: bool,
) -> Callable:
    """Bind one already-loaded actor to a factory accepted by ``game.Game``."""
    agent_class = HybridPPOAgent if shield else FrozenGreedyPPOAgent

    def factory(game_instance):
        return agent_class(game_instance, actor, device)

    factory.__name__ = (
        "HybridPPOFactory" if shield else "FrozenGreedyPPOFactory"
    )
    return factory
