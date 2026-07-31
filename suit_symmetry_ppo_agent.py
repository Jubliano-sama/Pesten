"""Suit-equivariant test-time ensemble around the frozen PPO champion.

Pesten is unchanged by any relabeling of the four ordinary suits.  The legacy
network is not architecturally equivariant, so this wrapper evaluates all 24
ordinary-suit permutations in one batch, maps their logits back to the original
labels, and averages them.  Special cards, PASS, player counts, direction, and
all other public features remain fixed.

The wrapper inherits the deployment-safe immediate draw-retry tracking.  It
uses only the production own-hand/public observation plus the just-drawn own
card and public turn number already used by that correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Callable, Sequence

import torch

from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_START,
    HAND_START,
    MODE_INDEX,
    PASS_ACTION,
    SORTS_PLAYED_START,
    SUIT_ACTION_START,
)
from corrected_ppo_agent import (
    DrawRetryCorrectedPPOAgent,
    draw_retry_action_mask,
)
from hybrid_agent import _resolve_device
from ppo2 import build_action_masks


STANDARD_SUIT_COUNT = 4
CARDS_PER_STANDARD_SUIT = 12
ALL_SUIT_PERMUTATIONS = tuple(permutations(range(STANDARD_SUIT_COUNT)))


@dataclass
class SuitSymmetryStats:
    decisions: int = 0
    ordinary_decisions: int = 0
    suit_decisions: int = 0
    changed_from_legacy: int = 0
    normal_draws_tracked: int = 0
    retry_decisions: int = 0
    retry_overrides: int = 0
    retry_legacy_wrong_card_actions: int = 0
    retry_drawn_card_actions: int = 0
    retry_pass_actions: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def permute_suit_observation(
    observation: torch.Tensor,
    suit_permutation: Sequence[int],
) -> torch.Tensor:
    """Relabel ordinary suits in one production observation."""
    if observation.ndim != 1 or len(observation) != 121:
        raise ValueError("expected one 121-feature observation")
    permutation = tuple(int(value) for value in suit_permutation)
    if sorted(permutation) != list(range(STANDARD_SUIT_COUNT)):
        raise ValueError("suit_permutation must permute 0,1,2,3")
    transformed = observation.clone()
    for original_suit, transformed_suit in enumerate(permutation):
        original_cards = slice(
            HAND_START + original_suit * CARDS_PER_STANDARD_SUIT,
            HAND_START + (original_suit + 1) * CARDS_PER_STANDARD_SUIT,
        )
        transformed_cards = slice(
            HAND_START + transformed_suit * CARDS_PER_STANDARD_SUIT,
            HAND_START + (transformed_suit + 1) * CARDS_PER_STANDARD_SUIT,
        )
        transformed[transformed_cards] = observation[original_cards]

        original_compatible = slice(
            COMPATIBLE_START
            + original_suit * CARDS_PER_STANDARD_SUIT,
            COMPATIBLE_START
            + (original_suit + 1) * CARDS_PER_STANDARD_SUIT,
        )
        transformed_compatible = slice(
            COMPATIBLE_START
            + transformed_suit * CARDS_PER_STANDARD_SUIT,
            COMPATIBLE_START
            + (transformed_suit + 1) * CARDS_PER_STANDARD_SUIT,
        )
        transformed[transformed_compatible] = observation[
            original_compatible
        ]
        transformed[SORTS_PLAYED_START + transformed_suit] = observation[
            SORTS_PLAYED_START + original_suit
        ]
    return transformed


def map_permuted_logits_back(
    permuted_logits: torch.Tensor,
    suit_permutation: Sequence[int],
) -> torch.Tensor:
    """Map card and suit-choice logits from relabeled coordinates to original."""
    if permuted_logits.ndim != 1 or len(permuted_logits) != ACTION_COUNT:
        raise ValueError(f"expected one {ACTION_COUNT}-action logit vector")
    permutation = tuple(int(value) for value in suit_permutation)
    if sorted(permutation) != list(range(STANDARD_SUIT_COUNT)):
        raise ValueError("suit_permutation must permute 0,1,2,3")
    original_logits = permuted_logits.clone()
    for original_suit, transformed_suit in enumerate(permutation):
        original_cards = slice(
            original_suit * CARDS_PER_STANDARD_SUIT,
            (original_suit + 1) * CARDS_PER_STANDARD_SUIT,
        )
        transformed_cards = slice(
            transformed_suit * CARDS_PER_STANDARD_SUIT,
            (transformed_suit + 1) * CARDS_PER_STANDARD_SUIT,
        )
        original_logits[original_cards] = permuted_logits[transformed_cards]
        original_logits[SUIT_ACTION_START + original_suit] = permuted_logits[
            SUIT_ACTION_START + transformed_suit
        ]
    return original_logits


def suit_averaged_logits(
    actor: torch.nn.Module,
    observation: torch.Tensor,
    *,
    suit_permutations=ALL_SUIT_PERMUTATIONS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return group-averaged logits and the unchanged legacy logits."""
    suit_permutations = tuple(suit_permutations)
    transformed = torch.stack(
        [
            permute_suit_observation(observation, permutation)
            for permutation in suit_permutations
        ]
    )
    with torch.inference_mode():
        transformed_logits = actor(transformed)
    mapped = torch.stack(
        [
            map_permuted_logits_back(logits, permutation)
            for logits, permutation in zip(
                transformed_logits, suit_permutations
            )
        ]
    )
    identity_index = suit_permutations.index(
        tuple(range(STANDARD_SUIT_COUNT))
    )
    return mapped.mean(dim=0), mapped[identity_index]


class SuitSymmetryPPOAgent(DrawRetryCorrectedPPOAgent):
    """Frozen PPO with full ordinary-suit group averaging at inference."""

    def __init__(self, game_instance, actor, device, *, stats=None):
        shared_stats = stats if stats is not None else SuitSymmetryStats()
        super().__init__(
            game_instance,
            actor,
            _resolve_device(device),
            stats=shared_stats,
        )
        self.type = "frozen_greedy_ppo_suit_symmetry_ensemble"

    def act_rl(self, observation, temperature=None):
        device_observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )
        averaged_logits, legacy_logits = suit_averaged_logits(
            self.actor,
            device_observation,
        )
        legacy_mask = build_action_masks(device_observation)
        legacy_action = int(
            torch.argmax(
                legacy_logits.masked_fill(
                    ~legacy_mask,
                    torch.finfo(legacy_logits.dtype).min,
                )
            ).item()
        )
        if self._in_draw_retry:
            if self._draw_retry_card_number is None:
                raise RuntimeError("draw retry has no tracked drawn card")
            action_mask = draw_retry_action_mask(
                device_observation,
                self._draw_retry_card_number,
            )
        else:
            action_mask = legacy_mask
        action = int(
            torch.argmax(
                averaged_logits.masked_fill(
                    ~action_mask,
                    torch.finfo(averaged_logits.dtype).min,
                )
            ).item()
        )

        self.stats.decisions += 1
        self.stats.suit_decisions += int(
            device_observation[MODE_INDEX] != 0
        )
        self.stats.ordinary_decisions += int(
            device_observation[MODE_INDEX] == 0
        )
        self.stats.changed_from_legacy += int(action != legacy_action)
        if self._in_draw_retry:
            self.stats.retry_decisions += 1
            self.stats.retry_overrides += int(action != legacy_action)
            self.stats.retry_legacy_wrong_card_actions += int(
                legacy_action
                not in (self._draw_retry_card_number, PASS_ACTION)
            )
            self.stats.retry_drawn_card_actions += int(
                action == self._draw_retry_card_number
            )
            self.stats.retry_pass_actions += int(action == PASS_ACTION)
        return (None if action == PASS_ACTION else action), 0.0


def make_suit_symmetry_ppo_factory(
    actor,
    device,
    *,
    stats: SuitSymmetryStats | None = None,
) -> Callable:
    shared_stats = stats if stats is not None else SuitSymmetryStats()

    def factory(game_instance):
        return SuitSymmetryPPOAgent(
            game_instance,
            actor,
            device,
            stats=shared_stats,
        )

    factory.__name__ = "SuitSymmetryPPOFactory"
    factory.stats = shared_stats
    return factory
