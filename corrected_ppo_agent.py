"""Frozen PPO wrapper with exact canonical draw-retry action semantics.

The legacy ``Game`` calls ``playCard`` a second time after a voluntary draw
when the drawn card is compatible.  On that immediate retry, the engine accepts
only the just-drawn card, while the historical PPO mask also exposed compatible
cards already in hand.  This wrapper keeps the frozen actor and its 121-feature
observation unchanged, but narrows that one retry mask to the drawn card plus
PASS.

Tracking uses only normal agent callbacks: an unpenalized pass arms a draw,
``addCard`` records that newly owned card and the public turn number, and the
state is consumed by an immediate same-turn retry.  It never reads opponent
card identities or the draw pile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import torch

from ai_observation import (
    MODE_INDEX,
    OBSERVATION_SIZE,
    PASS_ACTION,
)
from hybrid_agent import FrozenGreedyPPOAgent, _resolve_device
from ppo2 import build_action_masks


@dataclass
class CorrectedPPOStats:
    """Auditable decision counters shared by any number of wrapper instances."""

    outside_retry_decisions: int = 0
    outside_retry_action_mismatches: int = 0
    normal_draws_tracked: int = 0
    retry_decisions: int = 0
    retry_overrides: int = 0
    retry_legacy_wrong_card_actions: int = 0
    retry_drawn_card_actions: int = 0
    retry_pass_actions: int = 0

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def draw_retry_action_mask(
    observation: torch.Tensor,
    drawn_card_number: int,
) -> torch.Tensor:
    """Return exactly the just-drawn card and PASS for an ordinary retry."""
    observation = torch.as_tensor(observation)
    if observation.ndim != 1 or len(observation) != OBSERVATION_SIZE:
        raise ValueError(
            f"expected one {OBSERVATION_SIZE}-feature observation"
        )
    if observation[MODE_INDEX] != 0:
        raise ValueError("draw retry cannot be a suit-choice decision")
    if not 0 <= drawn_card_number < 50:
        raise ValueError("drawn card number must be in [0, 50)")
    legacy_mask = build_action_masks(observation)
    if not bool(legacy_mask[drawn_card_number]):
        raise RuntimeError("tracked draw-retry card is not currently legal")
    corrected = torch.zeros_like(legacy_mask)
    corrected[drawn_card_number] = True
    corrected[PASS_ACTION] = True
    return corrected


class DrawRetryCorrectedPPOAgent(FrozenGreedyPPOAgent):
    """Deployment wrapper around an immutable greedy production PPO actor."""

    def __init__(self, game_instance, actor, device, *, stats=None):
        super().__init__(game_instance, actor, _resolve_device(device))
        self.type = "frozen_greedy_ppo_draw_retry_corrected"
        self.stats = stats if stats is not None else CorrectedPPOStats()
        self._awaiting_normal_draw_turn: int | None = None
        self._draw_retry_turn: int | None = None
        self._draw_retry_card_number: int | None = None
        self._in_draw_retry = False

    @property
    def tracked_draw_retry(self) -> tuple[int, int] | None:
        """Expose the complete narrow tracking state for auditing/tests."""
        if (
            self._draw_retry_card_number is None
            or self._draw_retry_turn is None
        ):
            return None
        return self._draw_retry_card_number, self._draw_retry_turn

    def _clear_draw_tracking(self) -> None:
        self._awaiting_normal_draw_turn = None
        self._draw_retry_turn = None
        self._draw_retry_card_number = None
        self._in_draw_retry = False

    def addCard(self, drawn_card):
        super().addCard(drawn_card)
        # During reset/dealing there is no public current suit.  Clearing here
        # also prevents state from a prior game surviving reuse of an agent.
        if self.game.current_sort is None:
            self._clear_draw_tracking()
            return
        turn = int(self.game.turn_count)
        if self._awaiting_normal_draw_turn == turn:
            self._draw_retry_card_number = int(drawn_card.number)
            self._draw_retry_turn = turn
            self._awaiting_normal_draw_turn = None
            self.stats.normal_draws_tracked += 1

    def act_rl(self, observation, temperature=None):
        device_observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            logits = self.actor(device_observation.unsqueeze(0)).squeeze(0)
            legacy_mask = build_action_masks(device_observation)
            legacy_logits = logits.masked_fill(
                ~legacy_mask,
                torch.finfo(logits.dtype).min,
            )
            legacy_action = int(torch.argmax(legacy_logits).item())

            if self._in_draw_retry:
                if self._draw_retry_card_number is None:
                    raise RuntimeError("draw retry has no tracked drawn card")
                action_mask = draw_retry_action_mask(
                    device_observation,
                    self._draw_retry_card_number,
                )
                corrected_logits = logits.masked_fill(
                    ~action_mask,
                    torch.finfo(logits.dtype).min,
                )
                action = int(torch.argmax(corrected_logits).item())
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
            else:
                action = legacy_action
                self.stats.outside_retry_decisions += 1
                # Kept as an explicit deployment audit invariant rather than
                # inferred later from implementation structure.
                self.stats.outside_retry_action_mismatches += int(
                    action != legacy_action
                )

        return (None if action == PASS_ACTION else action), 0.0

    def playCard(self, current_sort, current_true_number, temperature=None):
        turn = int(self.game.turn_count)
        is_retry = (
            self._draw_retry_turn == turn
            and self._draw_retry_card_number is not None
        )
        if is_retry:
            self._in_draw_retry = True
            try:
                return super().playCard(
                    current_sort,
                    current_true_number,
                    temperature,
                )
            finally:
                self._clear_draw_tracking()

        # A compatible draw from an older turn never authorizes a later retry.
        self._clear_draw_tracking()
        selected = super().playCard(
            current_sort,
            current_true_number,
            temperature,
        )
        if (
            selected is None
            and self.game.penalty_amount == 0
            and not self.game.game_over
        ):
            self._awaiting_normal_draw_turn = turn
        return selected


def make_corrected_ppo_agent_factory(
    actor,
    device,
    *,
    stats: CorrectedPPOStats | None = None,
) -> Callable:
    """Bind one frozen actor and optional shared audit counters to ``Game``."""
    shared_stats = stats if stats is not None else CorrectedPPOStats()

    def factory(game_instance):
        return DrawRetryCorrectedPPOAgent(
            game_instance,
            actor,
            device,
            stats=shared_stats,
        )

    factory.__name__ = "DrawRetryCorrectedPPOFactory"
    factory.stats = shared_stats
    return factory
