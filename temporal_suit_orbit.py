"""Exact full-suit-orbit inference for recurrent temporal policies.

The four ordinary Pesten suits are interchangeable.  This module evaluates a
recurrent policy on every one of their 24 relabelings, preserves one recurrent
state per relabeling, maps action logits back to the caller's labels, and
averages them.  Jokers, PASS, ranks, player-relative fields, and public draw
counts are fixed by the group action.

Only :class:`temporal_observation.DecisionInput` crosses this wrapper.  Public
PLAY events are checked for the exact redundant card/suit relationship emitted
by both the scalar and batched game adapters before they are relabeled.
"""

from __future__ import annotations

from itertools import permutations
from typing import Sequence

import torch
from torch import nn

from temporal_observation import (
    ACTION_COUNT,
    CARD_TYPES,
    DecisionInput,
    EventKind,
    OBSERVATION_SIZE,
    PASS_ACTION,
    PlayerPrivateObservation,
    PublicEventToken,
    SUIT_ACTION_START,
)
from temporal_puffer_agent import PolicyStep


ORDINARY_SUIT_COUNT = 4
CARDS_PER_ORDINARY_SUIT = 12
SPECIAL_SUIT_ID = 4
ALL_SUIT_PERMUTATIONS = tuple(
    permutations(range(ORDINARY_SUIT_COUNT))
)
IDENTITY_SUIT_PERMUTATION = tuple(range(ORDINARY_SUIT_COUNT))

OWN_CARDS_START = 0
CURRENT_SUIT_START = 50
PUBLIC_SUIT_COUNTS_START = 82


def _checked_permutation(
    suit_permutation: Sequence[int],
) -> tuple[int, int, int, int]:
    permutation = tuple(int(value) for value in suit_permutation)
    if sorted(permutation) != list(range(ORDINARY_SUIT_COUNT)):
        raise ValueError("suit_permutation must permute 0,1,2,3")
    return permutation  # type: ignore[return-value]


def inverse_suit_permutation(
    suit_permutation: Sequence[int],
) -> tuple[int, int, int, int]:
    """Return the inverse of an original-to-relabeled suit permutation."""
    permutation = _checked_permutation(suit_permutation)
    return tuple(permutation.index(suit) for suit in range(4))  # type: ignore[return-value]


def relabel_card_id(card_id: int, suit_permutation: Sequence[int]) -> int:
    """Relabel one card ID, leaving the two special cards fixed."""
    permutation = _checked_permutation(suit_permutation)
    if not 0 <= card_id < CARD_TYPES:
        raise ValueError(f"card_id is out of range: {card_id}")
    if card_id >= ORDINARY_SUIT_COUNT * CARDS_PER_ORDINARY_SUIT:
        return card_id
    suit, rank_offset = divmod(card_id, CARDS_PER_ORDINARY_SUIT)
    return permutation[suit] * CARDS_PER_ORDINARY_SUIT + rank_offset


def relabel_suit_id(suit_id: int, suit_permutation: Sequence[int]) -> int:
    """Relabel an ordinary suit; keep the special/no-suit sentinels fixed."""
    permutation = _checked_permutation(suit_permutation)
    if 0 <= suit_id < ORDINARY_SUIT_COUNT:
        return permutation[suit_id]
    if suit_id in (-1, SPECIAL_SUIT_ID):
        return suit_id
    raise ValueError(f"suit_id is out of range: {suit_id}")


def relabel_action(action: int, suit_permutation: Sequence[int]) -> int:
    """Map one canonical action index into relabeled coordinates."""
    if not 0 <= action < ACTION_COUNT:
        raise ValueError(f"action is out of range: {action}")
    if action < 48:
        return relabel_card_id(action, suit_permutation)
    if SUIT_ACTION_START <= action < SUIT_ACTION_START + 4:
        permutation = _checked_permutation(suit_permutation)
        return SUIT_ACTION_START + permutation[action - SUIT_ACTION_START]
    # Special-card actions 48/49 and PASS 54 are group fixed points.
    return action


def validate_public_event_suit_contract(event: PublicEventToken) -> None:
    """Check the exact human-visible card/suit contract used by adapters."""
    if not isinstance(event, PublicEventToken):
        raise TypeError("public events must be PublicEventToken values")
    if event.kind is EventKind.PLAY:
        expected_suit = (
            event.card_id // CARDS_PER_ORDINARY_SUIT
            if event.card_id < 48
            else SPECIAL_SUIT_ID
        )
        if event.suit_id != expected_suit:
            raise ValueError(
                "public PLAY card_id/suit_id mismatch: "
                f"card {event.card_id} implies suit {expected_suit}, "
                f"got {event.suit_id}"
            )
        if event.count != 1:
            raise ValueError("public PLAY events must have count=1")
        return
    if event.kind is EventKind.CHOOSE_SUIT:
        if not 0 <= event.suit_id < ORDINARY_SUIT_COUNT:
            raise ValueError(
                "public CHOOSE_SUIT must name one ordinary suit"
            )
        if event.card_id != -1 or event.count != 0:
            raise ValueError(
                "public CHOOSE_SUIT cannot carry a card or count"
            )
        return
    if event.suit_id != -1:
        raise ValueError(
            f"public {event.kind.name} cannot carry a suit_id"
        )
    if event.kind is EventKind.EPISODE_START and (
        event.actor_relative != -1
        or event.card_id != -1
        or event.count != 0
    ):
        raise ValueError("EPISODE_START must contain only its event kind")
    if event.kind is EventKind.DRAW_COUNT and event.card_id != -1:
        raise ValueError("DRAW_COUNT cannot carry a card identity")


def relabel_public_event(
    event: PublicEventToken,
    suit_permutation: Sequence[int],
) -> PublicEventToken:
    """Relabel every and only suit-bearing field of one public event."""
    validate_public_event_suit_contract(event)
    if event.kind is EventKind.PLAY:
        return PublicEventToken(
            kind=event.kind,
            actor_relative=event.actor_relative,
            card_id=relabel_card_id(event.card_id, suit_permutation),
            suit_id=relabel_suit_id(event.suit_id, suit_permutation),
            count=event.count,
        )
    if event.kind is EventKind.CHOOSE_SUIT:
        return PublicEventToken(
            kind=event.kind,
            actor_relative=event.actor_relative,
            suit_id=relabel_suit_id(event.suit_id, suit_permutation),
        )
    return event


def relabel_observation(
    observation: PlayerPrivateObservation,
    suit_permutation: Sequence[int],
) -> PlayerPrivateObservation:
    """Relabel the three ordinary-suit blocks in the 94-value schema."""
    if not isinstance(observation, PlayerPrivateObservation):
        raise TypeError("observation must be PlayerPrivateObservation")
    permutation = _checked_permutation(suit_permutation)
    source = observation.values
    if len(source) != OBSERVATION_SIZE:
        raise ValueError(f"expected {OBSERVATION_SIZE} observation values")
    transformed = list(source)
    for original_suit, transformed_suit in enumerate(permutation):
        original_cards = slice(
            OWN_CARDS_START
            + original_suit * CARDS_PER_ORDINARY_SUIT,
            OWN_CARDS_START
            + (original_suit + 1) * CARDS_PER_ORDINARY_SUIT,
        )
        transformed_cards = slice(
            OWN_CARDS_START
            + transformed_suit * CARDS_PER_ORDINARY_SUIT,
            OWN_CARDS_START
            + (transformed_suit + 1) * CARDS_PER_ORDINARY_SUIT,
        )
        transformed[transformed_cards] = source[original_cards]
        transformed[CURRENT_SUIT_START + transformed_suit] = source[
            CURRENT_SUIT_START + original_suit
        ]
        transformed[PUBLIC_SUIT_COUNTS_START + transformed_suit] = source[
            PUBLIC_SUIT_COUNTS_START + original_suit
        ]
    return PlayerPrivateObservation(tuple(transformed))


def relabel_legal_actions(
    legal_actions: Sequence[bool],
    suit_permutation: Sequence[int],
) -> tuple[bool, ...]:
    """Map a canonical legal mask into relabeled action coordinates."""
    if len(legal_actions) != ACTION_COUNT:
        raise ValueError(f"expected {ACTION_COUNT} legal-action flags")
    transformed = [False] * ACTION_COUNT
    for original_action, is_legal in enumerate(legal_actions):
        transformed[relabel_action(original_action, suit_permutation)] = bool(
            is_legal
        )
    return tuple(transformed)


def relabel_decision(
    decision: DecisionInput,
    suit_permutation: Sequence[int],
) -> DecisionInput:
    """Relabel a complete safe decision, including its public event delta."""
    if not isinstance(decision, DecisionInput):
        raise TypeError("suit-orbit policies accept only DecisionInput")
    return DecisionInput(
        observation=relabel_observation(
            decision.observation, suit_permutation
        ),
        legal_actions=relabel_legal_actions(
            decision.legal_actions, suit_permutation
        ),
        public_events=tuple(
            relabel_public_event(event, suit_permutation)
            for event in decision.public_events
        ),
    )


def map_relabelled_logits_back(
    relabelled_logits: torch.Tensor,
    suit_permutation: Sequence[int],
) -> torch.Tensor:
    """Map a relabeled policy's logits into the caller's action labels."""
    if relabelled_logits.ndim != 1 or len(relabelled_logits) != ACTION_COUNT:
        raise ValueError(f"expected one {ACTION_COUNT}-action logit vector")
    original_logits = relabelled_logits.clone()
    for original_action in range(ACTION_COUNT):
        original_logits[original_action] = relabelled_logits[
            relabel_action(original_action, suit_permutation)
        ]
    return original_logits


class FullSuitOrbitRecurrentPolicy(nn.Module):
    """Maintain 24 recurrent worlds and average their canonical logits."""

    architecture = "full_24_suit_orbit_recurrent"
    critic_privileged = False

    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        required = ("initial_state", "step", "num_layers", "hidden_size", "cell")
        missing = tuple(name for name in required if not hasattr(policy, name))
        if missing:
            raise TypeError(
                "wrapped policy lacks recurrent contract fields: "
                + ", ".join(missing)
            )
        if getattr(policy, "cell") not in {"gru", "lstm"}:
            raise TypeError("wrapped policy cell must be GRU or LSTM")
        self.policy = policy
        self.num_layers = int(getattr(policy, "num_layers"))
        self.hidden_size = int(getattr(policy, "hidden_size"))
        self.cell = str(getattr(policy, "cell"))
        self.suit_permutations = ALL_SUIT_PERMUTATIONS

    @property
    def orbit_size(self) -> int:
        return len(self.suit_permutations)

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if batch_size != 1:
            raise ValueError(
                "FullSuitOrbitRecurrentPolicy represents one environment; "
                "its hidden batch axis is reserved for the 24 suit worlds"
            )
        return self.policy.initial_state(self.orbit_size, device=device)

    def _validate_orbit_state(
        self,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        expected = (self.num_layers, self.orbit_size, self.hidden_size)
        if self.cell == "gru":
            if not isinstance(state, torch.Tensor) or tuple(state.shape) != expected:
                raise TypeError(f"GRU orbit state must have shape {expected}")
            return
        if (
            not isinstance(state, tuple)
            or len(state) != 2
            or any(
                not isinstance(part, torch.Tensor)
                or tuple(part.shape) != expected
                for part in state
            )
        ):
            raise TypeError(
                f"LSTM orbit state must be a pair with shape {expected}"
            )

    @staticmethod
    def _state_slice(
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        index: int,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(state, torch.Tensor):
            return state[:, index : index + 1, :]
        return tuple(
            part[:, index : index + 1, :] for part in state
        )  # type: ignore[return-value]

    @staticmethod
    def _stack_states(
        states: Sequence[
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]
        ],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        first = states[0]
        if isinstance(first, torch.Tensor):
            return torch.cat(
                [state for state in states if isinstance(state, torch.Tensor)],
                dim=1,
            )
        return (
            torch.cat(
                [
                    state[0]
                    for state in states
                    if isinstance(state, tuple)
                ],
                dim=1,
            ),
            torch.cat(
                [
                    state[1]
                    for state in states
                    if isinstance(state, tuple)
                ],
                dim=1,
            ),
        )

    def step(
        self,
        decision: DecisionInput,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> PolicyStep:
        if not isinstance(decision, DecisionInput):
            raise TypeError("suit-orbit policies accept only DecisionInput")
        self._validate_orbit_state(state)
        # The target residual GRU can process the 24 independent recurrent
        # worlds as one true batch.  This is algebraically identical to the
        # reference loop below and avoids 24 tiny PyTorch dispatches per turn.
        from residual_temporal_policy import ResidualRecurrentTemporalPolicy

        if isinstance(self.policy, ResidualRecurrentTemporalPolicy):
            return self._step_residual_batch(decision, state)

        mapped_logits = []
        values = []
        next_states = []
        for orbit_index, permutation in enumerate(self.suit_permutations):
            result = self.policy.step(
                relabel_decision(decision, permutation),
                self._state_slice(state, orbit_index),
            )
            mapped_logits.append(
                map_relabelled_logits_back(result.logits, permutation)
            )
            values.append(result.value.reshape(()))
            next_states.append(result.state)

        stacked_logits = torch.stack(mapped_logits)
        legal = torch.tensor(
            decision.legal_actions,
            device=stacked_logits.device,
            dtype=torch.bool,
        )
        # Do not sum repeated finfo.min sentinels: that can overflow to -inf.
        averaged_logits = stacked_logits.masked_fill(~legal, 0.0).mean(dim=0)
        averaged_logits = averaged_logits.masked_fill(
            ~legal, torch.finfo(averaged_logits.dtype).min
        )
        return PolicyStep(
            logits=averaged_logits,
            value=torch.stack(values).mean(),
            state=self._stack_states(next_states),
        )

    def _step_residual_batch(
        self,
        decision: DecisionInput,
        state: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> PolicyStep:
        """Vectorized equivalent of 24 ``ResidualRecurrentPolicy.step`` calls."""
        from residual_temporal_policy import (
            ResidualRecurrentTemporalPolicy,
            safe_to_legacy_observation,
        )
        from temporal_model_zoo import decision_safe_tokens

        policy = self.policy
        if not isinstance(policy, ResidualRecurrentTemporalPolicy):
            raise TypeError("batched orbit path requires residual recurrent policy")
        transformed = tuple(
            relabel_decision(decision, permutation)
            for permutation in self.suit_permutations
        )
        parameter = next(policy.residual_actor.parameters())
        tokens = torch.tensor(
            [
                decision_safe_tokens(transformed_decision)
                for transformed_decision in transformed
            ],
            device=parameter.device,
            dtype=parameter.dtype,
        )
        hidden, next_state = policy.residual_recurrent(
            policy.token_projection(tokens), state
        )
        final_hidden = hidden[:, -1]
        teacher_parameter = next(policy.teacher.parameters())
        legacy = torch.stack(
            [
                safe_to_legacy_observation(transformed_decision)
                for transformed_decision in transformed
            ]
        ).to(
            device=teacher_parameter.device,
            dtype=teacher_parameter.dtype,
        )
        with torch.no_grad():
            base_logits = policy.teacher(legacy)
        combined = policy._combine(
            base_logits, policy.residual_actor(final_hidden)
        )
        legal_by_orbit = torch.tensor(
            [
                transformed_decision.legal_actions
                for transformed_decision in transformed
            ],
            device=combined.device,
            dtype=torch.bool,
        )
        combined = combined.masked_fill(
            ~legal_by_orbit, torch.finfo(combined.dtype).min
        )
        mapped_logits = torch.stack(
            [
                map_relabelled_logits_back(logits, permutation)
                for logits, permutation in zip(
                    combined, self.suit_permutations
                )
            ]
        )
        canonical_legal = torch.tensor(
            decision.legal_actions,
            device=mapped_logits.device,
            dtype=torch.bool,
        )
        averaged_logits = mapped_logits.masked_fill(
            ~canonical_legal, 0.0
        ).mean(dim=0)
        averaged_logits = averaged_logits.masked_fill(
            ~canonical_legal, torch.finfo(averaged_logits.dtype).min
        )
        return PolicyStep(
            logits=averaged_logits,
            value=policy.critic(final_hidden).squeeze(-1).mean(),
            state=next_state,
        )
