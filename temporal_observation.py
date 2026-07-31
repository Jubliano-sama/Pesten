"""Information-safe inputs for the recurrent Pesten policy.

This module deliberately has no dependency on ``game.Game``.  Hidden world
state cannot be represented by the policy input types: an observation contains
the observing player's own hand plus public counters, while event tokens
contain only facts that every player can observe.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
import json
from typing import Iterable, Sequence

import card


MAX_PLAYERS = 13
CARD_TYPES = 50
ACTION_COUNT = 55
SUIT_ACTION_START = 50
PASS_ACTION = 54

# 50 own-card counts, current suit (5), current rank (14), relative public hand
# sizes (13), public suit-play counts (5), and seven scalar public/private
# features.
OBSERVATION_SIZE = 94
EVENT_TOKEN_SIZE = 5


class EventKind(IntEnum):
    EPISODE_START = 0
    PLAY = 1
    DRAW_COUNT = 2
    CHOOSE_SUIT = 3


@dataclass(frozen=True, slots=True)
class PublicEventToken:
    """A public event with a structurally redacted draw representation.

    ``card_id`` is valid only for a publicly played card.  Draw events expose
    the actor and count, never a card identity or deck position.
    """

    kind: EventKind
    actor_relative: int = -1
    card_id: int = -1
    suit_id: int = -1
    count: int = 0

    def __post_init__(self) -> None:
        if self.kind is EventKind.PLAY:
            if not 0 <= self.card_id < CARD_TYPES:
                raise ValueError("a public play must contain a card id")
        elif self.card_id != -1:
            raise ValueError("only a public play may contain a card id")
        if self.kind is EventKind.DRAW_COUNT and self.count < 1:
            raise ValueError("a public draw event must have a positive count")

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.kind),
            self.actor_relative,
            self.card_id,
            self.suit_id,
            self.count,
        )


@dataclass(frozen=True, slots=True)
class PlayerPrivateObservation:
    """Immutable player-relative observation accepted by actor and critic."""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.values) != OBSERVATION_SIZE:
            raise ValueError(
                f"expected {OBSERVATION_SIZE} observation values, "
                f"got {len(self.values)}"
            )


@dataclass(frozen=True, slots=True)
class DecisionInput:
    """The complete and only input crossing the policy privacy boundary."""

    observation: PlayerPrivateObservation
    legal_actions: tuple[bool, ...]
    public_events: tuple[PublicEventToken, ...] = ()

    def __post_init__(self) -> None:
        if len(self.legal_actions) != ACTION_COUNT:
            raise ValueError(
                f"expected {ACTION_COUNT} legal-action flags, "
                f"got {len(self.legal_actions)}"
            )
        if not any(self.legal_actions):
            raise ValueError("at least one action must be legal")


def relative_index(actor: int, observer: int, player_count: int, direction: int) -> int:
    """Map an absolute public player id to observer-relative turn order."""
    if direction not in (-1, 1):
        raise ValueError("direction must be -1 or 1")
    if not 0 <= actor < player_count or not 0 <= observer < player_count:
        return -1
    return ((actor - observer) * direction) % player_count


def build_private_observation(
    *,
    own_cards: Sequence[int],
    current_suit: str,
    current_rank: int,
    public_hand_sizes: Sequence[int],
    observer_index: int,
    current_player_index: int,
    direction: int,
    penalty_amount: int,
    public_sorts_played: Sequence[int],
    change_suit_mode: bool = False,
) -> PlayerPrivateObservation:
    """Construct a policy observation from safe fields only.

    There is intentionally no argument for opponent hands, draw-pile contents,
    or the full game object.
    """
    player_count = len(public_hand_sizes)
    if not 2 <= player_count <= MAX_PLAYERS:
        raise ValueError(f"player count must be in [2, {MAX_PLAYERS}]")
    if not 0 <= observer_index < player_count:
        raise ValueError("observer index is outside the player list")
    if current_suit not in card.sorts:
        raise ValueError(f"unknown suit: {current_suit!r}")
    if not 0 <= current_rank <= 13:
        raise ValueError("current rank must be in [0, 13]")
    if len(public_sorts_played) < len(card.sorts):
        raise ValueError("public_sorts_played must contain all suits")

    values = [0.0] * OBSERVATION_SIZE
    for card_id in own_cards:
        if not 0 <= card_id < CARD_TYPES:
            raise ValueError(f"invalid own card id: {card_id}")
        values[card_id] += 1.0

    values[50 + card.sorts.index(current_suit)] = 1.0
    values[55 + current_rank] = 1.0

    for absolute_index, hand_size in enumerate(public_hand_sizes):
        slot = relative_index(
            absolute_index, observer_index, player_count, direction
        )
        values[69 + slot] = float(hand_size)

    for suit_index in range(5):
        values[82 + suit_index] = float(public_sorts_played[suit_index])

    values[87] = observer_index / max(1, player_count - 1)
    values[88] = relative_index(
        current_player_index, observer_index, player_count, direction
    ) / max(1, player_count - 1)
    values[89] = float(direction)
    values[90] = float(penalty_amount)
    values[91] = float(len(own_cards))
    values[92] = float(player_count)
    values[93] = float(change_suit_mode)
    return PlayerPrivateObservation(tuple(values))


def build_legal_actions(
    *,
    own_cards: Sequence[int],
    current_suit: str,
    current_rank: int,
    penalty_amount: int,
    change_suit_mode: bool = False,
) -> tuple[bool, ...]:
    """Build the immutable action mask without consulting hidden state."""
    mask = [False] * ACTION_COUNT
    if change_suit_mode:
        for action in range(SUIT_ACTION_START, SUIT_ACTION_START + 4):
            mask[action] = True
        return tuple(mask)

    for card_id in own_cards:
        candidate = card.Card(card_id)
        if penalty_amount > 0:
            legal = candidate.truenumber == current_rank
        else:
            legal = candidate.compatible(current_suit, current_rank)
        if legal:
            mask[card_id] = True
    mask[PASS_ACTION] = True
    return tuple(mask)


def event_to_safe_dict(event: PublicEventToken) -> dict[str, int | str]:
    """Serialize one event while omitting inapplicable sensitive fields."""
    result: dict[str, int | str] = {
        "kind": event.kind.name.lower(),
        "actor_relative": event.actor_relative,
    }
    if event.kind is EventKind.PLAY:
        result["card_id"] = event.card_id
    elif event.kind is EventKind.DRAW_COUNT:
        result["count"] = event.count
    elif event.kind is EventKind.CHOOSE_SUIT:
        result["suit_id"] = event.suit_id
    return result


def serialize_safe_history(decisions: Iterable[DecisionInput]) -> bytes:
    """Return canonical JSON containing only the safe policy-side history."""
    payload = []
    for decision in decisions:
        payload.append(
            {
                "observation": list(decision.observation.values),
                "legal_actions": list(decision.legal_actions),
                "public_events": [
                    event_to_safe_dict(event)
                    for event in decision.public_events
                ],
            }
        )
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def assert_policy_schema_is_hidden_state_free() -> None:
    """Fail loudly if a later edit broadens the immutable policy schema."""
    expected = {
        "PublicEventToken": {
            "kind", "actor_relative", "card_id", "suit_id", "count"
        },
        "PlayerPrivateObservation": {"values"},
        "DecisionInput": {"observation", "legal_actions", "public_events"},
    }
    actual = {
        cls.__name__: {field.name for field in fields(cls)}
        for cls in (
            PublicEventToken,
            PlayerPrivateObservation,
            DecisionInput,
        )
    }
    if actual != expected:
        raise AssertionError(f"unsafe policy schema change: {actual!r}")
