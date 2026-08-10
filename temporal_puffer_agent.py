"""Puffer-style recurrent policy and legacy-Game adapter for Pesten.

PyPI currently exposes PufferLib 3.0.0, whose recurrent helper is an LSTM
wrapper and which has no PufferNet/MinGRU API.  On Python 3.13 it also conflicts
with this project's NumPy 2 requirement through ``ml-dtypes``.  The compact GRU
below therefore uses Puffer's encode/decode conventions without claiming native
PufferLib 4 integration.  The legacy PPO in :mod:`ppo2` remains untouched and
supported.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, Optional

import torch
from torch import nn
from torch.distributions import Categorical

import card
import deck
from temporal_observation import (
    ACTION_COUNT,
    DecisionInput,
    EVENT_TOKEN_SIZE,
    EventKind,
    OBSERVATION_SIZE,
    PASS_ACTION,
    PublicEventToken,
    SUIT_ACTION_START,
    build_legal_actions,
    build_private_observation,
    build_draw_retry_legal_actions,
    relative_index,
)

PUFFERLIB4_BLOCKER = (
    "UV_CACHE_DIR=/tmp/pesten-uv-cache uv pip install --python "
    "venv/bin/python 'pufferlib>=4' --dry-run: only pufferlib<=3.0.0 "
    "is available, so the requirement is unsatisfiable; pufferlib==3.0.0 "
    "requires numpy<2 while ml-dtypes on Python 3.13 requires numpy>=2.1"
)


@dataclass(frozen=True, slots=True)
class PolicyStep:
    logits: torch.Tensor
    value: torch.Tensor
    state: object


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    decision: DecisionInput
    action: int
    old_log_prob: float
    value: float


class PufferStyleTemporalNet(nn.Module):
    """Small event-aware GRU policy with shared safe actor/critic features."""

    is_continuous = False
    native_pufferlib = False
    integration_status = (
        "Puffer-style recurrent adapter; PufferLib 4 is unavailable and "
        "PufferLib 3.0.0 has no PufferNet/MinGRU API"
    )

    def __init__(self, hidden_size: int = 96) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.observation_encoder = nn.Sequential(
            nn.Linear(OBSERVATION_SIZE, hidden_size),
            nn.Tanh(),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(EVENT_TOKEN_SIZE, hidden_size),
            nn.Tanh(),
        )
        self.event_gru = nn.GRUCell(hidden_size, hidden_size)
        self.decision_gru = nn.GRUCell(hidden_size, hidden_size)
        self.actor = nn.Linear(hidden_size, ACTION_COUNT)
        self.critic = nn.Linear(hidden_size, 1)

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.hidden_size,
            device=device or parameter.device,
            dtype=parameter.dtype,
        )

    @staticmethod
    def _event_tensor(
        events: tuple[PublicEventToken, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not events:
            return torch.empty((0, EVENT_TOKEN_SIZE), device=device, dtype=dtype)
        # Bounded scaling keeps all public token coordinates in a useful range.
        rows = [
            (
                int(event.kind) / 3.0,
                event.actor_relative / 12.0,
                (event.card_id + 1) / 50.0,
                (event.suit_id + 1) / 5.0,
                event.count / 54.0,
            )
            for event in events
        ]
        return torch.tensor(rows, device=device, dtype=dtype)

    def update_public_memory(
        self,
        events: tuple[PublicEventToken, ...],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Apply every public event in order, including between agent turns."""
        event_rows = self._event_tensor(
            events, device=state.device, dtype=state.dtype
        )
        next_state = state
        for event_row in event_rows:
            encoded = self.event_encoder(event_row.unsqueeze(0))
            next_state = self.event_gru(encoded, next_state)
        return next_state

    def encode_observations(
        self,
        observations: torch.Tensor,
        state: Optional[dict] = None,
    ) -> torch.Tensor:
        """Puffer-compatible observation encoder hook."""
        del state
        return self.observation_encoder(observations)

    def decode_actions(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Puffer-compatible action/value decoder hook."""
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def step(
        self,
        decision: DecisionInput,
        state: torch.Tensor,
    ) -> PolicyStep:
        """Evaluate actor and non-privileged critic through the safe schema."""
        if state.shape != (1, self.hidden_size):
            raise ValueError(
                f"state must have shape (1, {self.hidden_size}), "
                f"got {tuple(state.shape)}"
            )
        memory = self.update_public_memory(decision.public_events, state)
        observation = torch.tensor(
            decision.observation.values,
            device=state.device,
            dtype=state.dtype,
        ).unsqueeze(0)
        encoded = self.encode_observations(observation)
        next_state = self.decision_gru(encoded, memory)
        logits, values = self.decode_actions(next_state)
        legal = torch.tensor(
            decision.legal_actions, device=state.device, dtype=torch.bool
        ).unsqueeze(0)
        masked_logits = logits.masked_fill(
            ~legal, torch.finfo(logits.dtype).min
        )
        return PolicyStep(
            logits=masked_logits.squeeze(0),
            value=values.squeeze(0),
            state=next_state,
        )


class TemporalPestenAgent:
    """Adapter that keeps ``Game`` on the environment side of the boundary.

    The adapter may read its own cards and explicitly public game fields.  Its
    policy receives only :class:`DecisionInput` and recurrent state.
    """

    def __init__(
        self,
        game_instance,
        policy: Optional[PufferStyleTemporalNet] = None,
        *,
        deterministic: bool = False,
        seed: int = 0,
    ) -> None:
        self.game = game_instance
        self.policy = policy or PufferStyleTemporalNet()
        self.mydeck = deck.Deck([])
        self.type = "puffer_style_temporal"
        self.deterministic = deterministic
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.state = self.policy.initial_state()
        self.trajectory: list[TrajectoryStep] = []
        self.safe_history: list[DecisionInput] = []
        self._episode_marker: Optional[tuple[int, int]] = None
        self._public_cursor = 0
        self._last_hand_sizes: tuple[int, ...] = ()
        self._last_pass_turn: Optional[int] = None
        self._retry_card: Optional[int] = None
        self._retry_turn: Optional[int] = None

    def reset_episode(self) -> None:
        self.state = self.policy.initial_state()
        self.trajectory.clear()
        self.safe_history.clear()
        self._public_cursor = 0
        self._last_hand_sizes = ()
        self._last_pass_turn = None
        self._retry_card = None
        self._retry_turn = None

    def _retry_card_for_current_turn(self) -> Optional[int]:
        if self._retry_turn != self.game.turn_count:
            self._retry_card = None
            self._retry_turn = None
        return self._retry_card

    def _ensure_episode(self) -> None:
        marker = (id(self.game.grab_deck), id(self.game.game_deck))
        if marker == self._episode_marker:
            return
        self._episode_marker = marker
        self.reset_episode()
        self._last_hand_sizes = tuple(
            player.mydeck.cardCount() for player in self.game.players
        )

    def _observer_index(self) -> int:
        return self.game.players.index(self)

    def _sync_public_events(self) -> tuple[PublicEventToken, ...]:
        self._ensure_episode()
        observer = self._observer_index()
        player_count = len(self.game.players)
        direction = self.game.direction
        events: list[PublicEventToken] = []
        if self._public_cursor == 0 and not self.safe_history:
            events.append(PublicEventToken(EventKind.EPISODE_START))

        predicted_sizes = list(self._last_hand_sizes)
        new_history = self.game.public_history[self._public_cursor :]
        for item in new_history:
            actor = int(item.get("player", -1))
            actor_relative = relative_index(
                actor, observer, player_count, direction
            )
            action = item.get("action")
            if action == "play":
                events.append(
                    PublicEventToken(
                        EventKind.PLAY,
                        actor_relative=actor_relative,
                        card_id=int(item["card"]),
                        suit_id=card.sorts.index(item["sort"]),
                        count=1,
                    )
                )
                if 0 <= actor < len(predicted_sizes):
                    predicted_sizes[actor] -= 1
            elif action == "choose_sort":
                events.append(
                    PublicEventToken(
                        EventKind.CHOOSE_SUIT,
                        actor_relative=actor_relative,
                        suit_id=card.sorts.index(item["sort"]),
                    )
                )

        current_sizes = tuple(
            player.mydeck.cardCount() for player in self.game.players
        )
        # A legacy Game exposes hand-size changes but not opponent draw hooks.
        # Reconstruct the public draw count after accounting for public plays.
        for actor, (predicted, current) in enumerate(
            zip(predicted_sizes, current_sizes)
        ):
            draw_count = current - predicted
            if draw_count > 0:
                events.append(
                    PublicEventToken(
                        EventKind.DRAW_COUNT,
                        actor_relative=relative_index(
                            actor, observer, player_count, direction
                        ),
                        count=draw_count,
                    )
                )

        self._public_cursor = len(self.game.public_history)
        self._last_hand_sizes = current_sizes
        return tuple(events)

    def _decision(self, *, change_suit_mode: bool = False) -> DecisionInput:
        events = self._sync_public_events()
        observer = self._observer_index()
        own_cards = tuple(candidate.number for candidate in self.mydeck.cards)
        observation = build_private_observation(
            own_cards=own_cards,
            current_suit=self.game.current_sort,
            current_rank=self.game.current_true_number,
            public_hand_sizes=tuple(
                player.mydeck.cardCount() for player in self.game.players
            ),
            observer_index=observer,
            current_player_index=self.game.current_player_index,
            direction=self.game.direction,
            penalty_amount=self.game.penalty_amount,
            public_sorts_played=tuple(self.game.sorts_played),
            change_suit_mode=change_suit_mode,
        )
        legal = build_legal_actions(
            own_cards=own_cards,
            current_suit=self.game.current_sort,
            current_rank=self.game.current_true_number,
            penalty_amount=self.game.penalty_amount,
            change_suit_mode=change_suit_mode,
        )
        retry_card = self._retry_card_for_current_turn()
        if retry_card is not None and not change_suit_mode:
            legal = build_draw_retry_legal_actions(retry_card)
        return DecisionInput(observation, legal, events)

    def _act(self, decision: DecisionInput) -> int:
        with torch.no_grad():
            result = self.policy.step(decision, self.state)
            distribution = Categorical(logits=result.logits)
            if self.deterministic:
                action = int(torch.argmax(result.logits).item())
            else:
                probabilities = torch.softmax(result.logits, dim=-1).cpu()
                action = int(
                    torch.multinomial(
                        probabilities, 1, generator=self.generator
                    ).item()
                )
            old_log_prob = float(
                distribution.log_prob(
                    torch.tensor(action, device=result.logits.device)
                ).item()
            )
        if isinstance(result.state, torch.Tensor):
            self.state = result.state.detach()
        else:
            self.state = tuple(part.detach() for part in result.state)
        self.safe_history.append(decision)
        self.trajectory.append(
            TrajectoryStep(
                decision=decision,
                action=action,
                old_log_prob=old_log_prob,
                value=float(result.value.item()),
            )
        )
        return action

    def playCard(self, current_sort, current_true_number):
        del current_sort, current_true_number
        retry_active = self._retry_card_for_current_turn() is not None
        decision = self._decision()
        action = self._act(decision)
        if action == PASS_ACTION:
            self._last_pass_turn = (
                None
                if retry_active or self.game.penalty_amount > 0
                else self.game.turn_count
            )
            self._retry_card = None
            self._retry_turn = None
            return None
        self._last_pass_turn = None
        self._retry_card = None
        self._retry_turn = None
        return card.Card(action)

    def changeSort(self):
        action = self._act(self._decision(change_suit_mode=True))
        return card.sorts[action - SUIT_ACTION_START]

    def addCard(self, drawn_card) -> None:
        """Own draws stay private in ``mydeck`` and the next own observation."""
        if drawn_card is None:
            logging.error("None card detected")
            return
        # Game.reset clears hands before dealing.  Detect its first deal so
        # recurrent memory is reset at the boundary, not merely at the first
        # later policy call.  The final game-deck marker is checked again by
        # _ensure_episode once reset has finished.
        if (
            not self.mydeck.cards
            and self.game.turn_count == 0
            and not self.game.public_history
        ):
            deal_marker = (id(self.game.grab_deck), id(self.game.game_deck))
            if deal_marker != self._episode_marker:
                self._episode_marker = deal_marker
                self.reset_episode()
        self.mydeck.cards.append(drawn_card)
        if (
            self._last_pass_turn == self.game.turn_count
            and self.game.penalty_amount == 0
        ):
            self._retry_card = int(drawn_card.number)
            self._retry_turn = self.game.turn_count
            self._last_pass_turn = None

    def remove(self, played_card) -> None:
        for own_card in self.mydeck.cards:
            if own_card.number == played_card.number:
                self.mydeck.cards.remove(own_card)
                return


def temporal_agent_factory(
    policy: PufferStyleTemporalNet,
    *,
    deterministic: bool = False,
    seed: int = 0,
) -> Callable:
    """Create a legacy ``Game`` agent class sharing the supplied policy."""

    class ConfiguredTemporalAgent(TemporalPestenAgent):
        def __init__(self, game_instance) -> None:
            super().__init__(
                game_instance,
                policy,
                deterministic=deterministic,
                seed=seed,
            )

    ConfiguredTemporalAgent.__name__ = "ConfiguredTemporalAgent"
    return ConfiguredTemporalAgent
