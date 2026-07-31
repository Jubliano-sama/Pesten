"""PPO-guided Bayesian information-set search for Pesten.

The frozen production PPO remains the default policy.  Search is an optional
public-information refinement:

* the root is reconstructed from the agent's hand and public game state;
* hidden cards are sampled by the existing Bayesian information-set sampler;
* the PPO supplies the root action prior and the policy model for hypothetical
  opponents; and
* posterior search values may override PPO only after configurable margins.

The implementation never inspects ``grab_deck`` or opponent card identities.
Opponent hands inside search are posterior samples, not the live hidden hands.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Iterable, Optional, Sequence

import torch

from ai_observation import (
    COMPATIBLE_END,
    COMPATIBLE_START,
    OBSERVATION_SIZE,
    PASS_ACTION,
    SUIT_ACTION_START,
    populate_game_context,
)
from hardcodedAgent import (
    HardcodedStrategyAgent,
    SPECIAL_RANKS,
    STANDARD_SUITS,
    _Action,
    _SearchState,
    _card_rank,
    _compatible,
)
from hybrid_agent import FrozenGreedyPPOAgent, _resolve_device
from ppo2 import build_action_masks


@dataclass(frozen=True)
class GuidedSearchConfig:
    """Conservative search authority and resource limits."""

    belief_samples: int = 4
    search_depth: int = 3
    beam_width: int = 3
    trigger: str = "tactical"
    decision_mode: str = "blend"
    search_weight: float = 0.25
    override_margin: float = 0.08
    search_advantage_margin: float = 0.15
    minimum_candidate_prior: float = 0.025
    lower_tail_weight: float = 0.08

    def __post_init__(self) -> None:
        if self.belief_samples < 1:
            raise ValueError("belief_samples must be positive")
        if self.search_depth < 1:
            raise ValueError("search_depth must be positive")
        if self.beam_width < 1:
            raise ValueError("beam_width must be positive")
        if self.trigger not in {"tactical", "always"}:
            raise ValueError("trigger must be 'tactical' or 'always'")
        if self.decision_mode not in {"blend", "veto"}:
            raise ValueError("decision_mode must be 'blend' or 'veto'")
        if not 0.0 <= self.search_weight <= 1.0:
            raise ValueError("search_weight must be between zero and one")
        if self.override_margin < 0.0:
            raise ValueError("override_margin cannot be negative")
        if self.search_advantage_margin < 0.0:
            raise ValueError("search_advantage_margin cannot be negative")
        if not 0.0 <= self.minimum_candidate_prior <= 1.0:
            raise ValueError("minimum_candidate_prior must be between zero and one")
        if self.lower_tail_weight < 0.0:
            raise ValueError("lower_tail_weight cannot be negative")


def _minmax(values: dict[_Action, float]) -> dict[_Action, float]:
    """Normalize finite values to [0, 1], preserving equality."""
    if not values:
        return {}
    low = min(values.values())
    high = max(values.values())
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("search scores must be finite")
    if high <= low:
        return {action: 0.5 for action in values}
    scale = high - low
    return {action: (value - low) / scale for action, value in values.items()}


class PPOGuidedBayesianAgent(FrozenGreedyPPOAgent):
    """Frozen PPO with bounded Bayesian search in public tactical states."""

    def __init__(
        self,
        game_instance,
        actor,
        device,
        *,
        config: GuidedSearchConfig | None = None,
    ):
        super().__init__(game_instance, actor, _resolve_device(device))
        self.type = "ppo_guided_bayesian_search"
        self.config = config or GuidedSearchConfig()
        self._search_core = HardcodedStrategyAgent(
            game_instance,
            belief_samples=self.config.belief_samples,
            search_depth=self.config.search_depth,
            beam_width=self.config.beam_width,
        )
        self._planned_suit: Optional[str] = None
        self.last_search_diagnostics: dict = {}
        self.decision_count = 0
        self.search_count = 0
        self.override_count = 0
        self.decision_seconds = 0.0
        self.search_seconds = 0.0
        self.decision_latencies: list[float] = []

    def _root_information_state(self) -> tuple[_SearchState, int]:
        """Build an information state without reading live hidden identities."""
        root_index = self.game.players.index(self)
        own_hand = tuple(sorted(held.number for held in self.mydeck.cards))
        hands = []
        for player_index, player in enumerate(self.game.players):
            if player_index == root_index:
                hands.append(own_hand)
            else:
                # ``cardCount`` is public; ``cards`` must remain untouched.
                hands.append(tuple([None] * player.mydeck.cardCount()))
        discard = tuple(held.number for held in self.game.game_deck.cards)
        return (
            _SearchState(
                hands=tuple(hands),
                draw_pile=(),
                discard=discard,
                current_player=self.game.current_player_index,
                direction=self.game.direction,
                current_sort=self.game.current_sort,
                current_rank=self.game.current_true_number,
                penalty=self.game.penalty_amount,
                sorts_played=tuple(
                    int(value) for value in self.game.sorts_played
                ),
                winner=self.game.winner,
                game_over=self.game.game_over,
            ),
            root_index,
        )

    @staticmethod
    def _state_observation(
        state: _SearchState,
        player_index: int,
        *,
        suit_mode: bool = False,
    ) -> torch.Tensor:
        """Encode one posterior-sampled player's legitimate observation."""
        observation = torch.zeros(OBSERVATION_SIZE, dtype=torch.float32)
        for number in state.hands[player_index]:
            observation[number] += 1
            playable = (
                _card_rank(number) == state.current_rank
                if state.penalty > 0
                else _compatible(
                    number,
                    state.current_sort,
                    state.current_rank,
                )
            )
            if playable:
                observation[number + COMPATIBLE_START] += 1
        return populate_game_context(
            observation,
            sorts_played=state.sorts_played,
            hand_sizes=[len(hand) for hand in state.hands],
            player_index=player_index,
            direction=state.direction,
            is_change_sort=suit_mode,
        )

    def _actor_logits(self, observation: torch.Tensor) -> torch.Tensor:
        device_observation = observation.to(self.device)
        with torch.inference_mode():
            return (
                self.actor(device_observation.unsqueeze(0))
                .squeeze(0)
                .float()
                .cpu()
            )

    def _action_priors(
        self,
        state: _SearchState,
        actions: Sequence[_Action],
    ) -> dict[_Action, float]:
        """Return a normalized PPO prior over the supplied legal actions."""
        if not actions:
            return {}
        player_index = state.current_player
        observation = self._state_observation(state, player_index)
        logits = self._actor_logits(observation)
        legal_mask = build_action_masks(observation)
        masked_logits = logits.masked_fill(~legal_mask, -torch.inf)
        action_log_probabilities = torch.log_softmax(masked_logits, dim=0)

        needs_suit = any(action.chosen_suit for action in actions)
        suit_logits = None
        if needs_suit:
            raw_suit_logits = self._actor_logits(
                self._state_observation(
                    state,
                    player_index,
                    suit_mode=True,
                )
            )
            suit_logits = torch.log_softmax(
                raw_suit_logits[
                    SUIT_ACTION_START:
                    SUIT_ACTION_START + len(STANDARD_SUITS)
                ],
                dim=0,
            )

        scores = []
        for action in actions:
            action_index = (
                PASS_ACTION
                if action.card_number is None
                else action.card_number
            )
            score = float(action_log_probabilities[action_index].item())
            if action.chosen_suit in STANDARD_SUITS:
                assert suit_logits is not None
                suit_index = STANDARD_SUITS.index(action.chosen_suit)
                score += float(suit_logits[suit_index].item())
            scores.append(score)

        score_tensor = torch.tensor(scores, dtype=torch.float64)
        if not torch.isfinite(score_tensor).any():
            probability = 1.0 / len(actions)
            return {action: probability for action in actions}
        probabilities = torch.softmax(score_tensor, dim=0).tolist()
        return {
            action: float(probability)
            for action, probability in zip(actions, probabilities)
        }

    @staticmethod
    def _ensure_full_root_actions(
        state: _SearchState,
        actions: Iterable[_Action],
    ) -> list[_Action]:
        """Include voluntary pass, which is legal in the canonical engine."""
        root_actions = list(actions)
        if state.penalty > 0 and state.current_rank == 0:
            pass_actions = [
                _Action(None, chosen_suit=suit) for suit in STANDARD_SUITS
            ]
        else:
            pass_actions = [_Action(None)]
        for action in pass_actions:
            if action not in root_actions:
                root_actions.append(action)
        return root_actions

    @staticmethod
    def _prior_default(
        actions: Sequence[_Action],
        priors: dict[_Action, float],
    ) -> _Action:
        return max(
            actions,
            key=lambda action: (
                priors[action],
                -(action.card_number if action.card_number is not None else 10_000),
                -(
                    STANDARD_SUITS.index(action.chosen_suit)
                    if action.chosen_suit in STANDARD_SUITS
                    else 10_000
                ),
            ),
        )

    def _ppo_default_action(
        self,
        state: _SearchState,
        actions: Sequence[_Action],
        priors: dict[_Action, float],
    ) -> _Action:
        """Map PPO's greedy card decision to one search action.

        Jack/pass suit variants are only a search representation.  If search
        does not override, ``changeSort`` is deliberately deferred to the live
        PPO call at the same point in the engine lifecycle as the baseline.
        """
        observation = self._state_observation(
            state,
            state.current_player,
        )
        logits = self._actor_logits(observation)
        legal_mask = build_action_masks(observation)
        action_index = int(
            torch.argmax(logits.masked_fill(~legal_mask, -torch.inf)).item()
        )
        card_number = None if action_index == PASS_ACTION else action_index
        matching = [
            action
            for action in actions
            if action.card_number == card_number
        ]
        if not matching:
            raise RuntimeError("PPO default is absent from legal root actions")
        return self._prior_default(matching, priors)

    def _tactical_reasons(
        self,
        state: _SearchState,
        root_index: int,
        actions: Sequence[_Action],
    ) -> list[str]:
        reasons = []
        own_size = len(state.hands[root_index])
        opponent_sizes = [
            len(hand)
            for index, hand in enumerate(state.hands)
            if index != root_index
        ]
        distinct_cards = {
            action.card_number
            for action in actions
            if action.card_number is not None
        }
        if own_size <= 3:
            reasons.append("own_endgame")
        if min(opponent_sizes, default=99) <= 2:
            reasons.append("opponent_endgame")
        if state.penalty > 0:
            reasons.append("active_penalty")
        if len(distinct_cards) >= 2 and (
            own_size <= 5 or min(opponent_sizes, default=99) <= 4
        ):
            reasons.append("public_tactical_choice")
        return reasons

    def _blend_search_has_authority(
        self,
        actions: Sequence[_Action],
        priors: dict[_Action, float],
        ppo_default: _Action,
    ) -> bool:
        """Prove whether any posterior could clear the blend override gate."""
        if self.config.decision_mode != "blend":
            return True
        prior_normalized = _minmax(priors)
        ppo_floor = (
            (1.0 - self.config.search_weight)
            * prior_normalized[ppo_default]
        )
        return any(
            action != ppo_default
            and priors[action] >= self.config.minimum_candidate_prior
            and (
                (1.0 - self.config.search_weight)
                * prior_normalized[action]
                + self.config.search_weight
                - ppo_floor
                >= self.config.override_margin
            )
            and 1.0 >= self.config.search_advantage_margin
            for action in actions
        )

    def _guided_search(
        self,
        state: _SearchState,
        *,
        root_index: int,
        depth: int,
        cache: dict,
        prior_cache: dict,
    ) -> tuple[float, int]:
        if depth <= 0 or state.game_over:
            return self._search_core._evaluate(state, root_index), 1

        cache_key = (state, root_index, depth)
        if cache_key in cache:
            return cache[cache_key], 0

        actions = self._search_core._legal_actions(state)
        if not actions:
            return self._search_core._evaluate(state, root_index), 1
        if state in prior_cache:
            priors = prior_cache[state]
        else:
            priors = self._action_priors(state, actions)
            prior_cache[state] = priors

        # The frozen PPO is the branch prior.  A tiny heuristic tie breaker
        # keeps ordering stable if the network assigns equal probabilities.
        actions = sorted(
            actions,
            key=lambda action: (
                priors[action],
                self._search_core._action_order_score(state, action),
            ),
            reverse=True,
        )[: self.config.beam_width]

        values = []
        nodes = 1
        for action in actions:
            value, visited = self._guided_search(
                self._search_core._apply_action(state, action),
                root_index=root_index,
                depth=depth - 1,
                cache=cache,
                prior_cache=prior_cache,
            )
            nodes += visited
            values.append((action, value))

        if state.current_player == root_index:
            result = max(value for _, value in values)
        else:
            # Hypothetical opponents follow the frozen PPO distribution.  This
            # is an opponent model over sampled hands, never a read of reality.
            retained_mass = sum(priors[action] for action, _ in values)
            if retained_mass <= 0.0:
                result = sum(value for _, value in values) / len(values)
            else:
                result = sum(
                    priors[action] * value for action, value in values
                ) / retained_mass
        cache[cache_key] = result
        return result, nodes

    def _posterior_root_values(
        self,
        root_state: _SearchState,
        root_index: int,
        root_actions: Sequence[_Action],
    ) -> tuple[dict[_Action, float], int]:
        self._search_core.mydeck = self.mydeck
        sampled_states = self._search_core._sample_hidden_states(
            root_state,
            root_index,
        )
        samples = {action: [] for action in root_actions}
        nodes = 0
        for sampled_state in sampled_states:
            cache: dict = {}
            prior_cache: dict = {}
            for action in root_actions:
                value, visited = self._guided_search(
                    self._search_core._apply_action(sampled_state, action),
                    root_index=root_index,
                    depth=self.config.search_depth - 1,
                    cache=cache,
                    prior_cache=prior_cache,
                )
                samples[action].append(value)
                nodes += visited

        posterior = {}
        for action, values in samples.items():
            ordered = sorted(values)
            lower_quartile = ordered[
                max(0, math.ceil(len(ordered) * 0.25) - 1)
            ]
            posterior[action] = (
                sum(values) / len(values)
                + self.config.lower_tail_weight * lower_quartile
            )
        return posterior, nodes

    def _select_action(
        self,
        actions: Sequence[_Action],
        priors: dict[_Action, float],
        posterior: dict[_Action, float],
        ppo_default: _Action,
    ) -> tuple[_Action, dict]:
        prior_normalized = _minmax(priors)
        search_normalized = _minmax(posterior)

        if self.config.decision_mode == "veto":
            candidate = max(
                actions,
                key=lambda action: (
                    search_normalized[action],
                    priors[action],
                ),
            )
            combined = dict(search_normalized)
            clears_decision_margin = (
                search_normalized[candidate]
                - search_normalized[ppo_default]
                >= self.config.search_advantage_margin
            )
        else:
            combined = {
                action: (
                    (1.0 - self.config.search_weight)
                    * prior_normalized[action]
                    + self.config.search_weight
                    * search_normalized[action]
                )
                for action in actions
            }
            candidate = max(
                actions,
                key=lambda action: (
                    combined[action],
                    priors[action],
                ),
            )
            clears_decision_margin = (
                combined[candidate] - combined[ppo_default]
                >= self.config.override_margin
                and search_normalized[candidate]
                - search_normalized[ppo_default]
                >= self.config.search_advantage_margin
            )

        clears_prior_floor = (
            priors[candidate] >= self.config.minimum_candidate_prior
        )
        chosen = (
            candidate
            if candidate != ppo_default
            and clears_decision_margin
            and clears_prior_floor
            else ppo_default
        )
        return chosen, {
            "candidate": self._search_core._action_label(candidate),
            "overrode_ppo": chosen != ppo_default,
            "clears_decision_margin": clears_decision_margin,
            "clears_prior_floor": clears_prior_floor,
            "prior_normalized": {
                self._search_core._action_label(action): round(value, 6)
                for action, value in prior_normalized.items()
            },
            "search_normalized": {
                self._search_core._action_label(action): round(value, 6)
                for action, value in search_normalized.items()
            },
            "combined": {
                self._search_core._action_label(action): round(value, 6)
                for action, value in combined.items()
            },
        }

    def playCard(self, current_sort, current_true_number, temperature=None):
        """Choose a legal card using PPO by default and bounded search on demand."""
        started = time.perf_counter()
        self.decision_count += 1
        self._planned_suit = None
        search_elapsed = 0.0
        try:
            observation = self.obs()
            if not torch.any(
                observation[COMPATIBLE_START:COMPATIBLE_END] >= 1
            ):
                self.last_search_diagnostics = {
                    "invoked": False,
                    "reason": "forced_pass",
                    "chosen": "pass",
                }
                return None

            root_state, root_index = self._root_information_state()
            root_actions = self._ensure_full_root_actions(
                root_state,
                self._search_core._legal_actions(root_state),
            )
            priors = self._action_priors(root_state, root_actions)
            ppo_default = self._ppo_default_action(
                root_state,
                root_actions,
                priors,
            )
            tactical_reasons = self._tactical_reasons(
                root_state,
                root_index,
                root_actions,
            )
            should_search = (
                len(root_actions) >= 2
                and (
                    self.config.trigger == "always"
                    or bool(tactical_reasons)
                )
            )
            authority_limited = (
                should_search
                and not self._blend_search_has_authority(
                    root_actions,
                    priors,
                    ppo_default,
                )
            )
            should_search = should_search and not authority_limited
            if not should_search:
                chosen = ppo_default
                self.last_search_diagnostics = {
                    "invoked": False,
                    "reason": (
                        "search_cannot_clear_prior_margin"
                        if authority_limited
                        else "non_tactical_public_state"
                    ),
                    "tactical_reasons": tactical_reasons,
                    "ppo_default": self._search_core._action_label(
                        ppo_default
                    ),
                    "chosen": self._search_core._action_label(chosen),
                }
            else:
                search_started = time.perf_counter()
                posterior, nodes = self._posterior_root_values(
                    root_state,
                    root_index,
                    root_actions,
                )
                search_elapsed = time.perf_counter() - search_started
                self.search_count += 1
                self.search_seconds += search_elapsed
                chosen, selection = self._select_action(
                    root_actions,
                    priors,
                    posterior,
                    ppo_default,
                )
                self.last_search_diagnostics = {
                    "invoked": True,
                    "decision_mode": self.config.decision_mode,
                    "tactical_reasons": tactical_reasons,
                    "belief_samples": self.config.belief_samples,
                    "search_depth": self.config.search_depth,
                    "beam_width": self.config.beam_width,
                    "nodes": nodes,
                    "ppo_default": self._search_core._action_label(
                        ppo_default
                    ),
                    "chosen": self._search_core._action_label(chosen),
                    "posterior_action_values": {
                        self._search_core._action_label(action): round(
                            value,
                            6,
                        )
                        for action, value in posterior.items()
                    },
                    **selection,
                }
                if selection["overrode_ppo"]:
                    self.override_count += 1

            # Preserve the baseline engine timing for PPO's own jack/pass suit
            # choice. Only a genuine search override may pre-plan the suit.
            self._planned_suit = (
                chosen.chosen_suit
                if self.last_search_diagnostics.get("overrode_ppo", False)
                else None
            )
            if chosen.card_number is None:
                return None
            return next(
                held
                for held in self.mydeck.cards
                if held.number == chosen.card_number
            )
        finally:
            elapsed = time.perf_counter() - started
            self.decision_seconds += elapsed
            self.decision_latencies.append(elapsed)
            if self.last_search_diagnostics:
                self.last_search_diagnostics["decision_seconds"] = elapsed
                self.last_search_diagnostics["search_seconds"] = search_elapsed

    def changeSort(self):
        if self._planned_suit in STANDARD_SUITS:
            chosen = self._planned_suit
            self._planned_suit = None
            return chosen
        return super().changeSort()


def make_guided_search_factory(
    actor,
    device,
    *,
    config: GuidedSearchConfig | None = None,
    on_create: Callable[[PPOGuidedBayesianAgent], None] | None = None,
) -> Callable:
    """Bind one frozen actor and search configuration to a game factory."""
    frozen_config = config or GuidedSearchConfig()

    def factory(game_instance):
        agent = PPOGuidedBayesianAgent(
            game_instance,
            actor,
            device,
            config=frozen_config,
        )
        if on_create is not None:
            on_create(agent)
        return agent

    factory.__name__ = "PPOGuidedBayesianFactory"
    return factory
