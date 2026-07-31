"""Bayesian information-set tree-search agent for this Pesten ruleset.

Only public information is used:

* the agent's own hand;
* public hand sizes and turn state;
* the discard pile;
* cards each player previously played; and
* publicly announced suit choices.

Hidden hands are sampled from a Bayesian posterior.  A depth-limited paranoid
tree search then considers the plausible replies of every remaining player.
The policy is deterministic: its random generator is seeded from the public
information set, so an identical information set produces an identical move.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import random
from typing import Iterable, Optional

import card
import deck


SPECIAL_RANKS = frozenset({0, 1, 2, 7, 8, 13})
STANDARD_SUITS = card.sorts[:4]
FULL_DECK = tuple(range(50)) + (49, 48, 48, 48)
FULL_RANK_COUNTS = Counter(card.Card(number).truenumber for number in FULL_DECK)


@lru_cache(maxsize=None)
def _card_sort(number: int) -> str:
    return card.Card(number).sort


@lru_cache(maxsize=None)
def _card_rank(number: int) -> int:
    return card.Card(number).truenumber


def _compatible(number: int, current_sort: str, current_rank: int) -> bool:
    return (
        _card_rank(number) == current_rank
        or _card_rank(number) in {0, 13}
        or _card_sort(number) == current_sort
    )


@dataclass(frozen=True)
class _Action:
    card_number: Optional[int]
    chosen_suit: Optional[str] = None


@dataclass(frozen=True)
class _SearchState:
    hands: tuple[tuple[int, ...], ...]
    draw_pile: tuple[int, ...]
    discard: tuple[int, ...]
    current_player: int
    direction: int
    current_sort: str
    current_rank: int
    penalty: int
    sorts_played: tuple[int, ...]
    winner: Optional[int] = None
    game_over: bool = False


@dataclass
class _BayesianClues:
    """Sufficient statistics for a Dirichlet posterior per opponent."""

    suit_counts: Counter
    rank_counts: Counter
    standard_plays: int
    total_plays: int
    estimated_draws: int


class HardcodedStrategyAgent:
    """Deterministic Bayesian tree-search strategy.

    ``belief_samples`` controls posterior determinizations. ``search_depth`` is
    measured in completed player turns. ``beam_width`` bounds plausible moves
    at non-root nodes.
    """

    def __init__(
        self,
        game=None,
        *,
        belief_samples: int = 8,
        search_depth: int = 3,
        beam_width: int = 3,
    ):
        self.game = game
        self.mydeck = deck.Deck([])
        self.type = "bayesian_tree_search"
        self.belief_samples = max(1, belief_samples)
        self.search_depth = max(1, search_depth)
        self.beam_width = max(1, beam_width)
        self._planned_suit: Optional[str] = None
        self.last_search_diagnostics: dict = {}

    def addCard(self, drawn_card) -> None:
        if drawn_card is not None:
            self.mydeck.cards.append(drawn_card)

    def remove(self, played_card) -> None:
        for held_card in self.mydeck.cards:
            if held_card.number == played_card.number:
                self.mydeck.cards.remove(held_card)
                return

    def printCards(self) -> None:
        for held_card in self.mydeck.cards:
            print(held_card.toString())

    def playCard(self, current_sort, current_true_number):
        self._planned_suit = None
        playable = [
            held_card
            for held_card in self.mydeck.cards
            if held_card.compatible(current_sort, current_true_number)
        ]
        if not playable:
            return None

        # Without a live game, retain a deterministic rules-only fallback.
        if self.game is None or self not in getattr(self.game, "players", []):
            return max(playable, key=self._fallback_card_key)

        root_state, root_index = self._root_information_state()
        root_actions = self._legal_actions(root_state)
        if not root_actions:
            return None

        determinizations = self._sample_hidden_states(root_state, root_index)
        action_values = {action: [] for action in root_actions}
        node_count = 0

        for sampled_state in determinizations:
            cache = {}
            for action in root_actions:
                child = self._apply_action(sampled_state, action)
                value, visited = self._search(
                    child,
                    root_index=root_index,
                    depth=self.search_depth - 1,
                    cache=cache,
                )
                action_values[action].append(value)
                node_count += visited

        def posterior_value(action: _Action):
            values = action_values[action]
            mean = sum(values) / len(values)
            # A small lower-tail term prefers robust moves when posterior
            # samples disagree, while still optimizing expected utility.
            ordered = sorted(values)
            lower_quartile = ordered[max(0, len(ordered) // 4 - 1)]
            return (
                mean + 0.08 * lower_quartile,
                self._action_order_score(root_state, action),
                -(action.card_number if action.card_number is not None else 10_000),
                -(
                    STANDARD_SUITS.index(action.chosen_suit)
                    if action.chosen_suit in STANDARD_SUITS
                    else 10_000
                ),
            )

        chosen = max(root_actions, key=posterior_value)
        self._planned_suit = chosen.chosen_suit
        self.last_search_diagnostics = {
            "belief_samples": len(determinizations),
            "search_depth": self.search_depth,
            "beam_width": self.beam_width,
            "nodes": node_count,
            "posterior_action_values": {
                self._action_label(action): round(
                    sum(values) / len(values),
                    4,
                )
                for action, values in action_values.items()
            },
            "chosen": self._action_label(chosen),
        }

        if chosen.card_number is None:
            return None
        return next(
            held_card
            for held_card in self.mydeck.cards
            if held_card.number == chosen.card_number
        )

    def changeSort(self):
        if self._planned_suit in STANDARD_SUITS:
            chosen = self._planned_suit
            self._planned_suit = None
            return chosen
        return self._best_suit(
            tuple(held_card.number for held_card in self.mydeck.cards)
        )

    def _root_information_state(self) -> tuple[_SearchState, int]:
        root_index = self.game.players.index(self)
        hands = []
        for index, player in enumerate(self.game.players):
            if index == root_index:
                hands.append(tuple(sorted(c.number for c in self.mydeck.cards)))
            else:
                # Only the count is public. Posterior samples fill these slots.
                hands.append(tuple([None] * player.mydeck.cardCount()))
        discard = tuple(c.number for c in self.game.game_deck.cards)
        state = _SearchState(
            hands=tuple(hands),
            draw_pile=(),
            discard=discard,
            current_player=self.game.current_player_index,
            direction=self.game.direction,
            current_sort=self.game.current_sort,
            current_rank=self.game.current_true_number,
            penalty=self.game.penalty_amount,
            sorts_played=tuple(int(value) for value in self.game.sorts_played),
            winner=self.game.winner,
            game_over=self.game.game_over,
        )
        return state, root_index

    def _sample_hidden_states(
        self,
        information_state: _SearchState,
        root_index: int,
    ) -> list[_SearchState]:
        remaining = Counter(FULL_DECK)
        for number in information_state.hands[root_index]:
            remaining[number] -= 1
        for number in information_state.discard:
            remaining[number] -= 1
        pool = []
        for number, count in remaining.items():
            if count < 0:
                raise ValueError("public cards are inconsistent with the Pesten deck")
            pool.extend([number] * count)

        hand_sizes = [len(hand) for hand in information_state.hands]
        expected_unknown = sum(
            size for index, size in enumerate(hand_sizes) if index != root_index
        )
        if len(pool) < expected_unknown:
            raise ValueError("public hand sizes exceed the unseen card pool")

        clues = self._bayesian_clues(hand_sizes)
        seed = self._information_seed(information_state, root_index)
        sampled_states = []
        opponent_indices = [
            index for index in range(len(hand_sizes)) if index != root_index
        ]

        for sample_index in range(self.belief_samples):
            rng = random.Random(seed + 1_000_003 * sample_index)
            available = list(pool)
            sampled_hands = [None] * len(hand_sizes)
            sampled_hands[root_index] = information_state.hands[root_index]

            # Rotate allocation order to prevent the sequential posterior
            # sampler from systematically favoring a low-index player.
            offset = sample_index % max(1, len(opponent_indices))
            allocation_order = (
                opponent_indices[offset:] + opponent_indices[:offset]
            )
            for player_index in allocation_order:
                chosen_cards = []
                for _ in range(hand_sizes[player_index]):
                    weights = [
                        self._posterior_card_weight(
                            number,
                            clues[player_index],
                        )
                        for number in available
                    ]
                    chosen_at = self._weighted_index(weights, rng)
                    chosen_cards.append(available.pop(chosen_at))
                sampled_hands[player_index] = tuple(sorted(chosen_cards))

            rng.shuffle(available)
            sampled_states.append(
                _SearchState(
                    hands=tuple(sampled_hands),
                    draw_pile=tuple(available),
                    discard=information_state.discard,
                    current_player=information_state.current_player,
                    direction=information_state.direction,
                    current_sort=information_state.current_sort,
                    current_rank=information_state.current_rank,
                    penalty=information_state.penalty,
                    sorts_played=information_state.sorts_played,
                    winner=information_state.winner,
                    game_over=information_state.game_over,
                )
            )
        return sampled_states

    def _bayesian_clues(self, hand_sizes: list[int]) -> dict[int, _BayesianClues]:
        suit_counts = {index: Counter() for index in range(len(hand_sizes))}
        rank_counts = {index: Counter() for index in range(len(hand_sizes))}
        standard_plays = Counter()
        total_plays = Counter()

        for event in getattr(self.game, "public_history", []):
            player_index = event.get("player")
            if player_index not in suit_counts:
                continue
            if event.get("action") == "play":
                played_sort = event.get("sort")
                played_rank = event.get("rank")
                total_plays[player_index] += 1
                rank_counts[player_index][played_rank] += 1
                if played_sort in STANDARD_SUITS:
                    standard_plays[player_index] += 1
                    suit_counts[player_index][played_sort] += 1
            elif (
                event.get("action") == "choose_sort"
                and event.get("sort") in STANDARD_SUITS
            ):
                # A voluntary suit declaration is more informative than one
                # selected play. Treat it as three virtual observations.
                suit_counts[player_index][event["sort"]] += 3
                standard_plays[player_index] += 3

        clues = {}
        for player_index, current_size in enumerate(hand_sizes):
            plays = total_plays[player_index]
            # Everyone starts with seven. This lower-bound estimate measures
            # how much unseen replenishment has diluted early evidence.
            estimated_draws = max(0, current_size + plays - 7)
            clues[player_index] = _BayesianClues(
                suit_counts=suit_counts[player_index],
                rank_counts=rank_counts[player_index],
                standard_plays=standard_plays[player_index],
                total_plays=plays,
                estimated_draws=estimated_draws,
            )
        return clues

    @staticmethod
    def _posterior_card_weight(number: int, clues: _BayesianClues) -> float:
        """Posterior-predictive likelihood ratio for one unseen card.

        Suits use a symmetric Dirichlet(1,1,1,1) prior. Ranks use a
        Dirichlet prior proportional to their frequency in the 54-card deck.
        Played cards update those priors. The ratio to the prior predictive is
        used as an importance weight over the uniform hidden-deal prior.
        """

        dilution = 1.0 / (1.0 + 0.22 * clues.estimated_draws)
        played_sort = _card_sort(number)
        played_rank = _card_rank(number)

        suit_ratio = 1.0
        if played_sort in STANDARD_SUITS:
            posterior_suit = (
                1.0 + clues.suit_counts.get(played_sort, 0)
            ) / (4.0 + clues.standard_plays)
            suit_ratio = posterior_suit / 0.25

        rank_prior_total = float(sum(FULL_RANK_COUNTS.values()))
        rank_prior = FULL_RANK_COUNTS[played_rank] / rank_prior_total
        posterior_rank = (
            FULL_RANK_COUNTS[played_rank]
            + clues.rank_counts.get(played_rank, 0)
        ) / (rank_prior_total + clues.total_plays)
        rank_ratio = posterior_rank / rank_prior

        # Tempered likelihood avoids overconfidence because played-card choice
        # is evidence of both possession and strategy, not possession alone.
        return max(
            1e-6,
            math.pow(suit_ratio, 0.72 * dilution)
            * math.pow(rank_ratio, 0.28 * dilution),
        )

    def _search(
        self,
        state: _SearchState,
        *,
        root_index: int,
        depth: int,
        cache: dict,
    ) -> tuple[float, int]:
        if depth <= 0 or state.game_over:
            return self._evaluate(state, root_index), 1

        cache_key = (state, root_index, depth)
        if cache_key in cache:
            return cache[cache_key], 0

        actions = self._legal_actions(state)
        if not actions:
            return self._evaluate(state, root_index), 1
        actions = sorted(
            actions,
            key=lambda action: self._action_order_score(state, action),
            reverse=True,
        )[: self.beam_width]

        values = []
        nodes = 1
        for action in actions:
            child = self._apply_action(state, action)
            value, child_nodes = self._search(
                child,
                root_index=root_index,
                depth=depth - 1,
                cache=cache,
            )
            nodes += child_nodes
            values.append(value)

        if state.current_player == root_index:
            result = max(values)
        else:
            # Paranoid multi-player search: each plausible opponent reply is
            # evaluated against the root. This is exact minimax in two-player
            # games and conservative in games with more players.
            result = min(values)
        cache[cache_key] = result
        return result, nodes

    def _legal_actions(self, state: _SearchState) -> list[_Action]:
        if state.game_over:
            return []
        hand = state.hands[state.current_player]
        unique_cards = sorted(set(hand))
        actions = []

        if state.penalty > 0:
            card_numbers = [
                number
                for number in unique_cards
                if _card_rank(number) == state.current_rank
            ]
        else:
            card_numbers = [
                number
                for number in unique_cards
                if _compatible(number, state.current_sort, state.current_rank)
            ]

        for number in card_numbers:
            if _card_rank(number) == 13:
                actions.extend(
                    _Action(number, chosen_suit=suit)
                    for suit in STANDARD_SUITS
                )
            else:
                actions.append(_Action(number))

        # Passing is strategically meaningful when no effective card exists,
        # or when the only card would trigger the special-card finish penalty.
        # Considering voluntary passes everywhere made shallow search value a
        # speculative draw more than the certain reduction of the hand.
        should_offer_pass = not card_numbers or (
            len(hand) == 1
            and card_numbers
            and _card_rank(card_numbers[0]) in SPECIAL_RANKS
        )
        if should_offer_pass:
            if state.penalty > 0 and state.current_rank == 0:
                actions.extend(
                    _Action(None, chosen_suit=suit)
                    for suit in STANDARD_SUITS
                )
            else:
                actions.append(_Action(None))
        return actions

    def _apply_action(
        self,
        state: _SearchState,
        action: _Action,
    ) -> _SearchState:
        hands = [list(hand) for hand in state.hands]
        draw_pile = list(state.draw_pile)
        discard = list(state.discard)
        actor_index = state.current_player
        current_player = state.current_player
        direction = state.direction
        current_sort = state.current_sort
        current_rank = state.current_rank
        penalty = state.penalty
        sorts_played = list(state.sorts_played)
        winner = state.winner
        game_over = state.game_over

        def draw_one(player_index: int):
            nonlocal draw_pile, discard, game_over
            if not draw_pile and len(discard) > 1:
                # The real engine shuffles here. A posterior determinization
                # already randomized this pile; a deterministic recycle keeps
                # the search reproducible.
                draw_pile = list(reversed(discard[:-1]))
                discard = discard[-1:]
            if not draw_pile:
                game_over = True
                return None
            number = draw_pile.pop()
            hands[player_index].append(number)
            return number

        def process_card(number: int, chosen_suit: Optional[str]):
            nonlocal current_player, direction, current_sort, current_rank
            nonlocal penalty, winner, game_over
            is_last = len(hands[current_player]) == 1
            rank = _card_rank(number)
            if is_last and rank in SPECIAL_RANKS:
                draw_one(current_player)
                if not game_over:
                    draw_one(current_player)
                return

            hands[current_player].remove(number)
            discard.append(number)
            current_sort = _card_sort(number)
            current_rank = rank
            sorts_played[card.sorts.index(current_sort)] += 1

            if rank == 7:
                current_player = (
                    current_player - direction
                ) % len(hands)
            elif rank == 8:
                current_player = (
                    current_player + direction
                ) % len(hands)
            elif rank == 1:
                direction *= -1
            elif rank == 0:
                penalty += 5
            elif rank == 13:
                current_sort = (
                    chosen_suit
                    if chosen_suit in STANDARD_SUITS
                    else self._best_suit(tuple(hands[current_player]))
                )
            elif rank == 2:
                penalty += 2

        number = action.card_number
        if penalty > 0:
            if number is None or _card_rank(number) != current_rank:
                for _ in range(penalty):
                    if game_over:
                        break
                    draw_one(current_player)
                penalty = 0
                if current_rank == 0 and not game_over:
                    current_sort = (
                        action.chosen_suit
                        if action.chosen_suit in STANDARD_SUITS
                        else self._best_suit(tuple(hands[current_player]))
                    )
            else:
                process_card(number, action.chosen_suit)
        elif number is None:
            before_draw = tuple(hands[current_player])
            drawn = draw_one(current_player)
            if (
                drawn is not None
                and not game_over
                and _compatible(drawn, current_sort, current_rank)
                and not (
                    len(before_draw) == 1
                    and _card_rank(before_draw[0]) in SPECIAL_RANKS
                )
            ):
                drawn_suit = (
                    self._best_suit(before_draw)
                    if _card_rank(drawn) == 13
                    else None
                )
                process_card(drawn, drawn_suit)
        else:
            process_card(number, action.chosen_suit)

        if not game_over and not hands[actor_index]:
            winner = actor_index
            game_over = True
        if not game_over:
            current_player = (current_player + direction) % len(hands)

        return _SearchState(
            hands=tuple(tuple(sorted(hand)) for hand in hands),
            draw_pile=tuple(draw_pile),
            discard=tuple(discard),
            current_player=current_player,
            direction=direction,
            current_sort=current_sort,
            current_rank=current_rank,
            penalty=penalty,
            sorts_played=tuple(sorts_played),
            winner=winner,
            game_over=game_over,
        )

    def _evaluate(self, state: _SearchState, root_index: int) -> float:
        if state.winner == root_index:
            return 10_000.0
        if state.winner is not None:
            return -10_000.0
        if state.game_over:
            return 0.0

        own_hand = state.hands[root_index]
        opponent_sizes = [
            len(hand)
            for index, hand in enumerate(state.hands)
            if index != root_index
        ]
        score = 34.0 * (
            sum(opponent_sizes) / max(1, len(opponent_sizes))
            - len(own_hand)
        )
        score += 7.0 * min(opponent_sizes, default=0)
        score += self._hand_position_value(own_hand)
        if opponent_sizes:
            score -= sum(
                self._hand_position_value(hand)
                for index, hand in enumerate(state.hands)
                if index != root_index
            ) / len(opponent_sizes)

        current_hand = state.hands[state.current_player]
        current_mobility = sum(
            1
            for number in set(current_hand)
            if (
                _card_rank(number) == state.current_rank
                if state.penalty
                else _compatible(
                    number,
                    state.current_sort,
                    state.current_rank,
                )
            )
        )
        if state.current_player == root_index:
            score += 5.0 + 4.0 * current_mobility
            if state.penalty:
                score -= 4.0 * state.penalty
        else:
            score -= 2.5 * current_mobility
            if state.penalty:
                score += 3.0 * state.penalty
            if len(current_hand) == 1:
                # The persistent Luna benchmark exposed a useful endgame
                # distinction: an opponent holding one *ordinary* compatible
                # card can win immediately, while a compatible special still
                # cannot legally finish. Make that horizon risk explicit
                # instead of hoping a shallow search happens to reach it.
                can_finish_now = any(
                    _card_rank(number) not in SPECIAL_RANKS
                    and (
                        _card_rank(number) == state.current_rank
                        if state.penalty
                        else _compatible(
                            number,
                            state.current_sort,
                            state.current_rank,
                        )
                    )
                    for number in set(current_hand)
                )
                score += 40.0 if not can_finish_now else -180.0

        if len(own_hand) == 1:
            score += (
                45.0
                if _card_rank(own_hand[0]) not in SPECIAL_RANKS
                else -70.0
            )
        score -= 5.0 * sum(
            1 for number in own_hand if _card_rank(number) in SPECIAL_RANKS
        )
        return score

    @staticmethod
    def _hand_position_value(hand: Iterable[int]) -> float:
        hand = tuple(hand)
        if not hand:
            return 100.0
        suit_counts = Counter(
            _card_sort(number)
            for number in hand
            if _card_sort(number) in STANDARD_SUITS
        )
        concentration = max(suit_counts.values(), default=0)
        safe_cards = sum(
            1 for number in hand if _card_rank(number) not in SPECIAL_RANKS
        )
        special_liability = sum(
            1 for number in hand if _card_rank(number) in SPECIAL_RANKS
        )
        endgame_multiplier = max(0, 5 - len(hand))
        return (
            3.5 * concentration
            + 1.5 * safe_cards
            - 2.0 * endgame_multiplier * special_liability
        )

    def _action_order_score(
        self,
        state: _SearchState,
        action: _Action,
    ) -> float:
        hand = state.hands[state.current_player]
        if action.card_number is None:
            if (
                len(hand) == 1
                and _card_rank(hand[0]) in SPECIAL_RANKS
                and state.penalty == 0
            ):
                return 500.0
            return -200.0 - state.penalty

        number = action.card_number
        rank = _card_rank(number)
        remaining = list(hand)
        remaining.remove(number)
        if not remaining:
            return 2_000.0 if rank not in SPECIAL_RANKS else -1_000.0

        score = 100.0
        if len(remaining) == 1:
            score += (
                80.0
                if _card_rank(remaining[0]) not in SPECIAL_RANKS
                else -120.0
            )
        score += {7: 18.0, 8: 16.0, 0: 14.0, 2: 10.0, 13: 12.0}.get(
            rank,
            0.0,
        )
        opponent_at_one = any(
            len(other_hand) == 1
            for player_index, other_hand in enumerate(state.hands)
            if player_index != state.current_player
        )
        if opponent_at_one:
            # Save disruption for the point at which it denies an immediate
            # finish. Jack/Joker suit control and penalty/tempo ranks become
            # substantially more valuable under that public threat.
            score += {
                0: 42.0,
                13: 30.0,
                2: 26.0,
                8: 20.0,
                7: 18.0,
                1: 12.0,
            }.get(rank, 0.0)
        next_sort = (
            action.chosen_suit
            if rank == 13
            else _card_sort(number)
        )
        score += 5.0 * sum(
            1 for held in remaining if _card_sort(held) == next_sort
        )
        return score

    @staticmethod
    def _best_suit(hand: Iterable[int]) -> str:
        hand = tuple(hand)
        counts = Counter(
            _card_sort(number)
            for number in hand
            if _card_sort(number) in STANDARD_SUITS
        )
        safe_counts = Counter(
            _card_sort(number)
            for number in hand
            if _card_sort(number) in STANDARD_SUITS
            and _card_rank(number) not in SPECIAL_RANKS
        )
        return max(
            STANDARD_SUITS,
            key=lambda suit: (
                6 * counts[suit] + 3 * safe_counts[suit],
                counts[suit],
                -STANDARD_SUITS.index(suit),
            ),
        )

    @staticmethod
    def _weighted_index(weights: list[float], rng: random.Random) -> int:
        total = sum(weights)
        threshold = rng.random() * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if cumulative >= threshold:
                return index
        return len(weights) - 1

    def _information_seed(
        self,
        state: _SearchState,
        root_index: int,
    ) -> int:
        public_history = getattr(self.game, "public_history", [])
        payload = {
            "root": root_index,
            "hand": state.hands[root_index],
            "sizes": [len(hand) for hand in state.hands],
            "discard": state.discard,
            "player": state.current_player,
            "direction": state.direction,
            "sort": state.current_sort,
            "rank": state.current_rank,
            "penalty": state.penalty,
            "history": public_history,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).digest()
        return int.from_bytes(digest[:8], "big")

    @staticmethod
    def _action_label(action: _Action) -> str:
        card_label = (
            "pass"
            if action.card_number is None
            else str(action.card_number)
        )
        return (
            f"{card_label}->{action.chosen_suit}"
            if action.chosen_suit
            else card_label
        )

    @staticmethod
    def _fallback_card_key(held_card):
        rank = held_card.truenumber
        return (
            rank not in SPECIAL_RANKS,
            {7: 6, 8: 5, 0: 4, 2: 3, 13: 2, 1: 1}.get(rank, 0),
            -held_card.number,
        )


Agent = HardcodedStrategyAgent
