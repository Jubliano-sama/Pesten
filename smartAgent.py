import card
import deck
import logging

class Agent:
    # The smart agent’s default parameters.
    default_params = {
        'stop_bonus': -1.9467897139799917,         # Bonus for stopping a chain in a 2-player game if an 8 is in hand.
        'chain_bonus': 1.3119838792608858,            # Base bonus for playing a 7 in a chain.
        'hand_penalty': 1.0290830868190015,          # Penalty per card remaining in hand.
        'suit_common_weight': 0.43244878288347055,     # Weight for the total count of cards in a suit when choosing a sort.
        'suit_penalty_weight': -1.8463831782446556,     # Weight for the number of penalty cards in a suit when choosing a sort.
        'penalty_card_bonus': {                       # Bonus for playing a particular penalty card (if not a 7).
            8: -1.5328572573442552,
            1: 0.04122020478114498,
            0: -0.03556864491834867,
            13: -3.0043124890687474,
            2: 0.04665718005466267
        },
        'softmax_temperature': 1.0,  # (No longer used.)
        'end_penalty_bonus': 1.0     # Extra bonus per penalty card when computing a 7’s "end value."
    }
    
    def __init__(self, game=None):
        self.mydeck = deck.Deck([])
        self.type = "Smart"
        self.game = game
        # Load parameters from the default_params dictionary.
        self.stop_bonus = self.default_params.get('stop_bonus', 10.0)
        self.chain_bonus = self.default_params.get('chain_bonus', 1.0)
        self.hand_penalty = self.default_params.get('hand_penalty', 0.1)
        self.suit_common_weight = self.default_params.get('suit_common_weight', 1.0)
        self.suit_penalty_weight = self.default_params.get('suit_penalty_weight', 1.0)
        self.penalty_card_bonus = self.default_params.get('penalty_card_bonus', {8:5.0, 1:3.0, 0:4.0, 13:2.0, 2:2.0})
        self.softmax_temperature = self.default_params.get('softmax_temperature', 1.0)  # Unused now.
        self.end_penalty_bonus = self.default_params.get('end_penalty_bonus', 1.0)
    
    def changeSort(self):
        """
        Choose the next suit by scoring each suit as:
            score = suit_common_weight * (total cards in suit)
                    - suit_penalty_weight * (number of penalty cards in suit)
        """
        suits = card.sorts[:4]
        penalty_set = {7, 8, 1, 0, 13, 2}
        best_suit = None
        best_score = -float('inf')
        for s in suits:
            total_count = sum(1 for c in self.mydeck.cards if c.sort == s)
            penalty_count = sum(1 for c in self.mydeck.cards if c.sort == s and c.truenumber in penalty_set)
            score = self.suit_common_weight * total_count - self.suit_penalty_weight * penalty_count
            logging.debug(f"Suit {s}: total {total_count}, penalty {penalty_count}, score {score}")
            if score > best_score:
                best_score = score
                best_suit = s
        if best_suit is None:
            best_suit = suits[0]
        logging.debug(f"SmartAgent changes sort to: {best_suit} (score: {best_score})")
        return best_suit

    def playCard(self, current_sort, current_true_number):
        """
        Decide which card to play using a deterministic policy.

        1. If a penalty is active (i.e. current_true_number is in the penalty set),
           consider only candidate moves matching that penalty.
        2. For each candidate move, simulate the outcome:
             - If the candidate is a 7, use the chain simulation which now groups all playable 7’s so that
               their "end values" are compared.
             - Otherwise, compute the hand value via _hand_value() and add any bonus from penalty_card_bonus.
        3. Select the candidate with the highest score.
        """
        playable = [c for c in self.mydeck.cards if c.compatible(current_sort, current_true_number)]
        if not playable:
            return None

        penalty_set = {7, 8, 1, 0, 13, 2}

        # If in penalty state, restrict candidates to those matching the penalty.
        if current_true_number in penalty_set and current_true_number != 7:
            matching = [c for c in playable if c.truenumber == current_true_number]
            if matching:
                candidate_list = []
                candidate_scores = []
                for candidate in matching:
                    new_hand = self.mydeck.cards.copy()
                    try:
                        new_hand.remove(candidate)
                    except ValueError:
                        continue
                    if candidate.truenumber == 7:
                        # Group all playable 7's to compare their end values.
                        candidate_sevens = [c for c in matching if c.truenumber == 7]
                        candidate_end_values = {}
                        for c in candidate_sevens:
                            hand_after_removal = self.mydeck.cards.copy()
                            hand_after_removal.remove(c)
                            candidate_end_values[c] = self._end_value(c, hand_after_removal)
                        max_end = max(candidate_end_values.values())
                        candidate_score = self.chain_bonus + (max_end - candidate_end_values[candidate])
                    else:
                        candidate_score = self._hand_value(new_hand)
                        if candidate.truenumber in self.penalty_card_bonus:
                            candidate_score += self.penalty_card_bonus[candidate.truenumber]
                    candidate_list.append(candidate)
                    candidate_scores.append(candidate_score)
                max_index = candidate_scores.index(max(candidate_scores))
                chosen_candidate = candidate_list[max_index]
                logging.debug(f"SmartAgent (penalty state) selects {chosen_candidate.toString()} deterministically.")
                return chosen_candidate

        # Otherwise, evaluate all playable moves.
        candidate_list = []
        candidate_scores = []

        # Process candidate 7's collectively so that their end values are compared.
        candidate_sevens = [c for c in playable if c.truenumber == 7]
        if candidate_sevens:
            candidate_end_values = {}
            for candidate in candidate_sevens:
                new_hand = self.mydeck.cards.copy()
                new_hand.remove(candidate)
                candidate_end_values[candidate] = self._end_value(candidate, new_hand)
            max_end = max(candidate_end_values.values())
            for candidate in candidate_sevens:
                score = self.chain_bonus * len(candidate_sevens) * (max_end - candidate_end_values[candidate])
                candidate_list.append(candidate)
                candidate_scores.append(score)
                logging.debug(f"Candidate {candidate.toString()} chain score: {score:.2f}")

        # Process all non-7 candidates as before.
        for candidate in [c for c in playable if c.truenumber != 7]:
            new_hand = self.mydeck.cards.copy()
            new_hand.remove(candidate)
            candidate_score = self._hand_value(new_hand)
            if candidate.truenumber in self.penalty_card_bonus:
                candidate_score += self.penalty_card_bonus[candidate.truenumber]
            candidate_list.append(candidate)
            candidate_scores.append(candidate_score)
            logging.debug(f"Candidate {candidate.toString()} score: {candidate_score:.2f}")

        max_index = candidate_scores.index(max(candidate_scores))
        chosen_candidate = candidate_list[max_index]
        logging.debug(f"SmartAgent selects {chosen_candidate.toString()} deterministically (highest score).")
        return chosen_candidate

    def _end_value(self, card, hand):
        """
        Compute the "end value" for a candidate 7.
        Defined as:
            end_value = suit_common_weight * (number of cards in hand with same suit)
                        + end_penalty_bonus * (number of penalty cards in hand with same suit)
        """
        suit = card.sort
        total_count = sum(1 for c in hand if c.sort == suit)
        penalty_set = {7, 8, 1, 0, 13, 2}
        penalty_count = sum(1 for c in hand if c.sort == suit and c.truenumber in penalty_set)
        return self.suit_common_weight * total_count + self.end_penalty_bonus * penalty_count

    def _simulate_chain(self, available_sevens, hand):
        """
        Evaluate the bonus from playing a 7 in a chain.
        For the available 7(s) (typically a singleton), compute their end values.
        Let max_end be the highest end value among them. Then for each candidate 7,
        the bonus for playing it now is:
            bonus = chain_bonus + (max_end - end_value(candidate))
        Return the maximum bonus among candidates multiplied by the number of available sevens.
        (Note: In the original code this was used for a singleton 7, but if there are multiple 7's
         it is better to group them so that the differences in end values are taken into account.)
        """
        if not available_sevens:
            return self._stop_score(hand)
        end_values = [self._end_value(c, hand) for c in available_sevens]
        max_end = max(end_values)
        chain_values = [self.chain_bonus + (max_end - ev) for ev in end_values]
        bonus = max(chain_values)
        return bonus * len(available_sevens)

    def _stop_score(self, hand):
        """
        Compute a bonus for stopping the 7-chain.
         - In a 2-player game, if an 8 is in hand, return stop_bonus.
         - Otherwise, return the maximum count among the standard suits.
        """
        if self.game and self.game.num_players == 2:
            if any(c.truenumber == 8 for c in hand):
                return self.stop_bonus
        suits = card.sorts[:4]
        suit_counts = {s: 0 for s in suits}
        for c in hand:
            if c.sort in suits:
                suit_counts[c.sort] += 1
        return max(suit_counts.values()) if suit_counts else 0

    def _hand_value(self, hand):
        """
        Evaluate the quality of a hand.
        Computed as: stop_score minus a penalty proportional to the hand size.
        """
        base_value = self._stop_score(hand)
        penalty = self.hand_penalty * len(hand)
        return base_value - penalty

    def addCard(self, _card):
        if _card is None:
            logging.error("None card detected")
        else:
            self.mydeck.cards.append(_card)

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                return
        logging.error("Card not found in hand")


class SmarterAgent(Agent):
    """Measured upgrade of :class:`Agent` with public-state strategy rules.

    The original implementation accidentally valued cards that remained in the
    hand, removed every equal duplicate while evaluating a single play, ignored
    active-penalty stacking, and chose a Jack's suit from public discard counts.
    This policy keeps the tuned Smart scorer for multiplayer and penalty
    decisions.  In ordinary two-player turns it uses a compact rank-pressure
    rule distilled from the frozen dashboard teacher, then retains the
    independently useful endgame rules.  No model or hidden state is needed at
    runtime, and every input is visible to an ordinary player.
    """

    special_ranks = frozenset({0, 1, 2, 7, 8, 13})
    wild_ranks = frozenset({0, 13})
    # When preserving the only ordinary finisher, spend the most disruptive
    # special first. Immediate 7/8 finishes are handled separately below.
    preservation_order = {0: 0, 2: 1, 8: 2, 7: 3, 13: 4, 1: 5}
    # Log-odds derived from the dashboard teacher's deterministic two-player
    # play rates.  The first table applies normally; the second applies only
    # when the opponent is literally one card from winning.  A small remaining
    # suit-concentration term breaks strategically useful rank ties.
    two_player_rank_preference = (
        -2.325, 0.445, 0.075, -0.391, -0.301, -0.388, -0.359,
        0.130, -0.008, -0.091, -0.266, -0.153, -0.135, -3.166,
    )
    two_player_last_card_preference = (
        0.366, -0.015, 2.105, -0.756, -0.640, -0.770, -0.514,
        0.625, 0.442, -0.619, -0.886, -0.299, -0.336, -0.857,
    )
    rank_preference_weight = 6.0

    def __init__(self, game=None):
        super().__init__(game)
        self.type = "Smarter"

    def _penalty_active(self):
        return bool(self.game and getattr(self.game, "penalty_amount", 0) > 0)

    def _closest_opponent_hand(self):
        if not self.game or self not in getattr(self.game, "players", ()):
            return float("inf")
        return min(
            player.mydeck.cardCount()
            for player in self.game.players
            if player is not self
        )

    def _remaining_after(self, candidate):
        """Return the real one-card transition, including duplicate decks."""
        remaining = self.mydeck.cards.copy()
        remaining.remove(candidate)
        return remaining

    def _smart_choice(self, playable):
        """Apply the tuned Smart scorer to an already legal candidate set."""
        candidates = []
        scores = []
        sevens = [held for held in playable if held.truenumber == 7]
        if sevens:
            end_values = {
                held: self._end_value(held, self._remaining_after(held))
                for held in sevens
            }
            max_end = max(end_values.values())
            for held in sevens:
                candidates.append(held)
                scores.append(
                    self.chain_bonus
                    * len(sevens)
                    * (max_end - end_values[held])
                )
        for held in (item for item in playable if item.truenumber != 7):
            score = self._hand_value(self._remaining_after(held))
            score += self.penalty_card_bonus.get(held.truenumber, 0.0)
            candidates.append(held)
            scores.append(score)
        return candidates[scores.index(max(scores))]

    def _two_player_choice(self, playable, closest_opponent):
        """Choose by public rank pressure and the resulting suit concentration."""
        preferences = (
            self.two_player_last_card_preference
            if closest_opponent <= 1
            else self.two_player_rank_preference
        )
        best_card = playable[0]
        best_score = -float("inf")
        standard_suits = card.sorts[:4]
        for held in playable:
            remaining = self._remaining_after(held)
            concentration = max(
                sum(candidate.sort == suit for candidate in remaining)
                for suit in standard_suits
            )
            score = (
                concentration
                + self.rank_preference_weight
                * preferences[held.truenumber]
            )
            if score > best_score:
                best_score = score
                best_card = held
        return best_card

    def _preservation_card(self, specials, ordinary_finisher):
        tempo_ranks = {7}
        if self.game and self.game.num_players == 2:
            # In a two-player game an 8 skips the opponent and retains the turn.
            tempo_ranks.add(8)
        immediate = [
            held
            for held in specials
            if held.truenumber in tempo_ranks
            and ordinary_finisher.compatible(held.sort, held.truenumber)
        ]
        if immediate:
            return immediate[0]
        return min(
            specials,
            key=lambda held: self.preservation_order[held.truenumber],
        )

    def playCard(self, current_sort, current_true_number):
        playable = [
            held
            for held in self.mydeck.cards
            if held.compatible(current_sort, current_true_number)
        ]
        penalty_active = self._penalty_active()
        if penalty_active:
            # Pesten only permits stacking the same penalty rank. Returning a
            # different compatible wild card merely accepts the pending draw.
            playable = [
                held
                for held in playable
                if held.truenumber == current_true_number
            ]
        if not playable:
            return None

        hand_size = self.mydeck.cardCount()
        if (
            hand_size == 1
            and self.mydeck.cards[0].truenumber in self.special_ranks
            and not penalty_active
        ):
            # A special cannot finish the game and playing it draws two cards.
            # Passing first instead searches for an ordinary finishing card.
            return None

        closest_opponent = self._closest_opponent_hand()
        playable_ranks = {held.truenumber for held in playable}
        if not penalty_active and closest_opponent > 2:
            if hand_size >= 5 and playable_ranks <= self.wild_ranks:
                # The analyzed GRU teacher likewise saves Jack/Joker early and
                # uses them much more often once a rival is close.
                return None
            if (
                hand_size == 2
                and all(
                    held.truenumber in self.special_ranks
                    for held in self.mydeck.cards
                )
            ):
                # Fish for an ordinary finisher instead of guaranteeing that
                # the next remaining card is illegal as a final play.
                return None

        if (
            self.game
            and self.game.num_players == 2
            and not penalty_active
        ):
            chosen = self._two_player_choice(playable, closest_opponent)
        else:
            candidates = playable
            if closest_opponent <= 2:
                jokers = [held for held in playable if held.truenumber == 0]
                if jokers:
                    candidates = jokers
            chosen = self._smart_choice(candidates)

        ordinary = [
            held
            for held in self.mydeck.cards
            if held.truenumber not in self.special_ranks
        ]
        if (
            len(ordinary) == 1
            and hand_size <= 5
            and chosen.truenumber not in self.special_ranks
        ):
            specials = [
                held
                for held in playable
                if held.truenumber in self.special_ranks
            ]
            if specials:
                chosen = self._preservation_card(specials, ordinary[0])
        return chosen

    def changeSort(self):
        """Choose a suit actually represented in the remaining hand."""
        suits = card.sorts[:4]
        counts = [
            sum(held.sort == suit for held in self.mydeck.cards)
            for suit in suits
        ]
        return suits[counts.index(max(counts))]
