import card
import deck
import logging
import math
import numpy as np

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


class SmarterAgent:
    def __init__(self, game=None):
        self.mydeck = deck.Deck([])  # Initialize empty deck for the agent's hand
        self.game = game             # Store game instance for state access
        self.type = "Smarter"        # Identifier for debugging/logging

        # Scoring parameters (tunable via benchmarking)
        self.hand_penalty = 1.0       # Penalty per card in hand
        self.suit_bonus = 0.5         # Bonus for cards in the most common suit
        self.special_bonus = {        # Bonuses for keeping special cards in hand
            7: 2.0, 8: 1.5, 1: 1.0, 0: 1.5, 13: 1.0, 2: 1.0
        }
        self.skip_bonus = 5.0         # Base bonus for skipping a player (8)
        self.reverse_bonus = 0.5      # Bonus for reversing direction (1)
        self.penalty_bonus = 0.2      # Bonus per penalty point added (0, 2)
        self.suit_change_bonus = 1.0  # Base bonus for changing suit (13)
        self.suit_depletion_bonus = 0.1  # Bonus per depleted suit count
        self.seven_bonus = 2.0        # Bonus for extra turn (7)

    def addCard(self, _card):
        """Add a card to the agent's hand."""
        if _card is None:
            print("Warning: Attempted to add None card")
        else:
            self.mydeck.cards.append(_card)

    def remove(self, _card):
        """Remove a specific card from the agent's hand."""
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                return
        print("Error: Card not found in hand")

    def card_value(self, c, max_suit):
        """Compute the value of a card based on its properties."""
        value = 0
        if c.sort == max_suit:
            value += self.suit_bonus  # Bonus if it matches the dominant suit
        if c.truenumber in self.special_bonus:
            value += self.special_bonus[c.truenumber]  # Bonus for special cards
        return value

    def hand_value(self, hand):
        """Evaluate the hand’s value: penalize size, reward special cards and suit concentration."""
        if not hand:
            return 0  # Empty hand is a win, but we score before that happens
        suit_counts = {s: 0 for s in card.sorts[:4]}
        for c in hand:
            if c.sort in suit_counts:
                suit_counts[c.sort] += 1
        max_suit = max(suit_counts, key=suit_counts.get) if suit_counts else None
        return sum(self.card_value(c, max_suit) for c in hand) - self.hand_penalty * len(hand)

    def score_candidate(self, c, new_hand):
        """Score a playable card based on its effect and resulting hand value."""
        base_score = self.hand_value(new_hand)
        if not self.game:  # Fallback if game instance isn’t available
            return base_score + (self.special_bonus.get(c.truenumber, 0) * 0.5)
        
        if c.truenumber == 7:
            # Extra turn bonus
            return base_score + self.seven_bonus
        elif c.truenumber == 8:
            # Skip next player; more valuable if they have few cards
            try:
                next_idx = self.game.calculate_next_player(self.game.players.index(self), self.game.direction)
                next_count = self.game.players[next_idx].mydeck.cardCount()
                return base_score + self.skip_bonus / (next_count + 1)
            except (ValueError, AttributeError):
                return base_score + self.skip_bonus / 2  # Fallback
        elif c.truenumber == 1:
            # Reverse direction; small fixed bonus
            return base_score + self.reverse_bonus
        elif c.truenumber == 0:
            # Add 5 to penalty
            return base_score + self.penalty_bonus * 5
        elif c.truenumber == 2:
            # Add 2 to penalty
            return base_score + self.penalty_bonus * 2
        elif c.truenumber == 13:
            # Choose most depleted suit
            best_suit_idx = np.argmax(self.game.sorts_played[:4])
            depletion_value = self.game.sorts_played[best_suit_idx]
            return base_score + self.suit_change_bonus + depletion_value * self.suit_depletion_bonus
        else:
            # Normal card: favor depleted suits
            suit_idx = card.sorts.index(c.sort)
            return base_score + self.game.sorts_played[suit_idx] * self.suit_depletion_bonus

    def playCard(self, current_sort, current_true_number):
        """Choose the best playable card."""
        playable = [c for c in self.mydeck.cards if c.compatible(current_sort, current_true_number)]
        if not playable:
            return None
        scores = []
        for c in playable:
            new_hand = [card for card in self.mydeck.cards if card != c]
            score = self.score_candidate(c, new_hand)
            scores.append(score)
        return playable[np.argmax(scores)]

    def changeSort(self):
        """Choose the most depleted suit when playing a 13."""
        if not self.game or not hasattr(self.game, 'sorts_played'):
            return card.sorts[0]  # Fallback to first suit
        suits = card.sorts[:4]
        best_suit_idx = np.argmax(self.game.sorts_played[:4])
        return suits[best_suit_idx]