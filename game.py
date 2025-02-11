import logging
import numpy as np
from typing import List, Optional, Tuple, Type

import deck
import card

# Configure logging to only show critical errors by default.
logging.basicConfig(
    filename='gamepest.log',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.CRITICAL
)


class Game:
    """
    Represents a simulation of the card game.
    """

    def __init__(self, agent_classes: List[Type]) -> None:
        """
        Initialize the game with a set of agent classes.
        Each agent class is instantiated. If an agent's __init__ expects a game instance,
        it is passed 'self'; otherwise, the agent is instantiated without arguments.
        """
        self.players: List = []
        for agent_class in agent_classes:
            try:
                player = agent_class(self)
            except TypeError:
                player = agent_class()
            self.players.append(player)
        self.num_players: int = len(self.players)
        self.current_player_index: int = 0
        self.turn_count: int = 0
        self.direction: int = 1  # 1 = clockwise, -1 = anti-clockwise
        self.cards_played: int = 0

        # Game state variables
        self.current_sort: Optional[str] = None
        self.current_true_number: Optional[int] = None
        self.penalty_amount: int = 0
        self.sorts_played: np.ndarray = np.zeros(5, dtype=int)

        self.game_over: bool = False
        self.winner: Optional[int] = None

        # Decks used in the game
        self.grab_deck: Optional[deck.Deck] = None
        self.game_deck: Optional[deck.Deck] = None

    def reset(self) -> None:
        """
        Reset game state and initialize decks and player hands.
        """
        self.current_player_index = 0
        self.turn_count = 0
        self.direction = 1
        self.penalty_amount = 0
        self.current_sort = None
        self.current_true_number = None
        self.game_over = False
        self.winner = None
        self.sorts_played = np.zeros(5, dtype=int)

        if len(self.players) < 2:
            logging.critical("Cannot start game with less than two players")
            self.game_over = True
            return

        # Create and shuffle the grab deck
        self.grab_deck = deck.standardDeck()
        self.grab_deck.shuffle()

        # Deal 7 cards to each player
        for player in self.players:
            for _ in range(7):
                card_drawn = self.grab_deck.topCard()
                player.addCard(card_drawn)
                self.grab_deck.removeTopCard()

        # Initialize the game deck with a valid starting card
        self.game_deck = deck.Deck([self.grab_deck.topCard()])
        self.grab_deck.removeTopCard()
        top_card = self.game_deck.topCard()
        while top_card.truenumber in (13, 0, 1, 7, 8, 2):
            # If the starting card is invalid, add another card
            self.game_deck.cards.append(self.grab_deck.topCard())
            self.grab_deck.removeTopCard()
            top_card = self.game_deck.topCard()

        self.current_sort = top_card.sort
        self.current_true_number = top_card.truenumber

    def simulate_turn(self) -> None:
        """
        Simulate one turn of the game.
        """
        if self.game_over:
            return

        current_player = self.players[self.current_player_index]
        is_last_card = current_player.mydeck.cardCount() == 1

        if self.penalty_amount > 0:
            self._handle_penalty_turn(current_player, is_last_card)
        else:
            self._handle_normal_turn(current_player, is_last_card)

        self.turn_count += 1

        # Check win condition: a player wins by emptying their hand.
        if current_player.mydeck.cardCount() == 0:
            self.winner = self.current_player_index
            self.game_over = True
            logging.info(f"Player {self.winner} has won!")
            return

        # Update turn order if game is still running.
        if not self.game_over:
            self.current_player_index = self.calculate_next_player(self.current_player_index, self.direction)

    def _handle_penalty_turn(self, player, is_last_card: bool) -> None:
        """
        Handle a turn when a penalty is in effect.
        """
        played_card = player.playCard(self.current_sort, self.current_true_number)
        if played_card is not None:
            if played_card.truenumber != self.current_true_number:
                self._apply_penalty(player)
                if self.current_true_number == 0 and not self.game_over:
                    self.change_sort(player.changeSort())
            else:
                self._process_played_card(played_card, player, is_last_card)
        else:
            self._apply_penalty(player)
            if self.current_true_number == 0 and not self.game_over:
                self.change_sort(player.changeSort())

    def _handle_normal_turn(self, player, is_last_card: bool) -> None:
        """
        Handle a turn with no penalty in effect.
        """
        played_card = player.playCard(self.current_sort, self.current_true_number)
        self._process_played_card(played_card, player, is_last_card)

    def _process_played_card(self, played_card: Optional[card.Card], player, is_last_card: bool) -> None:
        """
        Process a played card: update counters, validate the play, update game state,
        and apply special effects.
        """
        self.cards_played += 1
        if self.cards_played > 500:
            self.game_over = True
            logging.info("Game ended due to too many cards played.")
            return

        if played_card is None:
            self._handle_no_card_play(player, is_last_card)
            return

        if not played_card.compatible(self.current_sort, self.current_true_number):
            logging.warning("Played card is not compatible with the current state.")
            return

        if is_last_card and self._is_penalty_card(played_card):
            self._handle_last_card_penalty(player)
            return

        self._update_state_with_card(played_card, player)
        self._apply_card_effects(played_card, player)

    def _apply_penalty(self, player) -> None:
        """
        Force the player to draw the required number of penalty cards.
        """
        logging.debug(f"Applying penalty: Player draws {self.penalty_amount} card(s).")
        for _ in range(self.penalty_amount):
            if self.game_over:
                return
            self.grab_card(player)
        self.penalty_amount = 0

    def grab_card(self, player) -> Optional[card.Card]:
        """
        Draw a card for the player from the grab deck. If the grab deck is empty,
        reshuffle the game deck (except the top card) into the grab deck.
        """
        drawn_card = None
        if self.grab_deck and self.grab_deck.cardCount() > 0:
            drawn_card = self.grab_deck.topCard()
            self.grab_deck.removeTopCard()
            logging.debug(f"Grabbed card: {drawn_card.toString()}")
            player.addCard(drawn_card)
        elif self.game_deck and self.game_deck.cardCount() > 1:
            logging.debug("Reshuffling game deck into grab deck")
            self.grab_deck.cards = self.game_deck.cards[:-1]
            self.grab_deck.shuffle()
            self.game_deck.cards = self.game_deck.cards[-1:]
            drawn_card = self.grab_deck.topCard()
            logging.debug(f"Grabbed card: {drawn_card.toString()}")
            self.grab_deck.removeTopCard()
            player.addCard(drawn_card)
        else:
            logging.debug("No cards left to grab; ending game.")
            self.game_over = True
        return drawn_card

    def change_sort(self, new_sort: str) -> None:
        """
        Change the current sort for the game.
        """
        self.current_sort = new_sort
        logging.debug(f"Sort changed to: {self.current_sort}")

    def _handle_no_card_play(self, player, is_last_card: bool) -> None:
        """
        If no card was played, force the player to grab a card and, if it's playable,
        allow the player to immediately play it.
        """
        logging.debug("No card played; forcing a card grab.")
        grabbed_card = self.grab_card(player)
        if self.game_over:
            return
        if grabbed_card and grabbed_card.compatible(self.current_sort, self.current_true_number):
            potential_play = player.playCard(self.current_sort, self.current_true_number)
            if potential_play is not None and potential_play == grabbed_card:
                logging.debug("Player plays the grabbed card.")
                self._process_played_card(grabbed_card, player, False)

    def _is_penalty_card(self, played_card: card.Card) -> bool:
        """
        Determine whether the played card triggers extra penalty when
        played as the last card.
        """
        return played_card.truenumber in (7, 8, 1, 0, 13, 2)

    def _handle_last_card_penalty(self, player) -> None:
        """
        If a penalty card is played as the player's last card, force the player to draw extra cards.
        """
        logging.debug("Last card penalty triggered. Player draws two cards.")
        self.grab_card(player)
        if not self.game_over:
            self.grab_card(player)

    def _update_state_with_card(self, played_card: card.Card, player) -> None:
        """
        Update the game state with the played card: append it to the game deck,
        update current sort/true number, record card usage, and remove the card from the player's hand.
        """
        logging.debug(f"Card played: {played_card.toString()}")
        self.game_deck.cards.append(played_card)
        self.current_sort = played_card.sort
        sort_index = card.sorts.index(played_card.sort)
        self.sorts_played[sort_index] += 1
        self.current_true_number = played_card.truenumber
        player.remove(played_card)

    def _apply_card_effects(self, played_card: card.Card, player) -> None:
        """
        Apply special effects based on the played card's true number.
        """
        if played_card.truenumber == 7:
            # Player gets another turn.
            self.current_player_index = (self.current_player_index - self.direction) % self.num_players
        elif played_card.truenumber == 8:
            # Skip the next player.
            self.current_player_index = (self.current_player_index + self.direction) % self.num_players
        elif played_card.truenumber == 1:
            self.direction *= -1
        elif played_card.truenumber == 0:
            self.penalty_amount += 5
        elif played_card.truenumber == 13:
            self.change_sort(player.changeSort())
        elif played_card.truenumber == 2:
            self.penalty_amount += 2

    def calculate_next_player(self, current_index: int, direction: int) -> int:
        """
        Calculate the next player's index using modulo arithmetic.
        """
        return (current_index + direction) % self.num_players

    def auto_simulate(self, max_turns: int = 1000) -> Tuple[int, Optional[int]]:
        """
        Automatically simulate the game until it ends or a maximum number of turns is reached.
        Returns:
            A tuple (turn_count, winner) where winner is None if the game ended in a draw.
        """
        self.reset()
        while not self.game_over and self.turn_count < max_turns:
            logging.debug(f"Turn: {self.turn_count}, Player: {self.current_player_index}")
            self.simulate_turn()
        if self.winner is None:
            logging.debug("Game ended in a draw!")
        return self.turn_count, self.winner
