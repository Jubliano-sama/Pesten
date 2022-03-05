import deck
import card
import logging
import randomAgent, neuralAgent
import numpy
from random import uniform, randint

logging.basicConfig(filename='gamepest.log', format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.CRITICAL)


class Game:
    def __init__(self, playerAmount):
        self.players = []
        self.currentPlayer = None
        self.canPlayGrabbedCard = False
        self.grabDeck = None
        self.gameDeck = None
        self.currentPlayerIndex = 0
        self.penaltyAmount = 0
        self.currentSort = None
        self.changeSortTurn = False
        self.currentTrueNumber = None
        self.amountOfPlayers = playerAmount
        self.winner = None
        self.gameEnded = False
        self.turn = 0
        self.direction = 1  # 1=clockwise -1=anti-clockwise
        self.sortsPlayed = numpy.zeros(shape=[5])

    def simTurn(self, _card):

        lastCard = False
        if self.currentPlayer.mydeck.cardCount() == 1:
            lastCard = True
        # print(turn)
        # checks if player has to grab penalty cards
        if self.penaltyAmount > 0:
            if _card is not None:
                # if the card the player wants to play is the same as the penalty card, play the card, else grab
                # cards
                if _card.truenumber != self.currentTrueNumber:
                    self.grabMultipleCards(self.penaltyAmount, self.currentPlayer)
                    # check if there werent enough cards to grab, ending the game in a draw
                    self.penaltyAmount = 0
                    if self.currentTrueNumber == 0 and self.gameEnded == False:
                        self.changeSort(self.currentPlayer.changeSort())
                else:
                    self.throwCard(_card, self.currentPlayer, lastCard)
            else:
                self.grabMultipleCards(self.penaltyAmount, self.currentPlayer)
                self.penaltyAmount = 0
                if self.currentTrueNumber == 0:
                    self.changeSort(self.currentPlayer.changeSort())
                # check if there werent enough cards to grab, ending the game in a draw
                if self.gameEnded == True:
                    logging.info("Game ended in a draw!")
        else:
            self.throwCard(_card, self.currentPlayer, lastCard)
        self.turn += 1
        if self.currentPlayer.mydeck.cardCount() == 0:
            self.winner = self.currentPlayerIndex
            self.gameEnded = True
            logging.info("Player: " + str(self.winner) + " has won!")

    def reset(self):
        self.currentPlayer = None
        self.canPlayGrabbedCard = False
        self.currentPlayerIndex = 0
        self.penaltyAmount = 0
        self.currentSort = None
        self.changeSortTurn = False
        self.currentTrueNumber = None
        self.winner = None
        self.gameEnded = False
        self.turn = 0
        self.direction = 1  # 1=clockwise -1=anti-clockwise
        self.sortsPlayed = numpy.zeros(shape=[5])

        if len(self.players) < 2:
            logging.critical("Cannot start game without or with only one player(s)")
            return
        self.grabDeck = deck.standardDeck()
        self.grabDeck.shuffle()
        # add shuffled cards to player decks
        for player in self.players:
            for i in range(7):
                player.addCard(self.grabDeck.topCard())
                self.grabDeck.removeTopCard()
        self.gameDeck = deck.Deck([self.grabDeck.cards[-1]])
        self.grabDeck.removeTopCard()
        # generate top card
        while self.gameDeck.cards[-1].truenumber == 13 or self.gameDeck.cards[-1].truenumber == 0 or \
                self.gameDeck.cards[-1].truenumber == 1 or self.gameDeck.cards[-1].truenumber == 7 \
                or self.gameDeck.cards[-1].truenumber == 8 or self.gameDeck.cards[-1].truenumber == 2:
            self.gameDeck.cards.append(self.grabDeck.cards[-1])
            self.grabDeck.removeTopCard()
        turn = 0
        self.currentSort = self.gameDeck.topCard().sort
        self.currentTrueNumber = self.gameDeck.topCard().truenumber

    def auto_sim(self):
        self.reset()
        while not self.gameEnded:
            logging.debug("Player " + str(self.currentPlayerIndex))
            self.currentPlayer = self.players[self.currentPlayerIndex]
            self.simTurn(self.currentPlayer.playCard(self.currentSort, self.currentTrueNumber))
            self.currentPlayerIndex = self.calculateNextPlayer(self.currentPlayerIndex, self.direction)
        if self.winner is None:
            logging.debug("Game ended in a draw!")
        return [self.turn, self.winner]

    def grabMultipleCards(self, amount, player):
        logging.debug("grabbing " + str(amount) + " cards")
        for i in range(amount):
            if self.gameEnded is not True:
                self.grabCard(player)
            else:
                return

    def grabCard(self, player):
        _card = None
        if self.grabDeck.cardCount() > 0:
            _card = self.grabDeck.topCard()
            self.grabDeck.removeTopCard()
            logging.debug("grabbed card:"  + _card.toString())
            player.addCard(_card)
        elif self.gameDeck.cardCount() > 1:
            logging.debug("RESHUFFLING")
            self.grabDeck.cards = self.gameDeck.cards[0:-1]
            self.grabDeck.shuffle()
            del self.gameDeck.cards[0:-1]
            _card = self.grabDeck.topCard()
            logging.debug("grabbed card: " + _card.toString())
            self.grabDeck.removeTopCard()
            player.addCard(_card)
        else:
            logging.debug("There are no cards left to grab")
            self.gameEnded = True
        return _card
    def changeSort(self, newSort):
        self.currentSort = newSort
        self.changeSortTurn = False
        logging.debug(("SORT CHANGED:", self.currentSort))

    def throwCard(self, _card, player, lastCard):
        if _card is None:
            logging.debug("Player does not play a card")
            _card = self.grabCard(player)
            self.canPlayGrabbedCard = _card.compatible(self.currentSort, self.currentTrueNumber)
            return
        if not _card.compatible(self.currentSort, self.currentTrueNumber):
            print("card is not compatible????")
        if lastCard is True:
            if _card.truenumber == 7 or _card.truenumber == 8 or _card.truenumber == 1 or _card.truenumber == 0 or _card.truenumber == 13 or _card.truenumber == 2:
                logging.debug("Last card played is pestkaart: " + _card.toString() + ", grabbing two penalty cards")
                self.grabCard(player)
                if self.gameEnded is not True:
                    self.grabCard(player)
                else:
                    return

        logging.debug("threw card:")
        logging.debug(_card.toString())
        self.gameDeck.cards.append(_card)
        self.currentSort = _card.sort
        self.sortsPlayed[card.sorts.index(self.currentSort)] += 1
        self.currentTrueNumber = _card.truenumber
        player.remove(_card)
        if _card.truenumber == 7:
            self.currentPlayerIndex -= self.direction
        elif _card.truenumber == 8:
            self.currentPlayerIndex += self.direction
        elif _card.truenumber == 1:
            self.direction *= -1
        # joker
        elif _card.truenumber == 0:
            self.penaltyAmount += 5
        # boer
        elif _card.truenumber == 13:
            self.changeSort(self.currentPlayer.changeSort())
        elif _card.truenumber == 2:
            self.penaltyAmount += 2
    def calculateNextPlayer(self, playerIndex, direction):
        # return next player
        playerIndex += direction
        if playerIndex >= self.amountOfPlayers:
            playerIndex = playerIndex - self.amountOfPlayers
        elif playerIndex <= -1:
            playerIndex = self.amountOfPlayers + playerIndex
        return playerIndex
