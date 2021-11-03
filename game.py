import deck
import card
import logging
import randomAgent
from random import uniform, randint

logging.basicConfig(filename='gamepest.log', format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.CRITICAL)


class Game:
    def __init__(self, playerAmount):
        self.players = []
        self.grabDeck = None
        self.gameDeck = None
        self.currentPlayerIndex = 0
        self.penaltyAmount = 0
        self.currentSort = None
        self.currentTrueNumber = None
        self.amountOfPlayers = playerAmount
        self.winner = None
        self.gameEnded = False
        self.direction = 1  # 1=clockwise -1=anti-clockwise

    # simulates game with random agents
    # TO DO move game logic to separate function, outside of class
    def randomSim(self):
        #generate random players
        for i in range(self.amountOfPlayers):
            self.players.append(randomAgent.Agent())
        return self.sim()

    def sim(self):
        if len(self.players) < 2:
            logging.critical("Cannot start game without or with only one player(s)")
            return
        self.grabDeck = deck.standardDeck()
        self.grabDeck.shuffle()
        #add shuffled cards to player decks
        for player in self.players:
            for i in range(7):
                player.addCard(self.grabDeck.topCard())
                self.grabDeck.removeTopCard()
        self.gameDeck = deck.Deck([self.grabDeck.cards[-1]])
        del self.grabDeck.cards[-1]
        # generate top card
        while self.gameDeck.cards[-1].truenumber == 13 or self.gameDeck.cards[-1].truenumber == 0 or \
                self.gameDeck.cards[-1].truenumber == 1 or self.gameDeck.cards[-1].truenumber == 7 \
                or self.gameDeck.cards[-1].truenumber == 8 or self.gameDeck.cards[-1].truenumber == 2:
            self.gameDeck.cards.append(self.grabDeck.cards[-1])
            del self.grabDeck.cards[-1]
        turn = 0
        self.currentSort = self.gameDeck.topCard().sort
        self.currentTrueNumber = self.gameDeck.topCard().truenumber

        #game loop
        turn = 0
        while not self.gameEnded:
            logging.debug("Turn: " + str(turn))
            currentPlayer = self.players[ self.currentPlayerIndex]
            logging.debug("Player: " + str(self.currentPlayerIndex) + " who has " + str(currentPlayer.mydeck.cardCount()) + " card(s)")
            _card = currentPlayer.playCard(self.currentSort, self.currentTrueNumber)
            #print(turn)
            # checks if player has to grab penalty cards
            if self.penaltyAmount > 0:
                if _card is not None:
                    if _card.truenumber != self.currentTrueNumber:
                        self.grabMultipleCards(self.penaltyAmount, currentPlayer)
                        #check if there werent enough cards to grab, ending the game in a draw
                        self.penaltyAmount = 0
                        if self.gameEnded == True:
                            break
                    else:
                        self.throwCard(_card, currentPlayer)
                else:
                    self.grabMultipleCards(self.penaltyAmount, currentPlayer)
                    self.penaltyAmount = 0
                    # check if there werent enough cards to grab, ending the game in a draw
                    if self.gameEnded == True:
                        break
            else:
                self.throwCard(_card, currentPlayer)
            turn += 1
            if currentPlayer.mydeck.cardCount() == 0:
                self.winner = self.currentPlayerIndex
                self.gameEnded = True
                logging.info("Player: " + str(self.winner) + " has won!")
                break

            self.currentPlayerIndex = self.calculateNextPlayer(self.currentPlayerIndex, self.direction)
        return [self.winner, turn]

    def grabMultipleCards(self, amount, player):
        logging.debug("grabbing " + str(amount) + " cards")
        for i in range(amount):
            if self.gameEnded is not True:
                self.grabCard(player)
            else:
                return

    def grabCard(self, player):
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

    def changeSort(self, newSort):
        self.currentSort = newSort
        logging.debug(("SORT CHANGED:", self.currentSort))

    def throwCard(self, _card, player):
        if _card is None:
            logging.debug("Player does not play a card")
            self.grabCard(player)
            return

        logging.debug("threw card:")
        logging.debug(_card.toString())
        self.gameDeck.cards.append(_card)
        self.currentSort = _card.sort
        self.currentTrueNumber = _card.truenumber
        self.players[self.currentPlayerIndex].mydeck.cards.remove(_card)
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
            self.changeSort(self.players[self.currentPlayerIndex].changeSort())
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
