import deck
import card
import randomAgent
from random import uniform, randint


class Game:
    def __init__(self):
        self.players = []
        self.grabDeck = None
        self.gameDeck = None
        self.currentPlayerIndex = 0
        self.penaltyAmount = 0
        self.direction = 1  # 1=clockwise -1=anti-clockwise

    # simulates game with random agents
    # TO DO move game logic to separate function, outside of class
    def randomSim(self, amountOfPlayers):
        self.grabDeck = deck.standardDeck()
        self.grabDeck.shuffle()

        for i in range(amountOfPlayers):
            self.players.append(randomAgent.Agent(self.grabDeck))
        self.gameDeck = deck.Deck([self.grabDeck.cards[-1]])
        del self.grabDeck.cards[-1]

        while self.gameDeck.cards[-1].truenumber == 13 or self.gameDeck.cards[-1].truenumber == 0 or \
                self.gameDeck.cards[-1].truenumber == 1 or self.gameDeck.cards[-1].truenumber == 7 \
                or self.gameDeck.cards[-1].truenumber == 8 or self.gameDeck.cards[-1].truenumber == 2:
            self.gameDeck.cards.append(self.grabDeck.cards[-1])
            del self.grabDeck.cards[-1]
        i = 0

        while i < 100:

            nextCard = self.players[self.currentPlayerIndex].playCard(self)
            currentPlayer = self.players[self.currentPlayerIndex]

            print("player", self.currentPlayerIndex)
            # checks if there are cards to be grabbed(joker/2)
            if self.penaltyAmount > 0:
                    # if players next action is playing the same card as the penalty card, add up penalty amounts
                if nextCard is not None:
                    if nextCard.truenumber == self.gameDeck.topCard().truenumber:
                       self.throwCard(nextCard)
                    else:
                        print("grabbing penalty cards:", self.penaltyAmount)
                        for i in range(self.penaltyAmount):
                            currentPlayer.addCard(self.grabCard())
                            self.penaltyAmount = 0
                else:
                    print("grabbing penalty cards", self.penaltyAmount)
                    for i in range(self.penaltyAmount):
                        currentPlayer.addCard(self.grabCard())
                        self.penaltyAmount = 0
            else:
                if nextCard is not None:
                    self.throwCard(nextCard)
                else:
                    print("cannot throw")
                    _grabcard = self.grabCard()
                    if _grabcard is not None:
                        print("grabbed card:")
                        _grabcard.vocalize()
                        self.players[self.currentPlayerIndex].mydeck.cards.append(_grabcard)
                        if _grabcard.compatible(self.gameDeck.cards[-1]) and uniform(0, 1) > 0.5:
                            self.throwCard(_grabcard)
                    else:
                        print("no cards available")
                        break
            self.currentPlayerIndex += self.direction
            if self.currentPlayerIndex >= amountOfPlayers:
                self.currentPlayerIndex = self.currentPlayerIndex - amountOfPlayers
            elif self.currentPlayerIndex <= -1:
                self.currentPlayerIndex = amountOfPlayers + self.currentPlayerIndex
            i += 1

    def grabCard(self):
        if len(self.grabDeck.cards) > 0:
            _card = self.grabDeck.topCard()
            self.grabDeck.removeTopCard()
            return _card
        elif len(self.gameDeck.cards) > 1:
            print("RESHUFFLING")
            self.grabDeck.cards = self.gameDeck.cards[0:-1]
            self.grabDeck.shuffle()
            del self.gameDeck.cards[0:-1]
            _card = self.grabDeck.topCard()
            self.grabDeck.removeTopCard()
            return _card
        else:
            return None

    def throwCard(self, _card):
        print("threw card:")
        _card.vocalize()

        self.gameDeck.cards.append(_card)
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
        elif _card.truenumber == 0:
            self.gameDeck.cards.append(card.Card(sort=randint(0, 4)))
            print(self.gameDeck.cards[-1].sort)
        elif _card.truenumber == 2:
            self.penaltyAmount += 2

    def calculateNextPlayer(self, currentPlayer):
        # return next player
        pass
