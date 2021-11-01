import deck
import card
import randomAgent
from random import uniform, randint


class Game:
    def __init__(self):
        self.players = []
        self.grabDeck = None
        self.gameDeck = None
        self.x = 0
        self.direction = 1

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
            _card = self.players[self.x].randCompatible(self.gameDeck.cards[-1])
            print("player", self.x)
            if _card is not None:
                self.throwCard(_card)
            else:
                print("cannot throw")
                _grabcard = self.grabCard()
                if _grabcard is not None:
                    print("grabbed card:")
                    _grabcard.vocalize()
                    self.players[self.x].mydeck.cards.append(_grabcard)
                    if _grabcard.compatible(self.gameDeck.cards[-1]) and uniform(0, 1) > 0.5:
                        self.throwCard(_grabcard)
                else:
                    print("no cards available")
                    break
            self.x += self.direction
            if self.x >= amountOfPlayers:
                self.x = self.x - amountOfPlayers
            elif self.x <= -1:
                self.x = amountOfPlayers + self.x
            i += 1

    def grabCard(self):
        if len(self.grabDeck.cards) > 0:
            _card = self.grabDeck.cards[-1]
            del self.grabDeck.cards[-1]
            return _card
        elif len(self.gameDeck.cards) > 1:
            print("RESHUFFLING")
            self.grabDeck.cards = self.gameDeck.cards[0:-1]
            del self.gameDeck.cards[0:-1]
            _card = self.grabDeck.cards[-1]
            del self.grabDeck.cards[-1]
            return _card
        else:
            return None

    def throwCard(self, _card):
        print("threw card:")
        _card.vocalize()

        self.gameDeck.cards.append(_card)
        self.players[self.x].mydeck.cards.remove(_card)
        if _card.truenumber == 7:
            self.x -= self.direction
        elif _card.truenumber == 8:
            self.x += self.direction
        elif _card.truenumber == 1:
            self.direction *= -1
        # joker
        elif _card.truenumber == 0:
            pass
        # boer
        elif _card.truenumber == 0:
            self.gameDeck.cards.append(card.Card(sort=randint(0, 4)))
            print(self.gameDeck.cards[-1].sort)
        elif _card.truenumber == 2:
            pass

    def handleNextPlayerGrab(self, grabAmount, _card, currentPlayer):
        #handle joker and 2s
        #recursion?
        pass

    def calculateNextPlayer(self, currentPlayer):
        #return next player
        pass