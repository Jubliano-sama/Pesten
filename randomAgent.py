import card
import deck


class Agent:
    def __init__(self, _deck):
        self.gamedeck = _deck
        self.mydeck = deck.Deck(_deck.cards[0:7])
        del _deck.cards[0:7]

    def randCompatible(self, compCard):
        self.mydeck.shuffle()
        for _card in self.mydeck.cards:
            if compCard.compatible(_card):
                return _card
        return None
    def printCards(self):
        self.mydeck.vocalize()
    def addCard(self, card):
        self.mydeck.cards.append(card)
    def playCard(self, game):
        return self.randCompatible(game.gameDeck.topCard())