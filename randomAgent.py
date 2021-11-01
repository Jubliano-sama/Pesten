import card
import deck


class Agent:
    def __init__(self, _deck):
        self.gamedeck = _deck
        self.mydeck = deck.Deck(_deck.cards[0:7])
        del _deck.cards[0:7]

    def randCompatible(self, card):
        for _card in self.mydeck.cards:
            if card.compatible(_card):
                return _card
        return None
    def printCards(self):
        self.mydeck.vocalize()
    def throwCard(self, game):
        return self.randCompatible(self, game.gameDeck.topCard())
        pass