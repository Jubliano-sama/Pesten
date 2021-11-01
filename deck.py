import numpy as np
import card


class Deck:
    def __init__(self, cards):
        self.cards = cards
        self.count = len(cards)

    def shuffle(self):
        np.random.shuffle(self.cards)

    def vocalize(self):
        for _card in self.cards:
            _card.vocalize()

    def topCard(self):
        return self.cards[-1]

    def removeTopCard(self):
        if self.cards is not None:
            del self.cards[-1]


def standardDeck():
    cards = []
    # 11 = koning #12 = koningin #1=aas 13= boer 0=joker
    for number in range(0, 50):
        _card = card.Card(number)
        cards.append(_card)
    cards.append(card.Card(49))
    cards.append(card.Card(48))
    cards.append(card.Card(48))
    cards.append(card.Card(48))
    return Deck(cards)
