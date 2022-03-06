from numpy import random
import card
import numpy

class Deck:
    def __init__(self, cards):
        self.cards = cards
        self.count = len(cards)

    def shuffle(self):
        #rng = numpy.random.RandomState(69)
        #rng.shuffle(self.cards)
        random.shuffle(self.cards)

    def vocalize(self):
        for _card in self.cards:
            _card.vocalize()

    def topCard(self):
        return self.cards[-1]

    def removeTopCard(self):
        if self.cards is not None:
            del self.cards[-1]

    def cardCount(self):
        return len(self.cards)


def standardDeck():
    """
    :return: standard deck of cards
    """
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
