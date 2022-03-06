import card
import deck
from random import randint
import logging


class Agent:
    def __init__(self):
        self.mydeck = deck.Deck([])
        self.type = "Random"
    def changeSort(self):
        return card.sorts[randint(0, 3)]

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                break

    def printCards(self):
        self.mydeck.vocalize()

    def addCard(self, _card):
        if not isinstance(_card, card.Card):
            logging.error("None card detected")
        self.mydeck.cards.append(_card)
    def playCard(self, sort, truenumber):
        if len(self.mydeck.cards) != 0:
            self.mydeck.shuffle()
            for _card in self.mydeck.cards:
                if _card.compatible(sort=sort, truenumber=truenumber):
                    return _card
            return None
        else:
            logging.debug("No cards?")
            print("ik heb geen kaarten?")
