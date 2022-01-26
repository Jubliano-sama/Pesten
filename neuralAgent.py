import card
import deck
from random import randint
import logging
import logging
import torch


class Agent:
    def __init__(self):
        self.mydeck = deck.Deck([])

    def returnState(self):
        # returns state
        pass

    def changeSort(self):
        # return sort
        pass

    def printCards(self):
        self.mydeck.vocalize()

    def addCard(self, _card):
        if not isinstance(_card, card.Card):
            logging.error("None card detected")
        self.mydeck.cards.append(_card)

    def playCard(self, sort, truenumber):
        if len(self.mydeck.cards) != 0:
            # return card to play
            pass
        else:
            logging.debug("No cards?")
            print("ik heb geen kaarten?")
