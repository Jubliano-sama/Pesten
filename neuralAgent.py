import card
import deck
from random import randint
import numpy
import logging
import logging
import torch
import game


class Agent:
    def __init__(self, _game):
        self.mydeck = deck.Deck([])
        self.type = "AI"
        self._game = _game

    def obs(self):
        # returns obs
        _obs = numpy.zeros(shape = [120])
        for _card in self.mydeck.cards:
            _obs[_card.truenumber] += 1
            if _card.compatible(self._game.currentSort, self._game.currentTrueNumber):
                _obs[_card.truenumber + 50] += 1
        for x in range(100, 105):
            _obs[x] = self._game.sortsPlayed[x - 100]
        players = self._game.players
        y = 105
        for x in players:
            _obs[y] = x.mydeck.cardCount()
            y += 1
        return _obs

    def changeSort(self):
        # return sort
        print("AI is not game ready yet")

        pass

    def printCards(self):
        self.mydeck.vocalize()

    def addCard(self, _card):
        if not isinstance(_card, card.Card):
            logging.error("None card detected")
        self.mydeck.cards.append(_card)

    def playCard(self, sort, truenumber):
        print("AI is not game ready yet")
