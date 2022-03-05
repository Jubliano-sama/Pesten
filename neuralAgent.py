import card
import deck
from random import randint
import numpy
import logging
import logging
import torch
import game
from network import FeedForwardNN

class Agent:
    def __init__(self, _game):
        self.mydeck = deck.Deck([])
        self.type = "AI"
        self.nn = FeedForwardNN(121, 54)
        self.game = _game
    def obs(self):
        # returns obs
        _obs = numpy.zeros(shape = [121])
        # adds one to element corresponding to either cards in deck(0,50) and cards in deck compatible(50,100)
        for _card in self.mydeck.cards:
            _obs[_card.number] += 1
            if _card.compatible(self.game.currentSort, self.game.currentTrueNumber):
                _obs[_card.number + 50] += 1
        for x in range(100, 105):
            _obs[x] = self.game.sortsPlayed[x - 100]
        players = self.game.players
        _obs[105] = int(self.game.changeSortTurn)
        y = 106
        for x in players:
            _obs[y] = x.mydeck.cardCount()
            y += 1
        return _obs
    def refine_action(self, action):

        #set all non-available actions to 0
        #if game.changeSortTurn == False:
        if not self.game.changeSortTurn:
            temp = torch.zeros(size=[54])
            for _card in self.mydeck.cards:
                if _card.compatible(self.game.currentSort, self.game.currentTrueNumber):
                    temp[_card.number] = action[_card.number]
            action = temp
        else:
            for x in range(0, 50):
                action[x] = 0
        return action

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                return
        print("card not found")

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
