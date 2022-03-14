import card
import deck
from random import randint
import numpy
import logging
import logging
import torch
import game
from network import FeedForwardNN
from torch.distributions import Categorical


class Agent:
    def __init__(self, _game):
        self.mydeck = deck.Deck([])
        self.type = "AI"
        self.nn = FeedForwardNN(121, 54)
        self.game = _game
        self.changesortbool = False
        self.device = torch.device("cpu")

    def obs(self):
        # returns obs
        _obs = torch.zeros(size=[121], requires_grad=False)
        # adds one to element corresponding to either cards in deck(0,50) and cards in deck compatible(50,100)
        for _card in self.mydeck.cards:
            _obs[_card.number] += 1
            if _card.compatible(self.game.currentSort, self.game.currentTrueNumber):
                _obs[_card.number + 50] += 1
        for x in range(100, 105):
            _obs[x] = self.game.sortsPlayed[x - 100]
        players = self.game.players
        _obs[105] = int(self.changesortbool)
        y = 106
        for x in players:
            _obs[y] = x.mydeck.cardCount()
            y += 1
        i = -1
        for x in range(len(self.game.players)):
            if self.game.players[x] == self:
                i = x
                break
        _obs[119] = i
        _obs[120] = self.game.direction
        return _obs

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                return
        print("card not found")

    def changeSort(self):
        self.changesortbool = True
        _obs = self.obs()
        self.changesortbool = False
        action = self.nn(_obs)
        mask = torch.zeros(size=[55], requires_grad=False)
        for x in range(50, 54):
            mask[x] = 1
        action = mask * action
        sample = torch.argmax(action)
        return card.sorts[sample - 50]

    def printCards(self):
        self.mydeck.vocalize()

    def addCard(self, _card):
        if not isinstance(_card, card.Card):
            logging.error("None card detected")
        self.mydeck.cards.append(_card)

    def playCard(self, sort, truenumber):
        _obs = self.obs()
        action = self.nn(_obs)
        mask = self.action_mask(single_obs=_obs)
        action = mask * action
        possibleActions = torch.count_nonzero(action[:54])
        logging.debug("AI can play " + str(possibleActions.item()) + " cards")
        if possibleActions > 0:
            sample = torch.argmax(action)
            if sample.item() == 54:
                logging.debug("AI willingly passes")
                return None
            return card.Card(sample.item())
        else:
            return None

    def action_mask(self, single_obs):
        single_obs = single_obs.numpy().tolist()
        mask = numpy.ones(shape=[55])
        if single_obs[105] == 0:
            for x in range(50, 100):
                if single_obs[x] == 0:
                    mask[x - 50] = 0
            for x in range(50, 54):
                mask[x] = 0
            mask[54] = 0
        else:
            for x in range(0, 50):
                mask[x] = 0
            mask[54] = 0
        return torch.tensor(mask, requires_grad=False, dtype=torch.int8)
