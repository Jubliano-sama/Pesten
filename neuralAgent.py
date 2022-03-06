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
        self.episode_obs = []
        self.episode_mask = []
        self.episode_act = []
        self.amount_of_steps = 0
        self.episode_logprobs = []
        self.changesortbool = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def obs(self):
        # returns obs
        _obs = numpy.zeros(shape=[121])
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
        return _obs

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                return
        print("card not found")

    def changeSort(self):
        self.amount_of_steps += 1
        self.changesortbool = True
        _obs = self.obs()
        self.changesortbool = False
        action = self.nn.forward(_obs)
        mask = torch.zeros(size=[55])
        for x in range(50, 54):
            mask[x] = 1
        self.episode_mask.append(mask.detach().numpy())
        action = mask * action
        dist = Categorical(action)
        sample = dist.sample()
        self.episode_logprobs.append(dist.log_prob(sample).detach())
        self.episode_obs.append(_obs)
        self.episode_act.append(sample)
        return card.sorts[sample.item() - 50]

    def printCards(self):
        self.mydeck.vocalize()

    def addCard(self, _card):
        if not isinstance(_card, card.Card):
            logging.error("None card detected")
        self.mydeck.cards.append(_card)

    def playCard(self, sort, truenumber):
        self.amount_of_steps += 1
        _obs = self.obs()
        action = self.nn.forward(_obs)
        mask = self.action_mask(single_obs=_obs)
        action = mask * action
        if torch.count_nonzero(action[:54]) > 0:
            dist = Categorical(action)
            sample = dist.sample()
            self.episode_logprobs.append(dist.log_prob(sample).detach())
            self.episode_obs.append(_obs)
            self.episode_act.append(sample)
            self.episode_mask.append(mask.detach().numpy())
            if sample.item() == 54:
                return None
            return card.Card(sample.item())
        else:
            return None


    def action_mask(self, batch_obs=None, single_obs=None):
        if batch_obs is None and single_obs is None:
            print("ERROR: need at least one input")
        elif batch_obs is not None and single_obs is not None:
            print("ERROR: can only handle one input at a time")
        elif batch_obs is not None:
            mask = torch.ones(size=(len(batch_obs), 54))  # creates a mask with all values enabled by default
            y = 0
            for obs in batch_obs:
                if obs[105] == 0:
                    compatible_cards = obs[50:100]
                    for x in range(0, 50):
                        if compatible_cards[x] == 0:
                            mask[y][x] = 0
                    for x in range(50, 54):
                        mask[y][x] = 0
                else:
                    for x in range(0, 50):
                        mask[y][x] = 0
                y += 1

            return mask
        else:
            mask = torch.ones(size=[55])
            if single_obs[105] == 0:
                compatible_cards = single_obs[50:100]
                for x in range(0, 50):
                    if compatible_cards[x] == 0:
                        mask[x] = 0
                for x in range(50, 54):
                    mask[x] = 0
                r = randint(0,63)
                mask[54] = 1 if r == 0 else 0
            else:
                for x in range(0, 50):
                    mask[x] = 0
                mask[54] = 0
            return mask
