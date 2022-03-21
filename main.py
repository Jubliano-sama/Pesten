import deck
import randomAgent, trainAgent
import game
import logging
import network
import torch
import ppo
from numpy import zeros
from torch.distributions import Categorical

turns = 0
winners = zeros(4)
def simRandomGame(playerAmount):
    global turns
    global winners
    _game = game.Game(4)
    data = _game.randomSim()
    turn = data[0]
    winners[data[1]] += 1
    turns += turn

print(torch.cuda.is_available())
torch.manual_seed(69)
_ppo = ppo.PPO()
#print(str(_ppo.test()) + "                                           ")
_ppo.learn()

