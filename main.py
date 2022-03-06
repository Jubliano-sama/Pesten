import deck
import randomAgent, neuralAgent
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
"""
 logging.critical("simulations started")
amount = -100
games = 100000
for i in range(games):
    simRandomGame(4)
    if i%100 == 0:
        amount += 100
        print("games done: " + str(amount))
logging.critical("simulations done")
print(turns//games)
print(winners)
"""

"""_game = game.Game(4)
_game.players.append(randomAgent.Agent())
_game.players.append(neuralAgent.Agent())
_game.players.append(randomAgent.Agent())
_game.players.append(randomAgent.Agent())
obs = _game.reset(neuralIndex=1)[0]
net = network.FeedForwardNN(121, 54)
pre_action = net.forward(obs)
action = _game.players[1].refine_action(pre_action, _game).squeeze()
for x in range(10):
    pre_action = net.forward(obs)
    action = _game.players[1].refine_action(pre_action, _game).squeeze()
    if action.sum() != 0:
        # Sample an action from the distribution
        dist = Categorical(action)
        _action = dist.sample()
        obs = _game.step(_action.item())[0]
    else:
        obs = _game.step(-1)[0]
        print("no actions available")
"""

print(torch.cuda.is_available())
_ppo = ppo.PPO()
_ppo.learn()

