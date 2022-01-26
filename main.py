import deck
import randomAgent
import game
import logging
import torch
from numpy import zeros

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