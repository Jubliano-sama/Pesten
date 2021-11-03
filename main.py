import deck
import randomAgent
import game
import logging

def simRandomGame(playerAmount):
    _game = game.Game(4)
    _game.randomSim()

logging.critical("simulations started")
amount = 1
for i in range(100000):
    simRandomGame(4)
    if i%100 == 0:
        amount += 100
        print("games done: " + str(amount -1))
logging.critical("simulations done")