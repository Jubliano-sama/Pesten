import deck
import randomAgent
import game


def simRandomGame(playerAmount):
    _game = game.Game(4)
    _game.randomSim()


for i in range(10000):
    simRandomGame(4)
