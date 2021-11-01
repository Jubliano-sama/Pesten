import deck
import randomAgent
import game


def simRandomGame(playerAmount):
    _game = game.Game()
    _game.randomSim(playerAmount)


for i in range(1):
    simRandomGame(4)
