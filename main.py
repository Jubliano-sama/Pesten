import deck
import randomAgent
import game
import logging

turns = 0
def simRandomGame(playerAmount):
    global turns
    _game = game.Game(4)
    turn = _game.randomSim()
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