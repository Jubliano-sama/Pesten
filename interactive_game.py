#!/usr/bin/env python
"""
interactive_game.py

an interactive wrapper to play the card game against any mix of agents.
you (human) get to make your moves while the others (random, smart, supervised)
do their thing. displays moves, your hand, and blocks invalid moves.
"""

import argparse
import sys

import game
import card
import randomAgent
import smartAgent
import supervisedAgent
import torch


# add a wrapper for supervised agent to load its weights properly
class SupervisedAgentWrapper(supervisedAgent.SupervisedAgent):
    def __init__(self, game_instance):
        super().__init__(game_instance)
        try:
            self.model.load_state_dict(torch.load("supervised_agent_V1.pth", map_location=torch.device("cpu")))
            self.model.eval()
        except Exception as e:
            print("failed to load supervised_agent weights:", e)


# human agent prompts for moves
class HumanAgent:
    def __init__(self, game_instance=None):
        from deck import Deck
        self.mydeck = Deck([])
        self.type = "Human"
        self.game = game_instance

    def addCard(self, _card):
        self.mydeck.cards.append(_card)

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                return
        print("card not found, bro")

    def printCards(self):
        print("your hand:")
        for idx, c in enumerate(self.mydeck.cards):
            print(f" {idx+1}: {c.toString()}")

    def playCard(self, current_sort, current_true_number):
        print("\n--- your turn ---")
        print(f"current table: {current_true_number} of {current_sort}")
        self.printCards()
        while True:
            move = input("pick a card (number) or type 'pass': ").strip().lower()
            if move == "pass":
                return None
            try:
                index = int(move) - 1
                if index < 0 or index >= len(self.mydeck.cards):
                    print("nah, invalid index. try again.")
                    continue
                chosen = self.mydeck.cards[index]
                if chosen.compatible(current_sort, current_true_number):
                    return chosen
                else:
                    print("that card ain't compatible. choose wisely, fam.")
            except Exception:
                print("invalid input, dude.")

    def changeSort(self):
        print("choose a suit:")
        for i, s in enumerate(card.sorts[:4]):
            print(f" {i+1}: {s}")
        while True:
            move = input("your choice: ").strip()
            try:
                idx = int(move) - 1
                if 0 <= idx < 4:
                    return card.sorts[idx]
                else:
                    print("invalid choice, try again.")
            except Exception:
                print("invalid input, bro.")


# interactive game subclass to print extra info each turn
class InteractiveGame(game.Game):
    def __init__(self, agent_classes):
        super().__init__(agent_classes)

    def simulate_turn(self):
        print(f"\n-- turn {self.turn_count} | current player: {self.current_player_index} --")
        super().simulate_turn()

    def _process_played_card(self, played_card, player, is_last_card):
        super()._process_played_card(played_card, player, is_last_card)
        player_idx = self.players.index(player)
        if played_card is not None:
            print(f">> player {player_idx} played: {played_card.toString()}")
        else:
            print(f">> player {player_idx} passed")


def main():
    parser = argparse.ArgumentParser(description="play interactively against various agents")
    parser.add_argument(
        "--agents",
        type=str,
        default="human,random,smart,supervised",
        help="comma-separated list of agents (human, random, smart, supervised)",
    )
    args = parser.parse_args()

    # mapping agent names to classes, using the wrapper for supervised
    mapping = {
        "human": HumanAgent,
        "random": randomAgent.Agent,
        "smart": smartAgent.Agent,
        "supervised": SupervisedAgentWrapper,
    }

    agent_names = [a.strip().lower() for a in args.agents.split(",")]
    agents = []
    for name in agent_names:
        agent_cls = mapping.get(name, randomAgent.Agent)
        agents.append(agent_cls)
    if "human" not in agent_names:
        print("yo, no human agent specified. adding one for you as player 0.")
        agents[0] = HumanAgent

    # create interactive game instance and reset state
    g = InteractiveGame(agents)
    g.reset()

    # main game loop
    while not g.game_over:
        g.simulate_turn()
    if g.winner is not None:
        print(f"\n*** game over: player {g.winner} wins! ***")
    else:
        print("\n*** game over: draw ***")
    sys.exit(0)


if __name__ == "__main__":
    main()
