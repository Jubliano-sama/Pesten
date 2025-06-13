#!/usr/bin/env python
import argparse
import multiprocessing
import time
import game
import randomAgent
import smartAgent
import supervisedAgent
import torch

# --- New Wrappers for Teacher and Student Actors ---

class TeacherAgentWrapper(supervisedAgent.SupervisedAgent):
    """
    Agent wrapper that loads the teacher actor (from ppo_actor.pth).
    """
    def __init__(self, game_instance):
        super().__init__(game_instance)
        try:
            self.model.load_state_dict(torch.load(
                "ppo_actor_OLD.pth",
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
            self.model.eval()
        except Exception as e:
            print("Failed to load teacher actor weights:", e)

import torch
from supervisedAgent import SupervisedAgent2
import matplotlib.pyplot as plt

class StudentAgentWrapper(SupervisedAgent2):
    """
    agent wrapper that loads the distilled student actor (from ppo_actor_80.pth)
    and gathers heuristics on how long cards (grouped by truenumber) are held each game.
    at the end of each game, call finalize_game() to compute per-truenumber averages.
    """
    # aggregated class-level statistics:
    # maps truenumber -> list of per-game average hold times
    aggregated_hold_times = {}
    aggregated_play_count = 0
    aggregated_change_sort_count = 0
    aggregated_games = 0

    def __init__(self, game_instance):
        super().__init__(game_instance)
        try:
            self.model.load_state_dict(torch.load(
                "ppo_actor_80.pth",
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
            self.model.eval()
        except Exception as e:
            print("failed to load student actor weights:", e)
        # track when each card is added (using id as key)
        self.card_entry_times = {}  # maps id(card) -> turn number when added
        # store hold times for this game per truenumber
        self.game_hold_times = {}   # maps truenumber -> list of hold times in this game
        self.play_count = 0
        self.change_sort_count = 0
        StudentAgentWrapper.aggregated_games += 1

    def addCard(self, _card):
        super().addCard(_card)
        card_id = id(_card)
        entry_turn = self.game.turn_count if hasattr(self.game, "turn_count") else 0
        self.card_entry_times[card_id] = entry_turn

    def remove(self, _card):
        card_id = id(_card)
        if card_id in self.card_entry_times:
            current_turn = self.game.turn_count if hasattr(self.game, "turn_count") else 0
            hold_time = current_turn - self.card_entry_times[card_id]
            key = _card.truenumber  # group by truenumber
            if key not in self.game_hold_times:
                self.game_hold_times[key] = []
            self.game_hold_times[key].append(hold_time)
            del self.card_entry_times[card_id]
        super().remove(_card)

    def playCard(self, current_sort, current_true_number, temperature=1e-3):
        self.play_count += 1
        StudentAgentWrapper.aggregated_play_count += 1
        return super().playCard(current_sort, current_true_number, temperature)

    def changeSort(self):
        self.change_sort_count += 1
        StudentAgentWrapper.aggregated_change_sort_count += 1
        return super().changeSort()

    def finalize_game(self):
        """
        call this method at the end of each game to compute, for each truenumber,
        the average hold time (over all cards of that type played in that game) and
        add this game-level average to the aggregated statistics.
        then, reset per-game stats.
        """
        for truenum, hold_list in self.game_hold_times.items():
            if hold_list:
                avg_hold = sum(hold_list) / len(hold_list)
                if truenum not in StudentAgentWrapper.aggregated_hold_times:
                    StudentAgentWrapper.aggregated_hold_times[truenum] = []
                StudentAgentWrapper.aggregated_hold_times[truenum].append(avg_hold)
        # reset game-specific stats
        self.game_hold_times = {}
        self.card_entry_times = {}

    @classmethod
    def report_aggregated_stats(cls):
        print("aggregated average hold times per game (grouped by truenumber):")
        for truenum, avg_list in cls.aggregated_hold_times.items():
            overall_avg = sum(avg_list) / len(avg_list)
            print(f"  truenumber {truenum}: overall avg = {overall_avg:.2f} turns over {len(avg_list)} games")
        print("total playCard calls:", cls.aggregated_play_count)
        print("total changeSort calls:", cls.aggregated_change_sort_count)
        print("games simulated:", cls.aggregated_games)

    @classmethod
    def plot_hold_time_histograms(cls):
        plt.figure(figsize=(10, 6))
        # for each truenumber, plot a histogram of game-level average hold times
        for truenum, avg_list in cls.aggregated_hold_times.items():
            plt.hist(avg_list, bins=20, alpha=0.5, label=f"truenumber {truenum}")
        plt.xlabel("Average Hold Time per Game (turns)")
        plt.ylabel("Frequency (number of games)")
        plt.title("Histogram of Average Card Hold Times by Truenumber")
        plt.legend()
        plt.show()



# --- Simulation Functions (as before) ---

def simulate_game(agent_classes):
    """
    Run a single game simulation using the given agent classes.
    Returns a tuple (turn_count, winner).
    """
    g = game.Game(agent_classes)
    for agent in agent_classes:
        agent.game = g
    turn_count, winner = g.auto_simulate()
    return turn_count, winner

def worker_simulate(args_tuple):
    """
    Run a batch of simulations.
    args_tuple is a tuple: (num_simulations, agent_classes).
    Returns a list of (turn_count, winner) tuples.
    """
    num_simulations, agent_classes = args_tuple
    results = []
    for _ in range(num_simulations):
        results.append(simulate_game(agent_classes))
    return results

def sequential_simulation(num_games, agent_classes):
    """
    Runs the simulations sequentially (single-threaded).
    """
    start_time = time.time()
    results = [simulate_game(agent_classes) for _ in range(num_games)]
    end_time = time.time()
    execution_time = end_time - start_time
    return results, execution_time

def parallel_simulation(num_games, agent_classes, batch_size):
    """
    Runs the simulations in parallel using multiprocessing.
    """
    num_batches = (num_games + batch_size - 1) // batch_size  # Ceiling division
    tasks = [(min(batch_size, num_games - i * batch_size), agent_classes) for i in range(num_batches)]
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        batch_results = pool.map(worker_simulate, tasks)
    end_time = time.time()
    execution_time = end_time - start_time
    results = [result for batch in batch_results for result in batch]
    return results, execution_time

def process_results(results, execution_time, label):
    """
    Processes and prints statistics from simulation results.
    """
    num_games = len(results)
    total_turns = sum(turns for turns, winner in results)
    wins = [0] * 4
    draws = 0
    for turns, winner in results:
        if winner is None:
            draws += 1
        else:
            wins[winner] += 1
    print(f"\n{label}:")
    print(f"  Simulated {num_games} game(s)")
    print(f"  Average turns per game: {total_turns / num_games:.2f}")
    print(f"  Wins by Agent 0: {wins[0]}")
    print(f"  Wins by Agent 1: {wins[1]}")
    print(f"  Wins by Agent 2: {wins[2]}")
    print(f"  Wins by Agent 3: {wins[3]}")
    print(f"  Draws: {draws}")
    print(f"  Execution Time: {execution_time:.4f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Simulate games between agents.')
    parser.add_argument('-n', '--num_games', type=int, default=100, help='Total number of games to simulate (default: 100)')
    parser.add_argument('--agents', type=str, default='teacher,student', help='Comma-separated list of agents. Options include teacher, student, smart, random, supervised')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for parallel execution (default: 10)')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel using multiprocessing')
    args = parser.parse_args()

    # Mapping agent names to classes.
    agent_mapping = {
        'teacher': TeacherAgentWrapper,
        'student': StudentAgentWrapper,
        'smart': smartAgent.Agent,
        'random': randomAgent.AgentRandom,
        'supervised': SupervisedAgentWrapper,
        'smarter': smartAgent.SmarterAgent
    }
    
    agent_names = [name.strip().lower() for name in args.agents.split(',')]
    agent_classes = [agent_mapping.get(name, randomAgent.AgentRandom) for name in agent_names]
    
    print("Simulating games with agent types:")
    for i, agent_cls in enumerate(agent_classes):
        print(f"  Player {i}: {agent_cls.__name__}")
    
    if args.parallel:
        results, execution_time = parallel_simulation(args.num_games, agent_classes, args.batch_size)
        process_results(results, execution_time, "Parallel Execution")
    else:
        results, execution_time = sequential_simulation(args.num_games, agent_classes)
        process_results(results, execution_time, "Single-Threaded Execution")

if __name__ == '__main__':
    main()
