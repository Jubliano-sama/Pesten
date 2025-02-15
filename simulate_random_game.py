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
                "ppo_actor.pth",
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
            self.model.eval()
        except Exception as e:
            print("Failed to load teacher actor weights:", e)

class StudentAgentWrapper(supervisedAgent.SupervisedAgent2):
    """
    Agent wrapper that loads the distilled student actor (from distilled_rl_agent.pth).
    """
    def __init__(self, game_instance):
        super().__init__(game_instance)
        try:
            self.model.load_state_dict(torch.load(
                "distilled_rl_agent.pth",
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
            self.model.eval()
        except Exception as e:
            print("Failed to load student actor weights:", e)

# --- Existing SupervisedAgentWrapper (if needed) ---
class SupervisedAgentWrapper(supervisedAgent.SupervisedAgent):
    def __init__(self, game_instance):
        super().__init__(game_instance)
        try:
            self.model.load_state_dict(torch.load(
                "ppo_actor_147.pth",  # if you still need this one
                map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")
            ))
            self.model.eval()
        except Exception as e:
            print("Failed to load supervised_agent weights:", e)

# --- Simulation Functions (as before) ---

def simulate_game(agent_classes):
    """
    Run a single game simulation using the given agent classes.
    Returns a tuple (turn_count, winner).
    """
    g = game.Game(agent_classes)
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
    wins = [0] * 2
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
        'random': randomAgent.Agent,
        'supervised': SupervisedAgentWrapper,
    }
    
    agent_names = [name.strip().lower() for name in args.agents.split(',')]
    agent_classes = [agent_mapping.get(name, randomAgent.Agent) for name in agent_names]
    
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
