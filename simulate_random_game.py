#!/usr/bin/env python
import argparse
import multiprocessing
import time
import game
import randomAgent
import smartAgent

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
    parser.add_argument('-n', '--num_games', type=int, default=1, help='Total number of games to simulate (default: 1)')
    parser.add_argument('--agents', type=str, default='smart,random', help='Comma-separated list of agents (smart, random)')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for parallel execution (default: 10)')
    parser.add_argument('--parallel', action='store_true', help='Run in parallel using multiprocessing')

    args = parser.parse_args()

    agent_mapping = {
        'smart': smartAgent.Agent,
        'random': randomAgent.Agent,
    }
    
    agent_names = [name.strip().lower() for name in args.agents.split(',')]
    agent_classes = [agent_mapping.get(name, randomAgent.Agent) for name in agent_names]

    if args.parallel:
        results, execution_time = parallel_simulation(args.num_games, agent_classes, args.batch_size)
        process_results(results, execution_time, "Parallel Execution")
    else:
        results, execution_time = sequential_simulation(args.num_games, agent_classes)
        process_results(results, execution_time, "Single-Threaded Execution")

if __name__ == '__main__':
    main()
