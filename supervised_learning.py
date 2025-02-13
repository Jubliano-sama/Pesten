#!/usr/bin/env python
"""
Supervised Agent Training & Evaluation with Action Masking, Updated Observation Ordering,
Batched Learning, and Debugging

This script collects training data from games played by smartAgent (via DataCollectorAgent),
trains a supervised network to mimic its decisions, and evaluates the trained network against
smartAgent over many games.

Batched Learning Modification:
------------------------------
Instead of collecting one huge training dataset and then training for many epochs,
we simulate a batch of games, train on that batch for several epochs, and then move on.
This file also prints debugging information (e.g. target distribution, mean logits)
to help diagnose why the loss might jump after iteration 1.
"""

import argparse
import multiprocessing
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import game
import smartAgent, supervisedAgent
import randomAgent
import deck, card

###############################################
# Data Collection Agent with Updated obs()    #
###############################################

class DataCollectorAgent(smartAgent.Agent):
    def __init__(self, game_instance=None):
        super().__init__(game_instance)
        self.collected_data = []  # list of (obs, target) tuples

    def obs(self):
        """
        Constructs a 121-dimensional observation vector.
        (Same as your existing implementation.)
        """
        import game
        _obs = torch.zeros(121, dtype=torch.float)
        # Indices 0-49: count each card in hand.
        for card_obj in self.mydeck.cards:
            _obs[card_obj.number] += 1
            # Indices 50-99: count playable cards.
            if card_obj.compatible(self.game.current_sort, self.game.current_true_number):
                _obs[card_obj.number + 50] += 1
        # Indices 100-104: game.sorts_played info.
        for i in range(5):
            _obs[100 + i] = self.game.sorts_played[i]
        # Index 105: changeSort flag.
        _obs[105] = 1.0 if getattr(self, "changesortbool", False) else 0.0
        # Indices 106-118: players' card counts in turn order.
        try:
            ai_index = self.game.players.index(self)
        except ValueError:
            ai_index = 0
        order = []
        cur_index = ai_index
        for _ in range(len(self.game.players)):
            order.append(cur_index)
            cur_index = self.game.calculate_next_player(cur_index, self.game.direction)
        idx = 106
        for player_index in order:
            player = self.game.players[player_index]
            _obs[idx] = player.mydeck.cardCount()
            idx += 1

        # Index 119: this agent's index.
        try:
            _obs[119] = self.game.players.index(self)
        except ValueError:
            _obs[119] = -1
        # Index 120: current game direction.
        _obs[120] = self.game.direction

        return _obs

    def playCard(self, current_sort, current_true_number):
        """
        Overrides the base playCard() to record (observation, target) pairs.
        **Modification:** If the agent passes (action is None), the sample is not recorded.
        """
        observation = self.obs()
        action = super().playCard(current_sort, current_true_number)
        # Only record if an actual card was played (i.e. action is not None).
        if action is None:
            return action

        target = action.number  # Use the card's number as the target.
        if isinstance(observation, torch.Tensor):
            obs_np = observation.detach().cpu().numpy()
        else:
            obs_np = np.array(observation)
        self.collected_data.append((obs_np, target))
        return action

    def changeSort(self):
        # Set flag so the observation indicates a sort change decision is in effect.
        self.changesortbool = True
        _obs = self.obs()
        self.changesortbool = False
        # Call the parent's changeSort method to get the chosen suit.
        chosen_sort = super().changeSort()
        # Determine the target action index.
        # Your action space uses outputs 50-53 for the four suits.
        # So if card.sorts is, say, ("harten", "ruiten", "schoppen", "klaveren", "special"),
        # then the target index is the index of the chosen suit plus 50.
        suit_index = card.sorts.index(chosen_sort)
        target = suit_index + 50

        # Record this training sample.
        # (Make sure to detach and move the observation to CPU if necessary.)
        if isinstance(_obs, torch.Tensor):
            obs_np = _obs.detach().cpu().numpy()
        else:
            obs_np = _obs
        self.collected_data.append((obs_np, target))
        return chosen_sort

    def addCard(self, _card):
        self.mydeck.cards.append(_card)

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                return
        print("Card not found")


###############################################
# Supervised Agent with Action Masking        #
###############################################

class DirectSkipNet(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(DirectSkipNet, self).__init__()
        # First layer: input -> hidden
        self.fc1 = nn.Linear(121, 200)
        self.bn1 = nn.BatchNorm1d(200)
        self.dropout1 = nn.Dropout(dropout_prob)
        
        # Second layer: hidden -> hidden
        self.fc2 = nn.Linear(200, 200)
        self.bn2 = nn.BatchNorm1d(200)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        # Third (output) layer: hidden -> output
        self.fc3 = nn.Linear(200, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_prob)

        self.fc4 = nn.Linear(128, 55)
        # Skip connections
        self.input_skip = nn.Linear(121, 55)
        self.input_skip2 = nn.Linear(200, 55)
        self.input_skip3 = nn.Linear(200, 128)

    def forward(self, x):
        # Skip connection directly from input.
        skip_out = self.input_skip(x)
        
        # First layer with ReLU, batch norm, and dropout.
        x = F.relu(self.fc1(x))
        x = self.bn1(x)
        x = self.dropout1(x)
        
        # Capture second skip connection after first hidden layer.
        skip_out2 = self.input_skip2(x)
        skip_out3 = self.input_skip3(x)
        # Second layer with ReLU, batch norm, and dropout.
        x = F.relu(self.fc2(x))
        x = self.bn2(x)
        x = self.dropout2(x)
        
        x = F.relu(self.fc3(x) + skip_out3)
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Final layer with no activation function.
        x = self.fc4(x)

        out = x + skip_out + skip_out2
        return out

###############################################
# Data Collection Functions                   #
###############################################

def simulate_game_collect_data(_):
    # Use four DataCollectorAgents.
    agents = [DataCollectorAgent for _ in range(4)]
    g = game.Game(agents)
    g.auto_simulate()
    game_data = []
    for agent in g.players:
        if isinstance(agent, DataCollectorAgent):
            game_data.extend(agent.collected_data)
    return game_data

def worker_collect_data(num_simulations):
    results = []
    for _ in range(num_simulations):
        results.append(simulate_game_collect_data(None))
    return results

def parallel_data_collection(num_games, batch_size):
    num_batches = (num_games + batch_size - 1) // batch_size
    tasks = [batch_size] * num_batches
    remainder = num_games - batch_size * (num_batches - 1)
    if remainder > 0:
        tasks[-1] = remainder
    all_data = []
    start_time = time.time()
    with multiprocessing.Pool() as pool:
        batch_results = pool.map(worker_collect_data, tasks)
    for batch in batch_results:
        for game_data in batch:
            all_data.extend(game_data)
    end_time = time.time()
    print(f"Data collection: {len(all_data)} samples collected from {num_games} games in {end_time - start_time:.2f} seconds")
    return all_data

def prepare_dataloader(training_data, batch_size=256):
    obs_list = [sample[0] for sample in training_data]
    target_list = [sample[1] for sample in training_data]
    # Convert the list of numpy arrays into one single numpy array.
    X = torch.tensor(np.array(obs_list), dtype=torch.float)
    y = torch.tensor(np.array(target_list), dtype=torch.int8)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader

def debug_target_distribution(training_data):
    targets = [sample[1] for sample in training_data]
    unique, counts = np.unique(targets, return_counts=True)
    print("Target distribution:", dict(zip(unique, counts)))

###############################################
# Evaluation Functions                        #
###############################################

global_supervised_agent = None

def init_worker(supervised_state_dict):
    global global_supervised_agent
    global_supervised_agent = supervisedAgent.SupervisedAgent()
    global_supervised_agent.model.load_state_dict(supervised_state_dict)
    global_supervised_agent.model.eval()

def simulate_game_eval(_):
    import copy
    global global_supervised_agent
    simulation_agent = copy.deepcopy(global_supervised_agent)
    g = game.Game([])
    # Insert the supervised agent as player 0.
    g.players.append(simulation_agent)
    simulation_agent.game = g
    # Append one smartAgent.
    for _ in range(1):
        g.players.append(smartAgent.Agent(g))
    g.num_players = len(g.players)
    turn_count, winner = g.auto_simulate()
    return turn_count, winner

def worker_eval(num_simulations):
    results = []
    for _ in range(num_simulations):
        results.append(simulate_game_eval(None))
    return results

def parallel_evaluation(num_games, batch_size, supervised_state_dict):
    num_batches = (num_games + batch_size - 1) // batch_size
    tasks = [batch_size] * num_batches
    remainder = num_games - batch_size * (num_batches - 1)
    if remainder > 0:
        tasks[-1] = remainder
    start_time = time.time()
    with multiprocessing.Pool(initializer=init_worker, initargs=(supervised_state_dict,)) as pool:
        batch_results = pool.map(worker_eval, tasks)
    end_time = time.time()
    all_results = [result for batch in batch_results for result in batch]
    exec_time = end_time - start_time
    return all_results, exec_time

def process_evaluation_results(results):
    num_games = len(results)
    wins_supervised = 0
    draws = 0
    for turns, winner in results:
        if winner is None:
            draws += 1
        elif winner == 0:
            wins_supervised += 1
    win_percentage = wins_supervised / num_games * 100
    print(f"Evaluation over {num_games} games:")
    print(f"  SupervisedAgent wins: {wins_supervised} ({win_percentage:.2f}%)")
    print(f"  Draws: {draws}")

###############################################
# Main Pipeline with Batched Learning         #
###############################################

def prepare_train_val_dataloaders(training_data, batch_size=256, validation_split=0.1):
    """
    Given training_data (a list of (obs, target) tuples), convert them into a TensorDataset,
    and split the dataset into training and validation sets (default 90/10 split).
    Returns two DataLoaders: one for training and one for validation.
    """
    obs_list = [sample[0] for sample in training_data]
    target_list = [sample[1] for sample in training_data]
    # Ensure targets are of integer type (using torch.long)
    X = torch.tensor(np.array(obs_list), dtype=torch.float)
    y = torch.tensor(np.array(target_list), dtype=torch.long)
    dataset = TensorDataset(X, y)
    # Calculate split sizes
    train_size = int((1 - validation_split) * len(dataset))
    val_size = len(dataset) - train_size
    # Randomly split the dataset
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def main():
    parser = argparse.ArgumentParser(description='Train a SupervisedAgent using smartAgent data with batched learning and debug info.')
    parser.add_argument('--collect_batch', type=int, default=1000, help='Batch size for parallel data collection (number of games per pool task; default: 10)')
    parser.add_argument('--train_epochs', type=int, default=5, help='Number of supervised training epochs per iteration (default: 5)')
    parser.add_argument('--eval_games', type=int, default=1000, help='Number of evaluation games (default: 1000)')
    parser.add_argument('--eval_batch', type=int, default=10000, help='Batch size for parallel evaluation (default: 10)')
    parser.add_argument('--iterations', type=int, default=1, help='Number of training iterations for batched learning (default: 10)')
    parser.add_argument('--batch_games', type=int, default=50000, help='Number of games to simulate per training iteration (default: 100)')
    args = parser.parse_args()

    print("Starting batched training...")
    supervised_agent = supervisedAgent.SupervisedAgent()
    # Create a persistent optimizer so we do not lose momentum across iterations.
    optimizer = optim.AdamW(supervised_agent.model.parameters(), lr=0.001, weight_decay=3e-5)
    supervised_agent.model.load_state_dict(torch.load("supervised_agent_V1.pth"))
    for it in range(args.iterations):
         print(f"\nIteration {it+1}/{args.iterations}: Collecting training data from {args.batch_games} game(s)...")
         training_data = parallel_data_collection(args.batch_games, args.collect_batch)
         debug_target_distribution(training_data)
         if len(training_data) == 0:
             print("No training data collected in this iteration. Skipping training update.")
             continue
         train_loader, val_loader = prepare_train_val_dataloaders(training_data, batch_size=2048, validation_split=0.1)
         print(f"Training on collected batch with {len(training_data)} samples...")
         optimizer = supervised_agent.train_supervised(train_loader, optimizer, num_epochs=args.train_epochs, val_loader=val_loader)
         torch.save(supervised_agent.model.state_dict(), f"supervised_agent_iter{it+1}.pth")
         print(f"Iteration {it+1} complete, model saved.")

    torch.save(supervised_agent.model.state_dict(), "supervised_agent.pth")
    print("\nBatched training complete.")

    print("\nEvaluating the trained SupervisedAgent against smartAgent...")
    trained_state = supervised_agent.model.state_dict()
    eval_results, eval_time = parallel_evaluation(args.eval_games, args.eval_batch, trained_state)
    process_evaluation_results(eval_results)
    print(f"Evaluation executed in {eval_time:.2f} seconds.")

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
