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
import smartAgent
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
    def __init__(self):
        super(DirectSkipNet, self).__init__()
        self.fc1 = nn.Linear(121, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 55)
        # This layer projects the input directly to the output space.
        self.input_skip = nn.Linear(121, 55)

    def forward(self, x):
        # Compute the skip connection.
        skip_out = self.input_skip(x)
        # Process x through the hidden layers.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # Add the skip connection to the final output.
        out = x + skip_out
        return out



class SupervisedAgent:
    def __init__(self, game_instance=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Replace your sequential model with the new DirectSkipNet.
        self.model = DirectSkipNet().to(self.device)
        self.game = game_instance
        self.mydeck = deck.Deck([])

    def obs(self):
        obs = torch.zeros(121, dtype=torch.float)
        for card_obj in self.mydeck.cards:
            obs[card_obj.number] += 1
            if card_obj.compatible(self.game.current_sort, self.game.current_true_number):
                obs[card_obj.number + 50] += 1
        for i in range(5):
            obs[100 + i] = self.game.sorts_played[i]
        obs[105] = 0
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
            obs[idx] = player.mydeck.cardCount()
            idx += 1
        try:
            obs[119] = self.game.players.index(self)
        except ValueError:
            obs[119] = -1
        obs[120] = self.game.direction
        return obs

    def get_action_mask(self, obs):
        obs_list = obs.detach().cpu().numpy().tolist() if isinstance(obs, torch.Tensor) else obs
        mask = np.ones(55, dtype=np.float32)
        if obs_list[105] == 0:
            for x in range(50, 100):
                if obs_list[x] == 0:
                    mask[x - 50] = 0
            for x in range(50, 54):
                mask[x] = 0
            mask[54] = 1
        else:
            for x in range(0, 50):
                mask[x] = 0
            mask[54] = 0
        return torch.tensor(mask, device=self.device, dtype=torch.float)

    def act(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        else:
            obs = obs.to(self.device)
        logits = self.model(obs)
        mask = self.get_action_mask(obs)
        masked_logits = torch.where(mask.bool(), logits, torch.tensor(-1e9, device=self.device))
        action_index = torch.argmax(masked_logits).item()
        if action_index == 54 or masked_logits[action_index] <= -1e9:
            return None
        else:
            return card.Card(action_index)

    def playCard(self, current_sort, current_true_number):
        observation = self.obs()
        return self.act(observation)

    def addCard(self, _card):
        self.mydeck.cards.append(_card)

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                return
        print("Card not found in deck.")

    def changeSort(self):
        observation = self.obs().to(self.device)
        logits = self.model(observation)
        suit_logits = logits[50:54]
        suit_index = torch.argmax(suit_logits).item()
        return card.sorts[suit_index]

    def printCards(self):
        for c in self.mydeck.cards:
            print(c.toString())

    def train_supervised(self, train_loader, optimizer, num_epochs=5, val_loader=None):
        """
        Train the network using cross-entropy loss on (observation, target) pairs.
        For each sample in the batch, an action mask is computed so that unavailable actions
        have their logits set to -1e9 (preventing them from affecting the loss/gradient).
        If a val_loader is provided, the function will also compute and print validation
        loss and accuracy after each epoch.
        """
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_correct = 0
            total_samples = 0
            for batch_obs, batch_targets in train_loader:
                batch_obs = batch_obs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_obs)  # shape: (batch_size, 55)
                
                # Compute a mask for each sample in the batch.
                batch_mask = []
                for obs in batch_obs:
                    batch_mask.append(self.get_action_mask(obs))
                batch_mask = torch.stack(batch_mask)  # shape: (batch_size, 55)
                
                # Mask out unavailable actions.
                masked_outputs = torch.where(batch_mask.bool(), outputs, torch.tensor(-1e9, device=self.device))
                loss = criterion(masked_outputs, batch_targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_obs.size(0)
                
                # Compute predictions and accuracy.
                preds = torch.argmax(masked_outputs, dim=1)
                total_correct += (preds == batch_targets).sum().item()
                total_samples += batch_obs.size(0)
            
            avg_loss = total_loss / total_samples
            train_accuracy = total_correct / total_samples * 100
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
            
            # Run validation if a validation DataLoader is provided.
            if val_loader is not None:
                self.model.eval()
                total_val_loss = 0.0
                total_val_correct = 0
                total_val_samples = 0
                with torch.no_grad():
                    for batch_obs, batch_targets in val_loader:
                        batch_obs = batch_obs.to(self.device)
                        batch_targets = batch_targets.to(self.device)
                        outputs = self.model(batch_obs)
                        
                        # Compute mask for validation batch.
                        batch_mask = []
                        for obs in batch_obs:
                            batch_mask.append(self.get_action_mask(obs))
                        batch_mask = torch.stack(batch_mask)
                        
                        masked_outputs = torch.where(batch_mask.bool(), outputs, torch.tensor(-1e9, device=self.device))
                        loss = criterion(masked_outputs, batch_targets)
                        total_val_loss += loss.item() * batch_obs.size(0)
                        preds = torch.argmax(masked_outputs, dim=1)
                        total_val_correct += (preds == batch_targets).sum().item()
                        total_val_samples += batch_obs.size(0)
                
                avg_val_loss = total_val_loss / total_val_samples
                val_accuracy = total_val_correct / total_val_samples * 100
                print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
                self.model.train()
        return optimizer


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
    global_supervised_agent = SupervisedAgent()
    global_supervised_agent.model.load_state_dict(supervised_state_dict)
    global_supervised_agent.eval_mode()

def simulate_game_eval(_):
    global global_supervised_agent
    g = game.Game([])
    # Insert the supervised agent as player 0.
    g.players.append(global_supervised_agent)
    global_supervised_agent.game = g
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
    parser.add_argument('--eval_games', type=int, default=10000, help='Number of evaluation games (default: 1000)')
    parser.add_argument('--eval_batch', type=int, default=100, help='Batch size for parallel evaluation (default: 10)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations for batched learning (default: 10)')
    parser.add_argument('--batch_games', type=int, default=5000, help='Number of games to simulate per training iteration (default: 100)')
    args = parser.parse_args()

    print("Starting batched training...")
    supervised_agent = SupervisedAgent()
    # Create a persistent optimizer so we do not lose momentum across iterations.
    optimizer = optim.Adam(supervised_agent.model.parameters(), lr=0.002, weight_decay=1e-4)

    for it in range(args.iterations):
         print(f"\nIteration {it+1}/{args.iterations}: Collecting training data from {args.batch_games} game(s)...")
         training_data = parallel_data_collection(args.batch_games, args.collect_batch)
         debug_target_distribution(training_data)
         if len(training_data) == 0:
             print("No training data collected in this iteration. Skipping training update.")
             continue
         train_loader, val_loader = prepare_train_val_dataloaders(training_data, batch_size=64, validation_split=0.1)
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
