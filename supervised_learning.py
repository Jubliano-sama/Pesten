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

class DenseSkipNet(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super(DenseSkipNet, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        
        # layer 1: input -> 200
        self.fc1 = nn.Linear(121, 200)
        self.bn1 = nn.BatchNorm1d(200)
        
        # layer 2: (input + layer1) -> 200
        self.fc2 = nn.Linear(121 + 200, 200)
        self.bn2 = nn.BatchNorm1d(200)
        
        # layer 3: (input + layer1 + layer2) -> 200
        self.fc3 = nn.Linear(121 + 200 + 200, 200)
        self.bn3 = nn.BatchNorm1d(200)
        
        # layer 4: (input + layer1 + layer2 + layer3) -> 128
        self.fc4 = nn.Linear(121 + 200 + 200 + 200, 128)
        self.bn4 = nn.BatchNorm1d(128)

        #layer 5: (input + layer1 + layer2 + layer3 + layer 4) -> 128
        self.fc5 = nn.Linear(121 + 200 + 200 + 200 + 128, 128)
        self.bn5 = nn.BatchNorm1d(128)
        
        # final layer: (input + layer1 + layer2 + layer3 + layer4) -> 55
        self.fc6 = nn.Linear(121 + 200 + 200 + 200 + 128 + 128, 55)

    def forward(self, x):
        # original input is preserved for dense concat.
        out0 = x  # (batch, 121)
        
        out1 = F.relu(self.bn1(self.fc1(out0)))  # (batch, 200)
        out1 = self.dropout(out1)
        cat1 = torch.cat([out0, out1], dim=1)  # (batch, 121+200)
        
        out2 = F.relu(self.bn2(self.fc2(cat1)))  # (batch, 200)
        out2 = self.dropout(out2)
        cat2 = torch.cat([cat1, out2], dim=1)  # (batch, 121+200+200)
        
        out3 = F.relu(self.bn3(self.fc3(cat2)))  # (batch, 200)
        out3 = self.dropout(out3)
        cat3 = torch.cat([cat2, out3], dim=1)  # (batch, 121+200+200+200)
        
        out4 = F.relu(self.bn4(self.fc4(cat3)))  # (batch, 128)
        out4 = self.dropout(out4)
        cat4 = torch.cat([cat3, out4], dim=1)  # (batch, 121+200+200+200+128)
        
        out5 = F.relu(self.bn5(self.fc5(cat4)))  # (batch, 128)
        out5 = self.dropout(out5)
        cat5 = torch.cat([cat4, out5], dim=1)  # (batch, 121+200+200+200+128+128)

        out = self.fc6(cat5)  # (batch, 55)
        return out


def compute_random_baseline_accuracy(batch_obs, batch_targets, get_action_mask, device):
    """
    Computes the expected random accuracy for a batch, assuming that the ground-truth
    action is among the available actions (as it should be). For each sample, if there are N allowed
    actions, the chance of picking the correct one is 1/N.
    """
    total_expected_accuracy = 0.0
    batch_size = batch_obs.size(0)
    # Loop over each sample in the batch.
    for i in range(batch_size):
        # Get the mask for this observation.
        mask = get_action_mask(batch_obs[i])
        # Count the number of available actions.
        num_available = int(mask.sum().item())
        # For debugging, ensure num_available is at least 1.
        if num_available < 1:
            continue
        # If the label is allowed (it should be), the expected accuracy is 1/num_available.
        # (If, for some reason, the label isn’t allowed, then it would be 0.)
        if mask[batch_targets[i]].item() > 0:
            total_expected_accuracy += 1.0 / num_available
        else:
            total_expected_accuracy += 0.0
    return total_expected_accuracy / batch_size

class SupervisedAgent:
    def __init__(self, game_instance=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Replace your sequential model with the new DirectSkipNet.
        self.model = DenseSkipNet(dropout_prob=0.35).to(self.device)
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
        At the end of each epoch, prints the max, min, and average gradient norm for that epoch.
        Gradients are clipped at a norm of 3.0 before the optimizer step.
        """
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for epoch in range(num_epochs):
            epoch_gradient_norms = []  # To store gradient norms for each batch.
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            for batch_obs, batch_targets in train_loader:
                batch_obs = batch_obs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                optimizer.zero_grad()

                # Forward pass.
                outputs = self.model(batch_obs)  # Shape: (batch_size, 55)
                # Compute the action mask for each sample.
                batch_mask = []
                for obs in batch_obs:
                    batch_mask.append(self.get_action_mask(obs))
                batch_mask = torch.stack(batch_mask)  # Shape: (batch_size, 55)
                # Apply the mask.
                masked_outputs = torch.where(batch_mask.bool(), outputs, torch.tensor(-1e9, device=self.device))
                
                # Compute loss.
                loss = criterion(masked_outputs, batch_targets)
                loss.backward()

                # Compute gradient norm before clipping.
                total_norm = 0.0
                for param in self.model.parameters():
                    if param.grad is not None:
                        total_norm += param.grad.data.norm(2).item() ** 2
                total_norm = total_norm ** 0.5
                epoch_gradient_norms.append(total_norm)

                # Clip gradients at a norm of 3.0.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 3.0)
                
                optimizer.step()

                total_loss += loss.item() * batch_obs.size(0)
                preds = torch.argmax(masked_outputs, dim=1)
                total_correct += (preds == batch_targets).sum().item()
                total_samples += batch_obs.size(0)

            # At the end of the epoch, compute max, min, and average gradient norm.
            if epoch_gradient_norms:
                max_norm = max(epoch_gradient_norms)
                min_norm = min(epoch_gradient_norms)
                avg_norm = sum(epoch_gradient_norms) / len(epoch_gradient_norms)
            else:
                max_norm = min_norm = avg_norm = float('nan')
            
            print(f"Epoch {epoch+1} Gradient Norms -- Max: {max_norm:.4f}, Min: {min_norm:.4f}, Avg: {avg_norm:.4f}")

            avg_loss = total_loss / total_samples
            train_accuracy = total_correct / total_samples * 100
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%")
            
            # Run validation if a validation loader is provided.
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
                        
                        # Compute mask for validation samples.
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
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of supervised training epochs per iteration (default: 5)')
    parser.add_argument('--eval_games', type=int, default=10000, help='Number of evaluation games (default: 1000)')
    parser.add_argument('--eval_batch', type=int, default=100, help='Batch size for parallel evaluation (default: 10)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations for batched learning (default: 10)')
    parser.add_argument('--batch_games', type=int, default=2000, help='Number of games to simulate per training iteration (default: 100)')
    args = parser.parse_args()

    print("Starting batched training...")
    supervised_agent = SupervisedAgent()
    # Create a persistent optimizer so we do not lose momentum across iterations.
    optimizer = optim.AdamW(supervised_agent.model.parameters(), lr=0.002, weight_decay=1e-4)
    # supervised_agent.model.load_state_dict(torch.load("supervised_agent_iter3.pth"))
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
