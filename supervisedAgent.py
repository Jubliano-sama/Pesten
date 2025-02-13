import torch
import torch.nn as nn
import torch.nn.functional as F
import card
import deck
import numpy as np
class DenseSkipNet(nn.Module):
    def __init__(self, input_dim=121, hidden_dims=None, output_dim=55, dropout_prob=0.3):
        super(DenseSkipNet, self).__init__()
        # default: 5 layers like [200, 200, 200, 128, 128]
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128, 128, 128, 128]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.dropout = nn.Dropout(dropout_prob)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()

        # each hidden layer receives the concatenation of the original input
        # and all previous hidden outputs, so input dims accumulate.
        current_input_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_input_dim, h_dim))
            self.bns.append(nn.BatchNorm1d(h_dim))
            current_input_dim += h_dim  # update for concatenation

        self.final_layer = nn.Linear(current_input_dim, output_dim)

    def forward(self, x):
        # x: (batch, input_dim)
        concat = x
        for fc, bn in zip(self.layers, self.bns):
            out = fc(concat)
            out = bn(out)
            out = F.relu(out)
            out = self.dropout(out)
            concat = torch.cat([concat, out], dim=1)
        return self.final_layer(concat)


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
        self.model = DenseSkipNet(dropout_prob=0.3).to(self.device)
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
        # ensure obs is a 1D tensor of shape [121]
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        else:
            obs = obs.to(self.device)
            
        # batchnorm wants shape [batch_size, 121], so add a dimension
        obs_batched = obs.unsqueeze(0)  # shape [1, 121]
        
        # forward pass
        logits = self.model(obs_batched)  # shape [1, 55]
        
        # squeeze back to shape [55] for argmax
        logits = logits.squeeze(0)
        
        # the action mask code expects the original shape [121], so just pass obs (the 1D version)
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
        observation[105] = 1.0
        obs_batched = observation.unsqueeze(0)  # shape [1, 121]
        # forward pass
        logits = self.model(obs_batched)  # shape [1, 55]
        
        # squeeze back to shape [55] for argmax
        logits = logits.squeeze(0)
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