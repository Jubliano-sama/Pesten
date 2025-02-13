import torch
import torch.nn as nn
import torch.nn.functional as F
import card
import deck
import numpy as np

class DenseSkipNet(nn.Module):
    def __init__(self, input_dim=121, hidden_dims=None, output_dim=55, dropout_prob=0.3):
        super(DenseSkipNet, self).__init__()
        if hidden_dims is None:
            hidden_dims = [128, 128, 128, 128, 128, 128, 128]
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.dropout = nn.Dropout(dropout_prob)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        current_input_dim = input_dim
        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(current_input_dim, h_dim))
            self.bns.append(nn.BatchNorm1d(h_dim))
            current_input_dim += h_dim
        self.final_layer = nn.Linear(current_input_dim, output_dim)

    def forward(self, x):
        concat = x
        for fc, bn in zip(self.layers, self.bns):
            out = fc(concat)
            if out.size(0) == 1:
                # use functional bn in eval mode if only one sample in batch
                out = nn.functional.batch_norm(out, bn.running_mean, bn.running_var, bn.weight, bn.bias, training=False)
            else:
                out = bn(out)
            out = F.relu(out)
            out = self.dropout(out)
            concat = torch.cat([concat, out], dim=1)
        return self.final_layer(concat)


class SupervisedAgent:
    def __init__(self, game_instance=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        obs[105] = 0  # flag for playCard
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
        if isinstance(obs, torch.Tensor):
            obs_list = obs.detach().cpu().numpy().tolist()
        else:
            obs_list = obs
        mask = np.zeros(55, dtype=np.float32)
        if obs_list[105] == 0:
            for x in range(50, 100):
                if obs_list[x] >= 1:
                    mask[x - 50] = 1
            mask[54] = 1
        else:
            for x in range(50, 54):
                mask[x] = 1
            mask[54] = 0
        return torch.tensor(mask, device=self.device, dtype=torch.float)

    def act(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        else:
            obs = obs.to(self.device)  # <<--- ensure obs is on the same device
        obs_batched = obs.unsqueeze(0)
        logits = self.model(obs_batched).squeeze(0)
        mask = self.get_action_mask(obs)
        masked_logits = torch.where(mask.bool(), logits, torch.tensor(-1e9, device=self.device))
        action_index = torch.argmax(masked_logits).item()
        return None if action_index == 54 or masked_logits[action_index] <= -1e9 else card.Card(action_index)


    def act_rl(self, obs, temperature=1.0):
        # ensure obs is a tensor on the right device
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        obs_batched = obs.unsqueeze(0)
        logits = self.model(obs_batched).squeeze(0)
        mask = self.get_action_mask(obs)
        masked_logits = torch.where(mask.bool(), logits, torch.tensor(-1e9, device=self.device))
        # apply temperature scaling
        scaled_logits = masked_logits / temperature
        dist = torch.distributions.Categorical(logits=scaled_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if action.item() == 54 or scaled_logits[action] <= -1e9:
            return None, log_prob
        else:
            return card.Card(action.item()), log_prob


    def playCard(self, current_sort, current_true_number):
        observation = self.obs()
        return self.act(observation)

    def changeSort(self):
        # decide suit change using outputs 50-53
        observation = self.obs().to(self.device)
        observation[105] = 1.0  # flag for sort change
        obs_batched = observation.unsqueeze(0)
        logits = self.model(obs_batched).squeeze(0)
        suit_logits = logits[50:54]
        dist = torch.distributions.Categorical(logits=suit_logits)
        suit_index = dist.sample().item()
        return card.sorts[suit_index]

    def printCards(self):
        for c in self.mydeck.cards:
            print(c.toString())

    def addCard(self, _card):
        self.mydeck.cards.append(_card)

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                return
        print("card not found in deck.")
