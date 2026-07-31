import torch
import torch.nn as nn
import torch.nn.functional as F
import card
import deck
import numpy as np
from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_END,
    COMPATIBLE_START,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PASS_ACTION,
    SUIT_ACTION_END,
    SUIT_ACTION_START,
    populate_game_context,
)

class DenseSkipNet(nn.Module):
    def __init__(
        self,
        input_dim=OBSERVATION_SIZE,
        hidden_dims=None,
        output_dim=ACTION_COUNT,
        dropout_prob=0.0,
    ):
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
        self.model = DenseSkipNet().to(self.device)
        self.game = game_instance
        self.mydeck = deck.Deck([])
    
    def obs(self):
        obs = torch.zeros(OBSERVATION_SIZE, dtype=torch.float)
        for card_obj in self.mydeck.cards:
            obs[card_obj.number] += 1
            if self.game.penalty_amount > 0:
                is_playable = (
                    card_obj.truenumber == self.game.current_true_number
                )
            else:
                is_playable = card_obj.compatible(
                    self.game.current_sort,
                    self.game.current_true_number,
                )
            if is_playable:
                obs[card_obj.number + 50] += 1
        try:
            ai_index = self.game.players.index(self)
        except ValueError:
            ai_index = -1
        return populate_game_context(
            obs,
            sorts_played=self.game.sorts_played,
            hand_sizes=[player.mydeck.cardCount() for player in self.game.players],
            player_index=ai_index,
            direction=self.game.direction,
        )

    def get_action_mask(self, obs):
        if isinstance(obs, torch.Tensor):
            obs_list = obs.detach().cpu().numpy().tolist()
        else:
            obs_list = obs
        mask = np.zeros(ACTION_COUNT, dtype=np.float32)
        if obs_list[MODE_INDEX] == 0:
            for x in range(COMPATIBLE_START, COMPATIBLE_END):
                if obs_list[x] >= 1:
                    mask[x - COMPATIBLE_START] = 1
            mask[PASS_ACTION] = 1
        else:
            for x in range(SUIT_ACTION_START, SUIT_ACTION_END):
                mask[x] = 1
            mask[PASS_ACTION] = 0
        return torch.tensor(mask, device=self.device, dtype=torch.float)

    def act(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        else:
            obs = obs.to(self.device)  # <<--- ensure obs is on the same device
        obs_batched = obs.unsqueeze(0)
        logits = self.model(obs_batched).squeeze(0)
        mask = self.get_action_mask(obs)
        masked_logits = torch.where(mask.bool(), logits, torch.tensor(-1e12, device=self.device))
        action_index = torch.argmax(masked_logits).item()
        return None if action_index == 54 or masked_logits[action_index] <= -1e9 else (action_index if 50 <= action_index <= 53 else card.Card(action_index))


    def playCard(self, current_sort, current_true_number):
        observation = self.obs()
        return self.act(observation)

    def changeSort(self):
        observation = self.obs()
        observation[MODE_INDEX] = 1
        observation = observation.to(self.device)
        action = self.act(observation)
        return card.sorts[action - 50]

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

class StudentNet(nn.Module):
    """
    Student network: a larger variant of DenseSkipNet with an extra layer and doubled hidden widths.
    Here we use 8 hidden layers of 256 units each.
    """
    def __init__(self, input_dim=OBSERVATION_SIZE, output_dim=ACTION_COUNT):
        super(StudentNet, self).__init__()
        hidden_dims = [256] * 8
        self.model = DenseSkipNet(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout_prob=0.3)
        
    def forward(self, x):
        return self.model(x)

class SupervisedAgent2:
    def __init__(self, game_instance=None, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = StudentNet().to(self.device)
        self.game = game_instance
        self.mydeck = deck.Deck([])
    
    def obs(self):
        obs = torch.zeros(OBSERVATION_SIZE, dtype=torch.float)
        for card_obj in self.mydeck.cards:
            obs[card_obj.number] += 1
            if self.game.penalty_amount > 0:
                is_playable = (
                    card_obj.truenumber == self.game.current_true_number
                )
            else:
                is_playable = card_obj.compatible(
                    self.game.current_sort,
                    self.game.current_true_number,
                )
            if is_playable:
                obs[card_obj.number + 50] += 1
        try:
            ai_index = self.game.players.index(self)
        except ValueError:
            ai_index = -1
        return populate_game_context(
            obs,
            sorts_played=self.game.sorts_played,
            hand_sizes=[player.mydeck.cardCount() for player in self.game.players],
            player_index=ai_index,
            direction=self.game.direction,
        )

    def get_action_mask(self, obs):
        if isinstance(obs, torch.Tensor):
            obs_list = obs.detach().cpu().numpy().tolist()
        else:
            obs_list = obs
        mask = np.zeros(ACTION_COUNT, dtype=np.float32)
        if obs_list[MODE_INDEX] == 0:
            for x in range(COMPATIBLE_START, COMPATIBLE_END):
                if obs_list[x] >= 1:
                    mask[x - COMPATIBLE_START] = 1
            mask[PASS_ACTION] = 1
        else:
            for x in range(SUIT_ACTION_START, SUIT_ACTION_END):
                mask[x] = 1
            mask[PASS_ACTION] = 0
        return torch.tensor(mask, device=self.device, dtype=torch.float)

    def act(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        else:
            obs = obs.to(self.device)  # <<--- ensure obs is on the same device
        obs_batched = obs.unsqueeze(0)
        logits = self.model(obs_batched).squeeze(0)
        mask = self.get_action_mask(obs)
        masked_logits = torch.where(mask.bool(), logits, torch.tensor(-1e12, device=self.device))
        action_index = torch.argmax(masked_logits).item()
        return None if action_index == 54 or masked_logits[action_index] <= -1e9 else (action_index if 50 <= action_index <= 53 else card.Card(action_index))


    def playCard(self, current_sort, current_true_number):
        observation = self.obs()
        return self.act(observation)

    def changeSort(self):
        observation = self.obs()
        observation[MODE_INDEX] = 1
        observation = observation.to(self.device)
        action = self.act(observation)
        return card.sorts[action - 50]

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
