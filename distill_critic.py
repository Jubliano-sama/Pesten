#!/usr/bin/env python
"""
distill_critic.py

This script distills the teacher critic (value-function) into a student critic model via supervised learning.
It performs the following steps:
  1. Data Generation via Self-Play:
       Uses the RL agent (trained by ppo2.py) to generate observations from simulated games.
       For each observation, the teacher critic's value (loaded from "ppo_critic.pth") is computed.
  2. Synthetic Data Generation:
       Generates synthetic observations (using a procedure similar to the actor distillation script)
       and computes the teacher critic’s value for each.
  3. Distillation Training:
       The self-play and synthetic datasets are combined (with a 90%/10% train/eval split),
       and a student critic network (a larger variant of DenseSkipNet with output_dim=1)
       is trained via MSE loss to mimic the teacher critic’s estimates.
  4. Saving:
       The distilled student critic is saved to disk.

Ensure your PYTHONPATH includes the repository so that modules (game, deck, card, etc.) can be imported.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random

# Import modules from the repository
import game
import deck
import card
from ppo2 import RL_SupervisedAgent  # RL agent wrapper defined in ppo2.py
from supervisedAgent import DenseSkipNet

# -----------------------------------------------------------------------------
# Global Settings
# -----------------------------------------------------------------------------
device = None
teacher_critic = None  # Teacher critic network (loaded from "ppo_critic.pth")
teacher_actor = None   # Teacher actor network (loaded from "ppo_actor.pth")
 
# -----------------------------------------------------------------------------
# Synthetic Observation Generator
# -----------------------------------------------------------------------------
def generate_synthetic_observation(playcard=True):
    """
    Generate a synthetic 121-dimensional observation.
      - Virtual hand: randomly choose between 1 and 15 cards from a standard shuffled deck.
      - For each card in the hand, increment its count (indices 0..49). For cards compatible
        with a randomly chosen "current card," increment indices 50..99.
      - "Sorts played" (indices 100..104): assign each sort an integer equal to a random 10–40%
        fraction of a random total (0–70).
      - Mode flag (index 105): 0 for playCard mode.
      - Player card counts (indices 106–118): simulate 2 players. Our count (index 106) equals our hand size;
        opponent's count (index 107) is random between 1 and 15; remaining indices are set to 0.
      - Player index (index 119): set to 0.
      - Game direction (index 120): set to 1.
    """
    obs = np.zeros(121, dtype=np.float32)
    # Virtual hand
    num_cards = random.randint(1, 15)
    d = deck.standardDeck()
    d.shuffle()
    hand_cards = d.cards[:num_cards]
    # Choose a "current card" (from after the hand)
    current_card = d.cards[num_cards] if len(d.cards) > num_cards else random.choice(d.cards)
    for c in hand_cards:
        obs[c.number] += 1
        if c.compatible(current_card.sort, current_card.truenumber):
            obs[c.number + 50] += 1
    # Sorts played (indices 100..104)
    total_sort = random.randint(0, 70)
    for i in range(5):
        frac = random.uniform(0.1, 0.4)
        obs[100 + i] = int(total_sort * frac)
    # Mode flag
    obs[105] = 0 if playcard else 1
    # Player card counts: simulate 2 players.
    obs[106] = num_cards           # our count
    obs[107] = random.randint(1, 15) # opponent's count
    for idx in range(108, 119):
        obs[idx] = 0
    # Player index and game direction.
    obs[119] = 0
    obs[120] = 1
    return obs

# -----------------------------------------------------------------------------
# Teacher Critic Value Computation
# -----------------------------------------------------------------------------
def compute_teacher_critic_value(obs):
    """
    Given an observation (numpy array or tensor), compute the teacher critic's scalar value estimate.
    """
    if not isinstance(obs, torch.Tensor):
        obs_tensor = torch.tensor(obs, dtype=torch.float, device=device)
    else:
        obs_tensor = obs.to(device)
    with torch.no_grad():
        # Teacher critic outputs a scalar value.
        value = teacher_critic(obs_tensor.unsqueeze(0)).squeeze(0)
    return value.item()

# -----------------------------------------------------------------------------
# Data Generation for Critic Distillation Using RL_SupervisedAgent
# -----------------------------------------------------------------------------
def collect_self_play_data_critic(num_points_target=10000):
    """
    Simulate games using the RL agent (trained by ppo2.py) to generate observations.
    For each observation (collected from agent.episode_obs), compute the teacher critic's value.
    """
    data = []
    collected = 0

    # Define a wrapper that instantiates an RL_SupervisedAgent using the teacher actor.
    def TeacherAgent(game_instance):
        return RL_SupervisedAgent(game_instance, teacher_actor, device)

    print("Starting self-play data collection for critic (using RL_SupervisedAgent)...")
    while collected < num_points_target:
        g = game.Game([TeacherAgent, TeacherAgent])
        g.reset()
        g.auto_simulate(max_turns=500)
        for agent in g.players:
            for obs in agent.episode_obs:
                obs_np = obs.detach().cpu().numpy() if isinstance(obs, torch.Tensor) else np.array(obs)
                value = compute_teacher_critic_value(obs_np)
                data.append((obs_np, value))
                collected += 1
                if collected >= num_points_target:
                    break
        if collected % 1000 < 50:
            print(f"Collected {collected} critic datapoints so far...")
    print("Self-play critic data collection complete.")
    return data

def generate_synthetic_data_critic(num_points):
    """
    Generate synthetic datapoints for critic distillation.
    For each synthetic observation, compute the teacher critic's value estimate.
    """
    data = []
    for _ in range(num_points):
        obs = generate_synthetic_observation(playcard=True)
        value = compute_teacher_critic_value(obs)
        data.append((obs, value))
    return data

# -----------------------------------------------------------------------------
# Student Critic Network
# -----------------------------------------------------------------------------
class StudentCritic(nn.Module):
    """
    A student critic network: a larger variant of DenseSkipNet with output_dim=1.
    Here we use 8 hidden layers of 256 units each.
    """
    def __init__(self, input_dim=121, output_dim=1):
        super(StudentCritic, self).__init__()
        hidden_dims = [512] * 8
        self.model = DenseSkipNet(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, dropout_prob=0.3)
    def forward(self, x):
        return self.model(x)

# -----------------------------------------------------------------------------
# Training Function for Critic Distillation
# -----------------------------------------------------------------------------
def train_student_critic(dataset, epochs=10, batch_size=256, learning_rate=1e-3):
    """
    Train the student critic network using MSE loss between teacher and student value estimates.
    The dataset consists of tuples (observation, teacher_value).
    A 90%/10% train/evaluation split is used to monitor performance.
    """
    random.shuffle(dataset)
    N = len(dataset)
    split_idx = int(0.9 * N)
    train_data = dataset[:split_idx]
    eval_data = dataset[split_idx:]
    
    train_obs = np.array([d[0] for d in train_data])
    train_values = np.array([d[1] for d in train_data])
    train_obs_tensor = torch.tensor(train_obs, dtype=torch.float)
    train_values_tensor = torch.tensor(train_values, dtype=torch.float).unsqueeze(1)
    train_ds = TensorDataset(train_obs_tensor, train_values_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    
    eval_obs = np.array([d[0] for d in eval_data])
    eval_values = np.array([d[1] for d in eval_data])
    eval_obs_tensor = torch.tensor(eval_obs, dtype=torch.float)
    eval_values_tensor = torch.tensor(eval_values, dtype=torch.float).unsqueeze(1)
    eval_ds = TensorDataset(eval_obs_tensor, eval_values_tensor)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False)
    
    student_critic = StudentCritic().to(device)
    optimizer = optim.Adam(student_critic.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    student_critic.train()
    print("Starting student critic training...")
    for ep in range(epochs):
        total_loss = 0.0
        for batch_obs, batch_values in train_loader:
            batch_obs = batch_obs.to(device)
            batch_values = batch_values.to(device)
            optimizer.zero_grad()
            student_values = student_critic(batch_obs)
            loss = criterion(student_values, batch_values)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_obs.size(0)
        avg_loss = total_loss / len(train_ds)
        
        student_critic.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for batch_obs, batch_values in eval_loader:
                batch_obs = batch_obs.to(device)
                batch_values = batch_values.to(device)
                student_values = student_critic(batch_obs)
                loss = criterion(student_values, batch_values)
                eval_loss += loss.item() * batch_obs.size(0)
        avg_eval_loss = eval_loss / len(eval_ds)
        student_critic.train()
        print(f"Epoch {ep+1}/{epochs} - Train MSE: {avg_loss:.4f} - Eval MSE: {avg_eval_loss:.4f}")
    print("Student critic training complete.")
    return student_critic

# -----------------------------------------------------------------------------
# Main: Load Teacher Critic & Actor, Generate Data, Train, and Save Student Critic
# -----------------------------------------------------------------------------
def main():
    global teacher_critic, teacher_actor, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading teacher critic model...")
    teacher_critic = DenseSkipNet(input_dim=121, output_dim=1, dropout_prob=0.3)
    teacher_critic.load_state_dict(torch.load("ppo_critic.pth", map_location=device))
    teacher_critic.to(device)
    teacher_critic.eval()
    print("Teacher critic model loaded.")
    
    print("Loading teacher actor model (for data generation) ...")
    teacher_actor = DenseSkipNet(input_dim=121, output_dim=55, dropout_prob=0.3)
    teacher_actor.load_state_dict(torch.load("ppo_actor.pth", map_location=device))
    teacher_actor.to(device)
    teacher_actor.eval()
    print("Teacher actor model loaded.")
    
    # Generate data via self-play using the RL agent from ppo2.py.
    self_play_data = collect_self_play_data_critic(num_points_target=100000)
    synthetic_data = generate_synthetic_data_critic(2000)
    combined_dataset = self_play_data + synthetic_data
    print(f"Total critic dataset size: {len(combined_dataset)} datapoints.")
    random.shuffle(combined_dataset)
    
    student_critic = train_student_critic(combined_dataset, epochs=14, batch_size=256, learning_rate=1e-3)
    
    torch.save(student_critic.state_dict(), "distilled_critic.pth")
    print("Distilled student critic model saved as 'distilled_critic.pth'.")

if __name__ == "__main__":
    main()
