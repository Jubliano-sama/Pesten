"""
ppo2.py

a runnable ppo implementation specialized for further training a pretrained supervised agent,
with shaping rewards:
  +0.1 for every card lost,
  -0.02 for every card gained, and
  -0.05 penalty for every pass.
additionally, samples where the agent’s legal move is ONLY “pass” (i.e. only one choice)
are filtered out so they aren’t used for training.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import game
import network2
import card
import deck

from supervisedAgent import DenseSkipNet

# --- Minimal RL Agent Wrapper using the supervised network ---
# in ppo2.py, modify rl_supervisedagent as follows:
class RL_SupervisedAgent:
    def __init__(self, game_instance, actor, device):
        self.game = game_instance
        self.actor = actor  # ppo actor network (dense skip net)
        self.device = device
        self.mydeck = deck.Deck([])
        self.type = "RL_supervised"
        self.prev_hand_count = 0
        self.lastaction = None
        # add buffers for training samples
        self.episode_obs = []
        self.episode_logprobs = []
        self.episode_rewards = []
        self.episode_act = []

    def reset_episode(self):
        self.episode_obs = []
        self.episode_logprobs = []
        self.episode_rewards = []
        self.episode_act = []

    def obs(self):
        obs = torch.zeros(121, dtype=torch.float)
        for c in self.mydeck.cards:
            obs[c.number] += 1
            if c.compatible(self.game.current_sort, self.game.current_true_number):
                obs[c.number + 50] += 1
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

    def act_rl(self, obs, temperature=1.0):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        obs = obs.to(self.device)
        obs_batched = obs.unsqueeze(0)
        logits = self.actor(obs_batched).squeeze(0)
        mask = self.get_action_mask(obs)
        masked_logits = torch.where(mask.bool(), logits, torch.tensor(-1e15, device=self.device))
        scaled_logits = masked_logits / temperature
        dist = torch.distributions.categorical.Categorical(logits=scaled_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if action.item() == 54:
            return None, log_prob
        else:
            return action.item(), log_prob

    def playCard(self, current_sort, current_true_number, temperature=1.0):
        observation = self.obs()
        legal_mask = self.get_action_mask(observation)

        # if agent can only pass
        test = legal_mask.sum().item()
        if int(legal_mask.sum().item()) == 1:
            return None
        
        observation = observation.to(self.device)
        action, log_prob = self.act_rl(observation, temperature)
        shaping_reward = self.compute_shaping_reward(action)

        # if agent can only play one card or pass
        if int(legal_mask.sum().item()) == 1:
            return card.Card(action) if action is not None else None
        if log_prob is None:
            return card.Card(action) if action is not None else None
        # record training sample
        self.episode_obs.append(observation.detach().cpu().numpy())
        self.episode_logprobs.append(log_prob.detach().cpu().item() if log_prob is not None else None)
        if len(self.episode_rewards) > 0:
            self.episode_rewards[-1] += shaping_reward
        self.episode_rewards.append(-10.0 if action is None else 0.0)
        self.episode_act.append(action if action is not None else 54)
        return card.Card(action) if action is not None else None

    def changeSort(self):
        observation = self.obs()
        observation[105] = 1
        observation = observation.to(self.device)
        mask = self.get_action_mask(observation)
        action, log_prob = self.act_rl(observation)
        shaping_reward = self.compute_shaping_reward(action)

        # record training sample
        self.episode_obs.append(observation.detach().cpu().numpy())
        self.episode_logprobs.append(log_prob.detach().cpu().item() if log_prob is not None else None)
        if len(self.episode_rewards) > 0:
            self.episode_rewards[-1] += shaping_reward
        self.episode_rewards.append(0)
        self.episode_act.append(action)
        return card.sorts[action - 50]

    def addCard(self, _card):
        self.mydeck.cards.append(_card)

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                return
        print("Error: Card not found in deck")

    def compute_shaping_reward(self, action):
        current_count = len(self.mydeck.cards)
        reward = 0.0
        if self.prev_hand_count:
            diff = self.prev_hand_count - current_count
            if diff > 0:
                reward += 0.1 * diff
            elif diff < 0:
                reward += -0.02 * abs(diff)
        if action is None or action == 54:
            reward += -0.05
        self.prev_hand_count = current_count
        return reward


# --- PPO for the Supervised Agent ---
class PPO_Supervised:
    def __init__(self, use_pretrained=True, pretrained_path="supervised_agent.pth"):
        self._init_hyperparameters()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = DenseSkipNet(input_dim=121, output_dim=55).to(self.device)
        if use_pretrained:
            try:
                self.actor.load_state_dict(torch.load(pretrained_path, map_location=self.device))
                print("Loaded pretrained supervised weights.")
            except Exception as e:
                print("Failed to load pretrained weights:", e)
        self.critic = network2.FeedForwardNN(121, 1).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        self.logger = {'t_so_far': 0, 'i_so_far': 0, 'batch_lens': []}

    def _init_hyperparameters(self):
        self.max_timesteps_per_episode = 5000
        self.n_updates_per_iteration = 8
        self.lr = 0.001
        self.gamma = 0.9
        self.clip = 0.2
        self.save_freq = 1

    def get_action_mask(self, obs):
        if isinstance(obs, torch.Tensor):
            obs_list = obs.detach().cpu().numpy().tolist()
        else:
            obs_list = obs
        mask = np.ones(55, dtype=np.float32)
        if obs_list[105] == 0:
            for x in range(50, 100):
                if obs_list[x] == 0:
                    mask[x-50] = 0
            for x in range(50, 54):
                mask[x] = 0
            mask[54] = 1
        else:
            for x in range(0,50):
                mask[x] = 0
            mask[54] = 0
        return torch.tensor(mask, device=self.device, dtype=torch.float)

    def generate_data(self, num_episodes=10, temperature=1.0):
        print(f"\n[data generation] starting data generation for {num_episodes} episode(s)...")
        batch_obs = []
        batch_actions = []
        batch_log_probs = []
        batch_rewards = []
        batch_lens = []
        for ep in range(num_episodes):
            print(f"  generating training episode {ep+1}/{num_episodes}...")
            g = game.Game([])
            agents = []
            rl_agent = RL_SupervisedAgent(g, self.actor, self.device)
            import randomAgent
            for _ in range(1):
                agents.append(randomAgent.Agent())
            agents.append(rl_agent)
            g.players = agents
            g.num_players = len(agents)
            g.reset()
            rl_agent.prev_hand_count = len(rl_agent.mydeck.cards)
            rl_agent.reset_episode()  # clear buffers at episode start
            g.auto_simulate(max_turns=500)
            # add terminal win/loss reward to last step
            if len(rl_agent.episode_obs) > 0:
                if g.winner == g.players.index(rl_agent):
                    rl_agent.episode_rewards[-1] += 1.0
                    print("we WON!")
                elif g.winner is None:
                    rl_agent.episode_rewards[-1] += 0.0
                else:
                    rl_agent.episode_rewards[-1] += -1.0
                    print("we LOST!")
            batch_obs.extend(rl_agent.episode_obs)
            batch_actions.extend(rl_agent.episode_act)
            batch_log_probs.extend(rl_agent.episode_logprobs)
            batch_rewards.extend(rl_agent.episode_rewards)
            batch_lens.append(len(rl_agent.episode_obs))
            print(f"    episode {ep+1} finished: {len(rl_agent.episode_obs)} step(s), winner = {g.winner}, total turns = {g.turn_count}")
        batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float, device=self.device)
        print("[Data Generation] Data generation complete.\n")
        return batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_lens

    def learn(self, num_iterations=10, episodes_per_iter=10, temperature=1.0):
        print(f"\n[Training] Starting training for {num_iterations} iteration(s), {episodes_per_iter} episode(s) per iteration.\n")
        for it in range(num_iterations):
            print(f"[Training] Iteration {it+1}/{num_iterations} started...")
            batch_obs, batch_actions, batch_log_probs, batch_rewards, ep_lens = self.generate_data(episodes_per_iter, temperature)
            # Compute reward-to-go for each episode
            rtgs = []
            idx = 0
            for l in ep_lens:
                ep_rewards = batch_rewards[idx:idx+l]
                discounted = []
                running = 0.0
                for r in reversed(ep_rewards.tolist()):
                    running = r + self.gamma * running
                    discounted.insert(0, running)
                rtgs.extend(discounted)
                idx += l
            rtgs = torch.tensor(rtgs, dtype=torch.float, device=self.device)
            values = self.critic(batch_obs).squeeze()
            advantages = rtgs - values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            for update in range(self.n_updates_per_iteration):
                values = self.critic(batch_obs).squeeze()
                logits = self.actor(batch_obs)
                masks = torch.stack([self.get_action_mask(o) for o in batch_obs])
                masked_logits = torch.where(masks.bool(), logits, torch.tensor(-1e9, device=self.device))
                dists = Categorical(logits=masked_logits)
                new_log_probs = dists.log_prob(batch_actions)
                ratios = torch.exp(new_log_probs - batch_log_probs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.MSELoss()(values, rtgs)
                total_loss = actor_loss + 0.5 * critic_loss
                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                total_loss.backward()
                self.actor_optim.step()
                self.critic_optim.step()
                print(f"    [Update {update+1}/{self.n_updates_per_iteration}] Actor loss: {actor_loss.item():.5f}, Critic loss: {critic_loss.item():.5f}")
            avg_ep_len = np.mean(ep_lens)
            print(f"[Training] Iteration {it+1} complete: Avg episode length = {avg_ep_len:.2f}")
            if it % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')
        print("[Training] Training complete.\n")

# --- Main function ---
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PPO RL training for pretrained supervised agent with shaping rewards and evaluation against a RandomAgent")
    parser.add_argument("--iterations", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--episodes", type=int, default=500, help="Episodes per iteration")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for action selection")
    args = parser.parse_args()

    ppo = PPO_Supervised(use_pretrained=True, pretrained_path="supervised_agent_V1.pth")
    ppo.learn(num_iterations=args.iterations, episodes_per_iter=args.episodes, temperature=args.temperature)

if __name__ == "__main__":
    main()
