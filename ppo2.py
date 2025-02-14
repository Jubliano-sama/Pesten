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
import os

import game
import network2
import card
import deck

from supervisedAgent import DenseSkipNet

# --- minimal rl agent wrapper using the supervised network ---
class RL_SupervisedAgent:
    def __init__(self, game_instance, actor, device):
        self.game = game_instance
        self.actor = actor  # ppo actor network (DenseSkipNet)
        self.device = device
        self.mydeck = deck.Deck([])
        self.type = "RL_supervised"
        self.prev_hand_count = 0
        self.lastaction = None
        # buffers for training samples
        self.episode_obs = []
        self.episode_logprobs = []
        self.episode_rewards = []
        self.episode_act = []
        # flag for evaluation mode; when true, forces greedy (near-zero temperature)
        self.eval_mode = False

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
        # if in eval mode, force near-greedy behavior
        if self.eval_mode:
            temperature = 1e-3
        scaled_logits = masked_logits / temperature
        dist = Categorical(logits=scaled_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        if action.item() == 54:
            return None, log_prob
        else:
            return action.item(), log_prob

    def playCard(self, current_sort, current_true_number, temperature=1.0):
        observation = self.obs()
        legal_mask = self.get_action_mask(observation)
        if int(legal_mask.sum().item()) == 1:
            return None
        observation = observation.to(self.device)
        # override temperature for eval mode
        if self.eval_mode:
            temperature = 1e-3
        action, log_prob = self.act_rl(observation, temperature)
        shaping_reward = self.compute_shaping_reward(action)
        # record training sample (even during eval, though you might choose not to)
        self.episode_obs.append(observation.detach().cpu().numpy())
        self.episode_logprobs.append(log_prob.detach().cpu().item() if log_prob is not None else None)
        if len(self.episode_rewards) > 0:
            self.episode_rewards[-1] += shaping_reward
        self.episode_rewards.append(-0.1 if action is None else 0.0)
        self.episode_act.append(action if action is not None else 54)
        return card.Card(action) if action is not None else None

    def changeSort(self):
        observation = self.obs()
        observation[105] = 1
        observation = observation.to(self.device)
        mask = self.get_action_mask(observation)
        # for eval mode, force greedy action selection
        temp = 1e-3 if self.eval_mode else 1.0
        action, log_prob = self.act_rl(observation, temperature=temp)
        shaping_reward = self.compute_shaping_reward(action)
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
        print("error: card not found in deck")

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


# --- ppo for the supervised agent ---
class PPO_Supervised:
    def __init__(self, use_pretrained=True,
                 actor_path="ppo_actor.pth",
                 critic_path="ppo_critic.pth",
                 actor_opt_path="ppo_actor_opt.pth",
                 critic_opt_path="ppo_critic_opt.pth",
                 max_saves=5):
        self._init_hyperparameters()
        self.max_saves = max_saves
        self.saved_checkpoints = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = DenseSkipNet(input_dim=121, output_dim=55).to(self.device)
        self.critic = network2.FeedForwardNN(121, 1).to(self.device)
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr)
        if use_pretrained:
            try:
                self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                print("loaded pretrained actor weights.")
            except Exception as e:
                print("failed to load pretrained actor weights:", e)
            try:
                self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
                print("loaded pretrained critic weights.")
            except Exception as e:
                print("failed to load pretrained critic weights:", e)
            try:
                self.actor_optim.load_state_dict(torch.load(actor_opt_path, map_location=self.device))
                print("loaded pretrained actor optimizer state.")
            except Exception as e:
                print("failed to load pretrained actor optimizer state:", e)
            try:
                self.critic_optim.load_state_dict(torch.load(critic_opt_path, map_location=self.device))
                print("loaded pretrained critic optimizer state.")
            except Exception as e:
                print("failed to load pretrained critic optimizer state:", e)
        self.logger = {'t_so_far': 0, 'i_so_far': 0, 'batch_lens': []}

    def _init_hyperparameters(self):
        self.max_timesteps_per_episode = 5000
        self.n_updates_per_iteration = 5
        self.lr = 0.001
        self.gamma = 0.9
        self.clip = 0.2
        # save every n iterations (can be set via cli)
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
                    mask[x - 50] = 0
            for x in range(50, 54):
                mask[x] = 0
            mask[54] = 1
        else:
            for x in range(0, 50):
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
        wins = 0
        for ep in range(num_episodes):
            print(f"  generating training episode {ep+1}/{num_episodes}...")
            g = game.Game([])
            agents = []
            rl_agent = RL_SupervisedAgent(g, self.actor, self.device)
            # training mode: ensure eval_mode is false
            rl_agent.eval_mode = False
            import randomAgent
            for _ in range(1):
                agents.append(randomAgent.Agent())
            agents.append(rl_agent)
            g.players = agents
            g.num_players = len(agents)
            g.reset()
            rl_agent.prev_hand_count = len(rl_agent.mydeck.cards)
            rl_agent.reset_episode()
            g.auto_simulate(max_turns=500)
            if len(rl_agent.episode_obs) > 0:
                if g.winner == g.players.index(rl_agent):
                    rl_agent.episode_rewards[-1] += 1.0
                    wins += 1
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
        win_rate = (wins / num_episodes) * 100
        print(f"[data generation] complete. win rate: {win_rate:.2f}%\n")
        batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float, device=self.device)
        return batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_lens

    def evaluate_policy(self, num_episodes=10):
        # set models to eval mode and force greedy behavior in agent
        self.actor.eval()
        self.critic.eval()
        print(f"\n[evaluation] starting evaluation for {num_episodes} episode(s)...")
        wins = 0
        total_turns = 0
        for ep in range(num_episodes):
            g = game.Game([])
            agents = []
            rl_agent = RL_SupervisedAgent(g, self.actor, self.device)
            # set evaluation flag to force near-greedy actions
            rl_agent.eval_mode = True
            rl_agent.reset_episode()
            import smartAgent
            for _ in range(1):
                agents.append(smartAgent.Agent())
            agents.append(rl_agent)
            g.players = agents
            g.num_players = len(agents)
            g.reset()
            rl_agent.prev_hand_count = len(rl_agent.mydeck.cards)
            g.auto_simulate(max_turns=500)
            total_turns += g.turn_count
            if g.winner is not None and g.winner == g.players.index(rl_agent):
                wins += 1
            print(f"  eval episode {ep+1}/{num_episodes} finished: turns = {g.turn_count}, winner = {g.winner}")
        win_rate = (wins / num_episodes) * 100
        avg_turns = total_turns / num_episodes
        print(f"[evaluation] complete. win rate: {win_rate:.2f}%, avg turns: {avg_turns:.2f}\n")
        # revert models back to training mode
        self.actor.train()
        self.critic.train()

    def learn(self, num_iterations=10, episodes_per_iter=10, temperature=1.0, eval_freq=5, eval_episodes=10):
        print(f"\n[training] starting training for {num_iterations} iteration(s), {episodes_per_iter} episode(s) per iteration.\n")
        for it in range(num_iterations):
            print(f"[training] iteration {it+1}/{num_iterations} started...")
            batch_obs, batch_actions, batch_log_probs, batch_rewards, ep_lens = self.generate_data(episodes_per_iter, temperature)
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
                print(f"    [update {update+1}/{self.n_updates_per_iteration}] actor loss: {actor_loss.item():.5f}, critic loss: {critic_loss.item():.5f}")
            avg_ep_len = np.mean(ep_lens)
            print(f"[training] iteration {it+1} complete: avg episode length = {avg_ep_len:.2f}")
            if (it+1) % eval_freq == 0:
                self.evaluate_policy(num_episodes=eval_episodes)
            # save checkpoints with iteration info and enforce max saves
            if (it+1) % self.save_freq == 0:
                ckpt_iter = it + 1
                actor_file = f"./ppo_actor_{ckpt_iter}.pth"
                critic_file = f"./ppo_critic_{ckpt_iter}.pth"
                actor_opt_file = f"./ppo_actor_opt_{ckpt_iter}.pth"
                critic_opt_file = f"./ppo_critic_opt_{ckpt_iter}.pth"
                torch.save(self.actor.state_dict(), actor_file)
                torch.save(self.critic.state_dict(), critic_file)
                torch.save(self.actor_optim.state_dict(), actor_opt_file)
                torch.save(self.critic_optim.state_dict(), critic_opt_file)
                self.saved_checkpoints.append(ckpt_iter)
                if len(self.saved_checkpoints) > self.max_saves:
                    old_ckpt = self.saved_checkpoints.pop(0)
                    old_files = [f"./ppo_actor_{old_ckpt}.pth", f"./ppo_critic_{old_ckpt}.pth",
                                 f"./ppo_actor_opt_{old_ckpt}.pth", f"./ppo_critic_opt_{old_ckpt}.pth"]
                    for file in old_files:
                        if os.path.exists(file):
                            os.remove(file)
        print("[training] training complete.\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ppo rl training for pretrained supervised agent with shaping rewards and evaluation against a randomagent")
    parser.add_argument("--iterations", type=int, default=1, help="number of training iterations")
    parser.add_argument("--episodes", type=int, default=100, help="episodes per iteration")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for action selection during training")
    parser.add_argument("--eval_freq", type=int, default=1, help="perform evaluation every n iterations")
    parser.add_argument("--eval_episodes", type=int, default=1000, help="number of evaluation episodes")
    parser.add_argument("--save_freq", type=int, default=1, help="save checkpoint every n iterations")
    parser.add_argument("--max_saves", type=int, default=10, help="maximum number of checkpoint saves to keep")
    args = parser.parse_args()

    ppo = PPO_Supervised(use_pretrained=True,
                         actor_path="ppo_actor.pth",
                         critic_path="ppo_critic.pth",
                         actor_opt_path="ppo_actor_opt.pth",
                         critic_opt_path="ppo_critic_opt.pth",
                         max_saves=args.max_saves)
    ppo.save_freq = args.save_freq
    ppo.learn(num_iterations=args.iterations, episodes_per_iter=args.episodes, temperature=args.temperature,
              eval_freq=args.eval_freq, eval_episodes=args.eval_episodes)


if __name__ == "__main__":
    main()
