import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from tqdm import tqdm
import logging

import game
import card
import deck
from random import shuffle
from supervisedAgent import DenseSkipNet

# enable bf16 autocast support
import torch.cuda.amp as amp

# --- minimal rl agent wrapper using the supervised network ---
class RL_SupervisedAgent:
    def __init__(self, game_instance, actor, device):
        self.game = game_instance
        self.actor = actor  # ppo actor network (DenseSkipNet)
        self.device = device
        self.mydeck = deck.Deck([])
        self.type = "rl_supervised"
        self.prev_hand_count = 0
        self.lastaction = None
        # buffers for training samples
        self.episode_obs = []
        self.episode_logprobs = []
        self.episode_rewards = []
        self.episode_act = []
        # flag for evaluation mode; when true, forces greedy (near-zero temperature)
        self.eval_mode = False
        # new flag: if false, no data will be collected
        self.collect_data = True

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
        temperature = 0.8
        if self.eval_mode:
            temperature = 1e-5
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float, device=self.device)
        obs = obs.to(self.device)
        obs_batched = obs.unsqueeze(0)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.actor(obs_batched).squeeze(0)
            mask = self.get_action_mask(obs)
            masked_logits = torch.where(mask.bool(), logits, torch.tensor(-1e15, device=self.device))
            scaled_logits = masked_logits / temperature
            dist = Categorical(logits=scaled_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            if action.item() == 54:
                if self.eval_mode:
                    logging.warning("Agent passed in eval.")
                else:
                    logging.warning("Agent passed in train.")

            return (None, log_prob) if action.item() == 54 else (action.item(), log_prob)

    def playCard(self, current_sort, current_true_number, temperature=1.0):
        observation = self.obs()
        legal_mask = self.get_action_mask(observation)
        if int(legal_mask.sum().item()) == 1:
            return None
        observation = observation.to(self.device)
        if self.eval_mode:
            temperature = 1e-3
        action, log_prob = self.act_rl(observation, temperature)
        if self.collect_data:
            self.episode_obs.append(observation.detach().cpu().numpy())
            self.episode_logprobs.append(log_prob.detach().cpu().item() if log_prob is not None else None)
            self.episode_rewards.append(-0.1 if action is None else 0.0)
            self.episode_act.append(action if action is not None else 54)
        return card.Card(action) if action is not None else None

    def changeSort(self):
        observation = self.obs()
        observation[105] = 1
        observation = observation.to(self.device)
        mask = self.get_action_mask(observation)
        temp = 1e-3 if self.eval_mode else 1.0
        action, log_prob = self.act_rl(observation, temperature=temp)
        if self.collect_data:
            self.episode_obs.append(observation.detach().cpu().numpy())
            self.episode_logprobs.append(log_prob.detach().cpu().item() if log_prob is not None else None)
            self.episode_rewards.append(0)
            self.episode_act.append(action)
        return card.sorts[action - 50]

    def addCard(self, _card):
        self.mydeck.cards.append(_card)

    def remove(self, _card):
        for c in self.mydeck.cards:
            if c.number == _card.number:
                self.mydeck.cards.remove(c)
                break

    def compute_shaping_reward(self, action):
        current_count = len(self.mydeck.cards)
        reward = 0.0
        if self.prev_hand_count:
            diff = self.prev_hand_count - current_count
            if diff > 0:
                reward += 0.05 * diff
            elif diff < 0:
                reward += 0.02 * diff
        self.prev_hand_count = current_count
        return reward


# --- student critic network ---
class StudentCritic(nn.Module):
    """
    a student critic network: a larger variant of denseskipnet with output_dim=1.
    here we use 8 hidden layers of 256 units each.
    """
    def __init__(self, input_dim=121, output_dim=1):
        super(StudentCritic, self).__init__()
        hidden_dims = [512] * 8
        self.model = DenseSkipNet(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    def forward(self, x):
        return self.model(x)


# --- student network ---
class StudentNet(nn.Module):
    """
    student network: a larger variant of denseskipnet with an extra layer and doubled hidden widths.
    here we use 8 hidden layers of 256 units each.
    """
    def __init__(self, input_dim=121, output_dim=55):
        super(StudentNet, self).__init__()
        hidden_dims = [256] * 8
        self.model = DenseSkipNet(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    def forward(self, x):
        return self.model(x)

def compute_gae(rewards, values, gamma_=0.97, lam=0.95):
    """
    compute generalized advantage estimation.
    
    args:
        rewards (tensor): 1d tensor of rewards from a rollout.
        values (tensor): 1d tensor of value estimates corresponding to the rewards.
        gamma (float): discount factor.
        lam (float): gae parameter controlling bias-variance tradeoff.
        
    returns:
        advantages (tensor): computed advantage estimates.
        returns (tensor): computed returns (advantages + values).
    """
    advantages = torch.zeros_like(rewards)
    last_adv = 0.0
    # assume values is same length as rewards; append one zero for bootstrap if needed
    for t in reversed(range(len(rewards))):
        next_val = values[t + 1] if t < len(rewards) - 1 else 0.0
        delta = rewards[t] + gamma_ * next_val - values[t]
        last_adv = delta + gamma_ * lam * last_adv
        advantages[t] = last_adv
    returns = advantages + values
    return advantages, returns

# --- ppo for the supervised agent ---
class PPO_Supervised:
    def __init__(self, use_pretrained=True,
                 actor_path="distilled_rl_agent_old.pth",
                 critic_path="ppo_critic_new.pth",
                 actor_opt_path="ppo_actor_opt.pth",
                 critic_opt_path="ppo_critic_opt_new.pth",
                 old_save = ['28run1','56run1', '28run2', '42run2', '56run2', '30run3', '80run3', '100run3'],
                 max_saves=5):
        self._init_hyperparameters()
        self.max_saves = max_saves
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = StudentNet(input_dim=121, output_dim=55).to(self.device)
        self.critic = StudentCritic().to(self.device)
        # add the base version to the saved list
        self.saved_checkpoints = old_save
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=self.lr, weight_decay=3e-6)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=self.lr, weight_decay=1e-5)
        if use_pretrained:
            try:
                self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
                print("loaded pretrained actor weights.")
                self.saved_checkpoints.append(0)
                torch.save(self.actor.state_dict(), "./ppo_actor_0.pth")
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
        self.scaler = torch.amp.GradScaler('cuda')

    def _init_hyperparameters(self):
        self.max_timesteps_per_episode = 5000
        self.n_updates_per_iteration = 3
        self.lr = 0.00003
        self.clip = 0.1
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

    def generate_data(self, num_episodes=10, temperature=1.0, agent_sets=[["smart", "rl"]]):
        print(f"\n[data generation] starting data generation for {num_episodes} episode(s)...")
        batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_lens = [], [], [], [], []
        global_wins = 0
        num_sets = len(agent_sets)
        for i, agent_set in enumerate(agent_sets):
            if not any(typ.lower() in ['rl', 'self'] for typ in agent_set):
                raise ValueError("each agent set must include at least one rl agent")
            if i < num_sets - 1:
                episodes_for_set = num_episodes // num_sets
            else:
                episodes_for_set = num_episodes - (num_sets - 1) * (num_episodes // num_sets)
            print(f"[data generation] collecting {episodes_for_set} episode(s) for agent set {i+1}: {agent_set}")
            set_wins = 0
            set_draws = 0
            for ep in tqdm(range(episodes_for_set), desc="data gen episodes", leave=True):
                g = game.Game([])
                agents = []
                rl_agents = []
                past_version_used = False  # ensure at most one past version per game
                pastProbability = np.random.random() < 0.2
                for typ in agent_set:
                    if typ.lower() in ['rl', 'self']:
                        if (not past_version_used) and (pastProbability) and (len(self.saved_checkpoints) > 0):
                            chosen_ckpt = np.random.choice(self.saved_checkpoints)
                            past_actor = StudentNet(input_dim=121, output_dim=55).to(self.device)
                            past_actor.load_state_dict(torch.load(f"./ppo_actor_{chosen_ckpt}.pth", map_location=self.device))
                            a = RL_SupervisedAgent(g, past_actor, self.device)
                            a.collect_data = False  # do not collect data from past version
                            a.eval_mode = True
                            past_version_used = True
                        else:
                            a = RL_SupervisedAgent(g, self.actor, self.device)
                            a.collect_data = True
                            a.eval_mode = False
                        a.reset_episode()
                        agents.append(a)
                        if a.collect_data:
                            rl_agents.append(a)
                    elif typ.lower() == 'smart':
                        import smartAgent
                        agents.append(smartAgent.Agent())
                    elif typ.lower() == 'random':
                        import randomAgent
                        agents.append(randomAgent.Agent())
                    else:
                        print(f"unknown agent type: {typ}")
                shuffle(agents)
                g.players = agents
                g.num_players = len(agents)
                g.reset()
                for a in rl_agents:
                    a.prev_hand_count = len(a.mydeck.cards)
                g.auto_simulate(max_turns=500)
                for a in rl_agents:
                    if len(a.episode_obs) > 0:
                        if g.winner == g.players.index(a):
                            a.episode_rewards[-1] += 1.0
                            set_wins += 1
                            global_wins += 1
                        elif g.winner is None:
                            a.episode_rewards[-1] += 0.0
                            set_draws += 1
                        else:
                            a.episode_rewards[-1] += -1.0
                        batch_obs.extend(a.episode_obs)
                        batch_actions.extend(a.episode_act)
                        batch_log_probs.extend(a.episode_logprobs)
                        batch_rewards.extend(a.episode_rewards)
                        batch_lens.append(len(a.episode_obs))
            set_win_rate = (set_wins / episodes_for_set) * 100
            print(f"[data generation] agent set {i+1} win rate: {set_win_rate:.2f}% draw rate: {set_draws/episodes_for_set * 100:.2f}% over {episodes_for_set} episodes")
        global_win_rate = (global_wins / num_episodes) * 100
        print(f"[data generation] complete. overall win rate: {global_win_rate:.2f}%\n")
        batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=self.device)
        batch_actions = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float, device=self.device)
        return batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_lens

    def evaluate_policy(self, num_episodes=10, agent_sets=[["smart", "rl"]]):
        self.actor.eval()
        self.critic.eval()
        print(f"\n[evaluation] starting evaluation for {num_episodes} episode(s)...")
        total_turns = 0
        global_wins = 0
        num_sets = len(agent_sets)
        for i, agent_set in enumerate(agent_sets):
            if not any(typ.lower() in ['rl', 'self'] for typ in agent_set):
                raise ValueError("each agent set must include at least one rl agent")
            if i < num_sets - 1:
                episodes_for_set = num_episodes // num_sets
            else:
                episodes_for_set = num_episodes - (num_sets - 1) * (num_episodes // num_sets)
            print(f"[evaluation] running {episodes_for_set} episode(s) for agent set {i+1}: {agent_set}")
            set_wins = 0
            for ep in tqdm(range(episodes_for_set), desc="evaluation episodes", leave=True):
                g = game.Game([])
                agents = []
                rl_agents = []
                past_version_used = False  # ensure at most one past version in evaluation as well
                for typ in agent_set:
                    if typ.lower() in ['rl', 'self']:
                        if (False):
                            chosen_ckpt = np.random.choice(self.saved_checkpoints)
                            past_actor = StudentNet(input_dim=121, output_dim=55).to(self.device)
                            past_actor.load_state_dict(torch.load(f"./ppo_actor_{chosen_ckpt}.pth", map_location=self.device))
                            a = RL_SupervisedAgent(g, past_actor, self.device)
                            a.collect_data = False
                            a.eval_mode = True
                            past_version_used = True
                        else:
                            a = RL_SupervisedAgent(g, self.actor, self.device)
                            a.collect_data = True
                            a.eval_mode = True
                        a.reset_episode()
                        agents.append(a)
                        rl_agents.append(a)
                    elif typ.lower() == 'smart':
                        import smartAgent
                        agents.append(smartAgent.Agent())
                    elif typ.lower() == 'random':
                        import randomAgent
                        agents.append(randomAgent.Agent())
                    else:
                        print(f"unknown agent type: {typ}")
                g.players = agents
                g.num_players = len(agents)
                g.reset()
                for a in rl_agents:
                    a.prev_hand_count = len(a.mydeck.cards)
                g.auto_simulate(max_turns=500)
                total_turns += g.turn_count
                for a in rl_agents:
                    if g.winner is not None and g.winner == g.players.index(a):
                        set_wins += 1
                        global_wins += 1
            set_win_rate = (set_wins / episodes_for_set) * 100
            print(f"[evaluation] agent set {i+1} win rate: {set_win_rate:.2f}% over {episodes_for_set} episodes")
        global_win_rate = (global_wins / num_episodes) * 100
        avg_turns = total_turns / num_episodes
        print(f"[evaluation] complete. overall win rate: {global_win_rate:.2f}%, avg turns: {avg_turns:.2f}\n")
        self.actor.train()
        self.critic.train()

    def learn(self, num_iterations=10, episodes_per_iter=10, temperature=1.0, eval_freq=5, eval_episodes=10, agent_sets=[["smart", "rl"]]):
        print(f"\n[training] starting training for {num_iterations} iteration(s), {episodes_per_iter} episode(s) per iteration.\n")
        for it in tqdm(range(num_iterations), desc="training iterations"):
            self.actor.train()
            tqdm.write(f"[training] iteration {it+1}/{num_iterations} started...")
            batch_obs, batch_actions, batch_log_probs, batch_rewards, ep_lens = self.generate_data(
                num_episodes=episodes_per_iter, temperature=temperature, agent_sets=agent_sets)
            # assuming batch_rewards is a 1d tensor and values = self.critic(batch_obs).squeeze()
            advantages, rtgs = compute_gae(batch_rewards, self.critic(batch_obs).squeeze(), lam=0.95)
            advantages = advantages.detach()
            rtgs = rtgs.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            for update in tqdm(range(self.n_updates_per_iteration), desc="policy updates", leave=False):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
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
                    total_loss = actor_loss + 0.8 * critic_loss

                    # compute approximate kl divergence: expectation_{a~old}[log(old_prob) - log(new_prob)]
                    kl_div = torch.mean(batch_log_probs - new_log_probs)

                self.actor_optim.zero_grad()
                self.critic_optim.zero_grad()
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.actor_optim)
                self.scaler.step(self.critic_optim)
                self.scaler.update()
                tqdm.write(f"    [update {update+1}/{self.n_updates_per_iteration}] actor loss: {actor_loss.item():.5f}, critic loss: {critic_loss.item():.5f}, kl: {kl_div.item():.5f}")
            avg_ep_len = np.mean(ep_lens)
            tqdm.write(f"[training] iteration {it+1} complete: avg episode length = {avg_ep_len:.2f}")
            if (it+1) % eval_freq == 0:
                self.evaluate_policy(num_episodes=eval_episodes)
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


def parse_agent_sets(agent_sets_str):
    # expected format: "smart,rl;random,rl" etc.
    sets = []
    for set_str in agent_sets_str.split(';'):
        agents = [a.strip() for a in set_str.split(',') if a.strip()]
        if agents:
            sets.append(agents)
    return sets


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="ppo rl training for pretrained supervised agent with shaping rewards and evaluation against specified opponent sets")
    parser.add_argument("--iterations", type=int, default=1000, help="number of training iterations")
    parser.add_argument("--episodes", type=int, default=1000, help="episodes per iteration")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for action selection during training")
    parser.add_argument("--eval_freq", type=int, default=10, help="perform evaluation every n iterations")
    parser.add_argument("--eval_episodes", type=int, default=4000, help="number of evaluation episodes")
    parser.add_argument("--save_freq", type=int, default=10, help="save checkpoint every n iterations")
    parser.add_argument("--max_saves", type=int, default=20, help="maximum number of checkpoint saves to keep")
    parser.add_argument("--agent_sets", type=str, default="smart,rl",
                        help="semicolon-separated sets of comma-separated agent types (each set must include at least one rl agent), e.g. \"smart,rl;random,rl\"")
    args = parser.parse_args()

    agent_sets = parse_agent_sets(args.agent_sets)
    # validate each set includes at least one rl agent
    for aset in agent_sets:
        if not any(a.lower() in ['rl', 'self'] for a in aset):
            raise ValueError("each agent set must include at least one rl agent")
    
    print("training on the following agent sets:")
    for idx, aset in enumerate(agent_sets):
        print(f" set {idx+1}: {aset}")
    
    ppo = PPO_Supervised(use_pretrained=True,
                         actor_path="ppo_actor_100run3.pth",
                         critic_path="ppo_critic_100run3.pth",
                         actor_opt_path="ppo_actor_opt_100run3.pth",
                         critic_opt_path="ppo_critic_opt_100run3.pth",
                         max_saves=args.max_saves)
    ppo.save_freq = args.save_freq
    ppo.learn(num_iterations=args.iterations, episodes_per_iter=args.episodes, temperature=args.temperature,
              eval_freq=args.eval_freq, eval_episodes=args.eval_episodes, agent_sets=agent_sets)


if __name__ == "__main__":
    main()
