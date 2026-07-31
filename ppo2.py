import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from contextlib import nullcontext
from tqdm import tqdm
import logging

import game
import card
import deck
from random import shuffle
from supervisedAgent import DenseSkipNet
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
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    POLICY_RANDOM,
    POLICY_RL,
    POLICY_SMART,
    collect_batched_rollout,
)

DEFAULT_ROLLOUT_TEMPERATURE = 0.8


def autocast_for(device):
    """Use the fastest supported CUDA training dtype, or no autocast on CPU."""
    if device.type != "cuda":
        return nullcontext()
    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )
    return torch.amp.autocast(device_type="cuda", dtype=dtype)


def build_action_masks(observations):
    """Build policy masks on the same device as one observation or a batch."""
    if not isinstance(observations, torch.Tensor):
        observations = torch.as_tensor(observations, dtype=torch.float)
    if observations.shape[-1] != OBSERVATION_SIZE:
        raise ValueError(
            f"expected observations with {OBSERVATION_SIZE} features, "
            f"got {observations.shape[-1]}"
        )

    mask_shape = (*observations.shape[:-1], ACTION_COUNT)
    play_masks = torch.zeros(
        mask_shape,
        device=observations.device,
        dtype=torch.bool,
    )
    card_action_count = COMPATIBLE_END - COMPATIBLE_START
    play_masks[..., :card_action_count] = (
        observations[..., COMPATIBLE_START:COMPATIBLE_END] >= 1
    )
    play_masks[..., PASS_ACTION] = True

    suit_masks = torch.zeros_like(play_masks)
    suit_masks[..., SUIT_ACTION_START:SUIT_ACTION_END] = True
    is_change_sort = observations[..., MODE_INDEX] != 0
    return torch.where(is_change_sort.unsqueeze(-1), suit_masks, play_masks)

# --- minimal rl agent wrapper using the supervised network ---
class RL_SupervisedAgent:
    def __init__(
        self,
        game_instance,
        actor,
        device,
        temperature=DEFAULT_ROLLOUT_TEMPERATURE,
    ):
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self.game = game_instance
        self.actor = actor  # ppo actor network (DenseSkipNet)
        self.device = device
        self.temperature = temperature
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
        obs = torch.zeros(OBSERVATION_SIZE, dtype=torch.float)
        for c in self.mydeck.cards:
            obs[c.number] += 1
            if self.game.penalty_amount > 0:
                is_playable = c.truenumber == self.game.current_true_number
            else:
                is_playable = c.compatible(
                    self.game.current_sort,
                    self.game.current_true_number,
                )
            if is_playable:
                obs[c.number + 50] += 1
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
        return build_action_masks(obs)

    def act_rl(self, obs, temperature=None):
        temperature = self.temperature if temperature is None else temperature
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if self.eval_mode:
            temperature = 1e-5
        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs, dtype=torch.float)
        device_obs = obs.to(self.device)
        obs_batched = device_obs.unsqueeze(0)
        with torch.inference_mode(), autocast_for(self.device):
            logits = self.actor(obs_batched).squeeze(0)
            mask = self.get_action_mask(device_obs)
            masked_logits = logits.masked_fill(
                ~mask,
                torch.finfo(logits.dtype).min,
            )
            scaled_logits = masked_logits / temperature
            dist = Categorical(logits=scaled_logits, validate_args=False)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action_value, log_prob_value = torch.stack(
                (action.float(), log_prob.float())
            ).cpu().tolist()
        action_value = int(action_value)
        if action_value == PASS_ACTION:
            logging.debug("RL agent passed.")
            return None, log_prob_value
        return action_value, log_prob_value

    def playCard(self, current_sort, current_true_number, temperature=None):
        observation = self.obs()
        if not torch.any(
            observation[COMPATIBLE_START:COMPATIBLE_END] >= 1
        ):
            return None
        action, log_prob = self.act_rl(observation, temperature)
        if self.collect_data:
            self.episode_obs.append(observation)
            self.episode_logprobs.append(log_prob)
            self.episode_rewards.append(-0.1 if action is None else 0.0)
            self.episode_act.append(
                action if action is not None else PASS_ACTION
            )
        return card.Card(action) if action is not None else None

    def changeSort(self):
        observation = self.obs()
        observation[MODE_INDEX] = 1
        action, log_prob = self.act_rl(observation)
        if self.collect_data:
            self.episode_obs.append(observation)
            self.episode_logprobs.append(log_prob)
            self.episode_rewards.append(0)
            self.episode_act.append(action)
        return card.sorts[action - SUIT_ACTION_START]

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
    def __init__(self, input_dim=OBSERVATION_SIZE, output_dim=1):
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
    def __init__(self, input_dim=OBSERVATION_SIZE, output_dim=ACTION_COUNT):
        super(StudentNet, self).__init__()
        hidden_dims = [256] * 8
        self.model = DenseSkipNet(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    def forward(self, x):
        return self.model(x)

def compute_gae(
    rewards,
    values,
    episode_lengths=None,
    gamma_=0.97,
    lam=0.95,
):
    """
    compute generalized advantage estimation.
    
    args:
        rewards (tensor): 1d tensor of rewards from a rollout.
        values (tensor): 1d tensor of value estimates corresponding to the rewards.
        gamma (float): discount factor.
        episode_lengths (sequence): number of samples in each episode.
        lam (float): gae parameter controlling bias-variance tradeoff.
        
    returns:
        advantages (tensor): computed advantage estimates.
        returns (tensor): computed returns (advantages + values).
    """
    if rewards.ndim != 1 or values.ndim != 1:
        raise ValueError("rewards and values must be one-dimensional")
    if len(rewards) != len(values):
        raise ValueError("rewards and values must have the same length")
    if episode_lengths is None:
        episode_lengths = [len(rewards)]
    if sum(episode_lengths) != len(rewards):
        raise ValueError("episode lengths must cover every reward exactly once")

    lengths = torch.as_tensor(
        episode_lengths,
        dtype=torch.long,
        device=values.device,
    )
    if torch.any(lengths < 0):
        raise ValueError("episode lengths cannot be negative")
    if len(rewards) == 0:
        return torch.zeros_like(rewards), values.clone()

    episode_ids = torch.repeat_interleave(
        torch.arange(len(lengths), device=values.device),
        lengths,
    )
    episode_starts = torch.cumsum(lengths, dim=0) - lengths
    time_indices = (
        torch.arange(len(rewards), device=values.device)
        - episode_starts[episode_ids]
    )
    max_length = int(lengths.max().item())
    padded_rewards = rewards.new_zeros((len(lengths), max_length))
    padded_values = values.new_zeros((len(lengths), max_length))
    padded_rewards[episode_ids, time_indices] = rewards
    padded_values[episode_ids, time_indices] = values
    padded_advantages = values.new_zeros((len(lengths), max_length))
    last_advantage = values.new_zeros(len(lengths))
    next_values = values.new_zeros(len(lengths))
    # One vector operation per time position replaces one scalar tensor
    # operation per collected decision.
    for time_index in range(max_length - 1, -1, -1):
        active = time_index < lengths
        delta = (
            padded_rewards[:, time_index]
            + gamma_ * next_values
            - padded_values[:, time_index]
        )
        last_advantage = torch.where(
            active,
            delta + gamma_ * lam * last_advantage,
            torch.zeros_like(last_advantage),
        )
        padded_advantages[:, time_index] = last_advantage
        next_values = torch.where(
            active,
            padded_values[:, time_index],
            torch.zeros_like(next_values),
        )
    advantages = padded_advantages[episode_ids, time_indices]
    returns = advantages + values
    return advantages, returns

# --- ppo for the supervised agent ---
class PPO_Supervised:
    def __init__(self, use_pretrained=True,
                 actor_path="distilled_rl_agent_old.pth",
                 critic_path="ppo_critic_new.pth",
                 actor_opt_path="ppo_actor_opt.pth",
                 critic_opt_path="ppo_critic_opt_new.pth",
                 old_save=('28run1', '56run1', '28run2', '42run2',
                           '56run2', '30run3', '80run3', '100run3'),
                 max_saves=5,
                 rollout_device="auto",
                 rollout_batch_size=256,
                 training_actor_device="auto",
                 compile_updates=None):
        self._init_hyperparameters()
        self.max_saves = max_saves
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if rollout_device == "auto":
            rollout_device = (
                "cpu" if self.device.type == "cuda" else self.device.type
            )
        self.rollout_device = torch.device(rollout_device)
        if self.rollout_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA rollout requested but CUDA is unavailable")

        if training_actor_device == "auto":
            training_actor_device = self.device.type
        self.actor_device = torch.device(training_actor_device)
        if self.actor_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA actor training requested but CUDA is unavailable")
        self.actor = StudentNet(
            input_dim=OBSERVATION_SIZE,
            output_dim=ACTION_COUNT,
        ).to(self.actor_device)
        self.critic = StudentCritic().to(self.device)
        if self.actor_device == self.rollout_device:
            self.rollout_actor = self.actor
        else:
            # Rollout inference stays on CPU, where batch sizes shrink toward
            # the tail of each game batch.  PPO updates use the accelerator.
            self.rollout_actor = StudentNet(
                input_dim=OBSERVATION_SIZE,
                output_dim=ACTION_COUNT,
            ).to(self.rollout_device)
        if compile_updates is None:
            compile_updates = (
                self.actor_device.type == "cuda"
                and self.device.type == "cuda"
            )
        self.compile_updates = bool(compile_updates)
        if self.compile_updates and (
            self.actor_device.type != "cuda"
            or self.device.type != "cuda"
        ):
            raise ValueError("compiled updates require CUDA actor and critic")
        if self.compile_updates:
            self._actor_update_model = torch.compile(
                self.actor,
                dynamic=True,
                mode="reduce-overhead",
            )
            self._critic_update_model = torch.compile(
                self.critic,
                dynamic=True,
                mode="reduce-overhead",
            )
        else:
            self._actor_update_model = self.actor
            self._critic_update_model = self.critic
        if rollout_batch_size < 1:
            raise ValueError("rollout_batch_size must be at least one")
        self.rollout_batch_size = int(rollout_batch_size)
        self._checkpoint_actor_cache = {}
        self.last_rollout_metrics = {}
        # add the base version to the saved list
        self.saved_checkpoints = list(old_save)
        self.actor_optim = optim.AdamW(
            self.actor.parameters(),
            lr=self.lr,
            weight_decay=3e-6,
            fused=self.actor_device.type == "cuda",
        )
        self.critic_optim = optim.AdamW(
            self.critic.parameters(),
            lr=self.lr,
            weight_decay=1e-5,
            fused=self.device.type == "cuda",
        )
        if use_pretrained:
            try:
                self.actor.load_state_dict(
                    torch.load(actor_path, map_location=self.actor_device)
                )
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
                self.actor_optim.load_state_dict(
                    torch.load(actor_opt_path, map_location=self.actor_device)
                )
                print("loaded pretrained actor optimizer state.")
            except Exception as e:
                print("failed to load pretrained actor optimizer state:", e)
            try:
                self.critic_optim.load_state_dict(torch.load(critic_opt_path, map_location=self.device))
                print("loaded pretrained critic optimizer state.")
            except Exception as e:
                print("failed to load pretrained critic optimizer state:", e)
        self.logger = {'t_so_far': 0, 'i_so_far': 0, 'batch_lens': []}
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=self.device.type == "cuda",
        )
        self._sync_rollout_actor()

    def _init_hyperparameters(self):
        self.max_timesteps_per_episode = 5000
        self.n_updates_per_iteration = 3
        self.lr = 0.00003
        self.clip = 0.1
        self.save_freq = 1

    def _sync_rollout_actor(self):
        self.actor.eval()
        if self.rollout_actor is not self.actor:
            self.rollout_actor.load_state_dict(self.actor.state_dict())
        self.rollout_actor.eval()

    def actor_update_autocast(self):
        """Match rollout numerics when training and rollout devices differ."""
        if self.actor_device != self.rollout_device:
            return nullcontext()
        return autocast_for(self.actor_device)

    def actor_update_forward(self, observations):
        return self._actor_update_model(observations)

    def critic_update_forward(self, observations):
        return self._critic_update_model(observations)

    def _available_actor_checkpoints(self):
        return [
            checkpoint
            for checkpoint in self.saved_checkpoints
            if os.path.exists(f"./ppo_actor_{checkpoint}.pth")
        ]

    def _checkpoint_actor(self, checkpoint):
        """Load each immutable historical opponent at most once per process."""
        actor = self._checkpoint_actor_cache.get(checkpoint)
        if actor is None:
            actor = StudentNet(
                input_dim=OBSERVATION_SIZE,
                output_dim=ACTION_COUNT,
            ).to(self.rollout_device)
            actor.load_state_dict(
                torch.load(
                    f"./ppo_actor_{checkpoint}.pth",
                    map_location=self.rollout_device,
                )
            )
            actor.eval()
            self._checkpoint_actor_cache[checkpoint] = actor
        return actor

    def get_action_mask(self, obs):
        return build_action_masks(obs)

    def get_action_masks(self, observations):
        return build_action_masks(observations)

    def generate_data_sequential(
        self,
        num_episodes=10,
        temperature=DEFAULT_ROLLOUT_TEMPERATURE,
        agent_sets=None,
    ):
        if agent_sets is None:
            agent_sets = [["smart", "rl"]]
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        self._sync_rollout_actor()
        print(f"\n[data generation] starting data generation for {num_episodes} episode(s)...")
        batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_lens = [], [], [], [], []
        global_wins = 0
        environment_turns = 0
        completed_games = 0
        num_sets = len(agent_sets)
        available_checkpoints = self._available_actor_checkpoints()
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
                        if (
                            not past_version_used
                            and pastProbability
                            and available_checkpoints
                        ):
                            chosen_ckpt = np.random.choice(available_checkpoints)
                            past_actor = StudentNet(
                                input_dim=OBSERVATION_SIZE,
                                output_dim=ACTION_COUNT,
                            ).to(self.rollout_device)
                            past_actor.load_state_dict(
                                torch.load(
                                    f"./ppo_actor_{chosen_ckpt}.pth",
                                    map_location=self.rollout_device,
                                )
                            )
                            past_actor.eval()
                            a = RL_SupervisedAgent(
                                g,
                                past_actor,
                                self.rollout_device,
                                temperature=temperature,
                            )
                            a.collect_data = False  # do not collect data from past version
                            a.eval_mode = True
                            past_version_used = True
                        else:
                            a = RL_SupervisedAgent(
                                g,
                                self.rollout_actor,
                                self.rollout_device,
                                temperature=temperature,
                            )
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
                        agents.append(randomAgent.AgentRandom())
                    else:
                        print(f"unknown agent type: {typ}")
                shuffle(agents)
                g.players = agents
                g.num_players = len(agents)
                for a in rl_agents:
                    a.prev_hand_count = len(a.mydeck.cards)
                g.auto_simulate(max_turns=500)
                environment_turns += g.turn_count
                completed_games += 1
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
        if not batch_obs:
            raise RuntimeError("data generation produced no policy samples")
        self.last_rollout_metrics = {
            "environment_turns": environment_turns,
            "completed_games": completed_games,
            "policy_decisions": len(batch_actions),
            "inference_calls": len(batch_actions),
            "peak_buffer_bytes": 0,
            "wins": global_wins,
            "draws": completed_games - global_wins,
        }
        batch_obs = torch.stack(batch_obs).to(
            device=self.actor_device,
            dtype=torch.float,
        )
        batch_actions = torch.as_tensor(
            batch_actions,
            dtype=torch.long,
            device=self.actor_device,
        )
        batch_log_probs = torch.as_tensor(
            batch_log_probs,
            dtype=torch.float,
            device=self.actor_device,
        )
        batch_rewards = torch.as_tensor(
            batch_rewards,
            dtype=torch.float,
            device=self.actor_device,
        )
        return batch_obs, batch_actions, batch_log_probs, batch_rewards, batch_lens

    def generate_data(
        self,
        num_episodes=10,
        temperature=DEFAULT_ROLLOUT_TEMPERATURE,
        agent_sets=None,
    ):
        """Collect rollouts in vectorized games with batched actor inference."""
        if agent_sets is None:
            agent_sets = [["smart", "rl"]]
        if num_episodes < 1:
            raise ValueError("num_episodes must be at least one")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if not agent_sets:
            raise ValueError("at least one agent set is required")
        self._sync_rollout_actor()

        all_observations = []
        all_actions = []
        all_log_probabilities = []
        all_rewards = []
        all_episode_lengths = []
        aggregate_metrics = {
            "environment_turns": 0,
            "completed_games": 0,
            "policy_decisions": 0,
            "inference_calls": 0,
            "peak_buffer_bytes": 0,
            "wins": 0,
            "draws": 0,
        }
        available_checkpoints = self._available_actor_checkpoints()
        current_key = "current"
        current_policy = ActorPolicy(
            actor=self.rollout_actor,
            device=self.rollout_device,
            temperature=temperature,
            collect_data=True,
        )

        episodes_per_set = [num_episodes // len(agent_sets)] * len(agent_sets)
        episodes_per_set[-1] += num_episodes - sum(episodes_per_set)
        for set_index, (agent_set, set_episodes) in enumerate(
            zip(agent_sets, episodes_per_set)
        ):
            normalized = [agent_type.lower() for agent_type in agent_set]
            if not any(agent_type in ("rl", "self") for agent_type in normalized):
                raise ValueError("each agent set must include at least one rl agent")
            unknown = set(normalized) - {"rl", "self", "smart", "random"}
            if unknown:
                raise ValueError(
                    "unknown agent type(s): " + ", ".join(sorted(unknown))
                )
            if set_episodes == 0:
                continue
            print(
                "[data generation] collecting "
                f"{set_episodes} episode(s) for agent set "
                f"{set_index + 1}: {agent_set}"
            )

            remaining = set_episodes
            while remaining:
                batch_games = min(self.rollout_batch_size, remaining)
                player_count = len(agent_set)
                policy_kinds = np.empty(
                    (batch_games, player_count),
                    dtype=np.int8,
                )
                actor_keys = np.empty(
                    (batch_games, player_count),
                    dtype=object,
                )
                actor_policies = {current_key: current_policy}

                for game_index in range(batch_games):
                    entries = []
                    use_past = (
                        bool(available_checkpoints)
                        and sum(
                            agent_type in ("rl", "self")
                            for agent_type in normalized
                        ) > 1
                        and np.random.random() < 0.2
                    )
                    past_version_used = False
                    for agent_type in normalized:
                        if agent_type in ("rl", "self"):
                            if use_past and not past_version_used:
                                checkpoint = np.random.choice(
                                    available_checkpoints
                                )
                                checkpoint_key = f"checkpoint:{checkpoint}"
                                entries.append((POLICY_RL, checkpoint_key))
                                if checkpoint_key not in actor_policies:
                                    actor_policies[checkpoint_key] = ActorPolicy(
                                        actor=self._checkpoint_actor(checkpoint),
                                        device=self.rollout_device,
                                        temperature=1e-5,
                                        collect_data=False,
                                        greedy=True,
                                    )
                                past_version_used = True
                            else:
                                entries.append((POLICY_RL, current_key))
                        elif agent_type == "smart":
                            entries.append((POLICY_SMART, None))
                        else:
                            entries.append((POLICY_RANDOM, None))
                    shuffle(entries)
                    for player_index, (kind, key) in enumerate(entries):
                        policy_kinds[game_index, player_index] = kind
                        actor_keys[game_index, player_index] = key

                seeds = np.random.randint(
                    0,
                    np.iinfo(np.uint32).max,
                    size=batch_games,
                    dtype=np.uint32,
                )
                engine = BatchedPesten(
                    policy_kinds,
                    actor_keys,
                    seeds,
                    max_turns=500,
                    # Sequential PPO constructs smartAgent.Agent() without the
                    # optional Game reference, so preserve that exact behavior.
                    smart_agents_have_game_reference=False,
                )
                result = collect_batched_rollout(
                    engine,
                    actor_policies,
                    output_device=self.actor_device,
                )
                all_observations.append(result.observations)
                all_actions.append(result.actions)
                all_log_probabilities.append(result.log_probabilities)
                all_rewards.append(result.rewards)
                all_episode_lengths.extend(result.episode_lengths)
                for metric in aggregate_metrics:
                    value = getattr(result, metric)
                    if metric == "peak_buffer_bytes":
                        aggregate_metrics[metric] = max(
                            aggregate_metrics[metric],
                            value,
                        )
                    else:
                        aggregate_metrics[metric] += value
                remaining -= batch_games

        self.last_rollout_metrics = aggregate_metrics
        print(
            "[data generation] complete: "
            f"{aggregate_metrics['completed_games']} games, "
            f"{aggregate_metrics['environment_turns']} turns, "
            f"{aggregate_metrics['policy_decisions']} collected decisions, "
            f"{aggregate_metrics['inference_calls']} actor batches"
        )
        return (
            torch.cat(all_observations),
            torch.cat(all_actions),
            torch.cat(all_log_probabilities),
            torch.cat(all_rewards),
            all_episode_lengths,
        )

    def evaluate_policy(self, num_episodes=10, agent_sets=None):
        if agent_sets is None:
            agent_sets = [["smart", "rl"]]
        self.actor.eval()
        self.critic.eval()
        self._sync_rollout_actor()
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
                for typ in agent_set:
                    if typ.lower() in ['rl', 'self']:
                        a = RL_SupervisedAgent(
                            g,
                            self.rollout_actor,
                            self.rollout_device,
                        )
                        a.collect_data = False
                        a.eval_mode = True
                        a.reset_episode()
                        agents.append(a)
                        rl_agents.append(a)
                    elif typ.lower() == 'smart':
                        import smartAgent
                        agents.append(smartAgent.Agent())
                    elif typ.lower() == 'random':
                        import randomAgent
                        agents.append(randomAgent.AgentRandom())
                    else:
                        print(f"unknown agent type: {typ}")
                g.players = agents
                g.num_players = len(agents)
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
        self.actor.eval()
        self.critic.train()

    def learn(
        self,
        num_iterations=10,
        episodes_per_iter=10,
        temperature=DEFAULT_ROLLOUT_TEMPERATURE,
        eval_freq=5,
        eval_episodes=10,
        agent_sets=None,
    ):
        if agent_sets is None:
            agent_sets = [["smart", "rl"]]
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        print(f"\n[training] starting training for {num_iterations} iteration(s), {episodes_per_iter} episode(s) per iteration.\n")
        for it in tqdm(range(num_iterations), desc="training iterations"):
            self.actor.eval()
            tqdm.write(f"[training] iteration {it+1}/{num_iterations} started...")
            batch_obs, batch_actions, batch_log_probs, batch_rewards, ep_lens = self.generate_data(
                num_episodes=episodes_per_iter, temperature=temperature, agent_sets=agent_sets)
            self.actor.eval()
            critic_obs = batch_obs.to(self.device)
            critic_rewards = batch_rewards.to(self.device)
            with torch.no_grad():
                advantages, rtgs = compute_gae(
                    critic_rewards,
                    self.critic_update_forward(critic_obs).squeeze(),
                    episode_lengths=ep_lens,
                    lam=0.95,
                )
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            actor_advantages = advantages.to(self.actor_device)
            masks = self.get_action_masks(batch_obs)
            for update in tqdm(range(self.n_updates_per_iteration), desc="policy updates", leave=False):
                with self.actor_update_autocast():
                    logits = self.actor_update_forward(batch_obs) / temperature
                    masked_logits = logits.masked_fill(
                        ~masks,
                        torch.finfo(logits.dtype).min,
                    )
                    dists = Categorical(
                        logits=masked_logits,
                        validate_args=False,
                    )
                    new_log_probs = dists.log_prob(batch_actions)
                    ratios = torch.exp(new_log_probs - batch_log_probs)
                    surr1 = ratios * actor_advantages
                    surr2 = (
                        torch.clamp(
                            ratios,
                            1 - self.clip,
                            1 + self.clip,
                        )
                        * actor_advantages
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()
                    # compute approximate kl divergence: expectation_{a~old}[log(old_prob) - log(new_prob)]
                    kl_div = torch.mean(batch_log_probs - new_log_probs)

                if update == 0:
                    max_initial_ratio_error = (
                        ratios.detach().sub(1).abs().max().item()
                    )
                    if max_initial_ratio_error > 0.02:
                        raise RuntimeError(
                            "rollout/update policy mismatch: unchanged-policy "
                            f"ratio error reached {max_initial_ratio_error:.5f}"
                        )

                with autocast_for(self.device):
                    values = self.critic_update_forward(critic_obs).squeeze()
                    critic_loss = nn.functional.mse_loss(values, rtgs)

                self.actor_optim.zero_grad(set_to_none=True)
                self.critic_optim.zero_grad(set_to_none=True)
                shared_cuda_update = (
                    self.actor_device == self.device
                    and self.device.type == "cuda"
                )
                if shared_cuda_update:
                    self.scaler.scale(
                        actor_loss + 0.8 * critic_loss
                    ).backward()
                else:
                    if self.actor_device.type == "cuda":
                        self.scaler.scale(actor_loss).backward()
                    else:
                        actor_loss.backward()
                    if self.device.type == "cuda":
                        self.scaler.scale(0.8 * critic_loss).backward()
                    else:
                        (0.8 * critic_loss).backward()
                if shared_cuda_update:
                    self.scaler.step(self.actor_optim)
                    self.scaler.step(self.critic_optim)
                    self.scaler.update()
                else:
                    if self.actor_device.type == "cuda":
                        self.scaler.step(self.actor_optim)
                    else:
                        self.actor_optim.step()
                    if self.device.type == "cuda":
                        self.scaler.step(self.critic_optim)
                        self.scaler.update()
                    else:
                        self.critic_optim.step()
                tqdm.write(f"    [update {update+1}/{self.n_updates_per_iteration}] actor loss: {actor_loss.item():.5f}, critic loss: {critic_loss.item():.5f}, kl: {kl_div.item():.5f}")
            avg_ep_len = np.mean(ep_lens)
            tqdm.write(f"[training] iteration {it+1} complete: avg episode length = {avg_ep_len:.2f}")
            if (it+1) % eval_freq == 0:
                self.evaluate_policy(
                    num_episodes=eval_episodes,
                    agent_sets=agent_sets,
                )
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_ROLLOUT_TEMPERATURE,
        help="temperature for action selection during training",
    )
    parser.add_argument("--eval_freq", type=int, default=10, help="perform evaluation every n iterations")
    parser.add_argument("--eval_episodes", type=int, default=4000, help="number of evaluation episodes")
    parser.add_argument("--save_freq", type=int, default=10, help="save checkpoint every n iterations")
    parser.add_argument("--max_saves", type=int, default=20, help="maximum number of checkpoint saves to keep")
    parser.add_argument(
        "--rollout_device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="device for batched policy inference (auto prefers CPU when training on CUDA)",
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=256,
        help="number of independent games advanced in each vectorized batch",
    )
    parser.add_argument(
        "--training_actor_device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="device for PPO actor updates (auto uses CUDA when available)",
    )
    parser.add_argument(
        "--compile_updates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="compile CUDA actor/critic updates (default: enabled on CUDA)",
    )
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
                         max_saves=args.max_saves,
                         rollout_device=args.rollout_device,
                         rollout_batch_size=args.rollout_batch_size,
                         training_actor_device=args.training_actor_device,
                         compile_updates=args.compile_updates)
    ppo.save_freq = args.save_freq
    ppo.learn(num_iterations=args.iterations, episodes_per_iter=args.episodes, temperature=args.temperature,
              eval_freq=args.eval_freq, eval_episodes=args.eval_episodes, agent_sets=agent_sets)


if __name__ == "__main__":
    main()
