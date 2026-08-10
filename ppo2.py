import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import hashlib
import multiprocessing
import os
import random
from numbers import Integral
from contextlib import nullcontext
from tqdm import tqdm
import logging
import warnings

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
    POLICY_SMARTER,
    collect_batched_rollout,
    collect_coordinated_batched_rollouts,
)
from seed_integrity import DEVELOPMENT_SEED_LIMIT

DEFAULT_ROLLOUT_TEMPERATURE = 0.8
DEFAULT_TORCH_THREADS = 6
DEFAULT_EVALUATION_WORKERS = 8
PARALLEL_EVALUATION_MIN_EPISODES = 128
DEFAULT_COMPILE_BREAK_EVEN_GAMES = 400_000

_EVALUATION_ACTOR = None


def resolve_compile_updates(
    requested,
    *,
    iterations,
    episodes_per_iteration,
    cuda_available,
    training_actor_device,
):
    """Choose compilation only when a requested run can repay its startup."""
    if requested is not None:
        return bool(requested)
    cuda_training = (
        bool(cuda_available)
        and training_actor_device in ("auto", "cuda")
    )
    requested_games = int(iterations) * int(episodes_per_iteration)
    return (
        cuda_training
        and requested_games >= DEFAULT_COMPILE_BREAK_EVEN_GAMES
    )


def _init_evaluation_worker(actor_state):
    """Build one canonical CPU evaluation actor per spawned worker."""
    global _EVALUATION_ACTOR
    torch.set_num_threads(1)
    actor = StudentNet(
        input_dim=OBSERVATION_SIZE,
        output_dim=ACTION_COUNT,
    ).eval()
    actor.load_state_dict(actor_state)
    _EVALUATION_ACTOR = PPO_Supervised._trace_cpu_actor(actor)


def _run_evaluation_game(task):
    """Run one independently seeded canonical evaluation game."""
    set_index, seed, normalized = task
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    match = game.Game([])
    agents = []
    rl_positions = []
    for agent_type in normalized:
        if agent_type in ("rl", "self"):
            agent = RL_SupervisedAgent(
                match,
                _EVALUATION_ACTOR,
                torch.device("cpu"),
            )
            agent.collect_data = False
            agent.eval_mode = True
            agent.reset_episode()
            rl_positions.append(len(agents))
        elif agent_type == "smart":
            import smartAgent

            agent = smartAgent.Agent()
        elif agent_type == "smarter":
            import smartAgent

            agent = smartAgent.SmarterAgent(match)
        elif agent_type == "random":
            import randomAgent

            agent = randomAgent.AgentRandom()
        else:
            raise ValueError(f"unknown agent type: {agent_type}")
        agents.append(agent)
    match.players = agents
    match.num_players = len(agents)
    for player_index in rl_positions:
        agents[player_index].prev_hand_count = len(
            agents[player_index].mydeck.cards
        )
    match.auto_simulate(max_turns=500)
    return (
        set_index,
        match.turn_count,
        int(match.winner in rl_positions),
    )


def non_rl_policy_kind(agent_type):
    """Map a parsed non-neural opponent name to its batched policy."""
    normalized = str(agent_type).lower()
    if normalized == "smart":
        return POLICY_SMART
    if normalized == "smarter":
        return POLICY_SMARTER
    if normalized == "random":
        return POLICY_RANDOM
    raise ValueError(f"unknown non-RL agent type: {agent_type}")


def coalesce_duplicate_agent_sets(agent_sets, episode_counts):
    """Combine exact duplicate configurations without changing their weight."""
    combined = []
    offsets = {}
    for agent_set, episode_count in zip(agent_sets, episode_counts):
        key = tuple(str(agent_type).lower() for agent_type in agent_set)
        offset = offsets.get(key)
        if offset is None:
            offsets[key] = len(combined)
            combined.append([list(agent_set), int(episode_count)])
        else:
            combined[offset][1] += int(episode_count)
    return [(agent_set, count) for agent_set, count in combined]


def group_compatible_agent_sets(scheduled_sets):
    """Group consecutive configurations that share one engine player count."""
    groups = []
    for agent_set, episode_count in scheduled_sets:
        if (
            groups
            and len(groups[-1][0][0]) == len(agent_set)
        ):
            groups[-1].append((agent_set, episode_count))
        else:
            groups.append([(agent_set, episode_count)])
    return groups


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
    masks = torch.zeros(
        mask_shape,
        device=observations.device,
        dtype=torch.bool,
    )
    card_action_count = COMPATIBLE_END - COMPATIBLE_START
    is_change_sort = observations[..., MODE_INDEX] != 0
    not_change_sort = ~is_change_sort
    masks[..., :card_action_count] = (
        (observations[..., COMPATIBLE_START:COMPATIBLE_END] >= 1)
        & not_change_sort.unsqueeze(-1)
    )
    masks[..., SUIT_ACTION_START:SUIT_ACTION_END] = (
        is_change_sort.unsqueeze(-1)
    )
    masks[..., PASS_ACTION] = not_change_sort
    return masks


def action_log_probabilities(logits, actions):
    """Match ``Categorical(logits=...).log_prob`` without wrapper overhead."""
    normalized = logits - logits.logsumexp(dim=-1, keepdim=True)
    return normalized.gather(
        dim=-1,
        index=actions.unsqueeze(-1),
    ).squeeze(-1)

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
        if self.collect_data:
            obs = torch.zeros(OBSERVATION_SIZE, dtype=torch.float)
        else:
            # Scalar updates to a tiny CPU tensor dispatch one Torch operation
            # per held card. Evaluation does not retain observations, so build
            # the identical float32 features cheaply in NumPy and expose them
            # to Torch without copying.
            obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
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
        obs = populate_game_context(
            obs,
            sorts_played=self.game.sorts_played,
            hand_sizes=[player.mydeck.cardCount() for player in self.game.players],
            player_index=ai_index,
            direction=self.game.direction,
        )
        if isinstance(obs, np.ndarray):
            return torch.from_numpy(obs)
        return obs

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
            if self.eval_mode and not self.collect_data:
                # The single-observation evaluation layout is known here.
                # Mask the two valid action regions directly instead of
                # allocating and populating a separate 55-element bool tensor.
                masked_logits = logits.clone()
                minimum = torch.finfo(logits.dtype).min
                if bool(device_obs[MODE_INDEX].item()):
                    masked_logits[
                        : COMPATIBLE_END - COMPATIBLE_START
                    ] = minimum
                    masked_logits[PASS_ACTION] = minimum
                else:
                    masked_logits[
                        : COMPATIBLE_END - COMPATIBLE_START
                    ].masked_fill_(
                        device_obs[
                            COMPATIBLE_START:COMPATIBLE_END
                        ] < 1,
                        minimum,
                    )
                    masked_logits[
                        SUIT_ACTION_START:SUIT_ACTION_END
                    ] = minimum
            else:
                mask = self.get_action_mask(device_obs)
                masked_logits = logits.masked_fill(
                    ~mask,
                    torch.finfo(logits.dtype).min,
                )
            scaled_logits = masked_logits / temperature
            if self.eval_mode and not self.collect_data:
                normalized = scaled_logits - scaled_logits.logsumexp(
                    dim=-1,
                    keepdim=True,
                )
                action = torch.multinomial(
                    normalized.softmax(dim=-1),
                    num_samples=1,
                ).squeeze(-1)
                # Canonical evaluation only consumes the action. Avoid the
                # distribution wrapper, log probability, and its transfer.
                action_value = int(action.item())
                log_prob_value = 0.0
            else:
                dist = Categorical(
                    logits=scaled_logits,
                    validate_args=False,
                )
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


class _PreallocatedStudentInference(nn.Module):
    """Inference-only StudentNet with one reusable dense-feature tensor."""

    def __init__(self, actor):
        super().__init__()
        self.model = actor.model
        self.input_dim = int(self.model.input_dim)
        self.total_features = int(self.model.final_layer.in_features)

    def forward(self, x):
        features = x.new_empty((x.size(0), self.total_features))
        features[:, : self.input_dim] = x
        end = self.input_dim
        for layer, batch_norm in zip(self.model.layers, self.model.bns):
            output = layer(features[:, :end])
            output = batch_norm(output)
            output = nn.functional.relu(output)
            next_end = end + output.size(1)
            features[:, end:next_end] = output
            end = next_end
        return self.model.final_layer(features)


class _HybridRolloutInference(nn.Module):
    """Run large policy forwards on CUDA while keeping rollout state on CPU."""

    def __init__(self, cpu_actor, gpu_actor, source_actor, threshold):
        super().__init__()
        self.cpu_actor = cpu_actor
        self.gpu_actor = gpu_actor
        self.threshold = int(threshold)
        source_state = source_actor.state_dict()
        gpu_state = gpu_actor.state_dict()
        self.gpu_actor_shares_source_storage = (
            source_state.keys() == gpu_state.keys()
            and all(
                source_state[name].data_ptr()
                == gpu_state[name].data_ptr()
                for name in source_state
            )
        )

    def sync_from(self, actor):
        state = actor.state_dict()
        self.cpu_actor.load_state_dict(state)
        if not self.gpu_actor_shares_source_storage:
            self.gpu_actor.load_state_dict(state)

    def forward(self, observations):
        if observations.size(0) < self.threshold:
            return self.cpu_actor(observations)
        # Strict FP32 preserves CPU actions in parity workloads.  Restore the
        # training setting immediately so compiled updates retain TF32-high.
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            return self.gpu_actor(
                observations.to(device="cuda")
            ).to(device="cpu")
        finally:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32


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
                 rollout_batch_size=1024,
                 training_actor_device="auto",
                 compile_updates=None,
                 skip_dynamic_cudagraphs=True,
                 hybrid_rollout_inference=True,
                 hybrid_rollout_threshold=64,
                 cuda_rollout_autocast=False,
                 reuse_cpu_critic_forward=True,
                 native_gae=True,
                 trace_cpu_rollout_actor=True,
                 native_smart_policy=True,
                 native_rollout_engine=True,
                 record_rollout_public_events=False,
                 direct_rollout_sampling=True,
                 numpy_rollout_masks=True,
                 reuse_rollout_workspaces=True,
                 coalesce_rollout_agent_sets=True,
                 group_compatible_rollout_agent_sets=True,
                 coordinate_rollout_engines=True,
                 drain_non_neural_rollout=True,
                 reuse_rollout_engines=True,
                 past_policy_pool_size=2,
                 past_policy_probability=0.2,
                 torch_threads=DEFAULT_TORCH_THREADS,
                 evaluation_workers=DEFAULT_EVALUATION_WORKERS,
                 evaluation_start_method="auto"):
        if (
            isinstance(torch_threads, bool)
            or not isinstance(torch_threads, Integral)
            or torch_threads < 1
        ):
            raise ValueError("torch_threads must be a positive integer")
        if (
            isinstance(evaluation_workers, bool)
            or not isinstance(evaluation_workers, Integral)
            or evaluation_workers < 1
        ):
            raise ValueError("evaluation_workers must be a positive integer")
        if evaluation_start_method not in (
            "auto",
            "spawn",
            "forkserver",
        ):
            raise ValueError(
                "evaluation_start_method must be auto, spawn, or forkserver"
            )
        if (
            isinstance(hybrid_rollout_threshold, bool)
            or not isinstance(hybrid_rollout_threshold, Integral)
            or hybrid_rollout_threshold < 1
        ):
            raise ValueError(
                "hybrid_rollout_threshold must be a positive integer"
            )
        # Small dense actor/critic updates and shrinking rollout batches
        # oversubscribe badly at the common 16-thread host default.
        self.torch_threads = int(torch_threads)
        self.evaluation_workers = int(evaluation_workers)
        self.evaluation_start_method = evaluation_start_method
        torch.set_num_threads(self.torch_threads)
        self._init_hyperparameters()
        if (
            not np.isfinite(past_policy_probability)
            or not 0.0 <= past_policy_probability <= 1.0
        ):
            raise ValueError("past_policy_probability must be in [0, 1]")
        if (
            past_policy_pool_size is not None
            and (
                isinstance(past_policy_pool_size, bool)
                or not isinstance(past_policy_pool_size, Integral)
                or past_policy_pool_size < 1
            )
        ):
            raise ValueError("past_policy_pool_size must be positive or None")
        self.past_policy_probability = float(past_policy_probability)
        self.past_policy_pool_size = (
            None
            if past_policy_pool_size is None
            else int(past_policy_pool_size)
        )
        self.direct_rollout_sampling = bool(direct_rollout_sampling)
        self.numpy_rollout_masks = bool(numpy_rollout_masks)
        self.reuse_rollout_workspaces = bool(reuse_rollout_workspaces)
        self.coalesce_rollout_agent_sets = bool(coalesce_rollout_agent_sets)
        self.group_compatible_rollout_agent_sets = bool(
            group_compatible_rollout_agent_sets
        )
        self.coordinate_rollout_engines = bool(
            coordinate_rollout_engines
        )
        self.drain_non_neural_rollout = bool(
            drain_non_neural_rollout
        )
        self.reuse_rollout_engines = bool(reuse_rollout_engines)
        self.record_rollout_public_events = bool(record_rollout_public_events)
        self.native_smart_policy = False
        if native_smart_policy:
            try:
                from numba_smart_policy import warm_native_smart_policy
            except ImportError:
                warm_native_smart_policy = None
            if (
                warm_native_smart_policy is not None
                and warm_native_smart_policy()
            ):
                self.native_smart_policy = True
        self.native_rollout_engine = False
        if native_rollout_engine:
            try:
                from numba_rollout_engine import warm_native_rollout_engine
            except ImportError:
                warm_native_rollout_engine = None
            if (
                warm_native_rollout_engine is not None
                and warm_native_rollout_engine()
            ):
                self.native_rollout_engine = True
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
        if (
            self.device.type == "cuda"
            or self.actor_device.type == "cuda"
        ):
            # Ampere tensor cores accelerate these dense FP32 updates while
            # retaining the numerical mode already used by the CUDA CLI and
            # live MLP trainer.
            torch.set_float32_matmul_precision("high")
        self.cuda_rollout_autocast = bool(cuda_rollout_autocast)
        self.reuse_cpu_critic_forward = bool(
            reuse_cpu_critic_forward
        )
        self.native_gae = False
        if native_gae:
            try:
                from numba_gae import warm_native_gae
            except ImportError:
                warm_native_gae = None
            if warm_native_gae is not None and warm_native_gae():
                self.native_gae = True
        self.actor = StudentNet(
            input_dim=OBSERVATION_SIZE,
            output_dim=ACTION_COUNT,
        ).to(self.actor_device)
        self.critic = StudentCritic().to(self.device)
        self.trace_cpu_rollout_actor = bool(
            trace_cpu_rollout_actor and self.rollout_device.type == "cpu"
        )
        if (
            self.actor_device == self.rollout_device
            and not self.trace_cpu_rollout_actor
        ):
            self.rollout_actor = self.actor
        else:
            # Rollout inference stays on CPU, where batch sizes shrink toward
            # the tail of each game batch.  PPO updates use the accelerator.
            self.rollout_actor = StudentNet(
                input_dim=OBSERVATION_SIZE,
                output_dim=ACTION_COUNT,
            ).to(self.rollout_device)
            if self.trace_cpu_rollout_actor:
                self.rollout_actor = self._trace_cpu_actor(
                    self.rollout_actor
                )
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
            # PPO minibatches vary in size.  Capturing every distinct dynamic
            # shape wastes compilation/runtime work and eventually triggers
            # Inductor's dynamic-CUDAGraph warning.  The compiled kernels still
            # run on CUDA when graph capture is skipped.
            torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = bool(
                skip_dynamic_cudagraphs
            )
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
        self.hybrid_rollout_threshold = int(hybrid_rollout_threshold)
        self.hybrid_rollout_inference = bool(
            hybrid_rollout_inference
            and self.actor_device.type == "cuda"
            and self.rollout_device.type == "cpu"
        )
        if self.hybrid_rollout_inference:
            self.rollout_actor = _HybridRolloutInference(
                self.rollout_actor,
                self._trace_gpu_actor(self.actor),
                self.actor,
                self.hybrid_rollout_threshold,
            ).eval()
        if rollout_batch_size < 1:
            raise ValueError("rollout_batch_size must be at least one")
        self.rollout_batch_size = int(rollout_batch_size)
        self._checkpoint_actor_cache = {}
        self._rollout_engine_cache = {}
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

    @staticmethod
    def _trace_cpu_actor(actor):
        """Remove Python dispatch from a dynamic-batch CPU inference actor."""
        actor.eval()
        if isinstance(actor, StudentNet):
            actor = _PreallocatedStudentInference(actor).eval()
        example = torch.zeros(
            (24, OBSERVATION_SIZE),
            dtype=torch.float32,
            device="cpu",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            return torch.jit.trace(
                actor,
                example,
                check_trace=False,
            ).eval()

    @staticmethod
    def _trace_gpu_actor(actor):
        """Trace a dynamic-batch CUDA actor used by hybrid rollout inference."""
        actor.eval()
        if isinstance(actor, StudentNet):
            actor = _PreallocatedStudentInference(actor).eval()
        example = torch.zeros(
            (256, OBSERVATION_SIZE),
            dtype=torch.float32,
            device="cuda",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", torch.jit.TracerWarning)
            warnings.simplefilter("ignore", DeprecationWarning)
            return torch.jit.trace(
                actor,
                example,
                check_trace=False,
            ).eval()

    def _init_hyperparameters(self):
        self.max_timesteps_per_episode = 5000
        self.n_updates_per_iteration = 3
        self.lr = 0.00003
        self.clip = 0.1
        self.save_freq = 1

    def _sync_rollout_actor(self):
        self.actor.eval()
        if isinstance(self.rollout_actor, _HybridRolloutInference):
            self.rollout_actor.sync_from(self.actor)
        elif self.rollout_actor is not self.actor:
            self.rollout_actor.load_state_dict(self.actor.state_dict())
        self.rollout_actor.eval()

    def actor_update_autocast(self):
        """Match rollout numerics when training and rollout devices differ."""
        if (
            self.actor_device != self.rollout_device
            or not self.cuda_rollout_autocast
        ):
            return nullcontext()
        return autocast_for(self.actor_device)

    def actor_update_forward(self, observations):
        return self._actor_update_model(observations)

    def critic_update_forward(self, observations):
        return self._critic_update_model(observations)

    def _critic_forward_with_two_bn_updates(self, observations):
        """Build one CPU graph while matching two BN running-stat updates."""
        batch_norms = [
            module
            for module in self.critic.modules()
            if (
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                and module.training
                and module.track_running_stats
                and module.momentum is not None
            )
        ]
        momentums = [float(module.momentum) for module in batch_norms]
        try:
            for module, momentum in zip(batch_norms, momentums):
                module.momentum = 1.0 - (1.0 - momentum) ** 2
            values = self.critic_update_forward(observations).squeeze(-1)
            with torch.no_grad():
                for module in batch_norms:
                    module.num_batches_tracked.add_(1)
            return values
        finally:
            for module, momentum in zip(batch_norms, momentums):
                module.momentum = momentum

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
            for parameter in actor.parameters():
                parameter.requires_grad_(False)
            if self.trace_cpu_rollout_actor:
                actor = self._trace_cpu_actor(actor)
            self._checkpoint_actor_cache[checkpoint] = actor
        return actor

    def _select_past_checkpoint(self, agent_types, available_checkpoints):
        """Select one fixed opponent for an eligible game, if configured."""
        eligible = (
            bool(available_checkpoints)
            and sum(
                agent_type in ("rl", "self")
                for agent_type in agent_types
            ) > 1
        )
        if not eligible or self.past_policy_probability == 0.0:
            return None, eligible
        if (
            self.past_policy_probability < 1.0
            and np.random.random() >= self.past_policy_probability
        ):
            return None, eligible
        return np.random.choice(available_checkpoints), eligible

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
        past_policy_eligible_games = 0
        past_policy_games = 0
        actor_key_counts = {}
        actor_key_layout = hashlib.sha256()
        num_sets = len(agent_sets)
        available_checkpoints = self._available_actor_checkpoints()
        available_checkpoint_count = len(available_checkpoints)
        if (
            self.past_policy_pool_size is not None
            and len(available_checkpoints) > self.past_policy_pool_size
        ):
            checkpoint_offsets = np.random.choice(
                len(available_checkpoints),
                size=self.past_policy_pool_size,
                replace=False,
            )
            available_checkpoints = [
                available_checkpoints[int(offset)]
                for offset in checkpoint_offsets
            ]
        for i, agent_set in enumerate(agent_sets):
            normalized = [typ.lower() for typ in agent_set]
            if not any(typ in ['rl', 'self'] for typ in normalized):
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
                agent_actor_keys = {}
                chosen_ckpt, past_eligible = self._select_past_checkpoint(
                    normalized, available_checkpoints
                )
                past_policy_eligible_games += int(past_eligible)
                past_version_used = False
                for typ in normalized:
                    if typ in ['rl', 'self']:
                        if (
                            not past_version_used
                            and chosen_ckpt is not None
                        ):
                            a = RL_SupervisedAgent(
                                g,
                                self._checkpoint_actor(chosen_ckpt),
                                self.rollout_device,
                                temperature=temperature,
                            )
                            a.collect_data = False  # do not collect data from past version
                            a.eval_mode = True
                            past_version_used = True
                            past_policy_games += 1
                            actor_key = f"checkpoint:{chosen_ckpt}"
                        else:
                            a = RL_SupervisedAgent(
                                g,
                                self.rollout_actor,
                                self.rollout_device,
                                temperature=temperature,
                            )
                            a.collect_data = True
                            a.eval_mode = False
                            actor_key = "current"
                        a.reset_episode()
                        agents.append(a)
                        agent_actor_keys[id(a)] = actor_key
                        actor_key_counts[actor_key] = (
                            actor_key_counts.get(actor_key, 0) + 1
                        )
                        if a.collect_data:
                            rl_agents.append(a)
                    elif typ == 'smart':
                        import smartAgent
                        agents.append(smartAgent.Agent())
                    elif typ == 'smarter':
                        import smartAgent
                        agents.append(smartAgent.SmarterAgent(g))
                    elif typ == 'random':
                        import randomAgent
                        agents.append(randomAgent.AgentRandom())
                    else:
                        print(f"unknown agent type: {typ}")
                shuffle(agents)
                actor_key_layout.update(
                    "|".join(
                        agent_actor_keys.get(id(agent), "")
                        for agent in agents
                    ).encode("utf-8")
                )
                actor_key_layout.update(b"\n")
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
            "past_policy_probability": self.past_policy_probability,
            "past_policy_eligible_games": past_policy_eligible_games,
            "past_policy_games": past_policy_games,
            "actor_key_counts": actor_key_counts,
            "actor_key_layout_sha256": actor_key_layout.hexdigest(),
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
        past_policy_eligible_games = 0
        past_policy_games = 0
        actor_key_counts = {}
        actor_key_layout = hashlib.sha256()
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
        available_checkpoint_count = len(available_checkpoints)
        if (
            self.past_policy_pool_size is not None
            and len(available_checkpoints) > self.past_policy_pool_size
        ):
            checkpoint_offsets = np.random.choice(
                len(available_checkpoints),
                size=self.past_policy_pool_size,
                replace=False,
            )
            available_checkpoints = [
                available_checkpoints[int(offset)]
                for offset in checkpoint_offsets
            ]
        current_key = "current"
        current_policy = ActorPolicy(
            actor=self.rollout_actor,
            device=self.rollout_device,
            temperature=temperature,
            collect_data=True,
            autocast=self.cuda_rollout_autocast,
        )

        episodes_per_set = [num_episodes // len(agent_sets)] * len(agent_sets)
        episodes_per_set[-1] += num_episodes - sum(episodes_per_set)
        scheduled_sets = list(zip(agent_sets, episodes_per_set))
        if self.coalesce_rollout_agent_sets:
            scheduled_sets = coalesce_duplicate_agent_sets(
                agent_sets,
                episodes_per_set,
            )
        rollout_groups = [
            [scheduled_set]
            for scheduled_set in scheduled_sets
        ]
        if self.group_compatible_rollout_agent_sets:
            rollout_groups = group_compatible_agent_sets(scheduled_sets)
        display_index = 0
        configuration_groups = []
        for rollout_group in rollout_groups:
            game_configurations = []
            for agent_set, set_episodes in rollout_group:
                display_index += 1
                normalized = [
                    agent_type.lower() for agent_type in agent_set
                ]
                if not any(
                    agent_type in ("rl", "self")
                    for agent_type in normalized
                ):
                    raise ValueError(
                        "each agent set must include at least one rl agent"
                    )
                unknown = set(normalized) - {
                    "rl", "self", "smart", "smarter", "random"
                }
                if unknown:
                    raise ValueError(
                        "unknown agent type(s): "
                        + ", ".join(sorted(unknown))
                    )
                if set_episodes == 0:
                    continue
                print(
                    "[data generation] collecting "
                    f"{set_episodes} episode(s) for agent set "
                    f"{display_index}: {agent_set}"
                )
                game_configurations.extend(
                    [normalized] * set_episodes
                )
            configuration_groups.append(game_configurations)

        def build_engine(
            batch_configurations,
            actor_policies,
            engine_cache_key,
        ):
            nonlocal past_policy_eligible_games, past_policy_games
            batch_games = len(batch_configurations)
            player_count = len(batch_configurations[0])
            policy_kinds = np.empty(
                (batch_games, player_count),
                dtype=np.int8,
            )
            actor_keys = np.empty(
                (batch_games, player_count),
                dtype=object,
            )
            for game_index, normalized in enumerate(batch_configurations):
                entries = []
                chosen_checkpoint, past_eligible = (
                    self._select_past_checkpoint(
                        normalized, available_checkpoints
                    )
                )
                past_policy_eligible_games += int(past_eligible)
                past_version_used = False
                for agent_type in normalized:
                    if agent_type in ("rl", "self"):
                        if (
                            chosen_checkpoint is not None
                            and not past_version_used
                        ):
                            checkpoint_key = (
                                f"checkpoint:{chosen_checkpoint}"
                            )
                            entries.append((POLICY_RL, checkpoint_key))
                            if checkpoint_key not in actor_policies:
                                actor_policies[checkpoint_key] = ActorPolicy(
                                    actor=self._checkpoint_actor(
                                        chosen_checkpoint
                                    ),
                                    device=self.rollout_device,
                                    temperature=1e-5,
                                    collect_data=False,
                                    greedy=True,
                                    autocast=self.cuda_rollout_autocast,
                                )
                            past_version_used = True
                            past_policy_games += 1
                        else:
                            entries.append((POLICY_RL, current_key))
                    else:
                        entries.append(
                            (non_rl_policy_kind(agent_type), None)
                        )
                shuffle(entries)
                for player_index, (kind, key) in enumerate(entries):
                    policy_kinds[game_index, player_index] = kind
                    actor_keys[game_index, player_index] = key
                    if key is not None:
                        actor_key_counts[key] = (
                            actor_key_counts.get(key, 0) + 1
                        )
                actor_key_layout.update(
                    "|".join(
                        "" if key is None else str(key)
                        for _, key in entries
                    ).encode("utf-8")
                )
                actor_key_layout.update(b"\n")

            seeds = np.random.randint(
                0,
                DEVELOPMENT_SEED_LIMIT,
                size=batch_games,
                dtype=np.uint32,
            )
            engine = (
                self._rollout_engine_cache.get(engine_cache_key)
                if self.reuse_rollout_engines
                else None
            )
            if engine is not None:
                engine.reset_games(policy_kinds, actor_keys, seeds)
                return engine
            engine = BatchedPesten(
                policy_kinds,
                actor_keys,
                seeds,
                max_turns=500,
                # Sequential PPO constructs smartAgent.Agent() without the
                # optional Game reference, so preserve that exact behavior.
                smart_agents_have_game_reference=False,
                native_smart_policy=self.native_smart_policy,
                native_rollout_engine=self.native_rollout_engine,
                record_public_events=self.record_rollout_public_events,
            )
            if self.reuse_rollout_engines:
                self._rollout_engine_cache[engine_cache_key] = engine
            return engine

        def record_result(result):
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

        if (
            self.coordinate_rollout_engines
            and len(configuration_groups) > 1
        ):
            largest_group = max(
                (len(configurations) for configurations in configuration_groups),
                default=0,
            )
            for batch_start in range(
                0,
                largest_group,
                self.rollout_batch_size,
            ):
                actor_policies = {current_key: current_policy}
                engines = []
                for group_index, game_configurations in enumerate(
                    configuration_groups
                ):
                    batch_configurations = game_configurations[
                        batch_start : batch_start + self.rollout_batch_size
                    ]
                    if not batch_configurations:
                        continue
                    engines.append(
                        build_engine(
                            batch_configurations,
                            actor_policies,
                            (
                                group_index,
                                len(batch_configurations),
                                len(batch_configurations[0]),
                            ),
                        )
                    )
                for result in collect_coordinated_batched_rollouts(
                    engines,
                    actor_policies,
                    output_device=self.actor_device,
                    direct_action_sampling=self.direct_rollout_sampling,
                    numpy_action_masks=self.numpy_rollout_masks,
                    reuse_rollout_workspaces=self.reuse_rollout_workspaces,
                    drain_non_neural=(
                        self.drain_non_neural_rollout
                        and len(engines) == 1
                        and engines[0].num_players == 2
                    ),
                ):
                    record_result(result)
        else:
            for group_index, game_configurations in enumerate(
                configuration_groups
            ):
                for batch_start in range(
                    0,
                    len(game_configurations),
                    self.rollout_batch_size,
                ):
                    batch_configurations = game_configurations[
                        batch_start : batch_start + self.rollout_batch_size
                    ]
                    actor_policies = {current_key: current_policy}
                    engine = build_engine(
                        batch_configurations,
                        actor_policies,
                        (
                            group_index,
                            len(batch_configurations),
                            len(batch_configurations[0]),
                        ),
                    )
                    result = collect_batched_rollout(
                        engine,
                        actor_policies,
                        output_device=self.actor_device,
                        direct_action_sampling=self.direct_rollout_sampling,
                        numpy_action_masks=self.numpy_rollout_masks,
                        reuse_rollout_workspaces=self.reuse_rollout_workspaces,
                        drain_non_neural=(
                            self.drain_non_neural_rollout
                            and engine.num_players == 2
                        ),
                    )
                    record_result(result)

        aggregate_metrics.update(
            {
                "torch_threads": self.torch_threads,
                "past_policy_probability": self.past_policy_probability,
                "past_policy_eligible_games": past_policy_eligible_games,
                "past_policy_games": past_policy_games,
                "past_policy_available_checkpoints": (
                    available_checkpoint_count
                ),
                "past_policy_pool_checkpoints": len(available_checkpoints),
                "coordinate_rollout_engines": (
                    self.coordinate_rollout_engines
                ),
                "drain_non_neural_rollout": (
                    self.drain_non_neural_rollout
                ),
                "reuse_rollout_engines": self.reuse_rollout_engines,
                "actor_key_counts": actor_key_counts,
                "actor_key_layout_sha256": actor_key_layout.hexdigest(),
            }
        )
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
        if (
            isinstance(num_episodes, bool)
            or not isinstance(num_episodes, Integral)
            or num_episodes < 1
        ):
            raise ValueError("num_episodes must be a positive integer")
        if not agent_sets:
            raise ValueError("agent_sets cannot be empty")

        supported_agents = {"rl", "self", "smart", "smarter", "random"}
        normalized_sets = []
        for agent_set in agent_sets:
            normalized = tuple(str(agent_type).lower() for agent_type in agent_set)
            if not any(agent_type in ("rl", "self") for agent_type in normalized):
                raise ValueError("each agent set must include at least one rl agent")
            unknown_agents = set(normalized) - supported_agents
            if unknown_agents:
                unknown = sorted(unknown_agents)[0]
                raise ValueError(f"unknown agent type: {unknown}")
            normalized_sets.append(normalized)

        self.actor.eval()
        self.critic.eval()
        try:
            self._sync_rollout_actor()
            print(
                f"\n[evaluation] starting evaluation for "
                f"{num_episodes} episode(s)..."
            )
            num_sets = len(normalized_sets)
            base_episodes = num_episodes // num_sets
            episode_counts = [base_episodes] * num_sets
            episode_counts[-1] += num_episodes - sum(episode_counts)
            for set_index, (agent_set, episodes_for_set) in enumerate(
                zip(agent_sets, episode_counts)
            ):
                print(
                    f"[evaluation] running {episodes_for_set} episode(s) "
                    f"for agent set {set_index + 1}: {agent_set}"
                )

            use_parallel = (
                self.evaluation_workers > 1
                and num_episodes >= PARALLEL_EVALUATION_MIN_EPISODES
            )
            if use_parallel:
                seeds = np.random.randint(
                    0,
                    DEVELOPMENT_SEED_LIMIT,
                    size=num_episodes,
                    dtype=np.int64,
                )
                tasks = []
                seed_index = 0
                for set_index, (normalized, episode_count) in enumerate(
                    zip(normalized_sets, episode_counts)
                ):
                    for _ in range(episode_count):
                        tasks.append(
                            (
                                set_index,
                                int(seeds[seed_index]),
                                normalized,
                            )
                        )
                        seed_index += 1
                actor_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.actor.state_dict().items()
                }
                worker_count = min(self.evaluation_workers, num_episodes)
                # Use several work units per process so unusually long card
                # games do not leave one worker holding the evaluation tail.
                chunksize = max(1, len(tasks) // (worker_count * 8))
                if self.evaluation_start_method == "auto":
                    available_methods = (
                        multiprocessing.get_all_start_methods()
                    )
                    start_methods = (
                        ["forkserver", "spawn"]
                        if "forkserver" in available_methods
                        else ["spawn"]
                    )
                else:
                    start_methods = [self.evaluation_start_method]
                for start_method in start_methods:
                    try:
                        context = multiprocessing.get_context(
                            start_method
                        )
                        with context.Pool(
                            worker_count,
                            initializer=_init_evaluation_worker,
                            initargs=(actor_state,),
                        ) as pool:
                            results = pool.map(
                                _run_evaluation_game,
                                tasks,
                                chunksize,
                            )
                        used_start_method = start_method
                        break
                    except OSError:
                        if start_method == start_methods[-1]:
                            raise
                        warnings.warn(
                            f"{start_method} evaluation unavailable; "
                            "falling back to spawn",
                            RuntimeWarning,
                            stacklevel=2,
                        )
            else:
                worker_count = 1
                used_start_method = None
                results = []
                for set_index, (normalized, episode_count) in enumerate(
                    zip(normalized_sets, episode_counts)
                ):
                    for _ in tqdm(
                        range(episode_count),
                        desc="evaluation episodes",
                        leave=True,
                    ):
                        match = game.Game([])
                        agents = []
                        rl_agents = []
                        for agent_type in normalized:
                            if agent_type in ("rl", "self"):
                                agent = RL_SupervisedAgent(
                                    match,
                                    self.rollout_actor,
                                    self.rollout_device,
                                )
                                agent.collect_data = False
                                agent.eval_mode = True
                                agent.reset_episode()
                                rl_agents.append(agent)
                            elif agent_type == "smart":
                                import smartAgent

                                agent = smartAgent.Agent()
                            elif agent_type == "smarter":
                                import smartAgent

                                agent = smartAgent.SmarterAgent(match)
                            else:
                                import randomAgent

                                agent = randomAgent.AgentRandom()
                            agents.append(agent)
                        match.players = agents
                        match.num_players = len(agents)
                        for agent in rl_agents:
                            agent.prev_hand_count = len(agent.mydeck.cards)
                        match.auto_simulate(max_turns=500)
                        results.append(
                            (
                                set_index,
                                match.turn_count,
                                int(
                                    match.winner is not None
                                    and any(
                                        match.winner == agents.index(agent)
                                        for agent in rl_agents
                                    )
                                ),
                            )
                        )

            set_wins = [0] * num_sets
            set_turns = [0] * num_sets
            for set_index, turn_count, won in results:
                set_turns[set_index] += turn_count
                set_wins[set_index] += won
            for set_index, (wins, episode_count) in enumerate(
                zip(set_wins, episode_counts)
            ):
                set_win_rate = (
                    100.0 * wins / episode_count
                    if episode_count
                    else 0.0
                )
                print(
                    f"[evaluation] agent set {set_index + 1} win rate: "
                    f"{set_win_rate:.2f}% over {episode_count} episodes"
                )
            total_turns = sum(set_turns)
            global_wins = sum(set_wins)
            global_win_rate = 100.0 * global_wins / num_episodes
            avg_turns = total_turns / num_episodes
            print(
                f"[evaluation] complete. overall win rate: "
                f"{global_win_rate:.2f}%, avg turns: {avg_turns:.2f} "
                f"({worker_count} worker(s))\n"
            )
            return {
                "win_rate": global_win_rate,
                "average_turns": avg_turns,
                "wins": global_wins,
                "episodes": int(num_episodes),
                "workers": worker_count,
                "parallel": use_parallel,
                "start_method": used_start_method,
                "set_wins": set_wins,
                "set_turns": set_turns,
            }
        finally:
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
            reusable_critic_values = None
            if (
                self.reuse_cpu_critic_forward
                and self.device.type == "cpu"
            ):
                reusable_critic_values = (
                    self._critic_forward_with_two_bn_updates(critic_obs)
                )
                gae_values = reusable_critic_values.detach()
            else:
                with torch.no_grad():
                    gae_values = self.critic_update_forward(
                        critic_obs
                    ).squeeze(-1)
            with torch.no_grad():
                if self.native_gae:
                    from numba_gae import compute_gae_native

                    advantages, rtgs = compute_gae_native(
                        critic_rewards,
                        gae_values,
                        ep_lens,
                        gamma=0.97,
                        lam=0.95,
                    )
                else:
                    advantages, rtgs = compute_gae(
                        critic_rewards,
                        gae_values,
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
                    new_log_probs = action_log_probabilities(
                        masked_logits,
                        batch_actions,
                    )
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
                    values = (
                        reusable_critic_values
                        if (
                            update == 0
                            and reusable_critic_values is not None
                        )
                        else self.critic_update_forward(
                            critic_obs
                        ).squeeze(-1)
                    )
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
        "--seed",
        type=int,
        default=20260731,
        help="seed for Python, NumPy, PyTorch, and CUDA RNGs",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_ROLLOUT_TEMPERATURE,
        help="temperature for action selection during training",
    )
    parser.add_argument("--eval_freq", type=int, default=10, help="perform evaluation every n iterations")
    parser.add_argument("--eval_episodes", type=int, default=4000, help="number of evaluation episodes")
    parser.add_argument(
        "--evaluation-workers",
        type=int,
        default=DEFAULT_EVALUATION_WORKERS,
        help=(
            "spawned canonical-game workers for evaluations of at least "
            f"{PARALLEL_EVALUATION_MIN_EPISODES} episodes "
            f"(default: {DEFAULT_EVALUATION_WORKERS})"
        ),
    )
    parser.add_argument(
        "--evaluation-start-method",
        choices=("auto", "spawn", "forkserver"),
        default="auto",
        help=(
            "safe multiprocessing start method for canonical evaluation; "
            "auto prefers forkserver and falls back to spawn"
        ),
    )
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
        default=1024,
        help=(
            "maximum independent games advanced in each vectorized batch "
            "(default: 1024)"
        ),
    )
    parser.add_argument(
        "--torch_threads",
        "--torch-threads",
        dest="torch_threads",
        type=int,
        default=DEFAULT_TORCH_THREADS,
        help=(
            "PyTorch intra-op threads for rollout and PPO updates "
            f"(default: {DEFAULT_TORCH_THREADS})"
        ),
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
        help=(
            "compile CUDA actor/critic updates; auto enables only for at "
            f"least {DEFAULT_COMPILE_BREAK_EVEN_GAMES:,} requested games"
        ),
    )
    parser.add_argument(
        "--skip-dynamic-cudagraphs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "skip CUDA Graph capture for dynamic PPO minibatch shapes while "
            "retaining compiled CUDA kernels (default: enabled)"
        ),
    )
    parser.add_argument(
        "--hybrid-rollout-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "run policy forwards of at least --hybrid-rollout-threshold "
            "rows on CUDA while keeping rollout state and sampling on CPU "
            "(default: enabled)"
        ),
    )
    parser.add_argument(
        "--hybrid-rollout-threshold",
        type=int,
        default=64,
        help=(
            "minimum policy-forward rows sent to CUDA by hybrid rollout "
            "(default: 64)"
        ),
    )
    parser.add_argument(
        "--cuda-rollout-autocast",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "use BF16/FP16 autocast for CUDA rollout actors and matching "
            "actor updates (default: disabled)"
        ),
    )
    parser.add_argument(
        "--reuse-cpu-critic-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reuse the CPU critic graph from GAE preparation for PPO update "
            "one while preserving BatchNorm statistics (default: enabled)"
        ),
    )
    parser.add_argument(
        "--native-gae",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use the exact segmented Numba advantage-estimation recurrence "
            "(default: enabled when installed)"
        ),
    )
    parser.add_argument(
        "--trace-cpu-rollout-actor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "trace the synchronized CPU rollout actor to remove Python "
            "dispatch (default: enabled)"
        ),
    )
    parser.add_argument(
        "--native-smart-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use the optional exact Numba Smart-opponent kernel "
            "(default: enabled when installed)"
        ),
    )
    parser.add_argument(
        "--direct-rollout-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "sample rollout actions and gather log probabilities directly "
            "without distribution-object overhead (default: enabled)"
        ),
    )
    parser.add_argument(
        "--native-rollout-engine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "use the optional exact Numba kernel for common batched state "
            "transitions (default: enabled when installed)"
        ),
    )
    parser.add_argument(
        "--record-rollout-public-events",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "retain per-game public event tuples during training rollouts "
            "(default: disabled; counters and game state remain enabled)"
        ),
    )
    parser.add_argument(
        "--numpy-rollout-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "build CPU rollout action masks with NumPy to avoid small-tensor "
            "Torch dispatch (default: enabled)"
        ),
    )
    parser.add_argument(
        "--reuse-rollout-workspaces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reuse caller-owned observation and action-mask arrays during "
            "vectorized rollouts (default: enabled)"
        ),
    )
    parser.add_argument(
        "--coalesce-rollout-agent-sets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "combine exact duplicate weighted agent sets into larger vector "
            "batches while preserving episode counts (default: enabled)"
        ),
    )
    parser.add_argument(
        "--group-compatible-rollout-agent-sets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "share vector engines across consecutive agent sets with the "
            "same player count (default: enabled)"
        ),
    )
    parser.add_argument(
        "--coordinate-rollout-engines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "batch actor requests across independent player-count engines "
            "(default: enabled)"
        ),
    )
    parser.add_argument(
        "--drain-non-neural-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "advance a lone two-player engine to each neural-policy boundary "
            "before inference (default: enabled)"
        ),
    )
    parser.add_argument(
        "--reuse-rollout-engines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "reset same-shaped vector game engines between iterations "
            "instead of reallocating state arrays (default: enabled)"
        ),
    )
    parser.add_argument(
        "--past-policy-probability",
        "--past_policy_probability",
        dest="past_policy_probability",
        type=float,
        default=0.2,
        help=(
            "per-game probability of replacing one of two or more RL seats "
            "with a fixed saved-checkpoint opponent"
        ),
    )
    parser.add_argument(
        "--past-policy-pool-size",
        type=int,
        default=2,
        help=(
            "maximum uniformly sampled historical checkpoints used within "
            "one rollout iteration (default: 2)"
        ),
    )
    parser.add_argument("--agent_sets", type=str, default="smart,rl",
                        help="semicolon-separated sets of comma-separated agent types (smart, smarter, random, rl/self; each set must include at least one rl agent), e.g. \"smarter,rl;random,rl\"")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    agent_sets = parse_agent_sets(args.agent_sets)
    # validate each set includes at least one rl agent
    for aset in agent_sets:
        if not any(a.lower() in ['rl', 'self'] for a in aset):
            raise ValueError("each agent set must include at least one rl agent")
    
    print("training on the following agent sets:")
    for idx, aset in enumerate(agent_sets):
        print(f" set {idx+1}: {aset}")

    compile_updates = resolve_compile_updates(
        args.compile_updates,
        iterations=args.iterations,
        episodes_per_iteration=args.episodes,
        cuda_available=torch.cuda.is_available(),
        training_actor_device=args.training_actor_device,
    )
    print(
        "CUDA update compilation: "
        f"{'enabled' if compile_updates else 'disabled'}"
    )
    
    ppo = PPO_Supervised(use_pretrained=True,
                         actor_path="ppo_actor_100run3.pth",
                         critic_path="ppo_critic_100run3.pth",
                         actor_opt_path="ppo_actor_opt_100run3.pth",
                         critic_opt_path="ppo_critic_opt_100run3.pth",
                         max_saves=args.max_saves,
                         rollout_device=args.rollout_device,
                         rollout_batch_size=args.rollout_batch_size,
                         training_actor_device=args.training_actor_device,
                         compile_updates=compile_updates,
                         skip_dynamic_cudagraphs=(
                             args.skip_dynamic_cudagraphs
                         ),
                         hybrid_rollout_inference=(
                             args.hybrid_rollout_inference
                         ),
                         hybrid_rollout_threshold=(
                             args.hybrid_rollout_threshold
                         ),
                         cuda_rollout_autocast=(
                             args.cuda_rollout_autocast
                         ),
                         reuse_cpu_critic_forward=(
                             args.reuse_cpu_critic_forward
                         ),
                         native_gae=args.native_gae,
                         trace_cpu_rollout_actor=args.trace_cpu_rollout_actor,
                         native_smart_policy=args.native_smart_policy,
                         native_rollout_engine=args.native_rollout_engine,
                         record_rollout_public_events=(
                             args.record_rollout_public_events
                         ),
                         direct_rollout_sampling=args.direct_rollout_sampling,
                         numpy_rollout_masks=args.numpy_rollout_masks,
                         reuse_rollout_workspaces=args.reuse_rollout_workspaces,
                         coalesce_rollout_agent_sets=(
                             args.coalesce_rollout_agent_sets
                         ),
                         group_compatible_rollout_agent_sets=(
                             args.group_compatible_rollout_agent_sets
                         ),
                         coordinate_rollout_engines=(
                             args.coordinate_rollout_engines
                         ),
                         drain_non_neural_rollout=(
                             args.drain_non_neural_rollout
                         ),
                         reuse_rollout_engines=(
                             args.reuse_rollout_engines
                         ),
                         past_policy_pool_size=args.past_policy_pool_size,
                         past_policy_probability=args.past_policy_probability,
                         torch_threads=args.torch_threads,
                         evaluation_workers=args.evaluation_workers,
                         evaluation_start_method=(
                             args.evaluation_start_method
                         ))
    ppo.save_freq = args.save_freq
    ppo.learn(num_iterations=args.iterations, episodes_per_iter=args.episodes, temperature=args.temperature,
              eval_freq=args.eval_freq, eval_episodes=args.eval_episodes, agent_sets=agent_sets)


if __name__ == "__main__":
    main()
