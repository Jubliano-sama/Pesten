#!/usr/bin/env python3
"""Packed PPO and a short privacy-safe league driver for temporal model-zoo policies.

Only :class:`DecisionInput` and already-redacted :class:`PublicEventToken`
values enter this module.  Whole episodes are independent batch rows, so
padding cannot join histories or leak state across episode boundaries.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

from temporal_model_zoo import (
    CausalTransformerTemporalPolicy,
    LargeRecurrentTemporalPolicy,
    SafeSequenceBatch,
    SafeTemporalPolicy,
    decision_safe_tokens,
    pack_safe_episodes,
)
from temporal_observation import DecisionInput, PublicEventToken
from temporal_puffer_agent import TrajectoryStep, temporal_agent_factory


SEALED_SEED_START = 92_000_000
DEFAULT_TEMPORAL_LEAGUE_MIXTURE = (
    ("current_selfplay", 4),
    ("historical_temporal", 2),
    ("ppo_champion", 2),
    ("bayesian_tree", 1),
    ("smart_rule", 1),
    ("smarter_rule", 0),
)


@contextmanager
def cpu_rollout_thread_scope(
    rollout_device: torch.device | str,
    cpu_threads: int,
):
    """Temporarily right-size Torch threads for sequential CPU inference.

    Recurrent legacy-Game rollouts issue many batch-one GRU and linear calls.
    Large intra-op pools spend substantially more time dispatching workers than
    doing arithmetic.  The setting is process-global, so keep the scope narrow
    and restore the training configuration even when collection raises.
    """
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    device = torch.device(rollout_device)
    if device.type != "cpu":
        yield
        return
    previous_threads = torch.get_num_threads()
    if previous_threads == cpu_threads:
        yield
        return
    torch.set_num_threads(cpu_threads)
    try:
        yield
    finally:
        torch.set_num_threads(previous_threads)


def run_rollout_operation(
    operation,
    *operation_args,
    rollout_device: torch.device | str,
    cpu_threads: int,
    **operation_kwargs,
):
    """Run one collection/evaluation call with scoped CPU thread control."""
    with cpu_rollout_thread_scope(rollout_device, cpu_threads):
        return operation(*operation_args, **operation_kwargs)


def configured_league_mixture(args) -> tuple[tuple[str, int], ...]:
    """Return CLI league weights in a stable, reproducible schedule order."""
    return (
        ("current_selfplay", args.league_current_weight),
        ("historical_temporal", args.league_historical_weight),
        ("ppo_champion", args.league_champion_weight),
        ("bayesian_tree", args.league_tree_weight),
        ("smart_rule", args.league_smart_weight),
        ("smarter_rule", args.league_smarter_weight),
    )


@dataclass(frozen=True, slots=True)
class PackedPPOBatch:
    """Safe sequence inputs plus aligned fixed behavior targets."""

    safe: SafeSequenceBatch
    old_log_probabilities: torch.Tensor
    behavior_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    episode_lengths: tuple[int, ...]
    episode_decision_offsets: tuple[int, ...]

    @property
    def sample_count(self) -> int:
        return int(self.safe.decision_count)


@dataclass(frozen=True, slots=True)
class PPOConfig:
    gamma: float = 0.97
    gae_lambda: float = 0.95
    step_reward: float = -0.01
    win_reward: float = 1.0
    loss_reward: float = -1.0
    clip_ratio: float = 0.2
    value_clip: float | None = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_grad_norm: float = 0.5
    normalize_advantages: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must be between zero and one")
        if self.clip_ratio < 0.0:
            raise ValueError("clip_ratio cannot be negative")
        if self.value_clip is not None and self.value_clip < 0.0:
            raise ValueError("value_clip cannot be negative")
        if self.value_coefficient < 0.0:
            raise ValueError("value_coefficient cannot be negative")
        if self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient cannot be negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")


def validate_development_seed(seed: int, *, span: int = 1) -> None:
    """Keep collection and derived seeds out of the sealed final range."""
    if seed < 0 or span < 1 or seed + span - 1 >= SEALED_SEED_START:
        raise ValueError(
            f"development seeds must stay in [0, {SEALED_SEED_START})"
        )


def _validate_trajectory(
    trajectory: Sequence[TrajectoryStep],
) -> None:
    for step in trajectory:
        if not isinstance(step, TrajectoryStep):
            raise TypeError("episodes must contain TrajectoryStep values")
        if not isinstance(step.decision, DecisionInput):
            raise TypeError("TrajectoryStep.decision must be DecisionInput")
        if any(
            not isinstance(event, PublicEventToken)
            for event in step.decision.public_events
        ):
            raise TypeError("decision events must be PublicEventToken values")
        if not 0 <= step.action < len(step.decision.legal_actions):
            raise ValueError(f"action is out of range: {step.action}")
        if not step.decision.legal_actions[step.action]:
            raise ValueError(f"behavior action {step.action} is not legal")
        if not math.isfinite(step.old_log_prob):
            raise ValueError("old_log_prob must be finite")
        if not math.isfinite(step.value):
            raise ValueError("behavior value must be finite")


def episode_safe_token_count(
    trajectory: Sequence[TrajectoryStep],
) -> int:
    _validate_trajectory(trajectory)
    return sum(
        len(decision_safe_tokens(step.decision)) for step in trajectory
    )


def gae_returns(
    trajectory: Sequence[TrajectoryStep],
    *,
    won: bool,
    config: PPOConfig = PPOConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute terminal, episode-local GAE targets from behavior values."""
    _validate_trajectory(trajectory)
    count = len(trajectory)
    if not count:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty
    rewards = torch.full(
        (count,), config.step_reward, dtype=torch.float32
    )
    rewards[-1] += config.win_reward if won else config.loss_reward
    behavior_values = torch.tensor(
        [step.value for step in trajectory], dtype=torch.float32
    )
    advantages = torch.empty_like(behavior_values)
    next_value = torch.tensor(0.0)
    running = torch.tensor(0.0)
    for index in range(count - 1, -1, -1):
        delta = (
            rewards[index]
            + config.gamma * next_value
            - behavior_values[index]
        )
        running = (
            delta
            + config.gamma * config.gae_lambda * running
        )
        advantages[index] = running
        next_value = behavior_values[index]
    return advantages + behavior_values, advantages


def advantage_statistics(
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    *,
    config: PPOConfig,
) -> tuple[float, float]:
    values = [
        advantages
        for trajectory, won in episodes
        if len(trajectory)
        for advantages in [gae_returns(trajectory, won=won, config=config)[1]]
    ]
    if not values:
        raise ValueError("PPO requires at least one decision")
    flat = torch.cat(values)
    return float(flat.mean()), float(flat.std(unbiased=False))


def pack_ppo_episodes(
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    config: PPOConfig = PPOConfig(),
    advantage_mean_std: tuple[float, float] | None = None,
) -> PackedPPOBatch:
    """Pack complete episodes without crossing or reordering their histories."""
    usable = [
        (tuple(trajectory), bool(won))
        for trajectory, won in episodes
        if trajectory
    ]
    if not usable:
        raise ValueError("PPO requires at least one decision")
    for trajectory, _ in usable:
        _validate_trajectory(trajectory)

    safe = pack_safe_episodes(
        [trajectory for trajectory, _ in usable],
        device=device,
        dtype=dtype,
    )
    old_log_probabilities = []
    behavior_values = []
    return_rows = []
    advantage_rows = []
    episode_lengths = []
    offsets = [0]
    for trajectory, won in usable:
        returns, advantages = gae_returns(
            trajectory,
            won=won,
            config=config,
        )
        old_log_probabilities.extend(
            step.old_log_prob for step in trajectory
        )
        behavior_values.extend(step.value for step in trajectory)
        return_rows.append(returns)
        advantage_rows.append(advantages)
        episode_lengths.append(len(trajectory))
        offsets.append(offsets[-1] + len(trajectory))

    advantages = torch.cat(advantage_rows)
    if config.normalize_advantages and advantages.numel() > 1:
        mean, std = (
            advantage_mean_std
            if advantage_mean_std is not None
            else (
                float(advantages.mean()),
                float(advantages.std(unbiased=False)),
            )
        )
        advantages = (advantages - mean) / (std + 1e-8)

    return PackedPPOBatch(
        safe=safe,
        old_log_probabilities=torch.tensor(
            old_log_probabilities, device=device, dtype=dtype
        ),
        behavior_values=torch.tensor(
            behavior_values, device=device, dtype=dtype
        ),
        returns=torch.cat(return_rows).to(device=device, dtype=dtype),
        advantages=advantages.to(device=device, dtype=dtype),
        episode_lengths=tuple(episode_lengths),
        episode_decision_offsets=tuple(offsets),
    )


def _ppo_objective(
    policy: SafeTemporalPolicy,
    batch: PackedPPOBatch,
    *,
    config: PPOConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, values = policy.forward_packed(batch.safe)
    distribution = Categorical(logits=logits)
    new_log_probabilities = distribution.log_prob(batch.safe.targets)
    log_ratios = (
        new_log_probabilities - batch.old_log_probabilities
    )
    ratios = log_ratios.exp()
    unclipped = ratios * batch.advantages
    clipped = torch.clamp(
        ratios,
        1.0 - config.clip_ratio,
        1.0 + config.clip_ratio,
    ) * batch.advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    value_errors = (values - batch.returns).square()
    if config.value_clip is not None:
        clipped_values = batch.behavior_values + torch.clamp(
            values - batch.behavior_values,
            -config.value_clip,
            config.value_clip,
        )
        value_errors = torch.maximum(
            value_errors,
            (clipped_values - batch.returns).square(),
        )
    value_loss = value_errors.mean()
    entropy = distribution.entropy().mean()
    loss = (
        policy_loss
        + config.value_coefficient * value_loss
        - config.entropy_coefficient * entropy
    )
    target_variance = batch.returns.var(unbiased=False)
    explained_variance = (
        torch.full((), float("nan"), device=values.device)
        if float(target_variance.detach()) == 0.0
        else 1.0
        - (batch.returns - values).var(unbiased=False)
        / target_variance
    )
    metrics = {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "old_approx_kl": (-log_ratios).mean(),
        "approx_kl": ((ratios - 1.0) - log_ratios).mean(),
        "clip_fraction": (
            (ratios - 1.0).abs() > config.clip_ratio
        ).float().mean(),
        "ratio_mean": ratios.mean(),
        "ratio_min": ratios.min(),
        "ratio_max": ratios.max(),
        "explained_variance": explained_variance,
        "predicted_values": values,
    }
    return loss, metrics


def _float_metrics(
    metrics: Mapping[str, torch.Tensor],
    *,
    samples: int,
    safe_tokens: int,
) -> dict[str, float | int]:
    result = {
        key: float(value.detach())
        for key, value in metrics.items()
        if key != "predicted_values"
    }
    result.update({"samples": samples, "safe_tokens": safe_tokens})
    return result


def ppo_update_packed(
    policy: SafeTemporalPolicy,
    batch: PackedPPOBatch,
    optimizer: torch.optim.Optimizer,
    *,
    config: PPOConfig = PPOConfig(),
) -> dict[str, float | int]:
    """Run one GPU-compatible packed PPO optimizer step."""
    if not isinstance(policy, SafeTemporalPolicy):
        raise TypeError("policy must implement SafeTemporalPolicy")
    policy.train()
    loss, tensors = _ppo_objective(policy, batch, config=config)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), config.max_grad_norm
    )
    optimizer.step()
    metrics = _float_metrics(
        tensors,
        samples=batch.sample_count,
        safe_tokens=batch.safe.token_count,
    )
    metrics["grad_norm"] = float(grad_norm.detach())
    return metrics


def evaluate_ppo_drift_packed(
    policy: SafeTemporalPolicy,
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    *,
    config: PPOConfig = PPOConfig(),
    minibatch_episodes: int = 16,
) -> dict[str, float | int]:
    """Evaluate current policy against fixed rollout behavior statistics."""
    usable = [episode for episode in episodes if episode[0]]
    if not usable:
        raise ValueError("PPO requires at least one decision")
    parameter = next(policy.parameters())
    normalization = (
        advantage_statistics(usable, config=config)
        if config.normalize_advantages
        else None
    )
    weighted: dict[str, float] = {}
    predictions = []
    targets = []
    samples = safe_tokens = 0
    was_training = policy.training
    policy.eval()
    with torch.no_grad():
        for start in range(0, len(usable), minibatch_episodes):
            batch = pack_ppo_episodes(
                usable[start : start + minibatch_episodes],
                device=parameter.device,
                dtype=parameter.dtype,
                config=config,
                advantage_mean_std=normalization,
            )
            _, tensors = _ppo_objective(policy, batch, config=config)
            count = batch.sample_count
            for key, value in tensors.items():
                if key in {"predicted_values", "explained_variance"}:
                    continue
                weighted[key] = weighted.get(key, 0.0) + float(value) * count
            predictions.append(tensors["predicted_values"].detach().float().cpu())
            targets.append(batch.returns.detach().float().cpu())
            samples += count
            safe_tokens += batch.safe.token_count
    policy.train(was_training)
    result = {key: value / samples for key, value in weighted.items()}
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    target_variance = target.var(unbiased=False)
    result["explained_variance"] = (
        math.nan
        if float(target_variance) == 0.0
        else float(
            1.0
            - (target - prediction).var(unbiased=False)
            / target_variance
        )
    )
    result.update({"samples": samples, "safe_tokens": safe_tokens})
    return result


def ppo_update_epochs(
    policy: SafeTemporalPolicy,
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    optimizer: torch.optim.Optimizer,
    *,
    config: PPOConfig = PPOConfig(),
    update_epochs: int = 2,
    minibatch_episodes: int = 8,
    seed: int = 0,
    target_kl: float | None = 0.02,
) -> dict:
    """Run shuffled whole-episode PPO minibatches with epoch-level KL stops."""
    validate_development_seed(seed)
    if update_epochs < 1 or minibatch_episodes < 1:
        raise ValueError("epochs and minibatch_episodes must be positive")
    usable = [episode for episode in episodes if episode[0]]
    if not usable:
        raise ValueError("PPO requires at least one decision")
    parameter = next(policy.parameters())
    normalization = (
        advantage_statistics(usable, config=config)
        if config.normalize_advantages
        else None
    )
    rng = random.Random(seed)
    updates = []
    epoch_diagnostics = []
    stopped_early = False
    for epoch in range(update_epochs):
        order = list(range(len(usable)))
        rng.shuffle(order)
        for start in range(0, len(order), minibatch_episodes):
            minibatch = [
                usable[index]
                for index in order[start : start + minibatch_episodes]
            ]
            batch = pack_ppo_episodes(
                minibatch,
                device=parameter.device,
                dtype=parameter.dtype,
                config=config,
                advantage_mean_std=normalization,
            )
            updates.append(
                ppo_update_packed(
                    policy, batch, optimizer, config=config
                )
            )
        diagnostics = evaluate_ppo_drift_packed(
            policy,
            usable,
            config=config,
            minibatch_episodes=minibatch_episodes,
        )
        epoch_diagnostics.append({"epoch": epoch + 1, **diagnostics})
        if (
            target_kl is not None
            and diagnostics["approx_kl"] > target_kl
        ):
            stopped_early = True
            break

    weight = sum(update["samples"] for update in updates)
    names = (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "old_approx_kl",
        "approx_kl",
        "clip_fraction",
        "ratio_mean",
        "explained_variance",
        "grad_norm",
    )
    aggregate = {
        name: sum(
            float(update[name]) * int(update["samples"])
            for update in updates
        )
        / weight
        for name in names
    }
    aggregate.update(
        {
            "samples": sum(len(trajectory) for trajectory, _ in usable),
            "safe_tokens": sum(
                episode_safe_token_count(trajectory)
                for trajectory, _ in usable
            ),
            "optimizer_steps": len(updates),
            "epochs_completed": len(epoch_diagnostics),
            "target_kl": target_kl,
            "stopped_early": stopped_early,
            "post_update_diagnostics": epoch_diagnostics[-1],
            "epoch_diagnostics": epoch_diagnostics,
        }
    )
    return aggregate


def critic_warmup_epochs(
    policy: SafeTemporalPolicy,
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    minibatch_episodes: int = 8,
    seed: int = 0,
) -> dict:
    """Fit Monte-Carlo values without changing any actor-head parameters."""
    validate_development_seed(seed)
    if epochs < 1 or minibatch_episodes < 1:
        raise ValueError("epochs and minibatch_episodes must be positive")
    usable = [episode for episode in episodes if episode[0]]
    if not usable:
        raise ValueError("critic warm-up requires at least one decision")
    parameter = next(policy.parameters())
    actor_parameters = {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.named_parameters()
        if "actor" in name
    }
    monte_carlo = PPOConfig(
        gae_lambda=1.0,
        normalize_advantages=False,
        entropy_coefficient=0.0,
    )
    rng = random.Random(seed)
    losses = []
    samples = 0
    started = time.perf_counter()
    policy.train()
    for _ in range(epochs):
        order = list(range(len(usable)))
        rng.shuffle(order)
        for start in range(0, len(order), minibatch_episodes):
            minibatch = [
                usable[index]
                for index in order[start : start + minibatch_episodes]
            ]
            batch = pack_ppo_episodes(
                minibatch,
                device=parameter.device,
                dtype=parameter.dtype,
                config=monte_carlo,
            )
            _, values = policy.forward_packed(batch.safe)
            loss = torch.nn.functional.smooth_l1_loss(
                values, batch.returns
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    candidate
                    for candidate in policy.parameters()
                    if candidate.grad is not None
                ],
                1.0,
            )
            optimizer.step()
            losses.append(float(loss.detach()))
            samples += batch.sample_count
    current_parameters = dict(policy.named_parameters())
    if any(
        not torch.equal(before, current_parameters[name].detach().cpu())
        for name, before in actor_parameters.items()
    ):
        raise RuntimeError("critic warm-up changed an actor-head parameter")
    return {
        "epochs": epochs,
        "optimizer_steps": len(losses),
        "processed_decisions": samples,
        "mean_loss": float(np.mean(losses)),
        "wall_seconds": time.perf_counter() - started,
    }


def context_compatible_episodes(
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    *,
    max_context: int | None,
) -> tuple[list[tuple[Sequence[TrajectoryStep], bool]], dict]:
    """Keep complete episodes only; never truncate or merge training history."""
    accepted = []
    dropped = 0
    longest = 0
    for episode in episodes:
        token_count = episode_safe_token_count(episode[0])
        longest = max(longest, token_count)
        if not episode[0]:
            continue
        if max_context is not None and token_count > max_context:
            dropped += 1
            continue
        accepted.append(episode)
    return accepted, {
        "accepted_episodes": len(accepted),
        "dropped_over_context": dropped,
        "longest_collected_safe_tokens": longest,
        "max_context": max_context,
    }


def frozen_snapshot(policy: SafeTemporalPolicy) -> SafeTemporalPolicy:
    snapshot = copy.deepcopy(policy).eval()
    for parameter in snapshot.parameters():
        parameter.requires_grad_(False)
    return snapshot


def build_policy(args) -> SafeTemporalPolicy:
    if args.architecture in {"residual_gru", "residual_lstm"}:
        from residual_temporal_policy import (
            ResidualRecurrentTemporalPolicy,
        )

        if args.checkpoint is not None:
            return ResidualRecurrentTemporalPolicy.load_checkpoint(
                args.checkpoint,
                champion_path=args.champion,
                map_location="cpu",
            )
        return ResidualRecurrentTemporalPolicy(
            champion_path=args.champion,
            hidden_size=args.hidden_size,
            num_layers=args.layers,
            cell=args.architecture.removeprefix("residual_"),
            residual_scale=args.residual_scale,
            policy_temperature=args.training_temperature,
        )
    if args.architecture == "residual":
        from residual_temporal_policy import ResidualCausalTemporalPolicy

        if args.checkpoint is not None:
            return ResidualCausalTemporalPolicy.load_checkpoint(
                args.checkpoint,
                champion_path=args.champion,
                map_location="cpu",
            )
        return ResidualCausalTemporalPolicy(
            champion_path=args.champion,
            d_model=args.hidden_size,
            num_layers=args.layers,
            num_heads=args.heads,
            feedforward_size=args.feedforward_size,
            max_context=args.max_context,
            residual_scale=args.residual_scale,
            logit_temperature=args.training_temperature,
        )
    if args.architecture == "causal":
        return CausalTransformerTemporalPolicy(
            d_model=args.hidden_size,
            num_layers=args.layers,
            num_heads=args.heads,
            feedforward_size=args.feedforward_size,
            max_context=args.max_context,
        )
    return LargeRecurrentTemporalPolicy(
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        cell=args.architecture,
    )


def load_checkpoint(policy: SafeTemporalPolicy, path: Path) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, Mapping):
        for key in ("policy_state_dict", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    policy.load_state_dict(payload, strict=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--champion",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument("--save", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--architecture",
        choices=(
            "causal",
            "gru",
            "lstm",
            "residual",
            "residual_gru",
            "residual_lstm",
        ),
        default="causal",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--feedforward-size", type=int, default=384)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--training-temperature", type=float, default=1.0)
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--rollout-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="device for sequential game inference; CPU is faster for small batches",
    )
    parser.add_argument(
        "--rollout-cpu-threads",
        type=int,
        default=1,
        help=(
            "Torch intra-op threads during sequential CPU rollout calls; "
            "batch-one recurrent inference is fastest with 1"
        ),
    )
    parser.add_argument("--seed", type=int, default=27_301)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--games-per-iteration", type=int, default=24)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-episodes", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--target-kl", type=float, default=0.015)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--eval-pairs", type=int, default=8)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument(
        "--league-current-weight",
        type=int,
        default=4,
        help="current-policy self-play slots in the deterministic league cycle",
    )
    parser.add_argument(
        "--league-historical-weight",
        type=int,
        default=2,
        help="frozen historical temporal-policy slots",
    )
    parser.add_argument(
        "--league-champion-weight",
        type=int,
        default=2,
        help="frozen feed-forward PPO champion slots",
    )
    parser.add_argument(
        "--league-tree-weight",
        type=int,
        default=1,
        help="public-information Bayesian tree-policy slots",
    )
    parser.add_argument(
        "--league-smart-weight",
        type=int,
        default=1,
        help="legacy SmartAgent rule-policy slots",
    )
    parser.add_argument(
        "--league-smarter-weight",
        type=int,
        default=0,
        help="stronger SmarterAgent rule-policy slots; opt-in by default",
    )
    parser.add_argument("--critic-warmup-games", type=int, default=0)
    parser.add_argument("--critic-warmup-epochs", type=int, default=3)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--checkpoint-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    # Keep packed PPO importable in the lightweight project environment. The
    # league dependency is needed only by the executable training driver.
    from benchmark_agents import load_ppo_agent
    from temporal_league import (
        CHAMPION_PATH,
        collect_league_batch,
        evaluate_paired,
        league_schedule,
        validate_league_mixture,
    )

    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if args.rollout_device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA rollout was requested but is unavailable")
    if args.rollout_cpu_threads < 1:
        raise SystemExit("rollout-cpu-threads must be positive")
    if min(
        args.iterations,
        args.games_per_iteration,
        args.update_epochs,
        args.minibatch_episodes,
        args.max_context,
    ) < 1:
        raise SystemExit("iteration, batch, and context sizes must be positive")
    if args.eval_pairs < 0 or args.history_size < 1:
        raise SystemExit("eval-pairs cannot be negative; history-size must be positive")
    if args.critic_warmup_games < 0 or args.critic_warmup_epochs < 1:
        raise SystemExit(
            "critic-warmup-games cannot be negative and epochs must be positive"
        )
    if args.checkpoint_every < 0:
        raise SystemExit("checkpoint-every cannot be negative")
    try:
        league_mixture = validate_league_mixture(
            configured_league_mixture(args)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    deterministic_league_schedule = league_schedule(league_mixture)
    seed_span = (
        args.iterations * 10_000
        + args.games_per_iteration
        + 6_000_000
        + max(1, args.eval_pairs)
    )
    validate_development_seed(args.seed, span=seed_span)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = build_policy(args)
    if not args.architecture.startswith("residual"):
        if args.checkpoint is None:
            raise SystemExit(
                "--checkpoint is required for non-residual architectures"
            )
        load_checkpoint(policy, args.checkpoint)
    policy.to(args.device)
    rollout_policy = copy.deepcopy(policy).to(args.rollout_device).eval()

    def sync_rollout_policy() -> None:
        rollout_policy.load_state_dict(policy.state_dict())
        rollout_policy.eval()

    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in policy.parameters()
            if parameter.requires_grad
        ],
        lr=args.learning_rate,
    )
    ppo_config = PPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
    )
    champion_factory = load_ppo_agent(
        CHAMPION_PATH, args.rollout_device
    )
    historical = [frozen_snapshot(rollout_policy)]
    candidate = temporal_agent_factory(
        rollout_policy, deterministic=True, seed=args.seed
    )
    initial_evaluation = (
        run_rollout_operation(
            evaluate_paired,
            candidate,
            champion_factory,
            pairs=args.eval_pairs,
            seed=args.seed + 5_000_000,
            candidate_name=policy.architecture,
            opponent_name="ppo_champion",
            rollout_device=args.rollout_device,
            cpu_threads=args.rollout_cpu_threads,
        )
        if args.eval_pairs
        else None
    )

    started = time.perf_counter()
    critic_warmup = None
    schedule_offset = 0
    if args.critic_warmup_games:
        warmup_episodes, warmup_contribution = run_rollout_operation(
            collect_league_batch,
            rollout_policy,
            champion_factory,
            historical,
            games=args.critic_warmup_games,
            seed=args.seed + 4_000_000,
            schedule_offset=schedule_offset,
            mixture=league_mixture,
            rollout_device=args.rollout_device,
            cpu_threads=args.rollout_cpu_threads,
        )
        schedule_offset += args.critic_warmup_games
        warmup_compatible, warmup_context = context_compatible_episodes(
            warmup_episodes,
            max_context=(
                args.max_context
                if args.architecture in {"causal", "residual"}
                else None
            ),
        )
        critic_warmup = {
            "contribution": warmup_contribution,
            "context_filter": warmup_context,
            "training": critic_warmup_epochs(
                policy,
                warmup_compatible,
                optimizer,
                epochs=args.critic_warmup_epochs,
                minibatch_episodes=args.minibatch_episodes,
                seed=args.seed + 4_100_000,
            ),
        }
        sync_rollout_policy()
        historical = [frozen_snapshot(rollout_policy)]
    iterations = []
    for iteration in range(args.iterations):
        collection_seed = args.seed + iteration * 10_000
        episodes, contribution = run_rollout_operation(
            collect_league_batch,
            rollout_policy,
            champion_factory,
            historical,
            games=args.games_per_iteration,
            seed=collection_seed,
            schedule_offset=schedule_offset,
            mixture=league_mixture,
            rollout_device=args.rollout_device,
            cpu_threads=args.rollout_cpu_threads,
        )
        schedule_offset += args.games_per_iteration
        compatible, context = context_compatible_episodes(
            episodes,
            max_context=(
                args.max_context
                if args.architecture in {"causal", "residual"}
                else None
            ),
        )
        if not compatible:
            raise RuntimeError(
                "all collected trajectories exceeded max_context; "
                "increase --max-context"
            )
        before = evaluate_ppo_drift_packed(
            policy,
            compatible,
            config=ppo_config,
            minibatch_episodes=args.minibatch_episodes,
        )
        update = ppo_update_epochs(
            policy,
            compatible,
            optimizer,
            config=ppo_config,
            update_epochs=args.update_epochs,
            minibatch_episodes=args.minibatch_episodes,
            seed=collection_seed + args.games_per_iteration,
            target_kl=args.target_kl,
        )
        sync_rollout_policy()
        historical.append(frozen_snapshot(rollout_policy))
        historical = historical[-args.history_size :]
        iteration_record = {
            "iteration": iteration + 1,
            "collection_seed": collection_seed,
            "contribution": contribution,
            "context_filter": context,
            "pre_update": before,
            "update": update,
        }
        if (
            args.checkpoint_every
            and (iteration + 1) % args.checkpoint_every == 0
        ):
            checkpoint_dir = args.checkpoint_dir or args.save.parent
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = (
                checkpoint_dir
                / f"{args.save.stem}_iter{iteration + 1}{args.save.suffix}"
            )
            if args.architecture.startswith("residual"):
                policy.save_checkpoint(checkpoint_path)
            else:
                torch.save(policy.state_dict(), checkpoint_path)
            iteration_record["checkpoint"] = str(checkpoint_path)
        iterations.append(iteration_record)

    candidate = temporal_agent_factory(
        rollout_policy, deterministic=True, seed=args.seed + 1
    )
    final_evaluation = (
        run_rollout_operation(
            evaluate_paired,
            candidate,
            champion_factory,
            pairs=args.eval_pairs,
            seed=args.seed + 5_500_000,
            candidate_name=policy.architecture,
            opponent_name="ppo_champion",
            rollout_device=args.rollout_device,
            cpu_threads=args.rollout_cpu_threads,
        )
        if args.eval_pairs
        else None
    )
    args.save.parent.mkdir(parents=True, exist_ok=True)
    if args.architecture.startswith("residual"):
        policy.save_checkpoint(args.save)
    else:
        torch.save(policy.state_dict(), args.save)
    payload = {
        "schema_version": 1,
        "scope": "short development-only packed temporal PPO fine-tuning",
        "architecture": policy.architecture,
        "checkpoint_in": (
            str(args.checkpoint) if args.checkpoint is not None else None
        ),
        "checkpoint_out": str(args.save),
        "device": args.device,
        "rollout_device": args.rollout_device,
        "rollout_cpu_threads": (
            args.rollout_cpu_threads
            if args.rollout_device == "cpu"
            else None
        ),
        "league_mixture": dict(league_mixture),
        "league_schedule": list(deterministic_league_schedule),
        "max_context": args.max_context,
        "seed": args.seed,
        "sealed_seed_start": SEALED_SEED_START,
        "information_boundary": (
            "DecisionInput plus redacted PublicEventToken only; no opponent "
            "identities or draw-pile order"
        ),
        "ppo": asdict(ppo_config),
        "critic_warmup": critic_warmup,
        "initial_evaluation": initial_evaluation,
        "iterations": iterations,
        "final_evaluation": final_evaluation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
