"""Focused rollout and PPO smoke training for the temporal Pesten policy."""

from __future__ import annotations

import json
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch
from torch.distributions import Categorical

import game
from randomAgent import AgentRandom
from temporal_puffer_agent import (
    PufferStyleTemporalNet,
    TrajectoryStep,
    temporal_agent_factory,
)


def _pack_native_token_batch(
    packed_rows: Sequence[Sequence[tuple[float, ...]]],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Pad ragged safe-token rows with one host-to-device materialization."""
    max_length = max(len(rows) for rows in packed_rows)
    token_size = len(packed_rows[0][0])
    token_batch = torch.zeros(
        len(packed_rows),
        max_length,
        token_size,
        device=device,
        dtype=dtype,
    )
    flat_rows = [row for rows in packed_rows for row in rows]
    flat_tokens = torch.tensor(
        flat_rows,
        device=device,
        dtype=dtype,
    )
    offset = 0
    for batch_index, rows in enumerate(packed_rows):
        length = len(rows)
        token_batch[batch_index, :length] = flat_tokens[
            offset : offset + length
        ]
        offset += length
    return token_batch


def _gather_native_decisions(
    all_logits: torch.Tensor,
    all_values: torch.Tensor,
    decision_indices: Sequence[Sequence[int]],
    *,
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather ragged decision positions in trajectory order with one index."""
    flat_indices = [
        batch_index * max_length + time_index
        for batch_index, indices in enumerate(decision_indices)
        for time_index in indices
    ]
    index_tensor = torch.tensor(
        flat_indices,
        device=device,
        dtype=torch.long,
    )
    flat_logits = all_logits.reshape(-1, all_logits.shape[-1])
    flat_values = all_values.reshape(-1)
    return (
        flat_logits.index_select(0, index_tensor),
        flat_values.index_select(0, index_tensor),
    )


def _discounted_returns(
    step_count: int,
    *,
    won: bool,
    gamma: float = 0.97,
) -> torch.Tensor:
    rewards = torch.full((step_count,), -0.01, dtype=torch.float32)
    if step_count:
        rewards[-1] += 1.0 if won else -1.0
    returns = torch.empty_like(rewards)
    running = torch.tensor(0.0)
    for index in reversed(range(step_count)):
        running = rewards[index] + gamma * running
        returns[index] = running
    return returns


def _gae_targets(
    trajectory: Sequence[TrajectoryStep],
    *,
    won: bool,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed behavior-value targets and generalized advantages."""
    step_count = len(trajectory)
    rewards = torch.full((step_count,), -0.01, dtype=torch.float32)
    rewards[-1] += 1.0 if won else -1.0
    behavior_values = torch.tensor(
        [step.value for step in trajectory], dtype=torch.float32
    )
    advantages = torch.empty_like(rewards)
    next_value = torch.tensor(0.0)
    running_advantage = torch.tensor(0.0)
    for index in reversed(range(step_count)):
        delta = (
            rewards[index]
            + gamma * next_value
            - behavior_values[index]
        )
        running_advantage = (
            delta + gamma * gae_lambda * running_advantage
        )
        advantages[index] = running_advantage
        next_value = behavior_values[index]
    return advantages + behavior_values, advantages


def ppo_update(
    policy: PufferStyleTemporalNet,
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    optimizer: torch.optim.Optimizer,
    *,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> dict[str, float]:
    """Perform one recurrent PPO update using safe decisions only."""
    policy.train()
    policy_losses = []
    value_losses = []
    entropies = []
    log_ratios = []
    clip_fractions = []
    predicted_values = []
    target_returns = []
    sample_count = 0

    for trajectory, won in episodes:
        if not trajectory:
            continue
        returns = _discounted_returns(len(trajectory), won=won).to(
            next(policy.parameters()).device
        )
        state = policy.initial_state()
        for index, old_step in enumerate(trajectory):
            result = policy.step(old_step.decision, state)
            state = result.state
            distribution = Categorical(logits=result.logits)
            action = torch.tensor(
                old_step.action, device=result.logits.device
            )
            new_log_prob = distribution.log_prob(action)
            old_log_prob = torch.tensor(
                old_step.old_log_prob, device=result.logits.device
            )
            advantage = returns[index] - result.value.detach()
            ratio = torch.exp(new_log_prob - old_log_prob)
            log_ratio = new_log_prob - old_log_prob
            unclipped = ratio * advantage
            clipped = torch.clamp(
                ratio, 1.0 - clip_ratio, 1.0 + clip_ratio
            ) * advantage
            policy_losses.append(-torch.minimum(unclipped, clipped))
            value_losses.append((result.value - returns[index]).square())
            entropies.append(distribution.entropy())
            log_ratios.append(log_ratio)
            clip_fractions.append(
                ((ratio - 1.0).abs() > clip_ratio).float()
            )
            predicted_values.append(result.value)
            target_returns.append(returns[index])
            sample_count += 1

    if not sample_count:
        raise ValueError("PPO update requires at least one decision")

    policy_loss = torch.stack(policy_losses).mean()
    value_loss = torch.stack(value_losses).mean()
    entropy = torch.stack(entropies).mean()
    stacked_log_ratios = torch.stack(log_ratios)
    stacked_ratios = stacked_log_ratios.exp()
    old_approx_kl = (-stacked_log_ratios).mean()
    approx_kl = ((stacked_ratios - 1.0) - stacked_log_ratios).mean()
    clip_fraction = torch.stack(clip_fractions).mean()
    predictions = torch.stack(predicted_values)
    targets = torch.stack(target_returns)
    target_variance = targets.var(unbiased=False)
    explained_variance = (
        torch.tensor(float("nan"))
        if target_variance == 0
        else 1.0
        - (targets - predictions).var(unbiased=False) / target_variance
    )
    loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "old_approx_kl": float(old_approx_kl.detach()),
        "approx_kl": float(approx_kl.detach()),
        "clip_fraction": float(clip_fraction.detach()),
        "explained_variance": float(explained_variance.detach()),
        "samples": sample_count,
    }


def ppo_update_batched(
    policy,
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    optimizer: torch.optim.Optimizer,
    *,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    gamma: float = 0.97,
    gae_lambda: float | None = None,
    normalize_advantages: bool = False,
) -> dict[str, float]:
    """Perform the same PPO update through native MinGRU sequence batches.

    Every trajectory is one independent batch row with a zero initial state.
    Public event tokens and private decision tokens retain their original
    order; losses are gathered only at decision positions.  Padding is applied
    strictly after the final decision and therefore cannot affect any gathered
    causal recurrent output.
    """
    required = (
        "_event_features",
        "_decision_features",
        "forward_native_sequence",
    )
    if any(not hasattr(policy, name) for name in required):
        raise TypeError(
            "ppo_update_batched requires NativePufferTemporalPolicy"
        )

    policy.train()
    parameter = next(policy.parameters())
    packed_rows: list[list[tuple[float, ...]]] = []
    decision_indices: list[list[int]] = []
    legal_masks: list[tuple[tuple[bool, ...], ...]] = []
    action_rows: list[tuple[int, ...]] = []
    old_log_probability_rows: list[tuple[float, ...]] = []
    return_rows: list[torch.Tensor] = []
    advantage_rows: list[torch.Tensor] = []

    for trajectory, won in episodes:
        if not trajectory:
            continue
        rows: list[tuple[float, ...]] = []
        indices: list[int] = []
        masks: list[tuple[bool, ...]] = []
        actions: list[int] = []
        old_log_probabilities: list[float] = []
        for old_step in trajectory:
            rows.extend(policy._event_features(old_step.decision))
            rows.append(policy._decision_features(old_step.decision))
            indices.append(len(rows) - 1)
            masks.append(old_step.decision.legal_actions)
            actions.append(old_step.action)
            old_log_probabilities.append(old_step.old_log_prob)
        packed_rows.append(rows)
        decision_indices.append(indices)
        legal_masks.append(tuple(masks))
        action_rows.append(tuple(actions))
        old_log_probability_rows.append(tuple(old_log_probabilities))
        if gae_lambda is None:
            targets = _discounted_returns(
                len(trajectory), won=won, gamma=gamma
            )
        else:
            targets, advantages = _gae_targets(
                trajectory,
                won=won,
                gamma=gamma,
                gae_lambda=gae_lambda,
            )
            advantage_rows.append(
                advantages.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )
        return_rows.append(
            targets.to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )

    sample_count = sum(len(indices) for indices in decision_indices)
    if not sample_count:
        raise ValueError("PPO update requires at least one decision")

    max_length = max(len(rows) for rows in packed_rows)
    token_batch = _pack_native_token_batch(
        packed_rows,
        device=parameter.device,
        dtype=parameter.dtype,
    )

    all_logits, all_values = policy.forward_native_sequence(token_batch)
    all_logits = all_logits.reshape(
        len(packed_rows), max_length, -1
    )
    all_values = all_values.reshape(len(packed_rows), max_length)

    gathered_masks = []
    gathered_actions = []
    gathered_old_log_probabilities = []
    for batch_index, indices in enumerate(decision_indices):
        gathered_masks.extend(legal_masks[batch_index])
        gathered_actions.extend(action_rows[batch_index])
        gathered_old_log_probabilities.extend(
            old_log_probability_rows[batch_index]
        )

    logits, values = _gather_native_decisions(
        all_logits,
        all_values,
        decision_indices,
        max_length=max_length,
        device=parameter.device,
    )
    masks = torch.tensor(
        gathered_masks, device=parameter.device, dtype=torch.bool
    )
    logits = logits.masked_fill(
        ~masks, torch.finfo(logits.dtype).min
    )
    actions = torch.tensor(
        gathered_actions, device=parameter.device, dtype=torch.long
    )
    old_log_probabilities = torch.tensor(
        gathered_old_log_probabilities,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    returns = torch.cat(return_rows)

    distribution = Categorical(logits=logits)
    new_log_probabilities = distribution.log_prob(actions)
    log_ratios = new_log_probabilities - old_log_probabilities
    ratios = log_ratios.exp()
    advantages = (
        torch.cat(advantage_rows)
        if gae_lambda is not None
        else returns - values.detach()
    )
    if normalize_advantages and advantages.numel() > 1:
        advantages = (
            advantages - advantages.mean()
        ) / (advantages.std(unbiased=False) + 1e-8)
    unclipped = ratios * advantages
    clipped = torch.clamp(
        ratios, 1.0 - clip_ratio, 1.0 + clip_ratio
    ) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = (values - returns).square().mean()
    entropy = distribution.entropy().mean()
    old_approx_kl = (-log_ratios).mean()
    approx_kl = ((ratios - 1.0) - log_ratios).mean()
    clip_fraction = (
        (ratios - 1.0).abs() > clip_ratio
    ).float().mean()
    target_variance = returns.var(unbiased=False)
    explained_variance = (
        torch.tensor(float("nan"), device=parameter.device)
        if target_variance == 0
        else 1.0
        - (returns - values).var(unbiased=False) / target_variance
    )
    loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
    optimizer.step()
    return {
        "loss": float(loss.detach()),
        "policy_loss": float(policy_loss.detach()),
        "value_loss": float(value_loss.detach()),
        "entropy": float(entropy.detach()),
        "old_approx_kl": float(old_approx_kl.detach()),
        "approx_kl": float(approx_kl.detach()),
        "clip_fraction": float(clip_fraction.detach()),
        "explained_variance": float(explained_variance.detach()),
        "samples": sample_count,
    }


def ppo_update_batched_epochs(
    policy,
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    optimizer: torch.optim.Optimizer,
    *,
    update_epochs: int = 4,
    minibatch_trajectories: int = 16,
    seed: int = 0,
    target_kl: float | None = 0.02,
    clip_ratio: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    gamma: float = 0.97,
    gae_lambda: float | None = 0.95,
    normalize_advantages: bool = True,
) -> dict:
    """Run reproducible PPO epochs/minibatches with a rollout-level KL stop."""
    if update_epochs < 1:
        raise ValueError("update_epochs must be positive")
    if minibatch_trajectories < 1:
        raise ValueError("minibatch_trajectories must be positive")
    usable = [episode for episode in episodes if episode[0]]
    if not usable:
        raise ValueError("PPO update requires at least one decision")
    rng = random.Random(seed)
    update_logs = []
    epoch_diagnostics = []
    stopped_early = False
    for epoch in range(update_epochs):
        order = list(range(len(usable)))
        rng.shuffle(order)
        for start in range(0, len(order), minibatch_trajectories):
            minibatch = [
                usable[index]
                for index in order[
                    start : start + minibatch_trajectories
                ]
            ]
            update_logs.append(
                ppo_update_batched(
                    policy,
                    minibatch,
                    optimizer,
                    clip_ratio=clip_ratio,
                    value_coefficient=value_coefficient,
                    entropy_coefficient=entropy_coefficient,
                    gamma=gamma,
                    gae_lambda=gae_lambda,
                    normalize_advantages=normalize_advantages,
                )
            )
        diagnostics = evaluate_ppo_drift_batched(
            policy, usable, clip_ratio=clip_ratio
        )
        epoch_diagnostics.append(
            {"epoch": epoch + 1, **diagnostics}
        )
        if (
            target_kl is not None
            and diagnostics["approx_kl"] > target_kl
        ):
            stopped_early = True
            break

    weighted_names = (
        "loss",
        "policy_loss",
        "value_loss",
        "entropy",
        "old_approx_kl",
        "approx_kl",
        "clip_fraction",
        "explained_variance",
    )
    weight = sum(item["samples"] for item in update_logs)
    aggregate = {
        name: sum(
            item[name] * item["samples"] for item in update_logs
        )
        / weight
        for name in weighted_names
    }
    aggregate.update(
        {
            "samples": sum(len(trajectory) for trajectory, _ in usable),
            "optimizer_steps": len(update_logs),
            "epochs_completed": len(epoch_diagnostics),
            "target_kl": target_kl,
            "stopped_early": stopped_early,
            "post_update_diagnostics": epoch_diagnostics[-1],
            "epoch_diagnostics": epoch_diagnostics,
        }
    )
    return aggregate


def evaluate_ppo_drift(
    policy,
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    *,
    clip_ratio: float = 0.2,
) -> dict[str, float]:
    """Comparable post-update diagnostics against the collected behavior."""
    log_ratios = []
    predictions = []
    targets = []
    entropies = []
    with torch.no_grad():
        for trajectory, won in episodes:
            if not trajectory:
                continue
            returns = _discounted_returns(
                len(trajectory), won=won
            ).to(next(policy.parameters()).device)
            state = policy.initial_state()
            for index, old_step in enumerate(trajectory):
                result = policy.step(old_step.decision, state)
                state = tuple(
                    part.detach() for part in result.state
                ) if not isinstance(result.state, torch.Tensor) else result.state.detach()
                distribution = Categorical(logits=result.logits)
                action = torch.tensor(
                    old_step.action, device=result.logits.device
                )
                log_ratios.append(
                    distribution.log_prob(action)
                    - torch.tensor(
                        old_step.old_log_prob,
                        device=result.logits.device,
                    )
                )
                predictions.append(result.value)
                targets.append(returns[index])
                entropies.append(distribution.entropy())
    logs = torch.stack(log_ratios)
    ratios = logs.exp()
    predicted = torch.stack(predictions)
    target = torch.stack(targets)
    variance = target.var(unbiased=False)
    explained = (
        torch.tensor(float("nan"), device=target.device)
        if variance == 0
        else 1
        - (target - predicted).var(unbiased=False) / variance
    )
    return {
        "approx_kl": float(
            ((ratios - 1.0) - logs).mean().detach()
        ),
        "old_approx_kl": float((-logs).mean().detach()),
        "clip_fraction": float(
            ((ratios - 1.0).abs() > clip_ratio).float().mean().detach()
        ),
        "entropy": float(torch.stack(entropies).mean().detach()),
        "value_mse": float((predicted - target).square().mean().detach()),
        "explained_variance": float(explained.detach()),
        "samples": len(log_ratios),
    }


def evaluate_ppo_drift_batched(
    policy,
    episodes: Sequence[tuple[Sequence[TrajectoryStep], bool]],
    *,
    clip_ratio: float = 0.2,
) -> dict[str, float]:
    """Evaluate post-update drift through one native MinGRU sequence batch."""
    required = (
        "_event_features",
        "_decision_features",
        "forward_native_sequence",
    )
    if any(not hasattr(policy, name) for name in required):
        raise TypeError(
            "evaluate_ppo_drift_batched requires "
            "NativePufferTemporalPolicy"
        )
    parameter = next(policy.parameters())
    packed_rows = []
    decision_indices = []
    legal_masks = []
    actions = []
    old_log_probabilities = []
    returns = []
    for trajectory, won in episodes:
        if not trajectory:
            continue
        rows = []
        indices = []
        masks = []
        for old_step in trajectory:
            rows.extend(policy._event_features(old_step.decision))
            rows.append(policy._decision_features(old_step.decision))
            indices.append(len(rows) - 1)
            masks.append(old_step.decision.legal_actions)
            actions.append(old_step.action)
            old_log_probabilities.append(old_step.old_log_prob)
        packed_rows.append(rows)
        decision_indices.append(indices)
        legal_masks.extend(masks)
        returns.append(
            _discounted_returns(len(trajectory), won=won).to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )
    sample_count = len(actions)
    if not sample_count:
        raise ValueError("drift evaluation requires at least one decision")

    max_length = max(len(rows) for rows in packed_rows)
    token_batch = _pack_native_token_batch(
        packed_rows,
        device=parameter.device,
        dtype=parameter.dtype,
    )

    with torch.no_grad():
        all_logits, all_values = policy.forward_native_sequence(token_batch)
        all_logits = all_logits.reshape(
            len(packed_rows), max_length, -1
        )
        all_values = all_values.reshape(len(packed_rows), max_length)
        logits, values = _gather_native_decisions(
            all_logits,
            all_values,
            decision_indices,
            max_length=max_length,
            device=parameter.device,
        )
        masks = torch.tensor(
            legal_masks, device=parameter.device, dtype=torch.bool
        )
        logits = logits.masked_fill(
            ~masks, torch.finfo(logits.dtype).min
        )
        distribution = Categorical(logits=logits)
        action_tensor = torch.tensor(
            actions, device=parameter.device, dtype=torch.long
        )
        old_log_probability_tensor = torch.tensor(
            old_log_probabilities,
            device=parameter.device,
            dtype=parameter.dtype,
        )
        logs = (
            distribution.log_prob(action_tensor)
            - old_log_probability_tensor
        )
        ratios = logs.exp()
        target = torch.cat(returns)
        variance = target.var(unbiased=False)
        explained = (
            torch.tensor(float("nan"), device=target.device)
            if variance == 0
            else 1
            - (target - values).var(unbiased=False) / variance
        )
        return {
            "approx_kl": float(
                ((ratios - 1.0) - logs).mean().detach()
            ),
            "old_approx_kl": float((-logs).mean().detach()),
            "clip_fraction": float(
                (
                    (ratios - 1.0).abs() > clip_ratio
                ).float().mean().detach()
            ),
            "entropy": float(distribution.entropy().mean().detach()),
            "value_mse": float(
                (values - target).square().mean().detach()
            ),
            "explained_variance": float(explained.detach()),
            "samples": sample_count,
        }


def collect_smoke_episodes(
    policy: PufferStyleTemporalNet,
    *,
    seed: int,
    episode_count: int = 2,
    max_turns: int = 160,
) -> tuple[list[tuple[Sequence[TrajectoryStep], bool]], dict[str, int]]:
    """Collect small deterministic-seed rollouts through the legacy game."""
    np.random.seed(seed)
    random.seed(seed)
    episodes: list[tuple[Sequence[TrajectoryStep], bool]] = []
    total_turns = 0
    wins = 0
    for episode_index in range(episode_count):
        agent_class = temporal_agent_factory(
            policy, deterministic=False, seed=seed + episode_index
        )
        match = game.Game([agent_class, AgentRandom, AgentRandom])
        turns, winner = match.auto_simulate(max_turns=max_turns)
        temporal_agent = match.players[0]
        won = winner == 0
        wins += int(won)
        total_turns += turns
        episodes.append((tuple(temporal_agent.trajectory), won))
    return episodes, {
        "episodes": episode_count,
        "turns": total_turns,
        "wins": wins,
        "decisions": sum(len(trajectory) for trajectory, _ in episodes),
    }


def run_smoke_benchmark(
    *,
    seed: int = 731,
    episode_count: int = 2,
    max_turns: int = 160,
) -> dict:
    """Run a deterministic rollout and one real recurrent PPO optimizer step."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    policy = PufferStyleTemporalNet(hidden_size=48)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    before = {
        name: parameter.detach().clone()
        for name, parameter in policy.named_parameters()
    }
    started = time.perf_counter()
    episodes, rollout = collect_smoke_episodes(
        policy,
        seed=seed,
        episode_count=episode_count,
        max_turns=max_turns,
    )
    metrics = ppo_update(policy, episodes, optimizer)
    elapsed = time.perf_counter() - started
    changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in policy.named_parameters()
    )
    return {
        "schema_version": 1,
        "seed": seed,
        "integration": {
            "native_pufferlib": policy.native_pufferlib,
            "status": policy.integration_status,
            "verified_pufferlib": "3.0.0",
            "installed_for_runtime": False,
            "pufferlib4_blocker": (
                "PyPI provides only pufferlib<=3.0.0"
            ),
            "pufferlib3_blocker": (
                "numpy<2 conflicts with ml-dtypes numpy>=2.1 on Python 3.13"
            ),
        },
        "model": {
            "name": type(policy).__name__,
            "recurrent_cell": "GRUCell",
            "hidden_size": policy.hidden_size,
            "actor_actions": 55,
            "critic_privileged": False,
        },
        "rollout": rollout,
        "training": {
            **metrics,
            "parameters_changed": changed,
        },
        "elapsed_seconds": round(elapsed, 6),
    }


def save_safe_checkpoint(
    path: str | Path,
    policy: PufferStyleTemporalNet,
) -> None:
    """Save model weights and non-sensitive schema metadata only."""
    torch.save(
        {
            "format": "pesten-temporal-safe-v1",
            "integration": policy.integration_status,
            "observation_schema": "PlayerPrivateObservation-v1",
            "state_dict": policy.state_dict(),
        },
        Path(path),
    )


def write_benchmark(path: str | Path, result: dict) -> None:
    Path(path).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    destination = Path("benchmarks/puffer_temporal_smoke_20260731.json")
    result = run_smoke_benchmark()
    write_benchmark(destination, result)
    print(json.dumps(result, indent=2, sort_keys=True))
