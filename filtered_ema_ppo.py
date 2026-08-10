#!/usr/bin/env python3
"""Audited top-advantage-filtered PPO continuation with an EMA actor.

This development-only experiment adapts two training ingredients from
"Superhuman AI for Generals.io Using Self-Play Reinforcement Learning":

* rank transitions by the critic's pre-update GAE estimate and apply the
  policy-gradient update to only the retained top fraction; and
* maintain a parameter exponential moving average for deployment evaluation.

The critic still learns from every collected transition.  Rollouts are made
from the current online actor, use only the existing 121 human-visible
features, and use explicit unique development deal seeds.  Both online and EMA
actors are saved as plain ``StudentNet`` state dictionaries for exact
suit-projected evaluation by the existing benchmark harness.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import torch
from torch.distributions import Categorical

from ai_observation import OBSERVATION_SIZE
from anchored_ppo import (
    CHAMPION_SHA256,
    enforce_initial_ratio_invariant,
    file_sha256,
)
from ppo2 import (
    DEFAULT_TORCH_THREADS,
    StudentCritic,
    StudentNet,
    build_action_masks,
    compute_gae,
)
from trpo_selfplay import (
    audit_terminal_payoffs,
    collect_explicit_seed_selfplay,
    module_sha256,
    recorded_deal_seeds,
    seed_everything,
    terminal_zero_sum_rewards,
    validate_development_seeds,
)
from seed_integrity import (
    DEVELOPMENT_SEED_LIMIT,
    RecordedSeedUse,
    scan_recorded_seed_interval,
)


PAPER_URL = "https://arxiv.org/abs/2606.23348"
DEFAULT_CHAMPION = Path("ppo_actor_100run3.pth")
DEFAULT_KEEP_FRACTIONS = (0.25, 0.5, 1.0)


def keep_fraction_slug(value: float) -> str:
    return format(float(value), "g").replace("-", "m").replace(".", "p")


def should_save_checkpoint(
    iteration: int,
    *,
    total_iterations: int,
    checkpoint_every: int,
) -> bool:
    """Save cadence milestones and always preserve the final iterate."""
    if min(iteration, total_iterations, checkpoint_every) < 1:
        raise ValueError("checkpoint iteration values must be positive")
    if iteration > total_iterations:
        raise ValueError("iteration cannot exceed total iterations")
    return iteration % checkpoint_every == 0 or iteration == total_iterations


def assert_no_recorded_seed_collisions(
    planned_seeds: Iterable[int],
    recorded_uses: Iterable[RecordedSeedUse],
) -> dict:
    """Fail closed when any planned seed already appears in an artifact."""
    planned = [int(seed) for seed in planned_seeds]
    if len(set(planned)) != len(planned):
        raise ValueError("planned seed namespaces overlap internally")
    recorded = list(recorded_uses)
    planned_set = set(planned)
    conflicts = [use for use in recorded if use.seed in planned_set]
    if conflicts:
        rendered = ", ".join(
            f"{use.seed}@{use.path}:{use.key_path}"
            for use in conflicts[:8]
        )
        raise RuntimeError(f"planned seeds already recorded: {rendered}")
    return {
        "recorded_seed_values_scanned": len(recorded),
        "planned_seed_count": len(planned),
        "recorded_conflict_count": 0,
        "recorded_conflicts": [],
    }


def top_advantage_indices(
    advantages: torch.Tensor,
    keep_fraction: float,
) -> torch.Tensor:
    """Return a deterministic, index-ordered top-advantage subset.

    Ties are broken by original transition index.  Returning the chosen
    indices in their original order makes minibatch construction independent
    of the details of the ranking implementation.
    """
    if advantages.ndim != 1 or len(advantages) == 0:
        raise ValueError("advantages must be a non-empty vector")
    if not 0 < keep_fraction <= 1 or not math.isfinite(keep_fraction):
        raise ValueError("keep fraction must be finite and in (0, 1]")
    if not bool(torch.isfinite(advantages).all()):
        raise ValueError("advantages must be finite")
    count = max(1, math.ceil(len(advantages) * keep_fraction))
    values = advantages.detach().cpu().numpy()
    original_indices = np.arange(len(values), dtype=np.int64)
    ranked = np.lexsort((original_indices, -values))
    chosen = np.sort(ranked[:count])
    return torch.as_tensor(chosen, dtype=torch.long, device=advantages.device)


def clipped_policy_surrogate(
    new_log_probabilities: torch.Tensor,
    behavior_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the standard PPO clipped surrogate and importance ratios."""
    if not (
        new_log_probabilities.shape
        == behavior_log_probabilities.shape
        == advantages.shape
    ):
        raise ValueError("PPO surrogate inputs must have equal shapes")
    if clip <= 0 or not math.isfinite(clip):
        raise ValueError("clip must be finite and positive")
    ratios = torch.exp(new_log_probabilities - behavior_log_probabilities)
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip, 1.0 + clip) * advantages
    return torch.minimum(unclipped, clipped).mean(), ratios


def initialize_ema(online: torch.nn.Module) -> torch.nn.Module:
    """Create a frozen, byte-identical EMA copy of an online actor."""
    ema = copy.deepcopy(online).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    if module_sha256(online) != module_sha256(ema):
        raise RuntimeError("EMA did not initialize from the exact online actor")
    return ema


def update_ema(
    ema: torch.nn.Module,
    online: torch.nn.Module,
    *,
    decay: float,
) -> None:
    """Apply ``ema = decay*ema + (1-decay)*online`` to parameters.

    Non-parameter buffers are copied from the online actor.  The current
    ``StudentNet`` has no training-varying buffers, but explicit copying keeps
    the checkpoint semantics defined if its architecture later gains any.
    """
    if not 0 <= decay < 1 or not math.isfinite(decay):
        raise ValueError("EMA decay must be finite and in [0, 1)")
    ema_parameters = dict(ema.named_parameters())
    online_parameters = dict(online.named_parameters())
    if ema_parameters.keys() != online_parameters.keys():
        raise ValueError("EMA and online parameter structures differ")
    with torch.no_grad():
        for name, ema_parameter in ema_parameters.items():
            ema_parameter.mul_(decay).add_(
                online_parameters[name],
                alpha=1.0 - decay,
            )
        ema_buffers = dict(ema.named_buffers())
        online_buffers = dict(online.named_buffers())
        if ema_buffers.keys() != online_buffers.keys():
            raise ValueError("EMA and online buffer structures differ")
        for name, ema_buffer in ema_buffers.items():
            ema_buffer.copy_(online_buffers[name])
    ema.eval()


def chunked_values(
    critic: torch.nn.Module,
    observations: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    values = []
    critic.eval()
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            values.append(
                critic(observations[start : start + batch_size]).squeeze(-1)
            )
    return torch.cat(values)


def monte_carlo_returns(
    rewards: torch.Tensor,
    episode_lengths: list[int],
    *,
    gamma: float,
) -> torch.Tensor:
    zeros = torch.zeros_like(rewards)
    _, returns = compute_gae(
        rewards,
        zeros,
        episode_lengths=episode_lengths,
        gamma_=gamma,
        lam=1.0,
    )
    return returns


def critic_update(
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    targets: torch.Tensor,
    *,
    batch_size: int,
    epochs: int,
    gradient_clip: float,
    generator: torch.Generator,
) -> dict:
    """Fit the critic on all transitions, independent of actor filtering."""
    if len(observations) != len(targets):
        raise ValueError("critic observations and targets must align")
    before = chunked_values(critic, observations, batch_size=batch_size)
    losses = []
    critic.train()
    for _ in range(epochs):
        order = torch.randperm(
            len(targets),
            generator=generator,
            device=observations.device,
        )
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            prediction = critic(observations[indices]).squeeze(-1)
            loss = torch.nn.functional.smooth_l1_loss(
                prediction,
                targets[indices],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                critic.parameters(),
                gradient_clip,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite critic gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
    after = chunked_values(critic, observations, batch_size=batch_size)
    return {
        "training_transitions": len(targets),
        "epochs": epochs,
        "mse_before": float(torch.nn.functional.mse_loss(before, targets)),
        "mse_after": float(torch.nn.functional.mse_loss(after, targets)),
        "mean_smooth_l1_loss": float(np.mean(losses)),
    }


def filtered_actor_update(
    actor: torch.nn.Module,
    ema_actor: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    observations: torch.Tensor,
    actions: torch.Tensor,
    behavior_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    legal_actions: torch.Tensor,
    *,
    keep_fraction: float,
    temperature: float,
    clip: float,
    entropy_coefficient: float,
    batch_size: int,
    epochs: int,
    gradient_clip: float,
    ema_decay: float,
    generator: torch.Generator,
) -> dict:
    """Update the actor only from the retained top-advantage transitions."""
    if observations.shape[-1] != OBSERVATION_SIZE:
        raise ValueError("actor may only consume public observations")
    if not (
        len(observations)
        == len(actions)
        == len(behavior_log_probabilities)
        == len(advantages)
        == len(legal_actions)
    ):
        raise ValueError("actor rollout tensors must have equal length")
    if legal_actions.dtype != torch.bool:
        raise ValueError("legal action mask must be boolean")
    if not bool(legal_actions.gather(1, actions[:, None]).all()):
        raise RuntimeError("rollout contains an illegal chosen action")
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")

    retained = top_advantage_indices(advantages, keep_fraction)
    normalized_advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)

    actor.eval()
    with torch.inference_mode():
        unchanged_logits = actor(observations) / temperature
        floor = torch.finfo(unchanged_logits.dtype).min
        unchanged_distribution = Categorical(
            logits=unchanged_logits.masked_fill(~legal_actions, floor),
            validate_args=False,
        )
        unchanged_log_probabilities = unchanged_distribution.log_prob(actions)
        initial_ratios = torch.exp(
            unchanged_log_probabilities - behavior_log_probabilities
        )
        maximum_initial_ratio_error = enforce_initial_ratio_invariant(
            initial_ratios
        )

    epoch_logs = []
    ema_updates = 0
    # DenseSkipNet contains BatchNorm layers.  Rollouts are collected in eval
    # mode, so gradients must also be computed in eval mode to preserve the
    # behavior-policy identity and avoid mutating running-stat buffers.
    actor.eval()
    for epoch in range(epochs):
        permutation = torch.randperm(
            len(retained),
            generator=generator,
            device=retained.device,
        )
        order = retained[permutation]
        losses = []
        entropies = []
        clip_fractions = []
        ratio_errors = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            logits = actor(observations[indices]) / temperature
            masks = legal_actions[indices]
            floor = torch.finfo(logits.dtype).min
            distribution = Categorical(
                logits=logits.masked_fill(~masks, floor),
                validate_args=False,
            )
            new_log_probabilities = distribution.log_prob(actions[indices])
            surrogate, ratios = clipped_policy_surrogate(
                new_log_probabilities,
                behavior_log_probabilities[indices],
                normalized_advantages[indices],
                clip=clip,
            )
            entropy = distribution.entropy().mean()
            loss = -surrogate - entropy_coefficient * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                gradient_clip,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("non-finite actor gradient")
            optimizer.step()
            update_ema(ema_actor, actor, decay=ema_decay)
            ema_updates += 1
            losses.append(float(loss.detach()))
            entropies.append(float(entropy.detach()))
            clip_fractions.append(
                float(((ratios - 1.0).abs() > clip).float().mean())
            )
            ratio_errors.append(
                float(ratios.detach().sub(1).abs().max())
            )
        epoch_logs.append(
            {
                "epoch": epoch + 1,
                "minibatches": math.ceil(len(order) / batch_size),
                "mean_loss": float(np.mean(losses)),
                "mean_entropy": float(np.mean(entropies)),
                "mean_clip_fraction": float(np.mean(clip_fractions)),
                "maximum_batch_ratio_error": float(max(ratio_errors)),
            }
        )
    actor.eval()
    ema_actor.eval()
    raw_advantages = advantages[retained]
    return {
        "rollout_transitions": len(advantages),
        "retained_transitions": len(retained),
        "realized_keep_fraction": len(retained) / len(advantages),
        "requested_keep_fraction": float(keep_fraction),
        "retained_minimum_raw_advantage": float(raw_advantages.min()),
        "retained_mean_raw_advantage": float(raw_advantages.mean()),
        "rollout_mean_raw_advantage": float(advantages.mean()),
        "maximum_initial_ratio_error": maximum_initial_ratio_error,
        "ema_updates": ema_updates,
        "epochs": epoch_logs,
    }


def _warm_critic(
    actor: torch.nn.Module,
    critic: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    deal_seeds: list[int],
    policy_seed: int,
    temperature: float,
    max_turns: int,
    gamma: float,
    batch_size: int,
    epochs: int,
    gradient_clip: float,
    device: torch.device,
    generator: torch.Generator,
) -> dict:
    actor_before = module_sha256(actor)
    rollout_actor = copy.deepcopy(actor).to("cpu").eval()
    for parameter in rollout_actor.parameters():
        parameter.requires_grad_(False)
    rollout, engine = collect_explicit_seed_selfplay(
        rollout_actor,
        deal_seeds=deal_seeds,
        policy_sampling_seed=policy_seed,
        temperature=temperature,
        max_turns=max_turns,
    )
    rewards = terminal_zero_sum_rewards(rollout.actions, rollout.rewards)
    payoff_audit = audit_terminal_payoffs(
        rewards,
        rollout.episode_lengths,
        expected_games=rollout.completed_games,
    )
    targets = monte_carlo_returns(
        rewards.to(device),
        rollout.episode_lengths,
        gamma=gamma,
    )
    optimization = critic_update(
        critic,
        optimizer,
        rollout.observations.to(device),
        targets,
        batch_size=batch_size,
        epochs=epochs,
        gradient_clip=gradient_clip,
        generator=generator,
    )
    if module_sha256(actor) != actor_before:
        raise RuntimeError("critic warm-up changed the frozen actor")
    return {
        "deal_seeds": recorded_deal_seeds(deal_seeds),
        "policy_sampling_seed": policy_seed,
        "rollout": {
            "completed_games": rollout.completed_games,
            "environment_turns": rollout.environment_turns,
            "policy_decisions": rollout.policy_decisions,
            "inference_calls": rollout.inference_calls,
            "draws": rollout.draws,
            "maximum_turns_observed": int(engine.turn_counts.max()),
        },
        "payoff_audit": payoff_audit,
        "optimization": optimization,
        "actor_unchanged": True,
    }


def run_pilot(args: argparse.Namespace) -> dict:
    champion = args.champion.resolve()
    if file_sha256(champion) != CHAMPION_SHA256:
        raise RuntimeError("refusing to train from an unverified champion")
    if not args.keep_fractions or any(
        not 0 < value <= 1 or not math.isfinite(value)
        for value in args.keep_fractions
    ):
        raise ValueError("keep fractions must be finite and in (0, 1]")
    if len(set(args.keep_fractions)) != len(args.keep_fractions):
        raise ValueError("keep fractions must be unique")
    positive_sizes = (
        args.iterations,
        args.games,
        args.critic_warmup_games,
        args.critic_warmup_epochs,
        args.actor_epochs,
        args.critic_epochs,
        args.batch_size,
        args.checkpoint_every,
        args.max_turns,
        args.torch_threads,
    )
    if min(positive_sizes) < 1:
        raise ValueError("pilot sizes must be positive")
    if not 0 <= args.entropy_coefficient < 1:
        raise ValueError("entropy coefficient must be in [0, 1)")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("EMA decay must be in [0, 1)")

    last_warmup_deal_seed = (
        args.warmup_deal_seed_start + args.critic_warmup_games - 1
    )
    last_training_deal_seed = (
        args.deal_seed_start + args.iterations * args.games - 1
    )
    last_policy_seed = args.policy_seed_start + args.iterations - 1
    validate_development_seeds(
        [
            args.training_seed,
            args.warmup_deal_seed_start,
            last_warmup_deal_seed,
            args.warmup_policy_seed,
            args.deal_seed_start,
            last_training_deal_seed,
            args.policy_seed_start,
            last_policy_seed,
        ]
    )
    warmup_seed_set = set(
        range(
            args.warmup_deal_seed_start,
            args.warmup_deal_seed_start + args.critic_warmup_games,
        )
    )
    training_seed_set = set(
        range(
            args.deal_seed_start,
            args.deal_seed_start + args.iterations * args.games,
        )
    )
    if warmup_seed_set & training_seed_set:
        raise ValueError("warm-up and training deal seeds must be disjoint")
    planned_seeds = [
        args.training_seed,
        *sorted(warmup_seed_set),
        args.warmup_policy_seed,
        *sorted(training_seed_set),
        *range(
            args.policy_seed_start,
            args.policy_seed_start + args.iterations,
        ),
    ]
    audit_roots = tuple(
        path.resolve()
        for path in (
            args.recorded_seed_root
            or (Path("benchmarks"), Path("experiments"))
        )
    )
    recorded_uses = scan_recorded_seed_interval(
        audit_roots,
        start=0,
        end=DEVELOPMENT_SEED_LIMIT,
        allow_sealed_results=True,
    )
    recorded_seed_audit = assert_no_recorded_seed_collisions(
        planned_seeds,
        recorded_uses,
    )

    torch.set_num_threads(args.torch_threads)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    champion_state = torch.load(champion, map_location="cpu", weights_only=True)

    seed_everything(args.training_seed)
    base_actor = StudentNet().to(device)
    base_actor.load_state_dict(champion_state)
    base_actor.eval()
    base_critic = StudentCritic().to(device)
    base_critic_optimizer = torch.optim.AdamW(
        base_critic.parameters(),
        lr=args.critic_learning_rate,
    )
    warmup_generator = torch.Generator(device=device).manual_seed(
        args.training_seed + 1
    )
    warmup_started = time.perf_counter()
    critic_warmup = _warm_critic(
        base_actor,
        base_critic,
        base_critic_optimizer,
        deal_seeds=list(
            range(
                args.warmup_deal_seed_start,
                args.warmup_deal_seed_start + args.critic_warmup_games,
            )
        ),
        policy_seed=args.warmup_policy_seed,
        temperature=args.temperature,
        max_turns=args.max_turns,
        gamma=args.gamma,
        batch_size=args.batch_size,
        epochs=args.critic_warmup_epochs,
        gradient_clip=args.critic_gradient_clip,
        device=device,
        generator=warmup_generator,
    )
    critic_warmup["wall_seconds"] = time.perf_counter() - warmup_started
    warmed_critic_state = copy.deepcopy(base_critic.state_dict())

    sweeps = []
    for keep_fraction in args.keep_fractions:
        seed_everything(args.training_seed)
        actor = StudentNet().to(device)
        actor.load_state_dict(champion_state)
        actor.eval()
        ema_actor = initialize_ema(actor).to(device)
        critic = StudentCritic().to(device)
        critic.load_state_dict(warmed_critic_state)
        actor_optimizer = torch.optim.AdamW(
            actor.parameters(),
            lr=args.actor_learning_rate,
        )
        critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=args.critic_learning_rate,
        )
        rollout_actor = copy.deepcopy(actor).to("cpu").eval()
        for parameter in rollout_actor.parameters():
            parameter.requires_grad_(False)
        sweep = {
            "keep_fraction": float(keep_fraction),
            "initial_online_actor_state_sha256": module_sha256(actor),
            "initial_ema_actor_state_sha256": module_sha256(ema_actor),
            "iterations": [],
        }
        for iteration in range(args.iterations):
            started = time.perf_counter()
            deal_seed_start = args.deal_seed_start + iteration * args.games
            deal_seeds = list(
                range(deal_seed_start, deal_seed_start + args.games)
            )
            policy_seed = args.policy_seed_start + iteration
            rollout_actor.load_state_dict(
                {
                    name: value.detach().cpu()
                    for name, value in actor.state_dict().items()
                }
            )
            rollout, engine = collect_explicit_seed_selfplay(
                rollout_actor,
                deal_seeds=deal_seeds,
                policy_sampling_seed=policy_seed,
                temperature=args.temperature,
                max_turns=args.max_turns,
            )
            rewards = terminal_zero_sum_rewards(
                rollout.actions,
                rollout.rewards,
            )
            payoff_audit = audit_terminal_payoffs(
                rewards,
                rollout.episode_lengths,
                expected_games=rollout.completed_games,
            )
            observations = rollout.observations.to(device)
            actions = rollout.actions.to(device)
            behavior_log_probabilities = rollout.log_probabilities.to(device)
            legal_actions = build_action_masks(observations)
            old_values = chunked_values(
                critic,
                observations,
                batch_size=args.batch_size,
            )
            with torch.inference_mode():
                advantages, returns = compute_gae(
                    rewards.to(device),
                    old_values,
                    episode_lengths=rollout.episode_lengths,
                    gamma_=args.gamma,
                    lam=args.gae_lambda,
                )
            actor_before = module_sha256(actor)
            actor_generator = torch.Generator(device=device).manual_seed(
                args.training_seed + 10_000 + iteration
            )
            actor_log = filtered_actor_update(
                actor,
                ema_actor,
                actor_optimizer,
                observations,
                actions,
                behavior_log_probabilities,
                advantages,
                legal_actions,
                keep_fraction=keep_fraction,
                temperature=args.temperature,
                clip=args.clip,
                entropy_coefficient=args.entropy_coefficient,
                batch_size=args.batch_size,
                epochs=args.actor_epochs,
                gradient_clip=args.actor_gradient_clip,
                ema_decay=args.ema_decay,
                generator=actor_generator,
            )
            critic_generator = torch.Generator(device=device).manual_seed(
                args.training_seed + 20_000 + iteration
            )
            critic_log = critic_update(
                critic,
                critic_optimizer,
                observations,
                returns.detach(),
                batch_size=args.batch_size,
                epochs=args.critic_epochs,
                gradient_clip=args.critic_gradient_clip,
                generator=critic_generator,
            )
            actor_after = module_sha256(actor)
            ema_after = module_sha256(ema_actor)
            if actor_after == actor_before:
                raise RuntimeError("filtered PPO update did not change actor")
            if ema_after == actor_before:
                raise RuntimeError("EMA update did not change actor")

            iteration_number = iteration + 1
            save_checkpoint = should_save_checkpoint(
                iteration_number,
                total_iterations=args.iterations,
                checkpoint_every=args.checkpoint_every,
            )
            iteration_record = {
                "iteration": iteration_number,
                "deal_seeds": recorded_deal_seeds(deal_seeds),
                "policy_sampling_seed": policy_seed,
                "rollout": {
                    "completed_games": rollout.completed_games,
                    "environment_turns": rollout.environment_turns,
                    "policy_decisions": rollout.policy_decisions,
                    "inference_calls": rollout.inference_calls,
                    "draws": rollout.draws,
                    "maximum_turns_observed": int(
                        engine.turn_counts.max()
                    ),
                },
                "payoff_audit": payoff_audit,
                "actor_optimization": actor_log,
                "critic_optimization": critic_log,
                "checkpoint_saved": save_checkpoint,
            }
            if save_checkpoint:
                slug = keep_fraction_slug(keep_fraction)
                online_name = (
                    f"filteredppo_keep_{slug}_iter_"
                    f"{iteration_number:02d}_online.pth"
                )
                ema_name = (
                    f"filteredppo_keep_{slug}_iter_"
                    f"{iteration_number:02d}_ema.pth"
                )
                online_path = output_dir / online_name
                ema_path = output_dir / ema_name
                torch.save(actor.state_dict(), online_path)
                torch.save(ema_actor.state_dict(), ema_path)
                iteration_record["checkpoints"] = {
                    "online": {
                        "path": online_name,
                        "sha256": file_sha256(online_path),
                        "actor_state_sha256": actor_after,
                    },
                    "ema": {
                        "path": ema_name,
                        "sha256": file_sha256(ema_path),
                        "actor_state_sha256": ema_after,
                    },
                }
            iteration_record["wall_seconds"] = time.perf_counter() - started
            sweep["iterations"].append(iteration_record)
        sweeps.append(sweep)

    payload = {
        "schema_version": 1,
        "method": "top_advantage_filtered_ema_ppo_shared_selfplay",
        "status": "complete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "name": (
                "Superhuman AI for Generals.io Using Self-Play "
                "Reinforcement Learning"
            ),
            "url": PAPER_URL,
            "transferred_ingredients": [
                "top-advantage actor-only filtering",
                "parameter exponential moving average",
            ],
        },
        "champion": {
            "path": str(champion),
            "sha256": CHAMPION_SHA256,
        },
        "privacy": {
            "observation_features": OBSERVATION_SIZE,
            "own_hand_plus_public_state_only": True,
            "opponent_card_identities": False,
            "draw_pile_order": False,
            "shared_symmetric_two_player_policy": True,
        },
        "seed_audit": {
            "training_seed": args.training_seed,
            "deal_seed_schedule": (
                "explicit unique contiguous seeds; common across fractions"
            ),
            "warmup_deal_seed_start": args.warmup_deal_seed_start,
            "warmup_policy_seed": args.warmup_policy_seed,
            "deal_seed_start": args.deal_seed_start,
            "policy_seed_start": args.policy_seed_start,
            "development_seed_limit": DEVELOPMENT_SEED_LIMIT,
            "development_only": True,
            "sealed_outcomes_used": False,
            "recorded_seed_roots": [str(path) for path in audit_roots],
            **recorded_seed_audit,
        },
        "recipe": {
            "keep_fractions": [
                float(value) for value in args.keep_fractions
            ],
            "iterations": args.iterations,
            "checkpoint_every": args.checkpoint_every,
            "checkpoint_policy": (
                "every checkpoint_every iterations and always final"
            ),
            "games_per_iteration": args.games,
            "critic_warmup_games": args.critic_warmup_games,
            "critic_warmup_epochs": args.critic_warmup_epochs,
            "actor_epochs": args.actor_epochs,
            "critic_epochs": args.critic_epochs,
            "batch_size": args.batch_size,
            "temperature": args.temperature,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip": args.clip,
            "entropy_coefficient": args.entropy_coefficient,
            "actor_learning_rate": args.actor_learning_rate,
            "critic_learning_rate": args.critic_learning_rate,
            "actor_gradient_clip": args.actor_gradient_clip,
            "critic_gradient_clip": args.critic_gradient_clip,
            "ema_decay": args.ema_decay,
            "maximum_turns": args.max_turns,
            "torch_threads": args.torch_threads,
            "device": str(device),
        },
        "critic_warmup": critic_warmup,
        "sweeps": sweeps,
        "promotion": {
            "eligible": False,
            "reason": "requires fresh direct and pool evaluation gates",
        },
    }
    manifest_path = output_dir / "filtered_ema_ppo_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--keep-fractions",
        type=float,
        nargs="+",
        default=list(DEFAULT_KEEP_FRACTIONS),
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="save every N iterations; the final iterate is always saved",
    )
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--critic-warmup-games", type=int, default=32)
    parser.add_argument("--critic-warmup-epochs", type=int, default=1)
    parser.add_argument("--actor-epochs", type=int, default=1)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--training-seed", type=int, default=89_110_001)
    parser.add_argument(
        "--warmup-deal-seed-start",
        type=int,
        default=89_111_000,
    )
    parser.add_argument("--warmup-policy-seed", type=int, default=89_112_000)
    parser.add_argument("--deal-seed-start", type=int, default=89_113_000)
    parser.add_argument("--policy-seed-start", type=int, default=89_114_000)
    parser.add_argument(
        "--recorded-seed-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "JSON/JSONL artifact root scanned before allocation; repeatable "
            "(default: benchmarks and experiments)"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.9)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.0)
    parser.add_argument("--actor-learning-rate", type=float, default=1e-5)
    parser.add_argument("--critic-learning-rate", type=float, default=3e-5)
    parser.add_argument("--actor-gradient-clip", type=float, default=0.267)
    parser.add_argument("--critic-gradient-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=DEFAULT_TORCH_THREADS,
    )
    parser.add_argument("--cpu-only", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    payload = run_pilot(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "method": payload["method"],
                "sweeps": len(payload["sweeps"]),
                "sealed_outcomes_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
