#!/usr/bin/env python3
"""Information-safe Q-boosting PPO experiment.

This is an isolated adaptation of the Q-boosting estimator from Fan and
Farina (2026), arXiv:2605.19235.  The paper's full VRPO algorithm uses a
centralized critic with privileged game state.  This project deliberately
does not expose that state: both actor and Q-critic receive only the existing
121-feature own-hand/public observation.  Consequently this module is a
Q-boosting PPO variant, not a claim of exact centralized-critic VRPO.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ai_observation import ACTION_COUNT, OBSERVATION_SIZE
from anchored_ppo import (
    CHAMPION_SHA256,
    enforce_initial_ratio_invariant,
    file_sha256,
    full_policy_kl,
    parse_agent_sets,
    potential_shaped_rewards,
    working_directory,
)
from benchmark_model_zoo import DEFAULT_TEACHER
from ppo2 import (
    DEFAULT_TORCH_THREADS,
    PPO_Supervised,
    StudentCritic,
    StudentNet,
)


PAPER = "Fan and Farina, arXiv:2605.19235"
SEALED_SEED_START = 92_000_000


class PublicObservationQCritic(StudentCritic):
    """Action-value critic restricted to the policy's human-visible input."""

    def __init__(self) -> None:
        super().__init__(
            input_dim=OBSERVATION_SIZE,
            output_dim=ACTION_COUNT,
        )


@dataclass(frozen=True, slots=True)
class QBoostEstimate:
    expected_values: torch.Tensor
    chosen_action_values: torch.Tensor
    expected_sarsa_residuals: torch.Tensor
    advantages: torch.Tensor
    q_targets: torch.Tensor


def validate_public_observations(observations: torch.Tensor) -> None:
    if observations.ndim != 2:
        raise ValueError("observations must be a two-dimensional batch")
    if observations.shape[1] != OBSERVATION_SIZE:
        raise ValueError(
            "Q-critic accepts only the existing "
            f"{OBSERVATION_SIZE}-feature human-visible observation"
        )


def expected_action_values(
    q_values: torch.Tensor,
    policy_logits: torch.Tensor,
    legal_actions: torch.Tensor,
) -> torch.Tensor:
    """Compute the policy expectation of Q over legal actions only."""
    if q_values.ndim != 2 or policy_logits.shape != q_values.shape:
        raise ValueError("Q values and policy logits must share [N, A] shape")
    if legal_actions.shape != q_values.shape:
        raise ValueError("legal-action mask must match Q values")
    if legal_actions.dtype is not torch.bool:
        raise ValueError("legal-action mask must be boolean")
    if torch.any(~legal_actions.any(dim=-1)):
        raise ValueError("every observation must have a legal action")
    floor = torch.finfo(policy_logits.dtype).min
    probabilities = torch.softmax(
        policy_logits.masked_fill(~legal_actions, floor),
        dim=-1,
    )
    return (probabilities * q_values).sum(dim=-1)


def qboost_estimate(
    *,
    rewards: torch.Tensor,
    q_values: torch.Tensor,
    actions: torch.Tensor,
    policy_logits: torch.Tensor,
    legal_actions: torch.Tensor,
    episode_lengths: Sequence[int],
    gamma: float = 0.97,
    trace_lambda: float = 0.95,
) -> QBoostEstimate:
    """Compute equations 3.2 and 3.3 using Expected-SARSA(lambda).

    For each sampled decision t:

      V_t = sum_a pi(a | o_t) Q(o_t, a)
      delta+_t = r_t + gamma V_{t+1} - Q(o_t, a_t)
      trace_t = delta+_t + gamma lambda trace_{t+1}
      Aboost_t = Q(o_t, a_t) - V_t + trace_t
      Qtarget_t = Q(o_t, a_t) + trace_t

    Terminal next-state values are exactly zero and traces never cross an
    episode boundary.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 <= trace_lambda <= 1.0:
        raise ValueError("trace_lambda must be in [0, 1]")
    if rewards.ndim != 1 or actions.ndim != 1:
        raise ValueError("rewards and actions must be one-dimensional")
    if len(rewards) != len(actions) or len(q_values) != len(rewards):
        raise ValueError("one reward, action, and Q row is required per step")
    lengths = tuple(int(length) for length in episode_lengths)
    if any(length < 0 for length in lengths):
        raise ValueError("episode lengths cannot be negative")
    if sum(lengths) != len(rewards):
        raise ValueError("episode lengths must cover every step exactly")
    if actions.device != q_values.device or rewards.device != q_values.device:
        raise ValueError("Q-boosting tensors must share one device")
    if actions.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("actions must contain integer indices")
    if torch.any(actions < 0) or torch.any(actions >= q_values.shape[1]):
        raise ValueError("action index is outside the Q-value table")
    action_indices = actions.to(dtype=torch.long)
    if torch.any(
        ~legal_actions.gather(1, action_indices.unsqueeze(1)).squeeze(1)
    ):
        raise ValueError("sampled action is not legal")

    expected_values = expected_action_values(
        q_values, policy_logits, legal_actions
    )
    chosen_values = q_values.gather(
        1, action_indices.unsqueeze(1)
    ).squeeze(1)
    next_values = torch.zeros_like(expected_values)
    offset = 0
    for length in lengths:
        stop = offset + length
        if length > 1:
            next_values[offset : stop - 1] = expected_values[
                offset + 1 : stop
            ]
        offset = stop
    residuals = rewards + gamma * next_values - chosen_values
    traces = torch.zeros_like(residuals)
    offset = 0
    for length in lengths:
        stop = offset + length
        following_trace = residuals.new_zeros(())
        for index in range(stop - 1, offset - 1, -1):
            following_trace = (
                residuals[index]
                + gamma * trace_lambda * following_trace
            )
            traces[index] = following_trace
        offset = stop
    return QBoostEstimate(
        expected_values=expected_values,
        chosen_action_values=chosen_values,
        expected_sarsa_residuals=residuals,
        advantages=chosen_values - expected_values + traces,
        q_targets=chosen_values + traces,
    )


def module_sha256(module: nn.Module) -> str:
    """Hash model state deterministically without changing its device."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(
            tensor.detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def fit_q_critic(
    ppo: PPO_Supervised,
    q_critic: nn.Module,
    q_optimizer: torch.optim.Optimizer,
    *,
    iterations: int,
    episodes: int,
    critic_epochs: int,
    batch_size: int,
    temperature: float,
    agent_sets: list[list[str]],
    gamma: float = 0.97,
    trace_lambda: float = 0.95,
    potential_shaping_coefficient: float = 0.0,
    seed: int = 0,
) -> list[dict]:
    """Warm the Q-critic on-policy while keeping the actor byte-identical."""
    if iterations < 0:
        raise ValueError("Q warm-up iterations cannot be negative")
    if iterations == 0:
        return []
    if min(episodes, critic_epochs, batch_size) < 1:
        raise ValueError("warm-up counts and batch size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    actor_before = module_sha256(ppo.actor)
    actor_training = ppo.actor.training
    actor_requires_grad = [
        parameter.requires_grad for parameter in ppo.actor.parameters()
    ]
    for parameter in ppo.actor.parameters():
        parameter.requires_grad_(False)
    ppo.actor.eval()
    q_critic.to(ppo.actor_device)
    generator = torch.Generator(device=ppo.actor_device).manual_seed(seed)
    history = []
    try:
        for iteration in range(iterations):
            started = time.perf_counter()
            observations, actions, _, rewards, lengths = ppo.generate_data(
                num_episodes=episodes,
                temperature=temperature,
                agent_sets=agent_sets,
            )
            validate_public_observations(observations)
            masks = ppo.get_action_masks(observations)
            training_rewards, shaping_metrics = potential_shaped_rewards(
                rewards,
                observations,
                lengths,
                gamma=gamma,
                coefficient=potential_shaping_coefficient,
            )
            q_critic.eval()
            with torch.inference_mode():
                behavior_logits = ppo.actor(observations) / temperature
                frozen_q_values = q_critic(observations)
                estimate = qboost_estimate(
                    rewards=training_rewards,
                    q_values=frozen_q_values,
                    actions=actions,
                    policy_logits=behavior_logits,
                    legal_actions=masks,
                    episode_lengths=lengths,
                    gamma=gamma,
                    trace_lambda=trace_lambda,
                )
                q_targets = estimate.q_targets.detach()

            q_critic.train()
            epoch_logs = []
            for epoch in range(critic_epochs):
                order = torch.randperm(
                    len(actions),
                    generator=generator,
                    device=ppo.actor_device,
                )
                losses = []
                for start in range(0, len(order), batch_size):
                    indices = order[start : start + batch_size]
                    predictions = q_critic(
                        observations[indices]
                    ).gather(
                        1, actions[indices].unsqueeze(1)
                    ).squeeze(1)
                    loss = torch.nn.functional.mse_loss(
                        predictions, q_targets[indices]
                    )
                    q_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        q_critic.parameters(), 1.0
                    )
                    q_optimizer.step()
                    losses.append(float(loss.detach()))
                epoch_logs.append(
                    {
                        "epoch": epoch + 1,
                        "q_loss": float(np.mean(losses)),
                    }
                )
            q_critic.eval()
            current_actor_hash = module_sha256(ppo.actor)
            if current_actor_hash != actor_before:
                raise RuntimeError(
                    "Q-critic warm-up changed frozen actor parameters"
                )
            history.append(
                {
                    "iteration": iteration + 1,
                    "games": episodes,
                    "decisions": len(actions),
                    "rollout": dict(ppo.last_rollout_metrics),
                    "trace_lambda": trace_lambda,
                    "potential_shaping": shaping_metrics,
                    "target_mean": float(q_targets.mean()),
                    "target_std": float(
                        q_targets.std(unbiased=False)
                    ),
                    "expected_sarsa_residual_rmse": float(
                        estimate.expected_sarsa_residuals.square()
                        .mean()
                        .sqrt()
                    ),
                    "critic_epochs": epoch_logs,
                    "actor_sha256_before": actor_before,
                    "actor_sha256_after": current_actor_hash,
                    "actor_unchanged": True,
                    "wall_seconds": time.perf_counter() - started,
                }
            )
    finally:
        for parameter, requires_grad in zip(
            ppo.actor.parameters(), actor_requires_grad
        ):
            parameter.requires_grad_(requires_grad)
        ppo.actor.train(actor_training)
    actor_after = module_sha256(ppo.actor)
    if actor_after != actor_before:
        raise RuntimeError(
            "Q-critic warm-up changed frozen actor parameters"
        )
    return history


def qboosted_anchored_iteration(
    ppo: PPO_Supervised,
    reference: StudentNet,
    q_critic: nn.Module,
    q_optimizer: torch.optim.Optimizer,
    *,
    episodes: int,
    actor_epochs: int,
    critic_epochs: int,
    batch_size: int,
    temperature: float,
    agent_sets: list[list[str]],
    kl_coefficient: float,
    entropy_coefficient: float,
    target_kl: float,
    gamma: float = 0.97,
    trace_lambda: float = 0.95,
    potential_shaping_coefficient: float = 0.0,
    seed: int = 0,
) -> dict:
    """Run one on-policy Q-boosted PPO update and Q-critic regression."""
    if min(episodes, actor_epochs, critic_epochs, batch_size) < 1:
        raise ValueError("iteration counts and batch size must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    started = time.perf_counter()
    observations, actions, old_log_probs, rewards, lengths = ppo.generate_data(
        num_episodes=episodes,
        temperature=temperature,
        agent_sets=agent_sets,
    )
    validate_public_observations(observations)
    masks = ppo.get_action_masks(observations)
    training_rewards, shaping_metrics = potential_shaped_rewards(
        rewards,
        observations,
        lengths,
        gamma=gamma,
        coefficient=potential_shaping_coefficient,
    )
    q_critic.to(ppo.actor_device)
    q_critic.eval()
    with torch.inference_mode():
        behavior_logits = ppo.actor(observations) / temperature
        frozen_q_values = q_critic(observations)
        behavior_estimate = qboost_estimate(
            rewards=training_rewards,
            q_values=frozen_q_values,
            actions=actions,
            policy_logits=behavior_logits,
            legal_actions=masks,
            episode_lengths=lengths,
            gamma=gamma,
            trace_lambda=trace_lambda,
        )
        fixed_q_targets = behavior_estimate.q_targets.detach()

    generator = torch.Generator(device=ppo.actor_device).manual_seed(seed)
    actor_logs = []
    max_initial_ratio_error = None
    fixed_q_for_actor = frozen_q_values.detach()
    for epoch in range(actor_epochs):
        with torch.inference_mode():
            current_logits = ppo.actor(observations) / temperature
            current_estimate = qboost_estimate(
                rewards=training_rewards,
                q_values=fixed_q_for_actor,
                actions=actions,
                policy_logits=current_logits,
                legal_actions=masks,
                episode_lengths=lengths,
                gamma=gamma,
                trace_lambda=trace_lambda,
            )
            advantages = current_estimate.advantages
            advantages = (
                advantages - advantages.mean()
            ) / (advantages.std(unbiased=False) + 1e-8)
        order = torch.randperm(
            len(actions), generator=generator, device=ppo.actor_device
        )
        losses = []
        anchor_kls = []
        entropies = []
        clip_fractions = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            logits = ppo.actor(observations[indices]) / temperature
            floor = torch.finfo(logits.dtype).min
            distribution = Categorical(
                logits=logits.masked_fill(~masks[indices], floor),
                validate_args=False,
            )
            new_log_probs = distribution.log_prob(actions[indices])
            ratios = (new_log_probs - old_log_probs[indices]).exp()
            if epoch == 0 and start == 0:
                max_initial_ratio_error = enforce_initial_ratio_invariant(
                    ratios
                )
            unclipped = ratios * advantages[indices]
            clipped = torch.clamp(
                ratios, 1.0 - ppo.clip, 1.0 + ppo.clip
            ) * advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            with torch.inference_mode():
                reference_logits = (
                    reference(observations[indices]) / temperature
                )
            anchor_kl = full_policy_kl(
                reference_logits, logits, masks[indices]
            )
            entropy = distribution.entropy().mean()
            loss = (
                policy_loss
                + kl_coefficient * anchor_kl
                - entropy_coefficient * entropy
            )
            ppo.actor_optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.actor.parameters(), 0.5)
            ppo.actor_optim.step()
            losses.append(float(policy_loss.detach()))
            anchor_kls.append(float(anchor_kl.detach()))
            entropies.append(float(entropy.detach()))
            clip_fractions.append(
                float(((ratios - 1.0).abs() > ppo.clip).float().mean())
            )
        actor_log = {
            "epoch": epoch + 1,
            "policy_loss": float(np.mean(losses)),
            "reference_kl": float(np.mean(anchor_kls)),
            "entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }
        actor_logs.append(actor_log)
        if actor_log["reference_kl"] > target_kl:
            break

    critic_logs = []
    q_critic.train()
    for epoch in range(critic_epochs):
        order = torch.randperm(
            len(actions), generator=generator, device=ppo.actor_device
        )
        losses = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            predicted_q = q_critic(observations[indices]).gather(
                1, actions[indices].unsqueeze(1)
            ).squeeze(1)
            loss = torch.nn.functional.mse_loss(
                predicted_q, fixed_q_targets[indices]
            )
            q_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_critic.parameters(), 1.0)
            q_optimizer.step()
            losses.append(float(loss.detach()))
        critic_logs.append(
            {
                "epoch": epoch + 1,
                "q_loss": float(np.mean(losses)),
            }
        )
    q_critic.eval()
    ppo._sync_rollout_actor()
    return {
        "algorithm": "information_safe_qboosted_ppo",
        "paper_exact_vrpo": False,
        "paper": PAPER,
        "critic_information": (
            "same 121-feature own-hand/public observation as actor"
        ),
        "games": episodes,
        "decisions": len(actions),
        "rollout": dict(ppo.last_rollout_metrics),
        "max_initial_ratio_error": max_initial_ratio_error,
        "trace_lambda": trace_lambda,
        "potential_shaping": shaping_metrics,
        "behavior_advantage_mean": float(
            behavior_estimate.advantages.mean()
        ),
        "behavior_advantage_std": float(
            behavior_estimate.advantages.std(unbiased=False)
        ),
        "behavior_expected_sarsa_residual_rmse": float(
            behavior_estimate.expected_sarsa_residuals.square().mean().sqrt()
        ),
        "actor_epochs": actor_logs,
        "critic_epochs": critic_logs,
        "wall_seconds": time.perf_counter() - started,
    }


def validate_development_seed(seed: int) -> None:
    if seed < 0 or seed >= SEALED_SEED_START:
        raise ValueError("Q-boosting pilot cannot use sealed final seeds")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion", type=Path, default=DEFAULT_TEACHER
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=27_601)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--actor-epochs", type=int, default=2)
    parser.add_argument("--critic-epochs", type=int, default=4)
    parser.add_argument(
        "--q-warmup-iterations",
        type=int,
        default=0,
        help=(
            "critic-only on-policy warm-up iterations before actor updates "
            "(default: disabled for backward compatibility)"
        ),
    )
    parser.add_argument("--q-warmup-games", type=int, default=512)
    parser.add_argument("--q-warmup-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=DEFAULT_TORCH_THREADS,
        help=(
            "PyTorch intra-op threads for rollout and updates "
            f"(default: {DEFAULT_TORCH_THREADS})"
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--trace-lambda", type=float, default=0.95)
    parser.add_argument("--kl-coefficient", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.003)
    parser.add_argument("--entropy-coefficient", type=float, default=0.002)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--q-learning-rate", type=float, default=3e-5)
    parser.add_argument(
        "--past-policy-probability", type=float, default=0.2
    )
    parser.add_argument(
        "--agent-sets", default="self,self;smart,rl;random,rl"
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    validate_development_seed(args.seed)
    champion = args.champion.resolve()
    if file_sha256(champion) != CHAMPION_SHA256:
        raise RuntimeError("refusing to train from an unverified champion")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    output_dir = args.output_dir.resolve()
    agent_sets = parse_agent_sets(args.agent_sets)

    with working_directory(output_dir):
        ppo = PPO_Supervised(
            use_pretrained=True,
            actor_path=str(champion),
            critic_path="missing_scalar_critic.pt",
            actor_opt_path="missing_actor_optimizer.pt",
            critic_opt_path="missing_scalar_critic_optimizer.pt",
            old_save=(),
            max_saves=50,
            rollout_device="cpu",
            rollout_batch_size=512,
            training_actor_device=(
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
            compile_updates=False,
            past_policy_probability=args.past_policy_probability,
            torch_threads=args.torch_threads,
        )
        for group in ppo.actor_optim.param_groups:
            group["lr"] = args.learning_rate
        reference = copy.deepcopy(ppo.actor).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        q_critic = PublicObservationQCritic().to(ppo.actor_device)
        q_optimizer = torch.optim.AdamW(
            q_critic.parameters(), lr=args.q_learning_rate
        )
        warmup = fit_q_critic(
            ppo,
            q_critic,
            q_optimizer,
            iterations=args.q_warmup_iterations,
            episodes=args.q_warmup_games,
            critic_epochs=args.q_warmup_epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            agent_sets=agent_sets,
            trace_lambda=args.trace_lambda,
            seed=args.seed + 50_000,
        )
        training = []
        for iteration in range(args.iterations):
            metrics = qboosted_anchored_iteration(
                ppo,
                reference,
                q_critic,
                q_optimizer,
                episodes=args.games,
                actor_epochs=args.actor_epochs,
                critic_epochs=args.critic_epochs,
                batch_size=args.batch_size,
                temperature=args.temperature,
                agent_sets=agent_sets,
                kl_coefficient=args.kl_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                target_kl=args.target_kl,
                trace_lambda=args.trace_lambda,
                seed=args.seed + iteration,
            )
            metrics["iteration"] = iteration + 1
            training.append(metrics)
            torch.save(
                ppo.actor.state_dict(),
                f"qboosted_actor_{iteration + 1}.pth",
            )
            torch.save(
                q_critic.state_dict(),
                f"qboosted_qcritic_{iteration + 1}.pth",
            )
        payload = {
            "schema_version": 1,
            "algorithm": "information_safe_qboosted_ppo",
            "paper": PAPER,
            "paper_exact_vrpo": False,
            "deviation_from_paper": (
                "human-information boundary replaces the paper's centralized "
                "full-state critic with the actor's own-hand/public observation"
            ),
            "seed": args.seed,
            "champion": {
                "path": str(champion),
                "sha256": CHAMPION_SHA256,
            },
            "information_boundary": {
                "critic_input_features": OBSERVATION_SIZE,
                "opponent_card_identities": False,
                "draw_pile_order": False,
            },
            "recipe": {
                "iterations": args.iterations,
                "games": args.games,
                "actor_epochs": args.actor_epochs,
                "critic_epochs": args.critic_epochs,
                "q_warmup_iterations": args.q_warmup_iterations,
                "q_warmup_games": args.q_warmup_games,
                "q_warmup_epochs": args.q_warmup_epochs,
                "batch_size": args.batch_size,
                "torch_threads": ppo.torch_threads,
                "temperature": args.temperature,
                "trace_lambda": args.trace_lambda,
                "kl_coefficient": args.kl_coefficient,
                "target_kl": args.target_kl,
                "entropy_coefficient": args.entropy_coefficient,
                "learning_rate": args.learning_rate,
                "q_learning_rate": args.q_learning_rate,
                "past_policy_probability": (
                    args.past_policy_probability
                ),
                "agent_sets": agent_sets,
            },
            "q_critic_warmup": warmup,
            "training": training,
            "final_actor": str(
                output_dir / f"qboosted_actor_{args.iterations}.pth"
            ),
            "final_qcritic": str(
                output_dir / f"qboosted_qcritic_{args.iterations}.pth"
            ),
        }
        Path("training_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
