#!/usr/bin/env python3
"""Critic-warm-started, frozen-champion-anchored PPO continuation.

The production checkpoint contains only actor weights.  Continuing PPO with a
random critic produced a negative pilot, so this recipe first fits the critic
to Monte-Carlo returns while the actor is frozen.  Policy updates then retain a
full-distribution KL anchor to the immutable production actor.

All rollouts use the existing 121-feature policy observation.  It contains the
player's own hand and public game state, never opponent card identities or the
draw-pile order.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.distributions import Categorical

from ai_observation import HAND_END, HAND_START, OBSERVATION_SIZE
from ppo2 import (
    PPO_Supervised,
    StudentNet,
    compute_gae,
)


CHAMPION_SHA256 = (
    "aac589b3c7300895604e6c77755631ab4a31c8bf9810768448ab2f3ff5bbe186"
)
MAX_INITIAL_RATIO_ERROR = 0.02
INITIAL_HAND_SIZE = 7.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_policy_kl(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    legal_actions: torch.Tensor,
) -> torch.Tensor:
    """Return mean KL(reference || candidate) over legal actions."""
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError("reference and candidate logits must have equal shape")
    if legal_actions.shape != reference_logits.shape:
        raise ValueError("legal mask must match the logits")
    floor = torch.finfo(reference_logits.dtype).min
    reference_log_probs = torch.log_softmax(
        reference_logits.masked_fill(~legal_actions, floor), dim=-1
    )
    candidate_log_probs = torch.log_softmax(
        candidate_logits.masked_fill(~legal_actions, floor), dim=-1
    )
    reference_probs = reference_log_probs.exp()
    return (
        reference_probs * (reference_log_probs - candidate_log_probs)
    ).sum(dim=-1).mean()


def monte_carlo_returns(
    rewards: torch.Tensor,
    episode_lengths: list[int],
    *,
    gamma: float,
) -> torch.Tensor:
    """Compute exact discounted rollout returns without a critic bootstrap."""
    zeros = torch.zeros_like(rewards)
    _, returns = compute_gae(
        rewards,
        zeros,
        episode_lengths=episode_lengths,
        gamma_=gamma,
        lam=1.0,
    )
    return returns


def own_hand_potential(observations: torch.Tensor) -> torch.Tensor:
    """Return a bounded progress potential from own-card counts only."""
    if observations.shape[-1] != OBSERVATION_SIZE:
        raise ValueError(
            f"expected observations with {OBSERVATION_SIZE} features, "
            f"got {observations.shape[-1]}"
        )
    own_hand_size = observations[..., HAND_START:HAND_END].sum(dim=-1)
    return -own_hand_size / INITIAL_HAND_SIZE


def potential_shaped_rewards(
    rewards: torch.Tensor,
    observations: torch.Tensor,
    episode_lengths: list[int],
    *,
    gamma: float,
    coefficient: float,
) -> tuple[torch.Tensor, dict]:
    """Add c * (gamma * Phi(next) - Phi(current)) within each episode."""
    if coefficient == 0.0:
        shaping = torch.zeros_like(rewards)
    else:
        if rewards.ndim != 1:
            raise ValueError("rewards must be one-dimensional")
        if len(observations) != len(rewards):
            raise ValueError("one observation is required per reward")
        if sum(episode_lengths) != len(rewards):
            raise ValueError(
                "episode lengths must cover every reward exactly once"
            )
        potentials = own_hand_potential(observations).to(rewards.device)
        next_potentials = torch.zeros_like(potentials)
        offset = 0
        for length in episode_lengths:
            if length < 0:
                raise ValueError("episode lengths cannot be negative")
            stop = offset + length
            if length > 1:
                next_potentials[offset : stop - 1] = potentials[
                    offset + 1 : stop
                ]
            # The final sample transitions to a terminal state with Phi=0.
            offset = stop
        shaping = coefficient * (
            gamma * next_potentials - potentials
        )
    shaped = rewards if coefficient == 0.0 else rewards + shaping
    return shaped, {
        "coefficient": float(coefficient),
        "potential": "negative_own_hand_count_over_initial_hand_size",
        "input_features": f"own_hand_counts[{HAND_START}:{HAND_END}]",
        "raw_reward_mean": float(rewards.mean()),
        "shaping_reward_mean": float(shaping.mean()),
        "max_abs_shaping_reward": float(shaping.abs().max()),
        "shaped_reward_mean": float(shaped.mean()),
    }


def explained_variance(
    predictions: torch.Tensor, targets: torch.Tensor
) -> float:
    target_variance = targets.var(unbiased=False)
    if float(target_variance) == 0.0:
        return float("nan")
    residual_variance = (targets - predictions).var(unbiased=False)
    return float(1.0 - residual_variance / target_variance)


def enforce_initial_ratio_invariant(ratios: torch.Tensor) -> float:
    """Verify that rollout and update policies agree before optimization."""
    max_error = float(ratios.detach().sub(1).abs().max())
    if not np.isfinite(max_error) or max_error > MAX_INITIAL_RATIO_ERROR:
        raise RuntimeError(
            "rollout/update policy mismatch: unchanged-policy "
            f"ratio error reached {max_error:.5f}"
        )
    return max_error


def _chunked_values(ppo: PPO_Supervised, observations, chunk_size: int):
    values = []
    with torch.inference_mode():
        for start in range(0, len(observations), chunk_size):
            values.append(
                ppo.critic(observations[start : start + chunk_size]).squeeze(-1)
            )
    return torch.cat(values)


def fit_critic(
    ppo: PPO_Supervised,
    *,
    iterations: int,
    episodes: int,
    epochs: int,
    batch_size: int,
    temperature: float,
    agent_sets: list[list[str]],
    gamma: float = 0.97,
    potential_shaping_coefficient: float = 0.0,
) -> list[dict]:
    """Fit only the critic; assert the production actor remains byte-identical."""
    actor_before = {
        key: value.detach().cpu().clone()
        for key, value in ppo.actor.state_dict().items()
    }
    history = []
    for iteration in range(iterations):
        started = time.perf_counter()
        observations, _, _, rewards, lengths = ppo.generate_data(
            num_episodes=episodes,
            temperature=temperature,
            agent_sets=agent_sets,
        )
        critic_observations = observations.to(ppo.device)
        training_rewards, shaping_metrics = potential_shaped_rewards(
            rewards,
            observations,
            lengths,
            gamma=gamma,
            coefficient=potential_shaping_coefficient,
        )
        targets = monte_carlo_returns(
            training_rewards.to(ppo.device), lengths, gamma=gamma
        )
        before = _chunked_values(ppo, critic_observations, batch_size)
        before_loss = float(torch.nn.functional.mse_loss(before, targets))
        generator = torch.Generator(device=ppo.device).manual_seed(
            70_001 + iteration
        )
        losses = []
        ppo.critic.train()
        for _ in range(epochs):
            order = torch.randperm(
                len(targets), generator=generator, device=ppo.device
            )
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                prediction = ppo.critic(
                    critic_observations[indices]
                ).squeeze(-1)
                loss = torch.nn.functional.smooth_l1_loss(
                    prediction, targets[indices]
                )
                ppo.critic_optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo.critic.parameters(), 1.0)
                ppo.critic_optim.step()
                losses.append(float(loss.detach()))
        ppo.critic.eval()
        after = _chunked_values(ppo, critic_observations, batch_size)
        history.append(
            {
                "iteration": iteration + 1,
                "games": episodes,
                "decisions": len(targets),
                "mse_before": before_loss,
                "mse_after": float(
                    torch.nn.functional.mse_loss(after, targets)
                ),
                "explained_variance_after": explained_variance(after, targets),
                "mean_training_loss": float(np.mean(losses)),
                "rollout": dict(ppo.last_rollout_metrics),
                "potential_shaping": shaping_metrics,
                "wall_seconds": time.perf_counter() - started,
            }
        )
    actor_after = ppo.actor.state_dict()
    if any(
        not torch.equal(actor_before[key], actor_after[key].detach().cpu())
        for key in actor_before
    ):
        raise RuntimeError("critic warm-up changed frozen actor parameters")
    return history


def anchored_iteration(
    ppo: PPO_Supervised,
    reference: StudentNet,
    *,
    episodes: int,
    update_epochs: int,
    batch_size: int,
    temperature: float,
    agent_sets: list[list[str]],
    kl_coefficient: float,
    entropy_coefficient: float,
    target_kl: float,
    gamma: float = 0.97,
    gae_lambda: float = 0.95,
    potential_shaping_coefficient: float = 0.0,
) -> dict:
    """Run one rollout and minibatched PPO update with a fixed actor anchor."""
    started = time.perf_counter()
    observations, actions, old_log_probs, rewards, lengths = ppo.generate_data(
        num_episodes=episodes,
        temperature=temperature,
        agent_sets=agent_sets,
    )
    critic_observations = observations.to(ppo.device)
    training_rewards, shaping_metrics = potential_shaped_rewards(
        rewards,
        observations,
        lengths,
        gamma=gamma,
        coefficient=potential_shaping_coefficient,
    )
    with torch.inference_mode():
        old_values = _chunked_values(
            ppo, critic_observations, batch_size
        )
        advantages, returns = compute_gae(
            training_rewards.to(ppo.device),
            old_values,
            episode_lengths=lengths,
            gamma_=gamma,
            lam=gae_lambda,
        )
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)
    actor_advantages = advantages.to(ppo.actor_device)
    actor_returns = returns.to(ppo.device)
    masks = ppo.get_action_masks(observations)
    epoch_logs = []
    generator = torch.Generator(device=ppo.actor_device).manual_seed(
        random.randrange(1, 2**31)
    )
    max_initial_ratio_error = None

    for epoch in range(update_epochs):
        order = torch.randperm(
            len(actions), generator=generator, device=ppo.actor_device
        )
        policy_losses = []
        critic_losses = []
        kls = []
        entropies = []
        clip_fractions = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_observations = observations[indices]
            batch_masks = masks[indices]
            candidate_logits = ppo.actor(batch_observations) / temperature
            with torch.inference_mode():
                reference_logits = reference(batch_observations) / temperature
            floor = torch.finfo(candidate_logits.dtype).min
            distribution = Categorical(
                logits=candidate_logits.masked_fill(~batch_masks, floor),
                validate_args=False,
            )
            new_log_probs = distribution.log_prob(actions[indices])
            ratios = (new_log_probs - old_log_probs[indices]).exp()
            if epoch == 0 and start == 0:
                max_initial_ratio_error = enforce_initial_ratio_invariant(
                    ratios
                )
            unclipped = ratios * actor_advantages[indices]
            clipped = torch.clamp(
                ratios, 1.0 - ppo.clip, 1.0 + ppo.clip
            ) * actor_advantages[indices]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            anchor_kl = full_policy_kl(
                reference_logits, candidate_logits, batch_masks
            )
            entropy = distribution.entropy().mean()

            critic_indices = indices.to(ppo.device)
            values = ppo.critic(
                critic_observations[critic_indices]
            ).squeeze(-1)
            critic_loss = torch.nn.functional.mse_loss(
                values, actor_returns[critic_indices]
            )
            loss = (
                policy_loss
                + kl_coefficient * anchor_kl
                - entropy_coefficient * entropy
                + 0.8 * critic_loss
            )
            ppo.actor_optim.zero_grad(set_to_none=True)
            ppo.critic_optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(ppo.critic.parameters(), 1.0)
            ppo.actor_optim.step()
            ppo.critic_optim.step()

            policy_losses.append(float(policy_loss.detach()))
            critic_losses.append(float(critic_loss.detach()))
            kls.append(float(anchor_kl.detach()))
            entropies.append(float(entropy.detach()))
            clip_fractions.append(
                float(((ratios - 1.0).abs() > ppo.clip).float().mean())
            )
        epoch_log = {
            "epoch": epoch + 1,
            "policy_loss": float(np.mean(policy_losses)),
            "critic_loss": float(np.mean(critic_losses)),
            "reference_kl": float(np.mean(kls)),
            "entropy": float(np.mean(entropies)),
            "clip_fraction": float(np.mean(clip_fractions)),
        }
        epoch_logs.append(epoch_log)
        if epoch_log["reference_kl"] > target_kl:
            break
    ppo._sync_rollout_actor()
    return {
        "games": episodes,
        "decisions": len(actions),
        "rollout": dict(ppo.last_rollout_metrics),
        "max_initial_ratio_error": max_initial_ratio_error,
        "potential_shaping": shaping_metrics,
        "update_epochs": epoch_logs,
        "wall_seconds": time.perf_counter() - started,
    }


def parse_agent_sets(value: str) -> list[list[str]]:
    groups = [
        [entry.strip() for entry in group.split(",") if entry.strip()]
        for group in value.split(";")
    ]
    return [group for group in groups if group]


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion", type=Path, default=Path("ppo_actor_100run3.pth")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--critic-iterations", type=int, default=4)
    parser.add_argument("--critic-games", type=int, default=2048)
    parser.add_argument("--critic-epochs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--kl-coefficient", type=float, default=1.0)
    parser.add_argument("--target-kl", type=float, default=0.003)
    parser.add_argument("--entropy-coefficient", type=float, default=0.002)
    parser.add_argument(
        "--champion-opponent-probability",
        "--past-policy-probability",
        dest="past_policy_probability",
        type=float,
        default=0.2,
        help=(
            "per eligible game probability of replacing one RL seat with "
            "the frozen checkpoint-0 champion (default: 0.2)"
        ),
    )
    parser.add_argument(
        "--potential-shaping-coefficient",
        type=float,
        default=0.0,
        help=(
            "coefficient for policy-invariant own-hand potential shaping "
            "(default: disabled)"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--agent-sets", default="self,self;self,self,rl;smart,rl;random,rl"
    )
    args = parser.parse_args()

    champion = args.champion.resolve()
    output_dir = args.output_dir.resolve()
    if file_sha256(champion) != CHAMPION_SHA256:
        raise RuntimeError("refusing to train from an unverified champion")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")
    agent_sets = parse_agent_sets(args.agent_sets)

    with working_directory(output_dir):
        ppo = PPO_Supervised(
            use_pretrained=True,
            actor_path=str(champion),
            critic_path="missing_critic.pt",
            actor_opt_path="missing_actor_optimizer.pt",
            critic_opt_path="missing_critic_optimizer.pt",
            old_save=(),
            max_saves=50,
            rollout_device="cpu",
            rollout_batch_size=512,
            training_actor_device="cuda" if torch.cuda.is_available() else "cpu",
            compile_updates=False,
            past_policy_probability=args.past_policy_probability,
        )
        for group in ppo.actor_optim.param_groups:
            group["lr"] = args.learning_rate
        reference = StudentNet().to(ppo.actor_device)
        reference.load_state_dict(
            torch.load(champion, map_location=ppo.actor_device, weights_only=True)
        )
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)

        warmup = fit_critic(
            ppo,
            iterations=args.critic_iterations,
            episodes=args.critic_games,
            epochs=args.critic_epochs,
            batch_size=args.batch_size,
            temperature=args.temperature,
            agent_sets=agent_sets,
            potential_shaping_coefficient=(
                args.potential_shaping_coefficient
            ),
        )
        training = []
        for iteration in range(args.iterations):
            metrics = anchored_iteration(
                ppo,
                reference,
                episodes=args.games,
                update_epochs=args.update_epochs,
                batch_size=args.batch_size,
                temperature=args.temperature,
                agent_sets=agent_sets,
                kl_coefficient=args.kl_coefficient,
                entropy_coefficient=args.entropy_coefficient,
                target_kl=args.target_kl,
                potential_shaping_coefficient=(
                    args.potential_shaping_coefficient
                ),
            )
            metrics["iteration"] = iteration + 1
            training.append(metrics)
            torch.save(
                ppo.actor.state_dict(),
                f"anchored_actor_{iteration + 1}.pth",
            )
        torch.save(ppo.critic.state_dict(), "anchored_critic_final.pth")
        payload = {
            "schema_version": 1,
            "seed": args.seed,
            "champion": {
                "path": str(champion),
                "sha256": CHAMPION_SHA256,
            },
            "privacy": {
                "observation": "own hand plus public state only",
                "opponent_card_identities": False,
                "draw_pile_order": False,
                "potential_shaping_inputs": "own hand counts only",
            },
            "recipe": {
                "critic_iterations": args.critic_iterations,
                "critic_games": args.critic_games,
                "critic_epochs": args.critic_epochs,
                "policy_iterations": args.iterations,
                "games_per_iteration": args.games,
                "update_epochs": args.update_epochs,
                "batch_size": args.batch_size,
                "temperature": args.temperature,
                "kl_coefficient": args.kl_coefficient,
                "target_kl": args.target_kl,
                "entropy_coefficient": args.entropy_coefficient,
                "champion_opponent_probability": (
                    args.past_policy_probability
                ),
                "potential_shaping_coefficient": (
                    args.potential_shaping_coefficient
                ),
                "learning_rate": args.learning_rate,
                "agent_sets": agent_sets,
            },
            "critic_warmup": warmup,
            "training": training,
            "final_actor": str(output_dir / f"anchored_actor_{args.iterations}.pth"),
        }
        Path("training_metrics.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "final_actor": payload["final_actor"],
                    "critic_warmup": warmup[-1] if warmup else None,
                    "last_iteration": training[-1] if training else None,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
