#!/usr/bin/env python3
"""KL-regularized Monte Carlo policy improvement against a random opponent.

The policy still receives only the established 121 public/own-hand features.
Fresh games are sampled from the warm actor, terminal outcomes are propagated
back through each trajectory, and a full-distribution KL term keeps every
update close to the behavior policy. This deliberately differs from PPO:
there is no clipped importance-ratio surrogate or learned value baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from ppo2 import PPO_Supervised, build_action_masks


def state_sha256(path: Path) -> str:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()

    def update(value) -> None:
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            for name, child in sorted(value.items(), key=lambda item: str(item[0])):
                digest.update(str(name).encode("utf-8"))
                update(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                update(child)
        else:
            digest.update(repr(value).encode("utf-8"))

    update(loaded)
    return digest.hexdigest()


def trajectory_advantages(
    rewards: torch.Tensor,
    lengths: list[int],
    *,
    gamma: float,
    balance_episodes: bool,
) -> tuple[torch.Tensor, dict]:
    advantages = torch.empty_like(rewards)
    offset = 0
    wins = losses = draws = 0
    for length in lengths:
        terminal = float(rewards[offset + length - 1])
        if terminal > 0.5:
            outcome = 1.0
            wins += 1
        elif terminal < -0.5:
            outcome = -1.0
            losses += 1
        else:
            outcome = 0.0
            draws += 1
        scale = 1.0 / math.sqrt(length) if balance_episodes else 1.0
        exponents = torch.arange(
            length - 1,
            -1,
            -1,
            dtype=rewards.dtype,
            device=rewards.device,
        )
        advantages[offset:offset + length] = (
            outcome * scale * torch.pow(gamma, exponents)
        )
        offset += length
    if offset != len(rewards):
        raise RuntimeError("episode lengths do not cover rewards")
    advantages = advantages - advantages.mean()
    advantages = advantages / advantages.std().clamp_min(1e-8)
    return advantages, {
        "games": len(lengths),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "sampled_score": (wins + 0.5 * draws) / len(lengths),
    }


def update_actor(
    ppo: PPO_Supervised,
    observations: torch.Tensor,
    actions: torch.Tensor,
    advantages: torch.Tensor,
    *,
    temperature: float,
    epochs: int,
    batch_size: int,
    kl_coefficient: float,
    entropy_coefficient: float,
    gradient_clip: float,
    seed: int,
) -> dict:
    actor = ppo.actor
    actor.eval()
    masks = build_action_masks(observations)
    with torch.inference_mode():
        old_logits = actor(observations) / temperature
        old_logits = old_logits.masked_fill(
            ~masks,
            torch.finfo(old_logits.dtype).min,
        )
        old_log_probabilities = torch.log_softmax(old_logits, dim=-1)
        old_probabilities = old_log_probabilities.exp()

    generator = torch.Generator(device=observations.device).manual_seed(seed)
    records = []
    for epoch in range(epochs):
        order = torch.randperm(
            len(observations),
            generator=generator,
            device=observations.device,
        )
        policy_losses = []
        kls = []
        entropies = []
        gradient_norms = []
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            logits = actor(observations[indices]) / temperature
            logits = logits.masked_fill(
                ~masks[indices],
                torch.finfo(logits.dtype).min,
            )
            log_probabilities = torch.log_softmax(logits, dim=-1)
            selected = log_probabilities.gather(
                1,
                actions[indices].unsqueeze(-1),
            ).squeeze(-1)
            policy_loss = -(
                advantages[indices].detach() * selected
            ).mean()
            kl = (
                old_probabilities[indices]
                * torch.where(
                    masks[indices],
                    old_log_probabilities[indices] - log_probabilities,
                    torch.zeros_like(log_probabilities),
                )
            ).sum(dim=-1).mean()
            entropy = -torch.where(
                masks[indices],
                log_probabilities.exp() * log_probabilities,
                torch.zeros_like(log_probabilities),
            ).sum(dim=-1).mean()
            loss = (
                policy_loss
                + kl_coefficient * kl
                - entropy_coefficient * entropy
            )
            ppo.actor_optim.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                actor.parameters(),
                gradient_clip,
            )
            ppo.actor_optim.step()
            policy_losses.append(float(policy_loss.detach()))
            kls.append(float(kl.detach()))
            entropies.append(float(entropy.detach()))
            gradient_norms.append(float(gradient_norm.detach()))
        records.append(
            {
                "epoch": epoch + 1,
                "policy_loss": float(np.mean(policy_losses)),
                "behavior_kl": float(np.mean(kls)),
                "entropy": float(np.mean(entropies)),
                "gradient_norm": float(np.mean(gradient_norms)),
            }
        )
    actor.eval()
    return {"epochs": records}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-actor", type=Path, required=True)
    parser.add_argument("--resume-critic", type=Path, required=True)
    parser.add_argument("--resume-actor-optimizer", type=Path, required=True)
    parser.add_argument("--resume-critic-optimizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--kl-coefficient", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.001)
    parser.add_argument("--gradient-clip", type=float, default=0.5)
    parser.add_argument("--torch-threads", type=int, default=6)
    parser.add_argument(
        "--balance-episodes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this experiment requires CUDA")
    if min(args.iterations, args.games, args.epochs, args.batch_size) < 1:
        raise ValueError("iterations, games, epochs, and batch size must be positive")
    if not 0 < args.temperature or not 0 < args.learning_rate:
        raise ValueError("temperature and learning rate must be positive")
    if not 0 < args.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if min(
        args.kl_coefficient,
        args.entropy_coefficient,
        args.gradient_clip,
    ) < 0:
        raise ValueError("regularization and clipping values cannot be negative")

    paths = {
        "actor": args.resume_actor.resolve(),
        "critic": args.resume_critic.resolve(),
        "actor_optimizer": args.resume_actor_optimizer.resolve(),
        "critic_optimizer": args.resume_critic_optimizer.resolve(),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {name}: {path}")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    ppo = PPO_Supervised(
        use_pretrained=True,
        actor_path=str(paths["actor"]),
        critic_path=str(paths["critic"]),
        actor_opt_path=str(paths["actor_optimizer"]),
        critic_opt_path=str(paths["critic_optimizer"]),
        old_save=(),
        max_saves=max(20, args.iterations + 2),
        rollout_device="cpu",
        rollout_batch_size=1024,
        training_actor_device="cuda",
        compile_updates=False,
        past_policy_probability=0.0,
        torch_threads=args.torch_threads,
    )
    for parameter_group in ppo.actor_optim.param_groups:
        parameter_group["lr"] = args.learning_rate

    started = time.perf_counter()
    curve = []
    for iteration in range(1, args.iterations + 1):
        iteration_started = time.perf_counter()
        observations, actions, _, rewards, lengths = ppo.generate_data(
            num_episodes=args.games,
            temperature=args.temperature,
            agent_sets=[["random", "rl"]],
        )
        advantages, outcomes = trajectory_advantages(
            rewards,
            lengths,
            gamma=args.gamma,
            balance_episodes=args.balance_episodes,
        )
        update = update_actor(
            ppo,
            observations,
            actions,
            advantages,
            temperature=args.temperature,
            epochs=args.epochs,
            batch_size=args.batch_size,
            kl_coefficient=args.kl_coefficient,
            entropy_coefficient=args.entropy_coefficient,
            gradient_clip=args.gradient_clip,
            seed=args.seed + iteration,
        )
        actor_path = output_dir / f"reinforce_actor_{iteration:03d}.pth"
        optimizer_path = output_dir / (
            f"reinforce_actor_optimizer_{iteration:03d}.pth"
        )
        torch.save(ppo.actor.state_dict(), actor_path)
        torch.save(ppo.actor_optim.state_dict(), optimizer_path)
        record = {
            "iteration": iteration,
            "outcomes": outcomes,
            "decisions": len(observations),
            "update": update,
            "actor": {
                "path": str(actor_path),
                "sha256": state_sha256(actor_path),
            },
            "actor_optimizer": str(optimizer_path),
            "wall_seconds": time.perf_counter() - iteration_started,
        }
        curve.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    payload = {
        "schema_version": 1,
        "kind": "kl_regularized_monte_carlo_policy_gradient",
        "seed": args.seed,
        "privacy": {
            "observation": "121 public and own-hand features only",
            "opponent_card_identities": False,
            "draw_pile_order": False,
        },
        "warm_parent": {
            name: {"path": str(path), "sha256": state_sha256(path)}
            for name, path in paths.items()
        },
        "config": {
            "iterations": args.iterations,
            "games": args.games,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "kl_coefficient": args.kl_coefficient,
            "entropy_coefficient": args.entropy_coefficient,
            "gradient_clip": args.gradient_clip,
            "balance_episodes": args.balance_episodes,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
        },
        "curve": curve,
        "elapsed_seconds": time.perf_counter() - started,
        "latest_actor": curve[-1]["actor"],
    }
    (output_dir / "reinforce_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
