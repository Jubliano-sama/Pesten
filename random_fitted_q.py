#!/usr/bin/env python3
"""Fit public-state action values from fresh games against random opponents."""

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
from torch.nn import functional as F

from fitted_q_policy import ActionValueNet
from ppo2 import PPO_Supervised


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_temperatures(value: str) -> list[float]:
    temperatures = [
        float(item.strip()) for item in value.split(",") if item.strip()
    ]
    if not temperatures or any(item <= 0 for item in temperatures):
        raise ValueError("temperatures must be positive")
    return temperatures


def return_targets(
    rewards: torch.Tensor,
    lengths: list[int],
    *,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    targets = torch.empty_like(rewards, device="cpu")
    episode_ids = torch.empty(len(rewards), dtype=torch.long)
    offset = 0
    wins = draws = losses = 0
    for episode, length in enumerate(lengths):
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
        exponents = torch.arange(length - 1, -1, -1, dtype=torch.float32)
        targets[offset : offset + length] = outcome * torch.pow(gamma, exponents)
        episode_ids[offset : offset + length] = episode
        offset += length
    if offset != len(rewards):
        raise RuntimeError("episode lengths do not cover rewards")
    return targets, episode_ids, {
        "games": len(lengths),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "sampled_score": (wins + 0.5 * draws) / len(lengths),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-actor", type=Path, required=True)
    parser.add_argument("--resume-critic", type=Path, required=True)
    parser.add_argument("--resume-actor-optimizer", type=Path, required=True)
    parser.add_argument("--resume-critic-optimizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=65_536)
    parser.add_argument("--temperatures", default="0.8,1.0,1.25")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--torch-threads", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temperatures = parse_temperatures(args.temperatures)
    if not torch.cuda.is_available():
        raise SystemExit("this experiment requires CUDA")
    if min(args.games, args.width, args.blocks, args.epochs, args.batch_size) < 1:
        raise ValueError("positive games/model/training sizes required")
    if not 0 < args.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be in (0, 0.5)")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("invalid optimizer settings")

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
        max_saves=2,
        rollout_device="cpu",
        rollout_batch_size=1024,
        training_actor_device="cuda",
        compile_updates=False,
        past_policy_probability=0.0,
        torch_threads=args.torch_threads,
    )
    ppo.actor.eval().requires_grad_(False)

    started = time.perf_counter()
    observations_parts = []
    actions_parts = []
    targets_parts = []
    episode_parts = []
    collection = []
    episode_offset = 0
    games_per_temperature = [args.games // len(temperatures)] * len(temperatures)
    games_per_temperature[-1] += args.games - sum(games_per_temperature)
    for temperature, games in zip(temperatures, games_per_temperature):
        observations, actions, _, rewards, lengths = ppo.generate_data(
            num_episodes=games,
            temperature=temperature,
            agent_sets=[["random", "rl"]],
        )
        targets, episode_ids, outcomes = return_targets(
            rewards.detach().cpu(),
            lengths,
            gamma=args.gamma,
        )
        observations_parts.append(observations.detach().cpu())
        actions_parts.append(actions.detach().cpu())
        targets_parts.append(targets)
        episode_parts.append(episode_ids + episode_offset)
        episode_offset += len(lengths)
        collection.append(
            {
                "temperature": temperature,
                "decisions": len(observations),
                **outcomes,
                "rollout": dict(ppo.last_rollout_metrics),
            }
        )
        print(json.dumps(collection[-1], sort_keys=True), flush=True)

    observations = torch.cat(observations_parts)
    actions = torch.cat(actions_parts)
    targets = torch.cat(targets_parts)
    episode_ids = torch.cat(episode_parts)
    episode_count = int(episode_ids.max()) + 1
    split_generator = torch.Generator().manual_seed(args.seed + 1)
    validation_episodes = torch.randperm(
        episode_count,
        generator=split_generator,
    )[: max(1, round(episode_count * args.validation_fraction))]
    validation_episode_mask = torch.zeros(episode_count, dtype=torch.bool)
    validation_episode_mask[validation_episodes] = True
    validation_mask = validation_episode_mask[episode_ids]
    train_indices = torch.nonzero(
        ~validation_mask,
        as_tuple=False,
    ).flatten()
    validation_indices = torch.nonzero(
        validation_mask,
        as_tuple=False,
    ).flatten()

    episode_lengths = torch.bincount(episode_ids, minlength=episode_count)
    sample_weights = episode_lengths[episode_ids].float().reciprocal()
    sample_weights *= sample_weights.numel() / sample_weights.sum()

    model = ActionValueNet(width=args.width, blocks=args.blocks).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    training_curve = []
    order_generator = torch.Generator().manual_seed(args.seed + 2)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_indices[
            torch.randperm(len(train_indices), generator=order_generator)
        ]
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            batch_observations = observations[indices].cuda(non_blocking=True)
            batch_actions = actions[indices].cuda(non_blocking=True)
            batch_targets = targets[indices].cuda(non_blocking=True)
            batch_weights = sample_weights[indices].cuda(non_blocking=True)
            predictions = model(batch_observations).gather(
                1,
                batch_actions.unsqueeze(-1),
            ).squeeze(-1)
            element_loss = F.smooth_l1_loss(
                predictions,
                batch_targets,
                reduction="none",
                beta=0.5,
            )
            loss = (element_loss * batch_weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        validation_losses = []
        validation_predictions = []
        validation_targets = []
        with torch.inference_mode():
            for start in range(0, len(validation_indices), args.batch_size):
                indices = validation_indices[start : start + args.batch_size]
                predicted = model(
                    observations[indices].cuda(non_blocking=True)
                ).gather(
                    1,
                    actions[indices].cuda(non_blocking=True).unsqueeze(-1),
                ).squeeze(-1)
                expected = targets[indices].cuda(non_blocking=True)
                validation_losses.append(
                    float(F.smooth_l1_loss(predicted, expected))
                )
                validation_predictions.append(predicted.cpu())
                validation_targets.append(expected.cpu())
        predicted = torch.cat(validation_predictions)
        expected = torch.cat(validation_targets)
        correlation = float(
            torch.corrcoef(torch.stack((predicted, expected)))[0, 1]
        )
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": float(np.mean(validation_losses)),
            "validation_correlation": correlation,
        }
        training_curve.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint_path = output_dir / "fitted_q.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "width": args.width,
            "blocks": args.blocks,
        },
        checkpoint_path,
    )
    manifest = {
        "schema_version": 1,
        "kind": "monte_carlo_fitted_action_value",
        "seed": args.seed,
        "privacy": {
            "observation": "121 public and own-hand features only",
            "opponent_card_identities": False,
            "draw_pile_order": False,
        },
        "warm_parent": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
        "config": {
            "games": args.games,
            "temperatures": temperatures,
            "width": args.width,
            "blocks": args.blocks,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gamma": args.gamma,
            "validation_fraction": args.validation_fraction,
        },
        "dataset": {
            "games": episode_count,
            "decisions": len(observations),
            "train_decisions": len(train_indices),
            "validation_decisions": len(validation_indices),
            "collection": collection,
        },
        "training_curve": training_curve,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }
    (output_dir / "fitted_q_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["checkpoint"], sort_keys=True))


if __name__ == "__main__":
    main()
