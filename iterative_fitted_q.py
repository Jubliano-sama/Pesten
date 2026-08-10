#!/usr/bin/env python3
"""One approximate-policy-iteration step for the fitted-Q Pesten actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    POLICY_RANDOM,
    POLICY_RL,
    collect_batched_rollout,
)
from fitted_q_policy import ActionValueNet, FittedQResidualPolicy
from ppo2 import StudentNet
from random_fitted_q import file_sha256, parse_temperatures, return_targets


def collect_random_games(
    actor: torch.nn.Module,
    *,
    games: int,
    temperature: float,
    rng: np.random.Generator,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], dict]:
    observations = []
    actions = []
    rewards = []
    lengths = []
    metrics = {
        "games": 0,
        "environment_turns": 0,
        "policy_decisions": 0,
        "inference_calls": 0,
    }
    for start in range(0, games, batch_size):
        count = min(batch_size, games - start)
        evaluated_seats = rng.integers(0, 2, size=count, dtype=np.int8)
        policy_kinds = np.full((count, 2), POLICY_RANDOM, dtype=np.int8)
        policy_kinds[np.arange(count), evaluated_seats] = POLICY_RL
        actor_keys = np.full((count, 2), None, dtype=object)
        actor_keys[np.arange(count), evaluated_seats] = "behavior"
        seeds = rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=count,
            dtype=np.uint32,
        )
        engine = BatchedPesten(
            policy_kinds,
            actor_keys,
            seeds,
            max_turns=500,
            smart_agents_have_game_reference=False,
        )
        result = collect_batched_rollout(
            engine,
            {
                "behavior": ActorPolicy(
                    actor,
                    device,
                    temperature=temperature,
                    collect_data=True,
                    greedy=False,
                    autocast=False,
                )
            },
            output_device=torch.device("cpu"),
            direct_action_sampling=True,
            reuse_rollout_workspaces=True,
            drain_non_neural=True,
        )
        observations.append(result.observations)
        actions.append(result.actions)
        rewards.append(result.rewards)
        lengths.extend(result.episode_lengths)
        metrics["games"] += result.completed_games
        metrics["environment_turns"] += result.environment_turns
        metrics["policy_decisions"] += result.policy_decisions
        metrics["inference_calls"] += result.inference_calls
    return (
        torch.cat(observations),
        torch.cat(actions),
        torch.cat(rewards),
        lengths,
        metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--behavior-q", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--behavior-scale", type=float, default=64.0)
    parser.add_argument("--behavior-gate", type=float, default=4.0)
    parser.add_argument("--games", type=int, default=131_072)
    parser.add_argument("--temperatures", default="0.8,1.0,1.25")
    parser.add_argument("--rollout-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
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
    if min(
        args.games,
        args.rollout_batch_size,
        args.epochs,
        args.batch_size,
        args.torch_threads,
    ) < 1:
        raise ValueError("positive collection and training sizes required")
    if args.behavior_scale < 0 or args.behavior_gate < 0:
        raise ValueError("behavior scale and gate cannot be negative")
    if not 0 < args.gamma <= 1:
        raise ValueError("gamma must be in (0, 1]")
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("validation-fraction must be in (0, 0.5)")

    base_path = args.base.resolve()
    behavior_q_path = args.behavior_q.resolve()
    if not base_path.is_file() or not behavior_q_path.is_file():
        raise FileNotFoundError("base actor or behavior Q checkpoint is missing")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(args.torch_threads)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    parent = StudentNet().to(device)
    parent.load_state_dict(
        torch.load(base_path, map_location=device, weights_only=True)
    )
    parent.eval().requires_grad_(False)
    saved = torch.load(
        behavior_q_path,
        map_location=device,
        weights_only=True,
    )
    behavior_q = ActionValueNet(
        width=int(saved["width"]),
        blocks=int(saved["blocks"]),
    ).to(device)
    behavior_q.load_state_dict(saved["model"])
    behavior_q.eval().requires_grad_(False)
    behavior = FittedQResidualPolicy(
        parent,
        behavior_q,
        scale=args.behavior_scale,
        normalize=False,
        confidence_gate=args.behavior_gate,
    ).to(device).eval()

    started = time.perf_counter()
    rng = np.random.default_rng(args.seed)
    observations_parts = []
    actions_parts = []
    targets_parts = []
    episode_parts = []
    collection = []
    episode_offset = 0
    game_counts = [args.games // len(temperatures)] * len(temperatures)
    game_counts[-1] += args.games - sum(game_counts)
    for temperature, games in zip(temperatures, game_counts):
        observations, actions, rewards, lengths, rollout = (
            collect_random_games(
                behavior,
                games=games,
                temperature=temperature,
                rng=rng,
                batch_size=args.rollout_batch_size,
                device=device,
            )
        )
        targets, episode_ids, outcomes = return_targets(
            rewards,
            lengths,
            gamma=args.gamma,
        )
        observations_parts.append(observations)
        actions_parts.append(actions)
        targets_parts.append(targets)
        episode_parts.append(episode_ids + episode_offset)
        episode_offset += len(lengths)
        record = {
            "temperature": temperature,
            "decisions": len(observations),
            "rollout": rollout,
            **outcomes,
        }
        collection.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

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

    model = ActionValueNet(
        width=int(saved["width"]),
        blocks=int(saved["blocks"]),
    ).to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        fused=True,
    )
    curve = []
    order_generator = torch.Generator().manual_seed(args.seed + 2)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = train_indices[
            torch.randperm(len(train_indices), generator=order_generator)
        ]
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = order[start : start + args.batch_size]
            predicted = model(
                observations[indices].to(device, non_blocking=True)
            ).gather(
                1,
                actions[indices].to(device, non_blocking=True).unsqueeze(-1),
            ).squeeze(-1)
            expected = targets[indices].to(device, non_blocking=True)
            weights = sample_weights[indices].to(device, non_blocking=True)
            loss = (
                F.smooth_l1_loss(
                    predicted,
                    expected,
                    reduction="none",
                    beta=0.5,
                )
                * weights
            ).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))

        model.eval()
        predicted_parts = []
        expected_parts = []
        with torch.inference_mode():
            for start in range(0, len(validation_indices), args.batch_size):
                indices = validation_indices[start : start + args.batch_size]
                predicted_parts.append(
                    model(
                        observations[indices].to(device, non_blocking=True)
                    ).gather(
                        1,
                        actions[indices]
                        .to(device, non_blocking=True)
                        .unsqueeze(-1),
                    )
                    .squeeze(-1)
                    .cpu()
                )
                expected_parts.append(targets[indices])
        predicted = torch.cat(predicted_parts)
        expected = torch.cat(expected_parts)
        record = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation_loss": float(F.smooth_l1_loss(predicted, expected)),
            "validation_correlation": float(
                torch.corrcoef(torch.stack((predicted, expected)))[0, 1]
            ),
        }
        curve.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint_path = output_dir / "fitted_q.pt"
    torch.save(
        {
            "schema_version": 1,
            "model": model.state_dict(),
            "width": int(saved["width"]),
            "blocks": int(saved["blocks"]),
        },
        checkpoint_path,
    )
    manifest = {
        "schema_version": 1,
        "kind": "iterative_monte_carlo_fitted_action_value",
        "seed": args.seed,
        "privacy": {
            "observation": "121 public and own-hand features only",
            "opponent_card_identities": False,
            "draw_pile_order": False,
        },
        "base": {"path": str(base_path), "sha256": file_sha256(base_path)},
        "behavior_q": {
            "path": str(behavior_q_path),
            "sha256": file_sha256(behavior_q_path),
        },
        "behavior": {
            "scale": args.behavior_scale,
            "confidence_gate": args.behavior_gate,
        },
        "config": {
            "games": args.games,
            "temperatures": temperatures,
            "rollout_batch_size": args.rollout_batch_size,
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
        "training_curve": curve,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
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

