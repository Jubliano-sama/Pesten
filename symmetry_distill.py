#!/usr/bin/env python3
"""Distill the frozen PPO's 24-suit orbit into one PPO-sized forward pass.

Rollouts and labels use only the established 121-feature policy observation:
own-card counts and public game counters.  Deal seeds are consecutive integers
in a development-only namespace; no hash-derived or sealed seeds are used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ai_observation import ACTION_COUNT, OBSERVATION_SIZE
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    POLICY_RANDOM,
    POLICY_RL,
    POLICY_SMART,
    POLICY_SMARTER,
    collect_batched_rollout,
)
from batched_suit_symmetry import BatchedSuitAveragedActor
from hybrid_agent import load_frozen_ppo_actor
from ppo2 import StudentNet, build_action_masks


DEVELOPMENT_SEED_LIMIT = 90_000_000
DEFAULT_SEED = 87_200_001
DEFAULT_TEACHER = Path("ppo_actor_100run3.pth")


@dataclass(frozen=True)
class SymmetryDistillConfig:
    seed: int = DEFAULT_SEED
    games: int = 64
    epochs: int = 3
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    validation_fraction: float = 0.20
    temperature: float = 1.0
    kl_weight: float = 1.0
    centered_mse_weight: float = 0.05
    hard_action_weight: float = 0.10
    max_turns: int = 500
    teacher_chunk_size: int = 128


def validate_development_seed_range(seed: int, count: int) -> None:
    """Require a non-hashed consecutive range entirely below 90 million."""
    if seed < 0 or count < 1:
        raise ValueError("seed must be non-negative and count positive")
    if seed + count - 1 >= DEVELOPMENT_SEED_LIMIT:
        raise ValueError(
            "development seed range must remain below "
            f"{DEVELOPMENT_SEED_LIMIT:,}"
        )


def _rollout_layout(games: int) -> tuple[np.ndarray, np.ndarray]:
    kinds = np.full((games, 2), POLICY_SMART, dtype=np.int8)
    keys = np.full((games, 2), None, dtype=object)
    opponents = (POLICY_RANDOM, POLICY_SMART, POLICY_SMARTER, POLICY_RL)
    for game_index in range(games):
        learner_seat = game_index % 2
        opponent_seat = 1 - learner_seat
        kinds[game_index, learner_seat] = POLICY_RL
        keys[game_index, learner_seat] = "collector"
        opponent = opponents[game_index % len(opponents)]
        kinds[game_index, opponent_seat] = opponent
        if opponent == POLICY_RL:
            keys[game_index, opponent_seat] = "collector"
    return kinds, keys


def collect_public_observations(
    actor: nn.Module,
    *,
    games: int,
    seed: int,
    device: torch.device,
    max_turns: int = 500,
) -> tuple[torch.Tensor, dict]:
    """Collect reachable observations using consecutive development seeds."""
    validate_development_seed_range(seed, games)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    kinds, keys = _rollout_layout(games)
    deal_seeds = np.arange(seed, seed + games, dtype=np.uint32)
    engine = BatchedPesten(
        kinds,
        keys,
        deal_seeds,
        max_turns=max_turns,
    )
    rollout = collect_batched_rollout(
        engine,
        {
            "collector": ActorPolicy(
                actor=actor,
                device=device,
                temperature=0.85,
                collect_data=True,
                greedy=False,
            )
        },
        output_device=torch.device("cpu"),
    )
    if (
        rollout.observations.ndim != 2
        or rollout.observations.shape[1] != OBSERVATION_SIZE
    ):
        raise RuntimeError("rollout crossed an unexpected policy boundary")
    return rollout.observations, {
        "games": games,
        "seed_strategy": "consecutive integers; no hashing",
        "minimum_deal_seed": int(deal_seeds.min()),
        "maximum_deal_seed": int(deal_seeds.max()),
        "all_deal_seeds_below_90m": bool(
            np.all(deal_seeds < DEVELOPMENT_SEED_LIMIT)
        ),
        "observations": len(rollout.observations),
        "environment_turns": rollout.environment_turns,
        "completed_games": rollout.completed_games,
        "policy_decisions": rollout.policy_decisions,
        "inference_calls": rollout.inference_calls,
    }


def compute_orbit_teacher_logits(
    observations: torch.Tensor,
    champion: nn.Module,
    *,
    device: torch.device,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Compute canonical logits from all 24 ordinary-suit relabelings."""
    teacher = BatchedSuitAveragedActor(
        champion,
        alpha=1.0,
        aggregation="logits",
        player_orbit_mode="none",
        base_chunk_size=chunk_size,
    ).to(device).eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(observations), chunk_size):
            parts.append(
                teacher(
                    observations[start : start + chunk_size].to(device)
                ).cpu()
            )
    return torch.cat(parts)


def _augment_by_suit(
    observations: torch.Tensor,
    targets: torch.Tensor,
    permutation_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Relabel inputs and equivariant teacher logits for per-row permutations."""
    helper = BatchedSuitAveragedActor(
        nn.Identity(),
        alpha=0.0,
    ).to(observations.device)
    observation_indices = helper.observation_indices[permutation_rows]
    action_destinations = helper.action_indices[permutation_rows]
    transformed_observations = torch.gather(
        observations, dim=1, index=observation_indices
    )
    transformed_targets = torch.empty_like(targets)
    transformed_targets.scatter_(1, action_destinations, targets)
    return transformed_observations, transformed_targets


def _masked_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observations: torch.Tensor,
    *,
    temperature: float,
) -> dict[str, float]:
    legal = build_action_masks(observations)
    floor = torch.finfo(logits.dtype).min
    student_log_probs = F.log_softmax(
        logits.masked_fill(~legal, floor) / temperature, dim=-1
    )
    teacher_log_probs = F.log_softmax(
        targets.masked_fill(~legal, floor) / temperature, dim=-1
    )
    teacher_probs = teacher_log_probs.exp()
    kl = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    student_actions = logits.masked_fill(~legal, floor).argmax(dim=-1)
    teacher_actions = targets.masked_fill(~legal, floor).argmax(dim=-1)
    legal_count = legal.sum(dim=-1, keepdim=True).clamp_min(1)
    student_centered = logits - (
        logits.masked_fill(~legal, 0.0).sum(dim=-1, keepdim=True)
        / legal_count
    )
    teacher_centered = targets - (
        targets.masked_fill(~legal, 0.0).sum(dim=-1, keepdim=True)
        / legal_count
    )
    mse = (
        (student_centered - teacher_centered).square()
        .masked_fill(~legal, 0.0)
        .sum()
        / legal.sum().clamp_min(1)
    )
    return {
        "kl": float(kl.mean().item()),
        "legal_action_agreement": float(
            (student_actions == teacher_actions).float().mean().item()
        ),
        "centered_legal_logit_rmse": float(mse.sqrt().item()),
    }


def _training_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    observations: torch.Tensor,
    config: SymmetryDistillConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    legal = build_action_masks(observations)
    floor = torch.finfo(logits.dtype).min
    student_log_probs = F.log_softmax(
        logits.masked_fill(~legal, floor) / config.temperature, dim=-1
    )
    teacher_log_probs = F.log_softmax(
        targets.masked_fill(~legal, floor) / config.temperature, dim=-1
    )
    teacher_probs = teacher_log_probs.exp()
    kl = (
        teacher_probs * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1).mean()
    legal_count = legal.sum(dim=-1, keepdim=True).clamp_min(1)
    student_centered = logits - (
        logits.masked_fill(~legal, 0.0).sum(dim=-1, keepdim=True)
        / legal_count
    )
    teacher_centered = targets - (
        targets.masked_fill(~legal, 0.0).sum(dim=-1, keepdim=True)
        / legal_count
    )
    centered_mse = (
        (student_centered - teacher_centered).square()
        .masked_fill(~legal, 0.0)
        .sum()
        / legal.sum().clamp_min(1)
    )
    hard_targets = targets.masked_fill(~legal, floor).argmax(dim=-1)
    hard_ce = F.cross_entropy(
        logits.masked_fill(~legal, floor), hard_targets
    )
    loss = (
        config.kl_weight * kl
        + config.centered_mse_weight * centered_mse
        + config.hard_action_weight * hard_ce
    )
    return loss, {
        "kl": float(kl.detach().item()),
        "centered_mse": float(centered_mse.detach().item()),
        "hard_ce": float(hard_ce.detach().item()),
    }


def distill_student(
    observations: torch.Tensor,
    targets: torch.Tensor,
    champion: nn.Module,
    *,
    config: SymmetryDistillConfig,
    device: torch.device,
) -> tuple[StudentNet, dict]:
    """Fine-tune a champion-initialized PPO architecture on orbit targets."""
    if len(observations) != len(targets) or len(observations) < 10:
        raise ValueError("distillation needs at least ten aligned samples")
    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(len(observations), generator=generator)
    validation_count = max(1, int(len(order) * config.validation_fraction))
    validation_indices = order[:validation_count]
    training_indices = order[validation_count:]
    train_observations = observations[training_indices]
    train_targets = targets[training_indices]
    validation_observations = observations[validation_indices].to(device)
    validation_targets = targets[validation_indices].to(device)

    student = StudentNet().to(device)
    student.load_state_dict(champion.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    champion.eval()
    with torch.inference_mode():
        baseline_metrics = _masked_metrics(
            champion(validation_observations),
            validation_targets,
            validation_observations,
            temperature=config.temperature,
        )

    history = []
    best_metrics = baseline_metrics
    best_epoch = 0
    best_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in student.state_dict().items()
    }
    for epoch in range(config.epochs):
        student.train()
        epoch_generator = torch.Generator().manual_seed(
            config.seed + epoch + 1
        )
        epoch_order = torch.randperm(
            len(train_observations), generator=epoch_generator
        )
        losses = []
        for start in range(0, len(epoch_order), config.batch_size):
            indices = epoch_order[start : start + config.batch_size]
            batch_observations = train_observations[indices].to(device)
            batch_targets = train_targets[indices].to(device)
            permutation_rows = torch.randint(
                0,
                24,
                (len(indices),),
                generator=epoch_generator,
            ).to(device)
            batch_observations, batch_targets = _augment_by_suit(
                batch_observations,
                batch_targets,
                permutation_rows,
            )
            logits = student(batch_observations)
            loss, components = _training_loss(
                logits, batch_targets, batch_observations, config
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().item()))
        student.eval()
        with torch.inference_mode():
            metrics = _masked_metrics(
                student(validation_observations),
                validation_targets,
                validation_observations,
                temperature=config.temperature,
            )
        epoch_record = {
            "epoch": epoch + 1,
            "training_loss": float(np.mean(losses)),
            "last_batch_components": components,
            "validation": metrics,
        }
        history.append(epoch_record)
        if (
            metrics["legal_action_agreement"]
            > best_metrics["legal_action_agreement"]
            or (
                metrics["legal_action_agreement"]
                == best_metrics["legal_action_agreement"]
                and metrics["kl"] < best_metrics["kl"]
            )
        ):
            best_metrics = metrics
            best_epoch = epoch + 1
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in student.state_dict().items()
            }

    student.load_state_dict(best_state, strict=True)
    student.eval()
    return student, {
        "training_samples": len(training_indices),
        "validation_samples": len(validation_indices),
        "baseline_champion": baseline_metrics,
        "selected_student": best_metrics,
        "selected_epoch": best_epoch,
        "selection_rule": (
            "highest held-out legal greedy-action agreement, then lower KL"
        ),
        "history": history,
        "augmentation": (
            "one uniformly sampled ordinary-suit permutation per sample "
            "per epoch; targets relabeled exactly"
        ),
    }


def train_pilot(
    *,
    teacher_path: Path,
    output_path: Path,
    report_path: Path,
    config: SymmetryDistillConfig,
    device: torch.device,
) -> dict:
    started = time.perf_counter()
    champion, _ = load_frozen_ppo_actor(teacher_path, device)
    observations, rollout = collect_public_observations(
        champion,
        games=config.games,
        seed=config.seed,
        device=device,
        max_turns=config.max_turns,
    )
    targets = compute_orbit_teacher_logits(
        observations,
        champion,
        device=device,
        chunk_size=config.teacher_chunk_size,
    )
    student, metrics = distill_student(
        observations,
        targets,
        champion,
        config=config,
        device=device,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), output_path)
    report = {
        "schema_version": 1,
        "method": "full 24-suit logit-orbit distillation",
        "teacher": str(teacher_path),
        "student_checkpoint": str(output_path),
        "student_architecture": "PPO StudentNet 121->55 initialized from teacher",
        "privacy_boundary": (
            "existing 121-feature own-hand and public-counter observation only"
        ),
        "seed_policy": (
            "consecutive development seeds below 90M; no hash derivation"
        ),
        "device": str(device),
        "config": asdict(config),
        "rollout": rollout,
        "metrics": metrics,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    config = SymmetryDistillConfig(
        seed=args.seed,
        games=args.games,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    report = train_pilot(
        teacher_path=args.teacher,
        output_path=args.output,
        report_path=args.report,
        config=config,
        device=torch.device(args.device),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
