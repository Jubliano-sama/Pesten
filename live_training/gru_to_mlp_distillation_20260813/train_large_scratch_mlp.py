#!/usr/bin/env python3
"""Train a fresh, oversized factorized MLP from a recurrent GRU teacher."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = Path(__file__).resolve().parent
for location in (REPO_ROOT, CAMPAIGN_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from ai_observation import ACTION_COUNT, OBSERVATION_SIZE
from distill_gru_to_mlp import (
    PLAYER_COUNTS,
    atomic_json,
    atomic_torch,
    collect_dataset,
    evaluate,
)
from ppo2 import StudentNet
from supervisedAgent import DenseSkipNet
from temporal_puffer_v4 import (
    FactorizedActionDecoder,
    load_temporal_policy_checkpoint,
)
from vector_puffer_rollout import VectorPufferRolloutPool


EXPANDED_OPPONENTS = (
    "current_selfplay",
    "current_selfplay",
    "historical_temporal",
    "ppo_champion",
    "canonical_random",
    "smart_rule",
    "smarter_rule",
)
TRAIN_KEYS = ("observations", "legal_actions", "teacher_logits")


class LargeFactorizedMLP(nn.Module):
    """Memoryless DenseSkip MLP with the GRU teacher's exact 23-score head."""

    def __init__(self, *, width: int = 384, layers: int = 8) -> None:
        super().__init__()
        self.width = int(width)
        self.layers = int(layers)
        self.model = DenseSkipNet(
            input_dim=OBSERVATION_SIZE,
            hidden_dims=[self.width] * self.layers,
            output_dim=23,
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        scores = self.model(observations)
        return FactorizedActionDecoder.expand_scores(scores)

    def config(self) -> dict:
        return {
            "architecture": "DenseSkipNet factorized MLP",
            "input_features": OBSERVATION_SIZE,
            "hidden_width": self.width,
            "hidden_layers": self.layers,
            "learned_action_scores": 23,
            "expanded_actions": ACTION_COUNT,
            "recurrent": False,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact(data: dict) -> dict:
    return {name: data[name] for name in TRAIN_KEYS}


def concatenate(first: dict, second: dict) -> dict:
    return {
        name: torch.cat((first[name], second[name]), dim=0)
        for name in TRAIN_KEYS
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--base-data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--historical", type=Path, action="append", default=[]
    )
    parser.add_argument("--additional-train-games", type=int, default=12_288)
    parser.add_argument("--additional-validation-games", type=int, default=3_072)
    parser.add_argument("--train-seed", type=int, default=132_000_000)
    parser.add_argument("--validation-seed", type=int, default=133_000_000)
    parser.add_argument("--collection-batch-games", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=2.0e-5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--seed", type=int, default=134_000_000)
    parser.add_argument("--torch-threads", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    teacher, teacher_payload = load_temporal_policy_checkpoint(
        args.teacher, device="cpu"
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    current_mlp_parameters = sum(
        parameter.numel() for parameter in StudentNet().parameters()
    )
    model = LargeFactorizedMLP(width=args.width, layers=args.layers)
    student_parameters = sum(parameter.numel() for parameter in model.parameters())
    minimum_parameters = math.ceil(teacher_parameters * 1.2)
    if student_parameters < minimum_parameters:
        raise ValueError(
            f"student has {student_parameters} parameters; needs {minimum_parameters}"
        )
    if student_parameters <= current_mlp_parameters:
        raise ValueError(
            "requested large student must also exceed the current MLP parameter count"
        )

    print(
        json.dumps(
            {
                "event": "architecture",
                "teacher_parameters": teacher_parameters,
                "minimum_120_percent": minimum_parameters,
                "current_mlp_parameters": current_mlp_parameters,
                "large_mlp_parameters": student_parameters,
                "large_over_teacher_ratio": student_parameters / teacher_parameters,
                "large_over_current_mlp_ratio": (
                    student_parameters / current_mlp_parameters
                ),
                "model": model.config(),
                "randomly_initialized": True,
            }
        ),
        flush=True,
    )

    base_train = compact(
        torch.load(
            args.base_data_dir / "train_data.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    base_validation = compact(
        torch.load(
            args.base_data_dir / "validation_data.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    with VectorPufferRolloutPool(
        teacher,
        workers=args.workers,
        champion_path=args.champion,
        historical_paths=args.historical,
    ) as pool:
        extra_train, extra_train_meta = collect_dataset(
            pool,
            teacher,
            games=args.additional_train_games,
            seed=args.train_seed,
            batch_games=args.collection_batch_games,
            schedule=EXPANDED_OPPONENTS,
            player_counts=PLAYER_COUNTS,
        )
        extra_validation, extra_validation_meta = collect_dataset(
            pool,
            teacher,
            games=args.additional_validation_games,
            seed=args.validation_seed,
            batch_games=args.collection_batch_games,
            schedule=EXPANDED_OPPONENTS,
            player_counts=PLAYER_COUNTS,
        )
    train_data = concatenate(base_train, compact(extra_train))
    validation_data = concatenate(base_validation, compact(extra_validation))
    del extra_train, extra_validation, base_train, base_validation
    atomic_torch(train_data, args.output_dir / "train_data.pt")
    atomic_torch(validation_data, args.output_dir / "validation_data.pt")

    initial_validation = evaluate(model, validation_data, batch_size=args.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.minimum_learning_rate,
    )
    generator = torch.Generator().manual_seed(args.seed + 1)
    samples = int(train_data["observations"].shape[0])
    history = []
    best = {"epoch": 0, **initial_validation}
    atomic_torch(model.state_dict(), args.output_dir / "best_mlp.pth")
    stale_epochs = 0
    print(
        json.dumps(
            {
                "event": "data_ready",
                "train_samples": samples,
                "validation_samples": int(validation_data["observations"].shape[0]),
                "initial_validation": initial_validation,
            }
        ),
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = torch.randperm(samples, generator=generator)
        totals = {"loss": 0.0, "kl": 0.0, "hard": 0.0, "margin": 0.0}
        started = time.perf_counter()
        for start in range(0, samples, args.batch_size):
            indices = permutation[start : start + args.batch_size]
            observations = train_data["observations"][indices]
            legal = train_data["legal_actions"][indices]
            teacher_logits = train_data["teacher_logits"][indices]
            teacher_actions = teacher_logits.argmax(dim=-1)
            logits = model(observations).masked_fill(~legal, -1.0e9)
            log_prob = torch.log_softmax(logits, dim=-1)
            teacher_prob = torch.softmax(teacher_logits, dim=-1)
            kl = F.kl_div(log_prob, teacher_prob, reduction="batchmean")
            hard = F.cross_entropy(logits, teacher_actions)
            chosen = logits.gather(1, teacher_actions[:, None]).squeeze(1)
            competitor = logits.masked_fill(
                F.one_hot(teacher_actions, ACTION_COUNT).bool(), -1.0e9
            ).max(dim=-1).values
            margin = F.relu(0.20 - chosen + competitor).mean()
            # Preserve dense distributional knowledge while prioritizing exact
            # behavior, which is the user's success criterion here.
            loss = 0.50 * kl + hard + 0.35 * margin
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = int(indices.numel())
            totals["loss"] += float(loss.detach()) * count
            totals["kl"] += float(kl.detach()) * count
            totals["hard"] += float(hard.detach()) * count
            totals["margin"] += float(margin.detach()) * count
        scheduler.step()
        validation = evaluate(model, validation_data, batch_size=args.batch_size)
        improved = (
            validation["action_agreement"] > best["action_agreement"]
            or (
                validation["action_agreement"] == best["action_agreement"]
                and validation["forward_kl"] < best["forward_kl"]
            )
        )
        if improved:
            best = {"epoch": epoch, **validation}
            stale_epochs = 0
            atomic_torch(model.state_dict(), args.output_dir / "best_mlp.pth")
        else:
            stale_epochs += 1
        record = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": totals["loss"] / samples,
            "train_forward_kl": totals["kl"] / samples,
            "train_hard_cross_entropy": totals["hard"] / samples,
            "train_margin_loss": totals["margin"] / samples,
            "validation": validation,
            "epoch_seconds": time.perf_counter() - started,
            "stale_epochs": stale_epochs,
        }
        history.append(record)
        atomic_torch(model.state_dict(), args.output_dir / "latest_mlp.pth")
        atomic_json(history, args.output_dir / "training_history.json")
        print(json.dumps({"event": "epoch", **record}), flush=True)
        if stale_epochs >= args.patience:
            break

    model.load_state_dict(
        torch.load(
            args.output_dir / "best_mlp.pth", map_location="cpu", weights_only=True
        )
    )
    best_train = evaluate(model, train_data, batch_size=args.batch_size)
    best_validation = evaluate(model, validation_data, batch_size=args.batch_size)
    result = {
        "schema_version": 1,
        "experiment": "large_scratch_factorized_mlp_distillation",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "teacher": {
            "path": str(args.teacher.resolve()),
            "sha256": sha256(args.teacher),
            "epoch": teacher_payload.get("epoch"),
            "model": teacher_payload.get("model"),
            "parameters": teacher_parameters,
        },
        "student": {
            **model.config(),
            "parameters": student_parameters,
            "minimum_requested_parameters": minimum_parameters,
            "over_teacher_ratio": student_parameters / teacher_parameters,
            "over_current_mlp_ratio": student_parameters / current_mlp_parameters,
            "randomly_initialized": True,
            "best_checkpoint": str((args.output_dir / "best_mlp.pth").resolve()),
            "best_checkpoint_sha256": sha256(args.output_dir / "best_mlp.pth"),
        },
        "data": {
            "base_train_samples": 220_288,
            "base_validation_samples": 54_755,
            "additional_train_games": args.additional_train_games,
            "additional_validation_games": args.additional_validation_games,
            "additional_opponent_schedule": list(EXPANDED_OPPONENTS),
            "player_count_schedule": list(PLAYER_COUNTS),
            "historical_checkpoints": [str(path.resolve()) for path in args.historical],
            "extra_train": extra_train_meta,
            "extra_validation": extra_validation_meta,
            "total_train_samples": samples,
            "total_validation_samples": int(validation_data["observations"].shape[0]),
        },
        "training": {
            "epochs_completed": len(history),
            "batch_size": args.batch_size,
            "initial_learning_rate": args.learning_rate,
            "minimum_learning_rate": args.minimum_learning_rate,
            "objective": "0.50 forward KL + hard CE + 0.35 top-action margin",
        },
        "initial_validation": initial_validation,
        "best": best,
        "best_train": best_train,
        "best_validation": best_validation,
        "perfect_on_train": best_train["action_agreement"] == 1.0,
        "perfect_on_held_out": best_validation["action_agreement"] == 1.0,
    }
    atomic_json(result, args.output_dir / "result.json")
    print(json.dumps({"event": "complete", "result": result}), flush=True)


if __name__ == "__main__":
    main()
