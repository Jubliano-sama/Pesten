#!/usr/bin/env python3
"""Continue GRU-to-MLP distillation with an exact-action-focused objective."""

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

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = Path(__file__).resolve().parent
for location in (REPO_ROOT, CAMPAIGN_ROOT):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from ai_observation import ACTION_COUNT
from distill_gru_to_mlp import atomic_json, atomic_torch, evaluate
from ppo2 import StudentNet


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=130_000_999)
    parser.add_argument("--torch-threads", type=int, default=12)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_data = torch.load(
        args.data_dir / "train_data.pt", map_location="cpu", weights_only=True
    )
    validation_data = torch.load(
        args.data_dir / "validation_data.pt", map_location="cpu", weights_only=True
    )
    model = StudentNet()
    model.load_state_dict(
        torch.load(args.resume, map_location="cpu", weights_only=True)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1.0e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.minimum_learning_rate,
    )
    generator = torch.Generator().manual_seed(args.seed)
    samples = int(train_data["observations"].shape[0])
    initial_train = evaluate(model, train_data, batch_size=4096)
    initial_validation = evaluate(model, validation_data, batch_size=4096)
    best = {
        "epoch": 0,
        "action_agreement": initial_validation["action_agreement"],
        "forward_kl": initial_validation["forward_kl"],
    }
    atomic_torch(model.state_dict(), args.output_dir / "best_mlp.pth")
    history = []
    stale_epochs = 0
    print(
        json.dumps(
            {
                "event": "start",
                "resume_sha256": sha256(args.resume),
                "initial_train": initial_train,
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
            margin = F.relu(0.25 - chosen + competitor).mean()
            loss = 0.25 * kl + hard + 0.50 * margin
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
        validation = evaluate(model, validation_data, batch_size=4096)
        improved = (
            validation["action_agreement"] > best["action_agreement"]
            or (
                validation["action_agreement"] == best["action_agreement"]
                and validation["forward_kl"] < best["forward_kl"]
            )
        )
        if improved:
            best = {
                "epoch": epoch,
                "action_agreement": validation["action_agreement"],
                "forward_kl": validation["forward_kl"],
            }
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
            args.output_dir / "best_mlp.pth",
            map_location="cpu",
            weights_only=True,
        )
    )
    final_train = evaluate(model, train_data, batch_size=4096)
    final_validation = evaluate(model, validation_data, batch_size=4096)
    result = {
        "schema_version": 1,
        "experiment": "gru_to_mlp_exact_action_continuation",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "resume": str(args.resume.resolve()),
        "resume_sha256": sha256(args.resume),
        "best_checkpoint": str((args.output_dir / "best_mlp.pth").resolve()),
        "best_checkpoint_sha256": sha256(args.output_dir / "best_mlp.pth"),
        "epochs_completed": len(history),
        "best": best,
        "initial_train": initial_train,
        "initial_validation": initial_validation,
        "best_train": final_train,
        "best_validation": final_validation,
        "perfect_on_train": final_train["action_agreement"] == 1.0,
        "perfect_on_held_out": final_validation["action_agreement"] == 1.0,
        "objective": "0.25 forward KL + hard CE + 0.50 top-action margin",
    }
    atomic_json(result, args.output_dir / "result.json")
    print(json.dumps({"event": "complete", "result": result}), flush=True)


if __name__ == "__main__":
    main()
