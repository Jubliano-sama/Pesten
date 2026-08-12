#!/usr/bin/env python3
"""Warm-start a larger Puffer MinGRU by distilling a proven smaller teacher."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_atomic(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunks-per-worker", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--minibatch-trajectories", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--strategic-pass-weight", type=float, default=2.0)
    parser.add_argument("--value-coefficient", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026081201)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("large-Puffer distillation requested unavailable CUDA")
    if args.hidden_size < 1 or args.num_layers < 1:
        raise ValueError("student dimensions must be positive")
    teacher_path = args.teacher.resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    from live_puffer_segment import _load_policy_payload
    from parallel_puffer_rollout import collect_league_batch_parallel
    from puffer_distillation import distill_puffer_policy
    from temporal_observation import PASS_ACTION
    from temporal_puffer_v4 import NativePufferTemporalPolicy

    payload, state, teacher_hidden, teacher_layers, teacher_heads = (
        _load_policy_payload(teacher_path)
    )
    if (
        args.hidden_size <= teacher_hidden
        and args.num_layers <= teacher_layers
    ):
        raise ValueError("student must be wider or deeper than its teacher")
    teacher_rollout = NativePufferTemporalPolicy(
        hidden_size=teacher_hidden,
        num_layers=teacher_layers,
        mode_conditioned_heads=teacher_heads,
    )
    teacher_rollout.load_state_dict(state)
    teacher_rollout.eval()
    for parameter in teacher_rollout.parameters():
        parameter.requires_grad_(False)

    recipe = payload.get("recipe", {})
    schedule = tuple(recipe.get("schedule", ("current_selfplay",)))
    mixture = tuple(
        (name, schedule.count(name)) for name in dict.fromkeys(schedule)
    )
    player_counts = tuple(recipe.get("player_count_schedule", (2,)))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    started = time.perf_counter()
    rollout_started = time.perf_counter()
    episodes, contribution = collect_league_batch_parallel(
        teacher_rollout,
        games=args.games,
        seed=args.seed,
        schedule_offset=0,
        mixture=mixture,
        player_counts=player_counts,
        workers=args.workers,
        chunks_per_worker=args.chunks_per_worker,
    )
    rollout_seconds = time.perf_counter() - rollout_started
    decisions = sum(len(trajectory) for trajectory, _ in episodes)
    strategic_steps = [
        step
        for trajectory, _ in episodes
        for step in trajectory
        if step.decision.legal_actions[PASS_ACTION]
        and any(step.decision.legal_actions[:PASS_ACTION])
    ]

    teacher = copy.deepcopy(teacher_rollout).to(args.device)
    student = NativePufferTemporalPolicy(
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        mode_conditioned_heads=True,
    ).to(args.device)
    optimizer = torch.optim.Adam(student.parameters(), lr=args.learning_rate)
    if args.device == "cuda":
        torch.cuda.synchronize()
    distill_started = time.perf_counter()
    curve = distill_puffer_policy(
        student,
        teacher,
        episodes,
        optimizer,
        epochs=args.epochs,
        minibatch_trajectories=args.minibatch_trajectories,
        seed=args.seed + 1_000_000,
        temperature=args.temperature,
        value_coefficient=args.value_coefficient,
        strategic_pass_weight=args.strategic_pass_weight,
    )
    if args.device == "cuda":
        torch.cuda.synchronize()
    distill_seconds = time.perf_counter() - distill_started
    cpu_state = {
        name: value.detach().cpu().clone()
        for name, value in student.state_dict().items()
    }
    checkpoint = {
        "format": "pesten-distilled-native-puffer-v1",
        "epoch": int(payload.get("epoch", 0)),
        "model": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "mode_conditioned_heads": True,
        },
        "parent_checkpoint": str(teacher_path),
        "teacher_sha256": _sha256(teacher_path),
        "state_dict": cpu_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "recipe": {
            **recipe,
            "distillation": {
                "teacher_hidden_size": teacher_hidden,
                "teacher_num_layers": teacher_layers,
                "games": args.games,
                "epochs": args.epochs,
                "temperature": args.temperature,
                "strategic_pass_weight": args.strategic_pass_weight,
                "value_coefficient": args.value_coefficient,
                "seed": args.seed,
            },
            "strategic_pass": {
                "action": PASS_ACTION,
                "legal_alongside_playable_cards": True,
                "teacher_opportunities": len(strategic_steps),
                "teacher_actions": sum(
                    step.action == PASS_ACTION for step in strategic_steps
                ),
            },
        },
    }
    _save_atomic(checkpoint, args.output)
    result = {
        "kind": "large_puffer_teacher_distillation",
        "teacher": {
            "path": str(teacher_path),
            "sha256": checkpoint["teacher_sha256"],
            "hidden_size": teacher_hidden,
            "num_layers": teacher_layers,
        },
        "student": checkpoint["model"],
        "output": str(args.output),
        "games": args.games,
        "trajectories": len(episodes),
        "decisions": decisions,
        "rollout_seconds": rollout_seconds,
        "games_per_second": args.games / rollout_seconds,
        "distillation_seconds": distill_seconds,
        "elapsed_seconds": time.perf_counter() - started,
        "contribution": contribution,
        "curve": curve,
        "runtime": {
            "device": str(next(student.parameters()).device),
            "cuda": torch.cuda.is_available(),
            "torch": torch.__version__,
        },
    }
    result_path = args.output.with_suffix(".distillation.json")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
