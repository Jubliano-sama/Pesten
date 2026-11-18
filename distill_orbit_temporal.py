#!/usr/bin/env python3
"""Development-only public-history distillation of a native suit-orbit teacher.

Collects safe trajectories using the unwrapped parent policy, then teaches a
same-size student the canonical mean logits of independent suit relabelings.
Saved checkpoints remain ordinary native MinGRUs; this runner never promotes
a checkpoint. Fidelity metrics are not substitutes for held-out game scores.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from sota_distill_temporal import atomic_json_save, atomic_torch_save, cpu_state, file_sha256
from strongest_policy_search import NativeSuitOrbitCore, TEACHER


OPPONENT_SCHEDULE = (
    "current_selfplay", "ppo_champion", "canonical_random", "smart_rule", "smarter_rule",
)
PLAYER_COUNT_SCHEDULE = tuple(range(2, 9))


def build_orbit_teacher(student, *, orbit_size: int, alpha: float):
    """Freeze a separate copy so neither targets nor collection mutate it."""
    teacher = deepcopy(student).eval()
    teacher.puffer_policy = NativeSuitOrbitCore(
        teacher.puffer_policy, orbit_size=orbit_size, alpha=alpha
    )
    # The wrapper registers relabeling maps on CPU even when its core was
    # already on CUDA. Move the complete wrapper before indexing event maps.
    teacher.to(next(student.parameters()).device)
    teacher.requires_grad_(False)
    return teacher


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, default=TEACHER)
    parser.add_argument("--champion", type=Path, default=Path("ppo_actor_100run3.pth"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=350)
    parser.add_argument("--validation-games", type=int, default=70)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch-trajectories", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--value-coefficient", type=float, default=0.1)
    parser.add_argument("--strategic-pass-weight", type=float, default=2.0)
    parser.add_argument("--reverse-kl-coefficient", type=float, default=0.25)
    parser.add_argument("--hard-target-coefficient", type=float, default=0.1)
    parser.add_argument("--disagreement-weight", type=float, default=1.0)
    parser.add_argument("--orbit-size", type=int, choices=(1, 4, 24), default=24)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=202609263)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    for name in ("games", "validation_games", "epochs", "minibatch_trajectories",
                 "workers", "torch_threads", "max_turns"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 <= args.alpha <= 1:
        parser.error("--alpha must be in [0, 1]")
    if args.learning_rate <= 0 or args.temperature <= 0 or args.value_coefficient < 0:
        parser.error("learning rate/temperature must be positive and value coefficient nonnegative")
    if args.strategic_pass_weight < 1 or args.disagreement_weight < 1:
        parser.error("pass/disagreement weights must be at least 1")
    if args.reverse_kl_coefficient < 0 or args.hard_target_coefficient < 0:
        parser.error("KL/hard-target coefficients must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA distillation requested but unavailable")
    teacher_path, champion_path = args.teacher.resolve(), args.champion.resolve()
    for path in (teacher_path, champion_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    from puffer_distillation import distill_puffer_policy, evaluate_puffer_distillation
    from temporal_puffer_v4 import load_temporal_policy_checkpoint, policy_model_config
    from vector_puffer_rollout import VectorPufferRolloutPool

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    torch.set_float32_matmul_precision("highest")
    student, parent_payload = load_temporal_policy_checkpoint(teacher_path, device=args.device)
    model_config = policy_model_config(student)
    if model_config["architecture"] != "min_gru":
        raise ValueError("suit-orbit distillation requires a simple native MinGRU parent")
    teacher = build_orbit_teacher(student, orbit_size=args.orbit_size, alpha=args.alpha)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    recipe = {
        "method": "public_history_native_suit_orbit_distillation",
        "arguments": {name: str(value) if isinstance(value, Path) else value
                      for name, value in vars(args).items()},
        "parent_checkpoint": str(teacher_path),
        "parent_sha256": file_sha256(teacher_path),
        "champion_checkpoint": str(champion_path),
        "champion_sha256": file_sha256(champion_path),
        "opponent_schedule": list(OPPONENT_SCHEDULE),
        "player_count_schedule": list(PLAYER_COUNT_SCHEDULE),
        "train_seed_start": args.seed,
        "validation_seed_start": args.seed + args.games,
        "rollout_precision": "float32",
        "max_public_events": None,
        "teacher_target": "canonical arithmetic mean of logits, independent recurrent states",
        "student_initialization": "exact parent clone",
        "weight_decay": 1e-5,
        "length_bucketed": True,
        "promotion": False,
        "sources_sha256": {name: file_sha256(Path(name)) for name in (
            "distill_orbit_temporal.py", "strongest_policy_search.py",
            "puffer_distillation.py", "vector_puffer_rollout.py", "temporal_puffer_v4.py",
        )},
    }
    atomic_json_save(recipe, output / "recipe.json")
    started = time.perf_counter()
    with VectorPufferRolloutPool(
        student, workers=args.workers, champion_path=champion_path,
        rollout_precision="float32", max_public_events=None,
    ) as pool:
        training, train_contribution = pool.collect(
            student, games=args.games, seed=args.seed, schedule_offset=0,
            schedule=OPPONENT_SCHEDULE, player_counts=PLAYER_COUNT_SCHEDULE,
            prepack_training=False, max_turns=args.max_turns,
        )
        validation, validation_contribution = pool.collect(
            student, games=args.validation_games, seed=args.seed + args.games,
            schedule_offset=args.games, schedule=OPPONENT_SCHEDULE,
            player_counts=PLAYER_COUNT_SCHEDULE, prepack_training=False,
            max_turns=args.max_turns,
        )
    collection_seconds = time.perf_counter() - started
    print(json.dumps({"stage": "collected", "seconds": collection_seconds,
                      "training_trajectories": len(training),
                      "validation_trajectories": len(validation)}), flush=True)
    initial = evaluate_puffer_distillation(
        student, teacher, validation, minibatch_trajectories=args.minibatch_trajectories
    )
    progress = {
        "recipe": recipe, "model": model_config, "initial_validation": initial,
        "collection_seconds": collection_seconds,
        "training_contribution": train_contribution,
        "validation_contribution": validation_contribution,
        "epochs": [], "status": "training", "promotion": False,
    }
    atomic_json_save(progress, output / "result.json")
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        metrics = distill_puffer_policy(
            student, teacher, training, optimizer, epochs=1,
            minibatch_trajectories=args.minibatch_trajectories, seed=args.seed + epoch,
            temperature=args.temperature, value_coefficient=args.value_coefficient,
            strategic_pass_weight=args.strategic_pass_weight,
            reverse_kl_coefficient=args.reverse_kl_coefficient,
            hard_target_coefficient=args.hard_target_coefficient,
            disagreement_weight=args.disagreement_weight, length_bucketed=True,
        )[0]
        train_seconds = time.perf_counter() - epoch_started
        validation_metrics = evaluate_puffer_distillation(
            student, teacher, validation,
            minibatch_trajectories=args.minibatch_trajectories,
        )
        metrics.update({"epoch": epoch, "training_seconds": train_seconds,
                        "samples_per_second": metrics["samples"] / train_seconds,
                        "validation": validation_metrics})
        progress["epochs"].append(metrics)
        checkpoint = output / f"epoch_{epoch:03d}.pt"
        atomic_torch_save({
            "format": "pesten-orbit-distilled-temporal-v1",
            "epoch": int(parent_payload.get("epoch", 0)),
            "distillation_epoch": epoch, "model": model_config,
            "parent_checkpoint": str(teacher_path),
            "parent_sha256": recipe["parent_sha256"],
            "state_dict": cpu_state(student), "optimizer_state_dict": optimizer.state_dict(),
            "recipe": recipe, "metrics": metrics,
        }, checkpoint)
        progress.update({"checkpoint": str(checkpoint),
                         "elapsed_seconds": time.perf_counter() - started,
                         "status": "complete" if epoch == args.epochs else "training"})
        atomic_json_save(progress, output / "result.json")
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
