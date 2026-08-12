#!/usr/bin/env python3
"""Distill the plateau MinGRU into a 23-score compositional action head.

The student keeps the teacher's recurrent body and value function, replaces
the arbitrary 55-logit policy head with suit-plus-rank scores, and uses DAgger
to cover states induced by both policies. This stage establishes parity; a
separate matched RLVR fork measures whether the simpler head learns faster.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from sota_distill_temporal import (
    OPPONENT_SCHEDULE,
    PLAYER_COUNT_SCHEDULE,
    atomic_json_save,
    atomic_torch_save,
    collect,
    cpu_state,
    player_distribution,
)


OUTPUT_LAYOUT = (
    {"indices": "0-3", "meaning": "playing-card suit scores: harten, ruiten, schoppen, klaveren"},
    {"indices": "4-15", "meaning": "ordinary rank scores: ace, 2 through 10, king, queen"},
    {"indices": "16", "meaning": "jack/boer score"},
    {"indices": "17", "meaning": "joker score"},
    {"indices": "18", "meaning": "pass score"},
    {"indices": "19-22", "meaning": "announced-suit scores after a jack"},
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-games", type=int, default=8192)
    parser.add_argument("--teacher-validation-games", type=int, default=4096)
    parser.add_argument("--student-validation-games", type=int, default=2048)
    parser.add_argument("--dagger-games", type=int, default=4096)
    parser.add_argument("--max-rounds", type=int, default=10)
    parser.add_argument("--epochs-per-round", type=int, default=4)
    parser.add_argument("--head-only-rounds", type=int, default=1)
    parser.add_argument("--minibatch-trajectories", type=int, default=128)
    parser.add_argument("--head-learning-rate", type=float, default=5e-4)
    parser.add_argument("--body-learning-rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=1.25)
    parser.add_argument("--strategic-pass-weight", type=float, default=3.0)
    parser.add_argument("--reverse-kl-coefficient", type=float, default=0.25)
    parser.add_argument("--hard-target-coefficient", type=float, default=0.5)
    parser.add_argument("--disagreement-weight", type=float, default=4.0)
    parser.add_argument("--value-coefficient", type=float, default=0.1)
    parser.add_argument("--target-agreement", type=float, default=0.999)
    parser.add_argument("--target-forward-kl", type=float, default=0.02)
    parser.add_argument("--target-pass-rate-delta", type=float, default=0.005)
    parser.add_argument("--target-value-mse", type=float, default=1e-5)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def _passes_gate(metrics: dict, args: argparse.Namespace) -> bool:
    teacher_pass = metrics["teacher_strategic_pass_rate"]
    student_pass = metrics["student_strategic_pass_rate"]
    pass_delta = (
        abs(student_pass - teacher_pass)
        if teacher_pass is not None and student_pass is not None
        else 0.0
    )
    return (
        metrics["action_agreement"] >= args.target_agreement
        and metrics["forward_kl"] <= args.target_forward_kl
        and metrics["value_mse"] <= args.target_value_mse
        and pass_delta <= args.target_pass_rate_delta
    )


def _checkpoint_payload(
    student,
    teacher_path: Path,
    teacher_sha256: str,
    teacher_epoch: int,
    optimizer,
    initialization: dict,
    rounds: list[dict],
    match: bool,
    args: argparse.Namespace,
) -> dict:
    from temporal_puffer_v4 import policy_model_config

    return {
        "format": "pesten-factorized-temporal-v1",
        "epoch": teacher_epoch,
        "model": policy_model_config(student),
        "parent_checkpoint": str(teacher_path),
        "teacher_sha256": teacher_sha256,
        "state_dict": cpu_state(student),
        "optimizer_state_dict": optimizer.state_dict(),
        "recipe": {
            "method": "projected_same_body_dagger_distillation",
            "output_layout": list(OUTPUT_LAYOUT),
            "schedule": list(OPPONENT_SCHEDULE),
            "player_count_schedule": list(PLAYER_COUNT_SCHEDULE),
            "head_learning_rate": args.head_learning_rate,
            "body_learning_rate": args.body_learning_rate,
            "head_only_rounds": args.head_only_rounds,
            "temperature": args.temperature,
            "strategic_pass_weight": args.strategic_pass_weight,
            "reverse_kl_coefficient": args.reverse_kl_coefficient,
            "hard_target_coefficient": args.hard_target_coefficient,
            "disagreement_weight": args.disagreement_weight,
            "value_coefficient": args.value_coefficient,
            "initialization": initialization,
            "rounds": rounds,
            "match_gate_passed": match,
        },
    }


def main() -> None:
    args = parse_args()
    counts = (
        args.teacher_games,
        args.teacher_validation_games,
        args.student_validation_games,
        args.dagger_games,
        args.max_rounds,
        args.epochs_per_round,
        args.minibatch_trajectories,
        args.workers,
    )
    if any(value < 1 for value in counts) or args.head_only_rounds < 0:
        raise ValueError("game, training, and worker counts must be valid")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA distillation requested but unavailable")
    teacher_path = args.teacher.resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from puffer_distillation import (
        distill_puffer_policy,
        evaluate_puffer_distillation,
        initialize_factorized_min_gru,
    )
    from temporal_puffer_v4 import (
        FACTORIZED_ACTION_HEAD,
        NativePufferTemporalPolicy,
        load_temporal_policy_checkpoint,
        policy_model_config,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    teacher_rollout, teacher_payload = load_temporal_policy_checkpoint(
        teacher_path
    )
    teacher_model = policy_model_config(teacher_rollout)
    if teacher_model["architecture"] != "min_gru":
        raise ValueError("factorized student requires a MinGRU teacher")
    teacher_sha256 = file_sha256(teacher_path)
    teacher_epoch = int(teacher_payload.get("epoch", 0))
    teacher = teacher_rollout.to(args.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = NativePufferTemporalPolicy(
        hidden_size=int(teacher_model["hidden_size"]),
        num_layers=int(teacher_model["num_layers"]),
        mode_conditioned_heads=True,
        action_head=FACTORIZED_ACTION_HEAD,
    ).to(args.device)
    initialization = initialize_factorized_min_gru(student, teacher)
    score_parameters = list(student.puffer_policy.decoder.scores.parameters())
    score_ids = {id(parameter) for parameter in score_parameters}
    body_parameters = [
        parameter
        for parameter in student.parameters()
        if id(parameter) not in score_ids
    ]
    optimizer = torch.optim.AdamW(
        (
            {
                "params": score_parameters,
                "lr": args.head_learning_rate,
                "name": "factorized_scores",
            },
            {
                "params": body_parameters,
                "lr": 0.0 if args.head_only_rounds else args.body_learning_rate,
                "name": "recurrent_body_and_value",
            },
        ),
        betas=(0.9, 0.95),
        weight_decay=1e-5,
    )

    started = time.perf_counter()
    teacher_episodes, teacher_contribution, teacher_seconds = collect(
        teacher_rollout,
        games=args.teacher_games,
        seed=args.seed,
        workers=args.workers,
    )
    fixed_validation, validation_contribution, validation_seconds = collect(
        teacher_rollout,
        games=args.teacher_validation_games,
        seed=args.seed + 10_000_000,
        workers=args.workers,
    )
    replay = list(teacher_episodes)
    initial_validation = evaluate_puffer_distillation(
        student,
        teacher,
        fixed_validation,
        minibatch_trajectories=args.minibatch_trajectories,
    )
    rounds = []
    match = False
    checkpoint = output_dir / "latest.pt"
    progress_path = output_dir / "distillation_progress.json"

    for round_index in range(1, args.max_rounds + 1):
        round_started = time.perf_counter()
        body_lr = (
            0.0
            if round_index <= args.head_only_rounds
            else args.body_learning_rate
        )
        optimizer.param_groups[1]["lr"] = body_lr
        curve = distill_puffer_policy(
            student,
            teacher,
            replay,
            optimizer,
            epochs=args.epochs_per_round,
            minibatch_trajectories=args.minibatch_trajectories,
            seed=args.seed + round_index * 1_000_000,
            temperature=args.temperature,
            value_coefficient=args.value_coefficient,
            strategic_pass_weight=args.strategic_pass_weight,
            reverse_kl_coefficient=args.reverse_kl_coefficient,
            hard_target_coefficient=args.hard_target_coefficient,
            disagreement_weight=args.disagreement_weight,
        )
        teacher_validation = evaluate_puffer_distillation(
            student,
            teacher,
            fixed_validation,
            minibatch_trajectories=args.minibatch_trajectories,
        )
        student_validation_episodes, student_contribution, student_seconds = collect(
            student.to("cpu"),
            games=args.student_validation_games,
            seed=args.seed + 20_000_000 + round_index * 100_000,
            workers=args.workers,
        )
        student.to(args.device)
        student_validation = evaluate_puffer_distillation(
            student,
            teacher,
            student_validation_episodes,
            minibatch_trajectories=args.minibatch_trajectories,
        )
        match = _passes_gate(teacher_validation, args) and _passes_gate(
            student_validation, args
        )
        row = {
            "round": round_index,
            "head_learning_rate": optimizer.param_groups[0]["lr"],
            "body_learning_rate": body_lr,
            "replay_trajectories": len(replay),
            "replay_decisions": sum(len(trajectory) for trajectory, _ in replay),
            "curve": curve,
            "teacher_policy_validation": teacher_validation,
            "student_policy_validation": student_validation,
            "student_validation_contribution": student_contribution,
            "student_validation_rollout_seconds": student_seconds,
            "match_gate_passed": match,
            "wall_seconds": time.perf_counter() - round_started,
        }
        rounds.append(row)
        payload = _checkpoint_payload(
            student,
            teacher_path,
            teacher_sha256,
            teacher_epoch,
            optimizer,
            initialization,
            rounds,
            match,
            args,
        )
        atomic_torch_save(payload, checkpoint)
        atomic_torch_save(
            payload, output_dir / f"factorized_round_{round_index:03d}.pt"
        )
        progress = {
            "schema_version": 1,
            "kind": "factorized_23_score_distillation_progress",
            "teacher": {
                "path": str(teacher_path),
                "sha256": teacher_sha256,
                "epoch": teacher_epoch,
                "model": teacher_model,
            },
            "student": {
                "path": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "model": policy_model_config(student),
                "parameters": sum(p.numel() for p in student.parameters()),
            },
            "output_layout": list(OUTPUT_LAYOUT),
            "initialization": initialization,
            "initial_validation": initial_validation,
            "rounds": rounds,
            "match_gate_passed": match,
        }
        atomic_json_save(progress, progress_path)
        print(
            json.dumps(
                {
                    "round": round_index,
                    "teacher_validation": teacher_validation,
                    "student_validation": student_validation,
                    "match_gate_passed": match,
                    "checkpoint": str(checkpoint),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if match:
            break
        dagger, dagger_contribution, dagger_seconds = collect(
            student.to("cpu"),
            games=args.dagger_games,
            seed=args.seed + 30_000_000 + round_index * args.dagger_games,
            workers=args.workers,
        )
        student.to(args.device)
        replay.extend(dagger)
        row["dagger_added_trajectories"] = len(dagger)
        row["dagger_contribution"] = dagger_contribution
        row["dagger_rollout_seconds"] = dagger_seconds

    result = {
        "schema_version": 1,
        "kind": "factorized_23_score_distillation",
        "teacher": {
            "path": str(teacher_path),
            "sha256": teacher_sha256,
            "epoch": teacher_epoch,
            "model": teacher_model,
            "parameters": sum(p.numel() for p in teacher.parameters()),
        },
        "student": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "model": policy_model_config(student),
            "parameters": sum(p.numel() for p in student.parameters()),
        },
        "output_layout": list(OUTPUT_LAYOUT),
        "initialization": initialization,
        "data": {
            "teacher_games": args.teacher_games,
            "teacher_contribution": teacher_contribution,
            "teacher_rollout_seconds": teacher_seconds,
            "teacher_validation_games": args.teacher_validation_games,
            "teacher_validation_contribution": validation_contribution,
            "teacher_validation_rollout_seconds": validation_seconds,
            "opponent_schedule": list(OPPONENT_SCHEDULE),
            "player_distribution": player_distribution(),
        },
        "initial_validation": initial_validation,
        "rounds": rounds,
        "match_gate": {
            "target_action_agreement": args.target_agreement,
            "target_forward_kl": args.target_forward_kl,
            "target_pass_rate_delta": args.target_pass_rate_delta,
            "target_value_mse": args.target_value_mse,
            "passed": match,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "device": str(next(student.parameters()).device),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
        },
    }
    atomic_json_save(result, output_dir / "distillation_result.json")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not match:
        raise SystemExit(
            "factorized student reached the configured DAgger round limit "
            "before matching the teacher"
        )


if __name__ == "__main__":
    main()
