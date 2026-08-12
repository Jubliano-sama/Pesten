#!/usr/bin/env python3
"""Build and verify an exact-teacher Hawk residual distillation student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from sota_distill_temporal import (
    atomic_json_save,
    atomic_torch_save,
    checkpoint_payload,
    collect,
    file_sha256,
    player_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-hidden-size", type=int, default=128)
    parser.add_argument("--adapter-layers", type=int, default=2)
    parser.add_argument("--adapter-recurrent-size", type=int, default=160)
    parser.add_argument("--adapter-mlp-expansion", type=int, default=2)
    parser.add_argument("--validation-games", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_813)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = (
        args.adapter_hidden_size,
        args.adapter_layers,
        args.adapter_recurrent_size,
        args.adapter_mlp_expansion,
        args.validation_games,
        args.workers,
    )
    if any(value < 1 for value in counts):
        raise ValueError("model, game, and worker counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA residual verification requested but unavailable")
    teacher_path = args.teacher.resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from puffer_distillation import (
        evaluate_puffer_distillation,
        initialize_hawk_residual,
    )
    from temporal_puffer_v4 import (
        HawkResidualTemporalPolicy,
        load_temporal_policy_checkpoint,
        policy_model_config,
    )

    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    teacher_rollout, _ = load_temporal_policy_checkpoint(teacher_path)
    teacher_model = policy_model_config(teacher_rollout)
    if teacher_model["architecture"] != "min_gru":
        raise ValueError("the frozen residual teacher must be a MinGRU")
    teacher = teacher_rollout.to(args.device).eval()
    student = HawkResidualTemporalPolicy(
        base_hidden_size=int(teacher_model["hidden_size"]),
        base_num_layers=int(teacher_model["num_layers"]),
        adapter_hidden_size=args.adapter_hidden_size,
        adapter_num_layers=args.adapter_layers,
        adapter_recurrent_size=args.adapter_recurrent_size,
        adapter_mlp_expansion=args.adapter_mlp_expansion,
        mode_conditioned_heads=bool(teacher_model["mode_conditioned_heads"]),
    ).to(args.device)
    initialization = initialize_hawk_residual(student, teacher)

    episodes, contribution, rollout_seconds = collect(
        teacher_rollout,
        games=args.validation_games,
        seed=args.seed + 10_000_000,
        workers=args.workers,
    )
    metrics = evaluate_puffer_distillation(
        student,
        teacher,
        episodes,
        minibatch_trajectories=128,
    )
    gate = (
        metrics["action_agreement"] == 1.0
        and metrics["forward_kl"] <= 1e-10
        and metrics["value_mse"] <= 1e-12
    )
    if not gate:
        raise RuntimeError(
            "function-preserving residual failed exact teacher gate: "
            + json.dumps(metrics, sort_keys=True)
        )

    teacher_sha256 = file_sha256(teacher_path)
    distillation = {
        **initialization,
        "validation_metrics": metrics,
        "validation_contribution": contribution,
        "validation_rollout_seconds": rollout_seconds,
        "exact_teacher_gate_passed": gate,
    }
    checkpoint = output_dir / "hawk_residual.pt"
    atomic_torch_save(
        checkpoint_payload(
            student,
            teacher_path=teacher_path,
            teacher_sha256=teacher_sha256,
            epoch=0,
            optimizer=None,
            distillation=distillation,
        ),
        checkpoint,
    )
    result = {
        "schema_version": 1,
        "kind": "function_preserving_hawk_residual_distillation",
        "teacher": {
            "path": str(teacher_path),
            "sha256": teacher_sha256,
            "model": teacher_model,
            "parameters": sum(p.numel() for p in teacher.parameters()),
        },
        "student": {
            "path": str(checkpoint),
            "sha256": file_sha256(checkpoint),
            "model": policy_model_config(student),
            "parameters": sum(p.numel() for p in student.parameters()),
            "recurrent_state_elements": sum(
                state.numel() for state in student.initial_state(1)
            ),
        },
        "initialization": initialization,
        "validation": {
            "games": args.validation_games,
            "metrics": metrics,
            "contribution": contribution,
            "rollout_seconds": rollout_seconds,
            "exact_teacher_gate_passed": gate,
        },
        "player_distribution": player_distribution(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json_save(result, output_dir / "hawk_residual_result.json")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
