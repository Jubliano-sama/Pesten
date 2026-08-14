#!/usr/bin/env python3
"""Distill an exact wider MinGRU and an on-policy Hawk RG-LRU student.

The wider student uses a function-preserving network morphism, then verifies
the inherited policy on held-out safe trajectories.  The alternative student
uses DAgger-style dataset aggregation: it repeatedly visits states induced by
its own policy and asks the frozen teacher for dense policy/value targets.
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


PLAYER_COUNT_SCHEDULE = (
    (2,) * 14
    + (3,) * 14
    + (4,) * 14
    + (5,) * 6
    + (6,) * 5
    + (7,) * 5
    + (8,) * 5
)
OPPONENT_SCHEDULE = (
    "current_selfplay",
    "current_selfplay",
    "ppo_champion",
    "random_rule",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_json_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def cpu_state(policy) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()
    }


def collect(policy, *, games: int, seed: int, workers: int):
    from vector_puffer_rollout import VectorPufferRolloutPool

    started = time.perf_counter()
    with VectorPufferRolloutPool(policy, workers=workers) as pool:
        episodes, contribution = pool.collect(
            policy,
            games=games,
            seed=seed,
            schedule_offset=0,
            schedule=OPPONENT_SCHEDULE,
            player_counts=PLAYER_COUNT_SCHEDULE,
            prepack_training=False,
        )
    return episodes, contribution, time.perf_counter() - started


def player_distribution() -> dict[str, object]:
    bulk = sum(count in (2, 3, 4) for count in PLAYER_COUNT_SCHEDULE)
    tail = len(PLAYER_COUNT_SCHEDULE) - bulk
    return {
        "cycle": list(PLAYER_COUNT_SCHEDULE),
        "cycle_length": len(PLAYER_COUNT_SCHEDULE),
        "counts": {
            str(players): PLAYER_COUNT_SCHEDULE.count(players)
            for players in range(2, 9)
        },
        "bulk_2_3_4_fraction": bulk / len(PLAYER_COUNT_SCHEDULE),
        "tail_5_8_fraction": tail / len(PLAYER_COUNT_SCHEDULE),
    }


def checkpoint_payload(
    policy,
    *,
    teacher_path: Path,
    teacher_sha256: str,
    epoch: int,
    optimizer,
    distillation: dict,
) -> dict:
    from temporal_puffer_v4 import policy_model_config

    payload = {
        "format": "pesten-sota-distilled-temporal-v1",
        "epoch": epoch,
        "model": policy_model_config(policy),
        "parent_checkpoint": str(teacher_path),
        "teacher_sha256": teacher_sha256,
        "state_dict": cpu_state(policy),
        "recipe": {
            "schedule": list(OPPONENT_SCHEDULE),
            "player_count_schedule": list(PLAYER_COUNT_SCHEDULE),
            "learning_rate": (
                optimizer.param_groups[0]["lr"]
                if optimizer is not None
                else 1e-4
            ),
            "distillation": distillation,
        },
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--hawk-resume",
        type=Path,
        help="continue the alternative student from a prior campaign checkpoint",
    )
    parser.add_argument(
        "--large-only",
        action="store_true",
        help="build and validate only the function-preserving wider MinGRU",
    )
    parser.add_argument("--large-hidden-size", type=int, default=512)
    parser.add_argument("--hawk-hidden-size", type=int, default=256)
    parser.add_argument("--hawk-layers", type=int, default=4)
    parser.add_argument("--hawk-recurrent-size", type=int, default=320)
    parser.add_argument("--hawk-mlp-expansion", type=int, default=2)
    parser.add_argument("--teacher-games", type=int, default=4096)
    parser.add_argument("--validation-games", type=int, default=1024)
    parser.add_argument("--dagger-games", type=int, default=2048)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--epochs-per-round", type=int, default=6)
    parser.add_argument("--minibatch-trajectories", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--strategic-pass-weight", type=float, default=2.5)
    parser.add_argument("--reverse-kl-coefficient", type=float, default=0.35)
    parser.add_argument("--hard-target-coefficient", type=float, default=0.15)
    parser.add_argument("--disagreement-weight", type=float, default=1.0)
    parser.add_argument("--value-coefficient", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_260_812)
    parser.add_argument("--target-agreement", type=float, default=0.985)
    parser.add_argument("--target-forward-kl", type=float, default=0.04)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = (
        args.teacher_games,
        args.validation_games,
        args.dagger_games,
        args.max_rounds,
        args.epochs_per_round,
        args.minibatch_trajectories,
        args.workers,
    )
    if any(value < 1 for value in counts):
        raise ValueError("game, round, epoch, minibatch, and worker counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA distillation requested but unavailable")
    if not 0 < args.target_agreement <= 1 or args.target_forward_kl < 0:
        raise ValueError("invalid distillation match targets")
    teacher_path = args.teacher.resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from puffer_distillation import (
        distill_puffer_policy,
        evaluate_puffer_distillation,
        initialize_widened_min_gru,
    )
    from temporal_puffer_v4 import (
        HawkRGLRUTemporalPolicy,
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
        raise ValueError("the frozen distillation teacher must be a MinGRU")
    teacher_sha256 = file_sha256(teacher_path)
    teacher = teacher_rollout.to(args.device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    started = time.perf_counter()
    teacher_episodes = None
    teacher_contribution = None
    teacher_rollout_seconds = None
    if not args.large_only:
        teacher_episodes, teacher_contribution, teacher_rollout_seconds = collect(
            teacher_rollout,
            games=args.teacher_games,
            seed=args.seed,
            workers=args.workers,
        )
    validation_episodes, validation_contribution, validation_rollout_seconds = collect(
        teacher_rollout,
        games=args.validation_games,
        seed=args.seed + 10_000_000,
        workers=args.workers,
    )

    large = NativePufferTemporalPolicy(
        hidden_size=args.large_hidden_size,
        num_layers=int(teacher_model["num_layers"]),
        mode_conditioned_heads=bool(
            teacher_model["mode_conditioned_heads"]
        ),
        action_head=str(teacher_model["action_head"]),
    ).to(args.device)
    widening = initialize_widened_min_gru(large, teacher)
    large_metrics = evaluate_puffer_distillation(
        large,
        teacher,
        validation_episodes,
        minibatch_trajectories=args.minibatch_trajectories,
    )
    large_match = (
        large_metrics["action_agreement"] >= 0.9999
        and abs(large_metrics["forward_kl"]) <= 1e-6
        and large_metrics["value_mse"] <= 1e-8
        and abs(
            large_metrics["student_strategic_pass_rate"]
            - large_metrics["teacher_strategic_pass_rate"]
        )
        <= 1e-4
    )
    if not large_match:
        raise RuntimeError(
            "function-preserving large student failed its exact match gate: "
            + json.dumps(large_metrics, sort_keys=True)
        )
    large_path = output_dir / (
        f"large_min_gru_{args.large_hidden_size}x"
        f"{teacher_model['num_layers']}.pt"
    )
    large_distillation = {
        "method": "function_preserving_widening_then_dense_teacher_anchor",
        "widening": widening,
        "held_out_metrics": large_metrics,
        "match_gate_passed": large_match,
    }
    atomic_torch_save(
        checkpoint_payload(
            large,
            teacher_path=teacher_path,
            teacher_sha256=teacher_sha256,
            epoch=int(teacher_payload.get("epoch", 0)),
            optimizer=None,
            distillation=large_distillation,
        ),
        large_path,
    )
    if args.large_only:
        result = {
            "schema_version": 1,
            "kind": "function_preserving_large_min_gru_distillation",
            "teacher": {
                "path": str(teacher_path),
                "sha256": teacher_sha256,
                "model": teacher_model,
                "parameters": sum(p.numel() for p in teacher.parameters()),
            },
            "student": {
                "path": str(large_path),
                "sha256": file_sha256(large_path),
                "model": policy_model_config(large),
                "parameters": sum(p.numel() for p in large.parameters()),
                "recurrent_state_elements": sum(
                    state.numel() for state in large.initial_state(1)
                ),
            },
            "initialization": widening,
            "validation": {
                "games": args.validation_games,
                "metrics": large_metrics,
                "contribution": validation_contribution,
                "rollout_seconds": validation_rollout_seconds,
                "match_gate_passed": large_match,
            },
            "player_distribution": player_distribution(),
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json_save(
            result, output_dir / "large_min_gru_result.json"
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return

    hawk_resume_payload = None
    if args.hawk_resume is not None:
        hawk_resume_path = args.hawk_resume.resolve()
        if not hawk_resume_path.is_file():
            raise FileNotFoundError(hawk_resume_path)
        hawk, hawk_resume_payload = load_temporal_policy_checkpoint(
            hawk_resume_path, device=args.device
        )
        if policy_model_config(hawk)["architecture"] != "hawk_rglru":
            raise ValueError("--hawk-resume must contain a Hawk RG-LRU policy")
    else:
        hawk = HawkRGLRUTemporalPolicy(
            hidden_size=args.hawk_hidden_size,
            num_layers=args.hawk_layers,
            recurrent_size=args.hawk_recurrent_size,
            mlp_expansion=args.hawk_mlp_expansion,
            mode_conditioned_heads=True,
        ).to(args.device)
    optimizer = torch.optim.AdamW(
        hawk.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=1e-4,
    )
    if hawk_resume_payload is not None and "optimizer_state_dict" in hawk_resume_payload:
        optimizer.load_state_dict(hawk_resume_payload["optimizer_state_dict"])
        parameter_device = next(hawk.parameters()).device
        for state_values in optimizer.state.values():
            for name, value in tuple(state_values.items()):
                if isinstance(value, torch.Tensor):
                    state_values[name] = value.to(parameter_device)
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate
    replay = list(teacher_episodes)
    rounds = []
    hawk_match = False
    hawk_path = output_dir / "hawk_rglru.pt"
    for round_index in range(1, args.max_rounds + 1):
        round_started = time.perf_counter()
        curve = distill_puffer_policy(
            hawk,
            teacher,
            replay,
            optimizer,
            epochs=args.epochs_per_round,
            minibatch_trajectories=args.minibatch_trajectories,
            seed=args.seed + 1_000_000 * round_index,
            temperature=args.temperature,
            value_coefficient=args.value_coefficient,
            strategic_pass_weight=args.strategic_pass_weight,
            reverse_kl_coefficient=args.reverse_kl_coefficient,
            hard_target_coefficient=args.hard_target_coefficient,
            disagreement_weight=args.disagreement_weight,
        )
        teacher_validation = evaluate_puffer_distillation(
            hawk,
            teacher,
            validation_episodes,
            minibatch_trajectories=args.minibatch_trajectories,
        )
        student_validation_episodes, student_contribution, student_seconds = collect(
            hawk,
            games=max(256, args.validation_games // 2),
            seed=args.seed + 20_000_000 + round_index * 100_000,
            workers=args.workers,
        )
        student_validation = evaluate_puffer_distillation(
            hawk,
            teacher,
            student_validation_episodes,
            minibatch_trajectories=args.minibatch_trajectories,
        )
        hawk_match = all(
            metrics["action_agreement"] >= args.target_agreement
            and metrics["forward_kl"] <= args.target_forward_kl
            for metrics in (teacher_validation, student_validation)
        )
        row = {
            "round": round_index,
            "replay_trajectories": len(replay),
            "replay_decisions": sum(len(trajectory) for trajectory, _ in replay),
            "curve": curve,
            "teacher_policy_validation": teacher_validation,
            "student_policy_validation": student_validation,
            "student_validation_contribution": student_contribution,
            "student_validation_rollout_seconds": student_seconds,
            "match_gate_passed": hawk_match,
            "wall_seconds": time.perf_counter() - round_started,
        }
        rounds.append(row)
        distillation = {
            "method": "dagger_bidirectional_kl_hard_target_value",
            "rounds": rounds,
            "loss": {
                "temperature": args.temperature,
                "strategic_pass_weight": args.strategic_pass_weight,
                "reverse_kl_coefficient": args.reverse_kl_coefficient,
                "hard_target_coefficient": args.hard_target_coefficient,
                "disagreement_weight": args.disagreement_weight,
                "value_coefficient": args.value_coefficient,
            },
            "target_action_agreement": args.target_agreement,
            "target_forward_kl": args.target_forward_kl,
            "match_gate_passed": hawk_match,
        }
        atomic_torch_save(
            checkpoint_payload(
                hawk,
                teacher_path=teacher_path,
                teacher_sha256=teacher_sha256,
                epoch=int(teacher_payload.get("epoch", 0)),
                optimizer=optimizer,
                distillation=distillation,
            ),
            hawk_path,
        )
        atomic_json_save(
            {
                "schema_version": 1,
                "kind": "sota_temporal_distillation_progress",
                "rounds": rounds,
                "latest_checkpoint": str(hawk_path),
                "match_gate_passed": hawk_match,
            },
            output_dir / "hawk_rglru.progress.json",
        )
        print(
            json.dumps(
                {
                    "round": round_index,
                    "teacher_validation": teacher_validation,
                    "student_validation": student_validation,
                    "match": hawk_match,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if hawk_match:
            break
        if round_index == args.max_rounds:
            continue
        dagger_episodes, dagger_contribution, dagger_seconds = collect(
            hawk,
            games=args.dagger_games,
            seed=args.seed + 30_000_000 + round_index * args.dagger_games,
            workers=args.workers,
        )
        replay.extend(dagger_episodes)
        row["dagger_contribution"] = dagger_contribution
        row["dagger_rollout_seconds"] = dagger_seconds
        row["dagger_added_trajectories"] = len(dagger_episodes)

    result = {
        "schema_version": 1,
        "kind": "sota_temporal_distillation_campaign",
        "teacher": {
            "path": str(teacher_path),
            "sha256": teacher_sha256,
            "epoch": int(teacher_payload.get("epoch", 0)),
            "model": teacher_model,
        },
        "data": {
            "teacher_games": args.teacher_games,
            "teacher_contribution": teacher_contribution,
            "teacher_rollout_seconds": teacher_rollout_seconds,
            "validation_games": args.validation_games,
            "validation_contribution": validation_contribution,
            "validation_rollout_seconds": validation_rollout_seconds,
            "opponent_schedule": list(OPPONENT_SCHEDULE),
            "player_distribution": player_distribution(),
        },
        "large_student": {
            "checkpoint": str(large_path),
            "model": policy_model_config(large),
            "parameters": sum(parameter.numel() for parameter in large.parameters()),
            "distillation": large_distillation,
        },
        "alternative_student": {
            "checkpoint": str(hawk_path),
            "model": policy_model_config(hawk),
            "parameters": sum(parameter.numel() for parameter in hawk.parameters()),
            "rounds": rounds,
            "match_gate_passed": hawk_match,
            "resume_checkpoint": (
                str(args.hawk_resume.resolve())
                if args.hawk_resume is not None
                else None
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "device": str(next(hawk.parameters()).device),
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
        },
    }
    atomic_json_save(result, output_dir / "distillation_result.json")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not hawk_match:
        raise SystemExit(
            "Hawk student reached the configured DAgger round limit before its match gate"
        )


if __name__ == "__main__":
    main()
