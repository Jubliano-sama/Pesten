#!/usr/bin/env python3
"""Create and rigorously validate an exact wider/deeper MinGRU student."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puffer_distillation import (
    evaluate_puffer_distillation,
    initialize_min_gru_residual,
)
from temporal_puffer_v4 import (
    MinGRUResidualTemporalPolicy,
    load_temporal_policy_checkpoint,
    policy_model_config,
)
from vector_puffer_rollout import VectorPufferRolloutPool


PLAYER_COUNTS = (
    (2,) * 14
    + (3,) * 14
    + (4,) * 14
    + (5,) * 6
    + (6,) * 5
    + (7,) * 5
    + (8,) * 5
)
OPPONENTS = (
    "current_selfplay",
    "current_selfplay",
    "historical_temporal",
    "ppo_champion",
    "canonical_random",
    "smart_rule",
    "smarter_rule",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_torch(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--historical", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--adapter-hidden-size", type=int, default=512)
    parser.add_argument("--adapter-layers", type=int, default=6)
    parser.add_argument("--validation-games", type=int, default=16_384)
    parser.add_argument("--shard-games", type=int, default=2_048)
    parser.add_argument("--validation-seed", type=int, default=140_000_000)
    parser.add_argument("--reload-games", type=int, default=2_048)
    parser.add_argument("--reload-seed", type=int, default=141_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--minibatch-trajectories", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=12)
    return parser.parse_args()


def weighted_metrics(rows: list[dict]) -> dict:
    samples = sum(int(row["samples"]) for row in rows)
    opportunities = sum(
        int(row["strategic_pass_opportunities"]) for row in rows
    )
    agreements = sum(
        round(float(row["action_agreement"]) * int(row["samples"]))
        for row in rows
    )
    teacher_passes = sum(
        round(
            float(row["teacher_strategic_pass_rate"])
            * int(row["strategic_pass_opportunities"])
        )
        for row in rows
    )
    student_passes = sum(
        round(
            float(row["student_strategic_pass_rate"])
            * int(row["strategic_pass_opportunities"])
        )
        for row in rows
    )
    return {
        "samples": samples,
        "agreement_count": agreements,
        "action_agreement": agreements / samples,
        "forward_kl": sum(
            float(row["forward_kl"]) * int(row["samples"])
            for row in rows
        )
        / samples,
        "value_mse": sum(
            float(row["value_mse"]) * int(row["samples"])
            for row in rows
        )
        / samples,
        "max_legal_logit_error": max(
            float(row["max_legal_logit_error"]) for row in rows
        ),
        "strategic_pass_opportunities": opportunities,
        "teacher_strategic_pass_rate": (
            teacher_passes / opportunities if opportunities else None
        ),
        "student_strategic_pass_rate": (
            student_passes / opportunities if opportunities else None
        ),
    }


def validate_sharded(
    pool,
    *,
    student,
    teacher,
    games: int,
    shard_games: int,
    seed: int,
    minibatch_trajectories: int,
) -> tuple[dict, list[dict], float]:
    rows = []
    started = time.perf_counter()
    for offset in range(0, games, shard_games):
        count = min(shard_games, games - offset)
        episodes, contribution = pool.collect(
            teacher,
            games=count,
            seed=seed + offset,
            schedule_offset=offset,
            schedule=OPPONENTS,
            player_counts=PLAYER_COUNTS,
            prepack_training=False,
            max_turns=500,
        )
        metrics = evaluate_puffer_distillation(
            student,
            teacher,
            episodes,
            minibatch_trajectories=minibatch_trajectories,
        )
        row = {
            "game_offset": offset,
            "games": count,
            "seed_start": seed + offset,
            "metrics": metrics,
            "contribution": contribution,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "event": "heldout_shard",
                    "games_complete": offset + count,
                    "games_total": games,
                    "metrics": metrics,
                }
            ),
            flush=True,
        )
        if metrics["action_agreement"] != 1.0:
            raise RuntimeError(
                "wider/deeper student failed exact action agreement: "
                + json.dumps(row, sort_keys=True)
            )
    return (
        weighted_metrics([row["metrics"] for row in rows]),
        rows,
        time.perf_counter() - started,
    )


def checkpoint_payload(student, teacher_path: Path, teacher_payload: dict, report: dict) -> dict:
    return {
        "format": "pesten-exact-wider-deeper-min-gru-v1",
        "epoch": int(teacher_payload.get("epoch", 0)),
        "model": policy_model_config(student),
        "parent_checkpoint": str(teacher_path.resolve()),
        "teacher_sha256": sha256(teacher_path),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in student.state_dict().items()
        },
        "recipe": {
            "distillation": report,
            "supervised_only": True,
            "rewards_used_for_updates": False,
            "rl_used_for_distillation": False,
        },
    }


def structural_exactness_proof(student, teacher) -> dict:
    """Verify the algebraic conditions that make the residual exactly zero."""
    teacher_state = teacher.state_dict()
    student_state = student.state_dict()
    mismatched_base_tensors = []
    for name, value in teacher_state.items():
        target = "puffer_policy.base." + name[len("puffer_policy.") :]
        if target not in student_state or not torch.equal(
            student_state[target], value
        ):
            mismatched_base_tensors.append(name)
    adapter_decoder = {
        name: value
        for name, value in student_state.items()
        if name.startswith("puffer_policy.adapter.decoder.")
    }
    nonzero_adapter_decoder_tensors = [
        name
        for name, value in adapter_decoder.items()
        if torch.count_nonzero(value).item() != 0
    ]
    passed = (
        not mismatched_base_tensors
        and bool(adapter_decoder)
        and not nonzero_adapter_decoder_tensors
    )
    return {
        "passed": passed,
        "teacher_base_tensor_count": len(teacher_state),
        "bit_exact_base_tensor_count": (
            len(teacher_state) - len(mismatched_base_tensors)
        ),
        "mismatched_base_tensors": mismatched_base_tensors,
        "adapter_decoder_tensor_count": len(adapter_decoder),
        "adapter_decoder_parameters": sum(
            value.numel() for value in adapter_decoder.values()
        ),
        "nonzero_adapter_decoder_tensors": nonzero_adapter_decoder_tensors,
        "reasoning": (
            "The base branch is tensor-identical to the teacher and the "
            "parallel branch contributes exactly zero logits and value. "
            "Therefore their sum equals the teacher for every finite input."
        ),
    }


def main() -> None:
    args = parse_args()
    if min(
        args.adapter_hidden_size,
        args.adapter_layers,
        args.validation_games,
        args.shard_games,
        args.reload_games,
        args.workers,
    ) < 1:
        raise ValueError("dimensions, games, and workers must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.validation_seed)
    np.random.seed(args.validation_seed)
    torch.manual_seed(args.validation_seed)

    teacher_path = args.teacher.resolve()
    teacher, teacher_payload = load_temporal_policy_checkpoint(
        teacher_path, device="cpu"
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    teacher_model = policy_model_config(teacher)
    student = MinGRUResidualTemporalPolicy(
        base_hidden_size=int(teacher_model["hidden_size"]),
        base_num_layers=int(teacher_model["num_layers"]),
        adapter_hidden_size=args.adapter_hidden_size,
        adapter_num_layers=args.adapter_layers,
        mode_conditioned_heads=bool(teacher_model["mode_conditioned_heads"]),
        action_head=str(teacher_model["action_head"]),
    )
    initialization = initialize_min_gru_residual(student, teacher)
    structural_proof = structural_exactness_proof(student, teacher)
    if not structural_proof["passed"]:
        raise RuntimeError(
            "function-preserving structural proof failed: "
            + json.dumps(structural_proof, sort_keys=True)
        )
    teacher_parameters = sum(parameter.numel() for parameter in teacher.parameters())
    student_parameters = sum(parameter.numel() for parameter in student.parameters())
    print(
        json.dumps(
            {
                "event": "initialized",
                "teacher_model": teacher_model,
                "student_model": policy_model_config(student),
                "teacher_parameters": teacher_parameters,
                "student_parameters": student_parameters,
                "parameter_ratio": student_parameters / teacher_parameters,
                "initialization": initialization,
                "structural_exactness_proof": structural_proof,
            }
        ),
        flush=True,
    )

    with VectorPufferRolloutPool(
        teacher,
        workers=args.workers,
        champion_path=args.champion,
        historical_paths=args.historical,
    ) as pool:
        aggregate, shards, validation_seconds = validate_sharded(
            pool,
            student=student,
            teacher=teacher,
            games=args.validation_games,
            shard_games=args.shard_games,
            seed=args.validation_seed,
            minibatch_trajectories=args.minibatch_trajectories,
        )
    exact = (
        aggregate["action_agreement"] == 1.0
        and aggregate["forward_kl"] == 0.0
        and aggregate["value_mse"] == 0.0
        and aggregate["max_legal_logit_error"] == 0.0
    )
    if not exact:
        raise RuntimeError(
            "student passed actions but failed exact logits/value gate: "
            + json.dumps(aggregate, sort_keys=True)
        )

    validation_report = {
        "games": args.validation_games,
        "seed_start": args.validation_seed,
        "opponent_schedule": list(OPPONENTS),
        "player_count_schedule": list(PLAYER_COUNTS),
        "metrics": aggregate,
        "shards": shards,
        "wall_seconds": validation_seconds,
        "exact_action_logit_and_value_match": exact,
    }
    checkpoint = args.output_dir / "exact_wider_deeper_gru.pt"
    atomic_torch(
        checkpoint_payload(
            student,
            teacher_path,
            teacher_payload,
            {
                "method": "exact_base_plus_zero_wider_deeper_min_gru_residual",
                "initialization": initialization,
                "structural_exactness_proof": structural_proof,
                "heldout": validation_report,
            },
        ),
        checkpoint,
    )

    # Reload through the production checkpoint path and verify an entirely
    # separate seed range so serialization cannot masquerade as success.
    reloaded, reloaded_payload = load_temporal_policy_checkpoint(checkpoint)
    reloaded.eval()
    with VectorPufferRolloutPool(
        teacher,
        workers=args.workers,
        champion_path=args.champion,
        historical_paths=args.historical,
    ) as pool:
        reload_metrics, reload_shards, reload_seconds = validate_sharded(
            pool,
            student=reloaded,
            teacher=teacher,
            games=args.reload_games,
            shard_games=args.shard_games,
            seed=args.reload_seed,
            minibatch_trajectories=args.minibatch_trajectories,
        )
    reload_exact = (
        reload_metrics["action_agreement"] == 1.0
        and reload_metrics["forward_kl"] == 0.0
        and reload_metrics["value_mse"] == 0.0
        and reload_metrics["max_legal_logit_error"] == 0.0
    )
    if not reload_exact:
        raise RuntimeError("reloaded exact student failed the independent gate")

    result = {
        "schema_version": 1,
        "kind": "exact_wider_deeper_min_gru_distillation",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "teacher": {
            "path": str(teacher_path),
            "sha256": sha256(teacher_path),
            "epoch": int(teacher_payload.get("epoch", 0)),
            "model": teacher_model,
            "parameters": teacher_parameters,
        },
        "student": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256(checkpoint),
            "model": policy_model_config(reloaded),
            "parameters": student_parameters,
            "parameter_ratio": student_parameters / teacher_parameters,
            "state_elements": sum(
                state.numel() for state in reloaded.initial_state(1)
            ),
        },
        "initialization": initialization,
        "structural_exactness_proof": structural_proof,
        "heldout_validation": validation_report,
        "independent_reload_validation": {
            "checkpoint_format": reloaded_payload.get("format"),
            "games": args.reload_games,
            "seed_start": args.reload_seed,
            "metrics": reload_metrics,
            "shards": reload_shards,
            "wall_seconds": reload_seconds,
            "exact_action_logit_and_value_match": reload_exact,
        },
        "distillation_updates": {
            "kind": "supervised function-preserving initialization",
            "rl_used": False,
            "reward_signal_used": False,
        },
    }
    atomic_json(result, args.output_dir / "result.json")
    print(json.dumps({"event": "complete", "result": result}), flush=True)


if __name__ == "__main__":
    main()
