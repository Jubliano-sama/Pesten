#!/usr/bin/env python3
"""Warm-start the deployed MLP and distill a recurrent GRU policy into it.

The held-out split uses disjoint game seeds.  Besides ordinary imitation
metrics, the script groups exact MLP observations and reports contradictory
teacher actions: one such collision is enough to show that no deterministic
memoryless policy with this input contract can match the recurrent teacher on
every encountered history.
"""

from __future__ import annotations

import argparse
from collections import Counter
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_observation import ACTION_COUNT, PASS_ACTION
from benchmark_model_zoo import safe_batch_to_legacy_observations
from ppo2 import StudentNet, build_action_masks
from puffer_distillation import _forward_decisions
from seed_integrity import DEVELOPMENT_SEED_LIMIT
from temporal_puffer_v4 import load_temporal_policy_checkpoint
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


def atomic_torch(payload, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total < 1:
        return [math.nan, math.nan]
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def flatten_episode_decisions(episodes):
    return [
        step.decision
        for trajectory, _ in episodes
        for step in trajectory
    ]


@torch.inference_mode()
def tensorize_episodes(policy, episodes, *, chunk_trajectories: int = 192) -> dict:
    output = {
        "observations": [],
        "safe_observations": [],
        "legal_actions": [],
        "legacy_derived_legal_actions": [],
        "teacher_logits": [],
    }
    usable = [episode for episode in episodes if episode[0]]
    for start in range(0, len(usable), chunk_trajectories):
        batch = usable[start : start + chunk_trajectories]
        decisions = flatten_episode_decisions(batch)
        safe = torch.tensor(
            [decision.observation.values for decision in decisions],
            dtype=torch.float32,
        )
        legal = torch.tensor(
            [decision.legal_actions for decision in decisions],
            dtype=torch.bool,
        )
        legacy = safe_batch_to_legacy_observations(safe, legal)
        derived_legal = build_action_masks(legacy)
        # Draw-retry is intentionally not recoverable from the frozen 121
        # features: the environment must supply its exact external mask. Keep
        # both masks so this contract gap is measured instead of hidden.
        logits, _, forward_legal = _forward_decisions(policy, batch)
        if not torch.equal(forward_legal.cpu(), legal):
            raise RuntimeError("teacher forward order does not match decisions")
        logits = logits.cpu()
        # Center finite logits. This preserves distributions and argmax while
        # making persisted targets numerically compact and easy to inspect.
        finite = logits.masked_fill(~legal, float("-inf"))
        centered = finite - finite.max(dim=1, keepdim=True).values
        centered = centered.masked_fill(~legal, -1.0e9)
        output["observations"].append(legacy.cpu())
        output["safe_observations"].append(safe.cpu())
        output["legal_actions"].append(legal.cpu())
        output["legacy_derived_legal_actions"].append(derived_legal.cpu())
        output["teacher_logits"].append(centered.cpu())
    if not output["observations"]:
        raise RuntimeError("rollout produced no teacher decisions")
    return {name: torch.cat(parts, dim=0) for name, parts in output.items()}


def collect_dataset(
    pool,
    teacher,
    *,
    games: int,
    seed: int,
    batch_games: int,
    schedule=OPPONENTS,
    player_counts=PLAYER_COUNTS,
) -> tuple[dict, dict]:
    tensors: dict[str, list[torch.Tensor]] = {}
    metadata = {
        "games": Counter(),
        "decisions": Counter(),
        "player_counts": Counter(),
        "total_turns": 0,
        "trajectories": 0,
        "wall_seconds": 0.0,
    }
    started = time.perf_counter()
    for offset in range(0, games, batch_games):
        count = min(batch_games, games - offset)
        episodes, batch_meta = pool.collect(
            teacher,
            games=count,
            seed=seed + offset,
            schedule_offset=offset,
            schedule=schedule,
            player_counts=player_counts,
            prepack_training=False,
            max_turns=500,
        )
        batch = tensorize_episodes(teacher, episodes)
        for name, value in batch.items():
            tensors.setdefault(name, []).append(value)
        metadata["games"].update(batch_meta.get("games", {}))
        metadata["decisions"].update(batch_meta.get("decisions", {}))
        metadata["player_counts"].update(batch_meta.get("player_counts", {}))
        metadata["total_turns"] += int(batch_meta.get("total_turns", 0))
        metadata["trajectories"] += int(batch_meta.get("trajectories", 0))
        print(
            json.dumps(
                {
                    "event": "collection_progress",
                    "games_complete": offset + count,
                    "games_total": games,
                    "decisions_so_far": sum(
                        part.shape[0] for part in tensors["observations"]
                    ),
                }
            ),
            flush=True,
        )
    metadata["wall_seconds"] = time.perf_counter() - started
    normalized_metadata = {
        **metadata,
        "games": dict(metadata["games"]),
        "decisions": dict(metadata["decisions"]),
        "player_counts": dict(metadata["player_counts"]),
    }
    return (
        {name: torch.cat(parts, dim=0) for name, parts in tensors.items()},
        normalized_metadata,
    )


@torch.inference_mode()
def evaluate(model: StudentNet, data: dict, *, batch_size: int) -> dict:
    model.eval()
    samples = int(data["observations"].shape[0])
    correct = 0
    forward_kl = 0.0
    top2_margin_error = 0.0
    strategic_total = strategic_correct = 0
    suit_total = suit_correct = 0
    pass_total = pass_correct = 0
    for start in range(0, samples, batch_size):
        stop = min(start + batch_size, samples)
        observations = data["observations"][start:stop]
        legal = data["legal_actions"][start:stop]
        teacher_logits = data["teacher_logits"][start:stop]
        logits = model(observations).masked_fill(~legal, -1.0e9)
        teacher_log = torch.log_softmax(teacher_logits, dim=-1)
        student_log = torch.log_softmax(logits, dim=-1)
        teacher_prob = teacher_log.exp()
        teacher_actions = teacher_logits.argmax(dim=-1)
        student_actions = logits.argmax(dim=-1)
        matches = teacher_actions == student_actions
        correct += int(matches.sum().item())
        forward_kl += float(
            (teacher_prob * (teacher_log - student_log)).sum(dim=-1).sum().item()
        )
        teacher_top2 = teacher_logits.topk(k=2, dim=-1).values
        student_on_teacher = logits.gather(1, teacher_actions[:, None]).squeeze(1)
        student_competitor = logits.masked_fill(
            F.one_hot(teacher_actions, ACTION_COUNT).bool(), -1.0e9
        ).max(dim=-1).values
        top2_margin_error += float(
            (
                (student_on_teacher - student_competitor)
                - (teacher_top2[:, 0] - teacher_top2[:, 1])
            ).abs().sum().item()
        )
        strategic = legal[:, PASS_ACTION] & legal[:, :PASS_ACTION].any(dim=-1)
        suit = legal[:, 50:54].all(dim=-1) & ~legal[:, PASS_ACTION]
        teacher_pass = teacher_actions == PASS_ACTION
        strategic_total += int(strategic.sum().item())
        strategic_correct += int((matches & strategic).sum().item())
        suit_total += int(suit.sum().item())
        suit_correct += int((matches & suit).sum().item())
        pass_total += int(teacher_pass.sum().item())
        pass_correct += int((matches & teacher_pass).sum().item())
    return {
        "samples": samples,
        "action_agreement": correct / samples,
        "agreement_count": correct,
        "agreement_wilson_95": wilson_interval(correct, samples),
        "forward_kl": forward_kl / samples,
        "teacher_margin_mae": top2_margin_error / samples,
        "strategic_samples": strategic_total,
        "strategic_action_agreement": (
            strategic_correct / strategic_total if strategic_total else None
        ),
        "suit_samples": suit_total,
        "suit_action_agreement": suit_correct / suit_total if suit_total else None,
        "teacher_pass_samples": pass_total,
        "teacher_pass_recall": pass_correct / pass_total if pass_total else None,
    }


def collision_analysis(data: dict, *, key_kind: str, examples: int = 3) -> dict:
    observations = data[
        "observations" if key_kind == "mlp" else "safe_observations"
    ].numpy()
    # Both policies receive this mask at the environment boundary. The MLP
    # does not need to infer draw-retry legality from its frozen 121 features.
    observations = np.concatenate(
        (observations, data["legal_actions"].numpy().astype(np.float32)),
        axis=1,
    )
    observations = np.ascontiguousarray(observations, dtype=np.float32)
    actions = data["teacher_logits"].argmax(dim=-1).numpy().astype(np.int64)
    row_type = np.dtype((np.void, observations.dtype.itemsize * observations.shape[1]))
    keys = observations.view(row_type).reshape(-1)
    _, inverse, counts = np.unique(
        keys, return_inverse=True, return_counts=True
    )
    pairs = np.stack((inverse.astype(np.int64), actions), axis=1)
    unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
    group_max = np.zeros(len(counts), dtype=np.int64)
    np.maximum.at(group_max, unique_pairs[:, 0], pair_counts)
    conflict_groups = np.flatnonzero(group_max < counts)
    irreducible = int(np.sum(counts - group_max))
    sample_examples = []
    for group in conflict_groups[:examples]:
        indices = np.flatnonzero(inverse == group)
        action_counts = Counter(int(actions[index]) for index in indices)
        sample_examples.append(
            {
                "occurrences": int(len(indices)),
                "teacher_action_counts": dict(sorted(action_counts.items())),
                "row_indices": [int(value) for value in indices[:8]],
            }
        )
    return {
        "key": (
            "exact deployed MLP 121-float observation plus external legal mask"
            if key_kind == "mlp"
            else "exact GRU 94-float current observation plus 55-action legal mask"
        ),
        "samples": int(len(actions)),
        "unique_inputs": int(len(counts)),
        "repeated_input_groups": int(np.count_nonzero(counts > 1)),
        "conflicting_teacher_action_groups": int(len(conflict_groups)),
        "samples_in_conflicting_groups": int(counts[conflict_groups].sum()),
        "empirical_minimum_errors_for_any_deterministic_snapshot_policy": irreducible,
        "empirical_maximum_action_agreement": 1.0 - irreducible / len(actions),
        "examples": sample_examples,
    }


def train(
    model: StudentNet,
    train_data: dict,
    validation_data: dict,
    *,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[list[dict], dict]:
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1.0e-5
    )
    generator = torch.Generator().manual_seed(seed)
    samples = int(train_data["observations"].shape[0])
    history = []
    best = {"action_agreement": -1.0, "forward_kl": math.inf, "epoch": 0}
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = torch.randperm(samples, generator=generator)
        loss_sum = kl_sum = hard_sum = margin_sum = 0.0
        seen = 0
        started = time.perf_counter()
        for start in range(0, samples, batch_size):
            indices = permutation[start : start + batch_size]
            observations = train_data["observations"][indices]
            legal = train_data["legal_actions"][indices]
            teacher_logits = train_data["teacher_logits"][indices]
            teacher_actions = teacher_logits.argmax(dim=-1)
            logits = model(observations).masked_fill(~legal, -1.0e9)
            teacher_prob = torch.softmax(teacher_logits, dim=-1)
            log_prob = torch.log_softmax(logits, dim=-1)
            kl = F.kl_div(log_prob, teacher_prob, reduction="batchmean")
            hard = F.cross_entropy(logits, teacher_actions)
            selected = logits.gather(1, teacher_actions[:, None]).squeeze(1)
            competitor = logits.masked_fill(
                F.one_hot(teacher_actions, ACTION_COUNT).bool(), -1.0e9
            ).max(dim=-1).values
            margin = F.relu(0.15 - selected + competitor).mean()
            loss = kl + 0.35 * hard + 0.15 * margin
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            count = int(indices.numel())
            loss_sum += float(loss.detach()) * count
            kl_sum += float(kl.detach()) * count
            hard_sum += float(hard.detach()) * count
            margin_sum += float(margin.detach()) * count
            seen += count
        validation = evaluate(model, validation_data, batch_size=batch_size)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "train_forward_kl": kl_sum / seen,
            "train_hard_cross_entropy": hard_sum / seen,
            "train_margin_loss": margin_sum / seen,
            "epoch_seconds": time.perf_counter() - started,
            "validation": validation,
        }
        history.append(record)
        improved = (
            validation["action_agreement"] > best["action_agreement"]
            or (
                validation["action_agreement"] == best["action_agreement"]
                and validation["forward_kl"] < best["forward_kl"]
            )
        )
        if improved:
            best = {
                "action_agreement": validation["action_agreement"],
                "forward_kl": validation["forward_kl"],
                "epoch": epoch,
            }
            atomic_torch(model.state_dict(), output_dir / "best_mlp.pth")
        atomic_torch(model.state_dict(), output_dir / "latest_mlp.pth")
        atomic_json(history, output_dir / "training_history.json")
        print(json.dumps({"event": "epoch", **record}), flush=True)
        if validation["action_agreement"] == 1.0:
            break
    model.load_state_dict(
        torch.load(output_dir / "best_mlp.pth", map_location="cpu", weights_only=True)
    )
    return history, best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-games", type=int, default=4096)
    parser.add_argument("--validation-games", type=int, default=1024)
    parser.add_argument("--collection-batch-games", type=int, default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--train-seed", type=int, default=130_000_000)
    parser.add_argument("--validation-seed", type=int, default=131_000_000)
    parser.add_argument("--torch-threads", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.train_games, args.validation_games, args.workers, args.epochs) < 1:
        raise ValueError("game, worker, and epoch counts must be positive")
    if args.train_seed + args.train_games >= DEVELOPMENT_SEED_LIMIT:
        raise ValueError("training seeds overlap the sealed namespace")
    if args.validation_seed + args.validation_games >= DEVELOPMENT_SEED_LIMIT:
        raise ValueError("validation seeds overlap the sealed namespace")
    train_range = set(range(args.train_seed, args.train_seed + args.train_games))
    validation_range = set(
        range(args.validation_seed, args.validation_seed + args.validation_games)
    )
    if train_range.intersection(validation_range):
        raise ValueError("training and validation deal seeds overlap")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)

    teacher, teacher_payload = load_temporal_policy_checkpoint(
        args.teacher, device="cpu"
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    mlp = StudentNet()
    mlp.load_state_dict(
        torch.load(args.warm_start, map_location="cpu", weights_only=True)
    )
    initial_mlp = StudentNet()
    initial_mlp.load_state_dict(mlp.state_dict())

    print(
        json.dumps(
            {
                "event": "start",
                "teacher": str(args.teacher),
                "teacher_sha256": sha256(args.teacher),
                "teacher_epoch": teacher_payload.get("epoch"),
                "teacher_model": teacher_payload.get("model"),
                "warm_start": str(args.warm_start),
                "warm_start_sha256": sha256(args.warm_start),
            }
        ),
        flush=True,
    )

    with VectorPufferRolloutPool(
        teacher,
        workers=args.workers,
        champion_path=args.warm_start,
    ) as pool:
        train_data, train_meta = collect_dataset(
            pool,
            teacher,
            games=args.train_games,
            seed=args.train_seed,
            batch_games=args.collection_batch_games,
        )
        validation_data, validation_meta = collect_dataset(
            pool,
            teacher,
            games=args.validation_games,
            seed=args.validation_seed,
            batch_games=args.collection_batch_games,
        )

    atomic_torch(train_data, args.output_dir / "train_data.pt")
    atomic_torch(validation_data, args.output_dir / "validation_data.pt")
    baseline = evaluate(initial_mlp, validation_data, batch_size=args.batch_size)
    mlp_collisions = collision_analysis(validation_data, key_kind="mlp")
    safe_collisions = collision_analysis(validation_data, key_kind="safe")
    validation_mask_mismatches = int(
        (
            validation_data["legacy_derived_legal_actions"]
            != validation_data["legal_actions"]
        ).any(dim=1).sum().item()
    )
    print(
        json.dumps(
            {
                "event": "data_ready",
                "train_samples": int(train_data["observations"].shape[0]),
                "validation_samples": int(validation_data["observations"].shape[0]),
                "warm_start_validation": baseline,
                "legacy_mask_mismatch_samples": validation_mask_mismatches,
                "mlp_collisions": mlp_collisions,
                "safe_collisions": safe_collisions,
            }
        ),
        flush=True,
    )

    history, best = train(
        mlp,
        train_data,
        validation_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.train_seed + 777,
    )
    final_metrics = evaluate(mlp, validation_data, batch_size=args.batch_size)
    result = {
        "schema_version": 1,
        "experiment": "final_gru_to_deployed_mlp_distillation",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "teacher": {
            "path": str(args.teacher.resolve()),
            "sha256": sha256(args.teacher),
            "epoch": teacher_payload.get("epoch"),
            "model": teacher_payload.get("model"),
        },
        "student": {
            "architecture": "ppo2.StudentNet DenseSkipNet 8x256",
            "warm_start_path": str(args.warm_start.resolve()),
            "warm_start_sha256": sha256(args.warm_start),
            "best_checkpoint": str((args.output_dir / "best_mlp.pth").resolve()),
            "best_checkpoint_sha256": sha256(args.output_dir / "best_mlp.pth"),
            "parameters": sum(parameter.numel() for parameter in mlp.parameters()),
        },
        "data": {
            "opponent_schedule": list(OPPONENTS),
            "player_count_schedule": list(PLAYER_COUNTS),
            "train_seed_start": args.train_seed,
            "validation_seed_start": args.validation_seed,
            "disjoint_seed_splits": True,
            "train": train_meta,
            "validation": validation_meta,
            "train_samples": int(train_data["observations"].shape[0]),
            "validation_samples": int(validation_data["observations"].shape[0]),
            "validation_legacy_mask_mismatch_samples": validation_mask_mismatches,
        },
        "warm_start_validation": baseline,
        "best": best,
        "best_validation": final_metrics,
        "collision_analysis": {
            "mlp_input": mlp_collisions,
            "gru_current_snapshot": safe_collisions,
        },
        "training": {
            "epochs_completed": len(history),
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "objective": "forward KL + 0.35 hard CE + 0.15 top-action margin",
        },
        "perfect_on_held_out": final_metrics["action_agreement"] == 1.0,
    }
    atomic_json(result, args.output_dir / "result.json")
    print(json.dumps({"event": "complete", "result": result}), flush=True)


if __name__ == "__main__":
    main()
