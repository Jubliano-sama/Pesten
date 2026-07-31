"""Development-only temporal checkpoint strength and disagreement audit.

The artifact produced here contains aggregate counters only.  It never
serializes a hand, a decision tensor, an event stream, or recurrent state.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch
from torch.distributions import Categorical

from benchmark_agents import load_ppo_agent
from ppo2 import StudentNet
from randomAgent import AgentRandom
from smartAgent import Agent as SmartAgent
from temporal_league import (
    CHAMPION_PATH,
    _batched_imitation_logits,
    collect_teacher_episodes,
    evaluate_paired,
    tree_factory,
)
from temporal_observation import EventKind
from temporal_puffer_agent import temporal_agent_factory
from temporal_puffer_v4 import NativePufferTemporalPolicy


SELECTION_WEIGHTS = {
    "ppo_champion": 0.5,
    "bayesian_tree": 0.2,
    "smart_rule": 0.2,
    "random": 0.1,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_policy(path: Path) -> NativePufferTemporalPolicy:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    model = payload.get("model", {})
    hidden_size = int(
        model.get(
            "hidden_size",
            state["puffer_policy.encoder.encoder.weight"].shape[0],
        )
    )
    num_layers = int(
        model.get(
            "num_layers",
            sum(
                key.startswith("puffer_policy.network.layers.")
                and key.endswith(".weight")
                for key in state
            ),
        )
    )
    mode_conditioned_heads = bool(
        model.get(
            "mode_conditioned_heads",
            any("card_and_pass" in key for key in state),
        )
    )
    policy = NativePufferTemporalPolicy(
        hidden_size=hidden_size,
        num_layers=num_layers,
        mode_conditioned_heads=mode_conditioned_heads,
    )
    policy.load_state_dict(state)
    policy.eval()
    return policy


def _checkpoint_paths(directory: Path, seeds: tuple[int, ...]) -> list[dict]:
    checkpoints = []
    for seed in seeds:
        imitation = directory / f"temporal_native_seed{seed}_imitation.pt"
        if imitation.is_file():
            checkpoints.append(
                {
                    "training_seed": seed,
                    "stage": "imitation",
                    "epoch": 0,
                    "path": imitation,
                }
            )
        for epoch in range(1, 1000):
            path = (
                directory
                / f"temporal_native_seed{seed}_league{epoch}.pt"
            )
            if not path.is_file():
                break
            checkpoints.append(
                {
                    "training_seed": seed,
                    "stage": "league",
                    "epoch": epoch,
                    "path": path,
                }
            )
    if not checkpoints:
        raise FileNotFoundError(f"no temporal checkpoints in {directory}")
    return checkpoints


def evaluate_checkpoint(
    checkpoint: dict,
    *,
    pairs: int,
    development_seed: int,
    champion_factory,
) -> dict:
    policy = _load_policy(checkpoint["path"])
    candidate = temporal_agent_factory(
        policy,
        deterministic=True,
        seed=development_seed,
    )
    opponents = {
        "ppo_champion": champion_factory,
        "bayesian_tree": tree_factory,
        "smart_rule": SmartAgent,
        "random": AgentRandom,
    }
    evaluations = {}
    for offset, (name, factory) in enumerate(opponents.items()):
        evaluations[name] = evaluate_paired(
            candidate,
            factory,
            pairs=pairs,
            seed=development_seed + offset * 100_000,
            candidate_name=(
                f"temporal_seed{checkpoint['training_seed']}_"
                f"{checkpoint['stage']}{checkpoint['epoch']}"
            ),
            opponent_name=name,
        )
    selection_score = sum(
        SELECTION_WEIGHTS[name] * evaluations[name]["score"]
        for name in SELECTION_WEIGHTS
    )
    return {
        "training_seed": checkpoint["training_seed"],
        "stage": checkpoint["stage"],
        "epoch": checkpoint["epoch"],
        "checkpoint": str(checkpoint["path"]),
        "checkpoint_sha256": _sha256(checkpoint["path"]),
        "development_seed": development_seed,
        "pairs_per_opponent": pairs,
        "pool": evaluations,
        "selection_score": selection_score,
    }


def _bucket(
    counters: dict[str, dict[str, list[float]]],
    category: str,
    label: str,
    *,
    correct: bool,
    cross_entropy: float,
) -> None:
    values = counters[category].setdefault(label, [0.0, 0.0, 0.0])
    values[0] += 1
    values[1] += int(correct)
    values[2] += cross_entropy


def _finalize_buckets(
    counters: dict[str, dict[str, list[float]]],
) -> dict:
    return {
        category: {
            label: {
                "samples": int(values[0]),
                "agreements": int(values[1]),
                "accuracy": values[1] / values[0],
                "mean_cross_entropy": values[2] / values[0],
            }
            for label, values in sorted(groups.items())
        }
        for category, groups in sorted(counters.items())
    }


def disagreement_audit(
    checkpoint: Path,
    *,
    games: int,
    development_seed: int,
) -> dict:
    policy = _load_policy(checkpoint)
    teacher = StudentNet()
    teacher.load_state_dict(
        torch.load(CHAMPION_PATH, map_location="cpu", weights_only=True)
    )
    teacher.eval()
    episodes = collect_teacher_episodes(
        policy,
        teacher,
        games=games,
        seed=development_seed,
    )

    counters: dict[str, dict[str, list[float]]] = defaultdict(dict)
    confusion = Counter()
    sequential_logits = []
    teacher_actions = []
    total = correct = illegal_candidate = illegal_teacher = 0
    reset_nonzero = 0
    cumulative_events = 0
    with torch.no_grad():
        for episode in episodes:
            state = policy.initial_state()
            reset_nonzero += int(
                any(bool(torch.count_nonzero(part)) for part in state)
            )
            episode_event_count = 0
            for decision_index, sample in enumerate(episode):
                result = policy.step(sample.decision, state)
                state = tuple(part.detach() for part in result.state)
                candidate_action = int(torch.argmax(result.logits).item())
                teacher_action = int(sample.action)
                legal = sample.decision.legal_actions
                illegal_candidate += int(not legal[candidate_action])
                illegal_teacher += int(not legal[teacher_action])
                distribution = Categorical(logits=result.logits)
                cross_entropy = float(
                    -distribution.log_prob(
                        torch.tensor(
                            teacher_action,
                            device=result.logits.device,
                        )
                    ).item()
                )
                agreed = candidate_action == teacher_action
                total += 1
                correct += int(agreed)
                confusion[(teacher_action, candidate_action)] += 1
                sequential_logits.append(result.logits.detach().cpu())
                teacher_actions.append(teacher_action)

                values = sample.decision.observation.values
                legal_count = sum(legal)
                hand_size = int(values[91])
                player_count = int(values[92])
                opponent_sizes = values[70 : 69 + player_count]
                opponent_minimum = min(opponent_sizes)
                after_own_draw = any(
                    event.kind is EventKind.DRAW_COUNT
                    and event.actor_relative == 0
                    for event in sample.decision.public_events
                )
                episode_event_count += len(
                    sample.decision.public_events
                )
                cumulative_events += len(sample.decision.public_events)

                _bucket(
                    counters,
                    "legal_action_count",
                    str(legal_count) if legal_count < 4 else "4+",
                    correct=agreed,
                    cross_entropy=cross_entropy,
                )
                _bucket(
                    counters,
                    "decision_mode",
                    "choose_suit" if values[93] else "play_card",
                    correct=agreed,
                    cross_entropy=cross_entropy,
                )
                _bucket(
                    counters,
                    "penalty",
                    "active" if values[90] > 0 else "none",
                    correct=agreed,
                    cross_entropy=cross_entropy,
                )
                _bucket(
                    counters,
                    "own_last_card",
                    "yes" if hand_size == 1 else "no",
                    correct=agreed,
                    cross_entropy=cross_entropy,
                )
                _bucket(
                    counters,
                    "after_own_draw_retry",
                    "yes" if after_own_draw else "no",
                    correct=agreed,
                    cross_entropy=cross_entropy,
                )
                _bucket(
                    counters,
                    "direction",
                    "reverse" if values[89] < 0 else "forward",
                    correct=agreed,
                    cross_entropy=cross_entropy,
                )
                _bucket(
                    counters,
                    "own_hand_size",
                    (
                        "1"
                        if hand_size == 1
                        else "2-3"
                        if hand_size <= 3
                        else "4-6"
                        if hand_size <= 6
                        else "7+"
                    ),
                    correct=agreed,
                    cross_entropy=cross_entropy,
                )
                _bucket(
                    counters,
                    "opponent_endgame",
                    "min<=2" if opponent_minimum <= 2 else "min>2",
                    correct=agreed,
                    cross_entropy=cross_entropy,
                )
                _bucket(
                    counters,
                    "decision_sequence_index",
                    (
                        "0-4"
                        if decision_index < 5
                        else "5-14"
                        if decision_index < 15
                        else "15+"
                    ),
                    correct=agreed,
                    cross_entropy=cross_entropy,
                )
                _bucket(
                    counters,
                    "public_event_sequence_length",
                    (
                        "0-15"
                        if episode_event_count <= 15
                        else "16-40"
                        if episode_event_count <= 40
                        else "41+"
                    ),
                    correct=agreed,
                    cross_entropy=cross_entropy,
                )

    batched_logits = []
    with torch.no_grad():
        for start in range(0, len(episodes), 16):
            logits, _ = _batched_imitation_logits(
                policy, episodes[start : start + 16]
            )
            batched_logits.append(logits.detach().cpu())
    sequential = torch.stack(sequential_logits)
    batched = torch.cat(batched_logits)
    legal_coordinates = torch.isfinite(sequential) & torch.isfinite(batched)
    maximum_native_difference = float(
        (sequential[legal_coordinates] - batched[legal_coordinates])
        .abs()
        .max()
        .item()
    )
    batched_actions = batched.argmax(dim=-1)
    sequential_actions = sequential.argmax(dim=-1)
    top_confusions = [
        {
            "teacher_action": teacher_action,
            "candidate_action": candidate_action,
            "count": count,
        }
        for (teacher_action, candidate_action), count in sorted(
            (
                (actions, count)
                for actions, count in confusion.items()
                if actions[0] != actions[1]
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:20]
        if teacher_action != candidate_action
    ]
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "development_seed": development_seed,
        "games": games,
        "samples": total,
        "accuracy": correct / total,
        "mean_public_events_per_decision": cumulative_events / total,
        "privacy": {
            "aggregate_only": True,
            "raw_decisions_serialized": False,
            "raw_events_serialized": False,
            "recurrent_states_serialized": False,
        },
        "inference_contract": {
            "policy_eval_mode": not policy.training,
            "state_reset_per_episode": True,
            "nonzero_initial_states": reset_nonzero,
            "action_count": 55,
            "illegal_candidate_actions": illegal_candidate,
            "illegal_teacher_actions": illegal_teacher,
            "sequential_vs_native_batched_argmax_mismatches": int(
                (sequential_actions != batched_actions).sum().item()
            ),
            "sequential_vs_native_batched_max_legal_logit_abs_diff": (
                maximum_native_difference
            ),
            "teacher_target_count_matches": (
                len(teacher_actions) == batched.shape[0]
            ),
        },
        "buckets": _finalize_buckets(counters),
        "top_action_confusions": top_confusions,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("benchmarks/puffer4_temporal_checkpoints"),
    )
    parser.add_argument("--training-seeds", default="731,1731,2731")
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--development-seed", type=int, default=4_310_000)
    parser.add_argument("--diagnostic-games", type=int, default=90)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/temporal/"
            "puffer4_temporal_checkpoint_diagnostics_20260731.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(
        int(value) for value in args.training_seeds.split(",")
    )
    checkpoints = _checkpoint_paths(args.checkpoint_dir, seeds)
    champion_factory = load_ppo_agent(CHAMPION_PATH, "cpu")
    started = time.perf_counter()
    evaluations = [
        evaluate_checkpoint(
            checkpoint,
            pairs=args.pairs,
            development_seed=args.development_seed,
            champion_factory=champion_factory,
        )
        for checkpoint in checkpoints
    ]
    best = max(evaluations, key=lambda item: item["selection_score"])
    audit = disagreement_audit(
        Path(best["checkpoint"]),
        games=args.diagnostic_games,
        development_seed=args.development_seed + 800_000,
    )
    payload = {
        "schema_version": 1,
        "purpose": (
            "development-only checkpoint selection and aggregate "
            "disagreement diagnosis; not held-out final evidence"
        ),
        "development_seed": args.development_seed,
        "selection_weights": SELECTION_WEIGHTS,
        "checkpoint_count": len(evaluations),
        "evaluations": evaluations,
        "best_checkpoint": best,
        "best_checkpoint_disagreement_audit": audit,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if not math.isfinite(payload["elapsed_seconds"]):
        raise RuntimeError("non-finite diagnostic runtime")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "checkpoint_count": len(evaluations),
                "best_checkpoint": best["checkpoint"],
                "selection_score": best["selection_score"],
                "accuracy": audit["accuracy"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
