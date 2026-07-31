"""Accepted random-initialized, reward-only temporal population training.

This module intentionally has no import path to the deployed PPO actor,
imitation datasets, teacher logits, or PPO-derived diagnostics.  Its only
learning signals are environment rewards, advantages, and value targets from
current-policy self-play, PFSP-prioritized temporal history, and fixed
rule/tree diversity anchors.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import random
import statistics
import time
from typing import Sequence

import numpy as np
import pufferlib
import torch

import game
from hardcodedAgent import HardcodedStrategyAgent
from randomAgent import AgentRandom
from smartAgent import Agent as SmartAgent
from temporal_batched_speed import (
    BatchRolloutStats,
    collect_league_batch_batched,
)
from temporal_puffer_agent import temporal_agent_factory
from temporal_puffer_v4 import (
    NativePufferTemporalPolicy,
    PUFFERLIB_REVISION,
)
from temporal_training import (
    ppo_update_batched_epochs as clipped_reward_policy_update,
)


ACCEPTED_SCHEDULE = (
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "historical_temporal",
    "historical_temporal",
    "bayesian_tree",
    "smart_rule",
)
NON_PPO_SELECTION_WEIGHTS = {
    "bayesian_tree": 0.35,
    "smart_rule": 0.30,
    "random": 0.15,
    "historical_worst": 0.20,
}
PROHIBITED_TRAINING_SOURCES = (
    "deployed PPO actor checkpoint",
    "PPO actions or logits",
    "PPO trajectories or rewards",
    "imitation or behavior cloning",
    "distillation or action-agreement loss",
)


def tree_factory(game_instance):
    return HardcodedStrategyAgent(
        game_instance,
        belief_samples=8,
        search_depth=3,
        beam_width=3,
    )


def _state_hash(policy: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(policy.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(policy: NativePufferTemporalPolicy):
    frozen = NativePufferTemporalPolicy(
        hidden_size=policy.hidden_size,
        num_layers=policy.num_layers,
        mode_conditioned_heads=policy.mode_conditioned_heads,
    )
    frozen.load_state_dict(copy.deepcopy(policy.state_dict()))
    frozen.eval()
    for parameter in frozen.parameters():
        parameter.requires_grad_(False)
    return frozen


def synchronize_cpu_shadow(
    source: NativePufferTemporalPolicy,
    shadow: NativePufferTemporalPolicy,
) -> None:
    """Copy update weights to the inference-only CPU policy in place."""
    source_state = {
        name: tensor.detach().cpu()
        for name, tensor in source.state_dict().items()
    }
    shadow.load_state_dict(source_state)
    shadow.eval()


def paired_bootstrap_interval(
    pair_scores: Sequence[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(pair_scores, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_paired(
    candidate_factory,
    opponent_factory,
    *,
    pairs: int,
    seed: int,
    candidate_name: str,
    opponent_name: str,
) -> dict:
    wins = losses = draws = turns = 0
    seat_wins = [0, 0]
    pair_scores = []
    started = time.perf_counter()
    for pair in range(pairs):
        deal_seed = seed + pair
        scores = []
        for candidate_seat in (0, 1):
            np.random.seed(deal_seed)
            random.seed(deal_seed)
            factories = (
                [candidate_factory, opponent_factory]
                if candidate_seat == 0
                else [opponent_factory, candidate_factory]
            )
            match = game.Game(factories)
            game_turns, winner = match.auto_simulate()
            turns += game_turns
            if winner is None:
                draws += 1
                scores.append(0.5)
            elif winner == candidate_seat:
                wins += 1
                seat_wins[candidate_seat] += 1
                scores.append(1.0)
            else:
                losses += 1
                scores.append(0.0)
        pair_scores.append(sum(scores) / 2)
    score = (wins + 0.5 * draws) / (2 * pairs)
    return {
        "candidate": candidate_name,
        "opponent": opponent_name,
        "pairs": pairs,
        "games": 2 * pairs,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": score,
        "paired_bootstrap_ci95": paired_bootstrap_interval(
            pair_scores, seed=seed + 99_991
        ),
        "wins_as_seat_0": seat_wins[0],
        "wins_as_seat_1": seat_wins[1],
        "average_turns": turns / (2 * pairs),
        "elapsed_seconds": time.perf_counter() - started,
        "seed": seed,
    }


def pfsp_weights(win_rates: Sequence[float]) -> tuple[float, ...]:
    """Prioritize historical opponents the current agent struggles against."""
    raw = [max(0.05, 1.0 - rate) ** 2 for rate in win_rates]
    total = sum(raw)
    return tuple(value / total for value in raw)


def prioritized_population(
    historical: Sequence[NativePufferTemporalPolicy],
    weights: Sequence[float],
    *,
    slots: int = 32,
) -> list[NativePufferTemporalPolicy]:
    if len(historical) != len(weights) or not historical:
        raise ValueError("PFSP history and weights must be nonempty and aligned")
    counts = [max(1, int(round(weight * slots))) for weight in weights]
    while sum(counts) > slots:
        index = max(range(len(counts)), key=lambda item: counts[item])
        if counts[index] == 1:
            break
        counts[index] -= 1
    while sum(counts) < slots:
        index = max(
            range(len(counts)),
            key=lambda item: weights[item] / counts[item],
        )
        counts[index] += 1
    return [
        historical[index]
        for index, count in enumerate(counts)
        for _ in range(count)
    ]


def evaluate_history_for_pfsp(
    policy: NativePufferTemporalPolicy,
    historical: Sequence[NativePufferTemporalPolicy],
    *,
    pairs: int,
    seed: int,
) -> tuple[list[float], tuple[float, ...]]:
    candidate = temporal_agent_factory(
        policy, deterministic=True, seed=seed
    )
    scores = []
    for index, opponent in enumerate(historical):
        evaluation = evaluate_paired(
            candidate,
            temporal_agent_factory(
                opponent, deterministic=True, seed=seed + index + 1
            ),
            pairs=pairs,
            seed=seed + index * 10_000,
            candidate_name="current_temporal",
            opponent_name=f"historical_{index}",
        )
        scores.append(evaluation["score"])
    return scores, pfsp_weights(scores)


def evaluate_non_ppo_pool(
    policy: NativePufferTemporalPolicy,
    historical: Sequence[NativePufferTemporalPolicy],
    *,
    pairs: int,
    seed: int,
) -> dict:
    candidate = temporal_agent_factory(
        policy, deterministic=True, seed=seed
    )
    pool = {
        "bayesian_tree": evaluate_paired(
            candidate,
            tree_factory,
            pairs=pairs,
            seed=seed,
            candidate_name="temporal_reward_only",
            opponent_name="bayesian_tree",
        ),
        "smart_rule": evaluate_paired(
            candidate,
            SmartAgent,
            pairs=pairs,
            seed=seed + 100_000,
            candidate_name="temporal_reward_only",
            opponent_name="smart_rule",
        ),
        "random": evaluate_paired(
            candidate,
            AgentRandom,
            pairs=pairs,
            seed=seed + 200_000,
            candidate_name="temporal_reward_only",
            opponent_name="random",
        ),
    }
    historical_scores = []
    for index, opponent in enumerate(historical[-4:]):
        result = evaluate_paired(
            candidate,
            temporal_agent_factory(
                opponent, deterministic=True, seed=seed + 300_000 + index
            ),
            pairs=pairs,
            seed=seed + 300_000 + index * 10_000,
            candidate_name="temporal_reward_only",
            opponent_name=f"historical_recent_{index}",
        )
        historical_scores.append(result)
    historical_worst = min(
        (item["score"] for item in historical_scores),
        default=0.5,
    )
    component_scores = {
        "bayesian_tree": pool["bayesian_tree"]["score"],
        "smart_rule": pool["smart_rule"]["score"],
        "random": pool["random"]["score"],
        "historical_worst": historical_worst,
    }
    selection_score = sum(
        NON_PPO_SELECTION_WEIGHTS[name] * component_scores[name]
        for name in NON_PPO_SELECTION_WEIGHTS
    )
    return {
        "pool": pool,
        "historical_recent": historical_scores,
        "historical_worst_score": historical_worst,
        "selection_components": component_scores,
        "selection_weights": NON_PPO_SELECTION_WEIGHTS,
        "selection_score": selection_score,
        "contains_ppo_benchmark": False,
    }


def save_reward_checkpoint(
    path: Path,
    policy: NativePufferTemporalPolicy,
    *,
    initialization_seed: int,
    initial_state_sha256: str,
    recipe: dict,
    epoch: int,
) -> dict:
    payload = {
        "format": "pesten-temporal-reward-only-v1",
        "observation_schema": "PlayerPrivateObservation-v1",
        "lineage": {
            "initialization": "torch_random_initialization",
            "initialization_seed": initialization_seed,
            "initial_state_sha256": initial_state_sha256,
            "parent_checkpoint": None,
            "imitation_checkpoint": None,
            "teacher_checkpoint": None,
            "ppo_training_opponent": False,
            "prohibited_training_sources_used": False,
        },
        "training": {
            "objective": (
                "environment reward, generalized advantage, clipped policy "
                "objective, entropy, and nonprivileged value target"
            ),
            "epoch": epoch,
            "recipe": recipe,
        },
        "state_dict": policy.state_dict(),
    }
    torch.save(payload, path)
    return {
        "path": str(path),
        "sha256": _file_hash(path),
        "lineage": payload["lineage"],
    }


def train_reward_population(
    *,
    seed: int,
    device: str,
    epochs: int,
    games_per_epoch: int,
    development_pairs: int,
    pfsp_pairs: int,
    hidden_size: int,
    num_layers: int,
    mode_conditioned_heads: bool,
    learning_rate: float,
    update_epochs: int,
    minibatch_trajectories: int,
    gamma: float,
    gae_lambda: float,
    clip_ratio: float,
    target_kl: float,
    entropy_coefficient: float,
    value_coefficient: float,
    output_dir: Path,
    schedule: Sequence[str] = ACCEPTED_SCHEDULE,
    pfsp_difficulty_prioritized: bool = True,
) -> dict:
    schedule = tuple(schedule)
    if "ppo_champion" in schedule:
        raise ValueError("accepted reward-only schedule cannot contain PPO")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    policy = NativePufferTemporalPolicy(
        hidden_size=hidden_size,
        num_layers=num_layers,
        mode_conditioned_heads=mode_conditioned_heads,
    ).to(device)
    initial_state_sha256 = _state_hash(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    initial_snapshot = snapshot(policy)
    # Keep environment rollout and scalar evaluation off CUDA.  Deep-copying
    # the already-created initial snapshot preserves the historical sequence
    # of global RNG consumption (constructing a second model would not).
    rollout_policy = copy.deepcopy(initial_snapshot)
    historical = [initial_snapshot]
    history_manifests = [
        {
            "history_index": 0,
            "source_epoch": 0,
            "state_sha256": initial_state_sha256,
        }
    ]
    recipe = {
        "schedule": list(schedule),
        "mixture": {
            name: schedule.count(name) / len(schedule)
            for name in sorted(set(schedule))
        },
        "learning_rate": learning_rate,
        "model": {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "mode_conditioned_heads": mode_conditioned_heads,
            "recurrent_context": "full episode",
        },
        "update_epochs": update_epochs,
        "minibatch_trajectories": minibatch_trajectories,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "clip_ratio": clip_ratio,
        "target_kl": target_kl,
        "entropy_coefficient": entropy_coefficient,
        "value_coefficient": value_coefficient,
        "execution_devices": {
            "update": str(next(policy.parameters()).device),
            "rollout": str(next(rollout_policy.parameters()).device),
            "evaluation": str(next(rollout_policy.parameters()).device),
        },
        "pfsp": {
            "weight": (
                "(max(0.05, 1 - current_win_rate))^2"
                if pfsp_difficulty_prioritized
                else "uniform historical sampling"
            ),
            "population_slots": 32,
            "evaluation_pairs_per_history": pfsp_pairs,
        },
        "prohibited_training_sources": list(PROHIBITED_TRAINING_SOURCES),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    curve = []
    checkpoints = []
    best = None
    total_decisions = 0
    for epoch in range(epochs):
        pfsp_history_manifest = copy.deepcopy(history_manifests)
        pfsp_scores, difficulty_weights = evaluate_history_for_pfsp(
            rollout_policy,
            historical,
            pairs=pfsp_pairs,
            seed=seed + 1_000_000 + epoch * 100_000,
        )
        weights = (
            difficulty_weights
            if pfsp_difficulty_prioritized
            else tuple(1.0 / len(historical) for _ in historical)
        )
        prioritized = prioritized_population(historical, weights)
        stats = BatchRolloutStats()
        epoch_started = time.perf_counter()
        episodes, contribution = collect_league_batch_batched(
            rollout_policy,
            None,
            prioritized,
            games=games_per_epoch,
            seed=seed + 2_000_000 + epoch * games_per_epoch,
            schedule_offset=epoch * games_per_epoch,
            schedule=schedule,
            stats=stats,
        )
        if "ppo_champion" in contribution["games"]:
            raise RuntimeError("PPO appeared in reward-only contribution")
        update = clipped_reward_policy_update(
            policy,
            episodes,
            optimizer,
            update_epochs=update_epochs,
            minibatch_trajectories=minibatch_trajectories,
            seed=seed + 3_000_000 + epoch,
            target_kl=target_kl,
            clip_ratio=clip_ratio,
            value_coefficient=value_coefficient,
            entropy_coefficient=entropy_coefficient,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=True,
        )
        epoch_decisions = sum(
            len(trajectory) for trajectory, _ in episodes
        )
        total_decisions += epoch_decisions
        frozen = snapshot(policy)
        sync_started = time.perf_counter()
        # ``frozen`` already contains the exact post-update CPU state, avoiding
        # a second device-to-host transfer solely for shadow synchronization.
        synchronize_cpu_shadow(frozen, rollout_policy)
        cpu_shadow_sync_seconds = time.perf_counter() - sync_started
        historical.append(frozen)
        history_manifests.append(
            {
                "history_index": len(history_manifests),
                "source_epoch": epoch + 1,
                "state_sha256": _state_hash(frozen),
            }
        )
        if len(historical) > 8:
            historical.pop(0)
            history_manifests.pop(0)

        checkpoint_path = output_dir / f"reward_seed{seed}_epoch{epoch + 1}.pt"
        checkpoint = save_reward_checkpoint(
            checkpoint_path,
            policy,
            initialization_seed=seed,
            initial_state_sha256=initial_state_sha256,
            recipe=recipe,
            epoch=epoch + 1,
        )
        checkpoints.append(checkpoint)
        evaluation = evaluate_non_ppo_pool(
            rollout_policy,
            historical[:-1],
            pairs=development_pairs,
            seed=seed + 4_000_000,
        )
        row = {
            "epoch": epoch + 1,
            "wall_seconds": time.perf_counter() - epoch_started,
            "cumulative_wall_seconds": time.perf_counter() - started,
            "epoch_decisions": epoch_decisions,
            "cumulative_decisions": total_decisions,
            "contribution": contribution,
            "batch_rollout_stats": {
                name: getattr(stats, name)
                for name in stats.__dataclass_fields__
            },
            "execution_devices": {
                "update": str(next(policy.parameters()).device),
                "rollout": str(next(rollout_policy.parameters()).device),
                "evaluation": str(next(rollout_policy.parameters()).device),
                "cpu_shadow_sync_seconds": cpu_shadow_sync_seconds,
            },
            "pfsp": {
                "history_win_rates": pfsp_scores,
                "history_weights": list(weights),
                "history_manifest": pfsp_history_manifest,
            },
            "reward_update": update,
            "development_evaluation": evaluation,
            "checkpoint": checkpoint,
        }
        curve.append(row)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "epoch": epoch + 1,
                    "decisions": epoch_decisions,
                    "selection_score": evaluation["selection_score"],
                    "policy_loss": update["policy_loss"],
                    "value_loss": update["value_loss"],
                    "entropy": update["entropy"],
                    "post_update_kl": update[
                        "post_update_diagnostics"
                    ]["approx_kl"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if best is None or evaluation["selection_score"] > best[
            "selection_score"
        ]:
            best = {
                "epoch": epoch + 1,
                "selection_score": evaluation["selection_score"],
                "checkpoint": checkpoint,
                "development_evaluation": evaluation,
            }

    assert best is not None
    manifest = {
        "format": "pesten-temporal-accepted-manifest-v1",
        "accepted_training_path": "random_initialized_reward_only",
        "random_initialization": {
            "seed": seed,
            "initial_state_sha256": initial_state_sha256,
            "parent_checkpoint": None,
        },
        "ppo_or_imitation_used_in_training": False,
        "recipe": recipe,
        "checkpoint_selection": {
            "development_pool_contains_ppo": False,
            "best_not_automatically_last": True,
            "weights": NON_PPO_SELECTION_WEIGHTS,
        },
        "best_checkpoint": best,
    }
    manifest_path = output_dir / f"reward_seed{seed}_accepted_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "seed": seed,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pufferlib": str(pufferlib.__version__),
            "pufferlib_revision": PUFFERLIB_REVISION,
            "device": device,
            "update_device": str(next(policy.parameters()).device),
            "rollout_device": str(
                next(rollout_policy.parameters()).device
            ),
            "evaluation_device": str(
                next(rollout_policy.parameters()).device
            ),
        },
        "lineage": manifest["random_initialization"],
        "recipe": recipe,
        "epochs": curve,
        "best_checkpoint": best,
        "accepted_manifest": {
            "path": str(manifest_path),
            "sha256": _file_hash(manifest_path),
        },
        "total_environment_decisions": total_decisions,
        "wall_seconds": time.perf_counter() - started,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="5401")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--games-per-epoch", type=int, default=100)
    parser.add_argument("--development-pairs", type=int, default=20)
    parser.add_argument("--pfsp-pairs", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--mode-conditioned-heads", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-trajectories", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--run-name", default="accepted_default")
    parser.add_argument(
        "--schedule",
        default=",".join(ACCEPTED_SCHEDULE),
        help=(
            "explicit reward-only opponent schedule; PPO entries are rejected"
        ),
    )
    parser.add_argument(
        "--uniform-history",
        action="store_true",
        help="ablate PFSP with uniform historical sampling",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/temporal/"
            "puffer4_temporal_reward_only_development_20260731.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(","))
    schedule = tuple(args.schedule.split(","))
    output_dir = (
        args.output.parent
        / "reward_only_checkpoints"
        / args.run_name
    )
    runs = [
        train_reward_population(
            seed=seed,
            device=args.device,
            epochs=args.epochs,
            games_per_epoch=args.games_per_epoch,
            development_pairs=args.development_pairs,
            pfsp_pairs=args.pfsp_pairs,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            mode_conditioned_heads=args.mode_conditioned_heads,
            learning_rate=args.learning_rate,
            update_epochs=args.update_epochs,
            minibatch_trajectories=args.minibatch_trajectories,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_ratio=args.clip_ratio,
            target_kl=args.target_kl,
            entropy_coefficient=args.entropy_coefficient,
            value_coefficient=args.value_coefficient,
            output_dir=output_dir / f"seed{seed}",
            schedule=schedule,
            pfsp_difficulty_prioritized=not args.uniform_history,
        )
        for seed in seeds
    ]
    payload = {
        "schema_version": 1,
        "purpose": (
            "accepted development-only random-initialized reward-only "
            "population league"
        ),
        "ppo_benchmark_used": False,
        "ppo_used_for_training_or_selection": False,
        "runs": runs,
        "aggregate": {
            "seeds": list(seeds),
            "best_development_scores": [
                run["best_checkpoint"]["selection_score"] for run in runs
            ],
            "mean_best_development_score": statistics.mean(
                run["best_checkpoint"]["selection_score"] for run in runs
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
