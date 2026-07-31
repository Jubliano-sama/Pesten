#!/usr/bin/env python3
"""Small common-development screen for privacy-safe temporal ensembles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_model_zoo import DEFAULT_TEACHER
from benchmark_residual_checkpoint import (
    deterministic_evaluation_agent_factory,
    deterministic_evaluation_runtime,
    evaluate_candidate_and_base_paired,
    stratified_paired_bootstrap_interval,
)
from temporal_ensemble import COMBINATION_MODES, TemporalRecurrentEnsemble


SEALED_SEED_START = 92_000_000
DEFAULT_DEVELOPMENT_SEED = 81_000_001
DEFAULT_MEMBERS = {
    "base": Path(
        "experiments/residual_gru_seed57301/residual_gru_final.pt"
    ),
    "aggressive": Path(
        "experiments/gru_aggressive_seed67301/gru_aggressive_final.pt"
    ),
    "lstm": Path(
        "experiments/lstm_aggressive_seed77301/lstm_aggressive_final.pt"
    ),
    "large": Path(
        "experiments/gru_large_seed87301/gru_large_final.pt"
    ),
}


def parse_member_specification(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError("member must use NAME=CHECKPOINT syntax")
    name, raw_path = specification.split("=", 1)
    if not name or not raw_path:
        raise ValueError("member must use non-empty NAME=CHECKPOINT syntax")
    return name, Path(raw_path)


def _evaluation_dependencies(champion_path: Path, device: str):
    from randomAgent import AgentRandom
    from smartAgent import Agent as SmartAgent, SmarterAgent
    from benchmark_agents import load_ppo_agent
    from hardcodedAgent import HardcodedStrategyAgent

    def tree_factory(game_instance):
        return HardcodedStrategyAgent(
            game_instance,
            belief_samples=8,
            search_depth=3,
            beam_width=3,
        )

    return {
        "ppo_champion": load_ppo_agent(champion_path, device),
        "bayesian_tree": tree_factory,
        "smart": SmartAgent,
        "smarter": SmarterAgent,
        "random": AgentRandom,
    }


def run_ensemble_screen(
    *,
    members: dict[str, Path],
    combinations: tuple[str, ...],
    champion_path: Path = DEFAULT_TEACHER,
    pairs: int = 10,
    seed: int = DEFAULT_DEVELOPMENT_SEED,
    device: str = "cpu",
    cpu_threads: int = 1,
    opponent_names: tuple[str, ...] | None = None,
    evaluate_pair=None,
    evaluate_comparison=None,
    opponents=None,
    base_factory=None,
) -> dict:
    if not members:
        raise ValueError("at least one ensemble member is required")
    if not combinations:
        raise ValueError("at least one combination mode is required")
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if any(mode not in COMBINATION_MODES for mode in combinations):
        raise ValueError("unknown ensemble combination mode")
    if seed < 0 or seed >= SEALED_SEED_START:
        raise ValueError("ensemble screen cannot use sealed final seeds")
    if opponents is None:
        opponents = _evaluation_dependencies(champion_path, device)
    if opponent_names is not None:
        unknown = set(opponent_names) - set(opponents)
        if unknown:
            raise ValueError(
                f"unknown opponents requested: {sorted(unknown)}"
            )
        opponents = {
            name: opponents[name] for name in opponent_names
        }
    if not opponents:
        raise ValueError("at least one opponent is required")
    if evaluate_comparison is None:
        if evaluate_pair is not None:
            raise ValueError(
                "custom evaluate_pair requires evaluate_comparison so "
                "same-deal candidate-minus-base samples remain available"
            )
        evaluate_comparison = evaluate_candidate_and_base_paired
    if base_factory is None:
        base_factory = opponents.get("ppo_champion")
    if base_factory is None:
        # A filtered pool still compares against the exact frozen PPO base.
        base_factory = _evaluation_dependencies(
            champion_path, device
        )["ppo_champion"]
    # Pool offsets and paired bootstrap offsets must also stay development-only.
    largest_seed = max(
        seed + 990_000,
        seed + max(0, len(opponents) - 1) * 100_000 + 190_000,
    )
    if largest_seed >= SEALED_SEED_START:
        raise ValueError("ensemble screen offsets reach sealed final seeds")

    screens = {}
    member_names = tuple(members)
    checkpoint_paths = tuple(members.values())
    member_metadata = None
    with deterministic_evaluation_runtime(
        device, cpu_threads=cpu_threads
    ):
        for combination in combinations:
            ensemble = TemporalRecurrentEnsemble.from_checkpoints(
                checkpoint_paths,
                champion_path=champion_path,
                device=device,
                combination=combination,
                member_names=member_names,
            ).eval()
            if member_metadata is None:
                member_metadata = {
                    name: metadata
                    for name, metadata in zip(
                        member_names, ensemble.checkpoint_metadata
                    )
                }
            candidate = deterministic_evaluation_agent_factory(
                ensemble, seed=seed
            )
            started = time.perf_counter()
            comparison_blocks = {}
            for index, (name, opponent) in enumerate(opponents.items()):
                opponent_pairs = (
                    pairs
                    if name == "ppo_champion"
                    else max(20, pairs // 2)
                )
                comparison_blocks[name] = evaluate_comparison(
                    candidate,
                    base_factory,
                    opponent,
                    pairs=opponent_pairs,
                    seed=seed + index * 100_000,
                    candidate_name=(
                        f"{ensemble.architecture}:{combination}"
                    ),
                    opponent_name=name,
                )
            results = {
                name: block["candidate_result"]
                for name, block in comparison_blocks.items()
            }
            base_results = {
                name: block["base_result"]
                for name, block in comparison_blocks.items()
            }
            paired_deltas = {
                name: block["comparison"]
                for name, block in comparison_blocks.items()
            }
            delta_groups = {
                name: block["_pair_deltas"]
                for name, block in comparison_blocks.items()
            }
            candidate_scores = [
                score
                for block in comparison_blocks.values()
                for score in block["_candidate_pair_scores"]
            ]
            base_scores = [
                score
                for block in comparison_blocks.values()
                for score in block["_base_pair_scores"]
            ]
            all_deltas = [
                delta
                for values in delta_groups.values()
                for delta in values
            ]
            combined_ci = stratified_paired_bootstrap_interval(
                delta_groups, seed=seed + 990_000
            )
            screens[combination] = {
                "results": results,
                "base_results": base_results,
                "paired_candidate_minus_base": paired_deltas,
                "combined_pool_delta": {
                    "opponents": list(opponents),
                    "pairs": len(all_deltas),
                    "weighting": "one vote per evaluated paired deal",
                    "bootstrap": "stratified within opponent",
                    "same_deals_and_rotated_seats": True,
                    "candidate_score": float(
                        np.mean(candidate_scores)
                    ),
                    "base_score": float(np.mean(base_scores)),
                    "candidate_minus_base_score": float(
                        np.mean(all_deltas)
                    ),
                    "paired_bootstrap_ci95": list(combined_ci),
                    "delta_gate_passed": combined_ci[0] > 0.0,
                    "seed": seed + 990_000,
                },
                "elapsed_seconds": time.perf_counter() - started,
            }

    return {
        "schema_version": 1,
        "scope": "small common-development temporal ensemble screen",
        "development_seed": seed,
        "sealed_seed_start": SEALED_SEED_START,
        "pairs_per_opponent": pairs,
        "opponents": list(opponents),
        "paired_deals_and_rotated_seats": True,
        "evaluation_runtime": {
            "deterministic_fast_path": True,
            "records_training_trajectories": False,
            "cpu_threads": (
                cpu_threads if torch.device(device).type == "cpu" else None
            ),
        },
        "information_boundary": (
            "DecisionInput plus redacted PublicEventToken history only; "
            "one independent recurrent state per member"
        ),
        "champion_path": str(champion_path),
        "members": {
            name: str(path) for name, path in members.items()
        },
        "member_metadata": member_metadata,
        "combinations": screens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--member",
        action="append",
        default=[],
        help="NAME=CHECKPOINT; repeat for every member",
    )
    parser.add_argument(
        "--combination",
        nargs="+",
        choices=COMBINATION_MODES,
        default=list(COMBINATION_MODES),
    )
    parser.add_argument("--champion", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--pairs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_DEVELOPMENT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="temporary PyTorch intra-op thread count for CPU evaluation",
    )
    parser.add_argument(
        "--opponents",
        nargs="+",
        choices=(
            "ppo_champion",
            "bayesian_tree",
            "smart",
            "smarter",
            "random",
        ),
        help="optional pool subset; defaults to all opponents",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    members = (
        dict(parse_member_specification(spec) for spec in args.member)
        if args.member
        else DEFAULT_MEMBERS
    )
    payload = run_ensemble_screen(
        members=members,
        combinations=tuple(args.combination),
        champion_path=args.champion,
        pairs=args.pairs,
        seed=args.seed,
        device=args.device,
        cpu_threads=args.cpu_threads,
        opponent_names=(
            tuple(args.opponents) if args.opponents else None
        ),
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
