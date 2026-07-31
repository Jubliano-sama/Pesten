#!/usr/bin/env python3
"""Bounded screen for robust aggregation of the 24-view suit orbit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from batched_pesten import ActorPolicy
from batched_suit_symmetry import BatchedSuitAveragedActor
from benchmark_checkpoint_batched import file_sha256, run_batched_specs
from benchmark_selective_suit_projection import (
    CHAMPION_KEY,
    DEFAULT_POOLS,
    build_specs,
    summarize_evaluation,
    validate_seed_ranges,
)
from hybrid_agent import load_frozen_ppo_actor
from robust_suit_aggregation import (
    ROBUST_AGGREGATIONS,
    RobustSuitAggregationActor,
)


MEAN_LOGITS_KEY = "mean_logits"
DEFAULT_SCREEN_SEED = 86_500_001
DEFAULT_CONFIRMATION_SEED = 88_500_001


def _actors(
    champion: torch.nn.Module,
    device: torch.device,
    robust_methods: tuple[str, ...] = ROBUST_AGGREGATIONS,
) -> tuple[dict[str, torch.nn.Module], dict[str, RobustSuitAggregationActor]]:
    robust = {
        method: RobustSuitAggregationActor(
            champion,
            aggregation=method,
            base_chunk_size=512,
        ).to(device).eval()
        for method in robust_methods
    }
    actors: dict[str, torch.nn.Module] = {
        CHAMPION_KEY: champion,
        MEAN_LOGITS_KEY: BatchedSuitAveragedActor(
            champion,
            alpha=1.0,
            aggregation="logits",
            player_orbit_mode="none",
            base_chunk_size=512,
        ).to(device).eval(),
        **robust,
    }
    return actors, robust


def _policies(
    actors: dict[str, torch.nn.Module],
    device: torch.device,
) -> dict[str, ActorPolicy]:
    return {
        key: ActorPolicy(
            actor,
            device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        )
        for key, actor in actors.items()
    }


def methods_passing_screen(screen_summary: dict) -> list[str]:
    """Frozen gate: beat mean pool, no champion or direct mean regression."""
    mean_pool = screen_summary["pool_comparisons"][MEAN_LOGITS_KEY][
        "combined_score"
    ]
    mean_vs_champion = screen_summary["direct"][
        f"{MEAN_LOGITS_KEY}_vs_{CHAMPION_KEY}"
    ]["evaluated_score"]
    passed = []
    for method in ROBUST_AGGREGATIONS:
        pool_score = screen_summary["pool_comparisons"][method][
            "combined_score"
        ]
        versus_champion = screen_summary["direct"][
            f"{method}_vs_{CHAMPION_KEY}"
        ]["evaluated_score"]
        versus_mean = screen_summary["direct"][
            f"{method}_vs_{MEAN_LOGITS_KEY}"
        ]["evaluated_score"]
        if (
            pool_score > mean_pool
            and versus_champion >= mean_vs_champion
            and versus_mean >= 0.5
        ):
            passed.append(method)
    return passed


def run_screen_and_optional_confirmation(
    *,
    champion_path: Path,
    device: torch.device,
    screen_pairs: int,
    confirmation_pairs: int,
    screen_seed: int,
    confirmation_seed: int,
    pool_names: tuple[str, ...] = DEFAULT_POOLS,
    max_turns: int = 1_000,
) -> dict:
    validate_seed_ranges(
        grid_seed=screen_seed,
        grid_pairs=screen_pairs,
        confirmation_seed=confirmation_seed,
        confirmation_pairs=confirmation_pairs,
        pool_count=len(pool_names),
    )
    champion, _ = load_frozen_ppo_actor(champion_path, device)
    actors, robust = _actors(champion, device)
    keys = (*ROBUST_AGGREGATIONS, MEAN_LOGITS_KEY, CHAMPION_KEY)
    direct_matchups = (
        ("majority_vote", CHAMPION_KEY),
        ("mean_borda", CHAMPION_KEY),
        (MEAN_LOGITS_KEY, CHAMPION_KEY),
        ("majority_vote", MEAN_LOGITS_KEY),
        ("mean_borda", MEAN_LOGITS_KEY),
    )
    specs = build_specs(
        base_seed=screen_seed,
        pairs=screen_pairs,
        direct_matchups=direct_matchups,
        pool_policy_keys=keys,
        pool_names=pool_names,
    )
    outcomes, throughput = run_batched_specs(
        specs, _policies(actors, device), max_turns=max_turns
    )
    screen = summarize_evaluation(
        specs,
        outcomes,
        pairs=screen_pairs,
        direct_matchups=direct_matchups,
        pool_policy_keys=keys,
        pool_names=pool_names,
        bootstrap_seed=screen_seed + 500_000,
    )
    screen["robust_statistics"] = {
        method: actor.statistics() for method, actor in robust.items()
    }
    screen["throughput"] = throughput
    passed = methods_passing_screen(screen)

    confirmation = None
    selected_method = None
    if passed:
        selected_method = max(
            passed,
            key=lambda method: (
                screen["pool_comparisons"][method]["combined_score"],
                screen["direct"][f"{method}_vs_{CHAMPION_KEY}"][
                    "evaluated_score"
                ],
                method,
            ),
        )
        confirmation_actors, confirmation_robust = _actors(
            champion, device, (selected_method,)
        )
        confirmation_keys = (
            selected_method,
            MEAN_LOGITS_KEY,
            CHAMPION_KEY,
        )
        confirmation_direct = (
            (selected_method, CHAMPION_KEY),
            (MEAN_LOGITS_KEY, CHAMPION_KEY),
            (selected_method, MEAN_LOGITS_KEY),
        )
        confirmation_specs = build_specs(
            base_seed=confirmation_seed,
            pairs=confirmation_pairs,
            direct_matchups=confirmation_direct,
            pool_policy_keys=confirmation_keys,
            pool_names=pool_names,
        )
        confirmation_outcomes, confirmation_throughput = run_batched_specs(
            confirmation_specs,
            _policies(confirmation_actors, device),
            max_turns=max_turns,
        )
        confirmation = summarize_evaluation(
            confirmation_specs,
            confirmation_outcomes,
            pairs=confirmation_pairs,
            direct_matchups=confirmation_direct,
            pool_policy_keys=confirmation_keys,
            pool_names=pool_names,
            bootstrap_seed=confirmation_seed + 500_000,
        )
        confirmation["robust_statistics"] = {
            selected_method: confirmation_robust[
                selected_method
            ].statistics()
        }
        confirmation["throughput"] = confirmation_throughput

    return {
        "schema_version": 1,
        "benchmark": "robust_full_suit_orbit_aggregation",
        "development_only": True,
        "champion": {
            "path": str(champion_path),
            "sha256": file_sha256(champion_path),
        },
        "privacy_boundary": (
            "current 121-feature own-hand/public observation and current "
            "24-view logits only"
        ),
        "predeclared_methods": list(ROBUST_AGGREGATIONS),
        "screen_gate": {
            "pool": "strictly higher combined score than mean logits",
            "champion_head": (
                "score versus champion at least mean-logit score"
            ),
            "mean_head": "at least 50 percent directly versus mean logits",
        },
        "seed_audit": {
            "strategy": "explicit consecutive common-deal ranges",
            "screen": {
                "base_seed": screen_seed,
                "pairs": screen_pairs,
            },
            "confirmation": {
                "base_seed": confirmation_seed,
                "pairs": confirmation_pairs,
                "used": confirmation is not None,
            },
            "ranges_disjoint": True,
            "all_below_90m": True,
        },
        "pools": list(pool_names),
        "methods_passing_screen": passed,
        "selected_method": selected_method,
        "screen": screen,
        "confirmation": confirmation,
        "confirmation_skip_reason": (
            None
            if confirmation is not None
            else "no robust method passed all predeclared screen gates"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion", type=Path, default=Path("ppo_actor_100run3.pth")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--screen-pairs", type=int, default=30)
    parser.add_argument("--confirmation-pairs", type=int, default=60)
    parser.add_argument("--screen-seed", type=int, default=DEFAULT_SCREEN_SEED)
    parser.add_argument(
        "--confirmation-seed",
        type=int,
        default=DEFAULT_CONFIRMATION_SEED,
    )
    parser.add_argument("--pool", default=",".join(DEFAULT_POOLS))
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    payload = run_screen_and_optional_confirmation(
        champion_path=args.champion,
        device=torch.device(args.device),
        screen_pairs=args.screen_pairs,
        confirmation_pairs=args.confirmation_pairs,
        screen_seed=args.screen_seed,
        confirmation_seed=args.confirmation_seed,
        pool_names=tuple(
            name.strip() for name in args.pool.split(",") if name.strip()
        ),
        max_turns=args.max_turns,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
