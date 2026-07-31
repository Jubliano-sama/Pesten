#!/usr/bin/env python3
"""Grid-screen and disjoint confirmation for selective suit projection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from batched_pesten import (
    ActorPolicy,
    POLICY_RANDOM,
    POLICY_RL,
    POLICY_SMART,
    POLICY_SMARTER,
)
from batched_suit_symmetry import BatchedSuitAveragedActor
from benchmark_checkpoint_batched import (
    BatchGameOutcome,
    BatchGameSpec,
    file_sha256,
    run_batched_specs,
)
from hybrid_agent import load_frozen_ppo_actor
from selective_suit_projection import (
    PREDECLARED_RULE_GRID,
    SelectiveSuitProjectionActor,
    SelectiveSuitRule,
)


DEVELOPMENT_SEED_LIMIT = 90_000_000
DEFAULT_GRID_SEED = 86_000_001
DEFAULT_CONFIRMATION_SEED = 88_000_001
DEFAULT_POOLS = ("random", "smart", "smarter")
POOL_KINDS = {
    "random": POLICY_RANDOM,
    "smart": POLICY_SMART,
    "smarter": POLICY_SMARTER,
}
CHAMPION_KEY = "champion"
ALWAYS_SUIT_KEY = "always_suit"


def validate_seed_ranges(
    *,
    grid_seed: int,
    grid_pairs: int,
    confirmation_seed: int,
    confirmation_pairs: int,
    pool_count: int,
) -> None:
    if grid_pairs < 1 or confirmation_pairs < 1:
        raise ValueError("pair counts must be positive")
    grid_end = grid_seed + 100_000 + pool_count * 10_000 + grid_pairs - 1
    confirmation_end = (
        confirmation_seed
        + 100_000
        + pool_count * 10_000
        + confirmation_pairs
        - 1
    )
    if min(grid_seed, confirmation_seed) < 0:
        raise ValueError("seeds cannot be negative")
    if max(grid_end, confirmation_end) >= DEVELOPMENT_SEED_LIMIT:
        raise ValueError("all development seeds must remain below 90M")
    if not (grid_end < confirmation_seed or confirmation_end < grid_seed):
        raise ValueError("grid and confirmation seed ranges must be disjoint")


def build_specs(
    *,
    base_seed: int,
    pairs: int,
    direct_matchups: Sequence[tuple[str, str]],
    pool_policy_keys: Sequence[str],
    pool_names: Sequence[str],
) -> list[BatchGameSpec]:
    """Use common consecutive deals within one explicit evaluation range."""
    unknown = tuple(name for name in pool_names if name not in POOL_KINDS)
    if unknown:
        raise ValueError(f"unknown pools: {unknown}")
    specs = []
    for evaluated_key, opponent_key in direct_matchups:
        for pair_index in range(pairs):
            deal_seed = base_seed + pair_index
            for evaluated_seat in (0, 1):
                specs.append(
                    BatchGameSpec(
                        section="direct",
                        opponent=opponent_key,
                        pair_index=pair_index,
                        evaluated_key=evaluated_key,
                        evaluated_seat=evaluated_seat,
                        deal_seed=deal_seed,
                        policy_kinds=(POLICY_RL, POLICY_RL),
                        actor_keys=(
                            (evaluated_key, opponent_key)
                            if evaluated_seat == 0
                            else (opponent_key, evaluated_key)
                        ),
                    )
                )
    for pool_index, pool_name in enumerate(pool_names):
        opponent_kind = POOL_KINDS[pool_name]
        for pair_index in range(pairs):
            deal_seed = (
                base_seed + 100_000 + pool_index * 10_000 + pair_index
            )
            for evaluated_key in pool_policy_keys:
                for evaluated_seat in (0, 1):
                    specs.append(
                        BatchGameSpec(
                            section="pool",
                            opponent=pool_name,
                            pair_index=pair_index,
                            evaluated_key=evaluated_key,
                            evaluated_seat=evaluated_seat,
                            deal_seed=deal_seed,
                            policy_kinds=(
                                (POLICY_RL, opponent_kind)
                                if evaluated_seat == 0
                                else (opponent_kind, POLICY_RL)
                            ),
                            actor_keys=(
                                (evaluated_key, None)
                                if evaluated_seat == 0
                                else (None, evaluated_key)
                            ),
                        )
                    )
    return specs


def _mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = 5_000,
) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _pair_scores(
    specs: Sequence[BatchGameSpec],
    outcomes: Sequence[BatchGameOutcome],
    *,
    section: str,
    evaluated_key: str,
    opponent: str,
    pairs: int,
) -> list[float]:
    indexed = tuple(zip(specs, outcomes))
    result = []
    for pair_index in range(pairs):
        selected = [
            outcome.score
            for spec, outcome in indexed
            if spec.section == section
            and spec.evaluated_key == evaluated_key
            and spec.opponent == opponent
            and spec.pair_index == pair_index
        ]
        if len(selected) != 2:
            raise RuntimeError("expected exactly one rotated seat pair")
        result.append(float(np.mean(selected)))
    return result


def summarize_evaluation(
    specs: Sequence[BatchGameSpec],
    outcomes: Sequence[BatchGameOutcome],
    *,
    pairs: int,
    direct_matchups: Sequence[tuple[str, str]],
    pool_policy_keys: Sequence[str],
    pool_names: Sequence[str],
    bootstrap_seed: int,
) -> dict:
    direct = {}
    for matchup_index, (evaluated_key, opponent_key) in enumerate(
        direct_matchups
    ):
        scores = _pair_scores(
            specs,
            outcomes,
            section="direct",
            evaluated_key=evaluated_key,
            opponent=opponent_key,
            pairs=pairs,
        )
        direct[f"{evaluated_key}_vs_{opponent_key}"] = {
            "evaluated_score": float(np.mean(scores)),
            "evaluated_advantage": float(2.0 * np.mean(scores) - 1.0),
            "score_ci95": _mean_ci(
                scores, seed=bootstrap_seed + matchup_index
            ),
        }

    pool_scores: dict[str, dict[str, list[float]]] = {}
    for policy_key in pool_policy_keys:
        pool_scores[policy_key] = {}
        for pool_name in pool_names:
            pool_scores[policy_key][pool_name] = _pair_scores(
                specs,
                outcomes,
                section="pool",
                evaluated_key=policy_key,
                opponent=pool_name,
                pairs=pairs,
            )
    pool_comparisons = {}
    champion_all = []
    for pool_name in pool_names:
        champion_all.extend(pool_scores[CHAMPION_KEY][pool_name])
    for policy_index, policy_key in enumerate(pool_policy_keys):
        per_pool = []
        all_deltas = []
        for pool_name in pool_names:
            candidate = pool_scores[policy_key][pool_name]
            champion = pool_scores[CHAMPION_KEY][pool_name]
            deltas = [
                candidate_score - champion_score
                for candidate_score, champion_score in zip(
                    candidate, champion
                )
            ]
            all_deltas.extend(deltas)
            per_pool.append(
                {
                    "opponent": pool_name,
                    "score": float(np.mean(candidate)),
                    "champion_score": float(np.mean(champion)),
                    "candidate_minus_champion": float(np.mean(deltas)),
                }
            )
        pool_comparisons[policy_key] = {
            "per_pool": per_pool,
            "combined_score": float(
                np.mean(
                    [
                        value
                        for values in pool_scores[policy_key].values()
                        for value in values
                    ]
                )
            ),
            "combined_candidate_minus_champion": float(
                np.mean(all_deltas)
            ),
            "combined_delta_ci95": _mean_ci(
                all_deltas,
                seed=bootstrap_seed + 10_000 + policy_index,
            ),
        }
    return {
        "direct": direct,
        "pool_comparisons": pool_comparisons,
        "champion_combined_pool_score": float(np.mean(champion_all)),
    }


def _make_actors(
    champion: torch.nn.Module,
    rules: Sequence[SelectiveSuitRule],
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, SelectiveSuitProjectionActor]]:
    selectors = {
        rule.name: SelectiveSuitProjectionActor(
            champion, rule, base_chunk_size=512
        ).to(device).eval()
        for rule in rules
    }
    actors: dict[str, torch.nn.Module] = {
        CHAMPION_KEY: champion,
        ALWAYS_SUIT_KEY: BatchedSuitAveragedActor(
            champion,
            alpha=1.0,
            aggregation="logits",
            player_orbit_mode="none",
            base_chunk_size=512,
        ).to(device).eval(),
        **selectors,
    }
    return actors, selectors


def _actor_policies(
    actors: Mapping[str, torch.nn.Module],
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


def select_top_rule(grid_summary: dict) -> str:
    """Predeclared selection: pool delta, then direct advantage, then name."""
    rule_names = tuple(rule.name for rule in PREDECLARED_RULE_GRID)
    return max(
        rule_names,
        key=lambda name: (
            grid_summary["pool_comparisons"][name][
                "combined_candidate_minus_champion"
            ],
            grid_summary["direct"][f"{name}_vs_{CHAMPION_KEY}"][
                "evaluated_advantage"
            ],
            name,
        ),
    )


def run_grid_and_confirmation(
    *,
    champion_path: Path,
    device: torch.device,
    grid_pairs: int,
    confirmation_pairs: int,
    grid_seed: int,
    confirmation_seed: int,
    pool_names: tuple[str, ...] = DEFAULT_POOLS,
    max_turns: int = 1_000,
) -> dict:
    validate_seed_ranges(
        grid_seed=grid_seed,
        grid_pairs=grid_pairs,
        confirmation_seed=confirmation_seed,
        confirmation_pairs=confirmation_pairs,
        pool_count=len(pool_names),
    )
    champion, _ = load_frozen_ppo_actor(champion_path, device)
    grid_actors, grid_selectors = _make_actors(
        champion, PREDECLARED_RULE_GRID, device
    )
    grid_keys = (
        *(rule.name for rule in PREDECLARED_RULE_GRID),
        ALWAYS_SUIT_KEY,
        CHAMPION_KEY,
    )
    grid_direct = tuple(
        (key, CHAMPION_KEY)
        for key in grid_keys
        if key != CHAMPION_KEY
    )
    grid_specs = build_specs(
        base_seed=grid_seed,
        pairs=grid_pairs,
        direct_matchups=grid_direct,
        pool_policy_keys=grid_keys,
        pool_names=pool_names,
    )
    grid_outcomes, grid_throughput = run_batched_specs(
        grid_specs,
        _actor_policies(grid_actors, device),
        max_turns=max_turns,
    )
    grid_summary = summarize_evaluation(
        grid_specs,
        grid_outcomes,
        pairs=grid_pairs,
        direct_matchups=grid_direct,
        pool_policy_keys=grid_keys,
        pool_names=pool_names,
        bootstrap_seed=grid_seed + 500_000,
    )
    grid_summary["selector_statistics"] = {
        name: selector.statistics()
        for name, selector in grid_selectors.items()
    }
    grid_summary["throughput"] = grid_throughput
    top_rule_name = select_top_rule(grid_summary)
    top_rule = next(
        rule
        for rule in PREDECLARED_RULE_GRID
        if rule.name == top_rule_name
    )

    # Fresh wrapper and counters: only the frozen rule definition crosses the
    # development/confirmation boundary.
    confirmation_actors, confirmation_selectors = _make_actors(
        champion, (top_rule,), device
    )
    confirmation_keys = (
        top_rule_name,
        ALWAYS_SUIT_KEY,
        CHAMPION_KEY,
    )
    confirmation_direct = (
        (top_rule_name, CHAMPION_KEY),
        (top_rule_name, ALWAYS_SUIT_KEY),
        (ALWAYS_SUIT_KEY, CHAMPION_KEY),
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
        _actor_policies(confirmation_actors, device),
        max_turns=max_turns,
    )
    confirmation_summary = summarize_evaluation(
        confirmation_specs,
        confirmation_outcomes,
        pairs=confirmation_pairs,
        direct_matchups=confirmation_direct,
        pool_policy_keys=confirmation_keys,
        pool_names=pool_names,
        bootstrap_seed=confirmation_seed + 500_000,
    )
    confirmation_summary["selector_statistics"] = {
        top_rule_name: confirmation_selectors[
            top_rule_name
        ].statistics()
    }
    confirmation_summary["throughput"] = confirmation_throughput

    return {
        "schema_version": 1,
        "benchmark": "selective_full_suit_projection",
        "development_only": True,
        "champion": {
            "path": str(champion_path),
            "sha256": file_sha256(champion_path),
        },
        "privacy_boundary": (
            "current 121-feature own-hand/public observation and current "
            "champion/orbit logits only"
        ),
        "grid_contract": {
            "rules_frozen_before_evaluation": [
                rule.to_dict() for rule in PREDECLARED_RULE_GRID
            ],
            "selection_rule": (
                "highest combined fixed-pool delta versus champion, then "
                "head-to-head advantage, then stable rule name"
            ),
            "no_seed_or_opponent_input": True,
        },
        "seed_audit": {
            "strategy": "explicit consecutive common-deal ranges",
            "grid": {
                "base_seed": grid_seed,
                "pairs": grid_pairs,
            },
            "confirmation": {
                "base_seed": confirmation_seed,
                "pairs": confirmation_pairs,
            },
            "ranges_disjoint": True,
            "all_below_90m": True,
        },
        "pools": list(pool_names),
        "selected_rule": top_rule.to_dict(),
        "grid": grid_summary,
        "confirmation": confirmation_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion", type=Path, default=Path("ppo_actor_100run3.pth")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--grid-pairs", type=int, default=20)
    parser.add_argument("--confirmation-pairs", type=int, default=60)
    parser.add_argument("--grid-seed", type=int, default=DEFAULT_GRID_SEED)
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
    payload = run_grid_and_confirmation(
        champion_path=args.champion,
        device=torch.device(args.device),
        grid_pairs=args.grid_pairs,
        confirmation_pairs=args.confirmation_pairs,
        grid_seed=args.grid_seed,
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
