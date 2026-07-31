#!/usr/bin/env python3
"""Development gameplay and speed screen for a suit-orbit student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Sequence

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
    BatchGameSpec,
    BatchGameOutcome,
    file_sha256,
    run_batched_specs,
)
from hybrid_agent import load_frozen_ppo_actor
from symmetry_distill import (
    DEVELOPMENT_SEED_LIMIT,
    collect_public_observations,
    validate_development_seed_range,
)


DEFAULT_SEED = 88_100_001
DEFAULT_POOLS = ("random", "smart", "smarter")
POOL_KINDS = {
    "random": POLICY_RANDOM,
    "smart": POLICY_SMART,
    "smarter": POLICY_SMARTER,
}


def build_linear_specs(
    *,
    pairs: int,
    base_seed: int,
    pool_names: Sequence[str] = DEFAULT_POOLS,
) -> list[BatchGameSpec]:
    """Create common-deal strata using explicit consecutive seed ranges."""
    if pairs < 1:
        raise ValueError("pairs must be positive")
    unknown = tuple(name for name in pool_names if name not in POOL_KINDS)
    if unknown:
        raise ValueError(f"unknown pools: {unknown}")
    highest_seed = base_seed + len(pool_names) * 100_000 + pairs - 1
    validate_development_seed_range(
        base_seed, highest_seed - base_seed + 1
    )
    specs = []
    for pair_index in range(pairs):
        deal_seed = base_seed + pair_index
        for candidate_seat in (0, 1):
            specs.append(
                BatchGameSpec(
                    section="direct",
                    opponent=None,
                    pair_index=pair_index,
                    evaluated_key="candidate",
                    evaluated_seat=candidate_seat,
                    deal_seed=deal_seed,
                    policy_kinds=(POLICY_RL, POLICY_RL),
                    actor_keys=(
                        ("candidate", "parent")
                        if candidate_seat == 0
                        else ("parent", "candidate")
                    ),
                )
            )
    for pool_index, pool_name in enumerate(pool_names, start=1):
        opponent_kind = POOL_KINDS[pool_name]
        for pair_index in range(pairs):
            deal_seed = base_seed + pool_index * 100_000 + pair_index
            for evaluated_key in ("candidate", "parent"):
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
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(samples, len(array)))
    means = array[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975)
    )


def _pair_score(outcomes: Sequence[BatchGameOutcome]) -> float:
    if len(outcomes) != 2:
        raise ValueError("rotated pair requires two outcomes")
    return float(np.mean([outcome.score for outcome in outcomes]))


def summarize_outcomes(
    specs: Sequence[BatchGameSpec],
    outcomes: Sequence[BatchGameOutcome],
    *,
    pairs: int,
    pool_names: Sequence[str],
    seed: int,
) -> dict:
    indexed = tuple(zip(specs, outcomes))
    direct_scores = []
    for pair_index in range(pairs):
        direct_scores.append(
            _pair_score(
                [
                    outcome
                    for spec, outcome in indexed
                    if spec.section == "direct"
                    and spec.pair_index == pair_index
                ]
            )
        )
    pools = []
    combined_deltas = []
    for pool_index, pool_name in enumerate(pool_names):
        candidate_scores = []
        parent_scores = []
        for pair_index in range(pairs):
            selected = [
                (spec, outcome)
                for spec, outcome in indexed
                if spec.section == "pool"
                and spec.opponent == pool_name
                and spec.pair_index == pair_index
            ]
            candidate_scores.append(
                _pair_score(
                    [
                        outcome
                        for spec, outcome in selected
                        if spec.evaluated_key == "candidate"
                    ]
                )
            )
            parent_scores.append(
                _pair_score(
                    [
                        outcome
                        for spec, outcome in selected
                        if spec.evaluated_key == "parent"
                    ]
                )
            )
        deltas = [
            candidate - parent
            for candidate, parent in zip(candidate_scores, parent_scores)
        ]
        combined_deltas.extend(deltas)
        pools.append(
            {
                "opponent": pool_name,
                "candidate_score": float(np.mean(candidate_scores)),
                "parent_score": float(np.mean(parent_scores)),
                "candidate_minus_parent": float(np.mean(deltas)),
                "delta_ci95": list(
                    _mean_ci(
                        deltas,
                        seed=seed + 500_000 + pool_index,
                    )
                ),
            }
        )
    return {
        "direct": {
            "candidate_score": float(np.mean(direct_scores)),
            "candidate_advantage": float(
                2.0 * np.mean(direct_scores) - 1.0
            ),
            "score_ci95": list(
                _mean_ci(direct_scores, seed=seed + 400_000)
            ),
        },
        "pool_results": pools,
        "combined_pool_delta": {
            "mean": float(np.mean(combined_deltas)),
            "ci95": list(
                _mean_ci(combined_deltas, seed=seed + 600_000)
            ),
            "pairs": len(combined_deltas),
        },
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_actor(
    actor: torch.nn.Module,
    observations: torch.Tensor,
    *,
    device: torch.device,
    iterations: int,
) -> dict:
    batch = observations.to(device)
    with torch.inference_mode():
        for _ in range(2):
            actor(batch)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            actor(batch)
        _synchronize(device)
    elapsed = time.perf_counter() - started
    evaluated = len(batch) * iterations
    return {
        "batch_size": len(batch),
        "iterations": iterations,
        "elapsed_seconds": elapsed,
        "observations_per_second": evaluated / elapsed,
        "microseconds_per_observation": elapsed * 1e6 / evaluated,
    }


def run_benchmark(
    *,
    student_path: Path,
    champion_path: Path,
    pairs: int,
    seed: int,
    pool_names: tuple[str, ...],
    device: torch.device,
    max_turns: int = 1_000,
) -> dict:
    student, _ = load_frozen_ppo_actor(student_path, device)
    champion, _ = load_frozen_ppo_actor(champion_path, device)
    orbit = BatchedSuitAveragedActor(
        champion,
        alpha=1.0,
        aggregation="logits",
        player_orbit_mode="none",
        base_chunk_size=128,
    ).to(device).eval()
    specs = build_linear_specs(
        pairs=pairs, base_seed=seed, pool_names=pool_names
    )
    policies = {
        "candidate": ActorPolicy(
            student, device, temperature=1.0, collect_data=False, greedy=True
        ),
        "parent": ActorPolicy(
            champion, device, temperature=1.0, collect_data=False, greedy=True
        ),
    }
    outcomes, gameplay_throughput = run_batched_specs(
        specs, policies, max_turns=max_turns
    )
    summary = summarize_outcomes(
        specs,
        outcomes,
        pairs=pairs,
        pool_names=pool_names,
        seed=seed,
    )

    speed_seed = seed + (len(pool_names) + 1) * 100_000
    speed_observations, speed_rollout = collect_public_observations(
        champion,
        games=8,
        seed=speed_seed,
        device=device,
        max_turns=300,
    )
    if len(speed_observations) < 128:
        repeats = (128 + len(speed_observations) - 1) // len(
            speed_observations
        )
        speed_observations = speed_observations.repeat((repeats, 1))
    timing = {}
    for batch_size in (1, 128):
        batch = speed_observations[:batch_size]
        student_timing = time_actor(
            student,
            batch,
            device=device,
            iterations=500 if batch_size == 1 else 50,
        )
        champion_timing = time_actor(
            champion,
            batch,
            device=device,
            iterations=500 if batch_size == 1 else 50,
        )
        orbit_timing = time_actor(
            orbit,
            batch,
            device=device,
            iterations=100 if batch_size == 1 else 5,
        )
        timing[str(batch_size)] = {
            "student": student_timing,
            "champion": champion_timing,
            "orbit_teacher": orbit_timing,
            "student_speedup_over_orbit": (
                student_timing["observations_per_second"]
                / orbit_timing["observations_per_second"]
            ),
            "student_speed_relative_to_champion": (
                student_timing["observations_per_second"]
                / champion_timing["observations_per_second"]
            ),
        }

    deal_seeds = [spec.deal_seed for spec in specs]
    return {
        "schema_version": 1,
        "benchmark": "single_pass_full_suit_orbit_student",
        "development_only": True,
        "student": {
            "path": str(student_path),
            "sha256": file_sha256(student_path),
            "architecture": "PPO StudentNet 121->55",
        },
        "champion": {
            "path": str(champion_path),
            "sha256": file_sha256(champion_path),
        },
        "teacher": "24-way ordinary-suit averaged champion logits",
        "privacy_boundary": (
            "existing 121-feature own-hand and public-counter observation only"
        ),
        "seed_audit": {
            "strategy": "explicit consecutive ranges; no hash derivation",
            "minimum_deal_seed": min(deal_seeds),
            "maximum_deal_seed": max(deal_seeds),
            "unique_deal_seeds": len(set(deal_seeds)),
            "all_below_90m": max(deal_seeds) < DEVELOPMENT_SEED_LIMIT,
            "speed_rollout": speed_rollout,
        },
        "pairs": pairs,
        "pools": list(pool_names),
        **summary,
        "throughput": {
            "gameplay": gameplay_throughput,
            "actor_microbenchmark": timing,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument(
        "--champion", type=Path, default=Path("ppo_actor_100run3.pth")
    )
    parser.add_argument("--pairs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pool", default=",".join(DEFAULT_POOLS))
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    payload = run_benchmark(
        student_path=args.student,
        champion_path=args.champion,
        pairs=args.pairs,
        seed=args.seed,
        pool_names=tuple(
            name.strip() for name in args.pool.split(",") if name.strip()
        ),
        device=torch.device(args.device),
        max_turns=args.max_turns,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
