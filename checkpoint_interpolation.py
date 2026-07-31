#!/usr/bin/env python3
"""Create and common-deal screen linear PPO checkpoint interpolations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from benchmark_hybrid import (
    bootstrap_mean_ci,
    file_sha256,
    run_seat_pair,
)
from hybrid_agent import load_frozen_ppo_actor, make_ppo_agent_factory
import randomAgent
import smartAgent


SEALED_SEED_START = 92_000_000
POOL = {
    "random": randomAgent.AgentRandom,
    "smart": smartAgent.Agent,
    "smarter": smartAgent.SmarterAgent,
}


def alpha_label(alpha: float) -> str:
    return f"{alpha:.6f}".rstrip("0").rstrip(".").replace(".", "p")


def parse_alpha(value: str) -> float:
    alpha = float(value)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be finite and in [0, 1]")
    return alpha


def interpolate_state_dicts(
    champion: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    alpha: float,
) -> dict[str, torch.Tensor]:
    """Return ``(1-alpha) * champion + alpha * candidate`` safely."""
    alpha = parse_alpha(str(alpha))
    if champion.keys() != candidate.keys():
        raise ValueError("checkpoint state-dict keys differ")
    interpolated = {}
    for name, champion_value in champion.items():
        candidate_value = candidate[name]
        if champion_value.shape != candidate_value.shape:
            raise ValueError(f"shape mismatch for {name}")
        if champion_value.dtype != candidate_value.dtype:
            raise ValueError(f"dtype mismatch for {name}")
        if torch.is_floating_point(champion_value):
            interpolated[name] = torch.lerp(
                champion_value,
                candidate_value,
                alpha,
            )
        else:
            if not torch.equal(champion_value, candidate_value):
                raise ValueError(
                    f"non-floating checkpoint value differs for {name}"
                )
            interpolated[name] = champion_value.clone()
    return interpolated


def create_interpolated_checkpoints(
    *,
    champion_path: Path,
    candidate_path: Path,
    alphas: Sequence[float],
    output_dir: Path,
) -> dict[float, Path]:
    champion = torch.load(
        champion_path,
        map_location="cpu",
        weights_only=True,
    )
    candidate = torch.load(
        candidate_path,
        map_location="cpu",
        weights_only=True,
    )
    normalized = tuple(parse_alpha(str(alpha)) for alpha in alphas)
    if len(normalized) != len(set(normalized)):
        raise ValueError("alphas must be unique")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for alpha in normalized:
        path = output_dir / f"alpha_{alpha_label(alpha)}.pth"
        torch.save(
            interpolate_state_dicts(
                champion,
                candidate,
                alpha=alpha,
            ),
            path,
        )
        paths[alpha] = path
    return paths


def validate_screen(
    *,
    pairs: int,
    seed: int,
    opponent_count: int,
    alpha_count: int = 1,
) -> None:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if opponent_count < 1 or alpha_count < 1:
        raise ValueError("opponent_count and alpha_count must be positive")
    if seed < 0:
        raise ValueError("seed cannot be negative")
    largest_deal_seed = seed + opponent_count * 100_000 + pairs - 1
    largest_bootstrap_seed = (
        seed + 900_000 + max(opponent_count, alpha_count - 1)
    )
    if max(largest_deal_seed, largest_bootstrap_seed) >= SEALED_SEED_START:
        raise ValueError("screen reaches sealed final seeds")


def summarize(values: Sequence[float], *, seed: int) -> dict:
    return {
        "mean": float(np.mean(values)),
        "bootstrap_ci95": list(bootstrap_mean_ci(values, seed=seed)),
    }


def _pair_scores(factory, opponent, *, pairs: int, seed: int) -> list[float]:
    return [
        run_seat_pair(
            factory,
            opponent,
            pair_index=pair_index,
            deal_seed=seed + pair_index,
        ).score
        for pair_index in range(pairs)
    ]


def screen_interpolations(
    *,
    champion_path: Path,
    candidate_path: Path,
    alphas: Sequence[float],
    pairs: int,
    seed: int,
    output_dir: Path,
    device: str = "cpu",
    opponents: Mapping[str, object] = POOL,
) -> dict:
    """Screen every alpha on identical deals against champion and a pool."""
    validate_screen(
        pairs=pairs,
        seed=seed,
        opponent_count=len(opponents),
        alpha_count=len(alphas),
    )
    started = time.perf_counter()
    paths = create_interpolated_checkpoints(
        champion_path=champion_path,
        candidate_path=candidate_path,
        alphas=alphas,
        output_dir=output_dir,
    )
    champion_actor, torch_device = load_frozen_ppo_actor(
        champion_path,
        device,
    )
    champion_factory = make_ppo_agent_factory(
        champion_actor,
        torch_device,
        shield=False,
    )
    factories = {}
    for alpha, path in paths.items():
        actor, _ = load_frozen_ppo_actor(path, device)
        factories[alpha] = make_ppo_agent_factory(
            actor,
            torch_device,
            shield=False,
        )

    pool_baselines = {}
    for opponent_index, (name, opponent) in enumerate(opponents.items()):
        opponent_seed = seed + opponent_index * 100_000
        pool_baselines[name] = _pair_scores(
            champion_factory,
            opponent,
            pairs=pairs,
            seed=opponent_seed,
        )

    results = {}
    for alpha_index, (alpha, factory) in enumerate(factories.items()):
        direct_seed = seed + len(opponents) * 100_000
        direct_scores = _pair_scores(
            factory,
            champion_factory,
            pairs=pairs,
            seed=direct_seed,
        )
        pool = {}
        combined_deltas = []
        for opponent_index, (name, opponent) in enumerate(opponents.items()):
            opponent_seed = seed + opponent_index * 100_000
            candidate_scores = _pair_scores(
                factory,
                opponent,
                pairs=pairs,
                seed=opponent_seed,
            )
            baseline_scores = pool_baselines[name]
            deltas = [
                candidate_score - baseline_score
                for candidate_score, baseline_score in zip(
                    candidate_scores,
                    baseline_scores,
                )
            ]
            combined_deltas.extend(deltas)
            pool[name] = {
                "deal_seed_start": opponent_seed,
                "candidate_score": summarize(
                    candidate_scores,
                    seed=seed + 500_000 + alpha_index * 100 + opponent_index,
                ),
                "champion_score": summarize(
                    baseline_scores,
                    seed=seed + 600_000 + opponent_index,
                ),
                "paired_delta": summarize(
                    deltas,
                    seed=seed + 700_000 + alpha_index * 100 + opponent_index,
                ),
                "pair_deltas": deltas,
            }
        results[alpha_label(alpha)] = {
            "alpha": alpha,
            "checkpoint": {
                "path": str(paths[alpha]),
                "sha256": file_sha256(paths[alpha]),
            },
            "direct_vs_champion": {
                "deal_seed_start": direct_seed,
                "score": summarize(
                    direct_scores,
                    seed=seed + 800_000 + alpha_index,
                ),
                "advantage": summarize(
                    [2.0 * score - 1.0 for score in direct_scores],
                    seed=seed + 810_000 + alpha_index,
                ),
                "pair_scores": direct_scores,
            },
            "pool": pool,
            "combined_pool_paired_delta": summarize(
                combined_deltas,
                seed=seed + 900_000 + alpha_index,
            ),
        }

    ranked = sorted(
        results,
        key=lambda label: (
            results[label]["combined_pool_paired_delta"]["mean"],
            results[label]["direct_vs_champion"]["score"]["mean"],
        ),
        reverse=True,
    )
    return {
        "schema_version": 1,
        "method": "linear_parameter_interpolation",
        "human_visible_information_only": True,
        "champion": {
            "path": str(champion_path),
            "sha256": file_sha256(champion_path),
        },
        "candidate": {
            "path": str(candidate_path),
            "sha256": file_sha256(candidate_path),
        },
        "pairs_per_matchup": pairs,
        "base_seed": seed,
        "sealed_seed_start": SEALED_SEED_START,
        "common_deals_across_alphas": True,
        "opponents": list(opponents),
        "results": results,
        "ranking_by_combined_pool_delta_then_direct": ranked,
        "best_alpha": results[ranked[0]]["alpha"],
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--champion",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--alphas",
        default="0,0.25,0.5,0.75,1",
    )
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=89_000_001)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = screen_interpolations(
        champion_path=args.champion,
        candidate_path=args.candidate,
        alphas=tuple(parse_alpha(value) for value in args.alphas.split(",")),
        pairs=args.pairs,
        seed=args.seed,
        output_dir=args.output_dir,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        label: {
            "alpha": block["alpha"],
            "direct": block["direct_vs_champion"]["score"]["mean"],
            "pool_delta": block["combined_pool_paired_delta"]["mean"],
            "pool_delta_ci95": block[
                "combined_pool_paired_delta"
            ]["bootstrap_ci95"],
        }
        for label, block in payload["results"].items()
    }
    print(
        json.dumps(
            {
                "best_alpha": payload["best_alpha"],
                "ranking": payload["ranking_by_combined_pool_delta_then_direct"],
                "results": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
