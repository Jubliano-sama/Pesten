#!/usr/bin/env python3
"""Cross-entropy policy search for rank/pass/suit biases versus random."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from batched_pesten import ActorPolicy
from benchmark_checkpoint_batched import build_game_specs, run_batched_specs
from benchmark_fitted_q import file_sha256
from evolutionary_rank_policy import BIAS_DIMENSIONS, RankBiasPolicy
from ppo2 import StudentNet


def random_specs(*, pairs: int, seed: int):
    return [
        spec
        for spec in build_game_specs(
            pairs=pairs,
            base_seed=seed,
            pool_names=("random",),
        )
        if spec.section == "pool" and spec.evaluated_key == "candidate"
    ]


def evaluate(
    base: StudentNet,
    biases: np.ndarray,
    *,
    specs,
    device: torch.device,
) -> tuple[float, dict]:
    actor = RankBiasPolicy(
        base,
        torch.as_tensor(biases, dtype=torch.float32, device=device),
    ).to(device).eval()
    outcomes, throughput = run_batched_specs(
        specs,
        {
            "candidate": ActorPolicy(
                actor,
                device,
                temperature=1.0,
                collect_data=False,
                greedy=True,
            )
        },
        max_turns=1_000,
    )
    return float(np.mean([outcome.score for outcome in outcomes])), throughput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--generations", type=int, default=16)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elite", type=int, default=6)
    parser.add_argument("--train-pairs", type=int, default=128)
    parser.add_argument("--validation-pairs", type=int, default=500)
    parser.add_argument("--initial-std", type=float, default=0.6)
    parser.add_argument("--minimum-std", type=float, default=0.04)
    parser.add_argument("--maximum-absolute-bias", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this experiment requires CUDA")
    if min(
        args.generations,
        args.population,
        args.elite,
        args.train_pairs,
        args.validation_pairs,
    ) < 1:
        raise ValueError("search sizes must be positive")
    if args.elite >= args.population:
        raise ValueError("elite must be smaller than population")
    if (
        args.initial_std <= 0
        or args.minimum_std <= 0
        or args.maximum_absolute_bias <= 0
    ):
        raise ValueError("search scales must be positive")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_path = args.base.resolve()
    device = torch.device("cuda")
    base = StudentNet().to(device)
    base.load_state_dict(
        torch.load(base_path, map_location=device, weights_only=True)
    )
    base.eval().requires_grad_(False)

    rng = np.random.default_rng(args.seed)
    mean = np.zeros(BIAS_DIMENSIONS, dtype=np.float64)
    std = np.full(BIAS_DIMENSIONS, args.initial_std, dtype=np.float64)
    validation_specs = random_specs(
        pairs=args.validation_pairs,
        seed=args.seed + 50_000,
    )
    baseline_score, baseline_throughput = evaluate(
        base,
        mean,
        specs=validation_specs,
        device=device,
    )
    best_validation = {
        "generation": 0,
        "score": baseline_score,
        "biases": mean.tolist(),
    }
    curve = []
    started = time.perf_counter()
    for generation in range(1, args.generations + 1):
        training_specs = random_specs(
            pairs=args.train_pairs,
            seed=args.seed + generation * 1_003,
        )
        population = rng.normal(
            loc=mean,
            scale=std,
            size=(args.population, BIAS_DIMENSIONS),
        )
        population[0] = mean
        population = np.clip(
            population,
            -args.maximum_absolute_bias,
            args.maximum_absolute_bias,
        )
        scores = np.empty(args.population, dtype=np.float64)
        for index, candidate in enumerate(population):
            scores[index], _ = evaluate(
                base,
                candidate,
                specs=training_specs,
                device=device,
            )
        elite_indices = np.argsort(scores)[-args.elite :][::-1]
        elites = population[elite_indices]
        elite_scores = scores[elite_indices]
        weights = np.linspace(args.elite, 1, args.elite, dtype=np.float64)
        weights /= weights.sum()
        new_mean = np.sum(elites * weights[:, None], axis=0)
        elite_variance = np.sum(
            weights[:, None] * np.square(elites - new_mean),
            axis=0,
        )
        mean = new_mean
        std = np.maximum(
            args.minimum_std,
            0.35 * std + 0.65 * np.sqrt(elite_variance + 1e-8),
        )
        validation_score, validation_throughput = evaluate(
            base,
            mean,
            specs=validation_specs,
            device=device,
        )
        if validation_score > best_validation["score"]:
            best_validation = {
                "generation": generation,
                "score": validation_score,
                "biases": mean.tolist(),
            }
        record = {
            "generation": generation,
            "train_best": float(elite_scores[0]),
            "train_elite_mean": float(elite_scores.mean()),
            "validation_score": validation_score,
            "mean_norm": float(np.linalg.norm(mean)),
            "mean_std": float(std.mean()),
            "validation_throughput": validation_throughput,
        }
        curve.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    checkpoint = {
        "schema_version": 1,
        "kind": "rank_bias_cross_entropy_search",
        "biases": torch.tensor(
            best_validation["biases"],
            dtype=torch.float32,
        ),
        "base_sha256": file_sha256(base_path),
    }
    checkpoint_path = output_dir / "rank_bias.pt"
    torch.save(checkpoint, checkpoint_path)
    manifest = {
        "schema_version": 1,
        "kind": "rank_bias_cross_entropy_search",
        "promotion_eligible": False,
        "promotion_reason": (
            "configuration search requires disjoint fresh-deal confirmation"
        ),
        "privacy": {
            "observation": "base actor public and own-hand 121 features only",
            "opponent_card_identities": False,
            "draw_pile_order": False,
        },
        "base": {"path": str(base_path), "sha256": file_sha256(base_path)},
        "seed": args.seed,
        "config": {
            "generations": args.generations,
            "population": args.population,
            "elite": args.elite,
            "train_pairs": args.train_pairs,
            "validation_pairs": args.validation_pairs,
            "initial_std": args.initial_std,
            "minimum_std": args.minimum_std,
            "maximum_absolute_bias": args.maximum_absolute_bias,
        },
        "validation": {
            "seed": args.seed + 50_000,
            "baseline_score": baseline_score,
            "baseline_throughput": baseline_throughput,
            "best": best_validation,
        },
        "curve": curve,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
    }
    (output_dir / "rank_bias_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["validation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
