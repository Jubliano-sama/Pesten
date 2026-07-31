#!/usr/bin/env python3
"""Bounded common-deal screen for the corrected iter5 GRU suit orbit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from benchmark_residual_checkpoint import (
    CHAMPION_PATH,
    _default_evaluation_dependencies,
    deterministic_evaluation_agent_factory,
    deterministic_evaluation_runtime,
    evaluate_candidate_and_base_paired,
    load_residual_checkpoint,
    stratified_paired_bootstrap_interval,
)
from temporal_suit_orbit import FullSuitOrbitRecurrentPolicy


DEFAULT_CHECKPOINT = Path(
    "experiments/gru_corrected_stronger_seed467301/checkpoints/"
    "gru_corrected_stronger_final_iter5.pt"
)
DEFAULT_SEED = 86_610_001
SEALED_SEED_FLOOR = 92_000_000
DEFAULT_OPPONENTS = ("ppo_champion", "random")


def run_screen(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    champion_path: Path = CHAMPION_PATH,
    pairs: int = 20,
    seed: int = DEFAULT_SEED,
    opponent_names: tuple[str, ...] = DEFAULT_OPPONENTS,
    device: str = "cpu",
    cpu_threads: int = 1,
) -> dict:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    deal_seeds = [
        seed + opponent_index * 100_000 + pair
        for opponent_index, _ in enumerate(opponent_names)
        for pair in range(pairs)
    ]
    if max(deal_seeds) >= SEALED_SEED_FLOOR:
        raise ValueError(
            "development deal seed reaches the sealed >=92m namespace"
        )
    candidate_base = load_residual_checkpoint(
        checkpoint,
        champion_path=champion_path,
        map_location=device,
    ).eval()
    comparison_base = load_residual_checkpoint(
        checkpoint,
        champion_path=champion_path,
        map_location=device,
    ).eval()
    orbit = FullSuitOrbitRecurrentPolicy(candidate_base).eval()
    candidate_factory = deterministic_evaluation_agent_factory(
        orbit, seed=seed
    )
    base_factory = deterministic_evaluation_agent_factory(
        comparison_base, seed=seed
    )
    _, available_opponents = _default_evaluation_dependencies(
        champion_path, device
    )
    unknown = tuple(
        name for name in opponent_names if name not in available_opponents
    )
    if unknown:
        raise ValueError(f"unknown opponents: {unknown}")

    started = time.perf_counter()
    blocks = {}
    with deterministic_evaluation_runtime(
        device, cpu_threads=cpu_threads
    ):
        for opponent_index, opponent_name in enumerate(opponent_names):
            blocks[opponent_name] = evaluate_candidate_and_base_paired(
                candidate_factory,
                base_factory,
                available_opponents[opponent_name],
                pairs=pairs,
                seed=seed + opponent_index * 100_000,
                candidate_name=orbit.architecture,
                opponent_name=opponent_name,
            )
            blocks[opponent_name]["base_result"]["candidate"] = (
                "corrected_residual_gru_iter5_base"
            )

    delta_groups = {
        name: block["_pair_deltas"] for name, block in blocks.items()
    }
    all_candidate_scores = [
        score
        for block in blocks.values()
        for score in block["_candidate_pair_scores"]
    ]
    all_base_scores = [
        score
        for block in blocks.values()
        for score in block["_base_pair_scores"]
    ]
    all_deltas = [
        delta for values in delta_groups.values() for delta in values
    ]
    combined_ci = stratified_paired_bootstrap_interval(
        delta_groups, seed=seed + 990_000
    )
    return {
        "schema_version": 1,
        "screen": "corrected_residual_gru_iter5_full_suit_orbit",
        "checkpoint": str(checkpoint),
        "checkpoint_format": candidate_base.loaded_checkpoint_format,
        "development_only": True,
        "information_boundary": (
            "DecisionInput plus exactly relabeled PublicEventToken fields; "
            "draws remain count-only"
        ),
        "orbit": {
            "ordinary_suit_permutations": orbit.orbit_size,
            "independent_recurrent_states": orbit.orbit_size,
            "state_shape": [
                orbit.num_layers,
                orbit.orbit_size,
                orbit.hidden_size,
            ],
            "logit_reduction": "canonical-label arithmetic mean",
            "value_reduction": "arithmetic mean",
        },
        "deal_seed_audit": {
            "minimum": min(deal_seeds),
            "maximum": max(deal_seeds),
            "count": len(deal_seeds),
            "sealed_seed_floor": SEALED_SEED_FLOOR,
            "all_below_92m": all(
                deal_seed < SEALED_SEED_FLOOR
                for deal_seed in deal_seeds
            ),
        },
        "pairs_per_opponent": pairs,
        "opponents": list(opponent_names),
        "results": {
            name: block["candidate_result"]
            for name, block in blocks.items()
        },
        "base_results": {
            name: block["base_result"] for name, block in blocks.items()
        },
        "paired_candidate_minus_base": {
            name: block["comparison"] for name, block in blocks.items()
        },
        "combined_pool_delta": {
            "pairs": len(all_deltas),
            "candidate_score": float(np.mean(all_candidate_scores)),
            "base_score": float(np.mean(all_base_scores)),
            "candidate_minus_base_score": float(np.mean(all_deltas)),
            "paired_bootstrap_ci95": list(combined_ci),
            "delta_gate_passed": combined_ci[0] > 0.0,
            "same_deals_and_rotated_seats": True,
            "bootstrap": "stratified within opponent",
        },
        "runtime": {
            "device": device,
            "cpu_threads": (
                cpu_threads if torch.device(device).type == "cpu" else None
            ),
            "elapsed_seconds": time.perf_counter() - started,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--champion", type=Path, default=CHAMPION_PATH)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--opponents",
        default=",".join(DEFAULT_OPPONENTS),
        help="comma-separated built-in opponent names",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    opponent_names = tuple(
        name.strip() for name in args.opponents.split(",") if name.strip()
    )
    payload = run_screen(
        checkpoint=args.checkpoint,
        champion_path=args.champion,
        pairs=args.pairs,
        seed=args.seed,
        opponent_names=opponent_names,
        device=args.device,
        cpu_threads=args.cpu_threads,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
