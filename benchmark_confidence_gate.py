#!/usr/bin/env python3
"""Common-deal development sweep for recurrent residual confidence gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import torch

from benchmark_model_zoo import DEFAULT_TEACHER, file_sha256
from benchmark_residual_checkpoint import (
    _evaluate_paired_with_scores,
    deterministic_evaluation_agent_factory,
    deterministic_evaluation_runtime,
    paired_bootstrap_interval,
    stratified_paired_bootstrap_interval,
)
from confidence_gated_residual import ConfidenceGatedResidualPolicy
from residual_temporal_policy import ResidualRecurrentTemporalPolicy


SEALED_SEED_START = 92_000_000
DEFAULT_DEVELOPMENT_SEED = 82_000_001
DEFAULT_THRESHOLDS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, math.inf)


def parse_threshold(value: str) -> float:
    normalized = value.strip().lower()
    if normalized in {"inf", "infinity", "unlimited"}:
        return math.inf
    threshold = float(normalized)
    if math.isnan(threshold) or threshold < 0.0:
        raise ValueError("thresholds must be non-negative")
    return threshold


def threshold_label(threshold: float) -> str:
    if math.isinf(threshold):
        return "infinity"
    return f"{threshold:g}"


def _json_threshold(threshold: float) -> float | str:
    return "infinity" if math.isinf(threshold) else threshold


def validate_sweep(
    *,
    thresholds: Sequence[float],
    pairs: int,
    seed: int,
    opponent_count: int,
) -> tuple[float, ...]:
    normalized = tuple(float(value) for value in thresholds)
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if not normalized or 0.0 not in normalized:
        raise ValueError("threshold sweep must include 0 (exact frozen PPO)")
    if len(normalized) != len(set(normalized)):
        raise ValueError("threshold sweep contains duplicates")
    if any(math.isnan(value) or value < 0.0 for value in normalized):
        raise ValueError("thresholds must be non-negative")
    if seed < 0:
        raise ValueError("development seed cannot be negative")
    largest_seed = max(
        seed + max(0, opponent_count - 1) * 100_000 + pairs - 1,
        seed + 990_000,
    )
    if largest_seed >= SEALED_SEED_START:
        raise ValueError("confidence sweep offsets reach sealed final seeds")
    return normalized


def _default_opponents(
    champion_path: Path,
    device: str,
) -> dict[str, object]:
    from benchmark_agents import load_ppo_agent
    from hardcodedAgent import HardcodedStrategyAgent
    from randomAgent import AgentRandom
    from smartAgent import Agent as SmartAgent, SmarterAgent

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


def _delta_block(
    candidate_scores: Sequence[float],
    base_scores: Sequence[float],
    *,
    seed: int,
) -> tuple[dict, list[float]]:
    if len(candidate_scores) != len(base_scores):
        raise AssertionError("candidate and threshold-zero pair counts differ")
    deltas = [
        float(candidate - base)
        for candidate, base in zip(candidate_scores, base_scores)
    ]
    ci = paired_bootstrap_interval(deltas, seed=seed)
    return (
        {
            "candidate_score": float(np.mean(candidate_scores)),
            "threshold_zero_score": float(np.mean(base_scores)),
            "candidate_minus_threshold_zero": float(np.mean(deltas)),
            "paired_bootstrap_ci95": list(ci),
            "same_deals_and_rotated_seats": True,
            "pair_deltas": deltas,
        },
        deltas,
    )


def screen_checkpoint(
    *,
    checkpoint: Path,
    champion_path: Path,
    thresholds: Sequence[float],
    pairs: int,
    seed: int,
    device: str,
    cpu_threads: int,
    opponents: Mapping[str, object],
) -> dict:
    thresholds = validate_sweep(
        thresholds=thresholds,
        pairs=pairs,
        seed=seed,
        opponent_count=len(opponents),
    )
    residual = ResidualRecurrentTemporalPolicy.load_checkpoint(
        checkpoint,
        champion_path=champion_path,
        map_location=device,
    ).eval()
    threshold_results = {}
    base_scores: dict[str, list[float]] = {}
    started = time.perf_counter()

    # Threshold zero is evaluated first even if the CLI lists it later.
    ordered = (0.0,) + tuple(value for value in thresholds if value != 0.0)
    with deterministic_evaluation_runtime(device, cpu_threads=cpu_threads):
        for threshold_index, threshold in enumerate(ordered):
            gate = ConfidenceGatedResidualPolicy(
                residual,
                margin_threshold=threshold,
            ).eval()
            candidate = deterministic_evaluation_agent_factory(
                gate,
                seed=seed,
            )
            opponent_blocks = {}
            delta_groups = {}
            for opponent_index, (opponent_name, opponent_factory) in enumerate(
                opponents.items()
            ):
                opponent_seed = seed + opponent_index * 100_000
                gate.reset_diagnostics()
                result, pair_scores = _evaluate_paired_with_scores(
                    candidate,
                    opponent_factory,
                    pairs=pairs,
                    seed=opponent_seed,
                    candidate_name=(
                        f"confidence_gate:{threshold_label(threshold)}"
                    ),
                    opponent_name=opponent_name,
                )
                if threshold == 0.0:
                    base_scores[opponent_name] = pair_scores
                delta, raw_deltas = _delta_block(
                    pair_scores,
                    base_scores[opponent_name],
                    seed=(
                        opponent_seed
                        + 190_000
                        + threshold_index * 1_000
                    ),
                )
                diagnostics = gate.diagnostics()
                diagnostics["margin_threshold"] = _json_threshold(threshold)
                opponent_blocks[opponent_name] = {
                    "result": result,
                    "same_deal_delta": delta,
                    "gate_diagnostics": diagnostics,
                    "pair_scores": pair_scores,
                }
                delta_groups[opponent_name] = raw_deltas

            combined_ci = stratified_paired_bootstrap_interval(
                delta_groups,
                seed=seed + 990_000 + threshold_index,
            )
            combined = [
                delta for values in delta_groups.values() for delta in values
            ]
            threshold_results[threshold_label(threshold)] = {
                "threshold": _json_threshold(threshold),
                "opponents": opponent_blocks,
                "combined_same_deal_delta": {
                    "opponents": list(opponents),
                    "pairs": len(combined),
                    "candidate_minus_threshold_zero": float(
                        np.mean(combined)
                    ),
                    "paired_bootstrap_ci95": list(combined_ci),
                    "same_deals_and_rotated_seats": True,
                },
            }

    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "architecture": residual.architecture,
        "threshold_zero_semantics": (
            "exact frozen PPO legal argmax while recurrent state still advances"
        ),
        "thresholds": threshold_results,
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_sweep(
    *,
    checkpoints: Mapping[str, Path],
    champion_path: Path = DEFAULT_TEACHER,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    pairs: int = 20,
    seed: int = DEFAULT_DEVELOPMENT_SEED,
    device: str = "cpu",
    cpu_threads: int = 1,
    opponent_names: Sequence[str] | None = None,
    opponents: Mapping[str, object] | None = None,
) -> dict:
    if not checkpoints:
        raise ValueError("at least one checkpoint is required")
    available = (
        dict(opponents)
        if opponents is not None
        else _default_opponents(champion_path, device)
    )
    names = tuple(opponent_names) if opponent_names is not None else tuple(available)
    unknown = sorted(set(names) - set(available))
    if unknown:
        raise ValueError(f"unknown opponents requested: {', '.join(unknown)}")
    selected = {name: available[name] for name in names}
    if not selected:
        raise ValueError("at least one opponent is required")
    validate_sweep(
        thresholds=thresholds,
        pairs=pairs,
        seed=seed,
        opponent_count=len(selected),
    )

    screens = {}
    for label, checkpoint in checkpoints.items():
        screens[label] = screen_checkpoint(
            checkpoint=checkpoint,
            champion_path=champion_path,
            thresholds=thresholds,
            pairs=pairs,
            seed=seed,
            device=device,
            cpu_threads=cpu_threads,
            opponents=selected,
        )
    return {
        "schema_version": 1,
        "scope": "common-development recurrent residual confidence-gate sweep",
        "development_seed": seed,
        "sealed_seed_start": SEALED_SEED_START,
        "pairs_per_opponent": pairs,
        "same_deals_across_thresholds": True,
        "rotated_seats": True,
        "base_threshold": 0.0,
        "thresholds": [_json_threshold(value) for value in thresholds],
        "opponents": list(selected),
        "information_boundary": (
            "DecisionInput plus typed public-event history and own hand only; "
            "no opponent identities or draw-pile order"
        ),
        "checkpoints": screens,
    }


def parse_checkpoint(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("checkpoint must use LABEL=PATH syntax")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise ValueError("checkpoint must use non-empty LABEL=PATH syntax")
    return label, Path(raw_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="repeatable LABEL=PATH recurrent residual checkpoint",
    )
    parser.add_argument(
        "--champion",
        type=Path,
        default=DEFAULT_TEACHER,
    )
    parser.add_argument(
        "--thresholds",
        default="0,0.25,0.5,1,2,4,infinity",
    )
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=DEFAULT_DEVELOPMENT_SEED)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--opponents",
        default="ppo_champion,bayesian_tree,smart,smarter,random",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.cpu_threads < 1:
        raise SystemExit("--cpu-threads must be positive")
    try:
        checkpoints = dict(parse_checkpoint(value) for value in args.checkpoint)
        thresholds = tuple(
            parse_threshold(value) for value in args.thresholds.split(",")
        )
        payload = run_sweep(
            checkpoints=checkpoints,
            champion_path=args.champion,
            thresholds=thresholds,
            pairs=args.pairs,
            seed=args.seed,
            device=args.device,
            cpu_threads=args.cpu_threads,
            opponent_names=tuple(
                name.strip()
                for name in args.opponents.split(",")
                if name.strip()
            ),
        )
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        raise SystemExit(str(error)) from error

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    summary = {
        checkpoint_name: {
            threshold: block["combined_same_deal_delta"]
            for threshold, block in checkpoint["thresholds"].items()
        }
        for checkpoint_name, checkpoint in payload["checkpoints"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
