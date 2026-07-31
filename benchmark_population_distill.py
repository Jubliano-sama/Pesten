#!/usr/bin/env python3
"""Fresh paired development benchmark for population-distilled students."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

import game
from benchmark_agents import load_ppo_agent
from hardcodedAgent import HardcodedStrategyAgent
from population_distill import (
    SEALED_FINAL_SEED_START,
    load_population_student,
    population_agent_factory,
    validate_development_seed,
)
from randomAgent import AgentRandom
from smartAgent import Agent as SmartAgent


DEFAULT_BENCHMARK_SEED = 82_000_000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    centre = p + z * z / (2.0 * trials)
    margin = z * math.sqrt(
        p * (1.0 - p) / trials + z * z / (4.0 * trials * trials)
    )
    return (
        (centre - margin) / denominator,
        (centre + margin) / denominator,
    )


def paired_bootstrap_interval(
    pair_scores: list[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    if not pair_scores:
        return 0.0, 0.0
    values = np.asarray(pair_scores, dtype=np.float64)
    rng = np.random.RandomState(seed)
    indices = rng.randint(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def inference_throughput(
    actor: torch.nn.Module,
    device: torch.device,
    *,
    batch_size: int = 1024,
    iterations: int = 100,
) -> dict:
    """Measure batched policy-only throughput on the selected runtime."""
    actor.eval()
    inputs = torch.zeros((batch_size, 121), device=device)
    with torch.inference_mode():
        for _ in range(5):
            actor(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(iterations):
            actor(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    samples = batch_size * iterations
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "samples": samples,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "parameters": sum(parameter.numel() for parameter in actor.parameters()),
    }


def benchmark_paired(
    candidate_factory,
    opponent_factory,
    *,
    candidate_name: str,
    opponent_name: str,
    pairs: int,
    seed: int,
) -> dict:
    validate_development_seed(seed, pairs)
    wins = losses = draws = total_turns = 0
    seat_wins = [0, 0]
    pair_scores = []
    started = time.perf_counter()
    for pair_index in range(pairs):
        deal_seed = seed + pair_index
        pair_points = 0.0
        for candidate_seat in (0, 1):
            np.random.seed(deal_seed)
            random.seed(deal_seed)
            torch.manual_seed(deal_seed)
            factories = (
                [candidate_factory, opponent_factory]
                if candidate_seat == 0
                else [opponent_factory, candidate_factory]
            )
            match = game.Game(factories)
            turns, winner = match.auto_simulate()
            total_turns += turns
            if winner is None:
                draws += 1
                pair_points += 0.5
            elif winner == candidate_seat:
                wins += 1
                seat_wins[candidate_seat] += 1
                pair_points += 1.0
            else:
                losses += 1
        pair_scores.append(pair_points / 2.0)

    games = pairs * 2
    bootstrap = paired_bootstrap_interval(
        pair_scores, seed=seed + 700_001
    )
    return {
        "candidate": candidate_name,
        "opponent": opponent_name,
        "pairs": pairs,
        "games": games,
        "seed": seed,
        "fresh_deal_per_pair": True,
        "same_deal_rotated_seats_within_pair": True,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / games,
        "win_rate_wilson95": wilson_interval(wins, games),
        "score_rate": (wins + 0.5 * draws) / games,
        "paired_score_ci95": bootstrap,
        "wins_as_seat_0": seat_wins[0],
        "wins_as_seat_1": seat_wins[1],
        "average_turns": total_turns / games,
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate", type=Path, default=Path("population_student.pt")
    )
    parser.add_argument(
        "--teacher", type=Path, default=Path("ppo_actor_100run3.pth")
    )
    parser.add_argument("--pairs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_BENCHMARK_SEED)
    parser.add_argument(
        "--opponents", default="ppo,smart,random,tree"
    )
    parser.add_argument("--belief-samples", type=int, default=4)
    parser.add_argument("--search-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/population_distill_development.json"),
    )
    parser.add_argument(
        "--continue-after-negative",
        action="store_true",
        help="run the rest of the pool even if PPO result is decisively negative",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_development_seed(args.seed, args.pairs * 10)
    if args.seed >= SEALED_FINAL_SEED_START:
        raise SystemExit("sealed final seeds are forbidden")
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    student, checkpoint = load_population_student(args.candidate, device)
    candidate_factory = population_agent_factory(student, device)
    ppo_factory = load_ppo_agent(args.teacher, device_name)
    teacher_actor = ppo_factory.shared_actor

    def tree_factory(game_instance):
        return HardcodedStrategyAgent(
            game_instance,
            belief_samples=args.belief_samples,
            search_depth=args.search_depth,
            beam_width=args.beam_width,
        )

    pool = {
        "ppo": ppo_factory,
        "smart": SmartAgent,
        "random": AgentRandom,
        "tree": tree_factory,
    }
    requested = [name.strip() for name in args.opponents.split(",") if name.strip()]
    unknown = set(requested) - set(pool)
    if unknown:
        raise SystemExit(f"unknown opponents: {sorted(unknown)}")
    # PPO is the early quality gate irrespective of display order.
    ordered = (["ppo"] if "ppo" in requested else []) + [
        name for name in requested if name != "ppo"
    ]
    results = {}
    stopped_early = False
    for index, name in enumerate(ordered):
        result = benchmark_paired(
            candidate_factory,
            pool[name],
            candidate_name="population_distilled",
            opponent_name=name,
            pairs=args.pairs,
            seed=args.seed + index * (args.pairs + 10_000),
        )
        results[name] = result
        if (
            name == "ppo"
            and result["paired_score_ci95"][1] < 0.5
            and not args.continue_after_negative
        ):
            stopped_early = True
            break

    report = {
        "schema_version": 1,
        "status": "complete",
        "development_only": True,
        "sealed_final_seed_start": SEALED_FINAL_SEED_START,
        "candidate": {
            "path": str(args.candidate),
            "sha256": file_sha256(args.candidate),
            "architecture": checkpoint["architecture"],
            "training_metadata": checkpoint.get("metadata", {}),
        },
        "teacher": {
            "path": str(args.teacher),
            "sha256": file_sha256(args.teacher),
        },
        "device": str(device),
        "inference_throughput": {
            "candidate": inference_throughput(student, device),
            "teacher": inference_throughput(teacher_actor, device),
        },
        "early_stop_on_decisive_ppo_loss": not args.continue_after_negative,
        "stopped_early": stopped_early,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
