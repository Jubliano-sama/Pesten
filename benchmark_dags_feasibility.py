#!/usr/bin/env python3
"""Development-only horizon/throughput audit for a possible DAGS pilot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from anchored_ppo import CHAMPION_SHA256, file_sha256
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    POLICY_RL,
    collect_batched_rollout,
)
from ppo2 import DEFAULT_TORCH_THREADS, StudentNet
from seed_integrity import DEVELOPMENT_SEED_LIMIT


PAPER_URL = "https://arxiv.org/abs/2605.14379v1"
DEFAULT_CHAMPION = Path("ppo_actor_100run3.pth")
DEFAULT_OUTPUT = Path("benchmarks/dags_feasibility_dev.json")
DEFAULT_GAMES = 256
DEFAULT_DEAL_SEED_START = 84_510_000
DEFAULT_POLICY_SEED = 84_510_999
DEFAULT_TEMPERATURE = 0.8
DEFAULT_MAX_TURNS = 500


def _validate_development_seeds(
    *,
    deal_seed_start: int,
    games: int,
    policy_seed: int,
) -> None:
    if games < 1:
        raise ValueError("games must be positive")
    seeds = (
        deal_seed_start,
        deal_seed_start + games - 1,
        policy_seed,
    )
    if any(seed < 0 or seed >= DEVELOPMENT_SEED_LIMIT for seed in seeds):
        raise ValueError("all DAGS feasibility seeds must be development-only")


def run_benchmark(
    *,
    champion: Path,
    games: int,
    deal_seed_start: int,
    policy_seed: int,
    temperature: float,
    max_turns: int,
    torch_threads: int,
) -> dict:
    """Run deterministic champion self-play and return compact measurements."""
    _validate_development_seeds(
        deal_seed_start=deal_seed_start,
        games=games,
        policy_seed=policy_seed,
    )
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if max_turns < 1 or torch_threads < 1:
        raise ValueError("max_turns and torch_threads must be positive")

    champion = champion.resolve()
    champion_sha256 = file_sha256(champion)
    if champion_sha256 != CHAMPION_SHA256:
        raise RuntimeError("refusing to benchmark an unverified champion")

    random.seed(policy_seed)
    np.random.seed(policy_seed)
    torch.manual_seed(policy_seed)
    torch.set_num_threads(torch_threads)

    actor = StudentNet()
    actor.load_state_dict(
        torch.load(champion, map_location="cpu", weights_only=True)
    )
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    deal_seeds = np.arange(
        deal_seed_start,
        deal_seed_start + games,
        dtype=np.uint32,
    )
    policy_kinds = np.full((games, 2), POLICY_RL, dtype=np.int8)
    actor_keys = np.full((games, 2), "champion", dtype=object)
    engine = BatchedPesten(
        policy_kinds,
        actor_keys,
        deal_seeds,
        max_turns=max_turns,
    )
    policy = ActorPolicy(
        actor=actor,
        device=torch.device("cpu"),
        temperature=temperature,
        collect_data=True,
    )

    started = time.perf_counter()
    rollout = collect_batched_rollout(
        engine,
        {"champion": policy},
        output_device=torch.device("cpu"),
    )
    wall_seconds = time.perf_counter() - started

    turns = engine.turn_counts.astype(np.float64)
    deterministic = {
        "games": games,
        "completed_games": rollout.completed_games,
        "environment_turns": rollout.environment_turns,
        "policy_decisions": rollout.policy_decisions,
        "policy_episodes": len(rollout.episode_lengths),
        "implicit_zero_decision_seats": (
            2 * games - len(rollout.episode_lengths)
        ),
        "inference_calls": rollout.inference_calls,
        "wins": rollout.wins,
        "draws": rollout.draws,
        "mean_turns_per_game": float(turns.mean()),
        "median_turns_per_game": float(np.median(turns)),
        "p90_turns_per_game": float(np.percentile(turns, 90)),
        "maximum_turns_observed": int(turns.max()),
        "truncated_games": int(np.count_nonzero(turns >= max_turns)),
        "mean_policy_decisions_per_game": rollout.policy_decisions / games,
        "observation_shape": list(rollout.observations.shape),
        "finite_log_probabilities": bool(
            torch.isfinite(rollout.log_probabilities).all()
        ),
    }
    throughput = {
        "rollout_wall_seconds": wall_seconds,
        "games_per_second": games / wall_seconds,
        "environment_turns_per_second": (
            rollout.environment_turns / wall_seconds
        ),
        "policy_decisions_per_second": (
            rollout.policy_decisions / wall_seconds
        ),
        "mean_policy_inference_batch_size": (
            rollout.policy_decisions / rollout.inference_calls
        ),
    }
    return {
        "schema_version": 1,
        "benchmark": "dags_feasibility_champion_horizon",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "name": "Data-Augmented Game Starts",
            "url": PAPER_URL,
        },
        "champion": {
            "path": str(champion),
            "sha256": champion_sha256,
        },
        "workload": {
            "players": 2,
            "agent_sets": [["champion", "champion"]],
            "games": games,
            "temperature": temperature,
            "maximum_turns_per_game": max_turns,
            "torch_threads": torch_threads,
            "rollout_device": "cpu",
            "policy": "StudentNet shared symmetrically by both seats",
        },
        "seed_audit": {
            "deal_seed_recipe": (
                f"contiguous uint32 range [{deal_seed_start}, "
                f"{deal_seed_start + games - 1}]"
            ),
            "deal_seed_start": deal_seed_start,
            "deal_seed_stop_inclusive": deal_seed_start + games - 1,
            "policy_sampling_seed": policy_seed,
            "development_seed_limit": DEVELOPMENT_SEED_LIMIT,
            "development_only": True,
            "sealed_outcomes_used": False,
        },
        "deterministic_measurements": deterministic,
        "throughput": throughput,
        "interpretation": {
            "dags_expected_value": "low_for_current_champion",
            "reason": (
                "Champion games already have short decision horizons relative "
                "to the long control bottlenecks where DAGS was most helpful."
            ),
            "timing_is_machine_state_dependent": True,
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument(
        "--deal-seed-start", type=int, default=DEFAULT_DEAL_SEED_START
    )
    parser.add_argument(
        "--policy-seed", type=int, default=DEFAULT_POLICY_SEED
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_TEMPERATURE
    )
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument(
        "--torch-threads", type=int, default=DEFAULT_TORCH_THREADS
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    payload = run_benchmark(
        champion=args.champion,
        games=args.games,
        deal_seed_start=args.deal_seed_start,
        policy_seed=args.policy_seed,
        temperature=args.temperature,
        max_turns=args.max_turns,
        torch_threads=args.torch_threads,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload["reproduction_command"] = (
        "./venv/bin/python benchmark_dags_feasibility.py "
        f"--champion {args.champion} --output {args.output} "
        f"--games {args.games} --deal-seed-start {args.deal_seed_start} "
        f"--policy-seed {args.policy_seed} "
        f"--temperature {args.temperature} --max-turns {args.max_turns} "
        f"--torch-threads {args.torch_threads}"
    )
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
