#!/usr/bin/env python3
"""Matched-deal benchmark for the frozen PPO and its tactical shield.

Each pair uses one fresh deal seed twice with the evaluated policy in opposite
seats.  Pool comparisons additionally run the plain PPO and shielded PPO on
the same two deals.  Confidence intervals bootstrap whole pairs, preserving
the matching and seat balance instead of treating games as independent.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

import game
from hardcodedAgent import HardcodedStrategyAgent
from hybrid_agent import load_frozen_ppo_actor, make_ppo_agent_factory
import randomAgent
from seed_integrity import DEVELOPMENT_SEED_LIMIT
import smartAgent


DEFAULT_FIXED_POOL = ("random", "smart", "smarter")


@dataclass(frozen=True)
class GameOutcome:
    evaluated_seat: int
    winner: int | None
    result: str
    score: float
    turns: int


@dataclass(frozen=True)
class SeatPairOutcome:
    pair_index: int
    deal_seed: int
    games: tuple[GameOutcome, GameOutcome]
    score: float


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_deal_seed(base_seed: int, namespace: str, pair_index: int) -> int:
    """Derive stable development deal seeds below reserved namespaces."""
    material = f"{base_seed}:{namespace}:{pair_index}".encode("utf-8")
    return (
        int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
        % DEVELOPMENT_SEED_LIMIT
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _play_game(
    evaluated_factory: Callable,
    opponent_factory: Callable,
    *,
    evaluated_seat: int,
    deal_seed: int,
    max_turns: int,
) -> GameOutcome:
    _seed_everything(deal_seed)
    factories = (
        [evaluated_factory, opponent_factory]
        if evaluated_seat == 0
        else [opponent_factory, evaluated_factory]
    )
    instance = game.Game(factories)
    turns, winner = instance.auto_simulate(max_turns=max_turns)
    if winner is None:
        result = "draw"
        score = 0.5
    elif winner == evaluated_seat:
        result = "win"
        score = 1.0
    else:
        result = "loss"
        score = 0.0
    return GameOutcome(
        evaluated_seat=evaluated_seat,
        winner=winner,
        result=result,
        score=score,
        turns=turns,
    )


def run_seat_pair(
    evaluated_factory: Callable,
    opponent_factory: Callable,
    *,
    pair_index: int,
    deal_seed: int,
    max_turns: int = 1_000,
) -> SeatPairOutcome:
    """Play the same shuffled deal with the evaluated policy in both seats."""
    games = tuple(
        _play_game(
            evaluated_factory,
            opponent_factory,
            evaluated_seat=seat,
            deal_seed=deal_seed,
            max_turns=max_turns,
        )
        for seat in (0, 1)
    )
    return SeatPairOutcome(
        pair_index=pair_index,
        deal_seed=deal_seed,
        games=games,
        score=sum(outcome.score for outcome in games) / len(games),
    )


def bootstrap_mean_ci(
    pair_values: Sequence[float],
    *,
    samples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI over independent matched-pair statistics."""
    values = np.asarray(pair_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("pair_values must be a non-empty one-dimensional sequence")
    if samples < 1:
        raise ValueError("samples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(values),
        size=(samples, len(values)),
    )
    means = values[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, (tail, 1.0 - tail))
    return float(low), float(high)


def _result_counts(pairs: Iterable[SeatPairOutcome]) -> dict[str, int]:
    counts = {"wins": 0, "losses": 0, "draws": 0}
    count_key = {"win": "wins", "loss": "losses", "draw": "draws"}
    for pair in pairs:
        for outcome in pair.games:
            counts[count_key[outcome.result]] += 1
    return counts


def _pair_summary(
    pairs: Sequence[SeatPairOutcome],
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    scores = [pair.score for pair in pairs]
    games = [outcome for pair in pairs for outcome in pair.games]
    return {
        "pairs": len(pairs),
        "games": len(games),
        **_result_counts(pairs),
        "mean_score": float(np.mean(scores)),
        "mean_score_bootstrap_ci95": bootstrap_mean_ci(
            scores,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        ),
        "average_turns": float(np.mean([outcome.turns for outcome in games])),
    }


def benchmark_head_to_head(
    shield_factory: Callable,
    plain_factory: Callable,
    *,
    pairs: int,
    base_seed: int,
    bootstrap_samples: int,
    max_turns: int = 1_000,
) -> dict:
    outcomes = []
    for pair_index in range(pairs):
        deal_seed = derive_deal_seed(base_seed, "shield_vs_plain", pair_index)
        outcomes.append(
            run_seat_pair(
                shield_factory,
                plain_factory,
                pair_index=pair_index,
                deal_seed=deal_seed,
                max_turns=max_turns,
            )
        )
    # In a zero-sum game with draws worth half, this is the shield's paired
    # score minus the plain PPO's paired score.
    advantages = [2.0 * pair.score - 1.0 for pair in outcomes]
    return {
        "matchup": "shield_vs_plain_ppo",
        "summary": {
            **_pair_summary(
                outcomes,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=derive_deal_seed(
                    base_seed,
                    "bootstrap_head_to_head",
                    0,
                ),
            ),
            "mean_paired_advantage": float(np.mean(advantages)),
            "paired_advantage_bootstrap_ci95": bootstrap_mean_ci(
                advantages,
                samples=bootstrap_samples,
                seed=derive_deal_seed(
                    base_seed,
                    "bootstrap_head_to_head_advantage",
                    0,
                ),
            ),
        },
        "per_pair": [asdict(pair) for pair in outcomes],
    }


def benchmark_against_pool(
    shield_factory: Callable,
    plain_factory: Callable,
    opponent_factories: dict[str, Callable],
    *,
    pairs: int,
    base_seed: int,
    bootstrap_samples: int,
    max_turns: int = 1_000,
) -> dict:
    matchups = []
    all_deltas = []
    used_seeds = set()

    for opponent_name, opponent_factory in opponent_factories.items():
        records = []
        for pair_index in range(pairs):
            deal_seed = derive_deal_seed(
                base_seed,
                f"pool:{opponent_name}",
                pair_index,
            )
            if deal_seed in used_seeds:
                raise RuntimeError("evaluation deal seed collision")
            used_seeds.add(deal_seed)
            shield_pair = run_seat_pair(
                shield_factory,
                opponent_factory,
                pair_index=pair_index,
                deal_seed=deal_seed,
                max_turns=max_turns,
            )
            plain_pair = run_seat_pair(
                plain_factory,
                opponent_factory,
                pair_index=pair_index,
                deal_seed=deal_seed,
                max_turns=max_turns,
            )
            delta = shield_pair.score - plain_pair.score
            all_deltas.append(delta)
            records.append(
                {
                    "pair_index": pair_index,
                    "deal_seed": deal_seed,
                    "shield": asdict(shield_pair),
                    "plain": asdict(plain_pair),
                    "paired_score_delta": delta,
                }
            )
        deltas = [record["paired_score_delta"] for record in records]
        matchups.append(
            {
                "opponent": opponent_name,
                "summary": {
                    "pairs": pairs,
                    "games_per_policy": pairs * 2,
                    "shield_mean_score": float(
                        np.mean(
                            [record["shield"]["score"] for record in records]
                        )
                    ),
                    "plain_mean_score": float(
                        np.mean(
                            [record["plain"]["score"] for record in records]
                        )
                    ),
                    "mean_paired_score_delta": float(np.mean(deltas)),
                    "paired_delta_bootstrap_ci95": bootstrap_mean_ci(
                        deltas,
                        samples=bootstrap_samples,
                        seed=derive_deal_seed(
                            base_seed,
                            f"bootstrap_pool:{opponent_name}",
                            0,
                        ),
                    ),
                },
                "per_pair": records,
            }
        )

    return {
        "fixed_pool": list(opponent_factories),
        "summary": {
            "matched_pairs": len(all_deltas),
            "mean_paired_score_delta": float(np.mean(all_deltas)),
            "paired_delta_bootstrap_ci95": bootstrap_mean_ci(
                all_deltas,
                samples=bootstrap_samples,
                seed=derive_deal_seed(base_seed, "bootstrap_pool_all", 0),
            ),
        },
        "matchups": matchups,
    }


def fixed_opponent_factories(
    names: Sequence[str],
    *,
    belief_samples: int = 8,
    search_depth: int = 3,
    beam_width: int = 3,
) -> dict[str, Callable]:
    factories = {
        "random": randomAgent.AgentRandom,
        "smart": smartAgent.Agent,
        "smarter": smartAgent.SmarterAgent,
        "bayesian": lambda game_instance: HardcodedStrategyAgent(
            game_instance,
            belief_samples=belief_samples,
            search_depth=search_depth,
            beam_width=beam_width,
        ),
    }
    unknown = sorted(set(names) - set(factories))
    if unknown:
        raise ValueError(f"unknown fixed-pool opponents: {', '.join(unknown)}")
    return {name: factories[name] for name in names}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument(
        "--mode",
        choices=("head-to-head", "pool", "both"),
        default="both",
    )
    parser.add_argument(
        "--pool",
        default=",".join(DEFAULT_FIXED_POOL),
        help="comma-separated subset of random,smart,smarter,bayesian",
    )
    parser.add_argument("--belief-samples", type=int, default=8)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.pairs < 1:
        raise SystemExit("--pairs must be positive")
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    if args.max_turns < 1:
        raise SystemExit("--max-turns must be positive")
    pool_names = tuple(
        name.strip() for name in args.pool.split(",") if name.strip()
    )
    if args.mode in {"pool", "both"} and not pool_names:
        raise SystemExit("--pool must name at least one opponent")

    actor, device = load_frozen_ppo_actor(args.model, args.device)
    plain_factory = make_ppo_agent_factory(actor, device, shield=False)
    shield_factory = make_ppo_agent_factory(actor, device, shield=True)
    started = time.perf_counter()
    payload = {
        "schema_version": 1,
        "benchmark": "hybrid_public_tactical_shield",
        "model": {
            "path": str(args.model),
            "sha256": file_sha256(args.model),
            "device": str(device),
            "frozen": True,
            "greedy": True,
        },
        "protocol": {
            "base_seed": args.seed,
            "pairs_per_matchup": args.pairs,
            "games_per_pair": 2,
            "seat_balanced": True,
            "matched_initial_deal": True,
            "bootstrap_unit": "seat_pair",
            "bootstrap_samples": args.bootstrap_samples,
            "max_turns": args.max_turns,
            "information_policy": (
                "own hand plus public Game fields only; no opponent card "
                "identities and no draw-pile inspection"
            ),
        },
    }
    if args.mode in {"head-to-head", "both"}:
        payload["head_to_head"] = benchmark_head_to_head(
            shield_factory,
            plain_factory,
            pairs=args.pairs,
            base_seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            max_turns=args.max_turns,
        )
    if args.mode in {"pool", "both"}:
        try:
            opponents = fixed_opponent_factories(
                pool_names,
                belief_samples=args.belief_samples,
                search_depth=args.search_depth,
                beam_width=args.beam_width,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        payload["pool_comparison"] = benchmark_against_pool(
            shield_factory,
            plain_factory,
            opponents,
            pairs=args.pairs,
            base_seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
            max_turns=args.max_turns,
        )
    payload["elapsed_seconds"] = time.perf_counter() - started

    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
