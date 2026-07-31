#!/usr/bin/env python3
"""Small paired development benchmark for PPO-guided search approaches.

Every pair reuses one fresh deal seed with the guided policy in opposite seats.
Confidence intervals bootstrap whole seat pairs.  The default sweep is broad
and intentionally small: tactical blend, tactical veto, and always-on blend.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Callable, Sequence

import numpy as np
import torch

import game
from guided_search_agent import (
    GuidedSearchConfig,
    PPOGuidedBayesianAgent,
    make_guided_search_factory,
)
from hybrid_agent import FrozenGreedyPPOAgent, load_frozen_ppo_actor
from seed_integrity import DEVELOPMENT_SEED_LIMIT


DEFAULT_APPROACHES = (
    "tactical_blend",
    "tactical_light",
    "tactical_blend_open",
    "tactical_veto",
    "always_blend",
)


@dataclass(frozen=True)
class GameResult:
    seat: int
    winner: int | None
    score: float
    turns: int
    elapsed_seconds: float


@dataclass(frozen=True)
class PairResult:
    pair_index: int
    deal_seed: int
    games: tuple[GameResult, GameResult]
    score: float
    advantage: float


class TimedFrozenPPOAgent(FrozenGreedyPPOAgent):
    def __init__(self, game_instance, actor, device):
        super().__init__(game_instance, actor, device)
        self.decision_count = 0
        self.decision_seconds = 0.0
        self.decision_latencies: list[float] = []

    def playCard(self, current_sort, current_true_number, temperature=None):
        started = time.perf_counter()
        try:
            return super().playCard(
                current_sort,
                current_true_number,
                temperature,
            )
        finally:
            elapsed = time.perf_counter() - started
            self.decision_count += 1
            self.decision_seconds += elapsed
            self.decision_latencies.append(elapsed)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(base_seed: int, pair_index: int) -> int:
    material = f"guided-search-dev:{base_seed}:{pair_index}".encode()
    return (
        int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
        % DEVELOPMENT_SEED_LIMIT
    )


def bootstrap_pair_ci(
    pair_values: Sequence[float],
    *,
    samples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    values = np.asarray(pair_values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("pair_values must be a non-empty vector")
    if samples < 1:
        raise ValueError("samples must be positive")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(values),
        size=(samples, len(values)),
    )
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def approach_config(name: str, base: GuidedSearchConfig) -> GuidedSearchConfig:
    if name == "tactical_blend":
        return replace(base, trigger="tactical", decision_mode="blend")
    if name == "tactical_light":
        return replace(
            base,
            belief_samples=min(base.belief_samples, 2),
            search_depth=min(base.search_depth, 2),
            beam_width=min(base.beam_width, 2),
            trigger="tactical",
            decision_mode="blend",
            search_weight=max(base.search_weight, 0.35),
            override_margin=0.05,
            search_advantage_margin=0.10,
        )
    if name == "tactical_blend_open":
        # Still PPO-majority, but useful as a development probe for whether
        # the posterior signal can improve play when given more authority.
        return replace(
            base,
            trigger="tactical",
            decision_mode="blend",
            search_weight=0.45,
            override_margin=0.02,
            search_advantage_margin=0.08,
            minimum_candidate_prior=0.01,
        )
    if name == "tactical_veto":
        return replace(base, trigger="tactical", decision_mode="veto")
    if name == "always_blend":
        return replace(
            base,
            trigger="always",
            decision_mode="blend",
            search_weight=max(base.search_weight, 0.30),
        )
    raise ValueError(f"unknown approach: {name}")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _capturing_factories(actor, device, config):
    guided_agents: list[PPOGuidedBayesianAgent] = []
    plain_agents: list[TimedFrozenPPOAgent] = []
    guided_factory = make_guided_search_factory(
        actor,
        device,
        config=config,
        on_create=guided_agents.append,
    )

    def plain_factory(game_instance):
        agent = TimedFrozenPPOAgent(game_instance, actor, device)
        plain_agents.append(agent)
        return agent

    return guided_factory, plain_factory, guided_agents, plain_agents


def _play_game(
    guided_factory: Callable,
    plain_factory: Callable,
    *,
    guided_seat: int,
    deal_seed: int,
    max_turns: int,
) -> GameResult:
    _seed_everything(deal_seed)
    factories = (
        [guided_factory, plain_factory]
        if guided_seat == 0
        else [plain_factory, guided_factory]
    )
    instance = game.Game(factories)
    started = time.perf_counter()
    turns, winner = instance.auto_simulate(max_turns=max_turns)
    elapsed = time.perf_counter() - started
    score = (
        0.5
        if winner is None
        else (1.0 if winner == guided_seat else 0.0)
    )
    return GameResult(
        seat=guided_seat,
        winner=winner,
        score=score,
        turns=turns,
        elapsed_seconds=elapsed,
    )


def _latency_summary(agents) -> dict:
    latencies = [
        value
        for agent in agents
        for value in agent.decision_latencies
    ]
    total_decisions = sum(agent.decision_count for agent in agents)
    total_seconds = sum(agent.decision_seconds for agent in agents)
    if not latencies:
        return {
            "decisions": 0,
            "mean_ms": 0.0,
            "p95_ms": 0.0,
            "total_seconds": 0.0,
        }
    return {
        "decisions": total_decisions,
        "mean_ms": 1_000.0 * total_seconds / total_decisions,
        "p95_ms": 1_000.0 * float(np.quantile(latencies, 0.95)),
        "total_seconds": total_seconds,
    }


def benchmark_approach(
    actor,
    device,
    *,
    name: str,
    config: GuidedSearchConfig,
    pairs: int,
    base_seed: int,
    bootstrap_samples: int,
    max_turns: int,
) -> dict:
    guided_factory, plain_factory, guided_agents, plain_agents = (
        _capturing_factories(actor, device, config)
    )
    pair_results = []
    for pair_index in range(pairs):
        deal_seed = derive_seed(base_seed, pair_index)
        games = tuple(
            _play_game(
                guided_factory,
                plain_factory,
                guided_seat=seat,
                deal_seed=deal_seed,
                max_turns=max_turns,
            )
            for seat in (0, 1)
        )
        score = sum(result.score for result in games) / 2.0
        pair_results.append(
            PairResult(
                pair_index=pair_index,
                deal_seed=deal_seed,
                games=games,
                score=score,
                advantage=2.0 * score - 1.0,
            )
        )

    advantages = [pair.advantage for pair in pair_results]
    guided_latency = _latency_summary(guided_agents)
    plain_latency = _latency_summary(plain_agents)
    searches = sum(agent.search_count for agent in guided_agents)
    overrides = sum(agent.override_count for agent in guided_agents)
    decisions = sum(agent.decision_count for agent in guided_agents)
    search_seconds = sum(agent.search_seconds for agent in guided_agents)
    wins = sum(
        result.score == 1.0
        for pair in pair_results
        for result in pair.games
    )
    losses = sum(
        result.score == 0.0
        for pair in pair_results
        for result in pair.games
    )
    draws = pairs * 2 - wins - losses
    return {
        "approach": name,
        "config": asdict(config),
        "strength": {
            "pairs": pairs,
            "games": pairs * 2,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "mean_score": float(
                np.mean([pair.score for pair in pair_results])
            ),
            "mean_paired_advantage": float(np.mean(advantages)),
            "paired_advantage_bootstrap_ci95": bootstrap_pair_ci(
                advantages,
                samples=bootstrap_samples,
                seed=derive_seed(base_seed + 1_000_000, 0),
            ),
        },
        "latency": {
            "guided": guided_latency,
            "plain_ppo": plain_latency,
            "mean_decision_slowdown": (
                guided_latency["mean_ms"] / plain_latency["mean_ms"]
                if plain_latency["mean_ms"] > 0.0
                else None
            ),
            "searches": searches,
            "search_invocation_rate": searches / max(1, decisions),
            "overrides": overrides,
            "override_rate_per_decision": overrides / max(1, decisions),
            "override_rate_when_searched": overrides / max(1, searches),
            "mean_search_ms_when_invoked": (
                1_000.0 * search_seconds / searches
                if searches
                else 0.0
            ),
        },
        "per_pair": [asdict(pair) for pair in pair_results],
    }


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
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2026073107)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument(
        "--approaches",
        default=",".join(DEFAULT_APPROACHES),
    )
    parser.add_argument("--belief-samples", type=int, default=4)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--search-weight", type=float, default=0.25)
    parser.add_argument("--override-margin", type=float, default=0.08)
    parser.add_argument(
        "--search-advantage-margin",
        type=float,
        default=0.15,
    )
    parser.add_argument("--minimum-candidate-prior", type=float, default=0.025)
    parser.add_argument("--torch-threads", type=int, default=1)
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
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    approaches = tuple(
        name.strip() for name in args.approaches.split(",") if name.strip()
    )
    if not approaches:
        raise SystemExit("--approaches must name at least one approach")

    torch.set_num_threads(args.torch_threads)
    actor, device = load_frozen_ppo_actor(args.model, args.device)
    base_config = GuidedSearchConfig(
        belief_samples=args.belief_samples,
        search_depth=args.search_depth,
        beam_width=args.beam_width,
        search_weight=args.search_weight,
        override_margin=args.override_margin,
        search_advantage_margin=args.search_advantage_margin,
        minimum_candidate_prior=args.minimum_candidate_prior,
    )
    started = time.perf_counter()
    try:
        results = [
            benchmark_approach(
                actor,
                device,
                name=name,
                config=approach_config(name, base_config),
                pairs=args.pairs,
                base_seed=args.seed,
                bootstrap_samples=args.bootstrap_samples,
                max_turns=args.max_turns,
            )
            for name in approaches
        ]
    except ValueError as error:
        raise SystemExit(str(error)) from error

    payload = {
        "schema_version": 1,
        "benchmark": "ppo_guided_bayesian_search_development",
        "model": {
            "path": str(args.model),
            "sha256": file_sha256(args.model),
            "device": str(device),
            "frozen": True,
            "role": "root_action_prior_and_sampled_opponent_model",
        },
        "protocol": {
            "base_seed": args.seed,
            "development_only": True,
            "pairs_per_approach": args.pairs,
            "games_per_pair": 2,
            "same_deal_opposite_seats": True,
            "same_deals_across_approaches": True,
            "bootstrap_unit": "seat_pair",
            "bootstrap_samples": args.bootstrap_samples,
            "information_policy": (
                "own hand and public game state only; hidden hands and draw "
                "pile are Bayesian posterior samples"
            ),
        },
        "approaches": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
