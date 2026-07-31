#!/usr/bin/env python3
"""Reproducible, seat-balanced head-to-head Pesten benchmarks."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from ai_observation import PASS_ACTION
import game
from hardcodedAgent import HardcodedStrategyAgent
from ppo2 import (
    RL_SupervisedAgent,
    StudentNet,
    build_action_masks,
)
import randomAgent
import smartAgent


@dataclass
class BenchmarkResult:
    agent: str
    opponent: str
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    win_rate_ci95: tuple[float, float]
    average_turns: float
    elapsed_seconds: float
    wins_as_seat_0: int
    games_as_seat_0: int
    wins_as_seat_1: int
    games_as_seat_1: int
    seed: int


class GreedyPPOAgent(RL_SupervisedAgent):
    """The deployed PPO architecture with deterministic masked argmax."""

    shared_actor = None
    shared_device = None

    def __init__(self, game_instance):
        super().__init__(
            game_instance,
            self.shared_actor,
            self.shared_device,
            temperature=1.0,
        )
        self.eval_mode = True
        self.collect_data = False

    def act_rl(self, observation, temperature=None):
        device_observation = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            logits = self.actor(device_observation.unsqueeze(0)).squeeze(0)
            mask = build_action_masks(device_observation)
            action = int(
                torch.argmax(
                    logits.masked_fill(
                        ~mask,
                        torch.finfo(logits.dtype).min,
                    )
                ).item()
            )
        return (None if action == PASS_ACTION else action), 0.0


def load_ppo_agent(model_path: Path, device_name: str):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    actor = StudentNet().to(device)
    try:
        state_dict = torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        state_dict = torch.load(model_path, map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
    GreedyPPOAgent.shared_actor = actor
    GreedyPPOAgent.shared_device = device
    return GreedyPPOAgent


def make_tree_agent(args):
    def factory(game_instance):
        return HardcodedStrategyAgent(
            game_instance,
            belief_samples=args.belief_samples,
            search_depth=args.search_depth,
            beam_width=args.beam_width,
        )

    return factory


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials
        + z * z / (4 * trials * trials)
    )
    return (
        (centre - margin) / denominator,
        (centre + margin) / denominator,
    )


def benchmark_pair(
    agent_factory,
    opponent_factory,
    *,
    agent_name: str,
    opponent_name: str,
    games: int,
    seed: int,
) -> BenchmarkResult:
    wins = losses = draws = total_turns = 0
    seat_games = [0, 0]
    seat_wins = [0, 0]
    started = time.perf_counter()

    for game_index in range(games):
        pair_seed = seed + game_index // 2
        np.random.seed(pair_seed)
        random.seed(pair_seed)
        agent_seat = game_index % 2
        factories = (
            [agent_factory, opponent_factory]
            if agent_seat == 0
            else [opponent_factory, agent_factory]
        )
        instance = game.Game(factories)
        turns, winner = instance.auto_simulate()
        total_turns += turns
        seat_games[agent_seat] += 1
        if winner is None:
            draws += 1
        elif winner == agent_seat:
            wins += 1
            seat_wins[agent_seat] += 1
        else:
            losses += 1

    elapsed = time.perf_counter() - started
    interval = wilson_interval(wins, games)
    return BenchmarkResult(
        agent=agent_name,
        opponent=opponent_name,
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=wins / games,
        win_rate_ci95=interval,
        average_turns=total_turns / games,
        elapsed_seconds=elapsed,
        wins_as_seat_0=seat_wins[0],
        games_as_seat_0=seat_games[0],
        wins_as_seat_1=seat_wins[1],
        games_as_seat_1=seat_games[1],
        seed=seed,
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opponent",
        choices=("ppo", "smart", "smarter", "random"),
        default="ppo",
    )
    parser.add_argument("--games", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260731)
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
    parser.add_argument("--belief-samples", type=int, default=8)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=3)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.games < 1:
        raise SystemExit("--games must be positive")

    metadata = {
        "tree_search": {
            "belief_samples": args.belief_samples,
            "search_depth": args.search_depth,
            "beam_width": args.beam_width,
            "belief_model": (
                "Dirichlet posterior over suits and ranks from public plays "
                "and announced suits"
            ),
        }
    }
    opponent_factories = {
        "smart": smartAgent.Agent,
        "smarter": smartAgent.SmarterAgent,
        "random": randomAgent.AgentRandom,
    }
    if args.opponent == "ppo":
        if not args.model.is_file():
            raise SystemExit(f"PPO checkpoint not found: {args.model}")
        opponent_factory = load_ppo_agent(args.model, args.device)
        metadata["ppo_checkpoint"] = {
            "path": str(args.model),
            "sha256": file_sha256(args.model),
        }
    else:
        opponent_factory = opponent_factories[args.opponent]

    result = benchmark_pair(
        make_tree_agent(args),
        opponent_factory,
        agent_name="bayesian_tree_search",
        opponent_name=args.opponent,
        games=args.games,
        seed=args.seed,
    )
    payload = {"result": asdict(result), **metadata}
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
