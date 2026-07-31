#!/usr/bin/env python3
"""Development-only multiplayer screen for symmetry-projected PPO players.

For every deal the evaluated policy is rotated through every seat while every
other seat uses the exact frozen PPO champion.  Suit-only and player-label
orbit projections therefore see identical deals and seats.  The evaluator
stores only public scores and seeds; policies receive the normal 121-feature
own-hand/public observation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from ai_observation import COMPATIBLE_END, COMPATIBLE_START, PASS_ACTION
from batched_pesten import ActorPolicy, BatchedPesten, PHASE_SUIT, POLICY_RL
from batched_suit_symmetry import BatchedSuitAveragedActor
from benchmark_checkpoint_batched import _synchronize_devices, summarize
from benchmark_model_zoo import DEFAULT_TEACHER
from hybrid_agent import load_frozen_ppo_actor
from ppo2 import build_action_masks


DEVELOPMENT_SEED_LIMIT = 90_000_000


@dataclass(frozen=True)
class MultiplayerSpec:
    variant: str
    deal_index: int
    evaluated_seat: int
    deal_seed: int
    actor_keys: tuple[str, ...]


def build_specs(*, deals: int, players: int, base_seed: int) -> list[MultiplayerSpec]:
    if deals < 1:
        raise ValueError("deals must be positive")
    # The standard 54-card deck can deal seven cards to at most seven players
    # and still leave a starting discard.  The 121-feature schema reserves 13
    # hand-size slots, but that storage capacity is not a valid deal size.
    if not 2 <= players <= 7:
        raise ValueError("players must be between 2 and 7")
    if base_seed < 0 or base_seed + deals > DEVELOPMENT_SEED_LIMIT:
        raise ValueError("development seeds must stay below 90,000,000")
    specs: list[MultiplayerSpec] = []
    for variant in ("suit_only", "player_orbit"):
        for deal_index in range(deals):
            deal_seed = base_seed + deal_index
            for evaluated_seat in range(players):
                keys = ["parent"] * players
                keys[evaluated_seat] = variant
                specs.append(
                    MultiplayerSpec(
                        variant=variant,
                        deal_index=deal_index,
                        evaluated_seat=evaluated_seat,
                        deal_seed=deal_seed,
                        actor_keys=tuple(keys),
                    )
                )
    return specs


def run_specs(
    specs: list[MultiplayerSpec],
    policies: dict[str, ActorPolicy],
    *,
    max_turns: int,
) -> tuple[list[float], dict]:
    if not specs:
        raise ValueError("at least one spec is required")
    players = len(specs[0].actor_keys)
    if any(len(spec.actor_keys) != players for spec in specs):
        raise ValueError("all specs must have the same player count")
    engine = BatchedPesten(
        np.full((len(specs), players), POLICY_RL, dtype=np.int8),
        np.asarray([spec.actor_keys for spec in specs], dtype=object),
        np.asarray([spec.deal_seed for spec in specs], dtype=np.uint32),
        max_turns=max_turns,
        smart_agents_have_game_reference=True,
    )
    inference_calls = 0
    policy_decisions = 0
    _synchronize_devices(policies)
    started = time.perf_counter()
    while True:
        active = engine.active_games()
        if not len(active):
            break
        grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
        immediate: list[int] = []
        for value in active:
            game_index = int(value)
            suit_mode = engine.phase[game_index] == PHASE_SUIT
            observation = engine.observation(
                game_index, suit_mode=suit_mode
            )
            if not suit_mode and not np.any(
                observation[COMPATIBLE_START:COMPATIBLE_END] >= 1
            ):
                immediate.append(game_index)
                continue
            seat = int(engine.current_player[game_index])
            key = str(engine.actor_keys[game_index, seat])
            grouped.setdefault(key, []).append((game_index, observation))
        for game_index in immediate:
            engine.apply_action(game_index, PASS_ACTION)
        for key, requests in grouped.items():
            policy = policies[key]
            observations = torch.from_numpy(
                np.stack([observation for _, observation in requests])
            ).to(policy.device)
            with torch.inference_mode():
                logits = policy.actor(observations)
                masks = build_action_masks(observations)
                actions = torch.argmax(
                    logits.masked_fill(
                        ~masks,
                        torch.finfo(logits.dtype).min,
                    ),
                    dim=-1,
                ).cpu()
            inference_calls += 1
            policy_decisions += len(requests)
            for (game_index, _), action in zip(requests, actions.tolist()):
                engine.apply_action(game_index, int(action))
    _synchronize_devices(policies)
    elapsed = time.perf_counter() - started
    scores = []
    for game_index, spec in enumerate(specs):
        winner = int(engine.winners[game_index])
        scores.append(
            (1.0 / players)
            if winner < 0
            else float(winner == spec.evaluated_seat)
        )
    return scores, {
        "games": len(specs),
        "turns": int(engine.turn_counts.sum()),
        "elapsed_seconds": elapsed,
        "games_per_second": len(specs) / elapsed,
        "policy_decisions": policy_decisions,
        "inference_calls": inference_calls,
        "mean_actor_batch_size": (
            policy_decisions / inference_calls if inference_calls else 0.0
        ),
    }


def assemble(
    specs: list[MultiplayerSpec],
    scores: list[float],
    *,
    deals: int,
    players: int,
    base_seed: int,
    bootstrap_samples: int,
) -> dict:
    indexed = list(zip(specs, scores))
    per_deal: dict[str, list[float]] = {}
    for variant in ("suit_only", "player_orbit"):
        values = []
        for deal_index in range(deals):
            rotated = [
                score
                for spec, score in indexed
                if spec.variant == variant and spec.deal_index == deal_index
            ]
            if len(rotated) != players:
                raise ValueError("incomplete seat rotation")
            values.append(float(np.mean(rotated)))
        per_deal[variant] = values
    paired = [
        orbit - suit
        for orbit, suit in zip(
            per_deal["player_orbit"], per_deal["suit_only"]
        )
    ]
    return {
        "deals": deals,
        "players": players,
        "base_seed": base_seed,
        "null_equal_policy_win_share": 1.0 / players,
        "suit_only_score": summarize(
            per_deal["suit_only"],
            seed=base_seed + 11,
            bootstrap_samples=bootstrap_samples,
        ),
        "player_orbit_score": summarize(
            per_deal["player_orbit"],
            seed=base_seed + 12,
            bootstrap_samples=bootstrap_samples,
        ),
        "paired_player_orbit_minus_suit_only": summarize(
            paired,
            seed=base_seed + 13,
            bootstrap_samples=bootstrap_samples,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--deals", type=int, default=200)
    parser.add_argument("--players", type=int, default=4)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--base-chunk-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    actor, device = load_frozen_ppo_actor(args.champion, args.device)
    suit = BatchedSuitAveragedActor(
        actor, player_orbit_average=False,
        base_chunk_size=args.base_chunk_size,
    ).to(device)
    orbit = BatchedSuitAveragedActor(
        actor, player_orbit_average=True,
        base_chunk_size=args.base_chunk_size,
    ).to(device)
    policies = {
        "parent": ActorPolicy(
            actor, device, temperature=1.0, greedy=True, collect_data=False
        ),
        "suit_only": ActorPolicy(
            suit, device, temperature=1.0, greedy=True, collect_data=False
        ),
        "player_orbit": ActorPolicy(
            orbit, device, temperature=1.0, greedy=True, collect_data=False
        ),
    }
    specs = build_specs(
        deals=args.deals, players=args.players, base_seed=args.base_seed
    )
    scores, throughput = run_specs(specs, policies, max_turns=args.max_turns)
    payload = {
        "schema_version": 1,
        "benchmark": "development_multiplayer_symmetry",
        "human_visible_information_only": True,
        "protocol": "candidate rotated through every seat; all others champion",
        **assemble(
            specs,
            scores,
            deals=args.deals,
            players=args.players,
            base_seed=args.base_seed,
            bootstrap_samples=args.bootstrap_samples,
        ),
        "throughput": throughput,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
