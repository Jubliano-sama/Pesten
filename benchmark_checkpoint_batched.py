#!/usr/bin/env python3
"""Exact-semantics batched comparison of two feed-forward PPO checkpoints.

The result schema and score/delta definitions mirror ``benchmark_checkpoint``.
All direct and fixed-pool games are evaluated in one ``BatchedPesten`` cohort,
so actor calls contain decisions from many independent matched deals.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
import time
from typing import Hashable, Mapping, Sequence

import numpy as np
import torch

from ai_observation import (
    COMPATIBLE_END,
    COMPATIBLE_START,
    PASS_ACTION,
    SUIT_ACTION_START,
)
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    PHASE_SUIT,
    POLICY_RANDOM,
    POLICY_RL,
    POLICY_SMART,
    POLICY_SMARTER,
)
from benchmark_hybrid import (
    bootstrap_mean_ci,
    derive_deal_seed,
    file_sha256,
)
import game
from hybrid_agent import (
    load_frozen_ppo_actor,
    make_ppo_agent_factory,
)
from ppo2 import build_action_masks
import randomAgent
import smartAgent


UNSEALED_SEED_LIMIT = 92_000_000
FIXED_POOL = {
    "random": POLICY_RANDOM,
    "smart": POLICY_SMART,
    "smarter": POLICY_SMARTER,
}
CANONICAL_POOL = {
    "random": randomAgent.AgentRandom,
    "smart": smartAgent.Agent,
    "smarter": smartAgent.SmarterAgent,
}


@dataclass(frozen=True)
class BatchGameSpec:
    section: str
    opponent: str | None
    pair_index: int
    evaluated_key: str
    evaluated_seat: int
    deal_seed: int
    policy_kinds: tuple[int, int]
    actor_keys: tuple[Hashable | None, Hashable | None]


@dataclass(frozen=True)
class BatchGameOutcome:
    evaluated_seat: int
    winner: int | None
    result: str
    score: float
    turns: int


def derive_unsealed_seed(
    base_seed: int,
    namespace: str,
    pair_index: int,
) -> int:
    """Use benchmark-compatible hashing while staying below sealed seed space."""
    if not 0 <= base_seed < UNSEALED_SEED_LIMIT:
        raise ValueError(
            f"base seed must be in [0, {UNSEALED_SEED_LIMIT})"
        )
    return derive_deal_seed(base_seed, namespace, pair_index) % UNSEALED_SEED_LIMIT


def build_game_specs(
    *,
    pairs: int,
    base_seed: int,
    pool_names: Sequence[str] = ("random", "smart", "smarter"),
) -> list[BatchGameSpec]:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    unknown = sorted(set(pool_names) - set(FIXED_POOL))
    if unknown:
        raise ValueError(f"unknown pool policies: {', '.join(unknown)}")
    if len(set(pool_names)) != len(pool_names):
        raise ValueError("pool policies must be unique")
    required_deals = pairs * (1 + len(pool_names))
    if required_deals > UNSEALED_SEED_LIMIT:
        raise ValueError(
            "requested distinct deals exceed the unsealed seed namespace"
        )

    specs: list[BatchGameSpec] = []
    seen_distinct_deals: set[int] = set()

    def deal_seed(namespace: str, pair_index: int) -> int:
        seed = derive_unsealed_seed(base_seed, namespace, pair_index)
        # Preserve every historical non-colliding seed. At large development
        # sweeps, resolve the inevitable birthday collisions deterministically
        # instead of aborting the benchmark.
        while seed in seen_distinct_deals:
            seed = (seed + 1) % UNSEALED_SEED_LIMIT
        seen_distinct_deals.add(seed)
        return seed

    for pair_index in range(pairs):
        seed = deal_seed("candidate_vs_parent", pair_index)
        specs.extend(
            (
                BatchGameSpec(
                    section="direct",
                    opponent=None,
                    pair_index=pair_index,
                    evaluated_key="candidate",
                    evaluated_seat=0,
                    deal_seed=seed,
                    policy_kinds=(POLICY_RL, POLICY_RL),
                    actor_keys=("candidate", "parent"),
                ),
                BatchGameSpec(
                    section="direct",
                    opponent=None,
                    pair_index=pair_index,
                    evaluated_key="candidate",
                    evaluated_seat=1,
                    deal_seed=seed,
                    policy_kinds=(POLICY_RL, POLICY_RL),
                    actor_keys=("parent", "candidate"),
                ),
            )
        )

    for opponent_name in pool_names:
        opponent_kind = FIXED_POOL[opponent_name]
        for pair_index in range(pairs):
            seed = deal_seed(f"pool:{opponent_name}", pair_index)
            for evaluated_key in ("candidate", "parent"):
                specs.extend(
                    (
                        BatchGameSpec(
                            section="pool",
                            opponent=opponent_name,
                            pair_index=pair_index,
                            evaluated_key=evaluated_key,
                            evaluated_seat=0,
                            deal_seed=seed,
                            policy_kinds=(POLICY_RL, opponent_kind),
                            actor_keys=(evaluated_key, None),
                        ),
                        BatchGameSpec(
                            section="pool",
                            opponent=opponent_name,
                            pair_index=pair_index,
                            evaluated_key=evaluated_key,
                            evaluated_seat=1,
                            deal_seed=seed,
                            policy_kinds=(opponent_kind, POLICY_RL),
                            actor_keys=(None, evaluated_key),
                        ),
                    )
                )
    return specs


def _exact_non_neural_action(
    engine: BatchedPesten,
    game_index: int,
    python_rng: random.Random,
) -> int:
    """Match canonical AgentRandom's split NumPy/Python RNG behavior."""
    player_index = int(engine.current_player[game_index])
    kind = int(engine.policy_kinds[game_index, player_index])
    if engine.phase[game_index] == PHASE_SUIT and kind == POLICY_RANDOM:
        return SUIT_ACTION_START + python_rng.randint(0, 3)
    return engine.non_neural_action(game_index)


def _synchronize_devices(actor_policies: Mapping[Hashable, ActorPolicy]) -> None:
    synchronized = set()
    for policy in actor_policies.values():
        device = policy.device
        identity = (device.type, device.index)
        if device.type == "cuda" and identity not in synchronized:
            torch.cuda.synchronize(device)
            synchronized.add(identity)


def run_batched_specs(
    specs: Sequence[BatchGameSpec],
    actor_policies: Mapping[Hashable, ActorPolicy],
    *,
    max_turns: int = 1_000,
) -> tuple[list[BatchGameOutcome], dict]:
    """Evaluate all specs, batching pending decisions by checkpoint."""
    if not specs:
        raise ValueError("at least one game spec is required")
    policy_kinds = np.asarray(
        [spec.policy_kinds for spec in specs],
        dtype=np.int8,
    )
    actor_keys = np.asarray(
        [spec.actor_keys for spec in specs],
        dtype=object,
    )
    seeds = np.asarray([spec.deal_seed for spec in specs], dtype=np.uint32)
    engine = BatchedPesten(
        policy_kinds,
        actor_keys,
        seeds,
        max_turns=max_turns,
        smart_agents_have_game_reference=True,
    )
    python_rngs = [random.Random(int(seed)) for seed in seeds]
    inference_calls = 0
    policy_decisions = 0

    _synchronize_devices(actor_policies)
    started = time.perf_counter()
    while True:
        active = engine.active_games()
        if len(active) == 0:
            break
        grouped: dict[Hashable, list[tuple[int, np.ndarray]]] = {}
        immediate: list[tuple[int, int]] = []
        for game_value in active:
            game_index = int(game_value)
            player_index = int(engine.current_player[game_index])
            kind = int(engine.policy_kinds[game_index, player_index])
            if kind != POLICY_RL:
                immediate.append(
                    (
                        game_index,
                        _exact_non_neural_action(
                            engine,
                            game_index,
                            python_rngs[game_index],
                        ),
                    )
                )
                continue
            suit_mode = engine.phase[game_index] == PHASE_SUIT
            observation = engine.observation(
                game_index,
                suit_mode=suit_mode,
            )
            # Do not use rl_has_card_decision here: its historical bitwise
            # check treats an even duplicate count (for example two jacks) as
            # absent.  The canonical PPO contract uses the compatible-count
            # observation, where every positive count is legal.
            if not suit_mode and not np.any(
                observation[COMPATIBLE_START:COMPATIBLE_END] >= 1
            ):
                immediate.append((game_index, PASS_ACTION))
                continue
            key = engine.actor_keys[game_index, player_index]
            if key not in actor_policies:
                raise KeyError(f"missing actor policy for key {key!r}")
            grouped.setdefault(key, []).append(
                (
                    game_index,
                    observation,
                )
            )

        for game_index, action in immediate:
            engine.apply_action(game_index, action)

        for key, requests in grouped.items():
            policy = actor_policies[key]
            observations = torch.from_numpy(
                np.stack([request[1] for request in requests], axis=0)
            ).to(device=policy.device)
            with torch.inference_mode():
                logits = policy.actor(observations)
                masks = build_action_masks(observations)
                actions = torch.argmax(
                    logits.masked_fill(
                        ~masks,
                        torch.finfo(logits.dtype).min,
                    ),
                    dim=-1,
                )
            actions_cpu = actions.to(
                device="cpu",
                dtype=torch.int64,
            ).numpy()
            inference_calls += 1
            policy_decisions += len(requests)
            for (game_index, _), action in zip(requests, actions_cpu):
                engine.apply_action(game_index, int(action))

    _synchronize_devices(actor_policies)
    elapsed = time.perf_counter() - started
    outcomes = []
    for game_index, spec in enumerate(specs):
        winner_value = int(engine.winners[game_index])
        winner = None if winner_value < 0 else winner_value
        if winner is None:
            result, score = "draw", 0.5
        elif winner == spec.evaluated_seat:
            result, score = "win", 1.0
        else:
            result, score = "loss", 0.0
        outcomes.append(
            BatchGameOutcome(
                evaluated_seat=spec.evaluated_seat,
                winner=winner,
                result=result,
                score=score,
                turns=int(engine.turn_counts[game_index]),
            )
        )
    statistics = {
        "games": len(specs),
        "turns": int(engine.turn_counts.sum()),
        "policy_decisions": policy_decisions,
        "inference_calls": inference_calls,
        "elapsed_seconds": elapsed,
        "games_per_second": len(specs) / elapsed,
        "turns_per_second": float(engine.turn_counts.sum()) / elapsed,
        "mean_actor_batch_size": (
            policy_decisions / inference_calls if inference_calls else 0.0
        ),
    }
    return outcomes, statistics


def _game_record(outcome: BatchGameOutcome) -> dict:
    return {
        "evaluated_seat": outcome.evaluated_seat,
        "winner": outcome.winner,
        "result": outcome.result,
        "score": outcome.score,
        "turns": outcome.turns,
    }


def _seat_pair_record(
    pair_index: int,
    deal_seed: int,
    outcomes: Sequence[BatchGameOutcome],
) -> dict:
    if len(outcomes) != 2:
        raise ValueError("a rotated seat pair must contain exactly two games")
    return {
        "pair_index": pair_index,
        "deal_seed": deal_seed,
        "games": [_game_record(outcome) for outcome in outcomes],
        "score": float(np.mean([outcome.score for outcome in outcomes])),
    }


def summarize(
    values: Sequence[float],
    *,
    seed: int,
    bootstrap_samples: int = 10_000,
) -> dict:
    return {
        "mean": float(np.mean(values)),
        "bootstrap_ci95": list(
            bootstrap_mean_ci(
                values,
                seed=seed,
                samples=bootstrap_samples,
            )
        ),
    }


def assemble_benchmark_payload(
    specs: Sequence[BatchGameSpec],
    outcomes: Sequence[BatchGameOutcome],
    *,
    pairs: int,
    base_seed: int,
    pool_names: Sequence[str],
    bootstrap_samples: int,
) -> dict:
    if len(specs) != len(outcomes):
        raise ValueError("spec and outcome counts differ")
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if len(set(pool_names)) != len(pool_names):
        raise ValueError("pool policies must be unique")
    direct_index: dict[
        int,
        list[tuple[BatchGameSpec, BatchGameOutcome]],
    ] = {}
    pool_index: dict[
        tuple[str | None, int, str],
        list[tuple[BatchGameSpec, BatchGameOutcome]],
    ] = {}
    for spec, outcome in zip(specs, outcomes):
        if not 0 <= spec.pair_index < pairs:
            raise ValueError("spec pair_index is outside requested pairs")
        if spec.section == "direct":
            if spec.opponent is not None or spec.evaluated_key != "candidate":
                raise ValueError("malformed direct benchmark spec")
            direct_index.setdefault(spec.pair_index, []).append((spec, outcome))
        elif spec.section == "pool":
            if (
                spec.opponent not in pool_names
                or spec.evaluated_key not in ("candidate", "parent")
            ):
                raise ValueError("malformed pool benchmark spec")
            pool_index.setdefault(
                (spec.opponent, spec.pair_index, spec.evaluated_key),
                [],
            ).append((spec, outcome))
        else:
            raise ValueError(f"unknown benchmark spec section: {spec.section}")

    def validated_pair(
        selected: Sequence[tuple[BatchGameSpec, BatchGameOutcome]],
        *,
        label: str,
    ) -> tuple[int, list[BatchGameOutcome]]:
        if len(selected) != 2:
            raise ValueError(f"{label} must contain exactly two games")
        selected_specs = [spec for spec, _ in selected]
        seats = [spec.evaluated_seat for spec in selected_specs]
        if seats != [0, 1]:
            raise ValueError(
                f"{label} must contain rotated seats 0 and 1 in order"
            )
        deal_seeds = {spec.deal_seed for spec in selected_specs}
        if len(deal_seeds) != 1:
            raise ValueError(f"{label} games must share one deal seed")
        for spec, outcome in selected:
            if outcome.evaluated_seat != spec.evaluated_seat:
                raise ValueError(
                    f"{label} outcome evaluated seat differs from its spec"
                )
        return deal_seeds.pop(), [outcome for _, outcome in selected]

    direct_pairs = []
    for pair_index in range(pairs):
        selected = direct_index.get(pair_index, [])
        deal_seed, pair_outcomes = validated_pair(
            selected,
            label=f"direct pair {pair_index}",
        )
        direct_pairs.append(
            _seat_pair_record(
                pair_index,
                deal_seed,
                pair_outcomes,
            )
        )
    direct_advantages = [
        2.0 * pair["score"] - 1.0 for pair in direct_pairs
    ]

    pool_results = []
    combined_deltas = []
    for opponent_name in pool_names:
        records = []
        for pair_index in range(pairs):
            candidate_selected = pool_index.get(
                (opponent_name, pair_index, "candidate"),
                [],
            )
            parent_selected = pool_index.get(
                (opponent_name, pair_index, "parent"),
                [],
            )
            candidate_seed, candidate_outcomes = validated_pair(
                candidate_selected,
                label=f"pool {opponent_name} candidate pair {pair_index}",
            )
            parent_seed, parent_outcomes = validated_pair(
                parent_selected,
                label=f"pool {opponent_name} parent pair {pair_index}",
            )
            if candidate_seed != parent_seed:
                raise ValueError(
                    f"pool {opponent_name} pair {pair_index} candidate and "
                    "parent must share one deal seed"
                )
            deal_seed = candidate_seed
            candidate_pair = _seat_pair_record(
                pair_index,
                deal_seed,
                candidate_outcomes,
            )
            parent_pair = _seat_pair_record(
                pair_index,
                deal_seed,
                parent_outcomes,
            )
            delta = candidate_pair["score"] - parent_pair["score"]
            combined_deltas.append(delta)
            records.append(
                {
                    "pair_index": pair_index,
                    "deal_seed": deal_seed,
                    "candidate": candidate_pair,
                    "parent": parent_pair,
                    "delta": delta,
                }
            )
        deltas = [record["delta"] for record in records]
        pool_results.append(
            {
                "opponent": opponent_name,
                "candidate_score": summarize(
                    [record["candidate"]["score"] for record in records],
                    seed=derive_unsealed_seed(
                        base_seed, f"candidate:{opponent_name}", 0
                    ),
                    bootstrap_samples=bootstrap_samples,
                ),
                "parent_score": summarize(
                    [record["parent"]["score"] for record in records],
                    seed=derive_unsealed_seed(
                        base_seed, f"parent:{opponent_name}", 0
                    ),
                    bootstrap_samples=bootstrap_samples,
                ),
                "paired_delta": summarize(
                    deltas,
                    seed=derive_unsealed_seed(
                        base_seed, f"delta:{opponent_name}", 0
                    ),
                    bootstrap_samples=bootstrap_samples,
                ),
                "per_pair": records,
            }
        )

    return {
        "pairs_per_matchup": pairs,
        "base_seed": base_seed,
        "direct": {
            "candidate_score": summarize(
                [pair["score"] for pair in direct_pairs],
                seed=derive_unsealed_seed(base_seed, "direct_score", 0),
                bootstrap_samples=bootstrap_samples,
            ),
            "candidate_advantage": summarize(
                direct_advantages,
                seed=derive_unsealed_seed(base_seed, "direct_advantage", 0),
                bootstrap_samples=bootstrap_samples,
            ),
            "per_pair": direct_pairs,
        },
        "pool": pool_results,
        "combined_pool_delta": summarize(
            combined_deltas,
            seed=derive_unsealed_seed(
                base_seed,
                "combined_pool_delta",
                0,
            ),
            bootstrap_samples=bootstrap_samples,
        ),
    }


def run_canonical_specs(
    specs: Sequence[BatchGameSpec],
    *,
    candidate_factory,
    parent_factory,
    max_turns: int = 1_000,
) -> tuple[list[BatchGameOutcome], float]:
    """Canonical reference used for parity tests and throughput calibration."""
    actor_factories = {
        "candidate": candidate_factory,
        "parent": parent_factory,
    }
    outcomes = []
    started = time.perf_counter()
    for spec in specs:
        random.seed(spec.deal_seed)
        np.random.seed(spec.deal_seed)
        factories = []
        for seat in range(2):
            key = spec.actor_keys[seat]
            if key is not None:
                factories.append(actor_factories[key])
            else:
                kind = spec.policy_kinds[seat]
                opponent_name = next(
                    name
                    for name, candidate_kind in FIXED_POOL.items()
                    if candidate_kind == kind
                )
                factories.append(CANONICAL_POOL[opponent_name])
        instance = game.Game(factories)
        turns, winner = instance.auto_simulate(max_turns=max_turns)
        if winner is None:
            result, score = "draw", 0.5
        elif winner == spec.evaluated_seat:
            result, score = "win", 1.0
        else:
            result, score = "loss", 0.0
        outcomes.append(
            BatchGameOutcome(
                evaluated_seat=spec.evaluated_seat,
                winner=winner,
                result=result,
                score=score,
                turns=turns,
            )
        )
    return outcomes, time.perf_counter() - started


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pool", default="random,smart,smarter")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument(
        "--canonical-throughput-pairs",
        type=int,
        default=20,
        help="canonical pairs used only to estimate evaluator speedup; 0 disables",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    pool_names = tuple(
        name.strip() for name in args.pool.split(",") if name.strip()
    )
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    if args.max_turns < 1:
        raise SystemExit("--max-turns must be positive")
    if args.canonical_throughput_pairs < 0:
        raise SystemExit("--canonical-throughput-pairs cannot be negative")
    try:
        specs = build_game_specs(
            pairs=args.pairs,
            base_seed=args.seed,
            pool_names=pool_names,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    candidate_actor, device = load_frozen_ppo_actor(
        args.candidate,
        args.device,
    )
    parent_actor, _ = load_frozen_ppo_actor(args.parent, args.device)
    policies = {
        "candidate": ActorPolicy(
            candidate_actor,
            device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        ),
        "parent": ActorPolicy(
            parent_actor,
            device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        ),
    }
    outcomes, throughput = run_batched_specs(
        specs,
        policies,
        max_turns=args.max_turns,
    )
    payload = {
        "schema_version": 1,
        "engine": "BatchedPesten exact-semantics evaluator",
        "candidate": {
            "path": str(args.candidate),
            "sha256": file_sha256(args.candidate),
        },
        "parent": {
            "path": str(args.parent),
            "sha256": file_sha256(args.parent),
        },
        **assemble_benchmark_payload(
            specs,
            outcomes,
            pairs=args.pairs,
            base_seed=args.seed,
            pool_names=pool_names,
            bootstrap_samples=args.bootstrap_samples,
        ),
        "throughput": {"batched": throughput},
    }

    reference_pairs = min(args.pairs, args.canonical_throughput_pairs)
    if reference_pairs:
        reference_specs = build_game_specs(
            pairs=reference_pairs,
            base_seed=args.seed,
            pool_names=pool_names,
        )
        candidate_factory = make_ppo_agent_factory(
            candidate_actor,
            device,
            shield=False,
        )
        parent_factory = make_ppo_agent_factory(
            parent_actor,
            device,
            shield=False,
        )
        canonical_outcomes, canonical_elapsed = run_canonical_specs(
            reference_specs,
            candidate_factory=candidate_factory,
            parent_factory=parent_factory,
            max_turns=args.max_turns,
        )
        batch_reference_outcomes, batch_reference_stats = run_batched_specs(
            reference_specs,
            policies,
            max_turns=args.max_turns,
        )
        if canonical_outcomes != batch_reference_outcomes:
            raise RuntimeError(
                "batched evaluator failed canonical parity calibration"
            )
        canonical_games_per_second = len(reference_specs) / canonical_elapsed
        payload["throughput"]["canonical_reference"] = {
            "pairs": reference_pairs,
            "games": len(reference_specs),
            "elapsed_seconds": canonical_elapsed,
            "games_per_second": canonical_games_per_second,
        }
        payload["throughput"]["matched_subset_batched"] = batch_reference_stats
        payload["throughput"]["speedup"] = (
            batch_reference_stats["games_per_second"]
            / canonical_games_per_second
        )

    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "direct": payload["direct"]["candidate_score"],
                "direct_advantage": payload["direct"][
                    "candidate_advantage"
                ],
                "combined_pool_delta": payload["combined_pool_delta"],
                "throughput": payload["throughput"],
            },
            indent=2,
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
