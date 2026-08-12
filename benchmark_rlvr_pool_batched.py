#!/usr/bin/env python3
"""Fast exact-engine matched pre/post evaluation for temporal RLVR models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import multiprocessing
from pathlib import Path
import random
import time

import numpy as np
import torch

from benchmark_rlvr_pool import (
    OPPONENTS,
    PLAYER_WEIGHTS,
    _stratum_summary,
    _weighted_summary,
    file_sha256,
)
from batched_pesten import BatchedPesten, PHASE_SUIT, POLICY_RANDOM, POLICY_RL


_POLICIES = None
_MLP_FACTORY = None


def _load_policy(path: Path):
    from temporal_puffer_v4 import load_temporal_policy_checkpoint

    policy, _ = load_temporal_policy_checkpoint(path)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def _initialize_worker(
    before: str,
    after: str,
    mlp: str,
    self_anchor: str,
) -> None:
    global _POLICIES, _MLP_FACTORY
    from benchmark_agents import load_ppo_agent

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    paths = {
        "before": Path(before),
        "after": Path(after),
        "self_anchor": Path(self_anchor),
    }
    loaded = {}
    cache = {}
    for name, path in paths.items():
        key = str(path.resolve())
        if key not in cache:
            cache[key] = _load_policy(path)
        loaded[name] = cache[key]
    _POLICIES = loaded
    _MLP_FACTORY = load_ppo_agent(Path(mlp), "cpu")


def _make_layout(task: dict):
    players = int(task["players"])
    blocks = int(task["blocks"])
    opponent = str(task["opponent"])
    records = []
    seeds = []
    roles = []
    for block in range(blocks):
        deal_seed = int(task["seed"]) + block
        for variant in ("before", "after"):
            for candidate_seat in range(players):
                row_roles = []
                for seat in range(players):
                    if seat == candidate_seat:
                        row_roles.append(("temporal", variant))
                    elif opponent == "self_anchor":
                        row_roles.append(("temporal", "self_anchor"))
                    elif opponent == "mlp":
                        row_roles.append(("mlp", None))
                    elif opponent == "canonical_random":
                        row_roles.append(("canonical_random", None))
                    else:
                        raise ValueError(f"unknown opponent: {opponent}")
                records.append((block, variant, candidate_seat, deal_seed))
                seeds.append(deal_seed)
                roles.append(row_roles)
    return records, roles, np.asarray(seeds, dtype=np.uint32)


def _evaluate_task(task: dict) -> dict:
    if _POLICIES is None or _MLP_FACTORY is None:
        raise RuntimeError("batched benchmark worker was not initialized")
    from temporal_batched_speed import (
        SafeEventCursor,
        TemporalSeatRuntime,
        _batch_champion_actions,
        _batch_temporal_actions,
        _champion_has_card_decision,
        build_safe_decision,
    )

    players = int(task["players"])
    blocks = int(task["blocks"])
    records, roles, seeds = _make_layout(task)
    policy_kinds = np.full((len(records), players), POLICY_RL, dtype=np.int8)
    actor_keys = np.empty((len(records), players), dtype=object)
    for row, row_roles in enumerate(roles):
        for seat, (kind, policy_name) in enumerate(row_roles):
            if kind == "canonical_random":
                policy_kinds[row, seat] = POLICY_RANDOM
            actor_keys[row, seat] = (kind, policy_name or "")
    engine = BatchedPesten(
        policy_kinds,
        actor_keys,
        seeds,
        max_turns=1000,
        record_public_events=True,
    )
    runtimes = {}
    suit_generators = {}
    for row, row_roles in enumerate(roles):
        for seat, (kind, policy_name) in enumerate(row_roles):
            if kind == "temporal":
                policy = _POLICIES[policy_name]
                runtimes[(row, seat)] = TemporalSeatRuntime(
                    game_index=row,
                    absolute_game_index=row,
                    seat=seat,
                    policy=policy,
                    generator=torch.Generator(device="cpu").manual_seed(
                        int(seeds[row]) * 2 + seat
                    ),
                    state=policy.initial_state(),
                    cursor=SafeEventCursor(),
                )
            elif kind == "canonical_random":
                suit_generators[row] = random.Random(int(seeds[row]))

    while True:
        active = engine.active_games()
        if len(active) == 0:
            break
        actions = {}
        temporal_groups = {}
        champion_games = []
        for row_value in active:
            row = int(row_value)
            seat = int(engine.current_player[row])
            kind, _ = roles[row][seat]
            if kind == "temporal":
                runtime = runtimes[(row, seat)]
                decision = build_safe_decision(
                    engine, row, seat, runtime.cursor
                )
                group = temporal_groups.setdefault(
                    id(runtime.policy), (runtime.policy, [])
                )
                group[1].append((runtime, decision))
            elif kind == "mlp":
                if (
                    int(engine.phase[row]) != PHASE_SUIT
                    and not _champion_has_card_decision(engine, row)
                ):
                    actions[row] = 54
                else:
                    champion_games.append(row)
            elif kind == "canonical_random":
                if int(engine.phase[row]) == PHASE_SUIT:
                    actions[row] = 50 + suit_generators[row].randint(0, 3)
                else:
                    actions[row] = engine.random_action(row)
            else:
                raise AssertionError(f"unexpected role {kind}")

        for _, (policy, entries) in temporal_groups.items():
            chosen = _batch_temporal_actions(
                policy,
                entries,
                None,
                deterministic=True,
            )
            for (runtime, _), action in zip(entries, chosen):
                actions[runtime.game_index] = action
        if champion_games:
            chosen = _batch_champion_actions(
                engine, champion_games, _MLP_FACTORY, None
            )
            actions.update(zip(champion_games, chosen))
        if len(actions) != len(active):
            raise RuntimeError("each active benchmark game needs one action")
        for row_value in active:
            row = int(row_value)
            engine.apply_action(row, actions[row])

    scores = {
        "before": np.zeros(blocks, dtype=np.float64),
        "after": np.zeros(blocks, dtype=np.float64),
    }
    totals = {
        "before": defaultdict(int),
        "after": defaultdict(int),
    }
    for row, (block, variant, candidate_seat, _) in enumerate(records):
        winner = int(engine.winners[row])
        totals[variant]["turns"] += int(engine.turn_counts[row])
        if winner < 0:
            totals[variant]["draws"] += 1
            scores[variant][block] += 1.0 / players
        elif winner == candidate_seat:
            totals[variant]["wins"] += 1
            scores[variant][block] += 1.0
        else:
            totals[variant]["losses"] += 1
    for variant in scores:
        scores[variant] /= players
    return {
        "opponent": task["opponent"],
        "players": players,
        "before_scores": scores["before"].tolist(),
        "after_scores": scores["after"].tolist(),
        "totals": {
            name: dict(values) for name, values in totals.items()
        },
    }


def run_batched_pool_benchmark(
    before: Path,
    after: Path,
    mlp: Path,
    self_anchor: Path,
    *,
    blocks_per_stratum: int,
    workers: int,
    chunks_per_stratum: int,
    seed: int,
) -> dict:
    if min(blocks_per_stratum, workers, chunks_per_stratum) < 1:
        raise ValueError("benchmark counts must be positive")
    tasks = []
    chunk_size = math.ceil(blocks_per_stratum / chunks_per_stratum)
    for opponent_index, opponent in enumerate(OPPONENTS):
        for players in range(2, 9):
            stratum_seed = seed + opponent_index * 100_000_000 + players * 1_000_000
            for start in range(0, blocks_per_stratum, chunk_size):
                tasks.append(
                    {
                        "opponent": opponent,
                        "players": players,
                        "seed": stratum_seed + start,
                        "blocks": min(
                            chunk_size, blocks_per_stratum - start
                        ),
                    }
                )
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with context.Pool(
        processes=min(workers, len(tasks)),
        initializer=_initialize_worker,
        initargs=(
            str(before),
            str(after),
            str(mlp),
            str(self_anchor),
        ),
    ) as pool:
        chunks = pool.map(_evaluate_task, tasks, chunksize=1)
    grouped = {}
    for chunk in chunks:
        key = (chunk["opponent"], chunk["players"])
        target = grouped.setdefault(
            key,
            {
                "opponent": chunk["opponent"],
                "players": chunk["players"],
                "before_scores": [],
                "after_scores": [],
                "totals": {
                    "before": defaultdict(int),
                    "after": defaultdict(int),
                },
            },
        )
        target["before_scores"].extend(chunk["before_scores"])
        target["after_scores"].extend(chunk["after_scores"])
        for variant in ("before", "after"):
            for name, value in chunk["totals"][variant].items():
                target["totals"][variant][name] += value
    raw = []
    strata = []
    for key in sorted(grouped):
        row = grouped[key]
        row["totals"] = {
            variant: dict(values)
            for variant, values in row["totals"].items()
        }
        raw.append(row)
        strata.append(_stratum_summary(row))
    return {
        "strata": strata,
        "weighted_result": _weighted_summary(raw),
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--self-anchor", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks-per-stratum", type=int, default=300)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--chunks-per-stratum", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before = args.before.resolve()
    after = args.after.resolve()
    mlp = args.mlp.resolve()
    self_anchor = (
        args.self_anchor.resolve() if args.self_anchor else before
    )
    for path in (before, after, mlp, self_anchor):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = run_batched_pool_benchmark(
        before,
        after,
        mlp,
        self_anchor,
        blocks_per_stratum=args.blocks_per_stratum,
        workers=args.workers,
        chunks_per_stratum=args.chunks_per_stratum,
        seed=args.seed,
    )
    payload = {
        "schema_version": 1,
        "kind": "matched_rlvr_pool_evaluation_batched_exact_engine",
        "before": {"path": str(before), "sha256": file_sha256(before)},
        "after": {"path": str(after), "sha256": file_sha256(after)},
        "self_anchor": {
            "path": str(self_anchor),
            "sha256": file_sha256(self_anchor),
        },
        "mlp": {"path": str(mlp), "sha256": file_sha256(mlp)},
        "protocol": {
            "opponents": list(OPPONENTS),
            "player_weights": {
                str(key): value for key, value in PLAYER_WEIGHTS.items()
            },
            "bulk_2_3_4_fraction": sum(
                PLAYER_WEIGHTS[key] for key in (2, 3, 4)
            ),
            "tail_5_8_fraction": sum(
                PLAYER_WEIGHTS[key] for key in (5, 6, 7, 8)
            ),
            "blocks_per_stratum": args.blocks_per_stratum,
            "seat_rotations_per_block": "equal to player count",
            "matched_pre_post_seeds": True,
            "fixed_self_anchor": True,
            "deterministic_temporal_actions": True,
            "engine": "BatchedPesten exact-semantics evaluator",
            "confidence": (
                "paired normal and distribution-free Hoeffding 99.9% intervals"
            ),
            "seed": args.seed,
        },
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(payload["weighted_result"], indent=2, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
