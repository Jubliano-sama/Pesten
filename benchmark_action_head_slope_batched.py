#!/usr/bin/env python3
"""Paired multi-checkpoint evaluation of RLVR action-head learning slopes."""

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

from batched_pesten import BatchedPesten, PHASE_SUIT, POLICY_RANDOM, POLICY_RL
from benchmark_rlvr_pool import OPPONENTS, PLAYER_WEIGHTS, file_sha256


_POLICIES = None
_MLP_FACTORY = None
_Z99_9 = 3.2905267314919255
_ALPHA = 0.001


def _load_policy(path: Path):
    from temporal_puffer_v4 import load_temporal_policy_checkpoint

    policy, _ = load_temporal_policy_checkpoint(path)
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    return policy


def _initialize_worker(
    candidate_paths: dict[str, str],
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
    loaded = {}
    cache = {}
    for label, raw_path in {
        **candidate_paths,
        "self_anchor": self_anchor,
    }.items():
        path = Path(raw_path)
        key = str(path.resolve())
        if key not in cache:
            cache[key] = _load_policy(path)
        loaded[label] = cache[key]
    _POLICIES = loaded
    _MLP_FACTORY = load_ppo_agent(Path(mlp), "cpu")


def _make_layout(task: dict):
    players = int(task["players"])
    blocks = int(task["blocks"])
    opponent = str(task["opponent"])
    labels = tuple(task["labels"])
    records = []
    seeds = []
    roles = []
    for block in range(blocks):
        deal_seed = int(task["seed"]) + block
        for label in labels:
            for candidate_seat in range(players):
                row_roles = []
                for seat in range(players):
                    if seat == candidate_seat:
                        row_roles.append(("temporal", label))
                    elif opponent == "self_anchor":
                        row_roles.append(("temporal", "self_anchor"))
                    elif opponent == "mlp":
                        row_roles.append(("mlp", ""))
                    elif opponent == "canonical_random":
                        row_roles.append(("canonical_random", ""))
                    else:
                        raise ValueError(f"unknown opponent: {opponent}")
                records.append((block, label, candidate_seat))
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
    labels = tuple(task["labels"])
    records, roles, seeds = _make_layout(task)
    policy_kinds = np.full((len(records), players), POLICY_RL, dtype=np.int8)
    actor_keys = np.empty((len(records), players), dtype=object)
    for row, row_roles in enumerate(roles):
        for seat, (kind, policy_name) in enumerate(row_roles):
            if kind == "canonical_random":
                policy_kinds[row, seat] = POLICY_RANDOM
            actor_keys[row, seat] = (kind, policy_name)
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
                temporal_groups.setdefault(
                    id(runtime.policy), (runtime.policy, [])
                )[1].append((runtime, decision))
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

        for policy, entries in temporal_groups.values():
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
        label: np.zeros(blocks, dtype=np.float64) for label in labels
    }
    totals = {label: defaultdict(int) for label in labels}
    for row, (block, label, candidate_seat) in enumerate(records):
        winner = int(engine.winners[row])
        totals[label]["turns"] += int(engine.turn_counts[row])
        if winner < 0:
            totals[label]["draws"] += 1
            scores[label][block] += 1.0 / players
        elif winner == candidate_seat:
            totals[label]["wins"] += 1
            scores[label][block] += 1.0
        else:
            totals[label]["losses"] += 1
    for label in labels:
        scores[label] /= players
    return {
        "opponent": task["opponent"],
        "players": players,
        "scores": {label: values.tolist() for label, values in scores.items()},
        "totals": {label: dict(values) for label, values in totals.items()},
    }


def _normal_interval(mean: float, standard_error: float) -> list[float]:
    return [
        mean - _Z99_9 * standard_error,
        mean + _Z99_9 * standard_error,
    ]


def _weighted_metric(
    raw: list[dict],
    value_fn,
    *,
    range_width_fn,
    opponent: str | None = None,
) -> dict:
    selected = [
        row for row in raw if opponent is None or row["opponent"] == opponent
    ]
    expected = len(PLAYER_WEIGHTS) * (len(OPPONENTS) if opponent is None else 1)
    if len(selected) != expected:
        raise ValueError(f"expected {expected} strata, got {len(selected)}")
    opponent_weight = 1 / len(OPPONENTS) if opponent is None else 1.0
    mean = 0.0
    variance = 0.0
    hoeffding_proxy = 0.0
    for row in selected:
        values = np.asarray(value_fn(row), dtype=np.float64)
        if values.ndim != 1 or len(values) < 1:
            raise ValueError("each metric needs a non-empty vector per stratum")
        weight = opponent_weight * PLAYER_WEIGHTS[int(row["players"])]
        mean += weight * float(values.mean())
        if len(values) > 1:
            variance += weight * weight * float(values.var(ddof=1)) / len(values)
        width = float(range_width_fn(row))
        hoeffding_proxy += (weight * width) ** 2 / len(values)
    standard_error = math.sqrt(variance)
    hoeffding_margin = math.sqrt(
        math.log(2.0 / _ALPHA) * hoeffding_proxy / 2.0
    )
    return {
        "estimate": mean,
        "standard_error": standard_error,
        "normal_ci99_9": _normal_interval(mean, standard_error),
        "hoeffding_ci99_9": [
            mean - hoeffding_margin,
            mean + hoeffding_margin,
        ],
        "hoeffding_margin99_9": hoeffding_margin,
    }


def _paired_summary(
    raw: list[dict],
    value_fn,
    *,
    score_range_width: float,
    opponent: str | None = None,
) -> dict:
    score = _weighted_metric(
        raw,
        value_fn,
        range_width_fn=lambda row: score_range_width,
        opponent=opponent,
    )
    normalized = _weighted_metric(
        raw,
        lambda row: np.asarray(value_fn(row), dtype=np.float64)
        * int(row["players"]),
        range_width_fn=lambda row: score_range_width * int(row["players"]),
        opponent=opponent,
    )
    return {"score": score, "normalized_win_share": normalized}


def _candidate_summary(
    raw: list[dict], label: str, *, opponent: str | None = None
) -> dict:
    return _paired_summary(
        raw,
        lambda row: row["scores"][label],
        score_range_width=1.0,
        opponent=opponent,
    )


def _difference_summary(
    raw: list[dict],
    after: str,
    before: str,
    *,
    opponent: str | None = None,
) -> dict:
    return _paired_summary(
        raw,
        lambda row: np.asarray(row["scores"][after])
        - np.asarray(row["scores"][before]),
        score_range_width=2.0,
        opponent=opponent,
    )


def _slope_values(row: dict, labels: list[str], phases: list[int]) -> np.ndarray:
    phase_values = np.asarray(phases, dtype=np.float64)
    centered = phase_values - phase_values.mean()
    coefficients = centered / np.square(centered).sum()
    matrix = np.stack(
        [np.asarray(row["scores"][label]) for label in labels], axis=0
    )
    return coefficients @ matrix


def _slope_summary(
    raw: list[dict],
    labels: list[str],
    phases: list[int],
    *,
    opponent: str | None = None,
) -> dict:
    phase_values = np.asarray(phases, dtype=np.float64)
    centered = phase_values - phase_values.mean()
    coefficient_l1 = float(np.abs(centered / np.square(centered).sum()).sum())
    return _paired_summary(
        raw,
        lambda row: _slope_values(row, labels, phases),
        score_range_width=coefficient_l1,
        opponent=opponent,
    )


def _slope_advantage_summary(
    raw: list[dict],
    factorized_labels: list[str],
    full_labels: list[str],
    phases: list[int],
    *,
    opponent: str | None = None,
) -> dict:
    phase_values = np.asarray(phases, dtype=np.float64)
    centered = phase_values - phase_values.mean()
    coefficient_l1 = float(np.abs(centered / np.square(centered).sum()).sum())
    return _paired_summary(
        raw,
        lambda row: _slope_values(row, factorized_labels, phases)
        - _slope_values(row, full_labels, phases),
        score_range_width=2.0 * coefficient_l1,
        opponent=opponent,
    )


def _endpoint_advantage_summary(
    raw: list[dict],
    factorized_labels: list[str],
    full_labels: list[str],
    *,
    opponent: str | None = None,
) -> dict:
    return _paired_summary(
        raw,
        lambda row: (
            np.asarray(row["scores"][factorized_labels[-1]])
            - np.asarray(row["scores"][factorized_labels[0]])
            - np.asarray(row["scores"][full_labels[-1]])
            + np.asarray(row["scores"][full_labels[0]])
        ),
        score_range_width=4.0,
        opponent=opponent,
    )


def _parse_checkpoints(values: list[list[str]], head: str) -> dict[int, Path]:
    checkpoints = {}
    for phase_text, path_text in values:
        phase = int(phase_text)
        path = Path(path_text).resolve()
        if phase < 0:
            raise ValueError(f"{head} phases must be non-negative")
        if phase in checkpoints:
            raise ValueError(f"duplicate {head} phase {phase}")
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoints[phase] = path
    if 0 not in checkpoints or len(checkpoints) < 3:
        raise ValueError(f"{head} needs phase 0 and at least two later phases")
    return dict(sorted(checkpoints.items()))


def run_benchmark(
    full: dict[int, Path],
    factorized: dict[int, Path],
    mlp: Path,
    self_anchor: Path,
    *,
    blocks_per_stratum: int,
    workers: int,
    chunks_per_stratum: int,
    seed: int,
) -> dict:
    if tuple(full) != tuple(factorized):
        raise ValueError("full and factorized checkpoints need identical phases")
    phases = list(full)
    candidate_paths = {
        **{f"full_{phase}": str(path) for phase, path in full.items()},
        **{
            f"factorized_{phase}": str(path)
            for phase, path in factorized.items()
        },
    }
    labels = tuple(candidate_paths)
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
                        "blocks": min(chunk_size, blocks_per_stratum - start),
                        "labels": labels,
                    }
                )
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with context.Pool(
        processes=min(workers, len(tasks)),
        initializer=_initialize_worker,
        initargs=(candidate_paths, str(mlp), str(self_anchor)),
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
                "scores": {label: [] for label in labels},
                "totals": {label: defaultdict(int) for label in labels},
            },
        )
        for label in labels:
            target["scores"][label].extend(chunk["scores"][label])
            for name, value in chunk["totals"][label].items():
                target["totals"][label][name] += value
    raw = []
    for key in sorted(grouped):
        row = grouped[key]
        row["totals"] = {
            label: dict(values) for label, values in row["totals"].items()
        }
        raw.append(row)

    full_labels = [f"full_{phase}" for phase in phases]
    factorized_labels = [f"factorized_{phase}" for phase in phases]
    candidate_results = {
        label: _candidate_summary(raw, label) for label in labels
    }
    improvement_curves = {}
    for head, head_labels in (
        ("full", full_labels),
        ("factorized", factorized_labels),
    ):
        improvement_curves[head] = {
            str(phase): _difference_summary(raw, label, head_labels[0])
            for phase, label in zip(phases, head_labels)
        }
    head_slopes = {
        "full": _slope_summary(raw, full_labels, phases),
        "factorized": _slope_summary(raw, factorized_labels, phases),
    }
    slope_advantage = _slope_advantage_summary(
        raw, factorized_labels, full_labels, phases
    )
    endpoint_advantage = _endpoint_advantage_summary(
        raw, factorized_labels, full_labels
    )
    opponent_results = {}
    for opponent in OPPONENTS:
        opponent_results[opponent] = {
            "head_slopes": {
                "full": _slope_summary(
                    raw, full_labels, phases, opponent=opponent
                ),
                "factorized": _slope_summary(
                    raw, factorized_labels, phases, opponent=opponent
                ),
            },
            "slope_advantage": _slope_advantage_summary(
                raw,
                factorized_labels,
                full_labels,
                phases,
                opponent=opponent,
            ),
            "endpoint_advantage": _endpoint_advantage_summary(
                raw,
                factorized_labels,
                full_labels,
                opponent=opponent,
            ),
            "endpoint_improvement": {
                "full": _difference_summary(
                    raw,
                    full_labels[-1],
                    full_labels[0],
                    opponent=opponent,
                ),
                "factorized": _difference_summary(
                    raw,
                    factorized_labels[-1],
                    factorized_labels[0],
                    opponent=opponent,
                ),
            },
        }
    strata = []
    for row in raw:
        strata.append(
            {
                "opponent": row["opponent"],
                "players": row["players"],
                "blocks": blocks_per_stratum,
                "candidate_scores": {
                    label: float(np.mean(values))
                    for label, values in row["scores"].items()
                },
                "totals": row["totals"],
            }
        )
    return {
        "phases": phases,
        "candidate_results": candidate_results,
        "improvement_curves": improvement_curves,
        "head_slopes_per_phase": head_slopes,
        "factorized_minus_full_slope": slope_advantage,
        "factorized_minus_full_endpoint_improvement": endpoint_advantage,
        "opponent_results": opponent_results,
        "strata": strata,
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="append",
        nargs=2,
        metavar=("PHASE", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument(
        "--factorized",
        action="append",
        nargs=2,
        metavar=("PHASE", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--self-anchor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks-per-stratum", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--chunks-per-stratum", type=int, default=12)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.blocks_per_stratum, args.workers, args.chunks_per_stratum) < 1:
        raise ValueError("benchmark counts must be positive")
    full = _parse_checkpoints(args.full, "full")
    factorized = _parse_checkpoints(args.factorized, "factorized")
    mlp = args.mlp.resolve()
    self_anchor = args.self_anchor.resolve()
    for path in (mlp, self_anchor):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = run_benchmark(
        full,
        factorized,
        mlp,
        self_anchor,
        blocks_per_stratum=args.blocks_per_stratum,
        workers=args.workers,
        chunks_per_stratum=args.chunks_per_stratum,
        seed=args.seed,
    )
    payload = {
        "schema_version": 1,
        "kind": "paired_action_head_rlvr_slope_evaluation_batched_exact_engine",
        "checkpoints": {
            "full": {
                str(phase): {"path": str(path), "sha256": file_sha256(path)}
                for phase, path in full.items()
            },
            "factorized": {
                str(phase): {"path": str(path), "sha256": file_sha256(path)}
                for phase, path in factorized.items()
            },
        },
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
            "common_deals_across_all_checkpoints": True,
            "fixed_self_anchor": True,
            "deterministic_temporal_actions": True,
            "confidence": "paired normal and distribution-free Hoeffding 99.9% intervals",
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
        json.dumps(
            {
                "factorized_minus_full_slope": payload[
                    "factorized_minus_full_slope"
                ],
                "factorized_minus_full_endpoint_improvement": payload[
                    "factorized_minus_full_endpoint_improvement"
                ],
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
