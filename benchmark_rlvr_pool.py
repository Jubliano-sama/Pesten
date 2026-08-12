#!/usr/bin/env python3
"""Matched pre/post evaluation in 2-8 player self, MLP, and random pools."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import random
import time

import numpy as np


PLAYER_WEIGHTS = {
    2: 14 / 63,
    3: 14 / 63,
    4: 14 / 63,
    5: 6 / 63,
    6: 5 / 63,
    7: 5 / 63,
    8: 5 / 63,
}
OPPONENTS = ("self_anchor", "mlp", "canonical_random")

_BEFORE_FACTORY = None
_AFTER_FACTORY = None
_MLP_FACTORY = None
_RANDOM_FACTORY = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _initialize_worker(before: str, after: str, mlp: str) -> None:
    global _BEFORE_FACTORY, _AFTER_FACTORY, _MLP_FACTORY, _RANDOM_FACTORY
    import torch

    torch.set_num_threads(1)
    from benchmark_agents import load_ppo_agent
    from randomAgent import AgentRandom
    from temporal_puffer_agent import temporal_agent_factory
    from temporal_puffer_v4 import load_temporal_policy_checkpoint

    before_policy, _ = load_temporal_policy_checkpoint(Path(before))
    after_policy, _ = load_temporal_policy_checkpoint(Path(after))
    before_policy.eval()
    after_policy.eval()
    _BEFORE_FACTORY = temporal_agent_factory(
        before_policy, deterministic=True
    )
    _AFTER_FACTORY = temporal_agent_factory(
        after_policy, deterministic=True
    )
    _MLP_FACTORY = load_ppo_agent(Path(mlp), "cpu")
    _RANDOM_FACTORY = AgentRandom


def _opponent_factory(name: str):
    if name == "self_anchor":
        return _BEFORE_FACTORY
    if name == "mlp":
        return _MLP_FACTORY
    if name == "canonical_random":
        return _RANDOM_FACTORY
    raise ValueError(f"unknown opponent: {name}")


def _candidate_score(
    candidate_factory,
    opponent_factory,
    *,
    players: int,
    seed: int,
) -> tuple[float, int, int, int, int]:
    import game

    wins = losses = draws = turns = 0
    for candidate_seat in range(players):
        np.random.seed(seed)
        random.seed(seed)
        factories = [opponent_factory] * players
        factories[candidate_seat] = candidate_factory
        match = game.Game(factories)
        game_turns, winner = match.auto_simulate()
        turns += game_turns
        if winner is None:
            draws += 1
        elif winner == candidate_seat:
            wins += 1
        else:
            losses += 1
    score = (wins + draws / players) / players
    return score, wins, losses, draws, turns


def _evaluate_task(task: dict) -> dict:
    if _BEFORE_FACTORY is None or _AFTER_FACTORY is None:
        raise RuntimeError("benchmark worker was not initialized")
    opponent = _opponent_factory(task["opponent"])
    before_scores = []
    after_scores = []
    totals = {
        "before": defaultdict(int),
        "after": defaultdict(int),
    }
    for offset in range(task["blocks"]):
        seed = task["seed"] + offset
        for label, candidate, sink in (
            ("before", _BEFORE_FACTORY, before_scores),
            ("after", _AFTER_FACTORY, after_scores),
        ):
            score, wins, losses, draws, turns = _candidate_score(
                candidate,
                opponent,
                players=task["players"],
                seed=seed,
            )
            sink.append(score)
            totals[label]["wins"] += wins
            totals[label]["losses"] += losses
            totals[label]["draws"] += draws
            totals[label]["turns"] += turns
    return {
        "opponent": task["opponent"],
        "players": task["players"],
        "before_scores": before_scores,
        "after_scores": after_scores,
        "totals": {name: dict(values) for name, values in totals.items()},
    }


def _normal_interval(mean: float, standard_error: float) -> list[float]:
    z = 3.2905267314919255
    return [mean - z * standard_error, mean + z * standard_error]


def _stratum_summary(row: dict) -> dict:
    before = np.asarray(row["before_scores"], dtype=np.float64)
    after = np.asarray(row["after_scores"], dtype=np.float64)
    delta = after - before
    blocks = len(delta)
    players = int(row["players"])
    summary = {
        "opponent": row["opponent"],
        "players": players,
        "blocks": blocks,
        "games_per_model": blocks * players,
        "before_score": float(before.mean()),
        "after_score": float(after.mean()),
        "delta": float(delta.mean()),
        "before_normalized_win_share": float(before.mean() * players),
        "after_normalized_win_share": float(after.mean() * players),
        "delta_standard_error": float(
            delta.std(ddof=1) / math.sqrt(blocks)
        ),
        "totals": row["totals"],
    }
    summary["delta_normal_ci99_9"] = _normal_interval(
        summary["delta"], summary["delta_standard_error"]
    )
    return summary


def _weighted_summary(strata: list[dict]) -> dict:
    opponent_weight = 1 / len(OPPONENTS)
    weighted = []
    variances = []
    normalized_variances = []
    hoeffding_variance_proxy = 0.0
    normalized_hoeffding_variance_proxy = 0.0
    before = after = delta = normalized_before = normalized_after = 0.0
    for row in strata:
        players = int(row["players"])
        weight = opponent_weight * PLAYER_WEIGHTS[players]
        before_values = np.asarray(row["before_scores"], dtype=np.float64)
        after_values = np.asarray(row["after_scores"], dtype=np.float64)
        differences = after_values - before_values
        before += weight * float(before_values.mean())
        after += weight * float(after_values.mean())
        delta += weight * float(differences.mean())
        normalized_before += weight * float(before_values.mean() * players)
        normalized_after += weight * float(after_values.mean() * players)
        sample_variance = (
            float(differences.var(ddof=1)) if len(differences) > 1 else 0.0
        )
        variances.append(weight * weight * sample_variance / len(differences))
        normalized_variances.append(
            weight
            * weight
            * players
            * players
            * sample_variance
            / len(differences)
        )
        hoeffding_variance_proxy += weight * weight / len(differences)
        normalized_hoeffding_variance_proxy += (
            weight * weight * players * players / len(differences)
        )
        weighted.append(
            {
                "opponent": row["opponent"],
                "players": players,
                "weight": weight,
                "blocks": len(differences),
            }
        )
    standard_error = math.sqrt(sum(variances))
    normalized_standard_error = math.sqrt(sum(normalized_variances))
    alpha = 0.001
    # The paired delta is in [-1, 1], hence range width two.
    hoeffding_margin = 2.0 * math.sqrt(
        math.log(2.0 / alpha) * hoeffding_variance_proxy / 2.0
    )
    normalized_hoeffding_margin = 2.0 * math.sqrt(
        math.log(2.0 / alpha)
        * normalized_hoeffding_variance_proxy
        / 2.0
    )
    normalized_delta = normalized_after - normalized_before
    return {
        "before_score": before,
        "after_score": after,
        "delta": delta,
        "before_normalized_win_share": normalized_before,
        "after_normalized_win_share": normalized_after,
        "delta_normalized_win_share": normalized_delta,
        "delta_normalized_standard_error": normalized_standard_error,
        "delta_normalized_ci99_9": _normal_interval(
            normalized_delta, normalized_standard_error
        ),
        "delta_normalized_hoeffding_ci99_9": [
            normalized_delta - normalized_hoeffding_margin,
            normalized_delta + normalized_hoeffding_margin,
        ],
        "delta_normalized_hoeffding_margin99_9": (
            normalized_hoeffding_margin
        ),
        "delta_standard_error": standard_error,
        "delta_normal_ci99_9": _normal_interval(delta, standard_error),
        "delta_hoeffding_ci99_9": [
            delta - hoeffding_margin,
            delta + hoeffding_margin,
        ],
        "delta_hoeffding_margin99_9": hoeffding_margin,
        "positive_normal_99_9": delta - 3.2905267314919255 * standard_error > 0,
        "positive_distribution_free_99_9": delta - hoeffding_margin > 0,
        "weights": weighted,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks-per-stratum", type=int, default=300)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunks-per-stratum", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.blocks_per_stratum,
        args.workers,
        args.chunks_per_stratum,
    ) < 1:
        raise ValueError("benchmark counts must be positive")
    paths = [args.before.resolve(), args.after.resolve(), args.mlp.resolve()]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    tasks = []
    chunk_size = math.ceil(
        args.blocks_per_stratum / args.chunks_per_stratum
    )
    for opponent_index, opponent in enumerate(OPPONENTS):
        for players in range(2, 9):
            stratum_seed = (
                args.seed + opponent_index * 100_000_000 + players * 1_000_000
            )
            for start in range(0, args.blocks_per_stratum, chunk_size):
                tasks.append(
                    {
                        "opponent": opponent,
                        "players": players,
                        "seed": stratum_seed + start,
                        "blocks": min(
                            chunk_size,
                            args.blocks_per_stratum - start,
                        ),
                    }
                )

    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with context.Pool(
        processes=min(args.workers, len(tasks)),
        initializer=_initialize_worker,
        initargs=tuple(str(path) for path in paths),
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
        for label in ("before", "after"):
            for name, value in chunk["totals"][label].items():
                target["totals"][label][name] += value
    rows = []
    raw_strata = []
    for key in sorted(grouped):
        row = grouped[key]
        row["totals"] = {
            label: dict(values) for label, values in row["totals"].items()
        }
        rows.append(_stratum_summary(row))
        raw_strata.append(row)

    weighted = _weighted_summary(raw_strata)
    result = {
        "schema_version": 1,
        "kind": "matched_rlvr_pool_evaluation",
        "before": {"path": str(paths[0]), "sha256": file_sha256(paths[0])},
        "after": {"path": str(paths[1]), "sha256": file_sha256(paths[1])},
        "mlp": {"path": str(paths[2]), "sha256": file_sha256(paths[2])},
        "protocol": {
            "opponents": list(OPPONENTS),
            "player_weights": {str(key): value for key, value in PLAYER_WEIGHTS.items()},
            "bulk_2_3_4_fraction": sum(PLAYER_WEIGHTS[key] for key in (2, 3, 4)),
            "tail_5_8_fraction": sum(PLAYER_WEIGHTS[key] for key in (5, 6, 7, 8)),
            "blocks_per_stratum": args.blocks_per_stratum,
            "seat_rotations_per_block": "equal to player count",
            "matched_pre_post_seeds": True,
            "confidence": "paired normal and distribution-free Hoeffding 99.9% intervals",
            "seed": args.seed,
        },
        "strata": rows,
        "weighted_result": weighted,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(weighted, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
