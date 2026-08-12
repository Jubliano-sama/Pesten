#!/usr/bin/env python3
"""Matched pre/post evaluation in 2-8 player Smart/Smarter pools."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import multiprocessing
from pathlib import Path
import time

from benchmark_rlvr_pool import (
    PLAYER_WEIGHTS,
    _stratum_summary,
    _weighted_summary,
    file_sha256,
)
from benchmark_rlvr_pool_batched import _evaluate_task, _initialize_worker


RULE_OPPONENTS = ("smart_rule", "smarter_rule")


def run_benchmark(
    before: Path,
    after: Path,
    mlp: Path,
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
    for opponent_index, opponent in enumerate(RULE_OPPONENTS):
        for players in range(2, 9):
            stratum_seed = seed + opponent_index * 100_000_000 + players * 1_000_000
            for start in range(0, blocks_per_stratum, chunk_size):
                tasks.append(
                    {
                        "opponent": opponent,
                        "players": players,
                        "seed": stratum_seed + start,
                        "blocks": min(chunk_size, blocks_per_stratum - start),
                    }
                )
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with context.Pool(
        processes=min(workers, len(tasks)),
        initializer=_initialize_worker,
        initargs=(str(before), str(after), str(mlp), str(before)),
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
    opponent_results = {
        opponent: _weighted_summary(
            [row for row in raw if row["opponent"] == opponent],
            opponent_weight=1.0,
        )
        for opponent in RULE_OPPONENTS
    }
    return {
        "strata": strata,
        "weighted_result": _weighted_summary(
            raw, opponent_weight=1 / len(RULE_OPPONENTS)
        ),
        "opponent_results": opponent_results,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks-per-stratum", type=int, default=300)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--chunks-per-stratum", type=int, default=6)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    before = args.before.resolve()
    after = args.after.resolve()
    mlp = args.mlp.resolve()
    for path in (before, after, mlp):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = run_benchmark(
        before,
        after,
        mlp,
        blocks_per_stratum=args.blocks_per_stratum,
        workers=args.workers,
        chunks_per_stratum=args.chunks_per_stratum,
        seed=args.seed,
    )
    payload = {
        "schema_version": 1,
        "kind": "matched_temporal_rule_pool_evaluation_batched_exact_engine",
        "before": {"path": str(before), "sha256": file_sha256(before)},
        "after": {"path": str(after), "sha256": file_sha256(after)},
        "mlp_loader_dependency": {
            "path": str(mlp),
            "sha256": file_sha256(mlp),
            "used_as_opponent": False,
        },
        "protocol": {
            "opponents": list(RULE_OPPONENTS),
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
            "deterministic_temporal_actions": True,
            "engine": "BatchedPesten exact-semantics evaluator",
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
    print(json.dumps(payload["weighted_result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
