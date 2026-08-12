#!/usr/bin/env python3
"""Large exact-engine 1v1 benchmark between two temporal checkpoints."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from benchmark_random_1v1_batched import (
    ALPHA,
    Z_99_9_TWO_SIDED,
    _hoeffding_interval,
)
from benchmark_rlvr_pool import file_sha256
from benchmark_rlvr_pool_batched import _evaluate_task, _initialize_worker


def _summary(scores: np.ndarray, totals: dict[str, int]) -> dict:
    score = float(scores.mean())
    standard_error = float(scores.std(ddof=1) / math.sqrt(len(scores)))
    normal = [
        max(0.0, score - Z_99_9_TWO_SIDED * standard_error),
        min(1.0, score + Z_99_9_TWO_SIDED * standard_error),
    ]
    hoeffding = _hoeffding_interval(score, len(scores))
    games = sum(totals.get(name, 0) for name in ("wins", "losses", "draws"))
    return {
        "blocks": len(scores),
        "games": games,
        "wins": int(totals.get("wins", 0)),
        "losses": int(totals.get("losses", 0)),
        "draws": int(totals.get("draws", 0)),
        "score": score,
        "paired_standard_error": standard_error,
        "paired_normal_ci99_9": normal,
        "paired_hoeffding_ci99_9": list(hoeffding),
    }


def run_benchmark(
    candidate: Path,
    opponent: Path,
    mlp: Path,
    *,
    blocks: int,
    workers: int,
    chunks: int,
    seed: int,
) -> dict:
    if min(blocks, workers, chunks) < 1:
        raise ValueError("benchmark counts must be positive")
    chunk_size = math.ceil(blocks / chunks)
    tasks = [
        {
            "opponent": "self_anchor",
            "players": 2,
            "seed": seed + start,
            "blocks": min(chunk_size, blocks - start),
        }
        for start in range(0, blocks, chunk_size)
    ]
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with context.Pool(
        processes=min(workers, len(tasks)),
        initializer=_initialize_worker,
        initargs=(str(opponent), str(candidate), str(mlp), str(opponent)),
    ) as pool:
        rows = pool.map(_evaluate_task, tasks, chunksize=1)
    scores = []
    totals = defaultdict(int)
    for row in rows:
        scores.extend(row["after_scores"])
        for name, value in row["totals"]["after"].items():
            totals[name] += int(value)
    return {
        "result": _summary(np.asarray(scores, dtype=np.float64), dict(totals)),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--chunks", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--noninferiority-margin", type=float, default=0.02)
    args = parser.parse_args()
    paths = tuple(
        path.resolve() for path in (args.candidate, args.opponent, args.mlp)
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0 < args.noninferiority_margin < 0.5:
        raise ValueError("noninferiority margin must be between zero and 0.5")
    benchmark = run_benchmark(
        paths[0],
        paths[1],
        paths[2],
        blocks=args.blocks,
        workers=args.workers,
        chunks=args.chunks,
        seed=args.seed,
    )
    floor = 0.5 - args.noninferiority_margin
    result = benchmark["result"]
    result["noninferiority_margin"] = args.noninferiority_margin
    result["noninferiority_floor"] = floor
    result["noninferiority_normal_99_9"] = (
        result["paired_normal_ci99_9"][0] >= floor
    )
    result["noninferiority_distribution_free_99_9"] = (
        result["paired_hoeffding_ci99_9"][0] >= floor
    )
    payload = {
        "schema_version": 1,
        "kind": "matched_temporal_1v1_batched_exact_engine",
        "candidate": {"path": str(paths[0]), "sha256": file_sha256(paths[0])},
        "opponent": {"path": str(paths[1]), "sha256": file_sha256(paths[1])},
        "mlp_loader_dependency": {
            "path": str(paths[2]),
            "sha256": file_sha256(paths[2]),
            "used_as_opponent": False,
        },
        "protocol": {
            "blocks": args.blocks,
            "games": args.blocks * 2,
            "candidate_seats_per_block": 2,
            "matched_seat_rotations": True,
            "deterministic_temporal_actions": True,
            "engine": "BatchedPesten exact-semantics evaluator",
            "confidence": (
                "two-sided paired normal and distribution-free Hoeffding "
                f"intervals at {100 * (1 - ALPHA):.1f}%"
            ),
            "seed_start_inclusive": args.seed,
            "seed_end_exclusive": args.seed + args.blocks,
        },
        **benchmark,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["result"], indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
