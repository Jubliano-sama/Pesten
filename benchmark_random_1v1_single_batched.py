#!/usr/bin/env python3
"""Efficient single-checkpoint 1v1 benchmark against canonical AgentRandom."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from benchmark_random_1v1_batched import _model_summary
from benchmark_rlvr_pool import file_sha256
from benchmark_rlvr_pool_batched import _evaluate_task, _initialize_worker
from temporal_batched_speed import CANONICAL_RANDOM_AGENT_CONTRACT


def run_single_benchmark(
    model: Path,
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
            "opponent": "canonical_random",
            "players": 2,
            "seed": seed + start,
            "blocks": min(chunk_size, blocks - start),
            "variants": ("after",),
        }
        for start in range(0, blocks, chunk_size)
    ]
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with context.Pool(
        processes=min(workers, len(tasks)),
        initializer=_initialize_worker,
        initargs=(str(model), str(model), str(mlp), str(model)),
    ) as pool:
        chunks_out = pool.map(_evaluate_task, tasks, chunksize=1)

    scores = []
    win_shares = []
    totals = defaultdict(int)
    for chunk in chunks_out:
        scores.extend(chunk["after_scores"])
        win_shares.extend(chunk["after_win_shares"])
        for name, value in chunk["totals"]["after"].items():
            totals[name] += int(value)
    return {
        "result": _model_summary(
            np.asarray(win_shares, dtype=np.float64),
            np.asarray(scores, dtype=np.float64),
            dict(totals),
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="model")
    parser.add_argument("--blocks", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=20)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    model = args.model.resolve()
    mlp = args.mlp.resolve()
    for path in (model, mlp):
        if not path.is_file():
            raise FileNotFoundError(path)
    result = run_single_benchmark(
        model,
        mlp,
        blocks=args.blocks,
        workers=args.workers,
        chunks=args.chunks,
        seed=args.seed,
    )
    payload = {
        "schema_version": 1,
        "kind": "single_canonical_random_1v1_batched_exact_engine",
        "model": {
            "label": args.label,
            "path": str(model),
            "sha256": file_sha256(model),
        },
        "mlp_loader_dependency": {
            "path": str(mlp),
            "sha256": file_sha256(mlp),
            "used_as_opponent": False,
        },
        "opponent": dict(CANONICAL_RANDOM_AGENT_CONTRACT),
        "protocol": {
            "blocks": args.blocks,
            "games": args.blocks * 2,
            "candidate_seats_per_block": 2,
            "deterministic_temporal_actions": True,
            "engine": "BatchedPesten exact-semantics evaluator",
            "confidence": (
                "two-sided 99.9% normal and distribution-free Hoeffding "
                "intervals over independent deal blocks"
            ),
            "seed_start_inclusive": args.seed,
            "seed_end_exclusive": args.seed + args.blocks,
        },
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({args.label: payload["result"], "elapsed_seconds": payload["elapsed_seconds"]}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
