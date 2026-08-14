#!/usr/bin/env python3
"""Seat-balanced exact-engine benchmark against the completed league catalog."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmark_random_1v1_batched import _model_summary
from benchmark_rlvr_pool import file_sha256
from benchmark_rlvr_pool_batched import _evaluate_task, _initialize_worker
from temporal_batched_speed import CANONICAL_RANDOM_AGENT_CONTRACT


OPPONENTS = {
    "canonical_random": "canonical_random",
    "smart_rule": "smart_rule",
    "smarter_rule": "smarter_rule",
    "deployed_mlp": "mlp",
    "pre_rlvr_distilled_gru_student": "self_anchor",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pre-rlvr-student", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--chunks-per-opponent", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    pre_rlvr_student = args.pre_rlvr_student.resolve()
    mlp = args.mlp.resolve()
    for path in (candidate, pre_rlvr_student, mlp):
        if not path.is_file():
            raise FileNotFoundError(path)
    if min(args.blocks, args.workers, args.chunks_per_opponent) < 1:
        raise ValueError("benchmark counts must be positive")

    chunk_size = math.ceil(args.blocks / args.chunks_per_opponent)
    tasks = []
    for label, engine_opponent in OPPONENTS.items():
        for start in range(0, args.blocks, chunk_size):
            tasks.append(
                {
                    "label": label,
                    "opponent": engine_opponent,
                    "players": 2,
                    "seed": args.seed + start,
                    "blocks": min(chunk_size, args.blocks - start),
                    "variants": ("after",),
                }
            )

    started = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        processes=min(args.workers, len(tasks)),
        initializer=_initialize_worker,
        initargs=(
            str(candidate),
            str(candidate),
            str(mlp),
            str(pre_rlvr_student),
        ),
    ) as pool:
        chunks = pool.map(_evaluate_task, tasks, chunksize=1)

    grouped = {
        label: {
            "win_shares": [],
            "scores": [],
            "totals": defaultdict(int),
        }
        for label in OPPONENTS
    }
    for task, chunk in zip(tasks, chunks):
        target = grouped[task["label"]]
        target["win_shares"].extend(chunk["after_win_shares"])
        target["scores"].extend(chunk["after_scores"])
        for name, value in chunk["totals"]["after"].items():
            target["totals"][name] += int(value)

    results = {}
    for label, values in grouped.items():
        results[label] = _model_summary(
            np.asarray(values["win_shares"], dtype=np.float64),
            np.asarray(values["scores"], dtype=np.float64),
            dict(values["totals"]),
        )

    payload = {
        "schema_version": 1,
        "kind": "confirmed_factorized_student_1v1_opponent_catalog",
        "candidate": {
            "path": str(candidate),
            "sha256": file_sha256(candidate),
        },
        "pre_rlvr_gru_baseline": {
            "representation": "perfectly_distilled_factorized_23_student",
            "path": str(pre_rlvr_student),
            "sha256": file_sha256(pre_rlvr_student),
            "original_full_action_gru_counted_separately": False,
        },
        "mlp": {"path": str(mlp), "sha256": file_sha256(mlp)},
        "opponents": {
            "canonical_random": dict(CANONICAL_RANDOM_AGENT_CONTRACT),
            "smart_rule": {"engine_role": "smart_rule"},
            "smarter_rule": {"engine_role": "smarter_rule"},
            "deployed_mlp": {"engine_role": "mlp"},
            "pre_rlvr_distilled_gru_student": {
                "engine_role": "self_anchor"
            },
        },
        "protocol": {
            "blocks_per_opponent": args.blocks,
            "games_per_opponent": args.blocks * 2,
            "candidate_seats_per_block": 2,
            "common_fresh_deal_bank_across_opponents": True,
            "seed_start_inclusive": args.seed,
            "seed_end_exclusive": args.seed + args.blocks,
            "deterministic_neural_actions": True,
            "engine": "BatchedPesten exact-semantics evaluator",
            "confidence": (
                "two-sided 99.9% normal and distribution-free Hoeffding "
                "intervals over independent deal blocks"
            ),
        },
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
