#!/usr/bin/env python3
"""Large matched 1v1 benchmark against canonical ``AgentRandom``."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np

from benchmark_rlvr_pool import file_sha256
from benchmark_rlvr_pool_batched import _evaluate_task, _initialize_worker
from temporal_batched_speed import CANONICAL_RANDOM_AGENT_CONTRACT


ALPHA = 0.001
Z_99_9_TWO_SIDED = 3.2905267314919255


def _bounded_normal_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(values.mean())
    standard_error = (
        float(values.std(ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    return (
        max(0.0, mean - Z_99_9_TWO_SIDED * standard_error),
        min(1.0, mean + Z_99_9_TWO_SIDED * standard_error),
        standard_error,
    )


def _hoeffding_interval(mean: float, blocks: int) -> tuple[float, float]:
    margin = math.sqrt(math.log(2.0 / ALPHA) / (2.0 * blocks))
    return max(0.0, mean - margin), min(1.0, mean + margin)


def _model_summary(
    win_shares: np.ndarray,
    scores: np.ndarray,
    totals: dict[str, int],
) -> dict:
    blocks = len(win_shares)
    games = int(sum(totals.get(key, 0) for key in ("wins", "losses", "draws")))
    win_rate = float(win_shares.mean())
    score_rate = float(scores.mean())
    win_normal_low, win_normal_high, win_standard_error = (
        _bounded_normal_interval(win_shares)
    )
    score_normal_low, score_normal_high, score_standard_error = (
        _bounded_normal_interval(scores)
    )
    win_hoeffding_low, win_hoeffding_high = _hoeffding_interval(
        win_rate, blocks
    )
    return {
        "blocks": blocks,
        "games": games,
        "wins": int(totals.get("wins", 0)),
        "losses": int(totals.get("losses", 0)),
        "draws": int(totals.get("draws", 0)),
        "outright_win_rate": win_rate,
        "outright_win_rate_normal_ci99_9": [
            win_normal_low,
            win_normal_high,
        ],
        "outright_win_rate_standard_error": win_standard_error,
        "outright_win_rate_hoeffding_ci99_9": [
            win_hoeffding_low,
            win_hoeffding_high,
        ],
        "score_rate_draws_half": score_rate,
        "score_rate_normal_ci99_9": [
            score_normal_low,
            score_normal_high,
        ],
        "score_rate_standard_error": score_standard_error,
    }


def _paired_summary(before: np.ndarray, after: np.ndarray) -> dict:
    delta = after - before
    mean = float(delta.mean())
    standard_error = (
        float(delta.std(ddof=1) / math.sqrt(len(delta)))
        if len(delta) > 1
        else 0.0
    )
    # A paired block delta lies in [-1, 1], whose range width is two.
    hoeffding_margin = math.sqrt(
        2.0 * math.log(2.0 / ALPHA) / len(delta)
    )
    return {
        "after_minus_before": mean,
        "standard_error": standard_error,
        "normal_ci99_9": [
            mean - Z_99_9_TWO_SIDED * standard_error,
            mean + Z_99_9_TWO_SIDED * standard_error,
        ],
        "hoeffding_ci99_9": [
            mean - hoeffding_margin,
            mean + hoeffding_margin,
        ],
    }


def run_benchmark(
    before: Path,
    after: Path,
    mlp: Path,
    *,
    blocks: int,
    workers: int,
    chunks: int,
    seed: int,
    disable_pass_before: bool = False,
    disable_pass_after: bool = False,
    pass_only_one_card_before: bool = False,
    pass_only_one_card_after: bool = False,
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
            "disable_pass_before": disable_pass_before,
            "disable_pass_after": disable_pass_after,
            "pass_only_one_card_before": pass_only_one_card_before,
            "pass_only_one_card_after": pass_only_one_card_after,
        }
        for start in range(0, blocks, chunk_size)
    ]
    context = multiprocessing.get_context("spawn")
    started = time.perf_counter()
    with context.Pool(
        processes=min(workers, len(tasks)),
        initializer=_initialize_worker,
        initargs=(str(before), str(after), str(mlp), str(before)),
    ) as pool:
        chunks_out = pool.map(_evaluate_task, tasks, chunksize=1)

    arrays = defaultdict(list)
    totals = {
        "before": defaultdict(int),
        "after": defaultdict(int),
    }
    for chunk in chunks_out:
        for name in (
            "before_scores",
            "after_scores",
            "before_win_shares",
            "after_win_shares",
        ):
            arrays[name].extend(chunk[name])
        for variant in ("before", "after"):
            for name, value in chunk["totals"][variant].items():
                totals[variant][name] += int(value)

    before_scores = np.asarray(arrays["before_scores"], dtype=np.float64)
    after_scores = np.asarray(arrays["after_scores"], dtype=np.float64)
    before_wins = np.asarray(arrays["before_win_shares"], dtype=np.float64)
    after_wins = np.asarray(arrays["after_win_shares"], dtype=np.float64)
    return {
        "before_result": _model_summary(
            before_wins, before_scores, dict(totals["before"])
        ),
        "after_result": _model_summary(
            after_wins, after_scores, dict(totals["after"])
        ),
        "paired_outright_win_rate": _paired_summary(
            before_wins, after_wins
        ),
        "paired_score_rate": _paired_summary(before_scores, after_scores),
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-label", default="before")
    parser.add_argument("--after-label", default="after")
    parser.add_argument("--blocks", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--chunks", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--disable-pass-before", action="store_true")
    parser.add_argument("--disable-pass-after", action="store_true")
    parser.add_argument("--pass-only-one-card-before", action="store_true")
    parser.add_argument("--pass-only-one-card-after", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.disable_pass_before and args.pass_only_one_card_before:
        raise ValueError("before pass ablations are mutually exclusive")
    if args.disable_pass_after and args.pass_only_one_card_after:
        raise ValueError("after pass ablations are mutually exclusive")
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
        blocks=args.blocks,
        workers=args.workers,
        chunks=args.chunks,
        seed=args.seed,
        disable_pass_before=args.disable_pass_before,
        disable_pass_after=args.disable_pass_after,
        pass_only_one_card_before=args.pass_only_one_card_before,
        pass_only_one_card_after=args.pass_only_one_card_after,
    )
    payload = {
        "schema_version": 1,
        "kind": "matched_canonical_random_1v1_batched_exact_engine",
        "before": {
            "label": args.before_label,
            "path": str(before),
            "sha256": file_sha256(before),
        },
        "after": {
            "label": args.after_label,
            "path": str(after),
            "sha256": file_sha256(after),
        },
        "mlp_loader_dependency": {
            "path": str(mlp),
            "sha256": file_sha256(mlp),
            "used_as_opponent": False,
        },
        "opponent": dict(CANONICAL_RANDOM_AGENT_CONTRACT),
        "protocol": {
            "blocks": args.blocks,
            "games_per_model": args.blocks * 2,
            "candidate_seats_per_block": 2,
            "matched_model_seeds": True,
            "deterministic_temporal_actions": True,
            "disable_strategic_pass_before": args.disable_pass_before,
            "disable_strategic_pass_after": args.disable_pass_after,
            "strategic_pass_only_with_one_card_before": (
                args.pass_only_one_card_before
            ),
            "strategic_pass_only_with_one_card_after": (
                args.pass_only_one_card_after
            ),
            "disabled_pass_definition": (
                "mask PASS_ACTION iff at least one playable card exists; "
                "retain forced draw and penalty acceptance"
            ),
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
    print(
        json.dumps(
            {
                args.before_label: payload["before_result"],
                args.after_label: payload["after_result"],
                "paired_outright_win_rate": payload[
                    "paired_outright_win_rate"
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
