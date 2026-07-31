#!/usr/bin/env python3
"""Audit the exact CPU TorchScript fast path used by the best player."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from ai_observation import (
    COMPATIBLE_END,
    COMPATIBLE_START,
    MODE_INDEX,
    OBSERVATION_SIZE,
)
from best_player import BestPlayerConfig, load_best_player
from ppo2 import build_action_masks


def _observations(
    batch_size: int,
    *,
    seed: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    observations = torch.randn(
        batch_size,
        OBSERVATION_SIZE,
        generator=generator,
    )
    observations[:, COMPATIBLE_START:COMPATIBLE_END] = (
        observations[:, COMPATIBLE_START:COMPATIBLE_END] > 0
    )
    observations[::2, MODE_INDEX] = 0
    observations[1::2, MODE_INDEX] = 1
    return observations


def _median_seconds(actor, observations, *, repetitions: int, rounds: int):
    samples = []
    with torch.inference_mode():
        actor(observations)
        for _ in range(rounds):
            started = time.perf_counter()
            for _ in range(repetitions):
                actor(observations)
            samples.append(
                (time.perf_counter() - started) / repetitions
            )
    return statistics.median(samples)


def run_audit(*, rounds: int, repetitions: int) -> dict:
    eager = load_best_player(
        BestPlayerConfig(device="cpu", trace_cpu_actor=False)
    )
    traced = load_best_player(
        BestPlayerConfig(device="cpu", trace_cpu_actor=True)
    )
    rows = []
    for batch_size in (1, 7, 64):
        observations = _observations(
            batch_size,
            seed=91_733_041 + batch_size,
        )
        with torch.inference_mode():
            expected = eager.actor(observations)
            actual = traced.actor(observations)
        masks = build_action_masks(observations)
        expected_actions = expected.masked_fill(
            ~masks, -torch.inf
        ).argmax(-1)
        actual_actions = actual.masked_fill(
            ~masks, -torch.inf
        ).argmax(-1)
        logits_exact = torch.equal(actual, expected)
        actions_exact = torch.equal(actual_actions, expected_actions)
        if not logits_exact or not actions_exact:
            raise RuntimeError(
                f"trace parity failed at batch size {batch_size}"
            )
        batch_repetitions = max(2, repetitions // batch_size)
        eager_seconds = _median_seconds(
            eager.actor,
            observations,
            repetitions=batch_repetitions,
            rounds=rounds,
        )
        traced_seconds = _median_seconds(
            traced.actor,
            observations,
            repetitions=batch_repetitions,
            rounds=rounds,
        )
        rows.append(
            {
                "batch_size": batch_size,
                "repetitions_per_round": batch_repetitions,
                "logits_bit_exact": logits_exact,
                "actions_exact": actions_exact,
                "eager_milliseconds": 1_000 * eager_seconds,
                "traced_milliseconds": 1_000 * traced_seconds,
                "speedup": eager_seconds / traced_seconds,
            }
        )
    return {
        "kind": "best_player_cpu_trace_performance_audit",
        "checkpoint_sha256": traced.metadata["checkpoint"]["sha256"],
        "torch_threads": torch.get_num_threads(),
        "aggregation": "24-way ordinary-suit mean logits",
        "base_actor_execution": {
            "reference": "eager",
            "optimized": "torchscript_trace",
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=256)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.rounds, args.repetitions, args.torch_threads) < 1:
        parser.error("rounds, repetitions, and torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    payload = run_audit(
        rounds=args.rounds,
        repetitions=args.repetitions,
    )
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized)
    print(serialized, end="")


if __name__ == "__main__":
    main()
