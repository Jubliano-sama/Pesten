#!/usr/bin/env python3
"""Profile exact production suit averaging at small and large batch sizes."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from ai_observation import (
    COMPATIBLE_END,
    COMPATIBLE_START,
    MODE_INDEX,
    OBSERVATION_SIZE,
)
from batched_suit_symmetry import (
    SUIT_PERMUTATIONS,
    BatchedSuitAveragedActor,
)
from best_player import (
    OPT_IN_CPU_BASE_CHUNK_SIZE,
    BestPlayerConfig,
    load_best_player,
)
from ppo2 import build_action_masks


FROZEN_BASE_CHUNK_SIZE = 512


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_actor(
    actor,
    observations,
    *,
    repetitions: int,
    rounds: int,
) -> float:
    samples = []
    for _ in range(rounds):
        _synchronize(observations.device)
        started = time.perf_counter()
        for _ in range(repetitions):
            actor(observations)
        _synchronize(observations.device)
        samples.append(
            (time.perf_counter() - started) / repetitions
        )
    return statistics.median(samples)


def _observations(
    batch_size: int,
    *,
    seed: int,
    device: torch.device,
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
    return observations.to(device)


def _largest_activation_bytes(
    base_actor,
    *,
    batch_size: int,
    base_chunk_size: int,
) -> int | None:
    dense_actor = getattr(base_actor, "model", base_actor)
    layers = getattr(dense_actor, "layers", None)
    if layers is None:
        return None
    feature_width = OBSERVATION_SIZE + sum(
        layer.out_features for layer in layers
    )
    expanded_batch = (
        min(batch_size, base_chunk_size) * len(SUIT_PERMUTATIONS)
    )
    return expanded_batch * feature_width * 4


def run_profile(
    *,
    batch_sizes: tuple[int, ...],
    rounds: int,
    device_name: str,
) -> dict:
    loaded = load_best_player(
        BestPlayerConfig(
            device=device_name,
            base_chunk_size=FROZEN_BASE_CHUNK_SIZE,
        )
    )
    base_actor = loaded.actor.actor
    device = loaded.device
    optimized_chunk_size = (
        OPT_IN_CPU_BASE_CHUNK_SIZE
        if device.type == "cpu"
        else FROZEN_BASE_CHUNK_SIZE
    )
    reference = BatchedSuitAveragedActor(
        base_actor,
        base_chunk_size=FROZEN_BASE_CHUNK_SIZE,
    ).to(device).eval()
    optimized = BatchedSuitAveragedActor(
        base_actor,
        base_chunk_size=optimized_chunk_size,
    ).to(device).eval()
    rows = []
    with torch.inference_mode():
        for batch_size in batch_sizes:
            observations = _observations(
                batch_size,
                seed=87_617_031 + batch_size,
                device=device,
            )
            expected = reference(observations)
            actual = optimized(observations)
            logits_exact = torch.equal(actual, expected)
            masks = build_action_masks(observations)
            expected_actions = expected.masked_fill(
                ~masks, -torch.inf
            ).argmax(-1)
            actual_actions = actual.masked_fill(
                ~masks, -torch.inf
            ).argmax(-1)
            actions_exact = torch.equal(actual_actions, expected_actions)
            if not logits_exact or not actions_exact:
                raise RuntimeError(
                    f"exact parity failed at batch size {batch_size}"
                )

            reference(observations)
            optimized(observations)
            repetitions = max(2, min(100, 512 // batch_size))
            reference_seconds = _time_actor(
                reference,
                observations,
                repetitions=repetitions,
                rounds=rounds,
            )
            optimized_seconds = _time_actor(
                optimized,
                observations,
                repetitions=repetitions,
                rounds=rounds,
            )
            reference_activation = _largest_activation_bytes(
                base_actor,
                batch_size=batch_size,
                base_chunk_size=FROZEN_BASE_CHUNK_SIZE,
            )
            optimized_activation = _largest_activation_bytes(
                base_actor,
                batch_size=batch_size,
                base_chunk_size=optimized_chunk_size,
            )
            rows.append(
                {
                    "batch_size": batch_size,
                    "repetitions_per_round": repetitions,
                    "logits_bit_exact": logits_exact,
                    "actions_exact": actions_exact,
                    "reference_milliseconds": reference_seconds * 1_000,
                    "optimized_milliseconds": optimized_seconds * 1_000,
                    "speedup": reference_seconds / optimized_seconds,
                    "largest_actor_activation_bytes": {
                        "reference": reference_activation,
                        "optimized": optimized_activation,
                    },
                }
            )
    return {
        "device": str(device),
        "torch_threads": torch.get_num_threads(),
        "checkpoint_sha256": loaded.metadata["checkpoint"]["sha256"],
        "aggregation": "24-way ordinary-suit mean logits",
        "reference_base_chunk_size": FROZEN_BASE_CHUNK_SIZE,
        "optimized_base_chunk_size": optimized_chunk_size,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", default="1,64,512")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.rounds < 1 or args.torch_threads < 1:
        parser.error("--rounds and --torch-threads must be positive")
    batch_sizes = tuple(
        int(value.strip())
        for value in args.batch_sizes.split(",")
        if value.strip()
    )
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        parser.error("--batch-sizes must contain positive integers")
    torch.set_num_threads(args.torch_threads)
    print(
        json.dumps(
            run_profile(
                batch_sizes=batch_sizes,
                rounds=args.rounds,
                device_name=args.device,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
