#!/usr/bin/env python3
"""Fresh-deal benchmark for a weighted ensemble of feed-forward PPO actors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from batched_pesten import ActorPolicy
from benchmark_checkpoint_batched import (
    assemble_benchmark_payload,
    build_game_specs,
    run_batched_specs,
)
from benchmark_hybrid import file_sha256
from hybrid_agent import load_frozen_ppo_actor


class WeightedCheckpointEnsemble(torch.nn.Module):
    """Combine compatible actor logits or probabilities at inference time."""

    def __init__(
        self,
        actors: list[torch.nn.Module],
        weights: list[float],
        *,
        aggregation: str,
    ) -> None:
        super().__init__()
        if len(actors) < 2:
            raise ValueError("an ensemble requires at least two actors")
        if len(actors) != len(weights):
            raise ValueError("actor and weight counts differ")
        if aggregation not in ("logits", "probabilities"):
            raise ValueError("aggregation must be logits or probabilities")
        normalized = torch.as_tensor(weights, dtype=torch.float32)
        if not torch.isfinite(normalized).all() or (normalized < 0).any():
            raise ValueError("weights must be finite and non-negative")
        if float(normalized.sum()) <= 0:
            raise ValueError("at least one weight must be positive")
        normalized /= normalized.sum()
        self.actors = torch.nn.ModuleList(actors)
        self.aggregation = aggregation
        self.register_buffer("weights", normalized, persistent=True)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        members = torch.stack(
            [actor(observations) for actor in self.actors],
            dim=0,
        )
        weights = self.weights.to(
            device=members.device,
            dtype=members.dtype,
        ).view(-1, 1, 1)
        if self.aggregation == "logits":
            return (members * weights).sum(dim=0)
        probabilities = (
            torch.softmax(members, dim=-1) * weights
        ).sum(dim=0)
        return probabilities.clamp_min(
            torch.finfo(probabilities.dtype).tiny
        ).log()


def parse_csv_paths(value: str) -> list[Path]:
    paths = [Path(item.strip()) for item in value.split(",") if item.strip()]
    if len(paths) < 2:
        raise ValueError("--models requires at least two paths")
    return paths


def parse_csv_floats(value: str | None, count: int) -> list[float]:
    if value is None:
        return [1.0] * count
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(weights) != count:
        raise ValueError("--weights must match --models")
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True)
    parser.add_argument("--weights")
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument(
        "--aggregation",
        choices=("logits", "probabilities"),
        default="logits",
    )
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pool", default="random")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_paths = [path.resolve() for path in parse_csv_paths(args.models)]
    weights = parse_csv_floats(args.weights, len(model_paths))
    pool_names = tuple(
        item.strip() for item in args.pool.split(",") if item.strip()
    )
    actors = []
    device = None
    for path in model_paths:
        actor, actor_device = load_frozen_ppo_actor(path, args.device)
        if device is None:
            device = actor_device
        elif actor_device != device:
            raise RuntimeError("ensemble actors were loaded on different devices")
        actors.append(actor)
    parent, parent_device = load_frozen_ppo_actor(args.parent, args.device)
    if parent_device != device:
        raise RuntimeError("parent was loaded on a different device")

    ensemble = WeightedCheckpointEnsemble(
        actors,
        weights,
        aggregation=args.aggregation,
    ).to(device).eval()
    ensemble.requires_grad_(False)
    specs = build_game_specs(
        pairs=args.pairs,
        base_seed=args.seed,
        pool_names=pool_names,
    )
    started = time.perf_counter()
    outcomes, throughput = run_batched_specs(
        specs,
        {
            "candidate": ActorPolicy(
                ensemble,
                device,
                temperature=1.0,
                collect_data=False,
                greedy=True,
            ),
            "parent": ActorPolicy(
                parent,
                device,
                temperature=1.0,
                collect_data=False,
                greedy=True,
            ),
        },
        max_turns=args.max_turns,
    )
    result = assemble_benchmark_payload(
        specs,
        outcomes,
        pairs=args.pairs,
        base_seed=args.seed,
        pool_names=pool_names,
        bootstrap_samples=args.bootstrap_samples,
    )
    payload = {
        "schema_version": 1,
        "kind": "weighted_checkpoint_ensemble",
        "models": [
            {
                "path": str(path),
                "sha256": file_sha256(path),
                "weight": float(ensemble.weights[index]),
            }
            for index, path in enumerate(model_paths)
        ],
        "parent": {
            "path": str(args.parent.resolve()),
            "sha256": file_sha256(args.parent),
        },
        "aggregation": args.aggregation,
        "device": str(device),
        "protocol": {
            "fresh_development_deals": True,
            "seat_balanced": True,
            "pairs_per_matchup": args.pairs,
            "base_seed": args.seed,
            "pool": list(pool_names),
        },
        "privacy": {
            "observation": "121 public and own-hand features only",
            "opponent_card_identities": False,
            "draw_pile_order": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "throughput": throughput,
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
                "output": str(args.output),
                "models": [str(path) for path in model_paths],
                "weights": ensemble.weights.tolist(),
                "aggregation": args.aggregation,
                "direct": {
                    "candidate_score": payload["direct"]["candidate_score"],
                    "candidate_advantage": payload["direct"][
                        "candidate_advantage"
                    ],
                },
                "pool": [
                    {
                        "opponent": item["opponent"],
                        "candidate_score": item["candidate_score"],
                        "paired_delta": item["paired_delta"],
                    }
                    for item in payload["pool"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
