#!/usr/bin/env python3
"""Paired gameplay evaluation for an information-safe model-zoo checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark_model_zoo import default_model_specs, file_sha256
from randomAgent import AgentRandom
from smartAgent import Agent as SmartAgent, SmarterAgent
from temporal_league import (
    CHAMPION_PATH,
    evaluate_paired,
    load_ppo_agent,
    tree_factory,
)
from temporal_puffer_agent import temporal_agent_factory


def load_policy(
    checkpoint: Path,
    *,
    model_name: str,
    max_context: int,
    device: str,
):
    specs = {
        spec.name: spec
        for spec in default_model_specs(max_context=max_context)
    }
    if model_name not in specs:
        raise ValueError(f"unsupported checkpoint model: {model_name}")
    policy = specs[model_name].factory().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model", default="causal_transformer_3x128"
    )
    parser.add_argument("--max-context", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.pairs < 1:
        raise ValueError("--pairs must be positive")
    policy = load_policy(
        args.checkpoint,
        model_name=args.model,
        max_context=args.max_context,
        device=args.device,
    )
    candidate = temporal_agent_factory(
        policy, deterministic=True, seed=args.seed
    )
    opponents = {
        "ppo_champion": load_ppo_agent(CHAMPION_PATH, args.device),
        "bayesian_tree": tree_factory,
        "smart": SmartAgent,
        "smarter": SmarterAgent,
        "random": AgentRandom,
    }
    results = {
        name: evaluate_paired(
            candidate,
            opponent,
            pairs=args.pairs if name == "ppo_champion" else max(20, args.pairs // 2),
            seed=args.seed + index * 100_000,
            candidate_name=args.model,
            opponent_name=name,
        )
        for index, (name, opponent) in enumerate(opponents.items())
    }
    payload = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": file_sha256(args.checkpoint),
            "model": args.model,
            "max_context": args.max_context,
        },
        "information_boundary": (
            "DecisionInput plus redacted PublicEventToken history only"
        ),
        "paired_deals_and_rotated_seats": True,
        "development_seed": args.seed,
        "results": results,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
