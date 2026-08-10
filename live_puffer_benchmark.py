#!/usr/bin/env python3
"""Benchmark one native PufferLib v4 checkpoint against old PPO and random."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"] if "state_dict" in payload else payload
    model = payload.get("model", {})
    hidden_size = int(
        model.get("hidden_size", state["puffer_policy.encoder.encoder.weight"].shape[0])
    )
    num_layers = int(
        model.get(
            "num_layers",
            sum(
                key.startswith("puffer_policy.network.layers.")
                and key.endswith(".weight")
                for key in state
            ),
        )
    )
    mode_conditioned_heads = bool(
        model.get(
            "mode_conditioned_heads",
            any("card_and_pass" in key for key in state),
        )
    )
    from temporal_puffer_v4 import NativePufferTemporalPolicy

    policy = NativePufferTemporalPolicy(
        hidden_size=hidden_size,
        num_layers=num_layers,
        mode_conditioned_heads=mode_conditioned_heads,
    )
    policy.load_state_dict(state)
    policy.eval()
    return policy, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs < 1:
        raise ValueError("pairs must be positive")
    checkpoint = args.checkpoint.resolve()
    policy, payload = _load(checkpoint)
    from benchmark_agents import load_ppo_agent
    from randomAgent import AgentRandom
    from temporal_league import evaluate_paired
    from temporal_puffer_agent import temporal_agent_factory

    candidate = temporal_agent_factory(policy, deterministic=True, seed=args.seed)
    old = load_ppo_agent(Path("ppo_actor_100run3.pth"), "cpu")
    started = time.perf_counter()
    results = {
        "old_mlp": evaluate_paired(
            candidate,
            old,
            pairs=args.pairs,
            seed=args.seed,
            candidate_name="native_puffer_candidate",
            opponent_name="old_mlp_ppo",
        ),
        "random": evaluate_paired(
            candidate,
            AgentRandom,
            pairs=args.pairs,
            seed=args.seed + 100_000,
            candidate_name="native_puffer_candidate",
            opponent_name="random",
        ),
    }
    result = {
        "schema_version": 1,
        "kind": "live_native_puffer_benchmark",
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _hash(checkpoint),
            "epoch": payload.get("epoch"),
        },
        "protocol": {
            "pairs_per_opponent": args.pairs,
            "games_per_pair": 2,
            "seat_balanced": True,
            "fresh_development_seeds": True,
            "candidate_observation": "DecisionInput plus public event tokens only",
        },
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        name: {
            "score": value["score"],
            "ci95": value["paired_bootstrap_ci95"],
            "games": value["games"],
        }
        for name, value in results.items()
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
