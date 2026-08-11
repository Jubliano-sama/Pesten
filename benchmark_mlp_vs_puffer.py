#!/usr/bin/env python3
"""Benchmark the latest warm MLP PPO against the latest warm native Puffer GRU."""

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


def _load_puffer(path: Path):
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
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--puffer", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=8_281_7041)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs < 1:
        raise ValueError("pairs must be positive")
    mlp_path = args.mlp.resolve()
    puffer_path = args.puffer.resolve()
    from benchmark_agents import load_ppo_agent
    from temporal_league import evaluate_paired
    from temporal_puffer_agent import temporal_agent_factory

    puffer_policy, puffer_payload = _load_puffer(puffer_path)
    puffer_factory = temporal_agent_factory(
        puffer_policy, deterministic=True, seed=args.seed
    )
    mlp_factory = load_ppo_agent(mlp_path, "cpu")
    started = time.perf_counter()
    results = {
        "mlp_vs_puffer": evaluate_paired(
            mlp_factory,
            puffer_factory,
            pairs=args.pairs,
            seed=args.seed,
            candidate_name="warm_mlp_ppo",
            opponent_name="warm_puffer_gru",
        ),
        "puffer_vs_mlp": evaluate_paired(
            puffer_factory,
            mlp_factory,
            pairs=args.pairs,
            seed=args.seed + 1_000_000,
            candidate_name="warm_puffer_gru",
            opponent_name="warm_mlp_ppo",
        ),
    }
    result = {
        "schema_version": 1,
        "kind": "warm_mlp_vs_warm_puffer_cross_benchmark",
        "mlp_checkpoint": {
            "path": str(mlp_path),
            "sha256": _hash(mlp_path),
        },
        "puffer_checkpoint": {
            "path": str(puffer_path),
            "sha256": _hash(puffer_path),
            "epoch": puffer_payload.get("epoch"),
        },
        "protocol": {
            "pairs_per_direction": args.pairs,
            "games_per_pair": 2,
            "seat_balanced": True,
            "fresh_development_seeds": True,
            "seed": args.seed,
        },
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                name: {
                    "score": value["score"],
                    "ci95": value["paired_bootstrap_ci95"],
                    "games": value["games"],
                    "pairs": value["pairs"],
                }
                for name, value in results.items()
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
