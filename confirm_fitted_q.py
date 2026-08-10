#!/usr/bin/env python3
"""Disjoint fresh-deal confirmation for one fitted-Q residual candidate."""

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
from benchmark_fitted_q import file_sha256
from fitted_q_policy import ActionValueNet, FittedQResidualPolicy
from ppo2 import StudentNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--gate", type=float)
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--pairs", type=int, default=2_500)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.pairs < 1 or args.bootstrap_samples < 1:
        raise ValueError("pairs and bootstrap samples must be positive")
    if args.scale < 0 or (args.gate is not None and args.gate < 0):
        raise ValueError("scale and gate must be non-negative")
    device = torch.device("cuda")
    base_path = args.base.resolve()
    q_path = args.q_checkpoint.resolve()

    parent = StudentNet().to(device)
    parent.load_state_dict(
        torch.load(base_path, map_location=device, weights_only=True)
    )
    parent.eval().requires_grad_(False)
    saved = torch.load(q_path, map_location=device, weights_only=True)
    action_value = ActionValueNet(
        width=int(saved["width"]),
        blocks=int(saved["blocks"]),
    ).to(device)
    action_value.load_state_dict(saved["model"])
    action_value.eval().requires_grad_(False)
    candidate = FittedQResidualPolicy(
        parent,
        action_value,
        scale=args.scale,
        normalize=args.normalize,
        confidence_gate=args.gate,
    ).to(device).eval()

    specs = build_game_specs(
        pairs=args.pairs,
        base_seed=args.seed,
        pool_names=("random",),
    )
    started = time.perf_counter()
    outcomes, throughput = run_batched_specs(
        specs,
        {
            "candidate": ActorPolicy(
                candidate,
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
    benchmark = assemble_benchmark_payload(
        specs,
        outcomes,
        pairs=args.pairs,
        base_seed=args.seed,
        pool_names=("random",),
        bootstrap_samples=args.bootstrap_samples,
    )
    payload = {
        "schema_version": 1,
        "kind": "fitted_q_residual_disjoint_confirmation",
        "promotion_eligible": True,
        "candidate": {
            "scale": args.scale,
            "confidence_gate": args.gate,
            "normalize_legal_q": args.normalize,
        },
        "base": {"path": str(base_path), "sha256": file_sha256(base_path)},
        "q_checkpoint": {"path": str(q_path), "sha256": file_sha256(q_path)},
        "protocol": {
            "fresh_development_deals": True,
            "disjoint_from_configuration_sweep": True,
            "seat_balanced": True,
            "pairs_per_matchup": args.pairs,
            "base_seed": args.seed,
            "pool": ["random"],
        },
        "privacy": {
            "observation": "121 public and own-hand features only",
            "opponent_card_identities": False,
            "draw_pile_order": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "throughput": throughput,
        **benchmark,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pool = payload["pool"][0]
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate": payload["candidate"],
                "direct": payload["direct"],
                "random": {
                    "candidate_score": pool["candidate_score"],
                    "parent_score": pool["parent_score"],
                    "paired_delta": pool["paired_delta"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

