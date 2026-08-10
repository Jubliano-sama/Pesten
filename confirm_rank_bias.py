#!/usr/bin/env python3
"""Disjoint fresh-deal confirmation for an evolved rank-bias policy."""

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
from evolutionary_rank_policy import RankBiasPolicy
from ppo2 import StudentNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--bias-checkpoint", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=2_500)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("this experiment requires CUDA")
    device = torch.device("cuda")
    base_path = args.base.resolve()
    bias_path = args.bias_checkpoint.resolve()
    parent = StudentNet().to(device)
    parent.load_state_dict(
        torch.load(base_path, map_location=device, weights_only=True)
    )
    parent.eval().requires_grad_(False)
    saved = torch.load(bias_path, map_location=device, weights_only=True)
    if saved.get("base_sha256") != file_sha256(base_path):
        raise RuntimeError("bias checkpoint base actor hash mismatch")
    candidate = RankBiasPolicy(
        parent,
        saved["biases"].to(device),
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
        max_turns=1_000,
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
        "kind": "rank_bias_disjoint_confirmation",
        "promotion_eligible": True,
        "base": {"path": str(base_path), "sha256": file_sha256(base_path)},
        "bias_checkpoint": {
            "path": str(bias_path),
            "sha256": file_sha256(bias_path),
            "biases": saved["biases"].cpu().tolist(),
        },
        "protocol": {
            "fresh_development_deals": True,
            "disjoint_from_search": True,
            "seat_balanced": True,
            "pairs_per_matchup": args.pairs,
            "base_seed": args.seed,
            "pool": ["random"],
        },
        "privacy": {
            "observation": "base actor public and own-hand 121 features only",
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
                "direct_score": payload["direct"]["candidate_score"],
                "random_candidate": pool["candidate_score"],
                "random_parent": pool["parent_score"],
                "random_paired_delta": pool["paired_delta"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

