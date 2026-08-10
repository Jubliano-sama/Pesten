#!/usr/bin/env python3
"""Common-deal development sweep for a fitted-Q residual policy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from batched_pesten import ActorPolicy
from benchmark_checkpoint_batched import build_game_specs, run_batched_specs
from fitted_q_policy import ActionValueNet, FittedQResidualPolicy
from ppo2 import StudentNet


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_floats(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 0 for item in values):
        raise ValueError("scales must be non-negative")
    return values


def parse_gates(value: str) -> list[float | None]:
    gates = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        gates.append(None if item == "none" else float(item))
    if not gates or any(item is not None and item < 0 for item in gates):
        raise ValueError("gates must be non-negative or none")
    return gates


def random_specs(*, pairs: int, seed: int):
    return [
        spec
        for spec in build_game_specs(
            pairs=pairs,
            base_seed=seed,
            pool_names=("random",),
        )
        if spec.section == "pool" and spec.evaluated_key == "candidate"
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--q-checkpoint", type=Path, required=True)
    parser.add_argument("--scales", default="0,.05,.1,.2,.3,.5,.75,1")
    parser.add_argument("--gates", default="none")
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.pairs < 1:
        raise ValueError("pairs must be positive")
    scales = parse_floats(args.scales)
    gates = parse_gates(args.gates)
    device = torch.device("cuda")
    base_path = args.base.resolve()
    q_path = args.q_checkpoint.resolve()
    base = StudentNet().to(device)
    base.load_state_dict(
        torch.load(base_path, map_location=device, weights_only=True)
    )
    base.eval().requires_grad_(False)
    saved = torch.load(q_path, map_location=device, weights_only=True)
    q_model = ActionValueNet(
        width=int(saved["width"]),
        blocks=int(saved["blocks"]),
    ).to(device)
    q_model.load_state_dict(saved["model"])
    q_model.eval().requires_grad_(False)
    specs = random_specs(pairs=args.pairs, seed=args.seed)

    started = time.perf_counter()
    results = []
    for gate in gates:
        for scale in scales:
            actor = FittedQResidualPolicy(
                base,
                q_model,
                scale=scale,
                normalize=args.normalize,
                confidence_gate=gate,
            ).to(device).eval()
            outcomes, throughput = run_batched_specs(
                specs,
                {
                    "candidate": ActorPolicy(
                        actor,
                        device,
                        temperature=1.0,
                        collect_data=False,
                        greedy=True,
                    )
                },
                max_turns=args.max_turns,
            )
            scores = np.asarray(
                [outcome.score for outcome in outcomes],
                dtype=np.float64,
            )
            result = {
                "scale": scale,
                "confidence_gate": gate,
                "score": float(scores.mean()),
                "wins": int(np.count_nonzero(scores == 1.0)),
                "draws": int(np.count_nonzero(scores == 0.5)),
                "losses": int(np.count_nonzero(scores == 0.0)),
                "average_turns": float(
                    np.mean([outcome.turns for outcome in outcomes])
                ),
                "throughput": throughput,
            }
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    results.sort(
        key=lambda item: (-item["score"], item["average_turns"])
    )
    payload = {
        "schema_version": 1,
        "kind": "common_deal_fitted_q_residual_screen",
        "promotion_eligible": False,
        "promotion_reason": (
            "development scale selection; requires disjoint confirmation"
        ),
        "base": {"path": str(base_path), "sha256": file_sha256(base_path)},
        "q_checkpoint": {"path": str(q_path), "sha256": file_sha256(q_path)},
        "privacy": {
            "observation": "121 public and own-hand features only",
            "opponent_card_identities": False,
            "draw_pile_order": False,
        },
        "protocol": {
            "base_seed": args.seed,
            "pairs": args.pairs,
            "games_per_configuration": len(specs),
            "seat_balanced": True,
            "common_deals": True,
            "normalize_legal_q": args.normalize,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"output": str(args.output), "top": results[:10]},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

