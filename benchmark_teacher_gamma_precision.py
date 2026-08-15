#!/usr/bin/env python3
"""One-pass, fixed-sample precision evaluation for matched gamma teachers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from benchmark_rlvr_pool import OPPONENTS, PLAYER_WEIGHTS, file_sha256
from benchmark_rlvr_pool_batched import run_multi_policy_benchmark


MODEL_LABEL = re.compile(r"^[a-z][a-z0-9_]*$")
CONFIDENCE_SPECS = {
    "0.95": (1.959963984540054, "95 percent"),
    "0.99": (2.5758293035489004, "99 percent"),
    "0.999": (3.2905267314919255, "99.9 percent"),
}


def parse_model(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not MODEL_LABEL.fullmatch(label) or not raw_path:
        raise argparse.ArgumentTypeError(
            "model must be a safe label and path formatted label=checkpoint"
        )
    return label, Path(raw_path)


def load_allocation(path: Path) -> tuple[dict[tuple[str, int], int], dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("kind") not in {
        "teacher_gamma_fixed_precision_allocation_v1",
        "teacher_gamma_fixed_precision_allocation_v2",
    }:
        raise ValueError("unsupported gamma precision allocation format")
    rows = payload.get("allocations")
    if not isinstance(rows, dict):
        raise ValueError("allocation has no strata")
    allocation = {}
    for opponent in OPPONENTS:
        counts = rows.get(opponent)
        if not isinstance(counts, dict):
            raise ValueError(f"allocation is missing opponent {opponent}")
        for players in range(2, 9):
            value = counts.get(str(players))
            if not isinstance(value, int) or value < 2:
                raise ValueError(
                    f"invalid allocation for {opponent}/{players}: {value}"
                )
            allocation[(opponent, players)] = value
    return allocation, payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        type=parse_model,
        required=True,
        help="candidate label and checkpoint; repeat for every gamma arm",
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--self-anchor", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--allocation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=14)
    parser.add_argument("--chunks-per-stratum", type=int, default=96)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--confidence",
        choices=tuple(CONFIDENCE_SPECS),
        default="0.999",
        help="paired normal confidence level used for the precision gate",
    )
    parser.add_argument(
        "--target-normalized-ci-half-width", type=float, default=0.005
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    policies = {}
    for label, raw_path in args.model:
        if label in policies:
            raise ValueError(f"duplicate model label: {label}")
        policies[label] = raw_path.resolve()
    if len(policies) < 2:
        raise ValueError("the gamma precision run requires at least two arms")
    if args.reference not in policies:
        raise ValueError("reference must name one --model arm")
    if args.workers < 1 or args.chunks_per_stratum < 1:
        raise ValueError("worker and chunk counts must be positive")
    if args.target_normalized_ci_half_width <= 0:
        raise ValueError("confidence-interval half-width must be positive")

    self_anchor = args.self_anchor.resolve()
    mlp = args.mlp.resolve()
    allocation_path = args.allocation.resolve()
    for path in (*policies.values(), self_anchor, mlp, allocation_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    allocation, allocation_payload = load_allocation(allocation_path)
    plan_confidence = str(
        allocation_payload.get("confidence_level", 0.999)
    )
    if plan_confidence != args.confidence:
        raise ValueError(
            "allocation confidence does not match requested confidence"
        )
    planned_target = float(
        allocation_payload.get(
            "target_normalized_ci_half_width",
            allocation_payload.get("target_normalized_ci99_9_half_width"),
        )
    )
    if planned_target > args.target_normalized_ci_half_width:
        raise ValueError("allocation was planned for a looser precision target")

    self_anchor_name = "self_anchor"
    if self_anchor_name in policies:
        raise ValueError(f"reserved model label: {self_anchor_name}")
    runtime_policies = {**policies, self_anchor_name: self_anchor}
    normal_quantile, confidence_label = CONFIDENCE_SPECS[args.confidence]
    result = run_multi_policy_benchmark(
        runtime_policies,
        mlp,
        self_anchor_name=self_anchor_name,
        reference_name=args.reference,
        blocks_by_stratum=allocation,
        workers=args.workers,
        chunks_per_stratum=args.chunks_per_stratum,
        seed=args.seed,
        target_normalized_ci_half_width=(
            args.target_normalized_ci_half_width
        ),
        normal_quantile=normal_quantile,
        confidence_label=confidence_label,
    )
    output = args.output.resolve()
    payload = {
        "schema_version": 1,
        "kind": "teacher_gamma_multiarm_fixed_precision_evaluation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            label: {"path": str(path), "sha256": file_sha256(path)}
            for label, path in policies.items()
        },
        "reference": args.reference,
        "self_anchor": {
            "path": str(self_anchor),
            "sha256": file_sha256(self_anchor),
        },
        "mlp": {"path": str(mlp), "sha256": file_sha256(mlp)},
        "allocation": {
            "path": str(allocation_path),
            "sha256": file_sha256(allocation_path),
            "payload": allocation_payload,
        },
        "protocol": {
            "opponents": list(OPPONENTS),
            "player_weights": {
                str(key): value for key, value in PLAYER_WEIGHTS.items()
            },
            "seed": args.seed,
            "workers": args.workers,
            "chunks_per_stratum": args.chunks_per_stratum,
            "same_deals_and_seat_rotations_for_every_arm": True,
            "fixed_self_anchor": True,
            "deterministic_temporal_actions": True,
            "confidence": f"paired normal {confidence_label}",
            "confidence_level": float(args.confidence),
            "normal_quantile": normal_quantile,
            "target_normalized_ci_half_width": (
                args.target_normalized_ci_half_width
            ),
            "stopping_rule": (
                "fixed allocation declared before corrected-model evaluation; "
                "no optional extension based on effect or significance"
            ),
        },
        **result,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        candidate: row["weighted_result"]
        for candidate, row in result["pairwise"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
