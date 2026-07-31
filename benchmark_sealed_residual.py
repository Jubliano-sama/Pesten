#!/usr/bin/env python3
"""Guarded sealed holdout evaluation for residual temporal checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

from benchmark_hybrid import file_sha256
from benchmark_residual_checkpoint import (
    CHAMPION_PATH,
    DEFAULT_OPPONENT_NAMES,
    _default_evaluation_dependencies,
    deterministic_evaluation_agent_factory,
    deterministic_evaluation_runtime,
    evaluate_candidate_and_base_paired,
    filter_opponents,
    load_residual_checkpoint,
    parse_opponent_names,
    stratified_paired_bootstrap_interval,
)
from benchmark_sealed_checkpoint import (
    SEALED_SEED_END,
    SEALED_SEED_START,
    refuse_output_overwrite,
    verify_candidate_identity,
)
from temporal_observation import assert_policy_schema_is_hidden_state_free
from seed_integrity import (
    DEFAULT_SEALED_CLAIM_LEDGER,
    claim_sealed_evaluation,
)


def require_sealed_confirmation(confirmed: bool) -> None:
    if confirmed is not True:
        raise ValueError("explicit sealed evaluation confirmation is required")


def sealed_residual_seed_plan(
    *,
    pairs: int,
    base_seed: int,
    opponent_names: Sequence[str] = DEFAULT_OPPONENT_NAMES,
) -> dict[str, range]:
    """Allocate one contiguous, collision-free sealed block per opponent."""
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if not SEALED_SEED_START <= base_seed < SEALED_SEED_END:
        raise ValueError(
            f"base seed must be in [{SEALED_SEED_START}, {SEALED_SEED_END})"
        )
    names = tuple(str(name).strip().lower() for name in opponent_names)
    if not names or any(not name for name in names):
        raise ValueError("at least one opponent must be selected")
    if len(names) != len(set(names)):
        raise ValueError("sealed opponent names must be unique")
    unknown = sorted(set(names) - set(DEFAULT_OPPONENT_NAMES))
    if unknown:
        raise ValueError(f"unknown opponent(s): {', '.join(unknown)}")
    required_seeds = pairs * len(names)
    if base_seed + required_seeds > SEALED_SEED_END:
        raise ValueError(
            "sealed opponent allocation exceeds the reserved interval "
            f"[{SEALED_SEED_START}, {SEALED_SEED_END})"
        )
    return {
        f"opponent:{name}": range(
            base_seed + index * pairs,
            base_seed + (index + 1) * pairs,
        )
        for index, name in enumerate(names)
    }


def assert_residual_information_boundary(policy) -> None:
    """Fail closed unless the loaded policy advertises the safe contract."""
    assert_policy_schema_is_hidden_state_free()
    if getattr(policy, "critic_privileged", None) is not False:
        raise RuntimeError(
            "sealed residual policy must have critic_privileged=False"
        )
    architecture = getattr(policy, "architecture", "")
    if not isinstance(architecture, str) or not architecture:
        raise RuntimeError("sealed residual policy architecture is missing")


def _plan_metadata(plan: Mapping[str, range]) -> dict[str, dict[str, int]]:
    return {
        namespace: {
            "start_inclusive": seeds.start,
            "end_exclusive": seeds.stop,
            "count": len(seeds),
        }
        for namespace, seeds in plan.items()
    }


def evaluate_sealed_residual(
    *,
    checkpoint: Path,
    expected_sha256: str,
    champion_path: Path,
    pairs: int,
    base_seed: int,
    opponent_names: Sequence[str],
    device: str,
    cpu_threads: int,
    confirmed: bool,
    output_path: Path,
    claim_ledger_path: Path = DEFAULT_SEALED_CLAIM_LEDGER,
) -> dict:
    """Run the sealed comparison after all caller-side guards have passed."""
    require_sealed_confirmation(confirmed)
    actual_sha256 = verify_candidate_identity(
        checkpoint,
        expected_sha256,
    )
    plan = sealed_residual_seed_plan(
        pairs=pairs,
        base_seed=base_seed,
        opponent_names=opponent_names,
    )
    champion_sha256 = file_sha256(champion_path)
    sealed_claim = claim_sealed_evaluation(
        evaluator="residual_temporal_checkpoint",
        candidate_path=checkpoint,
        candidate_sha256=actual_sha256,
        base_seed=base_seed,
        pairs=pairs,
        output_path=output_path,
        ledger_path=claim_ledger_path,
        configuration={
            "champion_path": str(champion_path.resolve()),
            "champion_sha256": champion_sha256,
            "opponents": list(opponent_names),
            "device": device,
            "cpu_threads": cpu_threads,
        },
    )
    policy = load_residual_checkpoint(
        checkpoint,
        champion_path=champion_path,
        map_location=device,
    ).eval()
    verify_candidate_identity(checkpoint, actual_sha256)
    if file_sha256(champion_path) != champion_sha256:
        raise RuntimeError(
            "champion checkpoint changed after the sealed claim was created"
        )
    assert_residual_information_boundary(policy)
    candidate_factory = deterministic_evaluation_agent_factory(
        policy,
        seed=base_seed,
    )
    _, available_opponents = _default_evaluation_dependencies(
        champion_path,
        device,
    )
    opponents = filter_opponents(
        available_opponents,
        opponent_names,
    )
    base_factory = available_opponents["ppo_champion"]

    blocks = {}
    with deterministic_evaluation_runtime(
        device,
        cpu_threads=cpu_threads,
    ):
        for name, opponent in opponents.items():
            block_seed = plan[f"opponent:{name}"].start
            blocks[name] = evaluate_candidate_and_base_paired(
                candidate_factory,
                base_factory,
                opponent,
                pairs=pairs,
                seed=block_seed,
                candidate_name=policy.architecture,
                opponent_name=name,
            )

    results = {
        name: block["candidate_result"]
        for name, block in blocks.items()
    }
    base_results = {
        name: block["base_result"]
        for name, block in blocks.items()
    }
    comparisons = {
        name: block["comparison"]
        for name, block in blocks.items()
    }
    candidate_scores = [
        score
        for block in blocks.values()
        for score in block["_candidate_pair_scores"]
    ]
    base_scores = [
        score
        for block in blocks.values()
        for score in block["_base_pair_scores"]
    ]
    delta_groups = {
        name: block["_pair_deltas"]
        for name, block in blocks.items()
    }
    all_deltas = [
        delta
        for deltas in delta_groups.values()
        for delta in deltas
    ]
    # This bootstrap seed never initializes a game or enters a deal plan.
    bootstrap_seed = base_seed - SEALED_SEED_START
    combined_ci = stratified_paired_bootstrap_interval(
        delta_groups,
        seed=bootstrap_seed,
    )
    return {
        "schema_version": 1,
        "engine": "canonical Game paired residual sealed evaluator",
        "sealed_holdout": {
            "interval_start_inclusive": SEALED_SEED_START,
            "interval_end_exclusive": SEALED_SEED_END,
            "one_way": True,
            "collision_free_namespace_plan": _plan_metadata(plan),
            "bootstrap_seed_not_used_for_games": bootstrap_seed,
            "global_claim": sealed_claim,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "format": policy.loaded_checkpoint_format,
        },
        "policy": {
            "class": type(policy).__name__,
            "architecture": policy.architecture,
            "recurrent_cell": getattr(policy, "cell", None),
            "critic_privileged": policy.critic_privileged,
        },
        "base_policy": {
            "path": str(champion_path),
            "sha256": champion_sha256,
            "frozen": True,
        },
        "information_boundary": (
            "DecisionInput plus redacted PublicEventToken history and own hand "
            "only; no opponent cards or draw-pile order"
        ),
        "human_visible_information_only": True,
        "paired_deals_and_rotated_seats": True,
        "pairs_per_opponent": pairs,
        "base_seed": base_seed,
        "opponents": list(opponents),
        "results": results,
        "base_results": base_results,
        "paired_candidate_minus_base": comparisons,
        "combined_pool_delta": {
            "opponents": list(opponents),
            "pairs": len(all_deltas),
            "weighting": "one vote per opponent-stratified paired deal",
            "same_deals_and_rotated_seats": True,
            "candidate_score": float(np.mean(candidate_scores)),
            "base_score": float(np.mean(base_scores)),
            "candidate_minus_base_score": float(np.mean(all_deltas)),
            "paired_bootstrap_ci95": list(combined_ci),
            "delta_gate_passed": combined_ci[0] > 0.0,
            "bootstrap_seed_not_used_for_games": bootstrap_seed,
        },
        "evaluation_runtime": {
            "deterministic_fast_path": True,
            "records_training_trajectories": False,
            "cpu_threads": (
                cpu_threads if torch.device(device).type == "cpu" else None
            ),
        },
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--champion", type=Path, default=CHAMPION_PATH)
    parser.add_argument("--pairs", type=int, default=1_000)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument(
        "--opponents",
        type=parse_opponent_names,
        default=DEFAULT_OPPONENT_NAMES,
        metavar="NAME[,NAME...]",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-sealed-evaluation",
        action="store_true",
        help="required acknowledgement that the sealed holdout is one-way",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.cpu_threads < 1:
        raise SystemExit("--cpu-threads must be positive")
    try:
        require_sealed_confirmation(args.confirm_sealed_evaluation)
        refuse_output_overwrite(args.output)
        # Perform identity and allocation checks before model construction.
        verify_candidate_identity(args.checkpoint, args.expected_sha256)
        sealed_residual_seed_plan(
            pairs=args.pairs,
            base_seed=args.base_seed,
            opponent_names=args.opponents,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error

    try:
        payload = evaluate_sealed_residual(
            checkpoint=args.checkpoint,
            expected_sha256=args.expected_sha256,
            champion_path=args.champion,
            pairs=args.pairs,
            base_seed=args.base_seed,
            opponent_names=args.opponents,
            device=args.device,
            cpu_threads=args.cpu_threads,
            confirmed=args.confirm_sealed_evaluation,
            output_path=args.output,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error
    payload["sealed_holdout"]["confirmed"] = True
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    refuse_output_overwrite(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
    except FileExistsError as error:
        raise SystemExit(
            f"refusing to overwrite sealed output: {args.output}"
        ) from error
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_sha256": payload["checkpoint"]["actual_sha256"],
                "combined_pool_delta": payload["combined_pool_delta"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
