#!/usr/bin/env python3
"""Guarded sealed holdout for the batched frozen-PPO suit symmetry wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch

from batched_pesten import ActorPolicy
from batched_suit_symmetry import BatchedSuitAveragedActor
from benchmark_checkpoint_batched import (
    assemble_benchmark_payload,
    run_batched_specs,
)
from benchmark_model_zoo import (
    DEFAULT_TEACHER,
    KNOWN_TEACHER_SHA256,
)
from benchmark_sealed_checkpoint import (
    DEFAULT_POOL,
    SEALED_SEED_END,
    SEALED_SEED_START,
    build_sealed_game_specs,
    refuse_output_overwrite,
    sealed_seed_plan,
    validate_expected_sha256,
    verify_candidate_identity,
)
from hybrid_agent import load_frozen_ppo_actor
from seed_integrity import (
    DEFAULT_SEALED_CLAIM_LEDGER,
    claim_sealed_evaluation,
)


def require_sealed_confirmation(confirmed: bool) -> None:
    if confirmed is not True:
        raise ValueError("explicit sealed evaluation confirmation is required")


def validate_champion_precommit(expected_sha256: str) -> str:
    """Bind this wrapper specifically to the reviewed production champion."""
    expected = validate_expected_sha256(expected_sha256)
    if expected != KNOWN_TEACHER_SHA256:
        raise ValueError(
            "suit symmetry sealed evaluation requires the exact production "
            f"champion SHA256 {KNOWN_TEACHER_SHA256}"
        )
    return expected


def validate_projection(
    *,
    alpha: float,
    aggregation: str,
    player_orbit_average: bool,
    base_chunk_size: int,
) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between zero and one")
    if aggregation not in {"logits", "probabilities"}:
        raise ValueError(
            "aggregation must be 'logits' or 'probabilities'"
        )
    if base_chunk_size < 1:
        raise ValueError("base_chunk_size must be positive")
    if not isinstance(player_orbit_average, bool):
        raise TypeError("player_orbit_average must be a bool")


def build_sealed_symmetry_specs(
    *,
    pairs: int,
    base_seed: int,
    pool_names: Sequence[str] = DEFAULT_POOL,
):
    """Return the guarded namespace plan and non-executing game specs."""
    pool_names = tuple(pool_names)
    plan = sealed_seed_plan(
        pairs=pairs,
        base_seed=base_seed,
        pool_names=pool_names,
    )
    specs = build_sealed_game_specs(
        pairs=pairs,
        base_seed=base_seed,
        pool_names=pool_names,
    )
    return plan, specs


def _plan_metadata(plan: dict[str, range]) -> dict[str, dict[str, int]]:
    return {
        namespace: {
            "start_inclusive": seeds.start,
            "end_exclusive": seeds.stop,
            "count": len(seeds),
        }
        for namespace, seeds in plan.items()
    }


def evaluate_sealed_suit_symmetry(
    *,
    champion_path: Path,
    expected_sha256: str,
    pairs: int,
    base_seed: int,
    pool_names: Sequence[str],
    device_name: str,
    max_turns: int,
    bootstrap_samples: int,
    alpha: float,
    aggregation: str,
    player_orbit_average: bool,
    base_chunk_size: int,
    confirmed: bool,
    output_path: Path,
    claim_ledger_path: Path = DEFAULT_SEALED_CLAIM_LEDGER,
) -> dict:
    """Execute one sealed batched comparison after every guard passes."""
    require_sealed_confirmation(confirmed)
    expected = validate_champion_precommit(expected_sha256)
    actual = verify_candidate_identity(champion_path, expected)
    validate_projection(
        alpha=alpha,
        aggregation=aggregation,
        player_orbit_average=player_orbit_average,
        base_chunk_size=base_chunk_size,
    )
    plan, specs = build_sealed_symmetry_specs(
        pairs=pairs,
        base_seed=base_seed,
        pool_names=pool_names,
    )
    sealed_claim = claim_sealed_evaluation(
        evaluator="suit_symmetry_projection",
        candidate_path=champion_path,
        candidate_sha256=actual,
        base_seed=base_seed,
        pairs=pairs,
        output_path=output_path,
        ledger_path=claim_ledger_path,
        configuration={
            "pool": list(pool_names),
            "device": device_name,
            "max_turns": max_turns,
            "bootstrap_samples": bootstrap_samples,
            "alpha": alpha,
            "aggregation": aggregation,
            "player_orbit_average": player_orbit_average,
            "base_chunk_size": base_chunk_size,
        },
    )

    champion_actor, device = load_frozen_ppo_actor(
        champion_path,
        device_name,
    )
    verify_candidate_identity(champion_path, actual)
    symmetry_actor = BatchedSuitAveragedActor(
        champion_actor,
        alpha=alpha,
        aggregation=aggregation,
        player_orbit_average=player_orbit_average,
        base_chunk_size=base_chunk_size,
    ).to(device)
    policies = {
        "candidate": ActorPolicy(
            symmetry_actor,
            device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        ),
        "parent": ActorPolicy(
            champion_actor,
            device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        ),
    }
    outcomes, throughput = run_batched_specs(
        specs,
        policies,
        max_turns=max_turns,
    )

    # The existing assembler only uses this value for bootstrap RNGs. No game
    # is ever initialized from it; all deal seeds come from the sealed plan.
    bootstrap_seed = base_seed - SEALED_SEED_START
    benchmark = assemble_benchmark_payload(
        specs,
        outcomes,
        pairs=pairs,
        base_seed=bootstrap_seed,
        pool_names=tuple(pool_names),
        bootstrap_samples=bootstrap_samples,
    )
    benchmark["base_seed"] = base_seed
    return {
        "schema_version": 1,
        "benchmark": "sealed_batched_suit_symmetry_projection",
        "engine": "BatchedPesten exact-semantics evaluator",
        "sealed_holdout": {
            "interval_start_inclusive": SEALED_SEED_START,
            "interval_end_exclusive": SEALED_SEED_END,
            "one_way": True,
            "confirmed": True,
            "collision_free_namespace_plan": _plan_metadata(plan),
            "bootstrap_seed_not_used_for_games": bootstrap_seed,
            "global_claim": sealed_claim,
        },
        "champion": {
            "path": str(champion_path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "frozen": True,
            "greedy": True,
        },
        "candidate": {
            "wrapper": "BatchedSuitAveragedActor",
            "ordinary_suit_permutations": 24,
            "alpha": alpha,
            "aggregation": aggregation,
            "player_orbit_average": player_orbit_average,
            "player_label_group": (
                "dihedral orbit of 2n relabelings"
                if player_orbit_average
                else "disabled"
            ),
            "base_chunk_size": base_chunk_size,
        },
        "legacy_comparison": {
            "actor": "exact same frozen champion",
            "direct": True,
            "same_deals_and_rotated_seats": True,
        },
        "information_boundary": {
            "input": "unchanged own-hand/public 121-feature observation",
            "synthetic_operation": "ordinary suit-label relabeling only",
            "opponent_card_identities": False,
            "draw_pile_inspection": False,
            "draw_retry": "just-drawn own card plus PASS via BatchedPesten",
        },
        "human_visible_information_only": True,
        **benchmark,
        "actor_statistics": symmetry_actor.statistics(),
        "throughput": throughput,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion",
        type=Path,
        default=DEFAULT_TEACHER,
    )
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--pairs", type=int, default=1_000)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument(
        "--aggregation",
        choices=("logits", "probabilities"),
        default="logits",
    )
    parser.add_argument(
        "--player-orbit-average",
        action="store_true",
        help="also average the public player-label dihedral orbit",
    )
    parser.add_argument("--base-chunk-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-sealed-evaluation",
        action="store_true",
        help="required acknowledgement that the sealed holdout is one-way",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    pool_names = tuple(
        name.strip() for name in args.pool.split(",") if name.strip()
    )
    if args.max_turns < 1:
        raise SystemExit("--max-turns must be positive")
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    try:
        require_sealed_confirmation(args.confirm_sealed_evaluation)
        refuse_output_overwrite(args.output)
        expected = validate_champion_precommit(args.expected_sha256)
        verify_candidate_identity(args.champion, expected)
        validate_projection(
            alpha=args.alpha,
            aggregation=args.aggregation,
            player_orbit_average=args.player_orbit_average,
            base_chunk_size=args.base_chunk_size,
        )
        build_sealed_symmetry_specs(
            pairs=args.pairs,
            base_seed=args.base_seed,
            pool_names=pool_names,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error

    torch.set_num_threads(args.torch_threads)
    try:
        payload = evaluate_sealed_suit_symmetry(
            champion_path=args.champion,
            expected_sha256=args.expected_sha256,
            pairs=args.pairs,
            base_seed=args.base_seed,
            pool_names=pool_names,
            device_name=args.device,
            max_turns=args.max_turns,
            bootstrap_samples=args.bootstrap_samples,
            alpha=args.alpha,
            aggregation=args.aggregation,
            player_orbit_average=args.player_orbit_average,
            base_chunk_size=args.base_chunk_size,
            confirmed=args.confirm_sealed_evaluation,
            output_path=args.output,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error
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
                "champion_sha256": payload["champion"]["actual_sha256"],
                "direct": payload["direct"]["candidate_score"],
                "combined_pool_delta": payload["combined_pool_delta"],
                "actor_statistics": payload["actor_statistics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
