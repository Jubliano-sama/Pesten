#!/usr/bin/env python3
"""One-way sealed holdout evaluation for feed-forward PPO checkpoints.

The evaluator deliberately requires the caller to bind a candidate path to an
exact SHA-256 digest before any sealed game is constructed. Deal seeds are
allocated arithmetically in a disjoint, mechanically pre-scanned high-uint32
namespace, making collisions and accidental development-seed reuse detectable.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
from pathlib import Path
import re
from typing import Sequence

from batched_pesten import ActorPolicy, POLICY_RL
from benchmark_checkpoint_batched import (
    BatchGameSpec,
    FIXED_POOL,
    assemble_benchmark_payload,
    run_batched_specs,
)
from benchmark_hybrid import file_sha256
from hybrid_agent import load_frozen_ppo_actor
from seed_integrity import (
    SEALED_SEED_END,
    SEALED_SEED_START,
    claim_sealed_evaluation,
)


DEFAULT_POOL = ("random", "smart", "smarter")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def validate_expected_sha256(expected_sha256: str) -> str:
    """Require the canonical complete digest, never a prefix."""
    if not SHA256_PATTERN.fullmatch(expected_sha256):
        raise ValueError(
            "expected SHA256 must be exactly 64 lowercase hexadecimal characters"
        )
    return expected_sha256


def verify_candidate_identity(
    candidate_path: Path,
    expected_sha256: str,
) -> str:
    """Verify the precommitted candidate identity before sealed evaluation."""
    expected = validate_expected_sha256(expected_sha256)
    if not candidate_path.is_file():
        raise FileNotFoundError(
            f"candidate checkpoint not found: {candidate_path}"
        )
    actual = file_sha256(candidate_path)
    if not hmac.compare_digest(actual, expected):
        raise ValueError(
            "candidate SHA256 mismatch: "
            f"expected {expected}, got {actual}"
        )
    return actual


def refuse_output_overwrite(output_path: Path) -> None:
    """Fail even for dangling symlinks; sealed results are append-never."""
    if os.path.lexists(output_path):
        raise FileExistsError(
            f"refusing to overwrite sealed output: {output_path}"
        )


def sealed_seed_plan(
    *,
    pairs: int,
    base_seed: int,
    pool_names: Sequence[str] = DEFAULT_POOL,
) -> dict[str, range]:
    """Allocate one collision-free contiguous block per logical matchup."""
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if not SEALED_SEED_START <= base_seed < SEALED_SEED_END:
        raise ValueError(
            f"base seed must be in [{SEALED_SEED_START}, {SEALED_SEED_END})"
        )
    normalized_pool = tuple(pool_names)
    if not normalized_pool:
        raise ValueError("sealed pool cannot be empty")
    if len(normalized_pool) != len(set(normalized_pool)):
        raise ValueError("sealed pool names must be unique")
    unknown = sorted(set(normalized_pool) - set(FIXED_POOL))
    if unknown:
        raise ValueError(f"unknown pool policies: {', '.join(unknown)}")

    namespaces = ("direct",) + tuple(
        f"pool:{name}" for name in normalized_pool
    )
    required_seeds = pairs * len(namespaces)
    if base_seed + required_seeds > SEALED_SEED_END:
        raise ValueError(
            "sealed namespace allocation exceeds the reserved interval "
            f"[{SEALED_SEED_START}, {SEALED_SEED_END})"
        )
    return {
        namespace: range(
            base_seed + namespace_index * pairs,
            base_seed + (namespace_index + 1) * pairs,
        )
        for namespace_index, namespace in enumerate(namespaces)
    }


def build_sealed_game_specs(
    *,
    pairs: int,
    base_seed: int,
    pool_names: Sequence[str] = DEFAULT_POOL,
) -> list[BatchGameSpec]:
    """Build matched, rotated BatchedPesten specs without playing any games."""
    pool_names = tuple(pool_names)
    plan = sealed_seed_plan(
        pairs=pairs,
        base_seed=base_seed,
        pool_names=pool_names,
    )
    specs = []
    for pair_index, deal_seed in enumerate(plan["direct"]):
        specs.extend(
            (
                BatchGameSpec(
                    section="direct",
                    opponent=None,
                    pair_index=pair_index,
                    evaluated_key="candidate",
                    evaluated_seat=0,
                    deal_seed=deal_seed,
                    policy_kinds=(POLICY_RL, POLICY_RL),
                    actor_keys=("candidate", "parent"),
                ),
                BatchGameSpec(
                    section="direct",
                    opponent=None,
                    pair_index=pair_index,
                    evaluated_key="candidate",
                    evaluated_seat=1,
                    deal_seed=deal_seed,
                    policy_kinds=(POLICY_RL, POLICY_RL),
                    actor_keys=("parent", "candidate"),
                ),
            )
        )

    for opponent_name in pool_names:
        opponent_kind = FIXED_POOL[opponent_name]
        for pair_index, deal_seed in enumerate(
            plan[f"pool:{opponent_name}"]
        ):
            for evaluated_key in ("candidate", "parent"):
                specs.extend(
                    (
                        BatchGameSpec(
                            section="pool",
                            opponent=opponent_name,
                            pair_index=pair_index,
                            evaluated_key=evaluated_key,
                            evaluated_seat=0,
                            deal_seed=deal_seed,
                            policy_kinds=(POLICY_RL, opponent_kind),
                            actor_keys=(evaluated_key, None),
                        ),
                        BatchGameSpec(
                            section="pool",
                            opponent=opponent_name,
                            pair_index=pair_index,
                            evaluated_key=evaluated_key,
                            evaluated_seat=1,
                            deal_seed=deal_seed,
                            policy_kinds=(opponent_kind, POLICY_RL),
                            actor_keys=(None, evaluated_key),
                        ),
                    )
                )
    return specs


def _seed_plan_metadata(plan: dict[str, range]) -> dict[str, dict[str, int]]:
    return {
        namespace: {
            "start_inclusive": seeds.start,
            "end_exclusive": seeds.stop,
            "count": len(seeds),
        }
        for namespace, seeds in plan.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument("--pairs", type=int, default=1_000)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pool", default=",".join(DEFAULT_POOL))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirm-sealed-evaluation",
        action="store_true",
        help="required acknowledgement that the sealed holdout is one-way",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.confirm_sealed_evaluation:
        raise SystemExit("--confirm-sealed-evaluation is required")
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    if args.max_turns < 1:
        raise SystemExit("--max-turns must be positive")

    pool_names = tuple(
        name.strip() for name in args.pool.split(",") if name.strip()
    )
    try:
        refuse_output_overwrite(args.output)
        actual_candidate_sha256 = verify_candidate_identity(
            args.candidate,
            args.expected_sha256,
        )
        plan = sealed_seed_plan(
            pairs=args.pairs,
            base_seed=args.base_seed,
            pool_names=pool_names,
        )
        specs = build_sealed_game_specs(
            pairs=args.pairs,
            base_seed=args.base_seed,
            pool_names=pool_names,
        )
        parent_sha256 = file_sha256(args.parent)
        sealed_claim = claim_sealed_evaluation(
            evaluator="feedforward_checkpoint",
            candidate_path=args.candidate,
            candidate_sha256=actual_candidate_sha256,
            base_seed=args.base_seed,
            pairs=args.pairs,
            output_path=args.output,
            configuration={
                "parent_path": str(args.parent.resolve()),
                "parent_sha256": parent_sha256,
                "pool": list(pool_names),
                "device": args.device,
                "max_turns": args.max_turns,
                "bootstrap_samples": args.bootstrap_samples,
            },
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        raise SystemExit(str(error)) from error

    candidate_actor, device = load_frozen_ppo_actor(
        args.candidate,
        args.device,
    )
    parent_actor, _ = load_frozen_ppo_actor(args.parent, args.device)
    verify_candidate_identity(args.candidate, actual_candidate_sha256)
    if file_sha256(args.parent) != parent_sha256:
        raise RuntimeError(
            "parent checkpoint changed after the sealed claim was created"
        )
    policies = {
        "candidate": ActorPolicy(
            candidate_actor,
            device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        ),
        "parent": ActorPolicy(
            parent_actor,
            device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        ),
    }
    outcomes, throughput = run_batched_specs(
        specs,
        policies,
        max_turns=args.max_turns,
    )

    # Bootstrap randomness does not generate games. Mapping the sealed base
    # into the existing unsealed assembler preserves its score schema without
    # creating an additional sealed deal namespace.
    bootstrap_seed = args.base_seed - SEALED_SEED_START
    benchmark = assemble_benchmark_payload(
        specs,
        outcomes,
        pairs=args.pairs,
        base_seed=bootstrap_seed,
        pool_names=pool_names,
        bootstrap_samples=args.bootstrap_samples,
    )
    benchmark["base_seed"] = args.base_seed
    payload = {
        "schema_version": 1,
        "engine": "BatchedPesten exact-semantics sealed evaluator",
        "sealed_holdout": {
            "interval": [SEALED_SEED_START, SEALED_SEED_END],
            "one_way": True,
            "confirmed": True,
            "collision_free_namespace_plan": _seed_plan_metadata(plan),
            "bootstrap_seed_not_used_for_games": bootstrap_seed,
            "global_claim": sealed_claim,
        },
        "information_boundary": (
            "own hand plus human-visible public game state only"
        ),
        "candidate": {
            "path": str(args.candidate),
            "expected_sha256": args.expected_sha256,
            "actual_sha256": actual_candidate_sha256,
        },
        "parent": {
            "path": str(args.parent),
            "sha256": parent_sha256,
        },
        **benchmark,
        "throughput": throughput,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    # Recheck immediately before the exclusive create to close the ordinary
    # race between the early fail-fast guard and final publication.
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
                "candidate_sha256": actual_candidate_sha256,
                "direct": payload["direct"]["candidate_score"],
                "combined_pool_delta": payload["combined_pool_delta"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
