#!/usr/bin/env python3
"""Fast BatchedPesten screen for blended suit-symmetry projections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from batched_pesten import ActorPolicy
from batched_suit_symmetry import (
    PLAYER_ORBIT_MODES,
    BatchedSuitAveragedActor,
)
from benchmark_checkpoint_batched import (
    assemble_benchmark_payload,
    build_game_specs,
    file_sha256,
    run_batched_specs,
    run_canonical_specs,
)
from corrected_ppo_agent import make_corrected_ppo_agent_factory
from hybrid_agent import load_frozen_ppo_actor
from suit_symmetry_ppo_agent import make_suit_symmetry_ppo_factory


def _canonical_candidate_factory(
    actor,
    device,
    *,
    alpha: float,
    aggregation: str,
    player_orbit_average: bool,
    base_chunk_size: int,
    player_orbit_mode: str | None = None,
):
    if (
        alpha == 1.0
        and aggregation == "logits"
        and not player_orbit_average
        and player_orbit_mode in (None, "none")
    ):
        return make_suit_symmetry_ppo_factory(actor, device)
    scalar_actor = BatchedSuitAveragedActor(
        actor,
        alpha=alpha,
        aggregation=aggregation,
        player_orbit_average=player_orbit_average,
        player_orbit_mode=player_orbit_mode,
        base_chunk_size=base_chunk_size,
    ).to(device)
    return make_corrected_ppo_agent_factory(scalar_actor, device)


def run_batched_symmetry_benchmark(
    actor,
    device: torch.device,
    *,
    pairs: int,
    base_seed: int,
    pool_names: tuple[str, ...],
    max_turns: int,
    bootstrap_samples: int,
    parent_actor=None,
    alpha: float = 1.0,
    aggregation: str = "logits",
    player_orbit_average: bool = False,
    player_orbit_mode: str | None = None,
    base_chunk_size: int = 512,
    canonical_throughput_pairs: int = 5,
) -> dict:
    resolved_parent_actor = actor if parent_actor is None else parent_actor
    specs = build_game_specs(
        pairs=pairs,
        base_seed=base_seed,
        pool_names=pool_names,
    )
    symmetry_actor = BatchedSuitAveragedActor(
        actor,
        alpha=alpha,
        aggregation=aggregation,
        player_orbit_average=player_orbit_average,
        player_orbit_mode=player_orbit_mode,
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
            resolved_parent_actor,
            device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        ),
    }
    outcomes, batched_statistics = run_batched_specs(
        specs,
        policies,
        max_turns=max_turns,
    )
    payload = {
        **assemble_benchmark_payload(
            specs,
            outcomes,
            pairs=pairs,
            base_seed=base_seed,
            pool_names=pool_names,
            bootstrap_samples=bootstrap_samples,
        ),
        "projection": {
            "alpha": alpha,
            "aggregation": aggregation,
            "ordinary_suit_permutations": 24,
            "player_label_group": (
                symmetry_actor.player_orbit_mode
            ),
            "player_orbit_average": symmetry_actor.player_orbit_average,
            "player_orbit_mode": symmetry_actor.player_orbit_mode,
            "base_chunk_size": base_chunk_size,
            "alpha_zero_is_exact_legacy_actor": True,
            "draw_retry": "just-drawn card plus PASS via BatchedPesten",
            "parent_actor_is_candidate": resolved_parent_actor is actor,
        },
        "actor_statistics": symmetry_actor.statistics(),
        "throughput": {"batched": batched_statistics},
    }

    reference_pairs = min(pairs, canonical_throughput_pairs)
    if reference_pairs:
        reference_specs = build_game_specs(
            pairs=reference_pairs,
            base_seed=base_seed,
            pool_names=pool_names,
        )
        candidate_factory = _canonical_candidate_factory(
            actor,
            device,
            alpha=alpha,
            aggregation=aggregation,
            player_orbit_average=player_orbit_average,
            player_orbit_mode=player_orbit_mode,
            base_chunk_size=base_chunk_size,
        )
        parent_factory = make_corrected_ppo_agent_factory(
            resolved_parent_actor, device
        )
        canonical_outcomes, canonical_elapsed = run_canonical_specs(
            reference_specs,
            candidate_factory=candidate_factory,
            parent_factory=parent_factory,
            max_turns=max_turns,
        )
        batched_reference, batched_reference_statistics = run_batched_specs(
            reference_specs,
            policies,
            max_turns=max_turns,
        )
        if canonical_outcomes != batched_reference:
            raise RuntimeError(
                "batched suit-symmetry evaluator failed canonical parity"
            )
        canonical_games_per_second = (
            len(reference_specs) / canonical_elapsed
        )
        payload["throughput"]["canonical_reference"] = {
            "pairs": reference_pairs,
            "games": len(reference_specs),
            "elapsed_seconds": canonical_elapsed,
            "games_per_second": canonical_games_per_second,
        }
        payload["throughput"]["matched_subset_batched"] = (
            batched_reference_statistics
        )
        payload["throughput"]["speedup"] = (
            batched_reference_statistics["games_per_second"]
            / canonical_games_per_second
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument(
        "--parent-model",
        type=Path,
        help=(
            "optional distinct parent checkpoint; defaults to --model for "
            "backward-compatible self-comparison"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=84_607_031)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--pool", default="random,smart,smarter")
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
        help=(
            "also average the valid cyclic/reflected absolute-player-label "
            "orbit; suit-only remains the default"
        ),
    )
    parser.add_argument(
        "--player-orbit-mode",
        choices=PLAYER_ORBIT_MODES,
        help=(
            "optional lower-cost player-label subgroup; the legacy "
            "--player-orbit-average flag remains an alias for full"
        ),
    )
    parser.add_argument("--base-chunk-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--canonical-throughput-pairs", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    if args.canonical_throughput_pairs < 0:
        raise SystemExit("--canonical-throughput-pairs cannot be negative")
    torch.set_num_threads(args.torch_threads)
    pool_names = tuple(
        name.strip() for name in args.pool.split(",") if name.strip()
    )
    actor, device = load_frozen_ppo_actor(args.model, args.device)
    if args.parent_model is None:
        parent_actor = actor
        parent_model_path = args.model
    else:
        parent_actor, parent_device = load_frozen_ppo_actor(
            args.parent_model, args.device
        )
        if parent_device != device:
            raise SystemExit("candidate and parent devices must match")
        parent_model_path = args.parent_model
    try:
        results = run_batched_symmetry_benchmark(
            actor,
            device,
            pairs=args.pairs,
            base_seed=args.seed,
            pool_names=pool_names,
            max_turns=args.max_turns,
            bootstrap_samples=args.bootstrap_samples,
            parent_actor=parent_actor,
            alpha=args.alpha,
            aggregation=args.aggregation,
            player_orbit_average=args.player_orbit_average,
            player_orbit_mode=args.player_orbit_mode,
            base_chunk_size=args.base_chunk_size,
            canonical_throughput_pairs=args.canonical_throughput_pairs,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    payload = {
        "schema_version": 1,
        "benchmark": "batched_suit_symmetry_projection",
        "engine": "BatchedPesten exact-semantics evaluator",
        "model": {
            "path": str(args.model),
            "sha256": file_sha256(args.model),
            "frozen": True,
            "greedy": True,
            "device": str(device),
        },
        "parent_model": {
            "path": str(parent_model_path),
            "sha256": file_sha256(parent_model_path),
            "frozen": True,
            "greedy": True,
            "device": str(device),
            "same_as_candidate": parent_actor is actor,
        },
        "protocol": {
            "scope": "common-development screen",
            "base_seed": args.seed,
            "all_deal_seeds_below_92m": True,
            "pairs_per_matchup": args.pairs,
            "same_deal_candidate_and_parent": True,
            "seat_balanced": True,
            "torch_threads": args.torch_threads,
        },
        "privacy": {
            "inputs": "unchanged own-hand/public 121-feature observation",
            "opponent_card_identities": False,
            "draw_pile_inspection": False,
        },
        **results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(
        json.dumps(
            {
                "projection": payload["projection"],
                "direct": {
                    "candidate_score": payload["direct"][
                        "candidate_score"
                    ],
                    "candidate_advantage": payload["direct"][
                        "candidate_advantage"
                    ],
                },
                "combined_pool_delta": payload["combined_pool_delta"],
                "throughput": payload["throughput"],
            },
            indent=2,
        )
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
