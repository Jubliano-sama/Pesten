#!/usr/bin/env python3
"""Development-only cross-play evaluation for PPO checkpoint populations.

Every neural matchup uses matched deals with both seat assignments.  Every
checkpoint also sees the same matched deals against random, Smart, and Smarter
fixed opponents.  The evaluator accepts explicit ``NAME=PATH`` checkpoints or
the ``nashpg_pilot_metrics.json`` manifest emitted by ``nashpg_ppo.py``.

This evaluator is deliberately outside the sealed protocol.  It accepts only
deal seeds below 92 million, audits recorded development artifacts before
allocating fresh explicit seeds, and refuses to overwrite an existing result.
Use a 100--200 pair ``screen`` first, then run only selected leaders as a
``confirmation`` with a different requested base seed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import torch

from batched_pesten import ActorPolicy, POLICY_RL
from batched_suit_symmetry import BatchedSuitAveragedActor
from benchmark_checkpoint_batched import (
    BatchGameOutcome,
    BatchGameSpec,
    FIXED_POOL,
    _seat_pair_record,
    derive_unsealed_seed,
    run_batched_specs,
    summarize,
)
from benchmark_hybrid import file_sha256
from hybrid_agent import load_frozen_ppo_actor
from seed_integrity import scan_recorded_seed_interval


DEVELOPMENT_SEED_LIMIT = 92_000_000
DEFAULT_INCUMBENT = Path("ppo_actor_100run3.pth")
MODEL_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
NASHPG_PROVENANCE = (
    "shared-symmetric two-player specialization of NashPG Algorithm 4; "
    "legal-masked PPO clipped surrogate minus alpha times "
    "KL(current||reference), with the reference frozen for K inner rounds "
    "and then refreshed"
)


@dataclass(frozen=True)
class CheckpointSpec:
    name: str
    path: Path
    sha256: str
    suit_projected: bool
    source: str


@dataclass(frozen=True)
class SeedAllocation:
    requested_base_seed: int
    used_recorded_seed_count: int
    skipped_recorded_seeds: tuple[int, ...]
    allocated_seeds: tuple[int, ...]


def _sha256(path: Path) -> str:
    return file_sha256(path)


def _validate_model_name(name: str) -> str:
    normalized = name.strip()
    if not normalized or not MODEL_NAME_PATTERN.fullmatch(normalized):
        raise ValueError(
            "model names must contain only letters, digits, '.', '_', or '-'"
        )
    return normalized


def parse_checkpoint_argument(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--checkpoint must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = _validate_model_name(name)
    if not raw_path.strip():
        raise ValueError("checkpoint path cannot be empty")
    return name, Path(raw_path.strip())


def _alpha_slug(value: object) -> str:
    rendered = format(float(value), ".12g")
    if rendered.startswith("-"):
        rendered = "m" + rendered[1:]
    return rendered.replace(".", "p")


def _resolve_manifest_checkpoint(manifest: Path, raw_path: object) -> Path:
    candidate = Path(str(raw_path))
    if candidate.is_absolute():
        return candidate.resolve()
    manifest_relative = (manifest.parent / candidate).resolve()
    cwd_relative = candidate.resolve()
    if manifest_relative.is_file():
        return manifest_relative
    if cwd_relative.is_file():
        return cwd_relative
    return manifest_relative


def checkpoint_specs_from_nashpg_manifest(
    manifest_path: Path,
) -> list[CheckpointSpec]:
    manifest = manifest_path.resolve()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rounds: list[tuple[Mapping[str, object], object]] = []

    def visit(value: object, inherited_alpha: object | None = None) -> None:
        if isinstance(value, Mapping):
            alpha = value.get("alpha", inherited_alpha)
            checkpoint = value.get("checkpoint")
            if (
                "outer_iteration" in value
                and alpha is not None
                and isinstance(checkpoint, Mapping)
                and "path" in checkpoint
            ):
                rounds.append((value, alpha))
            for child in value.values():
                visit(child, alpha)
        elif isinstance(value, list):
            for child in value:
                visit(child, inherited_alpha)

    visit(payload)
    if not rounds:
        raise ValueError(
            "NashPG manifest contains no outer round checkpoint records"
        )

    specs = []
    for record, alpha in rounds:
        checkpoint = record["checkpoint"]
        assert isinstance(checkpoint, Mapping)
        outer_iteration = int(record["outer_iteration"])
        name = _validate_model_name(
            f"alpha_{_alpha_slug(alpha)}_outer_{outer_iteration:02d}"
        )
        path = _resolve_manifest_checkpoint(manifest, checkpoint["path"])
        declared_sha = checkpoint.get("sha256")
        if declared_sha is None:
            raise ValueError(f"{name} is missing checkpoint.sha256")
        normalized_sha = str(declared_sha).strip().lower()
        if len(normalized_sha) != 64:
            raise ValueError(f"{name} has an invalid checkpoint SHA-256")
        specs.append(
            CheckpointSpec(
                name=name,
                path=path,
                sha256=normalized_sha,
                suit_projected=False,
                source=str(manifest),
            )
        )
    return specs


def resolve_checkpoint_specs(
    *,
    checkpoint_arguments: Sequence[str],
    nashpg_manifests: Sequence[Path],
    incumbent_path: Path,
    incumbent_name: str,
    suit_project_names: Iterable[str],
) -> list[CheckpointSpec]:
    raw_specs: list[CheckpointSpec] = []
    incumbent_name = _validate_model_name(incumbent_name)
    incumbent = incumbent_path.resolve()
    raw_specs.append(
        CheckpointSpec(
            name=incumbent_name,
            path=incumbent,
            sha256=_sha256(incumbent),
            suit_projected=False,
            source="--incumbent",
        )
    )
    for argument in checkpoint_arguments:
        name, path = parse_checkpoint_argument(argument)
        resolved = path.resolve()
        raw_specs.append(
            CheckpointSpec(
                name=name,
                path=resolved,
                sha256=_sha256(resolved),
                suit_projected=False,
                source="--checkpoint",
            )
        )
    for manifest in nashpg_manifests:
        raw_specs.extend(checkpoint_specs_from_nashpg_manifest(manifest))

    projected = {entry.strip() for entry in suit_project_names if entry.strip()}
    if "all" in projected:
        projected = {spec.name for spec in raw_specs}
    names = [spec.name for spec in raw_specs]
    if len(set(names)) != len(names):
        duplicates = sorted(
            name for name in set(names) if names.count(name) > 1
        )
        raise ValueError(f"duplicate model names: {', '.join(duplicates)}")
    unknown_projected = sorted(projected - set(names))
    if unknown_projected:
        raise ValueError(
            "unknown --suit-project model names: "
            + ", ".join(unknown_projected)
        )
    if len(raw_specs) < 2:
        raise ValueError("cross-play requires the incumbent and a candidate")

    specs = []
    for spec in raw_specs:
        if not spec.path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {spec.path}")
        actual_sha = _sha256(spec.path)
        if actual_sha != spec.sha256:
            raise ValueError(
                f"checkpoint SHA-256 mismatch for {spec.name}: "
                f"declared {spec.sha256}, found {actual_sha}"
            )
        specs.append(
            CheckpointSpec(
                name=spec.name,
                path=spec.path,
                sha256=actual_sha,
                suit_projected=spec.name in projected,
                source=spec.source,
            )
        )
    return specs


class _FreshSeedAllocator:
    def __init__(
        self,
        *,
        requested_base_seed: int,
        recorded_seeds: Iterable[int],
    ) -> None:
        if not 0 <= requested_base_seed < DEVELOPMENT_SEED_LIMIT:
            raise ValueError(
                f"base seed must be in [0, {DEVELOPMENT_SEED_LIMIT})"
            )
        self.requested_base_seed = int(requested_base_seed)
        self.recorded = {
            int(seed)
            for seed in recorded_seeds
            if 0 <= int(seed) < DEVELOPMENT_SEED_LIMIT
        }
        self.reserved: set[int] = set()
        self.skipped: list[int] = []
        self.cursor = int(requested_base_seed)

    def next(self) -> int:
        attempts = 0
        while self.cursor in self.recorded or self.cursor in self.reserved:
            if self.cursor in self.recorded:
                self.skipped.append(self.cursor)
            self.cursor = (self.cursor + 1) % DEVELOPMENT_SEED_LIMIT
            attempts += 1
            if attempts >= DEVELOPMENT_SEED_LIMIT:
                raise ValueError("development seed namespace is exhausted")
        allocated = self.cursor
        self.reserved.add(allocated)
        self.cursor = (self.cursor + 1) % DEVELOPMENT_SEED_LIMIT
        return allocated

    def result(self) -> SeedAllocation:
        return SeedAllocation(
            requested_base_seed=self.requested_base_seed,
            used_recorded_seed_count=len(self.recorded),
            skipped_recorded_seeds=tuple(self.skipped),
            allocated_seeds=tuple(sorted(self.reserved)),
        )


def build_crossplay_specs(
    *,
    model_names: Sequence[str],
    pairs: int,
    base_seed: int,
    pool_names: Sequence[str] = ("random", "smart", "smarter"),
    recorded_seeds: Iterable[int] = (),
) -> tuple[list[BatchGameSpec], SeedAllocation]:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if len(model_names) < 2 or len(set(model_names)) != len(model_names):
        raise ValueError("at least two unique model names are required")
    for name in model_names:
        _validate_model_name(name)
    if len(set(pool_names)) != len(pool_names):
        raise ValueError("pool policies must be unique")
    unknown_pool = sorted(set(pool_names) - set(FIXED_POOL))
    if unknown_pool:
        raise ValueError(f"unknown pool policies: {', '.join(unknown_pool)}")

    allocator = _FreshSeedAllocator(
        requested_base_seed=base_seed,
        recorded_seeds=recorded_seeds,
    )
    specs: list[BatchGameSpec] = []
    for model_a, model_b in itertools.combinations(model_names, 2):
        for pair_index in range(pairs):
            deal_seed = allocator.next()
            specs.extend(
                (
                    BatchGameSpec(
                        section="crossplay",
                        opponent=model_b,
                        pair_index=pair_index,
                        evaluated_key=model_a,
                        evaluated_seat=0,
                        deal_seed=deal_seed,
                        policy_kinds=(POLICY_RL, POLICY_RL),
                        actor_keys=(model_a, model_b),
                    ),
                    BatchGameSpec(
                        section="crossplay",
                        opponent=model_b,
                        pair_index=pair_index,
                        evaluated_key=model_a,
                        evaluated_seat=1,
                        deal_seed=deal_seed,
                        policy_kinds=(POLICY_RL, POLICY_RL),
                        actor_keys=(model_b, model_a),
                    ),
                )
            )

    for opponent_name in pool_names:
        opponent_kind = FIXED_POOL[opponent_name]
        for pair_index in range(pairs):
            # Common random numbers: every model sees the same deal and both
            # seat assignments against this fixed opponent.
            deal_seed = allocator.next()
            for model_name in model_names:
                specs.extend(
                    (
                        BatchGameSpec(
                            section="fixed_pool",
                            opponent=opponent_name,
                            pair_index=pair_index,
                            evaluated_key=model_name,
                            evaluated_seat=0,
                            deal_seed=deal_seed,
                            policy_kinds=(POLICY_RL, opponent_kind),
                            actor_keys=(model_name, None),
                        ),
                        BatchGameSpec(
                            section="fixed_pool",
                            opponent=opponent_name,
                            pair_index=pair_index,
                            evaluated_key=model_name,
                            evaluated_seat=1,
                            deal_seed=deal_seed,
                            policy_kinds=(opponent_kind, POLICY_RL),
                            actor_keys=(None, model_name),
                        ),
                    )
                )
    return specs, allocator.result()


def _validated_rotated_pair(
    selected: Sequence[tuple[BatchGameSpec, BatchGameOutcome]],
    *,
    label: str,
) -> tuple[int, list[BatchGameOutcome]]:
    if len(selected) != 2:
        raise ValueError(f"{label} must contain exactly two games")
    seats = [spec.evaluated_seat for spec, _ in selected]
    if seats != [0, 1]:
        raise ValueError(f"{label} must contain seats 0 and 1 in order")
    seeds = {spec.deal_seed for spec, _ in selected}
    if len(seeds) != 1:
        raise ValueError(f"{label} games must share one deal seed")
    for spec, outcome in selected:
        if spec.evaluated_seat != outcome.evaluated_seat:
            raise ValueError(f"{label} outcome seat does not match its spec")
    return seeds.pop(), [outcome for _, outcome in selected]


def assemble_crossplay_payload(
    specs: Sequence[BatchGameSpec],
    outcomes: Sequence[BatchGameOutcome],
    *,
    model_names: Sequence[str],
    incumbent_name: str,
    pairs: int,
    base_seed: int,
    pool_names: Sequence[str],
    bootstrap_samples: int,
) -> dict:
    if len(specs) != len(outcomes):
        raise ValueError("spec and outcome counts differ")
    if incumbent_name not in model_names:
        raise ValueError("incumbent name is absent from model names")

    cross_index: dict[
        tuple[str, str, int],
        list[tuple[BatchGameSpec, BatchGameOutcome]],
    ] = {}
    pool_index: dict[
        tuple[str, str, int],
        list[tuple[BatchGameSpec, BatchGameOutcome]],
    ] = {}
    for spec, outcome in zip(specs, outcomes):
        if not 0 <= spec.pair_index < pairs:
            raise ValueError("pair index is outside requested range")
        if spec.section == "crossplay":
            if spec.opponent not in model_names:
                raise ValueError("cross-play opponent is not a model")
            cross_index.setdefault(
                (spec.evaluated_key, str(spec.opponent), spec.pair_index),
                [],
            ).append((spec, outcome))
        elif spec.section == "fixed_pool":
            if spec.opponent not in pool_names:
                raise ValueError("fixed-pool opponent is unknown")
            pool_index.setdefault(
                (spec.evaluated_key, str(spec.opponent), spec.pair_index),
                [],
            ).append((spec, outcome))
        else:
            raise ValueError(f"unexpected spec section: {spec.section}")

    head_to_head = []
    crossplay_scores: dict[str, list[float]] = {
        name: [] for name in model_names
    }
    for model_a, model_b in itertools.combinations(model_names, 2):
        records = []
        for pair_index in range(pairs):
            deal_seed, pair_outcomes = _validated_rotated_pair(
                cross_index.get((model_a, model_b, pair_index), []),
                label=f"{model_a} vs {model_b} pair {pair_index}",
            )
            record = _seat_pair_record(
                pair_index, deal_seed, pair_outcomes
            )
            records.append(record)
            crossplay_scores[model_a].append(record["score"])
            crossplay_scores[model_b].append(1.0 - record["score"])
        a_scores = [record["score"] for record in records]
        advantages = [2.0 * score - 1.0 for score in a_scores]
        head_to_head.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "model_a_score": summarize(
                    a_scores,
                    seed=derive_unsealed_seed(
                        base_seed, f"cross:{model_a}:{model_b}", 0
                    ),
                    bootstrap_samples=bootstrap_samples,
                ),
                "model_a_advantage": summarize(
                    advantages,
                    seed=derive_unsealed_seed(
                        base_seed,
                        f"cross-advantage:{model_a}:{model_b}",
                        0,
                    ),
                    bootstrap_samples=bootstrap_samples,
                ),
                "per_pair": records,
            }
        )

    pool_pairs: dict[tuple[str, str], list[dict]] = {}
    for opponent_name in pool_names:
        for model_name in model_names:
            records = []
            for pair_index in range(pairs):
                deal_seed, pair_outcomes = _validated_rotated_pair(
                    pool_index.get(
                        (model_name, opponent_name, pair_index), []
                    ),
                    label=(
                        f"{model_name} vs {opponent_name} pair {pair_index}"
                    ),
                )
                records.append(
                    _seat_pair_record(
                        pair_index, deal_seed, pair_outcomes
                    )
                )
            pool_pairs[(model_name, opponent_name)] = records

    fixed_pool = []
    all_pool_scores: dict[str, list[float]] = {
        name: [] for name in model_names
    }
    all_incumbent_deltas: dict[str, list[float]] = {
        name: [] for name in model_names
    }
    for opponent_name in pool_names:
        incumbent_records = pool_pairs[(incumbent_name, opponent_name)]
        incumbent_scores = [record["score"] for record in incumbent_records]
        model_results = []
        for model_name in model_names:
            records = pool_pairs[(model_name, opponent_name)]
            scores = [record["score"] for record in records]
            if [record["deal_seed"] for record in records] != [
                record["deal_seed"] for record in incumbent_records
            ]:
                raise ValueError("fixed-pool common deal seeds diverged")
            deltas = [
                score - incumbent
                for score, incumbent in zip(scores, incumbent_scores)
            ]
            all_pool_scores[model_name].extend(scores)
            all_incumbent_deltas[model_name].extend(deltas)
            model_results.append(
                {
                    "model": model_name,
                    "score": summarize(
                        scores,
                        seed=derive_unsealed_seed(
                            base_seed,
                            f"pool:{opponent_name}:{model_name}",
                            0,
                        ),
                        bootstrap_samples=bootstrap_samples,
                    ),
                    "paired_delta_vs_incumbent": summarize(
                        deltas,
                        seed=derive_unsealed_seed(
                            base_seed,
                            f"pool-delta:{opponent_name}:{model_name}",
                            0,
                        ),
                        bootstrap_samples=bootstrap_samples,
                    ),
                    "per_pair": records,
                }
            )
        fixed_pool.append(
            {"opponent": opponent_name, "models": model_results}
        )

    model_summary = []
    for model_name in model_names:
        model_summary.append(
            {
                "model": model_name,
                "crossplay_mean_score": summarize(
                    crossplay_scores[model_name],
                    seed=derive_unsealed_seed(
                        base_seed, f"cross-summary:{model_name}", 0
                    ),
                    bootstrap_samples=bootstrap_samples,
                ),
                "combined_fixed_pool_score": summarize(
                    all_pool_scores[model_name],
                    seed=derive_unsealed_seed(
                        base_seed, f"pool-summary:{model_name}", 0
                    ),
                    bootstrap_samples=bootstrap_samples,
                ),
                "combined_fixed_pool_delta_vs_incumbent": summarize(
                    all_incumbent_deltas[model_name],
                    seed=derive_unsealed_seed(
                        base_seed, f"pool-delta-summary:{model_name}", 0
                    ),
                    bootstrap_samples=bootstrap_samples,
                ),
            }
        )
    return {
        "head_to_head": head_to_head,
        "fixed_pool": fixed_pool,
        "model_summary": model_summary,
        "ranking": {
            "by_combined_fixed_pool_score": [
                record["model"]
                for record in sorted(
                    model_summary,
                    key=lambda record: (
                        record["combined_fixed_pool_score"]["mean"],
                        record["crossplay_mean_score"]["mean"],
                        record["model"],
                    ),
                    reverse=True,
                )
            ],
            "by_crossplay_mean_score": [
                record["model"]
                for record in sorted(
                    model_summary,
                    key=lambda record: (
                        record["crossplay_mean_score"]["mean"],
                        record["combined_fixed_pool_score"]["mean"],
                        record["model"],
                    ),
                    reverse=True,
                )
            ],
        },
    }


def load_actor_policies(
    specs: Sequence[CheckpointSpec],
    *,
    device_name: str,
    base_chunk_size: int,
) -> dict[str, ActorPolicy]:
    actors: dict[tuple[Path, str], tuple[torch.nn.Module, torch.device]] = {}
    policies = {}
    for spec in specs:
        cache_key = (spec.path, spec.sha256)
        if cache_key not in actors:
            actors[cache_key] = load_frozen_ppo_actor(
                spec.path, device_name
            )
        actor, device = actors[cache_key]
        selected_actor: torch.nn.Module = actor
        if spec.suit_projected:
            selected_actor = BatchedSuitAveragedActor(
                actor,
                player_orbit_average=False,
                base_chunk_size=base_chunk_size,
            ).to(device)
            selected_actor.eval()
        policies[spec.name] = ActorPolicy(
            actor=selected_actor,
            device=device,
            temperature=1.0,
            collect_data=False,
            greedy=True,
        )
    return policies


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="candidate checkpoint; repeat for every explicit candidate",
    )
    parser.add_argument(
        "--nashpg-manifest",
        action="append",
        type=Path,
        default=[],
        help="NashPG pilot metrics manifest; every outer checkpoint is loaded",
    )
    parser.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--incumbent-name", default="incumbent")
    parser.add_argument(
        "--suit-project",
        action="append",
        default=[],
        help="model name to wrap in exact 24-suit averaging, or 'all'",
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=("screen", "confirmation", "harness_smoke"),
    )
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--pool", default="random,smart,smarter")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--base-chunk-size", type=int, default=512)
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument(
        "--recorded-seed-root",
        action="append",
        type=Path,
        default=[],
        help="artifact root used for freshness audit; defaults to benchmarks and experiments",
    )
    parser.add_argument(
        "--skip-recorded-seed-audit",
        action="store_true",
        help="test-only escape hatch; recorded seed freshness is not checked",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing result: {args.output}")
    if args.pairs < 1:
        raise SystemExit("--pairs must be positive")
    if args.stage == "screen" and not 100 <= args.pairs <= 200:
        raise SystemExit("screen stage requires 100--200 pairs")
    if args.stage == "confirmation" and args.pairs < 200:
        raise SystemExit("confirmation stage requires at least 200 pairs")
    if args.max_turns < 1:
        raise SystemExit("--max-turns must be positive")
    if args.bootstrap_samples < 1:
        raise SystemExit("--bootstrap-samples must be positive")
    if args.torch_threads < 1:
        raise SystemExit("--torch-threads must be positive")

    pool_names = tuple(
        value.strip() for value in args.pool.split(",") if value.strip()
    )
    try:
        checkpoint_specs = resolve_checkpoint_specs(
            checkpoint_arguments=args.checkpoint,
            nashpg_manifests=args.nashpg_manifest,
            incumbent_path=args.incumbent,
            incumbent_name=args.incumbent_name,
            suit_project_names=args.suit_project,
        )
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
        raise SystemExit(str(error)) from error

    recorded_uses = []
    audit_roots = (
        args.recorded_seed_root
        if args.recorded_seed_root
        else [Path("benchmarks"), Path("experiments")]
    )
    if not args.skip_recorded_seed_audit:
        recorded_uses = scan_recorded_seed_interval(
            audit_roots,
            start=0,
            end=DEVELOPMENT_SEED_LIMIT,
            allow_sealed_results=True,
        )
    try:
        specs, seed_allocation = build_crossplay_specs(
            model_names=[spec.name for spec in checkpoint_specs],
            pairs=args.pairs,
            base_seed=args.base_seed,
            pool_names=pool_names,
            recorded_seeds=(use.seed for use in recorded_uses),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    torch.set_num_threads(args.torch_threads)
    policies = load_actor_policies(
        checkpoint_specs,
        device_name=args.device,
        base_chunk_size=args.base_chunk_size,
    )
    outcomes, throughput = run_batched_specs(
        specs, policies, max_turns=args.max_turns
    )
    analysis = assemble_crossplay_payload(
        specs,
        outcomes,
        model_names=[spec.name for spec in checkpoint_specs],
        incumbent_name=args.incumbent_name,
        pairs=args.pairs,
        base_seed=args.base_seed,
        pool_names=pool_names,
        bootstrap_samples=args.bootstrap_samples,
    )
    payload = {
        "schema_version": 1,
        "benchmark": "development_checkpoint_crossplay",
        "stage": args.stage,
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "protocol": {
            "pairs_per_matchup": args.pairs,
            "matched_deals": True,
            "seat_rotation": [0, 1],
            "fixed_pool_common_deals_across_models": True,
            "pool": list(pool_names),
            "greedy_legal_masked_actions": True,
            "information_boundary": "own hand and public state only",
            "selection_rule": (
                "screen all candidates first; confirm only preselected "
                "leaders on a fresh explicit seed allocation"
            ),
        },
        "seed_audit": {
            "requested_base_seed": seed_allocation.requested_base_seed,
            "development_seed_limit_exclusive": DEVELOPMENT_SEED_LIMIT,
            "recorded_seed_roots": [str(path) for path in audit_roots],
            "recorded_seed_audit_skipped": args.skip_recorded_seed_audit,
            "recorded_seed_values_seen": (
                seed_allocation.used_recorded_seed_count
            ),
            "recorded_conflicts_skipped": list(
                seed_allocation.skipped_recorded_seeds
            ),
            "unique_deal_seed_count": len(seed_allocation.allocated_seeds),
            "minimum_deal_seed": min(seed_allocation.allocated_seeds),
            "maximum_deal_seed": max(seed_allocation.allocated_seeds),
            "all_deal_seeds_below_92m": all(
                seed < DEVELOPMENT_SEED_LIMIT
                for seed in seed_allocation.allocated_seeds
            ),
        },
        "models": [
            {
                "name": spec.name,
                "path": str(spec.path),
                "sha256": spec.sha256,
                "suit_projected": spec.suit_projected,
                "source": spec.source,
            }
            for spec in checkpoint_specs
        ],
        "candidate_training_provenance": (
            NASHPG_PROVENANCE if args.nashpg_manifest else None
        ),
        **analysis,
        "throughput": throughput,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "stage": args.stage,
                "models": [spec.name for spec in checkpoint_specs],
                "ranking": payload["ranking"],
                "seed_audit": payload["seed_audit"],
                "throughput": throughput,
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
