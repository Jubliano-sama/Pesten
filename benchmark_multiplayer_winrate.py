#!/usr/bin/env python3
"""Paired outright win rates against teacher4387/orbit24 and AgentRandom."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
import json
import math
import multiprocessing
from pathlib import Path
import platform
from statistics import NormalDist
import sys
import time

import numpy as np
import torch

from benchmark_rlvr_pool import file_sha256
import benchmark_rlvr_pool_batched as engine_eval
from multiplayer_eval_checkpoints import chunk_log
from strongest_policy_search import NativeSuitOrbitCore


ROOT = Path(__file__).resolve().parent
TEACHER = ROOT / "models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt"
TEACHER_SHA256 = "f8f0aece2d759eadc60b3cbccca30c321c01da13322f7cae511a73c8da3a9f93"
CELLS = tuple((opponent, players)
              for opponent in ("self_anchor", "canonical_random") for players in (2, 3, 4))
STRIDE = 1_000_000
VARIANTS = ("baseline", "candidate")
Z95 = NormalDist().inv_cdf(.975)
CANDIDATE_MODULES = {
    "residual": "train_outcome_residual",
    "lookahead": "public_lookahead",
    "rank-suit": "rank_suit_orbit",
}


def make_tasks(seed, blocks, chunk_blocks):
    if not 2 <= blocks <= STRIDE or chunk_blocks < 1:
        raise ValueError("use 2 to 1,000,000 blocks per cell and a positive chunk size")
    if seed < 0 or seed + (len(CELLS) - 1) * STRIDE + blocks > 2**32:
        raise ValueError("deal seeds must fit uint32")
    return [
        {"opponent": opponent, "players": players, "seed": seed + index * STRIDE + offset,
         "blocks": min(chunk_blocks, blocks - offset), "variants": VARIANTS,
         "policy_names": VARIANTS, "self_anchor_name": "baseline"}
        for index, (opponent, players) in enumerate(CELLS)
        for offset in range(0, blocks, chunk_blocks)
    ]


def orbit24(policy):
    policy.puffer_policy = NativeSuitOrbitCore(policy.puffer_policy, alpha=1, orbit_size=24)
    return policy.eval().requires_grad_(False)


def initialize(candidate, kind):
    # This common evaluator initializes the legacy MLP as a loader dependency.
    # Its only non-focal roles in this benchmark are the frozen teacher and Random.
    engine_eval._initialize_multi_worker({"baseline": str(TEACHER)}, str(ROOT / "ppo_actor_100run3.pth"))
    policies = engine_eval._POLICIES
    policies["baseline"] = orbit24(policies["baseline"])
    if kind == "native":
        policies["candidate"] = orbit24(engine_eval._load_policy(Path(candidate)))
    elif kind in CANDIDATE_MODULES:
        module = import_module(CANDIDATE_MODULES[kind])
        policies["candidate"] = module.load_candidate(candidate, device="cpu")
    else:
        raise ValueError(f"unknown candidate kind {kind}")
    for policy in policies.values():
        policy.to(device="cpu", dtype=torch.float32).eval().requires_grad_(False)


def evaluate_task(task):
    return {"seed": task["seed"], **engine_eval._evaluate_task(task)}


def source_hashes():
    # The batched evaluator imports its decision adapter on the first turn.
    import temporal_batched_speed

    files = set()
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if isinstance(path, str) and path.endswith(".py") and Path(path).parent == ROOT:
            files.add(Path(path))
    return {path.name: file_sha256(path) for path in sorted(files)}


def verify_freeze(path, candidate, kind):
    manifest = json.loads(path.read_text())
    if (ROOT / manifest["candidate"]).resolve() != candidate or manifest["candidate_kind"] != kind:
        raise ValueError("frozen profile differs from the requested candidate")
    for name, expected in manifest["files"].items():
        artifact = (ROOT / name).resolve()
        if not artifact.is_relative_to(ROOT) or file_sha256(artifact) != expected:
            raise ValueError(f"frozen file mismatch: {name}")
    return file_sha256(path)


def collect_with_progress(results, count, started, previous=(), persist=None):
    rows = list(previous)
    threshold = 10 * (100 * len(rows) // count // 10 + 1)
    for row in results:
        if persist is not None:
            persist(row)
        rows.append(row)
        completed = len(rows)
        percentage = 100 * completed // count
        if percentage >= threshold:
            print(f"{completed}/{count} tasks ({percentage}%), {time.monotonic() - started:.1f}s elapsed", flush=True)
            threshold = 10 * (percentage // 10 + 1)
    return rows


def summarize(rows, *, seed, blocks):
    by_cell = defaultdict(list)
    for row in rows:
        by_cell[row["opponent"], row["players"]].append(row)
    if set(by_cell) != set(CELLS):
        raise ValueError("incomplete or unexpected opponent/player-count cells")
    cells, raw_blocks = [], []
    for index, (opponent, players) in enumerate(CELLS):
        start = seed + index * STRIDE
        next_seed = start
        wins = {variant: [] for variant in VARIANTS}
        draws = {variant: [] for variant in VARIANTS}
        turns = dict.fromkeys(VARIANTS, 0)
        for row in sorted(by_cell[opponent, players], key=lambda item: item["seed"]):
            if row["seed"] != next_seed:
                raise ValueError("missing or duplicated deal blocks")
            chunk = len(row["baseline_win_shares"])
            if not chunk:
                raise ValueError("empty evaluation chunk")
            for variant in VARIANTS:
                totals = row["totals"][variant]
                counts = {}
                for name in ("win", "loss", "draw"):
                    shares = np.asarray(row[f"{variant}_{name}_shares"], dtype=float) * players
                    if (shares.shape != (chunk,) or not np.isfinite(shares).all()
                            or (shares < 0).any() or (shares > players).any()
                            or not np.allclose(shares, shares.round(), atol=1e-10, rtol=0)):
                        raise ValueError("invalid per-block outcome counts")
                    counts[name] = shares.round().astype(int)
                    field = {"win": "wins", "loss": "losses", "draw": "draws"}[name]
                    if counts[name].sum() != totals.get(field, 0):
                        raise ValueError("block and game outcome totals disagree")
                if not np.all(counts["win"] + counts["loss"] + counts["draw"] == players):
                    raise ValueError("seat rotations do not cover every seat")
                wins[variant].extend(counts["win"].tolist())
                draws[variant].extend(counts["draw"].tolist())
                turns[variant] += totals["turns"]
            next_seed += chunk
        if next_seed != start + blocks:
            raise ValueError("incomplete number of blocks")
        games = blocks * players
        raw = {"opponent": opponent, "players": players, "seed_start": start}
        cell = {"opponent": opponent, "players": players, "blocks": blocks, "games_per_policy": games}
        for variant in VARIANTS:
            win_count = sum(wins[variant])
            draw_count = sum(draws[variant])
            raw[variant] = {"wins": wins[variant], "draws": draws[variant]}
            cell[variant] = {"wins": win_count, "losses": games - win_count - draw_count,
                             "draws": draw_count, "turns": turns[variant], "win_rate": win_count / games}
        differences = (np.asarray(wins["candidate"]) - wins["baseline"]) / players
        raw["paired_win_rate_deltas"] = differences.tolist()
        cell["delta"] = float(differences.mean())
        cell["standard_error"] = float(differences.std(ddof=1) / math.sqrt(blocks))
        cell["paired_ci95"] = [cell["delta"] + sign * Z95 * cell["standard_error"] for sign in (-1, 1)]
        cells.append(cell)
        raw_blocks.append(raw)

    def average(group):
        size = len(group)
        means = {variant: sum(row[variant]["win_rate"] for row in group) / size for variant in VARIANTS}
        delta = means["candidate"] - means["baseline"]
        stderr = math.sqrt(sum(row["standard_error"]**2 for row in group)) / size
        return {**means, "delta": delta, "standard_error": stderr,
                "paired_ci95": [delta - Z95 * stderr, delta + Z95 * stderr]}

    return {
        "cells": cells,
        "by_opponent": {name: average([c for c in cells if c["opponent"] == name])
                        for name in ("self_anchor", "canonical_random")},
        "by_players": {p: average([c for c in cells if c["players"] == p]) for p in (2, 3, 4)},
        "macro": average(cells),
        "raw_blocks": raw_blocks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-kind", choices=("native", *CANDIDATE_MODULES), required=True)
    parser.add_argument("--blocks", type=int, required=True, help="independent deals per cell")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--chunk-blocks", type=int, default=48)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, help="verify all frozen sources and assets before and after play")
    parser.add_argument("--checkpoint", type=Path, help="durable completed-chunk log for this freeze")
    parser.add_argument("--resume", action="store_true", help="reuse the explicitly named matching checkpoint")
    args = parser.parse_args(argv)
    tasks = make_tasks(args.seed, args.blocks, args.chunk_blocks)
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    if args.resume and args.checkpoint is None:
        parser.error("--resume needs --checkpoint")
    if args.checkpoint is not None and (args.freeze_manifest is None or args.checkpoint.resolve() == args.output.resolve()):
        parser.error("a checkpoint requires a freeze manifest and a separate final output path")
    candidate_path = args.candidate.resolve()
    candidate_hash = file_sha256(candidate_path)
    freeze_sha = (verify_freeze(args.freeze_manifest, candidate_path, args.candidate_kind)
                  if args.freeze_manifest is not None else None)
    if file_sha256(TEACHER) != TEACHER_SHA256:
        raise ValueError("frozen orbit24 teacher checkpoint SHA-256 mismatch")
    started = time.monotonic()
    initargs = (str(candidate_path), args.candidate_kind)
    available_sources = {path.name: file_sha256(path) for path in ROOT.glob("*.py")}
    initialize(*initargs)
    sources = source_hashes()
    if args.freeze_manifest is not None:
        frozen = json.loads(args.freeze_manifest.read_text())
        sources.update({name: frozen["files"][name] for name in frozen["sources"]})
    header = {"kind": "pesten_matched_chunks_v1", "freeze_manifest_sha256": freeze_sha,
              "candidate_sha256": candidate_hash, "seed": args.seed, "blocks": args.blocks,
              "chunk_blocks": args.chunk_blocks}
    with chunk_log(args.checkpoint, header=header, tasks=tasks, resume=args.resume) as (saved, append):
        remaining = [task for task in tasks if task["seed"] not in saved]
        if args.resume:
            print(f"Resuming {len(saved)}/{len(tasks)} completed tasks", flush=True)
        if args.workers == 1 or not remaining:
            rows = collect_with_progress(map(evaluate_task, remaining), len(tasks), started, saved.values(), append)
        else:
            with ProcessPoolExecutor(
                max_workers=min(len(remaining), args.workers),
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize,
                initargs=initargs,
            ) as pool:
                pending = [pool.submit(evaluate_task, task) for task in remaining]
                rows = collect_with_progress((f.result() for f in as_completed(pending)), len(tasks), started,
                                             saved.values(), append)
    result = summarize(rows, seed=args.seed, blocks=args.blocks)
    if file_sha256(candidate_path) != candidate_hash or file_sha256(TEACHER) != TEACHER_SHA256:
        raise RuntimeError("checkpoint changed during evaluation")
    sources.update(source_hashes())
    if any(available_sources.get(name) != digest or file_sha256(ROOT / name) != digest
           for name, digest in sources.items()):
        raise RuntimeError("evaluation sources changed during play")
    if args.freeze_manifest is not None and verify_freeze(args.freeze_manifest, candidate_path, args.candidate_kind) != freeze_sha:
        raise RuntimeError("evaluation freeze changed during play")
    payload = {
        "baseline": {"path": str(TEACHER), "sha256": TEACHER_SHA256, "orbit_size": 24, "orbit_alpha": 1},
        "candidate": {"path": str(candidate_path), "sha256": candidate_hash, "kind": args.candidate_kind},
        "protocol": {"base_seed": args.seed, "blocks_per_cell": args.blocks, "seed_stride": STRIDE,
                     "games_per_policy": args.blocks * sum(players for _, players in CELLS),
                     "seat_rotation": "same deal for each focal seat and both focal policies",
                     "opponents": "all remaining seats are copies of the named opponent",
                     "engine": "BatchedPesten, canonical game.Game first finisher, 1000 turn cap",
                     "random": "randomAgent.AgentRandom semantics; shared MT19937 hand/deck stream and seeded Python suit RNG",
                     "policy": "greedy legal action, CPU float32",
                     "statistic": "wins / all games; equal weights for six cells, draws count as zero wins",
                     "intervals": "two-sided individual 95% normal intervals for paired win-rate differences; "
                                  "one independent sample per deal block, disjoint deal banks across cells",
                     "workers": args.workers, "chunk_blocks": args.chunk_blocks,
                     "python_version": platform.python_version(), "numpy_version": np.__version__,
                     "torch_version": str(torch.__version__)},
        "source_sha256": sources, "freeze_manifest_sha256": freeze_sha,
        "chunks_reused": len(saved), "wall_seconds": time.monotonic() - started,
        "wall_seconds_scope": "current invocation", **result,
    }
    args.output.parent.mkdir(exist_ok=True, parents=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["macro"], indent=2))


if __name__ == "__main__":
    main()
