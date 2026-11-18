"""Play paired, seat-balanced mixed-opponent games with the scalar Pesten engine."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import permutations
import json
import math
import multiprocessing
import os
from pathlib import Path
import random
from statistics import NormalDist, mean, stdev
import sys
import time


BASE_SEED = 1_524_837_801
STRIDE = 1_000_000
ROSTERS = (("best", "random"), ("best", "random", "random"), ("best", "best", "random"))
ARMS = ("baseline", "candidate")
FACTORIES = {}


def layouts(peers):
    """Return each focal seat and distinct clockwise order in absolute table order."""
    players = len(peers) + 1
    seats = []
    for focal in range(players):
        for order in sorted(set(permutations(peers))):
            clockwise = ("focal", *order)
            seats.append((focal, tuple(clockwise[(seat - focal) % players] for seat in range(players))))
    return seats


def use_root(root):
    os.chdir(root)
    sys.path.insert(0, str(root))
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = "1"


def initialize(root, candidate, kind):
    use_root(root)
    import benchmark_multiplayer_winrate as bench
    from randomAgent import AgentRandom
    from temporal_puffer_agent import temporal_agent_factory

    bench.initialize(str(candidate), kind)
    for arm in ARMS:
        FACTORIES[arm] = temporal_agent_factory(bench.engine_eval._POLICIES[arm], deterministic=True)
    FACTORIES.update(best=FACTORIES["baseline"], random=AgentRandom)


def play_chunk(roster, first_seed, count):
    import numpy as np
    import torch
    from game import Game

    games = [
        (arm, focal, [FACTORIES[arm if role == "focal" else role] for role in seats])
        for arm in ARMS for focal, seats in layouts(ROSTERS[roster])
    ]
    deals = []
    with torch.inference_mode():
        for seed in range(first_seed, first_seed + count):
            row = {"seed": seed, **{arm: dict(wins=0, losses=0, draws=0, turns=0) for arm in ARMS}}
            for arm, focal, factories in games:
                random.seed(seed)
                np.random.seed(seed)
                turns, winner = Game(factories).auto_simulate(max_turns=1000)
                outcome = "draws" if winner is None else "wins" if winner == focal else "losses"
                row[arm][outcome] += 1
                row[arm]["turns"] += turns
            deals.append(row)
    return roster, deals


def summarize(peers, rows, first_seed, blocks):
    rows.sort(key=lambda row: row["seed"])
    if [r["seed"] for r in rows] != list(range(first_seed, first_seed + blocks)):
        raise ValueError("missing or duplicate deal")
    configurations = layouts(peers)
    games = len(configurations)
    totals = {}
    for arm in ARMS:
        for row in rows:
            scores = row[arm]
            if sum(scores[k] for k in ("wins", "losses", "draws")) != games:
                raise ValueError("incomplete seat or opponent-order coverage")
        totals[arm] = {k: sum(row[arm][k] for row in rows) for k in ("wins", "losses", "draws", "turns")}
        totals[arm]["win_rate"] = totals[arm]["wins"] / (blocks * games)
    deltas = [(r["candidate"]["wins"] - r["baseline"]["wins"]) / games for r in rows]
    delta, se = mean(deltas), stdev(deltas) / math.sqrt(blocks)
    radius = NormalDist().inv_cdf(0.975) * se
    return {"peers": list(peers), "players": len(peers) + 1, "independent_deals": blocks,
            "games_per_arm": games * blocks, "layouts": [list(seats) for _, seats in configurations],
            **totals, "delta": delta, "paired_se": se, "paired_ci95": [delta - radius, delta + radius],
            "raw_deals": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="project or extracted freeze root (auto-detected by default)")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--freeze-manifest", type=Path, help="frozen_candidate.json, relative to --root")
    source.add_argument("--candidate", type=Path, help="development profile, relative to --root")
    parser.add_argument("--candidate-kind", choices=("native", "residual", "lookahead", "rank-suit"))
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--blocks", type=int, default=128, help="independent deals per roster")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-deals", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_deals < 1 or not 2 <= args.blocks <= STRIDE:
        parser.error("need positive workers and chunk size and 2 to 1,000,000 deals per roster")
    if args.seed < 0 or args.seed + (len(ROSTERS) - 1) * STRIDE + args.blocks > 2**32:
        parser.error("all deal seeds must fit uint32")
    if bool(args.candidate_kind) != bool(args.candidate):
        parser.error("use --candidate-kind only with --candidate, and specify it there")
    if args.candidate and args.seed == BASE_SEED:
        parser.error("use --freeze-manifest for the reserved mixed bank or a separate --seed for development")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    default_root = next((p for p in script.parents if (p / "benchmark_multiplayer_winrate.py").is_file()), Path.cwd())
    root = (args.root or default_root).resolve()
    use_root(root)
    import benchmark_multiplayer_winrate as bench

    if bench.ROOT != root:
        raise ValueError("loaded policy code outside the requested root")
    if bench.file_sha256(bench.TEACHER) != bench.TEACHER_SHA256:
        raise ValueError("incumbent checkpoint hash mismatch")
    manifest_path = (root / args.freeze_manifest).resolve() if args.freeze_manifest else None
    manifest_hash = None
    if manifest_path is not None:
        from live_training.improvement_20261118.freeze_candidate import runtime

        manifest = json.loads(manifest_path.read_text())
        candidate, kind = (root / manifest["candidate"]).resolve(), manifest["candidate_kind"]
        manifest_hash = bench.verify_freeze(manifest_path, candidate, kind)
        if runtime() != manifest["runtime"]:
            raise ValueError("installed runtime differs from the candidate freeze")
    else:
        candidate, kind = (root / args.candidate).resolve(), args.candidate_kind
    candidate_hash = bench.file_sha256(candidate)
    started = time.monotonic()
    rows = [[] for _ in ROSTERS]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=initialize, initargs=(root, candidate, kind)) as pool:
        futures = [pool.submit(play_chunk, roster, args.seed + roster * STRIDE + i,
                               min(args.chunk_deals, args.blocks - i))
                   for roster in range(len(ROSTERS)) for i in range(0, args.blocks, args.chunk_deals)]
        threshold = 10
        for finished, future in enumerate(as_completed(futures), start=1):
            roster, deals = future.result()
            rows[roster].extend(deals)
            percent = finished * 100 // len(futures)
            if percent >= threshold:
                print(f"{finished}/{len(futures)} chunks, {time.monotonic() - started:.1f}s elapsed",
                      file=sys.stderr, flush=True)
                threshold = (percent // 10 + 1) * 10
    seconds = time.monotonic() - started
    results = [summarize(peers, rows[i], args.seed + i * STRIDE, args.blocks) for i, peers in enumerate(ROSTERS)]
    if manifest_path is not None and bench.verify_freeze(manifest_path, candidate, kind) != manifest_hash:
        raise ValueError("candidate freeze changed during evaluation")
    if bench.file_sha256(candidate) != candidate_hash:
        raise ValueError("candidate changed during evaluation")
    games = sum(2 * result["games_per_arm"] for result in results)
    result = {"panel": "mixed opponents, supplementary", "base_seed": args.seed, "stride": STRIDE,
              "candidate": os.path.relpath(candidate, root), "candidate_kind": kind, "candidate_sha256": candidate_hash,
              "incumbent": str(bench.TEACHER.relative_to(root)), "incumbent_sha256": bench.TEACHER_SHA256,
              "freeze_manifest_sha256": manifest_hash,
              "driver_sha256": bench.file_sha256(script), "workers": args.workers, "chunk_deals": args.chunk_deals,
              "games": games, "seconds": seconds, "games_per_second": games / seconds,
              "interval": "pointwise 95% paired normal, independent unit = deal, delta = candidate minus baseline",
              "rosters": results}
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
