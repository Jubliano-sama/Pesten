"""Replay a frozen six-cell allocation with the canonical scalar Pesten agents."""

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import json
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mixed_tables as scalar


def initialize(root, candidate, kind, files):
    scalar.use_root(root)
    from live_training.improvement_20261118.freeze_candidate import capture_assets

    guard = ExitStack()
    guard.enter_context(capture_assets(files))
    atexit.register(guard.close)
    scalar.initialize(root, candidate, kind)


def play(task):
    from benchmark_rlvr_pool import _candidate_score

    players, opponent = task["players"], task["opponent"]
    other = scalar.FACTORIES["best" if opponent == "self_anchor" else "random"]
    seeds = range(task["seed"], task["seed"] + task["blocks"])
    row = {"seed": task["seed"], "players": players, "opponent": opponent, "totals": {}}
    for arm in task["variants"]:
        outcomes = [_candidate_score(scalar.FACTORIES[arm], other, players=players, seed=seed) for seed in seeds]
        scores, wins, losses, draws, turns = zip(*outcomes)
        row[f"{arm}_scores"] = scores
        for label, counts in (("win", wins), ("loss", losses), ("draw", draws)):
            row[f"{arm}_{label}_shares"] = [value / players for value in counts]
        row[f"{arm}_turns"] = turns
        row["totals"][arm] = dict(wins=sum(wins), losses=sum(losses), draws=sum(draws), turns=sum(turns))
    return row


def include_raw_turns(result, rows, variants):
    for raw, cell in zip(result["raw_blocks"], result["cells"], strict=True):
        players, opponent = raw["players"], raw["opponent"]
        chunks = sorted((r for r in rows if (r["players"], r["opponent"]) == (players, opponent)),
                        key=lambda r: r["seed"])
        for arm in variants:
            counts = raw[arm]
            counts["losses"] = [players - w - d for w, d in zip(counts["wins"], counts["draws"], strict=True)]
            turns = [t for chunk in chunks for t in chunk[f"{arm}_turns"]]
            if (len(turns) != cell["blocks"] or sum(turns) != cell[arm]["turns"] or
                    any(type(t) is not int or not 0 <= t <= players * 1000 for t in turns)):
                raise ValueError("per-deal turn counts disagree with cell totals")
            counts["turns"] = turns


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="extracted freeze directory")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-blocks", type=int, default=8)
    parser.add_argument("--checkpoint", type=Path, help="append completed chunks to this durable journal")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-seed", type=int, help="use a disjoint development allocation")
    parser.add_argument("--smoke-blocks", type=int)
    args = parser.parse_args()
    if (args.smoke_seed is None) != (args.smoke_blocks is None):
        parser.error("--smoke-seed and --smoke-blocks must be used together")
    if args.workers < 1 or args.chunk_blocks < 1 or (args.resume and not args.checkpoint):
        parser.error("use positive workers and chunk size and supply a checkpoint when resuming")
    root, output = args.root.resolve(), args.output.resolve()
    log = args.checkpoint.resolve() if args.checkpoint else None
    if output.exists() or output == log:
        parser.error("provide a new output path separate from the chunk journal")
    scalar.use_root(root)
    import benchmark_multiplayer_winrate as bench
    from live_training.improvement_20261118 import freeze_candidate as freeze

    if bench.ROOT != root or freeze.ROOT != root:
        raise RuntimeError("evaluation code loaded outside the specified freeze")
    manifest, candidate = freeze.load_freeze()
    allocation = freeze.holdout_allocation(json.loads((root / manifest["protocol"]).read_text()))
    if allocation != manifest["holdout"] or bench.file_sha256(bench.TEACHER) != bench.TEACHER_SHA256:
        raise ValueError("frozen allocation or incumbent mismatch")
    seed, blocks = allocation["base_seed"], allocation["blocks_per_cell"]
    if args.smoke_seed is not None:
        for i in range(len(bench.CELLS)):
            start = seed + i * bench.STRIDE
            for j in range(len(bench.CELLS)):
                test = args.smoke_seed + j * bench.STRIDE
                if max(start, test) < min(start + blocks, test + args.smoke_blocks):
                    parser.error("smoke allocation overlaps the frozen holdout")
        seed, blocks = args.smoke_seed, args.smoke_blocks
    tasks = bench.make_tasks(seed, blocks, args.chunk_blocks)
    driver_paths = [Path(__file__).resolve(), Path(scalar.__file__).resolve()]
    driver_hashes = {p.name: bench.file_sha256(p) for p in driver_paths}
    freeze_hash = bench.file_sha256(freeze.MANIFEST)
    header = {"kind": "pesten_scalar_six_cells_v1", "freeze_manifest_sha256": freeze_hash,
              "driver_sha256": driver_hashes, "seed": seed, "blocks": blocks, "chunk_blocks": args.chunk_blocks}
    started = time.monotonic()
    with bench.chunk_log(log, header=header, tasks=tasks, resume=args.resume) as (saved, append):
        remaining = [task for task in tasks if task["seed"] not in saved]
        rows = list(saved.values())
        if args.resume:
            print(f"Resuming {len(saved)}/{len(tasks)} completed chunks", flush=True)
        if remaining:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(remaining)),
                                     mp_context=multiprocessing.get_context("spawn"), initializer=initialize,
                                     initargs=(root, candidate, manifest["candidate_kind"], manifest["files"])) as pool:
                futures = [pool.submit(play, task) for task in remaining]
                rows = bench.collect_with_progress((f.result() for f in as_completed(futures)), len(tasks),
                                                   started, rows, append)
    result = bench.summarize(rows, seed=seed, blocks=blocks)
    include_raw_turns(result, rows, bench.VARIANTS)
    if (bench.verify_freeze(freeze.MANIFEST, candidate, manifest["candidate_kind"]) != freeze_hash or
            any(bench.file_sha256(p) != driver_hashes[p.name] for p in driver_paths)):
        raise RuntimeError("evaluation sources changed during play")
    payload = {"baseline": {"path": str(bench.TEACHER.relative_to(root)), "sha256": bench.TEACHER_SHA256,
                            "orbit_size": 24, "orbit_alpha": 1},
               "candidate": {"path": manifest["candidate"], "sha256": manifest["files"][manifest["candidate"]],
                             "kind": manifest["candidate_kind"]},
               "protocol": {"base_seed": seed, "blocks_per_cell": blocks, "seed_stride": bench.STRIDE,
                            "games_per_policy": blocks * sum(n for _, n in bench.CELLS), "freeze_protocol": manifest["protocol"],
                            "allocation": "frozen holdout" if args.smoke_seed is None else "disjoint development smoke",
                            "engine": "scalar game.Game.auto_simulate, 1000 turns", "random": "randomAgent.AgentRandom",
                            "seat_rotation": "all focal seats on the same deal in both arms, homogeneous nonfocal seats",
                            "policy": "deterministic temporal_agent_factory, single-thread CPU float32",
                            "statistic": "raw wins / all games, equal-weight mean of six cells",
                            "intervals": "pointwise paired normal 95%, stratified by cell, independent unit is a deal",
                            "workers": args.workers, "chunk_blocks": args.chunk_blocks},
               "source_sha256": {n: manifest["files"][n] for n in manifest["sources"]},
               "runtime": manifest["runtime"], "driver_sha256": driver_hashes, "freeze_manifest_sha256": freeze_hash,
               "chunks_reused": len(saved), "wall_seconds": time.monotonic() - started,
               "wall_seconds_scope": "current invocation", **result}
    output.parent.mkdir(exist_ok=True, parents=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent) as handle:
        json.dump(payload, handle, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        os.link(handle.name, output)


if __name__ == "__main__":
    main()
