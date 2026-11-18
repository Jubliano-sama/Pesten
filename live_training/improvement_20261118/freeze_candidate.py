"""Freeze one playable evaluation, replay canonical parity, or run its fixed holdout."""

import argparse
from contextlib import contextmanager
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
PROTOCOL = SCRIPT.with_name("protocol.json")
MANIFEST = ROOT / "frozen_candidate.json"
PARITY_SEED = 2417811052
DISCOVERY_SEED = 2308610497
sys.path.insert(0, str(ROOT))


def runtime():
    import torch
    versions = {"python": platform.python_version(), "torch": str(torch.__version__)}
    for package in ("numpy", "pufferlib", "numba", "llvmlite"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def holdout_allocation(document):
    """Read either campaign protocol while preserving its original file in the archive."""
    import benchmark_multiplayer_winrate as bench
    if "holdout" in document:
        bank = document["holdout"]
        blocks, stride, order = bank["independent_deals_per_cell"], bank["stride_per_cell"], bank["order"]
    else:
        bank = document
        blocks, stride, order = bank["blocks_per_cell"], bank["seed_stride"], bank["cell_order"]
    expected_order = [("teacher" if opponent == "self_anchor" else "random") + str(players)
                      for opponent, players in bench.CELLS]
    games_per_policy = blocks * sum(players for _, players in bench.CELLS)
    seed = bank["base_seed"]
    if (type(seed) is not int or type(blocks) is not int or stride != bench.STRIDE or order != expected_order
            or bank["games_per_policy"] != games_per_policy or bank["total_games"] != 2 * games_per_policy):
        raise ValueError("frozen protocol allocation disagrees with the six-cell evaluator")
    bench.make_tasks(seed, blocks, blocks)
    return {"base_seed": seed, "blocks_per_cell": blocks,
            "games_per_policy": games_per_policy, "total_games": 2 * games_per_policy}


@contextmanager
def capture_assets(expected_files=None):
    # Monitor actual loader reads. The normal loaders verify their embedded
    # digests and open nested priors, fitted heads and rollout artifacts here.
    assets = set()
    python_roots = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    enabled = True

    def on_open(event, values):
        if not enabled or event != "open" or not isinstance(values[0], str):
            return
        filename, _, flags = values
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
            return
        if not filename.lower().endswith((".pt", ".pth", ".json", ".npy", ".npz")):
            return
        path = Path(filename).resolve()
        if not path.is_file() or any(path.is_relative_to(directory) for directory in python_roots):
            return
        if not path.is_relative_to(ROOT):
            raise ValueError(f"evaluation data outside the snapshot: {path}")
        name = str(path.relative_to(ROOT))
        if expected_files is not None and name != MANIFEST.name and name not in expected_files:
            raise ValueError(f"unfrozen data read: {name}")
        assets.add(name)

    sys.addaudithook(on_open)
    try:
        yield assets
    finally:
        enabled = False


def freeze(args):
    import benchmark_multiplayer_winrate as bench
    candidate = args.candidate.relative_to(ROOT)
    protocol_path = str(args.protocol.relative_to(ROOT))
    protocol_bytes = args.protocol.read_bytes()
    allocation = holdout_allocation(json.loads(protocol_bytes))
    with capture_assets() as assets:
        bench.initialize(str(args.candidate), args.kind)
        # Several heads import their public-history and card-belief helpers
        # only on the first recurrent step. Exercise every table/opponent shape
        # on a dedicated bank before taking the source and artifact snapshot.
        for index, (opponent, players) in enumerate(bench.CELLS):
            bench.evaluate_task({"opponent": opponent, "players": players, "blocks": 1,
                                 "seed": DISCOVERY_SEED + index * bench.STRIDE,
                                 "variants": ("candidate",), "policy_names": bench.VARIANTS,
                                 "self_anchor_name": "baseline"})
    versions = runtime()
    sources = bench.source_hashes()
    required = {str(candidate), str(bench.TEACHER.relative_to(ROOT)), "ppo_actor_100run3.pth"}
    if not required <= assets or bench.file_sha256(bench.TEACHER) != bench.TEACHER_SHA256:
        raise RuntimeError("incomplete candidate or baseline dependency capture")
    dependencies = sorted(set(sources) | assets | {
        str(SCRIPT), str(PROTOCOL), protocol_path, "puffer4-native/pyproject.toml", "puffer4-native/uv.lock",
    })
    manifest = {"schema": 1, "candidate": str(candidate), "candidate_kind": args.kind,
                "sources": sorted(sources), "assets": sorted(assets), "runtime": versions,
                "protocol": protocol_path, "holdout": allocation, "parity_base_seed": PARITY_SEED,
                "discovery_base_seed": DISCOVERY_SEED, "files": {},
                "replay_from_fresh_extracted_directory": [
                    f"uv sync --locked --extra search --project puffer4-native --python {versions['python']}",
                    "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1",
                    f"puffer4-native/.venv/bin/python -I {SCRIPT} parity --output parity.json",
                    f"puffer4-native/.venv/bin/python -I {SCRIPT} run --workers 6 --chunk-blocks 48 --checkpoint chunks.jsonl --output result.json",
                ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        path = Path(temporary) / "freeze.zip"
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
            for name in dependencies:
                contents = protocol_bytes if name == protocol_path else (ROOT / name).read_bytes()
                manifest["files"][name] = hashlib.sha256(contents).hexdigest()
                if name in sources and manifest["files"][name] != sources[name]:
                    raise RuntimeError(f"module changed during freeze: {name}")
                archive.writestr(name, contents)
            archive.writestr(MANIFEST.name, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.link(path, args.output)
    print(f"{args.output}  SHA-256 {bench.file_sha256(args.output)}", flush=True)
    print(f"{len(sources)} imported project modules, {len(assets)} model/configuration assets", flush=True)


def load_freeze():
    import benchmark_multiplayer_winrate as bench
    manifest = json.loads(MANIFEST.read_text())
    candidate = ROOT / manifest["candidate"]
    bench.verify_freeze(MANIFEST, candidate, manifest["candidate_kind"])
    actual_runtime = runtime()
    if manifest["runtime"] != actual_runtime:
        raise RuntimeError(f"frozen runtime {manifest['runtime']} differs from {actual_runtime}")
    return manifest, candidate


def parity(args, manifest, candidate):
    import benchmark_multiplayer_winrate as bench
    from benchmark_rlvr_pool import _candidate_score
    from randomAgent import AgentRandom
    from temporal_puffer_agent import temporal_agent_factory

    bench.initialize(str(candidate), manifest["candidate_kind"])
    policies = bench.engine_eval._POLICIES
    factory = temporal_agent_factory(policies["candidate"], deterministic=True)
    teacher = temporal_agent_factory(policies["baseline"], deterministic=True)
    rows = []
    for index, (opponent, players) in enumerate(bench.CELLS):
        for offset in range(args.blocks):
            seed = manifest["parity_base_seed"] + index * bench.STRIDE + offset
            opponent_factory = teacher if opponent == "self_anchor" else AgentRandom
            _, wins, losses, draws, turns = _candidate_score(factory, opponent_factory, players=players, seed=seed)
            scalar = dict(wins=wins, losses=losses, draws=draws, turns=turns)
            values = bench.evaluate_task({
                "opponent": opponent, "players": players, "seed": seed, "blocks": 1,
                "variants": ("candidate",), "policy_names": bench.VARIANTS, "self_anchor_name": "baseline",
            })["totals"]["candidate"]
            batched = {name: values.get(name, 0) for name in scalar}
            rows.append({"opponent": opponent, "players": players, "deal_seed": seed,
                         "scalar": scalar, "batched": batched})
    imported = bench.source_hashes()
    for name, digest in imported.items():
        if manifest["files"].get(name) != digest:
            raise RuntimeError(f"unfrozen local module at replay: {name}")
    bench.verify_freeze(MANIFEST, candidate, manifest["candidate_kind"])
    passed = all(row["scalar"] == row["batched"] for row in rows)
    payload = {"passed": passed, "candidate": manifest["candidate"], "kind": manifest["candidate_kind"],
               "candidate_sha256": manifest["files"][manifest["candidate"]],
               "manifest_sha256": bench.file_sha256(MANIFEST), "runtime": runtime(),
               "games_per_engine": args.blocks * sum(p for _, p in bench.CELLS),
               "blocks_per_cell": args.blocks, "rows": rows}
    with args.output.open("x") as output:
        json.dump(payload, output, indent=2)
        output.write("\n")
    if not passed:
        raise RuntimeError(f"scalar/batched disagreement, see {args.output}")
    print(f"Canonical and batched W/L/D and turns agree in all six cells ({payload['games_per_engine']} games per engine).")


def run(args, manifest, candidate):
    import benchmark_multiplayer_winrate as bench
    bank = holdout_allocation(json.loads((ROOT / manifest["protocol"]).read_bytes()))
    if bank != manifest["holdout"]:
        raise ValueError("packaged protocol differs from the selected freeze")
    command = ["--candidate", str(candidate), "--candidate-kind", manifest["candidate_kind"],
               "--seed", str(bank["base_seed"]), "--blocks", str(bank["blocks_per_cell"]),
               "--chunk-blocks", str(args.chunk_blocks), "--workers", str(args.workers),
               "--freeze-manifest", str(MANIFEST), "--output", str(args.output)]
    if args.checkpoint is not None:
        command.extend(["--checkpoint", str(args.checkpoint)])
    if args.resume:
        command.append("--resume")
    bench.main(command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    create = actions.add_parser("freeze", help="bundle assets, source and protocol before evaluating the finalist")
    create.add_argument("--candidate", type=Path, required=True)
    create.add_argument("--kind", choices=("native", "residual", "lookahead", "rank-suit"), required=True)
    create.add_argument("--protocol", type=Path, default=ROOT / PROTOCOL,
                        help="campaign protocol to preserve and replay (defaults to the primary protocol)")
    check = actions.add_parser("parity", help="compare true scalar agents and batched play on separate smoke deals")
    check.add_argument("--blocks", type=int, default=2, choices=range(1, 101), metavar="1..100")
    evaluate = actions.add_parser("run", help="run the packaged holdout with its frozen dependencies")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--chunk-blocks", type=int, default=48)
    evaluate.add_argument("--checkpoint", type=Path, help="append finished chunks to this log")
    evaluate.add_argument("--resume", action="store_true", help="resume the same log and frozen configuration")
    for action in (create, check, evaluate):
        action.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    args.output.parent.mkdir(exist_ok=True, parents=True)
    if args.action == "freeze":
        args.candidate = args.candidate.resolve()
        args.protocol = args.protocol.resolve()
    elif args.action == "run" and args.checkpoint is not None:
        args.checkpoint = args.checkpoint.resolve()
    os.chdir(ROOT)
    if args.action == "freeze":
        freeze(args)
    else:
        manifest, candidate = load_freeze()
        with capture_assets(manifest["files"]):
            if args.action == "parity":
                parity(args, manifest, candidate)
            else:
                run(args, manifest, candidate)


if __name__ == "__main__":
    main()
