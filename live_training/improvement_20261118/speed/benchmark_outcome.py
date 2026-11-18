"""Matched CPU collect+fit trials for the exploratory and paired outcome recipes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
from zipfile import ZipFile

import numpy as np
import torch

from benchmark import ARMS, HERE, ROOT, digest, identical, stage_sources


def load_recipe(name, snapshot):
    # Keep repository-relative checkpoint paths even when replaying source
    # from an immutable scratch snapshot. Shared modules come from this arm.
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / f"{name}.py")
    sys.modules[name] = module
    exec(compile(snapshot.read_bytes(), module.__file__, "exec"), module.__dict__)
    return module


def trial(scratch, arm, seed, directory, chunks, games, paired, eager_after):
    outcome = load_recipe("train_outcome_residual", scratch / "outcome_source.py")
    if arm == "before" or eager_after:
        def eager_collection(orbit_size=1):
            policy = outcome.teacher(orbit_size)
            policy.puffer_policy = outcome._CollectionCore(policy.puffer_policy)
            return policy.eval()
        outcome.collection_policy = eager_collection

    queue = [
        ("collect", outcome, ["collect", "--directory", str(directory / "deals"),
            "--chunks", str(chunks), "--games", str(games), "--seed", str(seed), "--orbit", "1"]),
        ("fit", outcome, ["fit", "--directory", str(directory / "deals"), "--chunks", str(chunks),
            "--value-epochs", "7", "--head-epochs", "8", "--td-lambda", ".65",
            "--fit-seed", "513000001", "--scales", "24", "--output", str(directory / "fit")]),
    ]
    if paired:
        intervention = load_recipe("paired_outcome_residual", scratch / "paired_source.py")
        queue.extend([
            ("paired_collect", intervention, ["collect", "--directory", str(directory / "pairs"),
                "--chunks", str(chunks), "--pairs", str(games // 2), "--seed", str(seed + 9000),
                "--orbit", "1"]),
            ("paired_fit", intervention, ["fit", "--directory", str(directory / "pairs"),
                "--chunks", str(chunks), "--initial", str(directory / "fit/residual_suit24_24.pt"),
                "--epochs", "18", "--scales", "24", "--output", str(directory / "paired_fit")]),
        ])
    times = {}
    for label, recipe, arguments in queue:
        sys.argv = [recipe.__file__, *arguments]
        start = time.perf_counter()
        cpu = time.process_time()
        if hasattr(recipe, "main"):
            recipe.main()
        else:
            code = compile((scratch / "paired_source.py").read_bytes(), recipe.__file__, "exec")
            exec(code, {"__name__": "__main__", "__file__": recipe.__file__})
        times[label + "_seconds"] = time.perf_counter() - start
        times[label + "_cpu_seconds"] = time.process_time() - cpu
    (directory / "times.json").write_text(json.dumps(times))


def data_hash(path):
    result = hashlib.sha256()
    for shard in sorted(path.glob("*.npz")):
        with np.load(shard) as data:
            for name in sorted(data.files):
                array = data[name]
                result.update(str((shard.name, name, array.shape, array.dtype.str)).encode())
                result.update(array.tobytes())
    return result.hexdigest()


def run(args, arm, seed):
    directory = args.scratch / f"{arm}-{seed}"
    directory.mkdir()
    command = ["taskset", "-c", args.cpu, sys.executable, str(args.scratch / arm / "outcome.py"),
               str(args.scratch), arm, str(seed), str(directory), str(args.chunks), str(args.games),
               str(int(args.paired)), str(int(args.eager_after))]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(HERE), str(ROOT))),
           "MKL_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    started = time.perf_counter()
    with (directory / "run.log").open("w") as log:
        subprocess.run(command, check=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    fit = json.loads((directory / "fit/fit.json").read_text())
    times = json.loads((directory / "times.json").read_text())
    result = {**times, "process_seconds": elapsed, "data_sha256": data_hash(directory / "deals"),
              "games": fit["games"], "decisions": fit["decisions"],
              "train_games": fit["train_games"], "validation_games": fit["validation_games"],
              "history": fit["history"]}
    if args.paired:
        paired = json.loads((directory / "paired_fit/fit.json").read_text())
        result.update({"paired_data_sha256": data_hash(directory / "pairs"), "paired_games": paired["games"],
                       "paired_roots": paired["roots"], "paired_history": paired["validation_mse"]})
    return result


def matching_heads(scratch, seed, filename):
    heads = [torch.load(scratch / f"{arm}-{seed}" / filename, weights_only=True, map_location="cpu")
             for arm in ARMS]
    return identical(*heads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", default="3")
    parser.add_argument("--seeds", default="535000100,535020100,535040100")
    parser.add_argument("--chunks", type=int, default=9)
    parser.add_argument("--games", type=int, default=384)
    parser.add_argument("--paired", action="store_true", help="also collect and fit paired interventions")
    parser.add_argument("--eager-after", action="store_true", help="reproduce the shared-collector-only comparison")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    if args.chunks < 6 or args.games < 2 or args.games % 2:
        parser.error("need at least six shards and an even positive game count")
    args.scratch = args.scratch.resolve()
    args.scratch.mkdir(parents=True, exist_ok=False)
    stage_sources(args.scratch)
    snapshots = {}
    with ZipFile(HERE / "search_training_sources.zip") as archive:
        suite = archive.read("strongest_policy_search.py")
        for arm in ARMS:
            (args.scratch / arm / "strongest_policy_search.py").write_bytes(suite)
        snapshots["strongest_policy_search"] = hashlib.sha256(suite).hexdigest()
    with ZipFile(HERE / "recipes_at_benchmark.zip") as archive:
        for name, module in (("outcome", "train_outcome_residual"), ("paired", "paired_outcome_residual")):
            snapshot = args.scratch / f"{name}_source.py"
            snapshot.write_bytes(archive.read(f"{module}.py"))
            snapshots[module] = digest(snapshot)
    driver = """if __name__ == '__main__':
    import sys
    from pathlib import Path
    from benchmark_outcome import trial
    root, arm, seed, directory, chunks, games, paired, eager = sys.argv[1:]
    trial(Path(root), arm, int(seed), Path(directory), int(chunks), int(games), int(paired), int(eager))
"""
    for arm in ARMS:
        (args.scratch / arm / "outcome.py").write_text(driver)
    pairs = []
    for i, seed in enumerate(map(int, args.seeds.split(","))):
        order = list(ARMS if i % 2 == 0 else reversed(ARMS))
        pair = {"seed": seed, "order": order}
        for arm in order:
            pair[arm] = run(args, arm, seed)
            print(arm, seed, pair[arm]["process_seconds"], flush=True)
        pair["head_exact"] = matching_heads(args.scratch, seed, "fit/residual_suit24_24.pt")
        equality = ["data_sha256", "history"]
        if args.paired:
            pair["paired_head_exact"] = matching_heads(args.scratch, seed, "paired_fit/paired_24.pt")
            assert pair["paired_head_exact"]
            equality.extend(("paired_data_sha256", "paired_history"))
        assert pair["head_exact"] and all(pair["before"][key] == pair["after"][key] for key in equality)
        pairs.append(pair)
    metrics = ["collect_seconds", "fit_seconds", "collect_cpu_seconds", "fit_cpu_seconds",
               "process_seconds", "games", "decisions"]
    if args.paired:
        metrics.extend(("paired_games", "paired_roots", "paired_collect_seconds", "paired_fit_seconds",
                        "paired_collect_cpu_seconds", "paired_fit_cpu_seconds"))
    totals = {arm: {k: sum(p[arm][k] for p in pairs) for k in metrics} for arm in ARMS}
    for row in totals.values():
        row["games_per_second"] = (row["games"] + row.get("paired_games", 0)) / row["process_seconds"]
    result = {"scope": "CPU collection, compression, model fitting, checkpoint writes and process startup",
              "cpu": args.cpu, "torch": str(torch.__version__), "recipe_sha256": snapshots,
              "shards_per_run": args.chunks, "games_per_shard": args.games,
              "collection_orbit": 1, "value_epochs": 7, "head_epochs": 8,
              "eager_after": args.eager_after, "paired_stages": args.paired,
              "pairs": pairs, "totals": totals,
              "speedup": totals["before"]["process_seconds"] / totals["after"]["process_seconds"]}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
