"""Run paired one-core counterfactual collection and complete action-head fitting."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPEED = HERE.parent / "speed"
ACTOR = ROOT / "models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt"
LEAF = HERE.parent / "learned_search/fit2/leaf_weights.npy"
STYLE = HERE.parent / "style/classifier/style.pt"
FREQUENCY = HERE.parent / "lookahead/hand_frequencies.json"
ARMS = ("before", "after")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def child(arm, scratch, directory, seed, games, warm):
    import torch
    import temporal_batched_speed
    import train_outcome_residual as outcome

    assert Path(temporal_batched_speed.__file__).parent == scratch / arm
    assert Path(outcome.__file__).parent == scratch / "shared"
    if arm == "before":

        def eager(orbit_size):
            policy = outcome.teacher(orbit_size, trace_cpu=False)
            policy.puffer_policy = outcome._CollectionCore(policy.puffer_policy)
            return policy.eval()

        outcome.collection_policy = eager

    import rollout_policy_improvement as trainer

    torch.set_num_threads(1)
    directory.mkdir()
    data, out = directory / "data", directory / "actor"
    arguments = [
        "collect",
        "--directory",
        str(data),
        "--leaf",
        str(LEAF),
        "--seed",
        str(seed),
        "--chunks",
        "1" if warm else "6",
        "--games",
        str(games),
        "--orbit",
        "1",
        "--keep",
        ".65",
        "--explore",
        ".2",
        "--margin",
        "8",
        "--horizons",
        "1",
        "4",
        "12",
        "--beliefs",
    ]
    started = time.perf_counter()
    sys.argv = [trainer.__file__, *arguments]
    trainer.main()
    collected = time.perf_counter()
    if warm:
        return
    sys.argv = [
        trainer.__file__,
        "fit",
        "--directory",
        str(data),
        "--output",
        str(out),
        "--chunks",
        "6",
        "--device",
        "cpu",
        "--style",
        str(STYLE),
        "--beliefs",
        str(FREQUENCY),
        "--graph",
        "--epochs",
        "16",
        "--batch",
        "512",
        "--learning-rate",
        ".0015",
        "--weight-temperature",
        "1.5",
        "--fit-seed",
        "512750000",
        "--horizon-weights",
        ".12",
        ".25",
        ".63",
        "--margin",
        "8",
        "--scales",
        "195",
    ]
    trainer.main()
    finished = time.perf_counter()
    fit = json.loads((out / "fit.json").read_text())
    result = {
        "collection_seconds": collected - started,
        "fit_seconds": finished - collected,
        "shard_sha256": [digest(p) for p in sorted(data.glob("fork_*.npz"))],
        "head_sha256": digest(out / "branch_195.pt"),
        "history": fit["validation_mse"],
        "games": fit["games"],
        "roots": fit["roots"],
        "simulated_actions": fit["simulated_actions"],
        "training_roots": fit["train_roots"],
        "validation_roots": fit["validation_roots"],
    }
    (directory / "timing.json").write_text(json.dumps(result))


def trial(args, arm, seed, warm=False):
    scratch = args.scratch
    directory = scratch / ("warm" if warm else f"{arm}-{seed}")
    search_paths = [
        str(scratch / arm),
        str(scratch / "shared"),
        str(ROOT),
        os.environ.get("PYTHONPATH", ""),
    ]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(search_paths),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
        "NUMBA_CACHE_DIR": str(scratch / "numba-cache"),
    }
    command = [
        "taskset",
        "-c",
        args.cpu,
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        arm,
        str(scratch),
        str(directory),
        str(seed),
        str(12 if warm else args.games),
        str(int(warm)),
    ]
    start = time.perf_counter()
    with directory.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    elapsed = time.perf_counter() - start
    if not warm:
        row = json.loads((directory / "timing.json").read_text())
        return {**row, "process_wall_seconds": elapsed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", default="2")
    parser.add_argument(
        "--games", type=int, default=96, help="games per shard, with six shards per repetition"
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[647100000, 647200000, 647300000])
    args = parser.parse_args()
    if args.output.exists() or args.games < 12:
        parser.error("choose a new output path and at least twelve games per shard")
    args.scratch = args.scratch.resolve()
    args.scratch.mkdir(parents=True)
    # The shared benchmark verifies all restored modules against its before and after hashes.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from benchmark import stage_sources; from pathlib import Path; "
            "import sys; stage_sources(Path(sys.argv[1]))",
            str(args.scratch),
        ],
        cwd=SPEED,
        check=True,
    )
    shared = args.scratch / "shared"
    shared.mkdir()
    with ZipFile(HERE / "branch_training_sources.zip") as archive:
        archive.extractall(shared)
    trial(args, "after", 648999000, warm=True)
    result = {
        "scope": "CPU-only counterfactual collection, compression, 16-epoch graph fit, checkpoint writes and process startup using the same pretrained auxiliary inputs",
        "threads": 1,
        "cpu": args.cpu,
        "python": platform.python_version(),
        "games_per_shard": args.games,
        "shards_per_run": 6,
        "teacher_sha256": digest(ACTOR),
        "leaf_sha256": digest(LEAF),
        "style_sha256": digest(STYLE),
        "frequency_sha256": digest(FREQUENCY),
        "source_sha256": {p.name: digest(p) for p in sorted(shared.glob("*.py"))},
        "shared_engine_hashes": json.loads((SPEED / "sources.json").read_text()),
        "pairs": [],
    }
    for i, seed in enumerate(args.seeds):
        order = ARMS if i % 2 == 0 else ARMS[::-1]
        pair = {"seed": seed, "order": list(order)}
        for arm in order:
            pair[arm] = trial(args, arm, seed)
            print(arm, seed, round(pair[arm]["process_wall_seconds"], 2), flush=True)
        for key in (
            "shard_sha256",
            "head_sha256",
            "history",
            "games",
            "roots",
            "simulated_actions",
        ):
            if pair["before"][key] != pair["after"][key]:
                raise AssertionError(f"matched runs differ in {key} for {seed}")
        pair["all_outputs_exact"] = True
        result["pairs"].append(pair)
        sums = {
            arm: {
                k: sum(p[arm][k] for p in result["pairs"])
                for k in (
                    "collection_seconds",
                    "fit_seconds",
                    "process_wall_seconds",
                    "games",
                    "roots",
                    "simulated_actions",
                )
            }
            for arm in ARMS
        }
        result["totals"] = sums
        result["speedup"] = (
            sums["before"]["process_wall_seconds"] / sums["after"]["process_wall_seconds"]
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("Outputs matched. Aggregate end-to-end speedup", result["speedup"], flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--child"]:
        _, _, arm, scratch, directory, seed, games, warm = sys.argv
        child(arm, Path(scratch), Path(directory), int(seed), int(games), bool(int(warm)))
    else:
        main()
