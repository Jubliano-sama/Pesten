"""Time exploratory collection, hand-frequency training and linear continuation fitting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile

import numpy as np

from benchmark import ACTOR, ARMS, HERE, ROOT, digest, stage_sources
from benchmark_outcome import data_hash

FIT_SCRIPT = Path("live_training/improvement_20261118/lookahead/fit_rollout_policy.py")
COLLECT_DRIVER = """import sys
from pathlib import Path
import train_outcome_residual as outcome

arm = sys.argv.pop(1)
if arm == 'before':
    def eager(orbit_size):
        policy = outcome.teacher(orbit_size)
        policy.puffer_policy = outcome._CollectionCore(policy.puffer_policy)
        return policy.eval()
    outcome.collection_policy = eager
outcome.main()
"""


def command(args, arm, script, arguments, log):
    scratch = args.scratch
    paths = (scratch / arm, scratch / "shared", ROOT)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, paths)), "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "NUMBA_NUM_THREADS": "1",
           "NUMBA_CACHE_DIR": str(scratch / "numba-cache")}
    started = time.perf_counter()
    with log.open("w") as handle:
        subprocess.run(["taskset", "-c", args.cpu, sys.executable, str(script), *map(str, arguments)],
                       cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
    return time.perf_counter() - started


def stage(args):
    scratch = args.scratch
    source_hashes = stage_sources(scratch)
    expected = json.loads((HERE / "search_sources.json").read_text())
    shared = scratch / "shared"
    shared.mkdir()
    with ZipFile(HERE / "search_training_sources.zip") as archive:
        for name, sha in expected.items():
            path = shared / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
            assert digest(path) == sha, name
    with ZipFile(HERE / "recipes_at_benchmark.zip") as archive:
        (shared / "train_outcome_residual.py").write_bytes(archive.read("train_outcome_residual.py"))
    (shared / "collect.py").write_text(COLLECT_DRIVER)
    # Compile the unchanged numerical fitting code once on synthetic valid inputs.
    # Both measured arms subsequently load the same disk cache and train on real games.
    warm = scratch / "warm"
    warm.mkdir()
    obs = np.zeros((1, 94), dtype=np.float32)
    obs[0, [0, 1, 50, 56]] = 1
    obs[0, [70, 91, 92]] = [3, 2, 2]
    legal = np.zeros((1, 55), dtype=bool)
    legal[0, [0, 1, 54]] = True
    logits = np.zeros((1, 55), dtype=np.float32)
    logits[0, 0] = 1
    np.savez(warm / "chunk_0000.npz", obs=obs, logits=logits, legal=np.packbits(legal, axis=1, bitorder="little"))
    command(args, "after", shared / FIT_SCRIPT, [warm, warm / "fit.npy"], warm / "jit.log")
    return source_hashes, {**expected, "train_outcome_residual.py": digest(shared / "train_outcome_residual.py")}


def run(args, arm, seed):
    scratch = args.scratch
    shared = scratch / "shared"
    out = scratch / f"{arm}-{seed}"
    out.mkdir()
    deals = out / "deals"
    frequency = out / "hand_frequencies.json"
    continuation = out / "rollout_weights.npy"
    times = {}
    stages = [(1, 0, 6, 2.0, .04), (4, 6, 3, 1.65, .025)]
    for orbit, start, chunks, temperature, epsilon in stages:
        stage_args = [arm, "collect", "--directory", deals, "--seed", seed, "--games", args.games,
                      "--start", start, "--chunks", chunks, "--temperature", temperature,
                      "--epsilon", epsilon, "--orbit", orbit]
        times[f"orbit_{orbit}_seconds"] = command(args, arm, shared / "collect.py", stage_args,
                                                   out / f"orbit_{orbit}.log")
    times["hand_frequency_seconds"] = command(
        args, arm, shared / "train_public_lookahead.py",
        [deals, frequency, "--random-games", args.random_games, "--seed", seed + 8_000],
        out / "frequency.log",
    )
    times["continuation_seconds"] = command(args, arm, shared / FIT_SCRIPT, [deals, continuation], out / "fit.log")
    weights = np.load(continuation, allow_pickle=False)
    frequency_data = json.loads(frequency.read_text())
    comparisons = int(next(line.split()[1] for line in (out / "fit.log").read_text().splitlines() if line.startswith("pairs ")))
    assert weights.shape == (3, 8, 24) and np.isfinite(weights).all()
    return {
        **times, "total_seconds": sum(times.values()),
        "trajectory_sha256": data_hash(deals), "frequency_sha256": digest(frequency),
        "continuation_sha256": digest(continuation), "continuation_shape": list(weights.shape),
        "hand_frequencies_shape": list(np.asarray(frequency_data["weights"]).shape),
        "focal_games": args.games * 9, "random_games": 3 * args.random_games,
        "decisions": sum(row["decisions"] for row in frequency_data["strong_shards"]),
        "strong_hand_observations": frequency_data["strong_observations"],
        "random_hand_observations": frequency_data["random_observations"],
        "continuation_training_pairs": comparisons,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", default="2")
    parser.add_argument("--games", type=int, default=192)
    parser.add_argument("--random-games", type=int, default=64)
    parser.add_argument("--seeds", default="583000100,583100100,583200100")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")
    args.scratch = args.scratch.resolve()
    args.scratch.mkdir(parents=True, exist_ok=False)
    source_hashes, recipes = stage(args)
    result = {"cpu": args.cpu, "shard_games": args.games, "random_games_per_table": args.random_games,
              "actor_sha256": digest(ACTOR), "source_hashes": source_hashes, "recipes": recipes,
              "jit": "one shared, prewarmed Numba cache, identical fitting sources, BLAS and Numba threads=1",
              "pairs": []}
    for index, seed in enumerate(map(int, args.seeds.split(","))):
        order = list(ARMS if index % 2 == 0 else reversed(ARMS))
        pair = {"seed": seed, "order": order}
        for arm in order:
            pair[arm] = run(args, arm, seed)
            print(arm, seed, round(pair[arm]["total_seconds"], 3), flush=True)
        for key, value in pair["before"].items():
            if not key.endswith("_seconds"):
                assert value == pair["after"][key], f"{seed} {key}"
        pair["all_outputs_exact"] = True
        result["pairs"].append(pair)
        totals = {arm: {key: sum(p[arm][key] for p in result["pairs"])
                        for key in pair[arm] if key.endswith("_seconds")} for arm in ARMS}
        result["totals"] = totals
        result["speedup"] = totals["before"]["total_seconds"] / totals["after"]["total_seconds"]
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("verified exact", seed, "running speedup", result["speedup"], flush=True)


if __name__ == "__main__":
    main()
