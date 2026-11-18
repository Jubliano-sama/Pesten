"""Time the default privileged PPO learner and compare the exact training outputs."""
import argparse
import json
import os
from pathlib import Path
import platform
import hashlib
import shutil
import subprocess

import torch

from benchmark import ACTOR, ARMS, DRIVER, EPOCHS, HERE, ROOT, digest, identical, stage_sources, summarize, train


CRITIC = ROOT / "models/best/ppo_omniscient_mlp_gamma0p995_2to8_20260815.pt"
COMMON = [
    "--resume", str(ACTOR), "--privileged-critic", str(CRITIC),
    "--schedule", "current_selfplay,canonical_random", "--player-count-schedule", "2,3,4",
    "--games-per-epoch", "256", "--epochs", str(EPOCHS), "--max-turns", "1000",
    "--campaign-offset", "0", "--fresh-optimizer", "--learning-rate", "0.00005",
    "--device", "cuda", "--update-precision", "bfloat16", "--matmul-precision", "high",
    "--no-compile-updates", "--evaluate-every", "1000000", "--evaluation-pairs", "1",
    "--checkpoint-every", "1000000",
]


def cache_digest(directory):
    combined = hashlib.sha256()
    count = 0
    for path in sorted(directory.rglob("*")):
        if path.is_file():
            combined.update(path.relative_to(directory).as_posix().encode())
            combined.update(bytes.fromhex(digest(path)))
            count += 1
    return {"files": count, "sha256": combined.hexdigest()}


COMPILED_DRIVER = DRIVER + """
    import json
    import sys
    from pathlib import Path
    from torch._dynamo.utils import counters
    root = Path(sys.argv[sys.argv.index("--output-dir") + 1])
    selected = {k: dict(v) for k, v in counters.items() if k in ("inductor", "aot_autograd", "stats")}
    (root / "compiler_counters.json").write_text(json.dumps(selected, default=str))
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--cpus", default="0")
    parser.add_argument("--compiled", action="store_true", help="enable the normal compiled actor update")
    parser.add_argument("--warm-from", type=Path, help="use identical copies of these previously compiled arm caches")
    parser.add_argument("--require-hot", action="store_true", help="repeat both arms after any new compiled graph")
    parser.add_argument("--seeds", default="563000100,563020100,563040100")
    args = parser.parse_args()
    if args.require_hot and args.warm_from is None:
        parser.error("--require-hot requires --warm-from")
    if args.warm_from is not None and not args.compiled:
        parser.error("--warm-from requires --compiled")
    if args.output.exists():
        parser.error("choose a fresh output file")
    torch.set_num_threads(1)
    scratch = args.scratch.resolve()
    scratch.mkdir(parents=True, exist_ok=False)
    hashes = stage_sources(scratch)
    subprocess.run(["patch", "-s", "-p1", "-i", str(HERE / "privileged_optimization.patch")],
                   check=True, cwd=scratch / "after")
    for name, expected in json.loads((HERE / "privileged_sources.json").read_text()).items():
        if digest(scratch / "after" / name) != expected:
            raise ValueError(f"Source hash mismatch for {name}")
        hashes[name]["after"] = expected
    common = list(COMMON)
    initial_cache = None
    if args.compiled:
        for arm in ARMS:
            (scratch / arm / "run.py").write_text(COMPILED_DRIVER)
        if args.warm_from is not None:
            for arm in ARMS:
                dest = scratch / f"compiler-{arm}"
                for prior_arm in ARMS:
                    shutil.copytree(args.warm_from / f"compiler-{prior_arm}", dest, dirs_exist_ok=True)
            initial_cache = {arm: cache_digest(scratch / f"compiler-{arm}") for arm in ARMS}
            assert initial_cache["before"] == initial_cache["after"]
        common[common.index("--no-compile-updates")] = "--compile-updates"
        os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    seeds = [int(seed) for seed in args.seeds.split(",")]
    result = dict(
        torch=str(torch.__version__), python=platform.python_version(), gpu=torch.cuda.get_device_name(),
        cpus=args.cpus, actor_sha256=digest(ACTOR), critic_sha256=digest(CRITIC),
        source_hashes=hashes, common_arguments=common, seeds=seeds, pairs=[], initial_compiler_cache=initial_cache,
        steady_epochs="middle three of five; the full process includes initialization and final evaluation",
    )
    for index, seed in enumerate(seeds):
        order = list(ARMS if index % 2 == 0 else reversed(ARMS))
        pair = dict(seed=seed, order=order)
        for attempt in range(3):
            for arm in order:
                if args.compiled:
                    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(scratch / f"compiler-{arm}")
                pair[arm] = train(scratch, arm, seed, args.cpus, common)
                if args.compiled:
                    stats = scratch / f"{arm}-{seed}" / "compiler_counters.json"
                    pair[arm]["compiler_counters"] = json.loads(stats.read_text())
                print(f'{arm} {seed}: {pair[arm]["process_wall_seconds"]:.3f}s', flush=True)
            missed = args.require_hot and any(
                pair[arm]["compiler_counters"].get("inductor", {}).get("fxgraph_cache_miss", 0)
                or pair[arm]["compiler_counters"].get("aot_autograd", {}).get("autograd_cache_miss", 0)
                for arm in order
            )
            if not missed:
                break
            if attempt == 2:
                raise RuntimeError(f"New compiler graphs still required for seed {seed}")
            result.setdefault("excluded_cache_warmups", []).append({
                "seed": seed, "attempt": attempt + 1,
                "counters": {arm: pair[arm]["compiler_counters"] for arm in order},
                "process_wall_seconds": {arm: pair[arm]["process_wall_seconds"] for arm in order},
            })
            for arm in order:
                for suffix in ("", ".log"):
                    (scratch / f"{arm}-{seed}{suffix}").rename(scratch / f"priming-{attempt}-{arm}-{seed}{suffix}")
            print(f"Repeating both arms of {seed} after cache warmup", flush=True)
        checks = {}
        for key in ("games", "turns", "actor_decisions", "optimizer_steps", "ppo_epochs",
                    "state_sha256", "critic_update"):
            checks[f"all_epochs_{key}"] = all(
                left[key] == right[key] for left, right in zip(pair["before"]["epochs"], pair["after"]["epochs"])
            )
        for filename in ("latest.pt", "privileged_critic_latest.pt"):
            before, after = [torch.load(scratch / f"{arm}-{seed}" / filename, map_location="cpu", weights_only=False)
                             for arm in ARMS]
            for key in ("state_dict", "optimizer_state_dict"):
                checks[f"{filename}_{key}"] = identical(before[key], after[key])
        pair["exact"] = checks
        result["pairs"].append(pair)
        result["metrics"] = summarize(result["pairs"])
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f'{seed} exact={all(checks.values())} running speedup={result["metrics"]["steady_speedup"]:.3f}x', flush=True)
    if not all(all(pair["exact"].values()) for pair in result["pairs"]):
        raise AssertionError("Training differences recorded in results")


if __name__ == "__main__":
    main()
