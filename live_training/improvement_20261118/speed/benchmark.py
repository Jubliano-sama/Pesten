"""Reconstruct both trainer versions, alternate them on paired seeds, and check weights."""
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

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ACTOR = ROOT / "models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt"
GAMES = 768
EPOCHS = 5
ARMS = ("before", "after")
DRIVER = """if __name__ == '__main__':
    import torch
    import train_puffer_fast
    torch.use_deterministic_algorithms(True)
    train_puffer_fast.main()
"""
COMMON = [
    "--resume", str(ACTOR), "--no-privileged-critic",
    "--schedule", "current_selfplay,canonical_random", "--player-count-schedule", "2,3,4",
    "--games-per-epoch", str(GAMES), "--epochs", str(EPOCHS), "--rollout-workers", "4",
    "--max-turns", "1000", "--update-epochs", "2", "--minibatch-trajectories", "64",
    "--early-stop-diagnostic-trajectories", "128", "--diagnostic-minibatch-trajectories", "64",
    "--campaign-offset", "0", "--fresh-optimizer", "--learning-rate", "0.00005",
    "--device", "cuda", "--update-precision", "float32", "--matmul-precision", "highest",
    "--no-compile-updates", "--evaluate-every", "1000000", "--evaluation-pairs", "1",
    "--checkpoint-every", "1000000",
]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def stage_sources(scratch):
    hashes = json.loads((HERE / "sources.json").read_text())
    for arm in ARMS:
        stage = scratch / arm
        stage.mkdir()
        with ZipFile(HERE / "ppo_sources_at_benchmark.zip") as archive:
            for name in hashes:
                (stage / name).write_bytes(archive.read(name))
        if arm == "before":
            subprocess.run(["patch", "-s", "-p1", "-i", str(HERE / "restore_reference.patch")],
                           check=True, cwd=stage)
        for name, expected in hashes.items():
            if digest(stage / name) != expected[arm]:
                raise ValueError(f"Source hash mismatch for {arm}/{name}")
        (stage / "run.py").write_text(DRIVER)
    return hashes


def train(scratch, arm, seed, cpus, common=COMMON):
    out = scratch / f"{arm}-{seed}"
    command = ["taskset", "-c", cpus, sys.executable, str(scratch / arm / "run.py"),
               *common, "--seed", str(seed), "--output-dir", str(out)]
    env = {**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}
    started = time.perf_counter()
    with (scratch / f"{arm}-{seed}.log").open("w") as log:
        subprocess.run(command, check=True, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - started
    rows = [json.loads(line) for line in (out / "training_curve.jsonl").read_text().splitlines()]
    assert len(rows) == EPOCHS
    epochs = []
    for row in rows:
        epochs.append({
            "wall_seconds": row["wall_seconds"], "rollout_seconds": row["rollout_seconds"],
            "update_seconds": row["update_seconds"], "actor_decisions": row["strategic_pass"]["decisions"],
            "optimizer_steps": row["update"]["optimizer_steps"], "ppo_epochs": row["update"]["epochs_completed"],
            "state_sha256": row["checkpoint"]["state_sha256"], "games": sum(row["contribution"]["games"].values()),
            "turns": row["contribution"]["total_turns"], "draws": row["contribution"].get("turn_limit_draws"),
        })
        if row["critic_update"] is not None:
            epochs[-1]["critic_update"] = row["critic_update"]
            epochs[-1]["critic_update_seconds"] = row["critic_update_seconds"]
    return {"process_wall_seconds": elapsed, "epochs": epochs}


def identical(left, right):
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(identical(value, right[key]) for key, value in left.items())
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(identical(a, b) for a, b in zip(left, right))
    return left == right


def summarize(pairs):
    metrics = {}
    for arm in ARMS:
        steady = [e for p in pairs for e in p[arm]["epochs"][1:-1]]
        seconds = sum(e["wall_seconds"] for e in steady)
        metrics[arm] = {
            "steady_epochs": len(steady), "steady_wall_seconds": seconds,
            "games_per_second": sum(e["games"] for e in steady) / seconds,
            "actor_decisions_per_second": sum(e["actor_decisions"] for e in steady) / seconds,
            "mean_epoch_seconds": seconds / len(steady),
            "steady_rollout_seconds": sum(e["rollout_seconds"] for e in steady),
            "steady_update_seconds": sum(e["update_seconds"] for e in steady),
            "all_process_wall_seconds": sum(p[arm]["process_wall_seconds"] for p in pairs),
        }
    metrics["steady_speedup"] = metrics["after"]["games_per_second"] / metrics["before"]["games_per_second"]
    metrics["all_process_speedup"] = metrics["before"]["all_process_wall_seconds"] / metrics["after"]["all_process_wall_seconds"]
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True, help="new directory for raw run logs and checkpoints")
    parser.add_argument("--cpus", default="0-3,8-11")
    parser.add_argument("--seeds", default="202611180,202611197,202611214,202611231,202611248")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    args.scratch = args.scratch.resolve()
    args.scratch.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    hashes = stage_sources(args.scratch)
    seeds = [int(seed) for seed in args.seeds.split(",")]
    pairs = []
    for index, seed in enumerate(seeds):
        order = list(ARMS if index % 2 == 0 else reversed(ARMS))
        pair = {"seed": seed, "order": order}
        for arm in order:
            pair[arm] = train(args.scratch, arm, seed, args.cpus)
            print(f'{arm} {seed}: {pair[arm]["process_wall_seconds"]:.3f}s', flush=True)
        left, right = [torch.load(args.scratch / f"{arm}-{seed}" / "latest.pt",
                                 map_location="cpu", weights_only=False) for arm in ARMS]
        pair["every_epoch_state_and_work_exact"] = all(
            a[key] == b[key] for a, b in zip(pair["before"]["epochs"], pair["after"]["epochs"])
            for key in ("games", "turns", "draws", "actor_decisions", "optimizer_steps", "ppo_epochs", "state_sha256")
        )
        pair["final_actor_exact"] = identical(left["state_dict"], right["state_dict"])
        pair["final_optimizer_exact"] = identical(left["optimizer_state_dict"], right["optimizer_state_dict"])
        assert pair["every_epoch_state_and_work_exact"] and pair["final_actor_exact"] and pair["final_optimizer_exact"]
        pairs.append(pair)
        print(f'{seed} verified; aggregate steady speedup {summarize(pairs)["steady_speedup"]:.3f}x', flush=True)
    result = {"torch": str(torch.__version__), "python": platform.python_version(), "gpu": torch.cuda.get_device_name(),
              "cpus": args.cpus, "actor_sha256": digest(ACTOR), "source_hashes": hashes,
              "common_arguments": COMMON, "steady_epochs": "middle three of each five; exclude initialization and final evaluation",
              "seeds": seeds, "pairs": pairs, "metrics": summarize(pairs)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
