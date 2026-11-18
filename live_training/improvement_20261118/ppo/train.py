"""Targeted PPO continuation from a softened decision head. Run from the repo root."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

import torch


ROOT = Path(__file__).resolve().parents[3]
TEACHER = ROOT / "models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=631104699)
    parser.add_argument("--temperature", type=float, default=2.5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()
    if not args.temperature > 0:
        parser.error("temperature must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    if args.resume is None:
        payload = torch.load(TEACHER, map_location="cpu", weights_only=False)
        payload.pop("optimizer_state_dict", None)
        # A positive scale preserves the teacher's initial greedy actions.
        # Both policy heads are softened. The value head stays on its original scale.
        for name, value in payload["state_dict"].items():
            if ".decoder.card_and_pass." in name or ".decoder.suit." in name:
                value.div_(args.temperature)
        actor = args.output / "start.pt"
        torch.save(payload, actor)
    else:
        actor = args.resume
    # Each 12-game opponent block visits every seat equally at 2, 3 and 4 players.
    schedule = ["historical_temporal"] * 12 + ["canonical_random"] * 12
    counts = [str(n) for n in (2, 3, 4) for _ in range(24)]
    command = [
        sys.executable, "-u", str(ROOT / "train_puffer_fast.py"),
        "--resume", str(actor.resolve()), "--output-dir", str(args.output.resolve()),
        "--historical-checkpoint", str(TEACHER), "--no-privileged-critic",
        "--schedule", ",".join(schedule), "--player-count-schedule", ",".join(counts),
        "--campaign-offset", str(args.offset), "--seed", str(args.seed),
        "--games-per-epoch", "1152", "--epochs", str(args.epochs), "--max-turns", "1000",
        "--rollout-workers", "3", "--learning-rate", "0.00012", "--update-epochs", "3",
        "--minibatch-trajectories", "96", "--gamma", "1.0", "--gae-lambda", "0.98",
        "--decision-reward", "0", "--draw-reward", "-1", "--entropy-coefficient", "0",
        "--target-kl", "0.017", "--evaluate-every", "1000000", "--evaluation-pairs", "4",
        "--checkpoint-every", "8", "--no-compile-updates", "--update-precision", "float32",
        "--device", "cuda",
    ]
    if args.resume is None:
        command.append("--fresh-optimizer")
    subprocess.run(command, cwd=ROOT, check=True, env={
        **os.environ, "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    })


if __name__ == "__main__":
    main()
