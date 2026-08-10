#!/usr/bin/env python3
"""Rank warm MLP checkpoints against random on common fresh paired deals.

This is a development screen, not a promotion benchmark. Every checkpoint is
evaluated on exactly the same seat-balanced deals so the highest-ranked models
can be rechecked on a disjoint, larger confirmation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from batched_pesten import ActorPolicy
from benchmark_checkpoint_batched import build_game_specs, run_batched_specs
from ppo2 import StudentNet


ACTOR_NAMES = (
    re.compile(r"^ppo_actor_(\d+)\.pth$"),
    re.compile(r"^reinforce_actor_(\d+)\.pth$"),
)


def actor_iteration(path: Path) -> int:
    for pattern in ACTOR_NAMES:
        match = pattern.fullmatch(path.name)
        if match:
            return int(match.group(1))
    raise ValueError(f"unsupported actor checkpoint name: {path.name}")


def discover_checkpoints(root: Path) -> list[Path]:
    checkpoints = [
        path
        for path in root.rglob("*actor_*.pth")
        if any(pattern.fullmatch(path.name) for pattern in ACTOR_NAMES)
    ]
    return sorted(
        checkpoints,
        key=lambda path: (
            str(path.parent),
            actor_iteration(path),
        ),
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def random_candidate_specs(*, pairs: int, base_seed: int):
    return [
        spec
        for spec in build_game_specs(
            pairs=pairs,
            base_seed=base_seed,
            pool_names=("random",),
        )
        if spec.section == "pool" and spec.evaluated_key == "candidate"
    ]


def evaluate_checkpoint(
    actor: StudentNet,
    checkpoint: Path,
    *,
    device: torch.device,
    specs,
    max_turns: int,
) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor.load_state_dict(state)
    actor.eval()
    outcomes, statistics = run_batched_specs(
        specs,
        {
            "candidate": ActorPolicy(
                actor=actor,
                device=device,
                temperature=1.0,
                collect_data=False,
                greedy=True,
            )
        },
        max_turns=max_turns,
    )
    scores = np.asarray([outcome.score for outcome in outcomes], dtype=np.float64)
    turns = np.asarray([outcome.turns for outcome in outcomes], dtype=np.float64)
    segment = next(
        (
            int(parent.name.removeprefix("segment_"))
            for parent in checkpoint.parents
            if (
                parent.name.startswith("segment_")
                and parent.name.removeprefix("segment_").isdigit()
            )
        ),
        None,
    )
    return {
        "path": str(checkpoint),
        "segment": segment,
        "lane": checkpoint.parent.name,
        "iteration": actor_iteration(checkpoint),
        "score": float(scores.mean()),
        "wins": int(np.count_nonzero(scores == 1.0)),
        "draws": int(np.count_nonzero(scores == 0.5)),
        "losses": int(np.count_nonzero(scores == 0.0)),
        "average_turns": float(turns.mean()),
        "games": int(scores.size),
        "throughput": statistics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("live_training/long_segments"),
    )
    parser.add_argument(
        "--ranking",
        type=Path,
        help="screen only the leading checkpoints from a prior sweep JSON",
    )
    parser.add_argument(
        "--ranking-top",
        type=int,
        default=32,
        help="number of prior ranked checkpoints to rescreen",
    )
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-turns", type=int, default=1_000)
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pairs < 1:
        raise SystemExit("--pairs must be positive")
    if args.top < 1:
        raise SystemExit("--top must be positive")
    if args.ranking_top < 1:
        raise SystemExit("--ranking-top must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    root = args.root.resolve()
    if args.ranking is None:
        checkpoints = discover_checkpoints(root)
    else:
        prior = json.loads(args.ranking.read_text(encoding="utf-8"))
        checkpoints = [
            Path(item["path"]).resolve()
            for item in prior["results"][: args.ranking_top]
        ]
    if not checkpoints:
        raise SystemExit(f"no warm MLP checkpoints found under {root}")

    device = torch.device(args.device)
    actor = StudentNet().to(device).eval()
    specs = random_candidate_specs(pairs=args.pairs, base_seed=args.seed)
    started = time.perf_counter()
    results = []
    for index, checkpoint in enumerate(checkpoints, start=1):
        result = evaluate_checkpoint(
            actor,
            checkpoint.resolve(),
            device=device,
            specs=specs,
            max_turns=args.max_turns,
        )
        results.append(result)
        if index % 25 == 0 or index == len(checkpoints):
            best = max(results, key=lambda item: item["score"])
            print(
                json.dumps(
                    {
                        "completed": index,
                        "total": len(checkpoints),
                        "best_path": best["path"],
                        "best_score": best["score"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    ranking = sorted(
        results,
        key=lambda item: (-item["score"], item["average_turns"], item["path"]),
    )
    for item in ranking[: args.top]:
        item["sha256"] = file_sha256(Path(item["path"]))
    payload = {
        "schema_version": 1,
        "kind": "common_deal_random_checkpoint_screen",
        "promotion_eligible": False,
        "promotion_reason": (
            "development ranking only; requires disjoint fresh-deal confirmation"
        ),
        "root": str(root),
        "ranking_source": (
            None if args.ranking is None else str(args.ranking.resolve())
        ),
        "device": str(device),
        "base_seed": args.seed,
        "pairs": args.pairs,
        "games_per_checkpoint": len(specs),
        "checkpoint_count": len(checkpoints),
        "elapsed_seconds": time.perf_counter() - started,
        "top": ranking[: args.top],
        "results": ranking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "elapsed_seconds": payload["elapsed_seconds"],
                "top": [
                    {
                        "path": item["path"],
                        "score": item["score"],
                        "average_turns": item["average_turns"],
                    }
                    for item in payload["top"][:10]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
