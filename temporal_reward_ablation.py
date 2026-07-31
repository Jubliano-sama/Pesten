"""PPO-free development ablations for accepted reward-only training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

from temporal_reward_league import (
    ACCEPTED_SCHEDULE,
    train_reward_population,
)


SMART_HEAVY_SCHEDULE = (
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "historical_temporal",
    "historical_temporal",
    "bayesian_tree",
    "smart_rule",
    "smart_rule",
)
ANCHORLESS_SCHEDULE = (
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "current_selfplay",
    "historical_temporal",
    "historical_temporal",
)


@dataclass(frozen=True, slots=True)
class Ablation:
    name: str
    schedule: tuple[str, ...] = ACCEPTED_SCHEDULE
    pfsp: bool = True
    hidden_size: int = 192
    entropy_coefficient: float = 0.01


ABLATIONS = (
    Ablation(name="control_pfsp"),
    Ablation(name="uniform_history", pfsp=False),
    Ablation(name="smart_heavy", schedule=SMART_HEAVY_SCHEDULE),
    Ablation(name="anchorless", schedule=ANCHORLESS_SCHEDULE),
    Ablation(name="high_entropy", entropy_coefficient=0.02),
    Ablation(name="compact_128", hidden_size=128),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=6_401)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--games-per-epoch", type=int, default=60)
    parser.add_argument("--development-pairs", type=int, default=12)
    parser.add_argument("--pfsp-pairs", type=int, default=6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/temporal/"
            "puffer4_temporal_reward_only_ablation_20260731.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_root = (
        args.output.parent / "reward_only_ablation_checkpoints"
    )
    runs = []
    partial_path = args.output.with_suffix(".partial.json")
    for ablation in ABLATIONS:
        result = train_reward_population(
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
            games_per_epoch=args.games_per_epoch,
            development_pairs=args.development_pairs,
            pfsp_pairs=args.pfsp_pairs,
            hidden_size=ablation.hidden_size,
            num_layers=2,
            mode_conditioned_heads=True,
            learning_rate=3e-4,
            update_epochs=4,
            minibatch_trajectories=16,
            gamma=0.97,
            gae_lambda=0.95,
            clip_ratio=0.2,
            target_kl=0.02,
            entropy_coefficient=ablation.entropy_coefficient,
            value_coefficient=0.5,
            output_dir=checkpoint_root / ablation.name,
            schedule=ablation.schedule,
            pfsp_difficulty_prioritized=ablation.pfsp,
        )
        runs.append({"ablation": ablation.name, "result": result})
        partial_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "ppo_used": False,
                    "runs": runs,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    ranked = sorted(
        runs,
        key=lambda item: item["result"]["best_checkpoint"][
            "selection_score"
        ],
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "purpose": "reward-only non-PPO development ablation",
        "ppo_used_for_training_evaluation_or_selection": False,
        "seed": args.seed,
        "paired_budget": {
            "epochs": args.epochs,
            "games_per_epoch": args.games_per_epoch,
            "development_pairs": args.development_pairs,
            "pfsp_pairs": args.pfsp_pairs,
        },
        "ranking": [
            {
                "ablation": item["ablation"],
                "selection_score": item["result"]["best_checkpoint"][
                    "selection_score"
                ],
                "checkpoint": item["result"]["best_checkpoint"][
                    "checkpoint"
                ],
            }
            for item in ranked
        ],
        "runs": runs,
    }
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["ranking"], indent=2))


if __name__ == "__main__":
    main()
