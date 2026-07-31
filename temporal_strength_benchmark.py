"""Three-seed native-v4 temporal league training and champion evaluation."""

from __future__ import annotations

from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import pufferlib
import torch

from ppo2 import StudentNet
from randomAgent import AgentRandom
from smartAgent import Agent as SmartAgent
from temporal_league import (
    CHAMPION_PATH,
    LEAGUE_MIXTURE,
    collect_league_batch,
    collect_teacher_episodes,
    evaluate_paired,
    imitation_metrics,
    imitation_update,
    load_ppo_agent,
    tree_factory,
    verify_champion,
    _snapshot,
)
from temporal_puffer_agent import temporal_agent_factory
from temporal_puffer_v4 import (
    NativePufferTemporalPolicy,
    PUFFERLIB_REVISION,
)
from temporal_training import (
    evaluate_ppo_drift,
    ppo_update,
    save_safe_checkpoint,
)


def finite_metrics(metrics: dict) -> bool:
    return all(
        math.isfinite(value)
        for value in metrics.values()
        if isinstance(value, float)
    )


def train_seed(
    seed: int,
    *,
    imitation_games: int,
    validation_games: int,
    imitation_epochs: int,
    league_epochs: int,
    league_games_per_epoch: int,
    eval_pairs: int,
    output_dir: Path,
    device: str,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    champion_meta = verify_champion()
    champion_actor = StudentNet()
    champion_actor.load_state_dict(
        torch.load(CHAMPION_PATH, map_location="cpu", weights_only=True)
    )
    champion_actor.eval()
    champion_factory = load_ppo_agent(CHAMPION_PATH, "cpu")

    policy = NativePufferTemporalPolicy(
        hidden_size=128, num_layers=2
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=7e-4)

    train_teacher = collect_teacher_episodes(
        policy,
        champion_actor,
        games=imitation_games,
        seed=seed + 10_000,
    )
    validation = collect_teacher_episodes(
        policy,
        champion_actor,
        games=validation_games,
        seed=seed + 800_000,
    )
    initial_validation = imitation_metrics(policy, validation)
    started = time.perf_counter()
    imitation_curve = imitation_update(
        policy,
        train_teacher,
        optimizer,
        epochs=imitation_epochs,
        seed=seed + 20_000,
    )
    after_imitation = imitation_metrics(policy, validation)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_safe_checkpoint(
        output_dir / f"temporal_native_seed{seed}_imitation.pt", policy
    )

    historical = [_snapshot(policy)]
    league_curve = []
    data_contribution = {
        "games": defaultdict(int),
        "decisions": defaultdict(int),
    }
    total_decisions = sum(len(episode) for episode in train_teacher)
    time_to_gate = None
    for epoch in range(league_epochs):
        epoch_started = time.perf_counter()
        episodes, contribution = collect_league_batch(
            policy,
            champion_factory,
            historical,
            games=league_games_per_epoch,
            seed=seed + 100_000 + epoch * league_games_per_epoch,
            schedule_offset=epoch * league_games_per_epoch,
        )
        epoch_metrics = ppo_update(policy, episodes, optimizer)
        post_update = evaluate_ppo_drift(policy, episodes)
        total_decisions += int(epoch_metrics["samples"])
        for group in ("games", "decisions"):
            for name, value in contribution[group].items():
                data_contribution[group][name] += value
        validation_metrics = imitation_metrics(policy, validation)
        historical.append(_snapshot(policy))
        if len(historical) > 8:
            historical.pop(0)
        checkpoint = (
            output_dir / f"temporal_native_seed{seed}_league{epoch + 1}.pt"
        )
        save_safe_checkpoint(checkpoint, policy)
        quick_gate = evaluate_paired(
            temporal_agent_factory(policy, deterministic=True, seed=seed),
            champion_factory,
            pairs=max(10, min(30, eval_pairs // 4)),
            seed=seed + 900_000 + epoch * 1000,
            candidate_name=f"temporal_seed_{seed}",
            opponent_name="ppo_champion",
        )
        elapsed = time.perf_counter() - started
        if time_to_gate is None and quick_gate["gate_passed"]:
            time_to_gate = {
                "wall_seconds": elapsed,
                "environment_decisions": total_decisions,
                "checkpoint": str(checkpoint),
                "provisional_pairs": quick_gate["pairs"],
            }
        league_curve.append(
            {
                "epoch": epoch + 1,
                "wall_seconds": time.perf_counter() - epoch_started,
                "cumulative_wall_seconds": elapsed,
                "cumulative_decisions": total_decisions,
                "losses": epoch_metrics,
                "post_update_diagnostics": post_update,
                "validation": validation_metrics,
                "quick_ppo_gate": quick_gate,
                "checkpoint": str(checkpoint),
            }
        )

    candidate_factory = temporal_agent_factory(
        policy, deterministic=True, seed=seed
    )
    champion_eval = evaluate_paired(
        candidate_factory,
        champion_factory,
        pairs=eval_pairs,
        seed=seed + 2_000_000,
        candidate_name=f"temporal_seed_{seed}",
        opponent_name="ppo_champion",
    )
    pool = {
        "ppo_champion": champion_eval,
        "bayesian_tree": evaluate_paired(
            candidate_factory,
            tree_factory,
            pairs=max(20, eval_pairs // 2),
            seed=seed + 2_100_000,
            candidate_name=f"temporal_seed_{seed}",
            opponent_name="bayesian_tree",
        ),
        "smart_rule": evaluate_paired(
            candidate_factory,
            SmartAgent,
            pairs=max(20, eval_pairs // 2),
            seed=seed + 2_200_000,
            candidate_name=f"temporal_seed_{seed}",
            opponent_name="smart_rule",
        ),
        "random": evaluate_paired(
            candidate_factory,
            AgentRandom,
            pairs=max(20, eval_pairs // 2),
            seed=seed + 2_300_000,
            candidate_name=f"temporal_seed_{seed}",
            opponent_name="random",
        ),
    }
    final_validation = imitation_metrics(policy, validation)
    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "seed": seed,
        "champion": champion_meta,
        "training": {
            "imitation_games": imitation_games,
            "imitation_epochs": imitation_epochs,
            "league_epochs": league_epochs,
            "league_games_per_epoch": league_games_per_epoch,
            "league_mixture": [
                {"opponent": name, "parts": parts}
                for name, parts in LEAGUE_MIXTURE
            ],
            "opponent_data_contribution": {
                group: dict(values)
                for group, values in data_contribution.items()
            },
            "total_environment_decisions": total_decisions,
            "wall_seconds": elapsed,
            "decisions_per_second": total_decisions / elapsed,
            "imitation_curve": imitation_curve,
            "league_curve": league_curve,
            "time_to_provisional_gate": time_to_gate,
        },
        "heldout_champion_action_target": {
            "target": (
                "deployed frozen PPO masked-argmax action on held-out safe "
                "trajectories; this is actor agreement, not PPO loss"
            ),
            "initial": initial_validation,
            "after_imitation": after_imitation,
            "final": final_validation,
        },
        "evaluation": {
            "two_player_only": True,
            "paired_deal_seeds_and_rotated_seats": True,
            "pool": pool,
        },
        "stability": {
            "finite_imitation": all(
                finite_metrics(item) for item in imitation_curve
            ),
            "finite_league": all(
                finite_metrics(item["losses"]) for item in league_curve
            ),
        },
    }


def aggregate(runs: list[dict]) -> dict:
    scores = [
        run["evaluation"]["pool"]["ppo_champion"]["score"]
        for run in runs
    ]
    lower_bounds = [
        run["evaluation"]["pool"]["ppo_champion"][
            "paired_bootstrap_ci95"
        ][0]
        for run in runs
    ]
    return {
        "training_seeds": len(runs),
        "ppo_scores": scores,
        "mean_ppo_score": statistics.mean(scores),
        "min_seed_lower_ci95": min(lower_bounds),
        "all_seed_gates_passed": all(value > 0.5 for value in lower_bounds),
        "superiority_claim_allowed": (
            len(runs) >= 3 and all(value > 0.5 for value in lower_bounds)
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="731,1731,2731")
    parser.add_argument("--imitation-games", type=int, default=60)
    parser.add_argument("--validation-games", type=int, default=20)
    parser.add_argument("--imitation-epochs", type=int, default=4)
    parser.add_argument("--league-epochs", type=int, default=3)
    parser.add_argument("--league-games-per-epoch", type=int, default=30)
    parser.add_argument("--eval-pairs", type=int, default=100)
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/temporal/"
            "puffer4_temporal_vs_ppo_3seed_20260731.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    run_dir = args.output.parent / "puffer4_temporal_checkpoints"
    runs = [
        train_seed(
            seed,
            imitation_games=args.imitation_games,
            validation_games=args.validation_games,
            imitation_epochs=args.imitation_epochs,
            league_epochs=args.league_epochs,
            league_games_per_epoch=args.league_games_per_epoch,
            eval_pairs=args.eval_pairs,
            output_dir=run_dir,
            device=args.device,
        )
        for seed in seeds
    ]
    payload = {
        "schema_version": 1,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pufferlib": str(pufferlib.__version__),
            "pufferlib_revision": PUFFERLIB_REVISION,
            "device": args.device,
        },
        "budget_comparability": {
            "candidate_step_budget_recorded": True,
            "candidate_wall_budget_recorded": True,
            "ppo_champion_training_steps_available": False,
            "ppo_champion_training_wall_seconds_available": False,
            "step_matched_vs_champion": False,
            "wall_clock_matched_vs_champion": False,
            "reason": (
                "the deployed actor checkpoint contains weights only and no "
                "recoverable training step or wall-clock provenance"
            ),
        },
        "runs": runs,
        "aggregate": aggregate(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
