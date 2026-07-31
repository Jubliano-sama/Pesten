"""REJECTED imitation/distillation ablation; never use for accepted training.

The user explicitly rejected PPO mirroring as a training approach.  This file
is retained solely so earlier diagnostic results remain reproducible.  It
loads PPO actions/logits and is therefore prohibited from producing an
accepted or final checkpoint.  The accepted random-initialized reward-only
pipeline lives in ``temporal_reward_league.py``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import time

import numpy as np
import pufferlib
import torch

from benchmark_agents import load_ppo_agent
from ppo2 import StudentNet
from temporal_batched_speed import collect_league_batch_batched
from temporal_checkpoint_diagnostics import (
    SELECTION_WEIGHTS,
    disagreement_audit,
    evaluate_checkpoint,
)
from temporal_league import (
    CHAMPION_PATH,
    _snapshot,
    collect_teacher_episodes,
    imitation_metrics,
    imitation_update,
    imitation_update_targeted,
)
from temporal_puffer_v4 import (
    NativePufferTemporalPolicy,
    PUFFERLIB_REVISION,
)
from temporal_training import (
    ppo_update_batched,
    ppo_update_batched_epochs,
    save_safe_checkpoint,
)


DEVELOPMENT_TRAIN_SEEDS = (5_401, 15_401)
DEVELOPMENT_EVALUATION_SEED = 5_410_000
FRESH_FINAL_SEEDS = (90_731, 91_731, 92_731)


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    hidden_size: int = 128
    num_layers: int = 2
    mode_conditioned_heads: bool = False
    learning_rate: float = 7e-4
    targeted_imitation: bool = False
    distillation_weight: float = 0.5
    temperature: float = 1.0
    mode_balanced: bool = True
    suit_weight: float = 2.0
    many_legal_weight: float = 1.5
    early_weight: float = 1.25
    large_hand_weight: float = 1.5
    ppo_epochs: int = 1
    minibatch_trajectories: int = 16
    clip_ratio: float = 0.2
    target_kl: float | None = None
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    gae_lambda: float | None = None
    continued_imitation_epochs: int = 0
    context: str = "full_episode_recurrent"


CANDIDATES = {
    candidate.name: candidate
    for candidate in (
        Candidate(name="scalar_objective_control"),
        Candidate(
            name="targeted_gae",
            targeted_imitation=True,
            learning_rate=3e-4,
            ppo_epochs=4,
            target_kl=0.02,
            gae_lambda=0.95,
            continued_imitation_epochs=1,
        ),
        Candidate(
            name="mode_heads_targeted_gae",
            mode_conditioned_heads=True,
            targeted_imitation=True,
            learning_rate=3e-4,
            ppo_epochs=4,
            target_kl=0.02,
            gae_lambda=0.95,
            continued_imitation_epochs=1,
        ),
        Candidate(
            name="mode_heads_soft_distill",
            mode_conditioned_heads=True,
            targeted_imitation=True,
            distillation_weight=0.75,
            temperature=2.0,
            learning_rate=3e-4,
            ppo_epochs=4,
            target_kl=0.02,
            gae_lambda=0.95,
            continued_imitation_epochs=1,
        ),
        Candidate(
            name="wide_mode_heads",
            hidden_size=192,
            mode_conditioned_heads=True,
            targeted_imitation=True,
            distillation_weight=0.65,
            temperature=1.5,
            learning_rate=3e-4,
            ppo_epochs=4,
            target_kl=0.02,
            gae_lambda=0.95,
            continued_imitation_epochs=1,
        ),
        Candidate(
            name="deep_mode_heads",
            num_layers=3,
            mode_conditioned_heads=True,
            targeted_imitation=True,
            distillation_weight=0.65,
            temperature=1.5,
            learning_rate=3e-4,
            ppo_epochs=4,
            target_kl=0.02,
            gae_lambda=0.95,
            continued_imitation_epochs=1,
        ),
    )
}


def _imitation_train(
    candidate: Candidate,
    policy: NativePufferTemporalPolicy,
    episodes,
    optimizer,
    *,
    epochs: int,
    seed: int,
):
    if candidate.targeted_imitation:
        return imitation_update_targeted(
            policy,
            episodes,
            optimizer,
            epochs=epochs,
            seed=seed,
            batch_episodes=16,
            distillation_weight=candidate.distillation_weight,
            temperature=candidate.temperature,
            mode_balanced=candidate.mode_balanced,
            suit_weight=candidate.suit_weight,
            many_legal_weight=candidate.many_legal_weight,
            early_weight=candidate.early_weight,
            large_hand_weight=candidate.large_hand_weight,
        )
    return imitation_update(
        policy,
        episodes,
        optimizer,
        epochs=epochs,
        seed=seed,
        batch_episodes=16,
    )


def _ppo_train(candidate, policy, episodes, optimizer, *, seed):
    if candidate.ppo_epochs == 1 and candidate.gae_lambda is None:
        return ppo_update_batched(
            policy,
            episodes,
            optimizer,
            clip_ratio=candidate.clip_ratio,
            value_coefficient=candidate.value_coefficient,
            entropy_coefficient=candidate.entropy_coefficient,
        )
    return ppo_update_batched_epochs(
        policy,
        episodes,
        optimizer,
        update_epochs=candidate.ppo_epochs,
        minibatch_trajectories=candidate.minibatch_trajectories,
        seed=seed,
        target_kl=candidate.target_kl,
        clip_ratio=candidate.clip_ratio,
        value_coefficient=candidate.value_coefficient,
        entropy_coefficient=candidate.entropy_coefficient,
        gae_lambda=candidate.gae_lambda,
    )


def train_candidate(
    candidate: Candidate,
    *,
    train_seed: int,
    device: str,
    imitation_games: int,
    validation_games: int,
    imitation_epochs: int,
    league_epochs: int,
    league_games: int,
    development_pairs: int,
    output_dir: Path,
) -> dict:
    torch.manual_seed(train_seed)
    np.random.seed(train_seed)
    policy = NativePufferTemporalPolicy(
        hidden_size=candidate.hidden_size,
        num_layers=candidate.num_layers,
        mode_conditioned_heads=candidate.mode_conditioned_heads,
    ).to(device)
    optimizer = torch.optim.Adam(
        policy.parameters(), lr=candidate.learning_rate
    )
    teacher = StudentNet()
    teacher.load_state_dict(
        torch.load(CHAMPION_PATH, map_location="cpu", weights_only=True)
    )
    teacher.eval()
    champion_factory = load_ppo_agent(CHAMPION_PATH, "cpu")
    teacher_episodes = collect_teacher_episodes(
        policy,
        teacher,
        games=imitation_games,
        seed=train_seed + 100_000,
    )
    validation = collect_teacher_episodes(
        policy,
        teacher,
        games=validation_games,
        seed=DEVELOPMENT_EVALUATION_SEED + 700_000,
    )
    initial_validation = imitation_metrics(policy, validation)
    started = time.perf_counter()
    imitation_curve = _imitation_train(
        candidate,
        policy,
        teacher_episodes,
        optimizer,
        epochs=imitation_epochs,
        seed=train_seed + 200_000,
    )
    after_imitation = imitation_metrics(policy, validation)

    candidate_dir = output_dir / candidate.name / f"seed{train_seed}"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = []
    imitation_checkpoint = candidate_dir / "imitation.pt"
    save_safe_checkpoint(imitation_checkpoint, policy)
    checkpoints.append(
        {
            "training_seed": train_seed,
            "stage": "imitation",
            "epoch": 0,
            "path": imitation_checkpoint,
        }
    )

    historical = [_snapshot(policy)]
    league_curve = []
    total_decisions = sum(len(episode) for episode in teacher_episodes)
    for epoch in range(league_epochs):
        epoch_started = time.perf_counter()
        episodes, contribution = collect_league_batch_batched(
            policy,
            champion_factory,
            historical,
            games=league_games,
            seed=train_seed + 300_000 + epoch * league_games,
            schedule_offset=epoch * league_games,
        )
        ppo_metrics = _ppo_train(
            candidate,
            policy,
            episodes,
            optimizer,
            seed=train_seed + 400_000 + epoch,
        )
        continued_imitation = []
        if candidate.continued_imitation_epochs:
            continued_imitation = _imitation_train(
                candidate,
                policy,
                teacher_episodes,
                optimizer,
                epochs=candidate.continued_imitation_epochs,
                seed=train_seed + 500_000 + epoch,
            )
        total_decisions += sum(
            len(trajectory) for trajectory, _ in episodes
        )
        historical.append(_snapshot(policy))
        if len(historical) > 8:
            historical.pop(0)
        checkpoint = candidate_dir / f"league{epoch + 1}.pt"
        save_safe_checkpoint(checkpoint, policy)
        checkpoints.append(
            {
                "training_seed": train_seed,
                "stage": "league",
                "epoch": epoch + 1,
                "path": checkpoint,
            }
        )
        league_curve.append(
            {
                "epoch": epoch + 1,
                "wall_seconds": time.perf_counter() - epoch_started,
                "cumulative_decisions": total_decisions,
                "contribution": contribution,
                "ppo": ppo_metrics,
                "continued_imitation": continued_imitation,
                "validation": imitation_metrics(policy, validation),
                "checkpoint": str(checkpoint),
            }
        )

    evaluations = [
        evaluate_checkpoint(
            checkpoint,
            pairs=development_pairs,
            development_seed=DEVELOPMENT_EVALUATION_SEED,
            champion_factory=champion_factory,
        )
        for checkpoint in checkpoints
    ]
    best = max(evaluations, key=lambda item: item["selection_score"])
    bucket_audit = disagreement_audit(
        Path(best["checkpoint"]),
        games=validation_games,
        development_seed=DEVELOPMENT_EVALUATION_SEED + 800_000,
    )
    return {
        "candidate": asdict(candidate),
        "training_seed": train_seed,
        "selection_is_development_only": True,
        "imitation": {
            "games": imitation_games,
            "epochs": imitation_epochs,
            "initial_validation": initial_validation,
            "curve": imitation_curve,
            "after": after_imitation,
        },
        "league": {
            "epochs": league_epochs,
            "games_per_epoch": league_games,
            "curve": league_curve,
            "total_environment_decisions": total_decisions,
        },
        "checkpoint_evaluations": evaluations,
        "best_checkpoint": best,
        "best_checkpoint_bucket_audit": bucket_audit,
        "wall_seconds": time.perf_counter() - started,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-rejected-ablation",
        action="store_true",
        help=(
            "required acknowledgement: this PPO-derived ablation can never "
            "produce an accepted/final checkpoint"
        ),
    )
    parser.add_argument(
        "--candidates", default=",".join(CANDIDATES)
    )
    parser.add_argument("--train-seeds", default="5401")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--imitation-games", type=int, default=120)
    parser.add_argument("--validation-games", type=int, default=40)
    parser.add_argument("--imitation-epochs", type=int, default=6)
    parser.add_argument("--league-epochs", type=int, default=4)
    parser.add_argument("--league-games", type=int, default=40)
    parser.add_argument("--development-pairs", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/temporal/"
            "puffer4_temporal_development_sweep_20260731.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.allow_rejected_ablation:
        raise SystemExit(
            "rejected PPO imitation ablation disabled; use "
            "temporal_reward_league.py for accepted training"
        )
    names = tuple(args.candidates.split(","))
    unknown = set(names) - CANDIDATES.keys()
    if unknown:
        raise SystemExit(f"unknown candidates: {sorted(unknown)}")
    train_seeds = tuple(
        int(value) for value in args.train_seeds.split(",")
    )
    output_dir = args.output.parent / "development_sweep_checkpoints"
    runs = []
    for name in names:
        for seed in train_seeds:
            run = train_candidate(
                CANDIDATES[name],
                train_seed=seed,
                device=args.device,
                imitation_games=args.imitation_games,
                validation_games=args.validation_games,
                imitation_epochs=args.imitation_epochs,
                league_epochs=args.league_epochs,
                league_games=args.league_games,
                development_pairs=args.development_pairs,
                output_dir=output_dir,
            )
            runs.append(run)
            print(
                json.dumps(
                    {
                        "candidate": name,
                        "seed": seed,
                        "best_score": run["best_checkpoint"][
                            "selection_score"
                        ],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    ranked = sorted(
        runs,
        key=lambda item: item["best_checkpoint"]["selection_score"],
        reverse=True,
    )
    payload = {
        "schema_version": 1,
        "purpose": (
            "bounded development-only recipe selection; final seeds are "
            "declared but untouched"
        ),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pufferlib": str(pufferlib.__version__),
            "pufferlib_revision": PUFFERLIB_REVISION,
            "device": args.device,
        },
        "seed_policy": {
            "development_train_seeds_used": list(train_seeds),
            "development_evaluation_seed": DEVELOPMENT_EVALUATION_SEED,
            "fresh_final_seeds_reserved_untouched": list(
                FRESH_FINAL_SEEDS
            ),
        },
        "selection": {
            "weights": SELECTION_WEIGHTS,
            "losses_are_not_selection_criteria": True,
            "best_run": ranked[0]["best_checkpoint"],
            "ranking": [
                {
                    "candidate": run["candidate"]["name"],
                    "training_seed": run["training_seed"],
                    "selection_score": run["best_checkpoint"][
                        "selection_score"
                    ],
                    "checkpoint": run["best_checkpoint"]["checkpoint"],
                }
                for run in ranked
            ],
        },
        "puffer_tuning": {
            "used_native_v4_policy_and_mingru": True,
            "pufferl_autotune_used": False,
            "reason": (
                "the privacy adapter and mixed canonical/batched league are "
                "not a registered Puffer environment; the explicit grid is "
                "bounded, paired, and auditable"
            ),
        },
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(payload["selection"]["ranking"], indent=2)
    )


if __name__ == "__main__":
    main()
