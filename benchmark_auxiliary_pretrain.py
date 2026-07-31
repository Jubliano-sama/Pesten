#!/usr/bin/env python3
"""Development-only public-history pretraining and residual-GRU RL pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from auxiliary_pretrain import (
    PublicHistoryAuxiliaryModel,
    initialize_residual_encoder,
    pretrain_auxiliary_encoder,
    validate_development_seed,
)
from benchmark_model_zoo import DEFAULT_TEACHER
from benchmark_residual_checkpoint import (
    deterministic_evaluation_agent_factory,
    deterministic_evaluation_runtime,
    evaluate_candidate_and_base_paired,
)
from residual_temporal_policy import ResidualRecurrentTemporalPolicy
from temporal_observation import serialize_safe_history
from temporal_teacher_parallel import collect_teacher_episodes_parallel
from temporal_training import collect_smoke_episodes
from temporal_zoo_rl import (
    PPOConfig,
    cpu_rollout_thread_scope,
    ppo_update_epochs,
)


def canonical_behavior_digest(episodes) -> str:
    """Hash safe decisions/actions/outcomes while excluding critic values."""
    payload = []
    for trajectory, won in episodes:
        payload.append(
            {
                "safe_decisions": json.loads(
                    serialize_safe_history(
                        step.decision for step in trajectory
                    )
                ),
                "actions": [int(step.action) for step in trajectory],
                "won": bool(won),
            }
        )
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_auxiliary_pilot(
    *,
    seed: int = 63_301,
    teacher_path: Path = DEFAULT_TEACHER,
    auxiliary_games: int = 48,
    auxiliary_workers: int = 4,
    auxiliary_epochs: int = 4,
    hidden_size: int = 64,
    num_layers: int = 2,
    cell: str = "gru",
    device: str = "cpu",
    rl_games: int = 12,
    rl_update_epochs: int = 1,
    eval_pairs: int = 10,
    learning_rate: float = 3e-4,
    auxiliary_checkpoint: Path | None = None,
    pretrained_residual_out: Path | None = None,
    baseline_residual_out: Path | None = None,
) -> dict:
    span = 6_000_000 + max(auxiliary_games, rl_games, eval_pairs) + 200_000
    validate_development_seed(seed, span=span)
    if min(
        auxiliary_games,
        auxiliary_workers,
        auxiliary_epochs,
        hidden_size,
        num_layers,
        rl_games,
        rl_update_epochs,
        eval_pairs,
    ) < 1:
        raise ValueError("pilot counts and model sizes must be positive")
    training_device = torch.device(device)
    if training_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    started = time.perf_counter()

    collected = collect_teacher_episodes_parallel(
        teacher_path=teacher_path,
        games=auxiliary_games,
        seed=seed,
        max_turns=160,
        workers=auxiliary_workers,
        torch_threads_per_worker=1,
    )
    usable = tuple(episode for episode in collected if len(episode) >= 2)
    if len(usable) < 2:
        raise RuntimeError("auxiliary collection produced too little history")
    validation_count = max(1, round(len(usable) * 0.25))
    validation = usable[-validation_count:]
    training = usable[:-validation_count]

    torch.manual_seed(seed + 10_000)
    auxiliary = PublicHistoryAuxiliaryModel(
        hidden_size=hidden_size,
        num_layers=num_layers,
        cell=cell,
    )
    auxiliary_result = pretrain_auxiliary_encoder(
        auxiliary,
        training,
        validation,
        device=training_device,
        epochs=auxiliary_epochs,
        batch_episodes=8,
        learning_rate=1e-3,
        delta_loss_weight=0.25,
        seed=seed + 100_000,
    )
    if auxiliary_checkpoint is not None:
        auxiliary.save_checkpoint(auxiliary_checkpoint)

    policy_seed = seed + 200_000
    torch.manual_seed(policy_seed)
    baseline = ResidualRecurrentTemporalPolicy(
        champion_path=teacher_path,
        hidden_size=hidden_size,
        num_layers=num_layers,
        cell=cell,
    )
    torch.manual_seed(policy_seed)
    pretrained = ResidualRecurrentTemporalPolicy(
        champion_path=teacher_path,
        hidden_size=hidden_size,
        num_layers=num_layers,
        cell=cell,
    )
    actor_before = {
        name: tensor.detach().clone()
        for name, tensor in pretrained.state_dict().items()
        if "actor" in name or name.startswith("critic.")
    }
    transfer = initialize_residual_encoder(pretrained, auxiliary)
    if any(
        not torch.equal(before, pretrained.state_dict()[name])
        for name, before in actor_before.items()
    ):
        raise RuntimeError("auxiliary transfer changed actor or critic")

    rollout_seed = seed + 1_000_000
    baseline_episodes, baseline_collection = collect_smoke_episodes(
        baseline,
        seed=rollout_seed,
        episode_count=rl_games,
        max_turns=160,
    )
    pretrained_episodes, pretrained_collection = collect_smoke_episodes(
        pretrained,
        seed=rollout_seed,
        episode_count=rl_games,
        max_turns=160,
    )
    baseline_digest = canonical_behavior_digest(baseline_episodes)
    pretrained_digest = canonical_behavior_digest(pretrained_episodes)
    if baseline_digest != pretrained_digest:
        raise RuntimeError(
            "zero-residual policies did not collect identical behavior"
        )

    baseline.to(training_device)
    pretrained.to(training_device)
    baseline_optimizer = torch.optim.AdamW(
        [p for p in baseline.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    pretrained_optimizer = torch.optim.AdamW(
        [p for p in pretrained.parameters() if p.requires_grad],
        lr=learning_rate,
    )
    ppo_config = PPOConfig()
    update_seed = seed + 2_000_000
    baseline_update = ppo_update_epochs(
        baseline,
        baseline_episodes,
        baseline_optimizer,
        config=ppo_config,
        update_epochs=rl_update_epochs,
        minibatch_episodes=4,
        seed=update_seed,
        target_kl=0.02,
    )
    pretrained_update = ppo_update_epochs(
        pretrained,
        pretrained_episodes,
        pretrained_optimizer,
        config=ppo_config,
        update_epochs=rl_update_epochs,
        minibatch_episodes=4,
        seed=update_seed,
        target_kl=0.02,
    )
    baseline.to("cpu").eval()
    pretrained.to("cpu").eval()

    from benchmark_agents import load_ppo_agent

    champion_factory = load_ppo_agent(teacher_path, "cpu")
    candidate_factory = deterministic_evaluation_agent_factory(
        pretrained, seed=seed + 3_000_000
    )
    base_factory = deterministic_evaluation_agent_factory(
        baseline, seed=seed + 3_000_000
    )
    evaluation_seed = seed + 5_000_000
    with deterministic_evaluation_runtime("cpu", cpu_threads=1):
        comparison = evaluate_candidate_and_base_paired(
            candidate_factory,
            base_factory,
            champion_factory,
            pairs=eval_pairs,
            seed=evaluation_seed,
            candidate_name="auxiliary_pretrained_residual",
            opponent_name="ppo_champion",
        )

    if baseline_residual_out is not None:
        baseline.save_checkpoint(baseline_residual_out)
    if pretrained_residual_out is not None:
        pretrained.save_checkpoint(pretrained_residual_out)
    return {
        "schema_version": 1,
        "scope": (
            "development-only public-history auxiliary pretraining and "
            "common-behavior packed PPO pilot"
        ),
        "seed": seed,
        "sealed_seed_start": 92_000_000,
        "device": str(training_device),
        "information_boundary": (
            "typed DecisionInput/PublicEventToken history and public hand "
            "sizes only; no opponent identities or draw-pile order"
        ),
        "auxiliary_data": {
            "games_requested": auxiliary_games,
            "usable_episodes": len(usable),
            "training_episodes": len(training),
            "heldout_episodes": len(validation),
        },
        "auxiliary_training": auxiliary_result,
        "encoder_transfer": transfer,
        "rl_common_behavior": {
            "seed": rollout_seed,
            "sha256": baseline_digest,
            "byte_semantic_equal": True,
            "baseline_collection": baseline_collection,
            "pretrained_collection": pretrained_collection,
        },
        "rl_update": {
            "seed": update_seed,
            "baseline": baseline_update,
            "auxiliary_pretrained": pretrained_update,
        },
        "gameplay": {
            "seed": evaluation_seed,
            "pairs": eval_pairs,
            "auxiliary_pretrained": comparison["candidate_result"],
            "random_encoder_control": comparison["base_result"],
            "paired_candidate_minus_control": comparison["comparison"],
        },
        "outputs": {
            "auxiliary_checkpoint": (
                str(auxiliary_checkpoint)
                if auxiliary_checkpoint is not None
                else None
            ),
            "pretrained_residual": (
                str(pretrained_residual_out)
                if pretrained_residual_out is not None
                else None
            ),
            "baseline_residual": (
                str(baseline_residual_out)
                if baseline_residual_out is not None
                else None
            ),
        },
        "elapsed_seconds": time.perf_counter() - started,
    }


def run_auxiliary_pilot(*, cpu_threads: int = 1, **kwargs) -> dict:
    """Run the pilot with a right-sized, restored CPU intra-op pool."""
    device = kwargs.get("device", "cpu")
    with cpu_rollout_thread_scope(device, cpu_threads):
        result = _run_auxiliary_pilot(**kwargs)
    result["cpu_threads"] = cpu_threads if device == "cpu" else None
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=63_301)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--auxiliary-games", type=int, default=48)
    parser.add_argument("--auxiliary-workers", type=int, default=4)
    parser.add_argument("--auxiliary-epochs", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--cell", choices=("gru", "lstm"), default="gru")
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--rl-games", type=int, default=12)
    parser.add_argument("--rl-update-epochs", type=int, default=1)
    parser.add_argument("--eval-pairs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--auxiliary-checkpoint", type=Path)
    parser.add_argument("--pretrained-residual-out", type=Path)
    parser.add_argument("--baseline-residual-out", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_auxiliary_pilot(
        seed=args.seed,
        teacher_path=args.teacher,
        auxiliary_games=args.auxiliary_games,
        auxiliary_workers=args.auxiliary_workers,
        auxiliary_epochs=args.auxiliary_epochs,
        hidden_size=args.hidden_size,
        num_layers=args.layers,
        cell=args.cell,
        device=args.device,
        rl_games=args.rl_games,
        rl_update_epochs=args.rl_update_epochs,
        eval_pairs=args.eval_pairs,
        learning_rate=args.learning_rate,
        cpu_threads=args.cpu_threads,
        auxiliary_checkpoint=args.auxiliary_checkpoint,
        pretrained_residual_out=args.pretrained_residual_out,
        baseline_residual_out=args.baseline_residual_out,
    )
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
