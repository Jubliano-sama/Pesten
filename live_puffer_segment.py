#!/usr/bin/env python3
"""Run one warm, restartable native PufferLib 4 PPO segment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


def _state_hash(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _load_policy_payload(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"] if "state_dict" in payload else payload
    hidden_size = int(
        payload.get("model", {}).get("hidden_size", state["puffer_policy.encoder.encoder.weight"].shape[0])
    )
    num_layers = int(
        payload.get("model", {}).get(
            "num_layers",
            sum(
                key.startswith("puffer_policy.network.layers.")
                and key.endswith(".weight")
                for key in state
            ),
        )
    )
    mode_conditioned_heads = bool(
        payload.get("model", {}).get(
            "mode_conditioned_heads",
            any("card_and_pass" in key for key in state),
        )
    )
    return payload, state, hidden_size, num_layers, mode_conditioned_heads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--global-epoch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--games-per-epoch", type=int, default=100)
    parser.add_argument("--development-pairs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-trajectories", type=int, default=16)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--schedule",
        default="current_selfplay,current_selfplay,current_selfplay,current_selfplay,current_selfplay,current_selfplay,historical_temporal,historical_temporal,ppo_champion,bayesian_tree,smart_rule",
    )
    parser.add_argument(
        "--player-count-schedule",
        default="2,2,2,2,2,2,3,3,4,4,5,6,7,8",
        help="cyclic player counts; use two-player games most often",
    )
    return parser.parse_args()


def _parse_player_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not counts or any(count < 2 for count in counts):
        raise ValueError("player-count-schedule must contain integers >= 2")
    return counts


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parent
    sys.path.insert(0, str(repo))
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("native Puffer segment requested CUDA but it is unavailable")
    resume = args.resume.resolve()
    if not resume.is_file():
        raise FileNotFoundError(f"missing warm Puffer checkpoint: {resume}")
    if args.epochs < 1 or args.games_per_epoch < 1:
        raise ValueError("epochs and games per epoch must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from benchmark_agents import load_ppo_agent
    from temporal_league import (
        LEAGUE_MIXTURE,
        _snapshot,
        collect_league_batch,
        evaluate_paired,
        tree_factory,
    )
    from temporal_puffer_agent import temporal_agent_factory
    from temporal_puffer_v4 import NativePufferTemporalPolicy
    from temporal_training import ppo_update_batched_epochs
    from randomAgent import AgentRandom
    from smartAgent import Agent as SmartAgent

    payload, state, hidden_size, num_layers, mode_conditioned_heads = _load_policy_payload(resume)
    policy = NativePufferTemporalPolicy(
        hidden_size=hidden_size,
        num_layers=num_layers,
        mode_conditioned_heads=mode_conditioned_heads,
    ).to(args.device)
    policy.load_state_dict(state)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        # ``torch.load(..., map_location='cpu')`` keeps optimizer slots on
        # CPU.  Move them beside their resumed CUDA parameters before the
        # first update so a restart remains a true warm continuation.
        parameter_device = next(policy.parameters()).device
        for state_values in optimizer.state.values():
            for name, value in tuple(state_values.items()):
                if isinstance(value, torch.Tensor):
                    state_values[name] = value.to(parameter_device)
    start_epoch = int(payload.get("epoch", args.global_epoch - 1))
    if start_epoch != args.global_epoch - 1:
        raise ValueError(
            f"warm epoch mismatch: checkpoint={start_epoch}, requested={args.global_epoch - 1}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    champion_factory = load_ppo_agent(Path("ppo_actor_100run3.pth"), "cpu")
    schedule = tuple(value.strip() for value in args.schedule.split(",") if value.strip())
    if not schedule:
        raise ValueError("schedule cannot be empty")
    player_count_schedule = _parse_player_counts(args.player_count_schedule)
    started = time.perf_counter()
    curve = []
    for offset in range(args.epochs):
        epoch = args.global_epoch + offset
        epoch_started = time.perf_counter()
        rollout_policy = _snapshot(policy)
        historical = [_snapshot(rollout_policy)]
        episodes, contribution = collect_league_batch(
            rollout_policy,
            champion_factory,
            historical,
            games=args.games_per_epoch,
            seed=args.seed + epoch * args.games_per_epoch,
            schedule_offset=epoch * args.games_per_epoch,
            mixture=(
                tuple((name, schedule.count(name)) for name in dict.fromkeys(schedule))
            ),
            player_counts=player_count_schedule,
        )
        update = ppo_update_batched_epochs(
            policy,
            episodes,
            optimizer,
            update_epochs=args.update_epochs,
            minibatch_trajectories=args.minibatch_trajectories,
            seed=args.seed + 1_000_000 + epoch,
            target_kl=args.target_kl,
            clip_ratio=args.clip_ratio,
            value_coefficient=args.value_coefficient,
            entropy_coefficient=args.entropy_coefficient,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            normalize_advantages=True,
        )
        eval_policy = _snapshot(policy)
        candidate = temporal_agent_factory(eval_policy, deterministic=True, seed=args.seed + epoch)
        evaluations = {
            "ppo_champion": evaluate_paired(
                candidate, champion_factory, pairs=args.development_pairs,
                seed=args.seed + 2_000_000 + epoch * 10_000,
                candidate_name=f"puffer_epoch_{epoch}", opponent_name="ppo_champion",
            ),
            "random": evaluate_paired(
                candidate, AgentRandom, pairs=args.development_pairs,
                seed=args.seed + 3_000_000 + epoch * 10_000,
                candidate_name=f"puffer_epoch_{epoch}", opponent_name="random",
            ),
            "smart": evaluate_paired(
                candidate, SmartAgent, pairs=args.development_pairs,
                seed=args.seed + 4_000_000 + epoch * 10_000,
                candidate_name=f"puffer_epoch_{epoch}", opponent_name="smart",
            ),
        }
        checkpoint = args.output_dir / f"puffer_epoch_{epoch:04d}.pt"
        cpu_state = {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        }
        checkpoint_payload = {
            "format": "pesten-live-native-puffer-v1",
            "epoch": epoch,
            "model": {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "mode_conditioned_heads": mode_conditioned_heads,
            },
            "parent_checkpoint": str(resume),
            "state_dict": cpu_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "recipe": {
                "schedule": list(schedule),
                "player_count_schedule": list(player_count_schedule),
                "games_per_epoch": args.games_per_epoch,
                "update_epochs": args.update_epochs,
                "minibatch_trajectories": args.minibatch_trajectories,
                "learning_rate": args.learning_rate,
                "target_kl": args.target_kl,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
            },
        }
        torch.save(checkpoint_payload, checkpoint)
        torch.save(checkpoint_payload, args.output_dir / "latest.pt")
        row = {
            "epoch": epoch,
            "wall_seconds": time.perf_counter() - epoch_started,
            "decisions": sum(len(trajectory) for trajectory, _ in episodes),
            "contribution": contribution,
            "update": update,
            "evaluation": evaluations,
            "checkpoint": {
                "path": str(checkpoint),
                "state_sha256": _state_hash(cpu_state),
            },
        }
        curve.append(row)
        print(json.dumps({
            "epoch": epoch,
            "decisions": row["decisions"],
            "ppo_score": evaluations["ppo_champion"]["score"],
            "random_score": evaluations["random"]["score"],
            "smart_score": evaluations["smart"]["score"],
            "wall_seconds": row["wall_seconds"],
        }, sort_keys=True), flush=True)
        resume = checkpoint

    result = {
        "schema_version": 1,
        "kind": "warm_native_pufferlib_v4_segment",
        "seed": args.seed,
        "start_epoch": args.global_epoch,
        "end_epoch": args.global_epoch + args.epochs - 1,
        "curve": curve,
        "latest_checkpoint": str(args.output_dir / "latest.pt"),
        "elapsed_seconds": time.perf_counter() - started,
        "runtime": {
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device": str(next(policy.parameters()).device),
            "pufferlib": __import__("pufferlib").__version__,
        },
    }
    (args.output_dir / "segment_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
