#!/usr/bin/env python3
"""High-throughput, restartable native Puffer PPO training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _strategic_pass_metrics(episodes, pass_action: int) -> dict:
    opportunities = passes = total_passes = decisions = 0
    for trajectory, _ in episodes:
        for step in trajectory:
            decisions += 1
            if step.action == pass_action:
                total_passes += 1
            strategic = (
                step.decision.legal_actions[pass_action]
                and any(step.decision.legal_actions[:pass_action])
            )
            if strategic:
                opportunities += 1
                passes += step.action == pass_action
    return {
        "action": pass_action,
        "decisions": decisions,
        "opportunities": opportunities,
        "passes": passes,
        "rate": passes / opportunities if opportunities else None,
        "all_pass_actions": total_passes,
    }


def _move_optimizer(optimizer, device) -> None:
    for state_values in optimizer.state.values():
        for name, value in tuple(state_values.items()):
            if isinstance(value, torch.Tensor):
                state_values[name] = value.to(device)


def _append_jsonl_idempotent(path: Path, row: dict) -> None:
    if path.is_file() and path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            position = end - 1
            while position >= 0:
                handle.seek(position)
                if handle.read(1) == b"\n" and position < end - 1:
                    position += 1
                    break
                position -= 1
            handle.seek(max(0, position))
            last_line = handle.read(end - max(0, position)).decode("utf-8").strip()
        previous = json.loads(last_line)
        if previous.get("epoch") == row.get("epoch"):
            previous_hash = previous.get("checkpoint", {}).get("state_sha256")
            current_hash = row.get("checkpoint", {}).get("state_sha256")
            if previous_hash != current_hash:
                raise RuntimeError(
                    "refusing conflicting rows for the same training epoch"
                )
            return
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--games-per-epoch", type=int, default=2048)
    parser.add_argument("--rollout-workers", type=int, default=8)
    parser.add_argument("--chunks-per-worker", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-trajectories", type=int, default=128)
    parser.add_argument(
        "--diagnostic-minibatch-trajectories", type=int, default=256
    )
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--evaluation-pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026081202)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive = (
        args.epochs,
        args.games_per_epoch,
        args.rollout_workers,
        args.chunks_per_worker,
        args.update_epochs,
        args.minibatch_trajectories,
        args.diagnostic_minibatch_trajectories,
        args.checkpoint_every,
        args.evaluate_every,
        args.evaluation_pairs,
    )
    if any(value < 1 for value in positive):
        raise ValueError("count arguments must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("fast Puffer training requested unavailable CUDA")
    resume = args.resume.resolve()
    if not resume.is_file():
        raise FileNotFoundError(resume)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from benchmark_agents import load_ppo_agent
    from live_puffer_segment import _load_policy_payload, _state_hash
    from parallel_puffer_rollout import collect_league_batch_parallel
    from randomAgent import AgentRandom
    from smartAgent import Agent as SmartAgent
    from temporal_league import _snapshot, evaluate_paired
    from temporal_observation import PASS_ACTION
    from temporal_puffer_agent import temporal_agent_factory
    from temporal_puffer_v4 import NativePufferTemporalPolicy
    from temporal_training import ppo_update_batched_epochs

    payload, state, hidden_size, num_layers, heads = _load_policy_payload(resume)
    policy = NativePufferTemporalPolicy(
        hidden_size=hidden_size,
        num_layers=num_layers,
        mode_conditioned_heads=heads,
    ).to(args.device)
    policy.load_state_dict(state)
    recipe = payload.get("recipe", {})
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(recipe.get("learning_rate", 3e-4))
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        _move_optimizer(optimizer, next(policy.parameters()).device)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    schedule = tuple(recipe.get("schedule", ("current_selfplay",)))
    if not schedule:
        raise ValueError("checkpoint has an empty league schedule")
    mixture = tuple(
        (name, schedule.count(name)) for name in dict.fromkeys(schedule)
    )
    player_counts = tuple(recipe.get("player_count_schedule", (2,)))
    start_epoch = int(payload.get("epoch", 0)) + 1
    source_sha256 = _file_sha256(resume)
    champion_path = Path("ppo_actor_100run3.pth").resolve()
    champion_factory = load_ppo_agent(champion_path, "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    started = time.perf_counter()
    curve = []
    result_path = output_dir / "training_result.json"
    jsonl_path = output_dir / "training_curve.jsonl"
    for offset in range(args.epochs):
        epoch = start_epoch + offset
        epoch_started = time.perf_counter()
        rollout_policy = _snapshot(policy)
        rollout_started = time.perf_counter()
        episodes, contribution = collect_league_batch_parallel(
            rollout_policy,
            games=args.games_per_epoch,
            seed=args.seed + epoch * args.games_per_epoch,
            schedule_offset=epoch * args.games_per_epoch,
            mixture=mixture,
            player_counts=player_counts,
            workers=args.rollout_workers,
            chunks_per_worker=args.chunks_per_worker,
            champion_path=champion_path,
        )
        rollout_seconds = time.perf_counter() - rollout_started
        pass_metrics = _strategic_pass_metrics(episodes, PASS_ACTION)
        if args.device == "cuda":
            torch.cuda.synchronize()
        update_started = time.perf_counter()
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
            diagnostic_minibatch_trajectories=(
                args.diagnostic_minibatch_trajectories
            ),
        )
        if args.device == "cuda":
            torch.cuda.synchronize()
        update_seconds = time.perf_counter() - update_started

        evaluations = None
        if epoch % args.evaluate_every == 0 or offset == args.epochs - 1:
            evaluation_policy = _snapshot(policy)
            candidate = temporal_agent_factory(
                evaluation_policy, deterministic=True, seed=args.seed + epoch
            )
            evaluations = {
                "ppo_champion": evaluate_paired(
                    candidate,
                    champion_factory,
                    pairs=args.evaluation_pairs,
                    seed=args.seed + 2_000_000 + epoch * 10_000,
                    candidate_name=f"puffer_epoch_{epoch}",
                    opponent_name="ppo_champion",
                ),
                "random": evaluate_paired(
                    candidate,
                    AgentRandom,
                    pairs=args.evaluation_pairs,
                    seed=args.seed + 3_000_000 + epoch * 10_000,
                    candidate_name=f"puffer_epoch_{epoch}",
                    opponent_name="random",
                ),
                "smart": evaluate_paired(
                    candidate,
                    SmartAgent,
                    pairs=args.evaluation_pairs,
                    seed=args.seed + 4_000_000 + epoch * 10_000,
                    candidate_name=f"puffer_epoch_{epoch}",
                    opponent_name="smart",
                ),
            }

        cpu_state = {
            name: value.detach().cpu().clone()
            for name, value in policy.state_dict().items()
        }
        checkpoint_payload = {
            "format": "pesten-fast-native-puffer-v1",
            "epoch": epoch,
            "model": {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "mode_conditioned_heads": heads,
            },
            "parent_checkpoint": str(resume),
            "source_checkpoint_sha256": source_sha256,
            "state_dict": cpu_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "recipe": {
                **recipe,
                "schedule": list(schedule),
                "player_count_schedule": list(player_counts),
                "games_per_epoch": args.games_per_epoch,
                "rollout_workers": args.rollout_workers,
                "chunks_per_worker": args.chunks_per_worker,
                "update_epochs": args.update_epochs,
                "minibatch_trajectories": args.minibatch_trajectories,
                "diagnostic_minibatch_trajectories": (
                    args.diagnostic_minibatch_trajectories
                ),
                "learning_rate": learning_rate,
                "target_kl": args.target_kl,
                "gamma": args.gamma,
                "gae_lambda": args.gae_lambda,
                "strategic_pass": {
                    "action": PASS_ACTION,
                    "legal_alongside_playable_cards": True,
                    "tracked_each_epoch": True,
                },
            },
        }
        _atomic_torch_save(checkpoint_payload, output_dir / "latest.pt")
        checkpoint_path = output_dir / "latest.pt"
        if epoch % args.checkpoint_every == 0 or offset == args.epochs - 1:
            checkpoint_path = output_dir / f"puffer_epoch_{epoch:06d}.pt"
            _atomic_torch_save(checkpoint_payload, checkpoint_path)

        row = {
            "epoch": epoch,
            "wall_seconds": time.perf_counter() - epoch_started,
            "rollout_seconds": rollout_seconds,
            "update_seconds": update_seconds,
            "games_per_second": args.games_per_epoch / rollout_seconds,
            "decisions_per_second": pass_metrics["decisions"] / rollout_seconds,
            "contribution": contribution,
            "strategic_pass": pass_metrics,
            "update": update,
            "evaluation": evaluations,
            "checkpoint": {
                "path": str(checkpoint_path),
                "state_sha256": _state_hash(cpu_state),
            },
        }
        curve.append(row)
        _append_jsonl_idempotent(jsonl_path, row)
        result = {
            "schema_version": 1,
            "kind": "fast_native_pufferlib_v4_training",
            "source_checkpoint": str(resume),
            "source_checkpoint_sha256": source_sha256,
            "start_epoch": start_epoch,
            "last_completed_epoch": epoch,
            "planned_epochs": args.epochs,
            "latest_checkpoint": str(output_dir / "latest.pt"),
            "curve": curve,
            "elapsed_seconds": time.perf_counter() - started,
            "runtime": {
                "device": str(next(policy.parameters()).device),
                "torch": torch.__version__,
                "cuda": torch.cuda.is_available(),
                "pufferlib": __import__("pufferlib").__version__,
            },
        }
        _atomic_json(result, result_path)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "games_per_second": row["games_per_second"],
                    "decisions": pass_metrics["decisions"],
                    "rollout_seconds": rollout_seconds,
                    "update_seconds": update_seconds,
                    "strategic_pass_rate": pass_metrics["rate"],
                    "evaluation": {
                        name: value["score"]
                        for name, value in (evaluations or {}).items()
                    },
                    "checkpoint": str(checkpoint_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
