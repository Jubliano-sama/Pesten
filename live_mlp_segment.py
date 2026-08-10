#!/usr/bin/env python3
"""Run one warm, restartable segment of the dense MLP PPO trainer.

The process deliberately owns one short segment only.  A caller can start a
fresh process from the emitted actor, critic, and optimizer checkpoints so
changes to ``ppo2.py`` are picked up without throwing away learned state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch


def _state_hash(path: Path) -> str:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    digest = hashlib.sha256()
    if isinstance(loaded, dict):
        items = loaded.items()
    else:
        items = (("payload", loaded),)
    for name, value in sorted(items, key=lambda item: str(item[0])):
        digest.update(str(name).encode("utf-8"))
        if isinstance(value, torch.Tensor):
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(repr(sorted(value)).encode("utf-8"))
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _parse_agent_sets(value: str) -> list[list[str]]:
    sets = [
        [entry.strip() for entry in group.split(",") if entry.strip()]
        for group in value.split(";")
    ]
    sets = [group for group in sets if group]
    if not sets or any(
        not any(entry.lower() in {"rl", "self"} for entry in group)
        for group in sets
    ):
        raise ValueError("every agent set must contain rl or self")
    return sets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-actor", type=Path, required=True)
    parser.add_argument("--resume-critic", type=Path, required=True)
    parser.add_argument("--resume-actor-optimizer", type=Path, required=True)
    parser.add_argument("--resume-critic-optimizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--updates-per-iteration", type=int)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--agent-sets", default="smart,rl;random,rl;self,self,rl")
    parser.add_argument("--past-policy-probability", type=float, default=0.2)
    parser.add_argument(
        "--compile-updates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trace-cpu-rollout-actor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parent
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "actor": args.resume_actor,
            "critic": args.resume_critic,
            "actor_optimizer": args.resume_actor_optimizer,
            "critic_optimizer": args.resume_critic_optimizer,
        }.items()
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing warm MLP {name}: {path}")
    if args.iterations < 1 or args.episodes < 1:
        raise ValueError("iterations and episodes must be positive")
    if args.rollout_batch_size < 1 or args.torch_threads < 1:
        raise ValueError("rollout batch size and torch threads must be positive")
    if args.learning_rate is not None and args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if args.clip is not None and not 0 < args.clip < 1:
        raise ValueError("clip must be between zero and one")
    if (
        args.updates_per_iteration is not None
        and args.updates_per_iteration < 1
    ):
        raise ValueError("updates per iteration must be positive")

    # Import after resolving the repository explicitly; the script is also
    # usable when launched from a supervisor outside the checkout.
    sys.path.insert(0, str(repo))
    from ppo2 import DEFAULT_TORCH_THREADS, PPO_Supervised

    del DEFAULT_TORCH_THREADS
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(args.output_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    agent_sets = _parse_agent_sets(args.agent_sets)
    started = time.perf_counter()
    compile_updates = bool(args.compile_updates and torch.cuda.is_available())
    ppo = PPO_Supervised(
        use_pretrained=True,
        actor_path=str(paths["actor"]),
        critic_path=str(paths["critic"]),
        actor_opt_path=str(paths["actor_optimizer"]),
        critic_opt_path=str(paths["critic_optimizer"]),
        old_save=(),
        max_saves=max(20, args.iterations + 2),
        rollout_device="cpu",
        rollout_batch_size=args.rollout_batch_size,
        training_actor_device="cuda" if torch.cuda.is_available() else "cpu",
        compile_updates=compile_updates,
        trace_cpu_rollout_actor=args.trace_cpu_rollout_actor,
        past_policy_probability=args.past_policy_probability,
        torch_threads=args.torch_threads,
    )
    if args.learning_rate is not None:
        ppo.lr = float(args.learning_rate)
        for optimizer in (ppo.actor_optim, ppo.critic_optim):
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = ppo.lr
    if args.clip is not None:
        ppo.clip = float(args.clip)
    if args.updates_per_iteration is not None:
        ppo.n_updates_per_iteration = int(args.updates_per_iteration)
    ppo.save_freq = 1
    ppo.learn(
        num_iterations=args.iterations,
        episodes_per_iter=args.episodes,
        temperature=args.temperature,
        # Gameplay evaluation is run by the outer supervisor on common fresh
        # seeds; avoid inserting a large, non-comparable evaluation into a
        # training segment.
        eval_freq=args.iterations + 1,
        eval_episodes=1,
        agent_sets=agent_sets,
    )

    latest = args.output_dir / f"ppo_actor_{args.iterations}.pth"
    latest_critic = args.output_dir / f"ppo_critic_{args.iterations}.pth"
    latest_actor_optimizer = args.output_dir / f"ppo_actor_opt_{args.iterations}.pth"
    latest_critic_optimizer = args.output_dir / f"ppo_critic_opt_{args.iterations}.pth"
    required = {
        "actor": latest,
        "critic": latest_critic,
        "actor_optimizer": latest_actor_optimizer,
        "critic_optimizer": latest_critic_optimizer,
    }
    for name, path in required.items():
        if not path.is_file():
            raise RuntimeError(f"training did not emit {name}: {path}")
    result = {
        "schema_version": 1,
        "kind": "warm_mlp_ppo_segment",
        "seed": args.seed,
        "iterations": args.iterations,
        "episodes_per_iteration": args.episodes,
        "agent_sets": agent_sets,
        "hyperparameters": {
            "learning_rate": ppo.lr,
            "clip": ppo.clip,
            "updates_per_iteration": ppo.n_updates_per_iteration,
            "temperature": args.temperature,
            "past_policy_probability": args.past_policy_probability,
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device": str(ppo.device),
            "actor_device": str(ppo.actor_device),
            "rollout_device": str(ppo.rollout_device),
            "compile_updates": ppo.compile_updates,
            "torch_threads": ppo.torch_threads,
        },
        "warm_parent": {
            name: {"path": str(path), "sha256": _state_hash(path)}
            for name, path in paths.items()
        },
        "latest": {
            name: {"path": str(path), "sha256": _state_hash(path)}
            for name, path in required.items()
        },
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "segment_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
