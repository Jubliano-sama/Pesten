"""Reproducible cold/warm benchmark for sequential and batched PPO rollouts."""

from __future__ import annotations

import argparse
import copy
import contextlib
import io
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn.functional as functional
from torch.distributions import Categorical

from ai_observation import OBSERVATION_SIZE
from ppo2 import (
    PPO_Supervised,
    autocast_for,
    compute_gae,
)


DEFAULT_SEEDS = (20260731, 20260732, 20260733)
DEFAULT_OUTPUT = Path("benchmarks/training_speed_vectorized_20260731.json")


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _rss_mib(field: str = "VmRSS") -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(f"{field}:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(usage) / 1024.0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _training_update(ppo, batch, temperature: float) -> dict:
    observations, actions, old_log_probabilities, rewards, lengths = batch
    _synchronize()
    prep_start = time.perf_counter()
    critic_observations = observations.to(ppo.device)
    critic_rewards = rewards.to(ppo.device)
    with torch.no_grad():
        advantages, returns = compute_gae(
            critic_rewards,
            ppo.critic_update_forward(critic_observations).squeeze(-1),
            episode_lengths=lengths,
            lam=0.95,
        )
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std() + 1e-8)
    actor_advantages = advantages.to(ppo.actor_device)
    masks = ppo.get_action_masks(observations)
    _synchronize()
    prep_seconds = time.perf_counter() - prep_start

    first_ratio_error = None
    first_kl = None
    final_kl = None
    actor_loss_value = None
    critic_loss_value = None
    _synchronize()
    update_start = time.perf_counter()
    for update_index in range(ppo.n_updates_per_iteration):
        with ppo.actor_update_autocast():
            logits = ppo.actor_update_forward(observations) / temperature
            masked_logits = logits.masked_fill(
                ~masks,
                torch.finfo(logits.dtype).min,
            )
            distribution = Categorical(
                logits=masked_logits,
                validate_args=False,
            )
            new_log_probabilities = distribution.log_prob(actions)
            ratios = torch.exp(
                new_log_probabilities - old_log_probabilities
            )
            kl_divergence = torch.mean(
                old_log_probabilities - new_log_probabilities
            )
            surrogate = ratios * actor_advantages
            clipped_surrogate = (
                torch.clamp(
                    ratios,
                    1 - ppo.clip,
                    1 + ppo.clip,
                )
                * actor_advantages
            )
            actor_loss = -torch.min(
                surrogate,
                clipped_surrogate,
            ).mean()
        if update_index == 0:
            first_ratio_error = float(
                ratios.detach().sub(1).abs().max().item()
            )
            if first_ratio_error > 0.02:
                raise RuntimeError(
                    "unchanged-policy ratio error reached "
                    f"{first_ratio_error:.6f}"
                )
            first_kl = float(kl_divergence.detach().item())
        with autocast_for(ppo.device):
            values = ppo.critic_update_forward(
                critic_observations
            ).squeeze(-1)
            critic_loss = functional.mse_loss(values, returns)

        ppo.actor_optim.zero_grad(set_to_none=True)
        ppo.critic_optim.zero_grad(set_to_none=True)
        shared_cuda_update = (
            ppo.actor_device == ppo.device
            and ppo.device.type == "cuda"
        )
        if shared_cuda_update:
            ppo.scaler.scale(actor_loss + 0.8 * critic_loss).backward()
        else:
            if ppo.actor_device.type == "cuda":
                ppo.scaler.scale(actor_loss).backward()
            else:
                actor_loss.backward()
            if ppo.device.type == "cuda":
                ppo.scaler.scale(0.8 * critic_loss).backward()
            else:
                (0.8 * critic_loss).backward()
        if shared_cuda_update:
            ppo.scaler.step(ppo.actor_optim)
            ppo.scaler.step(ppo.critic_optim)
            ppo.scaler.update()
        else:
            if ppo.actor_device.type == "cuda":
                ppo.scaler.step(ppo.actor_optim)
            else:
                ppo.actor_optim.step()
            if ppo.device.type == "cuda":
                ppo.scaler.step(ppo.critic_optim)
                ppo.scaler.update()
            else:
                ppo.critic_optim.step()
        actor_loss_value = float(actor_loss.detach().item())
        critic_loss_value = float(critic_loss.detach().item())
        final_kl = float(kl_divergence.detach().item())
    _synchronize()
    return {
        "prep_seconds": prep_seconds,
        "update_seconds": time.perf_counter() - update_start,
        "first_update_max_ratio_error": first_ratio_error,
        "first_update_kl": first_kl,
        "final_kl": final_kl,
        "final_actor_loss": actor_loss_value,
        "final_critic_loss": critic_loss_value,
    }


def _worker(args) -> dict:
    inductor_cache = None
    if args.compile_updates and torch.cuda.is_available():
        inductor_cache = tempfile.TemporaryDirectory(
            prefix="pesten-inductor-"
        )
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache.name
    torch.set_num_threads(args.threads)
    _seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    ppo = PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device=args.rollout_device,
        rollout_batch_size=args.batch_size,
        compile_updates=(
            args.compile_updates and torch.cuda.is_available()
        ),
    )
    generator = (
        ppo.generate_data
        if args.implementation == "vectorized"
        else ppo.generate_data_sequential
    )
    if args.thermal_state == "warm":
        actor_state = {
            name: value.detach().cpu().clone()
            for name, value in ppo.actor.state_dict().items()
        }
        critic_state = {
            name: value.detach().cpu().clone()
            for name, value in ppo.critic.state_dict().items()
        }
        actor_optimizer_state = copy.deepcopy(
            ppo.actor_optim.state_dict()
        )
        critic_optimizer_state = copy.deepcopy(
            ppo.critic_optim.state_dict()
        )
        scaler_state = copy.deepcopy(ppo.scaler.state_dict())
        _seed_everything(args.seed ^ 0xA5A5A5A5)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            warmup_batch = generator(
                num_episodes=args.warmup_episodes,
                temperature=args.temperature,
                agent_sets=[["smart", "rl"]],
            )
            _training_update(ppo, warmup_batch, args.temperature)
        ppo.actor.load_state_dict(actor_state)
        ppo.critic.load_state_dict(critic_state)
        ppo.actor_optim.load_state_dict(actor_optimizer_state)
        ppo.critic_optim.load_state_dict(critic_optimizer_state)
        ppo.scaler.load_state_dict(scaler_state)
        ppo._sync_rollout_actor()
    # Model construction consumes Torch randomness.  Reset immediately before
    # every measured run so cold and warm trajectories for a given
    # implementation/seed are identical.
    _seed_everything(args.seed)

    rss_start_mib = _rss_mib()
    _synchronize()
    rollout_start = time.perf_counter()
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        batch = generator(
            num_episodes=args.episodes,
            temperature=args.temperature,
            agent_sets=[["smart", "rl"]],
        )
    _synchronize()
    rollout_seconds = time.perf_counter() - rollout_start
    update = _training_update(ppo, batch, args.temperature)
    total_seconds = (
        rollout_seconds
        + update["prep_seconds"]
        + update["update_seconds"]
    )
    decisions = len(batch[1])
    metrics = ppo.last_rollout_metrics
    rss_peak_mib = _rss_mib("VmHWM")
    cuda_peak_mib = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )
    result = {
        "implementation": args.implementation,
        "thermal_state": args.thermal_state,
        "seed": args.seed,
        "episodes": args.episodes,
        "environment_turns": metrics["environment_turns"],
        "collected_policy_decisions": decisions,
        "episode_samples": len(batch[4]),
        "inference_calls": metrics["inference_calls"],
        "mean_inference_batch_size": (
            decisions / metrics["inference_calls"]
            if metrics["inference_calls"]
            else 0.0
        ),
        "rollout_seconds": rollout_seconds,
        **update,
        "total_seconds": total_seconds,
        "environment_turns_per_second": (
            metrics["environment_turns"] / rollout_seconds
        ),
        "collected_policy_decisions_per_second": decisions / rollout_seconds,
        "games_per_second": args.episodes / rollout_seconds,
        "end_to_end_policy_decisions_per_second": decisions / total_seconds,
        "rss_start_mib": rss_start_mib,
        "rss_peak_mib": rss_peak_mib,
        "rss_peak_delta_mib": max(0.0, rss_peak_mib - rss_start_mib),
        "cuda_peak_allocated_mib": cuda_peak_mib,
        "actor_update_device": str(ppo.actor_device),
        "critic_update_device": str(ppo.device),
        "compiled_updates": ppo.compile_updates,
        "rollout_buffer_peak_mib": (
            metrics["peak_buffer_bytes"] / (1024 * 1024)
        ),
    }
    # Retain the temporary compiler cache through all timed work.  It is
    # discarded only after constructing the result so each worker is a true
    # compiler-cold process while warm measurements reuse their own cache.
    if inductor_cache is not None:
        inductor_cache.cleanup()
    return result


def _median_summary(runs: list[dict]) -> dict:
    fields = (
        "environment_turns",
        "collected_policy_decisions",
        "inference_calls",
        "mean_inference_batch_size",
        "rollout_seconds",
        "prep_seconds",
        "update_seconds",
        "total_seconds",
        "environment_turns_per_second",
        "collected_policy_decisions_per_second",
        "games_per_second",
        "end_to_end_policy_decisions_per_second",
        "rss_peak_mib",
        "rss_peak_delta_mib",
        "cuda_peak_allocated_mib",
        "rollout_buffer_peak_mib",
        "first_update_max_ratio_error",
        "first_update_kl",
        "final_kl",
    )
    return {
        field: statistics.median(run[field] for run in runs)
        for field in fields
    }


def _git_output(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _parent(args) -> dict:
    runs = []
    for implementation in ("sequential", "vectorized"):
        for thermal_state in ("cold", "warm"):
            for seed in args.seeds:
                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--implementation",
                    implementation,
                    "--thermal-state",
                    thermal_state,
                    "--seed",
                    str(seed),
                    "--episodes",
                    str(args.episodes),
                    "--warmup-episodes",
                    str(args.warmup_episodes),
                    "--temperature",
                    str(args.temperature),
                    "--batch-size",
                    str(args.batch_size),
                    "--threads",
                    str(args.threads),
                    "--rollout-device",
                    args.rollout_device,
                    (
                        "--compile-updates"
                        if args.compile_updates
                        else "--no-compile-updates"
                    ),
                ]
                completed = subprocess.run(
                    command,
                    check=True,
                    text=True,
                    capture_output=True,
                    env={**os.environ, "PYTHONHASHSEED": str(seed)},
                )
                runs.append(json.loads(completed.stdout))

    grouped = {}
    for implementation in ("sequential", "vectorized"):
        grouped[implementation] = {}
        for thermal_state in ("cold", "warm"):
            selected = [
                run
                for run in runs
                if run["implementation"] == implementation
                and run["thermal_state"] == thermal_state
            ]
            grouped[implementation][thermal_state] = {
                "runs": selected,
                "median": _median_summary(selected),
            }

    speedups = {}
    higher_is_better = (
        "environment_turns_per_second",
        "collected_policy_decisions_per_second",
        "games_per_second",
        "end_to_end_policy_decisions_per_second",
        "mean_inference_batch_size",
    )
    lower_is_better = (
        "rollout_seconds",
        "prep_seconds",
        "update_seconds",
        "total_seconds",
    )
    for thermal_state in ("cold", "warm"):
        before = grouped["sequential"][thermal_state]["median"]
        after = grouped["vectorized"][thermal_state]["median"]
        speedups[thermal_state] = {
            field: after[field] / before[field]
            for field in higher_is_better
        }
        speedups[thermal_state].update(
            {
                f"{field}_speedup": before[field] / after[field]
                for field in lower_is_better
            }
        )

    baseline_path = Path("benchmarks/training_performance_20260730.json")
    baseline = json.loads(baseline_path.read_text())
    status_lines = _git_output("status", "--short").splitlines()
    return {
        "status": "final",
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "baseline_reference": {
            "path": str(baseline_path),
            "recorded_at": baseline["recorded_at"],
            "final_median": baseline["final"]["median"],
            "comparability_caveat": baseline["final"][
                "comparability_caveat"
            ],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "torch_threads": args.threads,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "git": {
            "revision": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "worktree_dirty": bool(status_lines),
            "status_short": status_lines,
            "diff_stat": _git_output("diff", "--stat"),
        },
        "workload": {
            "implementations": ["sequential", "vectorized"],
            "thermal_states": {
                "cold": "fresh Python process with no rollout warmup",
                "warm": (
                    f"fresh Python process after {args.warmup_episodes} "
                    "unmeasured rollout games and one restored-state PPO update"
                ),
            },
            "seeds": list(args.seeds),
            "repeats_per_implementation_and_thermal_state": len(args.seeds),
            "episodes_per_run": args.episodes,
            "agent_sets": [["smart", "rl"]],
            "players_per_game": 2,
            "maximum_turns_per_game": 500,
            "pretrained": False,
            "historical_checkpoints": False,
            "temperature": args.temperature,
            "policy_updates": 3,
            "actor": "StudentNet: DenseSkipNet, eight 256-wide hidden layers",
            "critic": "StudentCritic: DenseSkipNet, eight 512-wide hidden layers",
            "rollout_device": args.rollout_device,
            "training_actor_device": (
                "cuda" if torch.cuda.is_available() else "cpu"
            ),
            "compiled_updates": (
                args.compile_updates and torch.cuda.is_available()
            ),
            "vector_game_batch_size": args.batch_size,
            "observation_features": OBSERVATION_SIZE,
            "actions": 55,
        },
        "results": grouped,
        "speedups_vectorized_over_sequential": speedups,
        "semantic_caveats": [
            (
                "Each fixed seed resets Python, NumPy, and Torch RNGs, but "
                "batched categorical sampling consumes the Torch stream in a "
                "different order from scalar sampling, so trajectories and "
                "decision counts are distribution-comparable rather than "
                "pairwise identical.  The vector engine also partitions deck "
                "randomness into one independent stream per game."
            ),
            (
                "The canonical differential suite separately proves exact "
                "rules, deck, hand-order, duplicate, terminal, and truncation "
                "equivalence under deterministic decisions across 128 seeds."
            ),
            (
                "Compiled CUDA updates have a large one-time graph-build cost. "
                "Cold workers use fresh temporary compiler caches; warm workers "
                "run one unmeasured update, restore model/optimizer/scaler "
                "state, and reset RNGs before measurement. Short runs can use "
                "--no-compile_updates."
            ),
            (
                "This machine exposed no CUDA device to the benchmark process; "
                "the recorded workload is CPU rollout and CPU PPO update."
            )
            if not torch.cuda.is_available()
            else (
                "Rollouts use the requested device and the critic uses CUDA; "
                "CUDA allocated memory excludes driver/context reservations."
            ),
        ],
        "commands": {
            "benchmark": " ".join(
                [
                    str(Path(sys.executable)),
                    str(Path(__file__).name),
                    "--episodes",
                    str(args.episodes),
                    "--seeds",
                    *[str(seed) for seed in args.seeds],
                    "--output",
                    str(args.output),
                ]
            ),
            "focused_tests": (
                f"{sys.executable} -m unittest "
                "tests.test_batched_pesten_equivalence -v"
            ),
            "full_tests": f"{sys.executable} -m unittest discover -s tests -v",
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--implementation",
        choices=("sequential", "vectorized"),
    )
    parser.add_argument(
        "--thermal-state",
        choices=("cold", "warm"),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--warmup-episodes", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--rollout-device",
        choices=("auto", "cpu", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--compile-updates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.worker:
        if (
            args.implementation is None
            or args.thermal_state is None
            or args.seed is None
        ):
            raise SystemExit(
                "--worker requires --implementation, --thermal-state, and --seed"
            )
        print(json.dumps(_worker(args), sort_keys=True))
        return
    result = _parent(args)
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite benchmark result: {args.output}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
