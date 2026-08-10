"""Reproducible cold/warm benchmark for sequential and batched PPO rollouts."""

from __future__ import annotations

import argparse
import copy
import contextlib
import hashlib
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
    action_log_probabilities,
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


def _module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _rollout_sha256(batch) -> str:
    digest = hashlib.sha256()
    for tensor in batch[:4]:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    digest.update(np.asarray(batch[4], dtype=np.int64).tobytes())
    return digest.hexdigest()


def _training_update(ppo, batch, temperature: float) -> dict:
    observations, actions, old_log_probabilities, rewards, lengths = batch
    _synchronize()
    prep_start = time.perf_counter()
    critic_observations = observations.to(ppo.device)
    critic_rewards = rewards.to(ppo.device)
    reusable_critic_values = None
    if (
        ppo.reuse_cpu_critic_forward
        and ppo.device.type == "cpu"
    ):
        reusable_critic_values = (
            ppo._critic_forward_with_two_bn_updates(
                critic_observations
            )
        )
        gae_values = reusable_critic_values.detach()
    else:
        with torch.no_grad():
            gae_values = ppo.critic_update_forward(
                critic_observations
            ).squeeze(-1)
    with torch.no_grad():
        if ppo.native_gae:
            from numba_gae import compute_gae_native

            advantages, returns = compute_gae_native(
                critic_rewards,
                gae_values,
                lengths,
                gamma=0.97,
                lam=0.95,
            )
        else:
            advantages, returns = compute_gae(
                critic_rewards,
                gae_values,
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
            new_log_probabilities = action_log_probabilities(
                masked_logits,
                actions,
            )
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
            values = (
                reusable_critic_values
                if (
                    update_index == 0
                    and reusable_critic_values is not None
                )
                else ppo.critic_update_forward(
                    critic_observations
                ).squeeze(-1)
            )
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
        hybrid_rollout_inference=args.hybrid_rollout_inference,
        hybrid_rollout_threshold=args.hybrid_rollout_threshold,
        cuda_rollout_autocast=args.cuda_rollout_autocast,
        reuse_cpu_critic_forward=args.reuse_cpu_critic_forward,
        native_gae=getattr(args, "native_gae", True),
        trace_cpu_rollout_actor=args.trace_cpu_rollout_actor,
        native_smart_policy=args.native_smart_policy,
        native_rollout_engine=args.native_rollout_engine,
        record_rollout_public_events=args.record_rollout_public_events,
        direct_rollout_sampling=args.direct_rollout_sampling,
        numpy_rollout_masks=args.numpy_rollout_masks,
        reuse_rollout_workspaces=args.reuse_rollout_workspaces,
        coalesce_rollout_agent_sets=args.coalesce_rollout_agent_sets,
        group_compatible_rollout_agent_sets=(
            args.group_compatible_rollout_agent_sets
        ),
        coordinate_rollout_engines=args.coordinate_rollout_engines,
        drain_non_neural_rollout=args.drain_non_neural_rollout,
        reuse_rollout_engines=getattr(
            args,
            "reuse_rollout_engines",
            True,
        ),
        past_policy_pool_size=args.past_policy_pool_size,
        torch_threads=args.threads,
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
        "rollout_sha256": _rollout_sha256(batch),
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
        "hybrid_rollout_inference": ppo.hybrid_rollout_inference,
        "hybrid_rollout_threshold": ppo.hybrid_rollout_threshold,
        "cuda_rollout_autocast": ppo.cuda_rollout_autocast,
        "reuse_cpu_critic_forward": ppo.reuse_cpu_critic_forward,
        "native_gae": ppo.native_gae,
        "traced_cpu_rollout_actor": ppo.trace_cpu_rollout_actor,
        "native_smart_policy": ppo.native_smart_policy,
        "native_rollout_engine": ppo.native_rollout_engine,
        "record_rollout_public_events": ppo.record_rollout_public_events,
        "direct_rollout_sampling": ppo.direct_rollout_sampling,
        "numpy_rollout_masks": ppo.numpy_rollout_masks,
        "reuse_rollout_workspaces": ppo.reuse_rollout_workspaces,
        "coalesce_rollout_agent_sets": ppo.coalesce_rollout_agent_sets,
        "group_compatible_rollout_agent_sets": (
            ppo.group_compatible_rollout_agent_sets
        ),
        "coordinate_rollout_engines": ppo.coordinate_rollout_engines,
        "drain_non_neural_rollout": ppo.drain_non_neural_rollout,
        "reuse_rollout_engines": ppo.reuse_rollout_engines,
        "past_policy_pool_size": ppo.past_policy_pool_size,
        "torch_threads": ppo.torch_threads,
        "actor_state_sha256": _module_sha256(ppo.actor),
        "critic_state_sha256": _module_sha256(ppo.critic),
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
                    (
                        "--cuda-rollout-autocast"
                        if args.cuda_rollout_autocast
                        else "--no-cuda-rollout-autocast"
                    ),
                    (
                        "--hybrid-rollout-inference"
                        if args.hybrid_rollout_inference
                        else "--no-hybrid-rollout-inference"
                    ),
                    "--hybrid-rollout-threshold",
                    str(args.hybrid_rollout_threshold),
                    (
                        "--reuse-cpu-critic-forward"
                        if args.reuse_cpu_critic_forward
                        else "--no-reuse-cpu-critic-forward"
                    ),
                    (
                        "--native-gae"
                        if args.native_gae
                        else "--no-native-gae"
                    ),
                    (
                        "--trace-cpu-rollout-actor"
                        if args.trace_cpu_rollout_actor
                        else "--no-trace-cpu-rollout-actor"
                    ),
                    (
                        "--native-smart-policy"
                        if args.native_smart_policy
                        else "--no-native-smart-policy"
                    ),
                    (
                        "--direct-rollout-sampling"
                        if args.direct_rollout_sampling
                        else "--no-direct-rollout-sampling"
                    ),
                    (
                        "--native-rollout-engine"
                        if args.native_rollout_engine
                        else "--no-native-rollout-engine"
                    ),
                    (
                        "--record-rollout-public-events"
                        if args.record_rollout_public_events
                        else "--no-record-rollout-public-events"
                    ),
                    (
                        "--numpy-rollout-masks"
                        if args.numpy_rollout_masks
                        else "--no-numpy-rollout-masks"
                    ),
                    (
                        "--reuse-rollout-workspaces"
                        if args.reuse_rollout_workspaces
                        else "--no-reuse-rollout-workspaces"
                    ),
                    (
                        "--coalesce-rollout-agent-sets"
                        if args.coalesce_rollout_agent_sets
                        else "--no-coalesce-rollout-agent-sets"
                    ),
                    (
                        "--group-compatible-rollout-agent-sets"
                        if args.group_compatible_rollout_agent_sets
                        else "--no-group-compatible-rollout-agent-sets"
                    ),
                    (
                        "--coordinate-rollout-engines"
                        if args.coordinate_rollout_engines
                        else "--no-coordinate-rollout-engines"
                    ),
                    (
                        "--drain-non-neural-rollout"
                        if args.drain_non_neural_rollout
                        else "--no-drain-non-neural-rollout"
                    ),
                    (
                        "--reuse-rollout-engines"
                        if args.reuse_rollout_engines
                        else "--no-reuse-rollout-engines"
                    ),
                    "--past-policy-pool-size",
                    str(args.past_policy_pool_size),
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
            "hybrid_rollout_inference": (
                args.hybrid_rollout_inference
                and torch.cuda.is_available()
                and args.rollout_device in ("auto", "cpu")
            ),
            "hybrid_rollout_threshold": args.hybrid_rollout_threshold,
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
        "--cuda-rollout-autocast",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--hybrid-rollout-inference",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--hybrid-rollout-threshold",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--reuse-cpu-critic-forward",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--native-gae",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--trace-cpu-rollout-actor",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--native-smart-policy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--direct-rollout-sampling",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--native-rollout-engine",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--record-rollout-public-events",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--numpy-rollout-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reuse-rollout-workspaces",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--coalesce-rollout-agent-sets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--group-compatible-rollout-agent-sets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--coordinate-rollout-engines",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--drain-non-neural-rollout",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reuse-rollout-engines",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--past-policy-pool-size", type=int, default=2)
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
