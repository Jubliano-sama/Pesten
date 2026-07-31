#!/usr/bin/env python3
"""Paired three-seed speed profile for temporal Puffer league training.

Each repeat uses the same initialized model, frozen historical snapshot,
champion, deal seeds, action-sampling seeds, fixed 40/20/20/10/10 league mix,
PPO objective, drift diagnostics, and checkpoint round-trip.  The only
difference is scalar versus batched rollout/replay execution.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import tempfile
import time
from typing import Callable

import numpy as np
import pufferlib
import torch

from benchmark_agents import load_ppo_agent
from temporal_batched_speed import (
    BatchRolloutStats,
    LEAGUE_SCHEDULE,
    collect_league_batch_batched,
)
from temporal_league import (
    CHAMPION_PATH,
    CHAMPION_SHA256,
    _snapshot,
    collect_league_batch,
    verify_champion,
)
from temporal_puffer_v4 import (
    NativePufferTemporalPolicy,
    PUFFERLIB_REVISION,
)
from temporal_training import (
    evaluate_ppo_drift,
    evaluate_ppo_drift_batched,
    ppo_update,
    ppo_update_batched,
    save_safe_checkpoint,
)


DEFAULT_SEEDS = (731, 1731, 2731)
METRIC_TOLERANCE = 2e-5
DRIFT_TOLERANCE = 5e-4
# Explained variance divides two separately reduced variances and is more
# sensitive than loss/KL to the native batch reduction order.  A 0.2% absolute
# bound is still materially strict while avoiding false failures on long games.
EXPLAINED_VARIANCE_TOLERANCE = 2e-3
# Adam's first normalized step can turn an otherwise negligible reduction-
# order gradient delta around zero into opposite +/-lr parameter steps.  The
# strict bound is therefore just above 2*7e-4; loss/KL stay on the tighter
# tolerance above and every serialized tensor must still round-trip exactly.
CHECKPOINT_ABS_TOLERANCE = 1.5e-3


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, function: Callable):
    _synchronize(device)
    started = time.perf_counter()
    result = function()
    _synchronize(device)
    return result, time.perf_counter() - started


def _git_output(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ("git", *arguments),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clone_policy(
    source: NativePufferTemporalPolicy,
    device: torch.device,
) -> NativePufferTemporalPolicy:
    clone = NativePufferTemporalPolicy(
        hidden_size=source.hidden_size,
        num_layers=source.num_layers,
    ).to(device)
    clone.load_state_dict(copy.deepcopy(source.state_dict()))
    return clone


def _current_rss_mib() -> float:
    # Linux reports KiB.  macOS reports bytes, but this benchmark targets RTX
    # Linux and records its platform explicitly.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _memory_snapshot(device: torch.device) -> dict[str, float]:
    result = {"rss_peak_mib": _current_rss_mib()}
    if device.type == "cuda":
        result.update(
            {
                "cuda_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated(device) / 2**20
                ),
                "cuda_peak_reserved_mib": (
                    torch.cuda.max_memory_reserved(device) / 2**20
                ),
            }
        )
    return result


def _reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _episode_decisions(episodes) -> int:
    return sum(len(trajectory) for trajectory, _ in episodes)


def _compare_rollouts(scalar_episodes, batched_episodes) -> dict:
    if len(scalar_episodes) != len(batched_episodes):
        raise AssertionError("trajectory count differs")
    max_log_probability_delta = 0.0
    max_value_delta = 0.0
    decisions = 0
    for scalar_episode, batched_episode in zip(
        scalar_episodes, batched_episodes
    ):
        scalar_trajectory, scalar_won = scalar_episode
        batched_trajectory, batched_won = batched_episode
        if scalar_won != batched_won:
            raise AssertionError("episode outcome differs")
        if len(scalar_trajectory) != len(batched_trajectory):
            raise AssertionError("episode decision count differs")
        for scalar_step, batched_step in zip(
            scalar_trajectory, batched_trajectory
        ):
            if scalar_step.decision != batched_step.decision:
                raise AssertionError(
                    "safe observation/event/action-mask stream differs"
                )
            if scalar_step.action != batched_step.action:
                raise AssertionError("sampled action differs")
            max_log_probability_delta = max(
                max_log_probability_delta,
                abs(
                    scalar_step.old_log_prob
                    - batched_step.old_log_prob
                ),
            )
            max_value_delta = max(
                max_value_delta,
                abs(scalar_step.value - batched_step.value),
            )
            decisions += 1
    return {
        "exact_trajectories": True,
        "exact_outcomes": True,
        "exact_decisions": True,
        "exact_actions": True,
        "max_old_log_probability_abs_delta": (
            max_log_probability_delta
        ),
        "max_behavior_value_abs_delta": max_value_delta,
        "decisions_compared": decisions,
    }


def _compare_metrics(
    reference: dict,
    candidate: dict,
    *,
    tolerance: float = METRIC_TOLERANCE,
) -> dict:
    if reference.keys() != candidate.keys():
        raise AssertionError("metric keys differ")
    deltas = {}
    tolerances = {}
    for name, reference_value in reference.items():
        candidate_value = candidate[name]
        metric_tolerance = (
            max(tolerance, EXPLAINED_VARIANCE_TOLERANCE)
            if name == "explained_variance"
            else tolerance
        )
        tolerances[name] = metric_tolerance
        if isinstance(reference_value, int):
            if candidate_value != reference_value:
                raise AssertionError(f"{name} differs")
            deltas[name] = 0
            continue
        if math.isnan(reference_value) and math.isnan(candidate_value):
            deltas[name] = 0.0
            continue
        delta = abs(reference_value - candidate_value)
        deltas[name] = delta
        if delta > metric_tolerance:
            raise AssertionError(
                f"{name} delta {delta} exceeds {metric_tolerance}"
            )
    return {
        "tolerance": tolerance,
        "per_metric_tolerance": tolerances,
        "max_abs_delta": max(deltas.values(), default=0.0),
        "abs_deltas": deltas,
    }


def _state_delta(reference: torch.nn.Module, candidate: torch.nn.Module) -> dict:
    reference_state = reference.state_dict()
    candidate_state = candidate.state_dict()
    if reference_state.keys() != candidate_state.keys():
        raise AssertionError("checkpoint parameter keys differ")
    max_absolute = 0.0
    max_relative = 0.0
    squared_difference = 0.0
    absolute_difference = 0.0
    element_count = 0
    elements_over_metric_tolerance = 0
    for name, reference_value in reference_state.items():
        candidate_value = candidate_state[name]
        difference = (reference_value - candidate_value).abs()
        max_absolute = max(max_absolute, float(difference.max().item()))
        denominator = reference_value.abs().clamp_min(1e-12)
        max_relative = max(
            max_relative,
            float((difference / denominator).max().item()),
        )
        squared_difference += float(difference.square().sum().item())
        absolute_difference += float(difference.sum().item())
        element_count += difference.numel()
        elements_over_metric_tolerance += int(
            (difference > METRIC_TOLERANCE).sum().item()
        )
    if max_absolute > CHECKPOINT_ABS_TOLERANCE:
        raise AssertionError(
            f"updated checkpoint delta {max_absolute} exceeds tolerance"
        )
    return {
        "same_keys": True,
        "max_parameter_abs_delta": max_absolute,
        "max_parameter_relative_delta": max_relative,
        "mean_parameter_abs_delta": (
            absolute_difference / element_count
        ),
        "rms_parameter_delta": math.sqrt(
            squared_difference / element_count
        ),
        "fraction_over_loss_metric_tolerance": (
            elements_over_metric_tolerance / element_count
        ),
        "absolute_tolerance": CHECKPOINT_ABS_TOLERANCE,
    }


def _checkpoint_round_trip(
    policy: NativePufferTemporalPolicy,
    *,
    label: str,
    directory: Path,
    device: torch.device,
) -> dict:
    path = directory / f"{label}.pt"
    _, elapsed = _timed(
        device, lambda: save_safe_checkpoint(path, policy)
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    saved_state = payload["state_dict"]
    current_state = policy.state_dict()
    if current_state.keys() != saved_state.keys():
        raise AssertionError("checkpoint round-trip changed keys")
    for name, value in current_state.items():
        if not torch.equal(value.detach().cpu(), saved_state[name]):
            raise AssertionError(
                f"checkpoint round-trip changed tensor {name}"
            )
    return {
        "seconds": elapsed,
        "bytes": path.stat().st_size,
        "sha256": _file_sha256(path),
        "format": payload["format"],
        "exact_round_trip": True,
    }


def _phase_rates(
    *,
    rollout_seconds: float,
    update_seconds: float,
    drift_seconds: float,
    checkpoint_seconds: float,
    turns: int,
    decisions: int,
    games: int,
) -> dict:
    total = (
        rollout_seconds
        + update_seconds
        + drift_seconds
        + checkpoint_seconds
    )
    return {
        "rollout_turns_per_second": turns / rollout_seconds,
        "rollout_decisions_per_second": decisions / rollout_seconds,
        "rollout_games_per_second": games / rollout_seconds,
        "update_decisions_per_second": decisions / update_seconds,
        "end_to_end_decisions_per_second": decisions / total,
        "total_seconds": total,
    }


def _compile_sequence_forward(
    policy: NativePufferTemporalPolicy,
) -> None:
    policy.forward_native_sequence = torch.compile(
        policy.forward_native_sequence,
        dynamic=True,
        mode="reduce-overhead",
    )


def _run_seed(
    seed: int,
    *,
    games: int,
    device: torch.device,
    compile_updates: bool,
    checkpoint_directory: Path,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    policy = NativePufferTemporalPolicy(
        hidden_size=128, num_layers=2
    ).to(device)
    policy.eval()
    historical = [_snapshot(policy)]
    champion = load_ppo_agent(CHAMPION_PATH, "cpu")
    league_seed = seed + 100_000

    _reset_memory(device)
    (scalar_episodes, scalar_contribution), scalar_rollout_seconds = _timed(
        device,
        lambda: collect_league_batch(
            policy,
            champion,
            historical,
            games=games,
            seed=league_seed,
            schedule_offset=0,
        ),
    )
    scalar_rollout_memory = _memory_snapshot(device)

    batch_stats = BatchRolloutStats()
    _reset_memory(device)
    (batched_episodes, batched_contribution), batched_rollout_seconds = _timed(
        device,
        lambda: collect_league_batch_batched(
            policy,
            champion,
            historical,
            games=games,
            seed=league_seed,
            schedule_offset=0,
            schedule=LEAGUE_SCHEDULE,
            stats=batch_stats,
        ),
    )
    batched_rollout_memory = _memory_snapshot(device)
    if scalar_contribution != batched_contribution:
        raise AssertionError("league contribution metadata differs")
    rollout_parity = _compare_rollouts(
        scalar_episodes, batched_episodes
    )
    decisions = _episode_decisions(scalar_episodes)

    scalar_policy = _clone_policy(policy, device)
    scalar_optimizer = torch.optim.Adam(
        scalar_policy.parameters(), lr=7e-4
    )
    _reset_memory(device)
    scalar_metrics, scalar_update_seconds = _timed(
        device,
        lambda: ppo_update(
            scalar_policy, scalar_episodes, scalar_optimizer
        ),
    )
    scalar_drift, scalar_drift_seconds = _timed(
        device,
        lambda: evaluate_ppo_drift(
            scalar_policy, scalar_episodes
        ),
    )
    scalar_checkpoint = _checkpoint_round_trip(
        scalar_policy,
        label=f"seed{seed}_scalar",
        directory=checkpoint_directory,
        device=device,
    )
    scalar_memory = _memory_snapshot(device)

    batched_policy = _clone_policy(policy, device)
    if compile_updates:
        _compile_sequence_forward(batched_policy)
    batched_optimizer = torch.optim.Adam(
        batched_policy.parameters(), lr=7e-4
    )
    _reset_memory(device)
    cold_metrics, cold_update_seconds = _timed(
        device,
        lambda: ppo_update_batched(
            batched_policy, batched_episodes, batched_optimizer
        ),
    )
    cold_drift, cold_drift_seconds = _timed(
        device,
        lambda: evaluate_ppo_drift_batched(
            batched_policy, batched_episodes
        ),
    )
    cold_checkpoint = _checkpoint_round_trip(
        batched_policy,
        label=f"seed{seed}_batched_cold",
        directory=checkpoint_directory,
        device=device,
    )
    cold_memory = _memory_snapshot(device)
    cold_metric_parity = _compare_metrics(
        scalar_metrics, cold_metrics
    )
    cold_drift_parity = _compare_metrics(
        scalar_drift, cold_drift, tolerance=DRIFT_TOLERANCE
    )
    cold_state_parity = _state_delta(
        scalar_policy, batched_policy
    )

    # Reuse the compiled graph but restore the exact pre-update checkpoint and
    # an empty Adam state, producing a like-for-like steady-state measurement.
    batched_policy.load_state_dict(copy.deepcopy(policy.state_dict()))
    batched_optimizer = torch.optim.Adam(
        batched_policy.parameters(), lr=7e-4
    )
    _reset_memory(device)
    warm_metrics, warm_update_seconds = _timed(
        device,
        lambda: ppo_update_batched(
            batched_policy, batched_episodes, batched_optimizer
        ),
    )
    warm_drift, warm_drift_seconds = _timed(
        device,
        lambda: evaluate_ppo_drift_batched(
            batched_policy, batched_episodes
        ),
    )
    warm_checkpoint = _checkpoint_round_trip(
        batched_policy,
        label=f"seed{seed}_batched_warm",
        directory=checkpoint_directory,
        device=device,
    )
    warm_memory = _memory_snapshot(device)
    warm_metric_parity = _compare_metrics(
        scalar_metrics, warm_metrics
    )
    warm_drift_parity = _compare_metrics(
        scalar_drift, warm_drift, tolerance=DRIFT_TOLERANCE
    )
    warm_state_parity = _state_delta(
        scalar_policy, batched_policy
    )

    scalar_rates = _phase_rates(
        rollout_seconds=scalar_rollout_seconds,
        update_seconds=scalar_update_seconds,
        drift_seconds=scalar_drift_seconds,
        checkpoint_seconds=scalar_checkpoint["seconds"],
        turns=scalar_contribution["total_turns"],
        decisions=decisions,
        games=games,
    )
    cold_rates = _phase_rates(
        rollout_seconds=batched_rollout_seconds,
        update_seconds=cold_update_seconds,
        drift_seconds=cold_drift_seconds,
        checkpoint_seconds=cold_checkpoint["seconds"],
        turns=batched_contribution["total_turns"],
        decisions=decisions,
        games=games,
    )
    warm_rates = _phase_rates(
        rollout_seconds=batched_rollout_seconds,
        update_seconds=warm_update_seconds,
        drift_seconds=warm_drift_seconds,
        checkpoint_seconds=warm_checkpoint["seconds"],
        turns=batched_contribution["total_turns"],
        decisions=decisions,
        games=games,
    )
    attributed_vector_seconds = sum(
        (
            batch_stats.safe_observation_event_seconds,
            batch_stats.temporal_feature_encoding_seconds,
            batch_stats.temporal_recurrent_inference_transfer_seconds,
            batch_stats.temporal_sampling_state_scatter_seconds,
            batch_stats.champion_inference_transfer_seconds,
            batch_stats.rules_stepping_seconds,
        )
    )
    phase_profile = {
        "before_scalar": {
            "environment_rules_safe_encoding_recurrent_inference_inclusive": (
                scalar_rollout_seconds
            ),
            "loss_backward_optimizer_inclusive": scalar_update_seconds,
            "evaluation_drift": scalar_drift_seconds,
            "checkpoint": scalar_checkpoint["seconds"],
        },
        "after_batched": {
            "environment_rules_stepping": (
                batch_stats.rules_stepping_seconds
            ),
            "safe_observation_event_encoding": (
                batch_stats.safe_observation_event_seconds
            ),
            "temporal_token_feature_encoding": (
                batch_stats.temporal_feature_encoding_seconds
            ),
            "temporal_recurrent_inference_cpu_gpu_transfer": (
                batch_stats.temporal_recurrent_inference_transfer_seconds
            ),
            "temporal_sampling_state_scatter": (
                batch_stats.temporal_sampling_state_scatter_seconds
            ),
            "ppo_opponent_inference_cpu_device_transfer": (
                batch_stats.champion_inference_transfer_seconds
            ),
            "bayesian_tree_scalar_environment_and_search": (
                batch_stats.tree_seconds
            ),
            "vector_rollout_unattributed_orchestration": max(
                0.0,
                batch_stats.vector_seconds - attributed_vector_seconds,
            ),
            "loss_backward_optimizer_cold_inclusive": (
                cold_update_seconds
            ),
            "loss_backward_optimizer_warm_inclusive": (
                warm_update_seconds
            ),
            "evaluation_drift_cold": cold_drift_seconds,
            "evaluation_drift_warm": warm_drift_seconds,
            "checkpoint_cold": cold_checkpoint["seconds"],
            "checkpoint_warm": warm_checkpoint["seconds"],
        },
        "counter_semantics": {
            "inference_transfer": (
                "wall time through the synchronizing device-to-CPU action "
                "probability or argmax transfer; includes recurrent kernels, "
                "token/state host-to-device construction, and transfer"
            ),
            "loss_backward_optimizer": (
                "inclusive native replay token packing, forward, PPO loss, "
                "backward, clipping, and Adam step"
            ),
        },
    }
    return {
        "seed": seed,
        "league_seed": league_seed,
        "games": games,
        "turns": scalar_contribution["total_turns"],
        "training_decisions": decisions,
        "contribution": scalar_contribution,
        "rollout_parity": rollout_parity,
        "before_scalar": {
            "phase_seconds": {
                "rollout": scalar_rollout_seconds,
                "ppo_update": scalar_update_seconds,
                "post_update_drift": scalar_drift_seconds,
                "checkpoint": scalar_checkpoint["seconds"],
            },
            "rates": scalar_rates,
            "metrics": scalar_metrics,
            "drift": scalar_drift,
            "checkpoint": scalar_checkpoint,
            "memory": {
                "rollout": scalar_rollout_memory,
                "training": scalar_memory,
            },
        },
        "after_batched_cold": {
            "phase_seconds": {
                "rollout": batched_rollout_seconds,
                "ppo_update": cold_update_seconds,
                "post_update_drift": cold_drift_seconds,
                "checkpoint": cold_checkpoint["seconds"],
            },
            "rates": cold_rates,
            "metrics": cold_metrics,
            "drift": cold_drift,
            "checkpoint": cold_checkpoint,
            "memory": {
                "rollout": batched_rollout_memory,
                "training": cold_memory,
            },
            "metric_parity": cold_metric_parity,
            "drift_parity": cold_drift_parity,
            "checkpoint_parity": cold_state_parity,
        },
        "after_batched_warm": {
            "phase_seconds": {
                "rollout": batched_rollout_seconds,
                "ppo_update": warm_update_seconds,
                "post_update_drift": warm_drift_seconds,
                "checkpoint": warm_checkpoint["seconds"],
            },
            "rates": warm_rates,
            "metrics": warm_metrics,
            "drift": warm_drift,
            "checkpoint": warm_checkpoint,
            "memory": {
                "rollout": batched_rollout_memory,
                "training": warm_memory,
            },
            "metric_parity": warm_metric_parity,
            "drift_parity": warm_drift_parity,
            "checkpoint_parity": warm_state_parity,
        },
        "batch_rollout_stats": {
            name: getattr(batch_stats, name)
            for name in batch_stats.__dataclass_fields__
        },
        "phase_profile": phase_profile,
        "speedups": {
            "rollout": (
                scalar_rollout_seconds / batched_rollout_seconds
            ),
            "cold_update": (
                scalar_update_seconds / cold_update_seconds
            ),
            "warm_update": (
                scalar_update_seconds / warm_update_seconds
            ),
            "cold_drift": (
                scalar_drift_seconds / cold_drift_seconds
            ),
            "warm_drift": (
                scalar_drift_seconds / warm_drift_seconds
            ),
            "cold_total": (
                scalar_rates["total_seconds"]
                / cold_rates["total_seconds"]
            ),
            "warm_total": (
                scalar_rates["total_seconds"]
                / warm_rates["total_seconds"]
            ),
            "warm_end_to_end_decisions": (
                warm_rates["end_to_end_decisions_per_second"]
                / scalar_rates["end_to_end_decisions_per_second"]
            ),
        },
    }


def _median(runs: list[dict], path: tuple[str, ...]) -> float:
    values = []
    for run in runs:
        value = run
        for key in path:
            value = value[key]
        values.append(value)
    return statistics.median(values)


def _aggregate(runs: list[dict]) -> dict:
    paths = {
        "scalar_total_seconds": (
            "before_scalar", "rates", "total_seconds"
        ),
        "batched_cold_total_seconds": (
            "after_batched_cold", "rates", "total_seconds"
        ),
        "batched_warm_total_seconds": (
            "after_batched_warm", "rates", "total_seconds"
        ),
        "scalar_rollout_decisions_per_second": (
            "before_scalar", "rates", "rollout_decisions_per_second"
        ),
        "batched_rollout_decisions_per_second": (
            "after_batched_warm",
            "rates",
            "rollout_decisions_per_second",
        ),
        "scalar_end_to_end_decisions_per_second": (
            "before_scalar",
            "rates",
            "end_to_end_decisions_per_second",
        ),
        "batched_warm_end_to_end_decisions_per_second": (
            "after_batched_warm",
            "rates",
            "end_to_end_decisions_per_second",
        ),
        "rollout_speedup": ("speedups", "rollout"),
        "cold_update_speedup": ("speedups", "cold_update"),
        "warm_update_speedup": ("speedups", "warm_update"),
        "warm_drift_speedup": ("speedups", "warm_drift"),
        "cold_total_speedup": ("speedups", "cold_total"),
        "warm_total_speedup": ("speedups", "warm_total"),
    }
    return {
        f"median_{name}": _median(runs, path)
        for name, path in paths.items()
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="must be a multiple of 10 to preserve the exact league mixture",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda"
    )
    parser.add_argument(
        "--no-compile-updates",
        action="store_true",
        help="disable torch.compile around native sequence replay",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "benchmarks/temporal/"
            "temporal_batched_speed_3seed_20260731.json"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed repeats from the adjacent partial artifact",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.games <= 0 or args.games % 10:
        raise SystemExit("--games must be a positive multiple of 10")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    seeds = tuple(int(value) for value in args.seeds.split(","))
    if len(seeds) != 3:
        raise SystemExit("the published profile requires exactly three seeds")

    champion = verify_champion()
    compile_updates = not args.no_compile_updates
    partial_path = args.output.with_name(
        args.output.stem + ".partial.json"
    )
    runs = []
    if args.resume and partial_path.is_file():
        partial_payload = json.loads(
            partial_path.read_text(encoding="utf-8")
        )
        runs = partial_payload["runs"]
    completed_seeds = {run["seed"] for run in runs}
    with tempfile.TemporaryDirectory(
        prefix="pesten-temporal-speed-checkpoints-"
    ) as checkpoint_directory_name:
        checkpoint_directory = Path(checkpoint_directory_name)
        for seed in seeds:
            if seed in completed_seeds:
                continue
            if compile_updates:
                torch.compiler.reset()
                cache_directory = (
                    checkpoint_directory / f"inductor-seed-{seed}"
                )
                cache_directory.mkdir()
                os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(
                    cache_directory
                )
            run = _run_seed(
                seed,
                games=args.games,
                device=device,
                compile_updates=compile_updates,
                checkpoint_directory=checkpoint_directory,
            )
            runs.append(run)
            partial_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "complete": False,
                        "runs": runs,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "seed": seed,
                        "speedups": run["speedups"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pufferlib": str(pufferlib.__version__),
        "pufferlib_revision": PUFFERLIB_REVISION,
        "device": str(device),
        "compile_updates": compile_updates,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_status_short": _git_output("status", "--short"),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        runtime["cuda"] = {
            "name": properties.name,
            "total_memory_mib": properties.total_memory / 2**20,
            "compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),
        }
    payload = {
        "schema_version": 1,
        "benchmark": "temporal_batched_league_epoch",
        "runtime": runtime,
        "workload": {
            "seeds": list(seeds),
            "repeats": len(seeds),
            "games_per_repeat": args.games,
            "league_epochs_per_repeat": 1,
            "league_mix": {
                "current_selfplay": 0.4,
                "historical_temporal": 0.2,
                "ppo_champion": 0.2,
                "bayesian_tree": 0.1,
                "smart_rule": 0.1,
            },
            "policy": {
                "class": "NativePufferTemporalPolicy",
                "hidden_size": 128,
                "num_layers": 2,
                "initialization": "torch manual seed for each repeat",
            },
            "optimizer": {
                "class": "Adam",
                "learning_rate": 7e-4,
            },
            "champion_path": str(CHAMPION_PATH),
            "champion_sha256": CHAMPION_SHA256,
            "champion_verified": champion,
            "timed_phases": [
                "league rollout",
                "one PPO update",
                "post-update PPO drift diagnostics",
                "safe checkpoint save and exact round-trip verification",
            ],
            "bayesian_tree_execution": (
                "canonical scalar fallback in both variants"
            ),
        },
        "parity_contract": {
            "exact": [
                "deal seeds",
                "sampling seeds",
                "opponent schedule",
                "game outcomes",
                "safe observations",
                "public event ordering",
                "legal action masks",
                "sampled actions",
                "training decision counts",
                "checkpoint keys and round trips",
            ],
            "numeric_tolerance": METRIC_TOLERANCE,
            "default_drift_metric_tolerance": DRIFT_TOLERANCE,
            "explained_variance_tolerance": (
                EXPLAINED_VARIANCE_TOLERANCE
            ),
            "numeric": [
                "behavior log probabilities",
                "behavior values",
                "PPO loss components",
                "KL diagnostics",
                "updated checkpoint parameters",
            ],
        },
        "phase_profile_contract": {
            "required_labels_present": [
                "environment and rules stepping",
                "safe observation and event encoding",
                "temporal recurrent inference",
                "PPO opponent inference",
                "CPU/GPU transfer and recurrent-state scatter",
                "loss, backward, and optimizer",
                "checkpoint",
                "evaluation and drift",
            ],
            "scalar_rollout_subphases": (
                "reported as one inclusive baseline because instrumenting the "
                "legacy per-agent call stack would change the frozen baseline"
            ),
        },
        "runs": runs,
        "aggregate": _aggregate(runs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
