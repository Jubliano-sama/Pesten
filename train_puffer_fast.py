#!/usr/bin/env python3
"""High-throughput, restartable native Puffer PPO training."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch


AUTO_INT8_ROLLOUT_PARAMETER_THRESHOLD = 1_000_000
INT8_RESIDUAL_BASE_WORKER_CAP = 8


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
        if hasattr(trajectory, "legal_masks"):
            masks = trajectory.legal_masks
            actions = trajectory.actions
            decisions += len(trajectory)
            strategic = masks[:, pass_action] & masks[:, :pass_action].any(
                axis=1
            )
            opportunities += int(strategic.sum())
            passes += int(((actions == pass_action) & strategic).sum())
            total_passes += int((actions == pass_action).sum())
            continue
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


def _prune_frozen_optimizer_state(optimizer) -> dict[str, int]:
    """Drop stale Adam moments for parameters excluded from backprop."""
    removed_parameter_tensors = 0
    removed_parameters = 0
    removed_state_tensors = 0
    removed_state_bytes = 0
    for parameter, state_values in tuple(optimizer.state.items()):
        if parameter.requires_grad:
            continue
        optimizer.state.pop(parameter)
        removed_parameter_tensors += 1
        removed_parameters += parameter.numel()
        for value in state_values.values():
            if isinstance(value, torch.Tensor):
                removed_state_tensors += 1
                removed_state_bytes += value.numel() * value.element_size()
    return {
        "removed_parameter_tensors": removed_parameter_tensors,
        "removed_parameters": removed_parameters,
        "removed_state_tensors": removed_state_tensors,
        "removed_state_bytes": removed_state_bytes,
    }


def _last_jsonl_row(path: Path) -> dict | None:
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
        return json.loads(last_line)
    return None


def _append_jsonl_idempotent(path: Path, row: dict) -> None:
    previous = _last_jsonl_row(path)
    if previous is not None:
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


def _effective_vector_workers(
    requested_workers: int,
    schedule: tuple[str, ...],
    player_counts: tuple[int, ...],
    games: int | None = None,
    rollout_precision: str = "float32",
) -> int:
    """Use the requested pool now that large vector cohorts can be sharded."""
    del schedule, player_counts
    effective = (
        min(requested_workers, games)
        if games is not None
        else requested_workers
    )
    if rollout_precision == "int8_residual_base":
        effective = min(effective, INT8_RESIDUAL_BASE_WORKER_CAP)
    return effective


def _resolve_rollout_precision(
    requested: str,
    parameter_count: int,
    architecture: str | None = None,
) -> str:
    """Use int8 behavior inference automatically for genuinely large models."""
    if requested not in (
        "auto",
        "float32",
        "int8_recurrent",
        "int8_residual_base",
    ):
        raise ValueError(f"unsupported rollout precision: {requested}")
    if requested != "auto":
        return requested
    if (
        parameter_count >= AUTO_INT8_ROLLOUT_PARAMETER_THRESHOLD
        and architecture == "min_gru_residual"
    ):
        return "int8_residual_base"
    return (
        "int8_recurrent"
        if parameter_count >= AUTO_INT8_ROLLOUT_PARAMETER_THRESHOLD
        else "float32"
    )


def _resolve_rollout_max_public_events(
    requested: int | None,
    rollout_precision: str,
) -> int | None:
    """Bound behavior history for the throughput-first residual actor."""
    if requested is not None:
        if requested < 0:
            raise ValueError("rollout max public events cannot be negative")
        return requested or None
    return 5 if rollout_precision == "int8_residual_base" else None


def _configure_trainable_policy(
    policy,
    architecture: str,
    requested_adapter_layers: int | None,
) -> dict[str, float | int | str | None]:
    """Restrict residual backprop while retaining the complete forward model."""
    total = sum(parameter.numel() for parameter in policy.parameters())
    if architecture != "min_gru_residual":
        if requested_adapter_layers not in (None, -1):
            raise ValueError(
                "trainable adapter layers require min_gru_residual"
            )
        for parameter in policy.parameters():
            parameter.requires_grad_(True)
        return {
            "mode": "all",
            "adapter_layers": None,
            "trainable_parameters": total,
            "total_parameters": total,
            "trainable_fraction": 1.0,
        }

    core = policy.puffer_policy
    base = getattr(core, "base", None)
    adapter = getattr(core, "adapter", None)
    layers = getattr(getattr(adapter, "network", None), "layers", None)
    if base is None or adapter is None or layers is None:
        raise ValueError("residual policy does not expose adapter layers")
    layer_count = len(layers)
    selected = (
        min(2, layer_count)
        if requested_adapter_layers is None
        else requested_adapter_layers
    )
    if selected < -1 or selected > layer_count:
        raise ValueError(
            f"trainable adapter layers must be in [-1, {layer_count}]"
        )
    for parameter in policy.parameters():
        parameter.requires_grad_(selected == -1)
    if selected != -1:
        for layer in layers[layer_count - selected :]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)
        for parameter in adapter.decoder.parameters():
            parameter.requires_grad_(True)
    trainable = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )
    return {
        "mode": "all" if selected == -1 else "top_adapter_layers_and_heads",
        "adapter_layers": None if selected == -1 else selected,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
    }


def _apply_residual_base_freeze(
    policy,
    architecture: str,
    requested: bool,
    training_scope: dict[str, float | int | str | None],
) -> dict[str, float | int | str | bool | None]:
    """Preserve the explicit adapter-only compatibility control."""
    if not requested:
        return training_scope
    if architecture not in {"min_gru_residual", "hawk_residual"}:
        raise ValueError(
            "--freeze-residual-base requires a residual temporal policy"
        )
    for parameter in policy.puffer_policy.base.parameters():
        parameter.requires_grad_(False)
    trainable = sum(
        parameter.numel()
        for parameter in policy.parameters()
        if parameter.requires_grad
    )
    total = int(training_scope["total_parameters"])
    return {
        **training_scope,
        "mode": (
            "residual_adapter_only"
            if training_scope["mode"] == "all"
            else training_scope["mode"]
        ),
        "freeze_residual_base": True,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total,
    }


def _resolve_update_precision(
    requested: str,
    *,
    device_type: str,
    bf16_supported: bool,
) -> str:
    """Prefer native BF16 tensor cores while retaining an FP32 escape hatch."""
    if requested not in ("auto", "float32", "bfloat16"):
        raise ValueError(f"unsupported update precision: {requested}")
    if requested == "float32":
        return requested
    if requested == "bfloat16":
        if device_type != "cuda" or not bf16_supported:
            raise ValueError("bfloat16 updates require supported CUDA hardware")
        return requested
    return (
        "bfloat16"
        if device_type == "cuda" and bf16_supported
        else "float32"
    )


def _update_autocast(device_type: str, update_precision: str):
    if update_precision == "bfloat16":
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return nullcontext()


def _resolve_compile_updates(
    requested: bool | None,
    *,
    pipelined_rollouts: bool,
    parameter_count: int,
) -> bool:
    if requested is not None:
        return requested
    # Compiled updates are faster in isolation, but an explicitly requested
    # large-model pipeline can overlap most eager BF16 work with CPU rollout.
    # Avoid several minutes of multi-shape compilation unless the caller opts
    # into it explicitly.
    return not (
        pipelined_rollouts
        and parameter_count >= AUTO_INT8_ROLLOUT_PARAMETER_THRESHOLD
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--champion",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
        help="frozen feed-forward MLP opponent checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--games-per-epoch", type=int, default=2048)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=1000,
        help="training-only per-game turn cap; evaluation keeps its own contract",
    )
    parser.add_argument("--rollout-workers", type=int, default=10)
    parser.add_argument("--chunks-per-worker", type=int, default=3)
    parser.add_argument(
        "--rollout-engine",
        choices=("vector", "process"),
        default="vector",
    )
    parser.add_argument(
        "--rollout-precision",
        choices=(
            "auto",
            "float32",
            "int8_recurrent",
            "int8_residual_base",
        ),
        default="auto",
        help=(
            "behavior-policy precision; auto uses weight-only recurrent int8 "
            "for large models and the preserved int8 base actor for large "
            "MinGRU residual policies; learner precision is independent"
        ),
    )
    parser.add_argument(
        "--trace-int8-rollouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "trace quantized MinGRU decision/event graphs to remove Python "
            "dispatch while preserving exact actor outputs"
        ),
    )
    parser.add_argument(
        "--rollout-max-public-events",
        type=int,
        default=None,
        help=(
            "maximum recent events consumed by the behavior actor; auto uses "
            "five for int8_residual_base, while 0 restores full history"
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument(
        "--fresh-optimizer",
        action="store_true",
        help="ignore any checkpoint optimizer state and start a fresh Adam",
    )
    parser.add_argument(
        "--trainable-adapter-layers",
        type=int,
        default=None,
        help=(
            "number of final residual-adapter GRU layers to backpropagate; "
            "auto uses two, 0 trains heads only, and -1 trains everything"
        ),
    )
    parser.add_argument(
        "--freeze-residual-base",
        action="store_true",
        help=(
            "freeze the exact base branch of a residual temporal policy; "
            "combine with --trainable-adapter-layers -1 to train its entire "
            "added branch"
        ),
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=1,
        help=(
            "PPO replay passes per rollout; the throughput-biased default "
            "trades some sample reuse for much shorter updates"
        ),
    )
    parser.add_argument("--minibatch-trajectories", type=int, default=128)
    parser.add_argument(
        "--diagnostic-minibatch-trajectories", type=int, default=256
    )
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.02)
    parser.add_argument(
        "--early-stop-diagnostic-trajectories", type=int, default=512
    )
    parser.add_argument(
        "--full-rollout-diagnostics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "rescore the complete rollout after every update; the faster "
            "default samples 512 trajectories and expands only near target KL"
        ),
    )
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--evaluation-pairs", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026081202)
    parser.add_argument(
        "--campaign-offset",
        type=int,
        help=(
            "optional zero-based phase offset for common rollout, update, "
            "and evaluation randomness across differently aged checkpoints"
        ),
    )
    parser.add_argument(
        "--schedule",
        help=(
            "optional comma-separated opponent cycle overriding the "
            "checkpoint recipe"
        ),
    )
    parser.add_argument(
        "--player-count-schedule",
        help=(
            "optional comma-separated 2-8 player cycle overriding the "
            "checkpoint recipe"
        ),
    )
    parser.add_argument(
        "--historical-checkpoint",
        action="append",
        type=Path,
        help=(
            "frozen temporal opponent checkpoint; repeat to form a bounded "
            "historical pool"
        ),
    )
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--compile-updates",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "compile recurrent CUDA updates; auto skips compilation when a "
            "large-model rollout pipeline already hides eager update time"
        ),
    )
    parser.add_argument(
        "--update-precision",
        choices=("auto", "float32", "bfloat16"),
        default="auto",
        help=(
            "PPO forward/backward precision; auto selects native CUDA BF16 "
            "while retaining FP32 parameters and optimizer state"
        ),
    )
    parser.add_argument(
        "--persistent-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pipelined-rollouts",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "overlap the next CPU vector rollout with the current CUDA update; "
            "opt-in for short eager runs because sustained compiled training "
            "is faster and avoids one-update policy lag"
        ),
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high"),
        default="high",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive = (
        args.epochs,
        args.games_per_epoch,
        args.max_turns,
        args.rollout_workers,
        args.chunks_per_worker,
        args.update_epochs,
        args.minibatch_trajectories,
        args.diagnostic_minibatch_trajectories,
        args.checkpoint_every,
        args.evaluate_every,
        args.evaluation_pairs,
        args.early_stop_diagnostic_trajectories,
    )
    if any(value < 1 for value in positive):
        raise ValueError("count arguments must be positive")
    if args.campaign_offset is not None and args.campaign_offset < 0:
        raise ValueError("campaign offset must be nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("fast Puffer training requested unavailable CUDA")
    pipelined_rollouts = args.pipelined_rollouts
    if pipelined_rollouts and args.rollout_engine != "vector":
        raise ValueError("pipelined rollouts require --rollout-engine vector")
    resume = args.resume.resolve()
    if not resume.is_file():
        raise FileNotFoundError(resume)
    champion_path = args.champion.resolve()
    if not champion_path.is_file():
        raise FileNotFoundError(champion_path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    from benchmark_agents import load_ppo_agent
    from live_puffer_segment import _state_hash
    from parallel_puffer_rollout import collect_league_batch_parallel
    from randomAgent import AgentRandom
    from smartAgent import Agent as SmartAgent
    from temporal_league import _snapshot, evaluate_paired
    from temporal_observation import PASS_ACTION
    from temporal_puffer_agent import temporal_agent_factory
    from temporal_puffer_v4 import (
        load_temporal_policy_checkpoint,
        policy_model_config,
    )
    from temporal_training import ppo_update_batched_epochs
    from vector_puffer_rollout import VectorPufferRolloutPool

    policy, payload = load_temporal_policy_checkpoint(
        resume, device=args.device
    )
    model_config = policy_model_config(policy)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())
    if (
        args.rollout_engine != "vector"
        and args.rollout_precision in (
            "int8_recurrent",
            "int8_residual_base",
        )
    ):
        raise ValueError("int8 rollouts require --rollout-engine vector")
    rollout_precision = (
        _resolve_rollout_precision(
            args.rollout_precision,
            parameter_count,
            str(model_config.get("architecture", "min_gru")),
        )
        if args.rollout_engine == "vector"
        else "float32"
    )
    update_precision = _resolve_update_precision(
        args.update_precision,
        device_type=args.device,
        bf16_supported=(
            torch.cuda.is_bf16_supported() if args.device == "cuda" else False
        ),
    )
    rollout_max_public_events = _resolve_rollout_max_public_events(
        args.rollout_max_public_events,
        rollout_precision,
    )
    training_scope = _configure_trainable_policy(
        policy,
        str(model_config.get("architecture", "min_gru")),
        args.trainable_adapter_layers,
    )
    training_scope = _apply_residual_base_freeze(
        policy,
        str(model_config.get("architecture", "min_gru")),
        args.freeze_residual_base,
        training_scope,
    )
    compile_updates = _resolve_compile_updates(
        args.compile_updates,
        pipelined_rollouts=pipelined_rollouts,
        parameter_count=parameter_count,
    )
    torch.set_float32_matmul_precision(args.matmul_precision)
    if compile_updates:
        torch._dynamo.config.recompile_limit = 32
        policy.forward_native_sequence = torch.compile(
            policy.forward_native_sequence,
            dynamic=False,
            mode="reduce-overhead",
        )
    recipe = payload.get("recipe", {})
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(recipe.get("learning_rate", 3e-4))
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    if "optimizer_state_dict" in payload and not args.fresh_optimizer:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    optimizer_state_pruning = _prune_frozen_optimizer_state(optimizer)
    _move_optimizer(optimizer, next(policy.parameters()).device)

    schedule = tuple(
        value.strip()
        for value in (
            args.schedule.split(",")
            if args.schedule is not None
            else recipe.get("schedule", ("current_selfplay",))
        )
        if value.strip()
    )
    if not schedule:
        raise ValueError("league schedule cannot be empty")
    mixture = tuple(
        (name, schedule.count(name)) for name in dict.fromkeys(schedule)
    )
    player_counts = tuple(
        int(str(value).strip())
        for value in (
            args.player_count_schedule.split(",")
            if args.player_count_schedule is not None
            else recipe.get("player_count_schedule", (2,))
        )
        if str(value).strip()
    )
    if not player_counts or any(count < 2 or count > 8 for count in player_counts):
        raise ValueError("player-count schedule must contain values from 2 to 8")
    requested_historical = args.historical_checkpoint
    if requested_historical is None:
        requested_historical = []
        for entry in recipe.get("historical_checkpoints", ()):
            requested_historical.append(
                Path(entry["path"] if isinstance(entry, dict) else entry)
            )
    if "historical_temporal" in schedule and not requested_historical:
        requested_historical = [resume]
    historical_paths = []
    historical_seen = set()
    for raw_path in requested_historical:
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in historical_seen:
            historical_paths.append(path)
            historical_seen.add(path)
    historical_records = [
        {"path": str(path), "sha256": _file_sha256(path)}
        for path in historical_paths
    ]
    start_epoch = int(payload.get("epoch", 0)) + 1
    source_sha256 = _file_sha256(resume)
    champion_factory = load_ppo_agent(champion_path, "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    started = time.perf_counter()
    curve = []
    result_path = output_dir / "training_result.json"
    jsonl_path = output_dir / "training_curve.jsonl"
    previous_row = _last_jsonl_row(jsonl_path)
    if (
        previous_row is not None
        and int(previous_row.get("epoch", -1)) >= start_epoch
    ):
        raise RuntimeError(
            "output training curve already reaches epoch "
            f"{previous_row.get('epoch')}; resume from that epoch's checkpoint "
            "or choose a new output directory"
        )
    vector_rollout_pool = (
        VectorPufferRolloutPool(
            policy,
            workers=_effective_vector_workers(
                args.rollout_workers,
                schedule,
                player_counts,
                args.games_per_epoch,
                rollout_precision,
            ),
            champion_path=champion_path,
            historical_paths=historical_paths,
            rollout_precision=rollout_precision,
            trace_min_gru=args.trace_int8_rollouts,
            max_public_events=rollout_max_public_events,
        )
        if args.rollout_engine == "vector"
        else None
    )
    def campaign_phase_for_offset(offset: int) -> int:
        return (
            args.campaign_offset + offset
            if args.campaign_offset is not None
            else start_epoch + offset
        )

    def collect_phase(rollout_policy, campaign_phase: int):
        rollout_started = time.perf_counter()
        rollout_arguments = {
            "games": args.games_per_epoch,
            "seed": args.seed + campaign_phase * args.games_per_epoch,
            "schedule_offset": campaign_phase * args.games_per_epoch,
            "player_counts": player_counts,
            "max_turns": args.max_turns,
        }
        if args.rollout_engine == "vector":
            episodes, contribution = vector_rollout_pool.collect(
                rollout_policy,
                schedule=schedule,
                prepack_training=args.persistent_replay,
                **rollout_arguments,
            )
        else:
            episodes, contribution = collect_league_batch_parallel(
                rollout_policy,
                mixture=mixture,
                workers=args.rollout_workers,
                chunks_per_worker=args.chunks_per_worker,
                champion_path=champion_path,
                historical_paths=historical_paths,
                **rollout_arguments,
            )
        return (
            episodes,
            contribution,
            time.perf_counter() - rollout_started,
        )

    rollout_executor = (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="puffer-rollout")
        if pipelined_rollouts
        else None
    )
    pending_rollout = None
    for offset in range(args.epochs):
        epoch = start_epoch + offset
        campaign_phase = campaign_phase_for_offset(offset)
        epoch_started = time.perf_counter()
        rollout_wait_started = time.perf_counter()
        if pending_rollout is not None:
            episodes, contribution, rollout_seconds = pending_rollout.result()
            pending_rollout = None
        else:
            episodes, contribution, rollout_seconds = collect_phase(
                _snapshot(policy), campaign_phase
            )
        rollout_wait_seconds = time.perf_counter() - rollout_wait_started
        if rollout_executor is not None and offset + 1 < args.epochs:
            pending_rollout = rollout_executor.submit(
                collect_phase,
                _snapshot(policy),
                campaign_phase_for_offset(offset + 1),
            )
        pass_metrics = _strategic_pass_metrics(episodes, PASS_ACTION)
        if args.device == "cuda":
            torch.cuda.synchronize()
        update_started = time.perf_counter()
        with _update_autocast(args.device, update_precision):
            update = ppo_update_batched_epochs(
                policy,
                episodes,
                optimizer,
                update_epochs=args.update_epochs,
                minibatch_trajectories=args.minibatch_trajectories,
                seed=args.seed + 1_000_000 + campaign_phase,
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
                early_stop_diagnostic_trajectories=(
                    args.early_stop_diagnostic_trajectories
                ),
                static_compile_padding=compile_updates,
                length_bucketed_minibatches=args.persistent_replay,
                persistent_prepared_batches=args.persistent_replay,
                full_final_diagnostic=args.full_rollout_diagnostics,
            )
        if args.device == "cuda":
            torch.cuda.synchronize()
        update_seconds = time.perf_counter() - update_started

        evaluations = None
        if epoch % args.evaluate_every == 0 or offset == args.epochs - 1:
            evaluation_policy = _snapshot(policy)
            candidate = temporal_agent_factory(
                evaluation_policy,
                deterministic=True,
                seed=args.seed + campaign_phase,
            )
            evaluations = {
                "ppo_champion": evaluate_paired(
                    candidate,
                    champion_factory,
                    pairs=args.evaluation_pairs,
                    seed=(
                        args.seed
                        + 2_000_000
                        + campaign_phase * 10_000
                    ),
                    candidate_name=f"puffer_epoch_{epoch}",
                    opponent_name="ppo_champion",
                ),
                "random": evaluate_paired(
                    candidate,
                    AgentRandom,
                    pairs=args.evaluation_pairs,
                    seed=(
                        args.seed
                        + 3_000_000
                        + campaign_phase * 10_000
                    ),
                    candidate_name=f"puffer_epoch_{epoch}",
                    opponent_name="random",
                ),
                "smart": evaluate_paired(
                    candidate,
                    SmartAgent,
                    pairs=args.evaluation_pairs,
                    seed=(
                        args.seed
                        + 4_000_000
                        + campaign_phase * 10_000
                    ),
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
            "model": model_config,
            "parent_checkpoint": str(resume),
            "source_checkpoint_sha256": source_sha256,
            "state_dict": cpu_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "recipe": {
                **recipe,
                "schedule": list(schedule),
                "player_count_schedule": list(player_counts),
                "historical_checkpoints": historical_records,
                "games_per_epoch": args.games_per_epoch,
                "max_turns": args.max_turns,
                "rollout_workers": args.rollout_workers,
                "effective_rollout_workers": (
                    _effective_vector_workers(
                        args.rollout_workers,
                        schedule,
                        player_counts,
                        args.games_per_epoch,
                        rollout_precision,
                    )
                    if args.rollout_engine == "vector"
                    else args.rollout_workers
                ),
                "chunks_per_worker": args.chunks_per_worker,
                "rollout_engine": args.rollout_engine,
                "rollout_precision": rollout_precision,
                "trace_int8_rollouts": args.trace_int8_rollouts,
                "rollout_max_public_events": rollout_max_public_events,
                "requested_rollout_precision": args.rollout_precision,
                "pipelined_rollouts": pipelined_rollouts,
                "requested_pipelined_rollouts": args.pipelined_rollouts,
                "rollout_policy_lag_updates": (
                    1 if pipelined_rollouts else 0
                ),
                "update_epochs": args.update_epochs,
                "freeze_residual_base": args.freeze_residual_base,
                "training_scope": training_scope,
                "optimizer_state_pruning": optimizer_state_pruning,
                "minibatch_trajectories": args.minibatch_trajectories,
                "diagnostic_minibatch_trajectories": (
                    args.diagnostic_minibatch_trajectories
                ),
                "learning_rate": learning_rate,
                "fresh_optimizer": args.fresh_optimizer,
                "target_kl": args.target_kl,
                "early_stop_diagnostic_trajectories": (
                    args.early_stop_diagnostic_trajectories
                ),
                "full_rollout_diagnostics": args.full_rollout_diagnostics,
                "compile_updates": compile_updates,
                "requested_compile_updates": args.compile_updates,
                "update_precision": update_precision,
                "requested_update_precision": args.update_precision,
                "persistent_replay": args.persistent_replay,
                "matmul_precision": args.matmul_precision,
                "champion_checkpoint": str(champion_path),
                "champion_checkpoint_sha256": _file_sha256(champion_path),
                "campaign_offset": args.campaign_offset,
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
            "campaign_phase": campaign_phase,
            "rollout_precision": rollout_precision,
            "trace_int8_rollouts": args.trace_int8_rollouts,
            "rollout_max_public_events": rollout_max_public_events,
            "update_precision": update_precision,
            "training_scope": training_scope,
            "optimizer_state_pruning": optimizer_state_pruning,
            "wall_seconds": time.perf_counter() - epoch_started,
            "rollout_seconds": rollout_seconds,
            "rollout_wait_seconds": rollout_wait_seconds,
            "rollout_overlap_seconds": max(
                0.0, rollout_seconds - rollout_wait_seconds
            ),
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
                "rollout_precision": rollout_precision,
                "trace_int8_rollouts": args.trace_int8_rollouts,
                "rollout_max_public_events": rollout_max_public_events,
                "update_precision": update_precision,
                "training_scope": training_scope,
                "optimizer_state_pruning": optimizer_state_pruning,
                "pipelined_rollouts": pipelined_rollouts,
                "rollout_policy_lag_updates": (
                    1 if pipelined_rollouts else 0
                ),
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
                    "rollout_precision": rollout_precision,
                    "update_precision": update_precision,
                    "games_per_second": row["games_per_second"],
                    "decisions": pass_metrics["decisions"],
                    "rollout_seconds": rollout_seconds,
                    "rollout_wait_seconds": rollout_wait_seconds,
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
    if rollout_executor is not None:
        rollout_executor.shutdown(wait=True, cancel_futures=True)
    if vector_rollout_pool is not None:
        vector_rollout_pool.close()


if __name__ == "__main__":
    main()
