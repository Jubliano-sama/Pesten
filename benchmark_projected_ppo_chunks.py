#!/usr/bin/env python3
"""Benchmark exact projected-PPO training across base chunk sizes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

from anchored_ppo import CHAMPION_SHA256, file_sha256
from nashpg_ppo import module_sha256, seed_everything
from projected_ppo import (
    collect_common_projected_batch,
    configure_projected_ppo,
    train_common_batch_candidate,
    warm_projected_critic,
)
from seed_integrity import DEVELOPMENT_SEED_LIMIT


REQUIRED_CHUNK_SIZES = (32, 64, 128, 256, 512)
BENCHMARK_CONFIG = {
    "name": "kl0p5_lr3em6_ent1em3",
    "learning_rate": 3e-6,
    "kl_coefficient": 0.5,
    "entropy_coefficient": 1e-3,
    "target_kl": 1e-3,
}


def structured_sha256(value: Any) -> str:
    """Hash nested state dictionaries without serialization metadata."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode())
            digest.update(
                np.asarray(tensor.shape, dtype=np.int64).tobytes()
            )
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda entry: str(entry)):
                visit(str(key))
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        elif isinstance(item, np.ndarray):
            digest.update(b"ndarray")
            digest.update(str(item.dtype).encode())
            digest.update(np.asarray(item.shape, dtype=np.int64).tobytes())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif item is None:
            digest.update(b"none")
        elif isinstance(item, bool):
            digest.update(b"true" if item else b"false")
        elif isinstance(item, (int, float, str)):
            digest.update(type(item).__name__.encode())
            digest.update(repr(item).encode())
        else:
            raise TypeError(f"unsupported state value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    return structured_sha256(tensor)


def actor_state_difference(
    candidate: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
) -> dict:
    if candidate.keys() != reference.keys():
        raise ValueError("actor state dictionaries have different keys")
    differing = 0
    maximum = 0.0
    sum_squared = 0.0
    elements = 0
    for key in candidate:
        left = candidate[key].detach().cpu()
        right = reference[key].detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"actor tensor mismatch at {key}")
        delta = left.to(torch.float64) - right.to(torch.float64)
        differing += int(torch.count_nonzero(delta))
        if delta.numel():
            maximum = max(maximum, float(delta.abs().max()))
        sum_squared += float(delta.square().sum())
        elements += delta.numel()
    return {
        "differing_parameter_elements": differing,
        "total_parameter_elements": elements,
        "max_abs_parameter_difference": maximum,
        "rms_parameter_difference": (
            (sum_squared / elements) ** 0.5 if elements else 0.0
        ),
        "bit_identical": differing == 0,
    }


def prepare_reset_template(
    champion: Path,
    *,
    seed: int,
    critic_games: int,
    critic_epochs: int,
    minibatch_size: int,
    temperature: float,
    rollout_batch_size: int,
    torch_threads: int,
) -> tuple[dict, dict]:
    """Warm one critic and freeze the exact reset states for every trial."""
    seed_everything(seed)
    ppo, base_actor = configure_projected_ppo(
        champion,
        learning_rate=BENCHMARK_CONFIG["learning_rate"],
        rollout_batch_size=rollout_batch_size,
        base_chunk_size=64,
        torch_threads=torch_threads,
    )
    warmup = warm_projected_critic(
        ppo,
        games=critic_games,
        epochs=critic_epochs,
        minibatch_size=minibatch_size,
        temperature=temperature,
    )
    template = {
        "actor": copy.deepcopy(base_actor.state_dict()),
        "critic": copy.deepcopy(ppo.critic.state_dict()),
        "actor_optimizer": copy.deepcopy(ppo.actor_optim.state_dict()),
        "critic_optimizer": copy.deepcopy(ppo.critic_optim.state_dict()),
    }
    return template, warmup


def run_chunk_trial(
    champion: Path,
    output_dir: Path,
    reset_template: dict,
    *,
    chunk_size: int,
    common_seed: int,
    games: int,
    update_epochs: int,
    minibatch_size: int,
    diagnostic_chunk_size: int,
    rollout_batch_size: int,
    temperature: float,
    torch_threads: int,
) -> tuple[dict, dict[str, torch.Tensor]]:
    """Collect and update from an exactly reset projected PPO state."""
    seed_everything(common_seed)
    ppo, base_actor = configure_projected_ppo(
        champion,
        learning_rate=BENCHMARK_CONFIG["learning_rate"],
        rollout_batch_size=rollout_batch_size,
        base_chunk_size=chunk_size,
        torch_threads=torch_threads,
    )
    base_actor.load_state_dict(reset_template["actor"])
    ppo.critic.load_state_dict(reset_template["critic"])
    ppo.actor_optim.load_state_dict(
        copy.deepcopy(reset_template["actor_optimizer"])
    )
    ppo.critic_optim.load_state_dict(
        copy.deepcopy(reset_template["critic_optimizer"])
    )
    initial = {
        "actor_state_sha256": structured_sha256(
            base_actor.state_dict()
        ),
        "critic_state_sha256": structured_sha256(ppo.critic.state_dict()),
        "actor_optimizer_sha256": structured_sha256(
            ppo.actor_optim.state_dict()
        ),
        "critic_optimizer_sha256": structured_sha256(
            ppo.critic_optim.state_dict()
        ),
    }
    collection_started = time.perf_counter()
    common = collect_common_projected_batch(
        ppo,
        games=games,
        temperature=temperature,
        diagnostic_chunk_size=diagnostic_chunk_size,
    )
    collection_wall = time.perf_counter() - collection_started
    initial_logits_sha = tensor_sha256(common["old_logits"])
    update_started = time.perf_counter()
    update = train_common_batch_candidate(
        ppo,
        base_actor,
        common,
        BENCHMARK_CONFIG,
        update_epochs=update_epochs,
        minibatch_size=minibatch_size,
        temperature=temperature,
        permutation_seed=common_seed + 20_000,
        diagnostic_chunk_size=diagnostic_chunk_size,
    )
    update_wall = time.perf_counter() - update_started
    if not update["policy_updated"] or not update["target_kl_satisfied"]:
        raise RuntimeError("chunk trial did not retain a valid policy update")
    checkpoint_name = f"projected_ppo_chunk_{chunk_size}.pth"
    checkpoint = output_dir / checkpoint_name
    state = copy.deepcopy(base_actor.state_dict())
    torch.save(state, checkpoint)
    public_common = {
        key: value
        for key, value in common.items()
        if not isinstance(value, torch.Tensor)
    }
    return (
        {
            "base_chunk_size": chunk_size,
            "initial_state": initial,
            "common_rollout": public_common,
            "common_batch_sha256": common["sha256"],
            "initial_projected_logits_sha256": initial_logits_sha,
            "exact_initial_logits_match_behavior": (
                update["initial_exact_behavior_kl_old_to_new"] == 0.0
            ),
            "max_initial_ratio_error": update[
                "max_initial_ratio_error"
            ],
            "collection_wall_seconds": collection_wall,
            "update_wall_seconds": update_wall,
            "total_wall_seconds": collection_wall + update_wall,
            "final_exact_behavior_kl_old_to_new": update[
                "final_exact_behavior_kl_old_to_new"
            ],
            "target_kl": BENCHMARK_CONFIG["target_kl"],
            "target_kl_satisfied": update["target_kl_satisfied"],
            "peak_augmented_base_actor_call_observations": (
                min(chunk_size, minibatch_size) * 24
            ),
            "update": update,
            "checkpoint": {
                "path": checkpoint_name,
                "sha256": file_sha256(checkpoint),
                "actor_state_sha256": module_sha256(base_actor),
            },
        },
        state,
    )


def run_benchmark(
    champion: Path,
    output_dir: Path,
    *,
    chunk_sizes: Sequence[int],
    seed: int,
    critic_games: int,
    critic_epochs: int,
    games: int,
    update_epochs: int,
    minibatch_size: int,
    diagnostic_chunk_size: int,
    rollout_batch_size: int,
    temperature: float,
    torch_threads: int,
) -> dict:
    if len(set(chunk_sizes)) != len(chunk_sizes):
        raise ValueError("chunk sizes must be unique")
    if any(size < 1 for size in chunk_sizes):
        raise ValueError("chunk sizes must be positive")
    if seed < 0 or seed + 10_000 >= DEVELOPMENT_SEED_LIMIT:
        raise ValueError("benchmark seeds must remain development-only")
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    reset_template, warmup = prepare_reset_template(
        champion,
        seed=seed,
        critic_games=critic_games,
        critic_epochs=critic_epochs,
        minibatch_size=minibatch_size,
        temperature=temperature,
        rollout_batch_size=rollout_batch_size,
        torch_threads=torch_threads,
    )
    reset_hashes = {
        key: structured_sha256(value)
        for key, value in reset_template.items()
    }
    trials = []
    final_states = {}
    common_seed = seed + 10_000
    for chunk_size in chunk_sizes:
        trial, final_state = run_chunk_trial(
            champion,
            output_dir,
            reset_template,
            chunk_size=chunk_size,
            common_seed=common_seed,
            games=games,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            diagnostic_chunk_size=diagnostic_chunk_size,
            rollout_batch_size=rollout_batch_size,
            temperature=temperature,
            torch_threads=torch_threads,
        )
        if trial["initial_state"] != {
            "actor_state_sha256": reset_hashes["actor"],
            "critic_state_sha256": reset_hashes["critic"],
            "actor_optimizer_sha256": reset_hashes["actor_optimizer"],
            "critic_optimizer_sha256": reset_hashes[
                "critic_optimizer"
            ],
        }:
            raise RuntimeError("chunk trial did not start from reset template")
        trials.append(trial)
        final_states[chunk_size] = final_state
    batch_hashes = {trial["common_batch_sha256"] for trial in trials}
    logits_hashes = {
        trial["initial_projected_logits_sha256"]
        for trial in trials
    }
    if len(batch_hashes) != 1 or len(logits_hashes) != 1:
        raise RuntimeError("chunk sizes did not use identical common data")
    reference_chunk = (
        512 if 512 in final_states else max(final_states)
    )
    reference_state = final_states[reference_chunk]
    reference_file_sha = next(
        trial["checkpoint"]["sha256"]
        for trial in trials
        if trial["base_chunk_size"] == reference_chunk
    )
    for trial in trials:
        trial["checkpoint_difference_vs_reference"] = {
            "reference_chunk_size": reference_chunk,
            "file_sha256_identical": (
                trial["checkpoint"]["sha256"] == reference_file_sha
            ),
            **actor_state_difference(
                final_states[trial["base_chunk_size"]],
                reference_state,
            ),
        }
    payload = {
        "schema_version": 1,
        "benchmark": "projected_ppo_base_chunk_throughput",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "champion": {
            "path": str(champion.resolve()),
            "sha256": CHAMPION_SHA256,
        },
        "projection": {
            "ordinary_suit_permutations": 24,
            "aggregation": "arithmetic_mean_logits",
            "same_actor_for_rollout_and_update": True,
        },
        "reset_contract": {
            "identical_actor": True,
            "identical_critic": True,
            "identical_actor_optimizer": True,
            "identical_critic_optimizer": True,
            "identical_common_batch": True,
            "identical_permutation_seed": True,
            "state_sha256": reset_hashes,
        },
        "seed_audit": {
            "root_seed": seed,
            "common_rollout_seed": common_seed,
            "development_seed_limit": DEVELOPMENT_SEED_LIMIT,
            "development_only": True,
            "sealed_inputs_used": False,
            "common_deals_intentionally_reused_across_trials": True,
        },
        "recipe": {
            "chunk_sizes": list(chunk_sizes),
            "config": dict(BENCHMARK_CONFIG),
            "critic_games": critic_games,
            "critic_epochs": critic_epochs,
            "games": games,
            "update_epochs": update_epochs,
            "minibatch_size": minibatch_size,
            "diagnostic_chunk_size": diagnostic_chunk_size,
            "rollout_batch_size": rollout_batch_size,
            "temperature": temperature,
            "torch_threads": torch_threads,
        },
        "critic_warmup": warmup,
        "common_batch_sha256": next(iter(batch_hashes)),
        "initial_projected_logits_sha256": next(iter(logits_hashes)),
        "reference_chunk_size": reference_chunk,
        "trials": trials,
        "wall_seconds": time.perf_counter() - started,
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--champion",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/projected_ppo_chunk_benchmark_dev"),
    )
    parser.add_argument("--seed", type=int, default=84_900_001)
    parser.add_argument(
        "--chunk-sizes",
        type=int,
        nargs="+",
        default=list(REQUIRED_CHUNK_SIZES),
    )
    parser.add_argument("--critic-games", type=int, default=64)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--diagnostic-chunk-size", type=int, default=64)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if tuple(args.chunk_sizes) != REQUIRED_CHUNK_SIZES:
        raise SystemExit(
            "production benchmark requires chunk sizes 32 64 128 256 512"
        )
    if min(
        args.critic_games,
        args.critic_epochs,
        args.games,
        args.update_epochs,
        args.minibatch_size,
        args.diagnostic_chunk_size,
        args.rollout_batch_size,
        args.torch_threads,
    ) < 1:
        raise SystemExit("all size/count arguments must be positive")
    if args.temperature <= 0 or not np.isfinite(args.temperature):
        raise SystemExit("temperature must be finite and positive")
    payload = run_benchmark(
        args.champion,
        args.output_dir,
        chunk_sizes=args.chunk_sizes,
        seed=args.seed,
        critic_games=args.critic_games,
        critic_epochs=args.critic_epochs,
        games=args.games,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        diagnostic_chunk_size=args.diagnostic_chunk_size,
        rollout_batch_size=args.rollout_batch_size,
        temperature=args.temperature,
        torch_threads=args.torch_threads,
    )
    manifest = args.output_dir / "projected_ppo_chunk_benchmark.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest.resolve()),
                "chunk_sizes": args.chunk_sizes,
                "common_batch_sha256": payload["common_batch_sha256"],
                "wall_seconds": payload["wall_seconds"],
                "sealed_inputs_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
