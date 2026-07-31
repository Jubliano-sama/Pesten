#!/usr/bin/env python3
"""Bounded iterative continuation for the exact projected PPO actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

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


ITERATIVE_LANES = (
    {
        "name": "kl0p5_lr3em6_ent1em3",
        "learning_rate": 3e-6,
        "kl_coefficient": 0.5,
        "entropy_coefficient": 1e-3,
        "target_kl": 1e-3,
    },
    {
        "name": "kl0p5_lr3em6_ent0",
        "learning_rate": 3e-6,
        "kl_coefficient": 0.5,
        "entropy_coefficient": 0.0,
        "target_kl": 1e-3,
    },
)


def register_unique_deal_seeds(
    registry: set[int],
    rollout: dict,
    *,
    label: str,
) -> list[int]:
    """Fail closed unless every replay-verified development deal is new."""
    if not rollout.get("deal_seed_rng_replay_verified"):
        raise RuntimeError(f"{label}: deal RNG replay was not verified")
    records = rollout.get("deal_seeds")
    if not isinstance(records, list):
        raise RuntimeError(f"{label}: deal seed records are missing")
    values = [int(record["deal_seed"]) for record in records]
    if len(values) != rollout.get("deal_seed_count"):
        raise RuntimeError(f"{label}: deal seed count mismatch")
    if len(values) != len(set(values)):
        raise RuntimeError(f"{label}: duplicate deal within rollout")
    if any(
        seed < 0 or seed >= DEVELOPMENT_SEED_LIMIT
        for seed in values
    ):
        raise RuntimeError(f"{label}: deal entered sealed seed namespace")
    overlap = registry.intersection(values)
    if overlap:
        raise RuntimeError(
            f"{label}: deal seed reused across training rollouts"
        )
    registry.update(values)
    return values


def _public_batch(batch: dict) -> dict:
    return {
        key: value
        for key, value in batch.items()
        if not isinstance(value, torch.Tensor)
    }


def run_iterative_lane(
    champion: Path,
    output_dir: Path,
    config: dict,
    *,
    lane_seed: int,
    iterations: int,
    critic_games: int,
    critic_epochs: int,
    games: int,
    update_epochs: int,
    minibatch_size: int,
    diagnostic_chunk_size: int,
    base_chunk_size: int,
    rollout_batch_size: int,
    temperature: float,
    torch_threads: int,
    deal_registry: set[int],
) -> dict:
    """Run one champion-started projected lane and save every iterate."""
    seed_everything(lane_seed)
    ppo, base_actor = configure_projected_ppo(
        champion,
        learning_rate=config["learning_rate"],
        rollout_batch_size=rollout_batch_size,
        base_chunk_size=base_chunk_size,
        torch_threads=torch_threads,
    )
    champion_actor_sha = module_sha256(base_actor)
    warmup = warm_projected_critic(
        ppo,
        games=critic_games,
        epochs=critic_epochs,
        minibatch_size=minibatch_size,
        temperature=temperature,
    )
    register_unique_deal_seeds(
        deal_registry,
        warmup["rollout"],
        label=f"{config['name']}:critic",
    )
    lane = {
        "name": config["name"],
        "config": dict(config),
        "lane_seed": lane_seed,
        "initial_actor_state_sha256": champion_actor_sha,
        "critic_warmup": warmup,
        "iterations": [],
    }
    prior_batches: set[str] = set()
    for iteration in range(iterations):
        ppo.actor.reset_statistics()
        started = time.perf_counter()
        batch = collect_common_projected_batch(
            ppo,
            games=games,
            temperature=temperature,
            diagnostic_chunk_size=diagnostic_chunk_size,
        )
        if batch["sha256"] in prior_batches:
            raise RuntimeError("iterative lane reused a common rollout batch")
        prior_batches.add(batch["sha256"])
        register_unique_deal_seeds(
            deal_registry,
            batch["rollout"],
            label=f"{config['name']}:iteration:{iteration + 1}",
        )
        update = train_common_batch_candidate(
            ppo,
            base_actor,
            batch,
            config,
            update_epochs=update_epochs,
            minibatch_size=minibatch_size,
            temperature=temperature,
            permutation_seed=lane_seed + 20_000 + iteration,
            diagnostic_chunk_size=diagnostic_chunk_size,
        )
        if not update["policy_updated"]:
            raise RuntimeError(
                f"{config['name']} iteration {iteration + 1} retained no update"
            )
        if not update["target_kl_satisfied"]:
            raise RuntimeError("retained projected iterate exceeds target KL")
        checkpoint_name = (
            f"projected_iter_{config['name']}_{iteration + 1:02d}.pth"
        )
        checkpoint = output_dir / checkpoint_name
        torch.save(base_actor.state_dict(), checkpoint)
        lane["iterations"].append(
            {
                "iteration": iteration + 1,
                "fresh_rollout": _public_batch(batch),
                "update": update,
                "checkpoint": {
                    "path": checkpoint_name,
                    "sha256": file_sha256(checkpoint),
                    "actor_state_sha256": module_sha256(base_actor),
                    "raw_base_requires_24_suit_projection": True,
                },
                "wall_seconds": time.perf_counter() - started,
            }
        )
    return lane


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
        default=Path("benchmarks/projected_ppo_iterative_dev"),
    )
    parser.add_argument("--seed", type=int, default=84_700_001)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--critic-games", type=int, default=256)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--diagnostic-chunk-size", type=int, default=64)
    parser.add_argument("--base-chunk-size", type=int, default=64)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--torch-threads", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.seed < 0 or args.seed + 1_000_000 >= DEVELOPMENT_SEED_LIMIT:
        raise SystemExit("lane root seeds must remain in development range")
    if not 6 <= args.iterations <= 10:
        raise SystemExit("bounded iterative lane requires 6-10 iterations")
    if min(
        args.critic_games,
        args.critic_epochs,
        args.games,
        args.update_epochs,
        args.minibatch_size,
        args.diagnostic_chunk_size,
        args.base_chunk_size,
        args.rollout_batch_size,
        args.torch_threads,
    ) < 1:
        raise SystemExit("all size/count arguments must be positive")
    if args.temperature <= 0 or not np.isfinite(args.temperature):
        raise SystemExit("temperature must be finite and positive")
    if file_sha256(args.champion) != CHAMPION_SHA256:
        raise SystemExit("production champion SHA-256 mismatch")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deal_registry: set[int] = set()
    lanes = []
    for lane_index, config in enumerate(ITERATIVE_LANES):
        lanes.append(
            run_iterative_lane(
                args.champion,
                args.output_dir,
                config,
                lane_seed=args.seed + lane_index * 1_000_000,
                iterations=args.iterations,
                critic_games=args.critic_games,
                critic_epochs=args.critic_epochs,
                games=args.games,
                update_epochs=args.update_epochs,
                minibatch_size=args.minibatch_size,
                diagnostic_chunk_size=args.diagnostic_chunk_size,
                base_chunk_size=args.base_chunk_size,
                rollout_batch_size=args.rollout_batch_size,
                temperature=args.temperature,
                torch_threads=args.torch_threads,
                deal_registry=deal_registry,
            )
        )
    payload = {
        "schema_version": 1,
        "algorithm": "exact_24_suit_projected_ppo_iterative",
        "champion": {
            "path": str(args.champion.resolve()),
            "file_sha256": CHAMPION_SHA256,
        },
        "projection": {
            "ordinary_suit_permutations": 24,
            "aggregation": "arithmetic_mean_logits",
            "alpha": 1.0,
            "player_orbit_mode": "none",
            "rollout_actor_is_update_actor": True,
            "trainable_shared_base": True,
            "saved_checkpoints_are_raw_base_weights": True,
            "production_legal_mask_preserved": True,
        },
        "privacy": {
            "players": 2,
            "self_play": True,
            "own_hand_plus_public_state_only": True,
            "opponent_card_identities": False,
            "draw_pile_order": False,
        },
        "payoff": {
            "terminal_two_player_zero_sum_only": True,
            "standard_per_pass_shaping_removed": True,
        },
        "seed_audit": {
            "root_seed": args.seed,
            "development_seed_limit": DEVELOPMENT_SEED_LIMIT,
            "development_only": True,
            "sealed_outcomes_used": False,
            "per_deal_seeds_recorded": True,
            "rng_replay_verified_per_rollout": True,
            "globally_unique_training_deals": True,
            "unique_deal_count": len(deal_registry),
        },
        "recipe": {
            "iterations": args.iterations,
            "critic_games_per_lane": args.critic_games,
            "critic_epochs": args.critic_epochs,
            "games_per_iteration": args.games,
            "update_epochs": args.update_epochs,
            "minibatch_size": args.minibatch_size,
            "temperature": args.temperature,
            "base_chunk_size": args.base_chunk_size,
            "diagnostic_chunk_size": args.diagnostic_chunk_size,
            "rollout_batch_size": args.rollout_batch_size,
            "torch_threads": args.torch_threads,
            "lanes": list(ITERATIVE_LANES),
        },
        "lanes": lanes,
        "wall_seconds": time.perf_counter() - started,
    }
    manifest = args.output_dir / "projected_ppo_iterative_metrics.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest.resolve()),
                "lanes": len(lanes),
                "checkpoints": sum(
                    len(lane["iterations"])
                    for lane in lanes
                ),
                "unique_training_deals": len(deal_registry),
                "sealed_outcomes_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
