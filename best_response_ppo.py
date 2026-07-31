#!/usr/bin/env python3
"""Development-only PPO approximate-best-response probe.

This diagnostic trains one candidate against the immutable raw, unprojected
incumbent.  Every explicit deal is played twice with the candidate
in opposite seats.  Only the candidate's ordinary 121-feature observation is
collected, and the optimized reward is the undiscounted terminal zero-sum
win/draw/loss payoff.

The result is an approximate response found by one finite optimizer, not an
exact best response and therefore not an exploitability certificate.  The
default sweep varies incumbent-KL anchoring and entropy regularization to expose
optimization instability and overfitting before independent cross-play.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Iterable, Sequence

import numpy as np
import torch

from ai_observation import OBSERVATION_SIZE, PASS_ACTION
from anchored_ppo import (
    CHAMPION_SHA256,
    enforce_initial_ratio_invariant,
    file_sha256,
)
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    POLICY_RL,
    collect_batched_rollout,
)
from ppo2 import DEFAULT_TORCH_THREADS, StudentNet, build_action_masks, compute_gae
from seed_integrity import scan_recorded_seed_interval


DEFAULT_INCUMBENT = Path("ppo_actor_100run3.pth")
DEFAULT_OUTPUT_DIR = Path("benchmarks/best_response_ppo_sweep_dev")
DEVELOPMENT_SEED_LIMIT = 92_000_000
CANDIDATE_KEY = "candidate"
INCUMBENT_KEY = "raw_incumbent"
PPO_PAPER_URL = "https://arxiv.org/abs/1707.06347"
PSRO_PAPER_URL = "https://arxiv.org/abs/1711.00832"


@dataclass(frozen=True)
class SweepVariant:
    name: str
    champion_kl_coefficient: float
    entropy_coefficient: float


DEFAULT_VARIANTS = (
    SweepVariant("plain", 0.0, 0.0),
    SweepVariant("entropy_0p01", 0.0, 0.01),
    SweepVariant("anchor_0p02", 0.02, 0.0),
    SweepVariant("anchor_0p02_entropy_0p01", 0.02, 0.01),
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def validate_variant(variant: SweepVariant) -> None:
    if not variant.name or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789_"
        for character in variant.name
    ):
        raise ValueError("variant names must be non-empty lowercase slugs")
    for label, value in (
        ("champion KL coefficient", variant.champion_kl_coefficient),
        ("entropy coefficient", variant.entropy_coefficient),
    ):
        if value < 0 or not np.isfinite(value):
            raise ValueError(f"{label} must be finite and non-negative")


def parse_variant(text: str) -> SweepVariant:
    """Parse NAME:CHAMPION_KL:ENTROPY."""
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "variant must be NAME:CHAMPION_KL:ENTROPY"
        )
    try:
        variant = SweepVariant(parts[0], float(parts[1]), float(parts[2]))
        validate_variant(variant)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return variant


def allocate_fresh_seed_plan(
    *,
    requested_start: int,
    deal_count: int,
    policy_seed_count: int,
    shuffle_seed_count: int,
    recorded_seeds: Iterable[int],
) -> dict:
    """Allocate one explicit collision-free development seed plan."""
    if requested_start < 0 or requested_start >= DEVELOPMENT_SEED_LIMIT:
        raise ValueError("requested seed start must be below 92 million")
    if min(deal_count, policy_seed_count, shuffle_seed_count) < 1:
        raise ValueError("seed plan counts must be positive")
    recorded = {
        int(seed)
        for seed in recorded_seeds
        if 0 <= int(seed) < DEVELOPMENT_SEED_LIMIT
    }
    cursor = int(requested_start)
    skipped: list[int] = []

    def allocate(count: int) -> list[int]:
        nonlocal cursor
        result = []
        while len(result) < count:
            if cursor >= DEVELOPMENT_SEED_LIMIT:
                raise ValueError("best-response seed plan crossed 92 million")
            if cursor in recorded:
                skipped.append(cursor)
            else:
                result.append(cursor)
            cursor += 1
        return result

    training_seed = allocate(1)[0]
    deal_seeds = allocate(deal_count)
    policy_sampling_seeds = allocate(policy_seed_count)
    shuffle_seeds = allocate(shuffle_seed_count)
    all_allocated = [
        training_seed,
        *deal_seeds,
        *policy_sampling_seeds,
        *shuffle_seeds,
    ]
    if len(set(all_allocated)) != len(all_allocated):
        raise RuntimeError("best-response seed allocator emitted a duplicate")
    return {
        "training_seed": training_seed,
        "deal_seeds": deal_seeds,
        "policy_sampling_seeds": policy_sampling_seeds,
        "shuffle_seeds": shuffle_seeds,
        "requested_start": int(requested_start),
        "minimum_seed": min(all_allocated),
        "maximum_seed": max(all_allocated),
        "skipped_recorded_seeds": skipped,
        "recorded_seed_values_seen": len(recorded),
        "all_seeds_below_92m": max(all_allocated) < DEVELOPMENT_SEED_LIMIT,
    }


def recorded_deal_seeds(seeds: Iterable[int]) -> list[dict[str, int]]:
    return [{"deal_seed": int(seed)} for seed in seeds]


def build_matched_fixed_opponent_layout(
    deal_seeds: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return two games per deal with candidate/incumbent seats exchanged."""
    seeds = np.asarray(deal_seeds, dtype=np.uint32)
    if seeds.ndim != 1 or len(seeds) < 1:
        raise ValueError("at least one one-dimensional deal seed is required")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("common deal seeds must be unique")
    games = 2 * len(seeds)
    policy_kinds = np.full((games, 2), POLICY_RL, dtype=np.int8)
    actor_keys = np.empty((games, 2), dtype=object)
    actor_keys[0::2] = (CANDIDATE_KEY, INCUMBENT_KEY)
    actor_keys[1::2] = (INCUMBENT_KEY, CANDIDATE_KEY)
    repeated_seeds = np.repeat(seeds, 2)
    candidate_seats = np.tile(np.asarray([0, 1], dtype=np.int8), len(seeds))
    return policy_kinds, actor_keys, repeated_seeds, candidate_seats


def terminal_only_rewards(
    actions: torch.Tensor,
    rollout_rewards: torch.Tensor,
) -> torch.Tensor:
    """Remove the collector's pass penalty and retain terminal payoff only."""
    if actions.shape != rollout_rewards.shape:
        raise ValueError("actions and rewards must have equal shape")
    return rollout_rewards + 0.1 * (actions == PASS_ACTION).to(
        rollout_rewards.dtype
    )


def audit_terminal_candidate_payoffs(
    rewards: torch.Tensor,
    episode_lengths: Sequence[int],
    *,
    engine: BatchedPesten,
    candidate_seats: np.ndarray,
    atol: float = 1e-6,
) -> dict:
    """Audit terminal samples and environment-level two-player zero-sum payoff."""
    if len(candidate_seats) != engine.num_games:
        raise ValueError("one candidate seat is required per game")
    if sum(int(length) for length in episode_lengths) != len(rewards):
        raise ValueError("episode lengths do not cover candidate rewards")
    if any(int(length) <= 0 for length in episode_lengths):
        raise ValueError("populated candidate episodes must be positive")
    offset = 0
    sampled_returns = []
    for length_value in episode_lengths:
        length = int(length_value)
        sampled_returns.append(rewards[offset : offset + length].sum())
        offset += length
    if sampled_returns:
        sampled = torch.stack(sampled_returns)
        valid = torch.stack(
            [
                torch.isclose(
                    sampled,
                    sampled.new_full((), value),
                    atol=atol,
                    rtol=0,
                )
                for value in (-1.0, 0.0, 1.0)
            ]
        ).any(dim=0)
        if not bool(valid.all()):
            raise RuntimeError("candidate episode return is not terminal payoff")

    winners = engine.winners.astype(np.int64, copy=False)
    seats = np.asarray(candidate_seats, dtype=np.int64)
    candidate_payoffs = np.where(
        winners < 0,
        0.0,
        np.where(winners == seats, 1.0, -1.0),
    )
    opponent_payoffs = -candidate_payoffs
    if not np.allclose(candidate_payoffs + opponent_payoffs, 0.0):
        raise RuntimeError("fixed-opponent games are not zero-sum")
    first = candidate_payoffs[0::2]
    second = candidate_payoffs[1::2]
    pair_scores = (first + second + 2.0) / 4.0
    return {
        "games": int(engine.num_games),
        "matched_deal_pairs": int(engine.num_games // 2),
        "candidate_seat_0_games": int(np.count_nonzero(seats == 0)),
        "candidate_seat_1_games": int(np.count_nonzero(seats == 1)),
        "candidate_wins": int(np.count_nonzero(candidate_payoffs == 1.0)),
        "candidate_draws": int(np.count_nonzero(candidate_payoffs == 0.0)),
        "candidate_losses": int(np.count_nonzero(candidate_payoffs == -1.0)),
        "candidate_mean_payoff": float(candidate_payoffs.mean()),
        "candidate_matched_pair_score": float(pair_scores.mean()),
        "populated_candidate_episodes": len(episode_lengths),
        "implicit_zero_decision_candidate_games": (
            int(engine.num_games) - len(episode_lengths)
        ),
        "terminal_payoffs_verified": True,
        "environment_zero_sum_verified": True,
    }


def collect_fixed_opponent_response(
    candidate: torch.nn.Module,
    champion: torch.nn.Module,
    *,
    deal_seeds: Sequence[int],
    policy_sampling_seed: int,
    temperature: float,
    max_turns: int,
    device: torch.device,
) -> tuple:
    if not 0 <= policy_sampling_seed < DEVELOPMENT_SEED_LIMIT:
        raise ValueError("policy sampling seed must stay below 92 million")
    if any(
        not 0 <= int(seed) < DEVELOPMENT_SEED_LIMIT for seed in deal_seeds
    ):
        raise ValueError("deal seeds must stay below 92 million")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    seed_everything(policy_sampling_seed)
    policy_kinds, actor_keys, seeds, candidate_seats = (
        build_matched_fixed_opponent_layout(deal_seeds)
    )
    engine = BatchedPesten(
        policy_kinds,
        actor_keys,
        seeds,
        max_turns=max_turns,
    )
    candidate.eval()
    champion.eval()
    rollout = collect_batched_rollout(
        engine,
        {
            CANDIDATE_KEY: ActorPolicy(
                actor=candidate,
                device=device,
                temperature=temperature,
                collect_data=True,
            ),
            INCUMBENT_KEY: ActorPolicy(
                actor=champion,
                device=device,
                temperature=temperature,
                collect_data=False,
            ),
        },
        output_device=device,
    )
    if rollout.completed_games != len(seeds):
        raise RuntimeError("best-response rollout did not complete every game")
    if rollout.observations.shape[-1] != OBSERVATION_SIZE:
        raise RuntimeError("candidate received a non-public observation")
    return rollout, engine, candidate_seats


def _masked_log_probabilities(
    logits: torch.Tensor,
    legal_actions: torch.Tensor,
) -> torch.Tensor:
    floor = torch.finfo(logits.dtype).min
    return torch.log_softmax(logits.masked_fill(~legal_actions, floor), dim=-1)


def masked_candidate_champion_kl(
    candidate_logits: torch.Tensor,
    champion_logits: torch.Tensor,
    legal_actions: torch.Tensor,
) -> torch.Tensor:
    """Return mean KL(candidate || champion) over legal actions."""
    if candidate_logits.shape != champion_logits.shape:
        raise ValueError("candidate and champion logits must match")
    if legal_actions.shape != candidate_logits.shape:
        raise ValueError("legal mask must match logits")
    if legal_actions.dtype != torch.bool:
        raise ValueError("legal mask must be boolean")
    candidate_log = _masked_log_probabilities(
        candidate_logits, legal_actions
    )
    champion_log = _masked_log_probabilities(champion_logits, legal_actions)
    candidate_probabilities = candidate_log.exp()
    return (
        candidate_probabilities * (candidate_log - champion_log)
    ).sum(dim=-1).mean()


def clipped_response_objective(
    ratios: torch.Tensor,
    advantages: torch.Tensor,
    entropy: torch.Tensor,
    champion_kl: torch.Tensor,
    *,
    clip: float,
    entropy_coefficient: float,
    champion_kl_coefficient: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if clip < 0 or not np.isfinite(clip):
        raise ValueError("clip must be finite and non-negative")
    variant = SweepVariant(
        "objective",
        float(champion_kl_coefficient),
        float(entropy_coefficient),
    )
    validate_variant(variant)
    surrogate = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip, 1.0 + clip) * advantages
    clipped_surrogate = torch.minimum(surrogate, clipped).mean()
    objective = (
        clipped_surrogate
        + entropy_coefficient * entropy
        - champion_kl_coefficient * champion_kl
    )
    return objective, clipped_surrogate


def exact_terminal_advantages(
    rewards: torch.Tensor,
    episode_lengths: Sequence[int],
) -> torch.Tensor:
    """Return normalized undiscounted Monte-Carlo terminal returns."""
    zeros = torch.zeros_like(rewards)
    _, returns = compute_gae(
        rewards,
        zeros,
        episode_lengths=list(episode_lengths),
        gamma_=1.0,
        lam=1.0,
    )
    return (returns - returns.mean()) / (
        returns.std(unbiased=False) + 1e-8
    )


def ppo_best_response_update(
    candidate: torch.nn.Module,
    champion: torch.nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    behavior_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    *,
    optimizer: torch.optim.Optimizer,
    temperature: float,
    clip: float,
    entropy_coefficient: float,
    champion_kl_coefficient: float,
    epochs: int,
    batch_size: int,
    shuffle_seed: int,
    maximum_gradient_norm: float = 1.0,
) -> dict:
    if min(epochs, batch_size) < 1:
        raise ValueError("epochs and batch size must be positive")
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if not (
        len(observations)
        == len(actions)
        == len(behavior_log_probabilities)
        == len(advantages)
    ):
        raise ValueError("PPO response batch tensors must have equal length")
    legal_actions = build_action_masks(observations)
    if not bool(legal_actions.gather(1, actions[:, None]).all()):
        raise ValueError("rollout contains an illegal candidate action")
    champion.eval()
    champion_hash_before = module_sha256(champion)
    candidate.eval()
    with torch.no_grad():
        initial_logits = candidate(observations) / temperature
        initial_selected = _masked_log_probabilities(
            initial_logits, legal_actions
        ).gather(1, actions[:, None]).squeeze(1)
        initial_ratios = torch.exp(
            initial_selected - behavior_log_probabilities
        )
        maximum_initial_ratio_error = enforce_initial_ratio_invariant(
            initial_ratios
        )

    generator = torch.Generator(device=observations.device).manual_seed(
        int(shuffle_seed)
    )
    epoch_metrics = []
    candidate.train()
    for epoch in range(epochs):
        order = torch.randperm(
            len(observations),
            generator=generator,
            device=observations.device,
        )
        objectives = []
        surrogates = []
        entropies = []
        kls = []
        gradient_norms = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_observations = observations[indices]
            batch_actions = actions[indices]
            batch_masks = legal_actions[indices]
            logits = candidate(batch_observations) / temperature
            log_probabilities = _masked_log_probabilities(logits, batch_masks)
            selected = log_probabilities.gather(
                1, batch_actions[:, None]
            ).squeeze(1)
            ratios = torch.exp(
                selected - behavior_log_probabilities[indices]
            )
            probabilities = log_probabilities.exp()
            entropy = -(
                probabilities * log_probabilities
            ).sum(dim=-1).mean()
            with torch.no_grad():
                champion_logits = champion(batch_observations) / temperature
            champion_kl = masked_candidate_champion_kl(
                logits,
                champion_logits,
                batch_masks,
            )
            objective, clipped_surrogate = clipped_response_objective(
                ratios,
                advantages[indices],
                entropy,
                champion_kl,
                clip=clip,
                entropy_coefficient=entropy_coefficient,
                champion_kl_coefficient=champion_kl_coefficient,
            )
            optimizer.zero_grad(set_to_none=True)
            (-objective).backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                candidate.parameters(), maximum_gradient_norm
            )
            optimizer.step()
            objectives.append(float(objective.detach()))
            surrogates.append(float(clipped_surrogate.detach()))
            entropies.append(float(entropy.detach()))
            kls.append(float(champion_kl.detach()))
            gradient_norms.append(float(gradient_norm.detach()))
        epoch_metrics.append(
            {
                "epoch": epoch + 1,
                "mean_objective": float(np.mean(objectives)),
                "mean_clipped_surrogate": float(np.mean(surrogates)),
                "mean_entropy": float(np.mean(entropies)),
                "mean_candidate_champion_kl": float(np.mean(kls)),
                "maximum_preclip_gradient_norm": float(max(gradient_norms)),
            }
        )
    candidate.eval()
    if module_sha256(champion) != champion_hash_before:
        raise RuntimeError("frozen champion changed during response update")
    return {
        "maximum_initial_ratio_error": maximum_initial_ratio_error,
        "unchanged_policy_ratio_guard_passed": True,
        "epochs": epoch_metrics,
        "candidate_actor_state_sha256_after": module_sha256(candidate),
        "frozen_champion_actor_state_sha256": champion_hash_before,
    }


def run_sweep(args: argparse.Namespace) -> dict:
    champion_path = args.incumbent.resolve()
    if file_sha256(champion_path) != CHAMPION_SHA256:
        raise RuntimeError("refusing to train against an unverified champion")
    if min(
        args.iterations,
        args.deal_pairs,
        args.epochs,
        args.batch_size,
        args.max_turns,
        args.torch_threads,
    ) < 1:
        raise ValueError("best-response sweep sizes must be positive")
    if args.learning_rate <= 0 or not np.isfinite(args.learning_rate):
        raise ValueError("learning rate must be finite and positive")
    variants = tuple(args.variant or DEFAULT_VARIANTS)
    if len({variant.name for variant in variants}) != len(variants):
        raise ValueError("best-response variant names must be unique")
    for variant in variants:
        validate_variant(variant)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    audit_roots = tuple(path.resolve() for path in args.recorded_seed_root)
    recorded = scan_recorded_seed_interval(
        audit_roots,
        start=args.seed_start,
        end=DEVELOPMENT_SEED_LIMIT,
        allow_sealed_results=True,
    )
    seed_plan = allocate_fresh_seed_plan(
        requested_start=args.seed_start,
        deal_count=args.iterations * args.deal_pairs,
        policy_seed_count=args.iterations,
        shuffle_seed_count=args.iterations,
        recorded_seeds=(use.seed for use in recorded),
    )
    torch.set_num_threads(args.torch_threads)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and not args.cpu_only
        else "cpu"
    )
    champion_state = torch.load(
        champion_path, map_location="cpu", weights_only=True
    )
    champion = StudentNet().to(device)
    champion.load_state_dict(champion_state)
    champion.eval()
    for parameter in champion.parameters():
        parameter.requires_grad_(False)
    champion_actor_hash = module_sha256(champion)

    sweeps = []
    started = time.perf_counter()
    for variant in variants:
        seed_everything(seed_plan["training_seed"])
        candidate = StudentNet().to(device)
        candidate.load_state_dict(champion_state)
        candidate.eval()
        if module_sha256(candidate) != champion_actor_hash:
            raise RuntimeError("candidate did not start from champion state")
        optimizer = torch.optim.Adam(
            candidate.parameters(), lr=args.learning_rate
        )
        iterations = []
        for iteration in range(args.iterations):
            iteration_started = time.perf_counter()
            start = iteration * args.deal_pairs
            stop = start + args.deal_pairs
            deal_seeds = seed_plan["deal_seeds"][start:stop]
            policy_sampling_seed = seed_plan["policy_sampling_seeds"][
                iteration
            ]
            rollout, engine, candidate_seats = (
                collect_fixed_opponent_response(
                    candidate,
                    champion,
                    deal_seeds=deal_seeds,
                    policy_sampling_seed=policy_sampling_seed,
                    temperature=args.temperature,
                    max_turns=args.max_turns,
                    device=device,
                )
            )
            rewards = terminal_only_rewards(
                rollout.actions, rollout.rewards
            )
            payoff_audit = audit_terminal_candidate_payoffs(
                rewards,
                rollout.episode_lengths,
                engine=engine,
                candidate_seats=candidate_seats,
            )
            advantages = exact_terminal_advantages(
                rewards, rollout.episode_lengths
            )
            update = ppo_best_response_update(
                candidate,
                champion,
                rollout.observations,
                rollout.actions,
                rollout.log_probabilities,
                advantages,
                optimizer=optimizer,
                temperature=args.temperature,
                clip=args.clip,
                entropy_coefficient=variant.entropy_coefficient,
                champion_kl_coefficient=(
                    variant.champion_kl_coefficient
                ),
                epochs=args.epochs,
                batch_size=args.batch_size,
                shuffle_seed=seed_plan["shuffle_seeds"][iteration],
            )
            if module_sha256(champion) != champion_actor_hash:
                raise RuntimeError("champion changed between iterations")
            iterations.append(
                {
                    "iteration": iteration + 1,
                    "deal_seeds": recorded_deal_seeds(deal_seeds),
                    "policy_sampling_seed": policy_sampling_seed,
                    "shuffle_seed": seed_plan["shuffle_seeds"][iteration],
                    "seat_rotation": (
                        "each deal candidate seat0 then candidate seat1"
                    ),
                    "rollout": {
                        "completed_games": rollout.completed_games,
                        "policy_decisions": rollout.policy_decisions,
                        "environment_turns": rollout.environment_turns,
                        "inference_calls": rollout.inference_calls,
                    },
                    "payoff_audit": payoff_audit,
                    "update": update,
                    "wall_seconds": time.perf_counter()
                    - iteration_started,
                }
            )
        checkpoint_path = output_dir / f"best_response_{variant.name}.pth"
        torch.save(candidate.state_dict(), checkpoint_path)
        sweeps.append(
            {
                "variant": asdict(variant),
                "initial_actor_state_sha256": champion_actor_hash,
                "iterations": iterations,
                "checkpoint": {
                    "path": checkpoint_path.name,
                    "sha256": file_sha256(checkpoint_path),
                    "actor_state_sha256": module_sha256(candidate),
                },
            }
        )

    manifest = {
        "schema_version": 1,
        "method": "finite learned PPO approximate best response",
        "status": "final",
        "development_only": True,
        "sealed_seeds_or_outcomes_used": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": {
            "claim": (
                "diagnostic lower bound from responses found by finite PPO"
            ),
            "not_claimed": (
                "exact best response, exploitability certificate, or "
                "exact exploitability or production promotion"
            ),
            "overfitting_warning": (
                "the response may overfit the frozen raw incumbent; it must "
                "pass raw direct and fixed-pool gates, then fresh exact "
                "projected-vs-projected production gates"
            ),
        },
        "references": {
            "ppo": PPO_PAPER_URL,
            "policy_space_response_oracles": PSRO_PAPER_URL,
        },
        "fixed_raw_incumbent": {
            "path": str(champion_path),
            "sha256": file_sha256(champion_path),
            "actor_state_sha256": champion_actor_hash,
            "projected": False,
            "frozen": True,
        },
        "observation_contract": {
            "size": OBSERVATION_SIZE,
            "candidate_only_collection": True,
            "opponent_private_state_visible": False,
            "draw_pile_order_visible": False,
        },
        "reward": {
            "terminal_zero_sum_win_draw_loss_only": True,
            "discount": 1.0,
            "collector_pass_penalty_removed": True,
        },
        "protocol": {
            "iterations": args.iterations,
            "deal_pairs_per_iteration": args.deal_pairs,
            "games_per_iteration": 2 * args.deal_pairs,
            "temperature": args.temperature,
            "ppo_clip": args.clip,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "maximum_turns": args.max_turns,
            "device": str(device),
            "torch_threads": args.torch_threads,
        },
        "seed_audit": {
            **seed_plan,
            "deal_seeds": recorded_deal_seeds(seed_plan["deal_seeds"]),
            "recorded_seed_roots": [str(path) for path in audit_roots],
            "common_across_variants": True,
            "both_seats_per_deal": True,
        },
        "sweeps": sweeps,
        "wall_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "best_response_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--incumbent", type=Path, default=DEFAULT_INCUMBENT
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--variant",
        action="append",
        type=parse_variant,
        help=(
            "NAME:CHAMPION_KL:ENTROPY; repeat for a custom sweep. "
            "Defaults to a four-way 2x2 stability grid."
        ),
    )
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--deal-pairs", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--clip", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument(
        "--seed-start",
        type=int,
        default=90_500_001,
        help="requested start of the explicit sub-92m development plan",
    )
    parser.add_argument(
        "--recorded-seed-root",
        action="append",
        type=Path,
        default=None,
        help="artifact root scanned for prior seed use; repeat as needed",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=DEFAULT_TORCH_THREADS
    )
    parser.add_argument("--cpu-only", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    if args.recorded_seed_root is None:
        args.recorded_seed_root = [Path("benchmarks"), Path("experiments")]
    manifest = run_sweep(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "variants": [
                    sweep["variant"]["name"]
                    for sweep in manifest["sweeps"]
                ],
                "seed_audit": {
                    key: manifest["seed_audit"][key]
                    for key in (
                        "requested_start",
                        "minimum_seed",
                        "maximum_seed",
                        "skipped_recorded_seeds",
                        "all_seeds_below_92m",
                    )
                },
                "wall_seconds": manifest["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
