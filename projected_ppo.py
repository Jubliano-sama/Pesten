#!/usr/bin/env python3
"""Exact 24-suit mean-logit PPO continuation from the production champion.

The same differentiable :class:`BatchedSuitAveragedActor` instance is used for
rollout behavior and PPO updates.  Only its shared base actor is optimized and
saved; every update gradient flows through the observation permutation,
base-network evaluation, inverse action permutation, and 24-way arithmetic
mean.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.distributions import Categorical

from ai_observation import OBSERVATION_SIZE
from anchored_ppo import (
    CHAMPION_SHA256,
    enforce_initial_ratio_invariant,
    file_sha256,
)
from batched_suit_symmetry import BatchedSuitAveragedActor
from nashpg_ppo import (
    SELF_PLAY_AGENT_SETS,
    assert_two_player_zero_sum,
    generate_audited_self_play_data,
    module_sha256,
    seed_everything,
    terminal_zero_sum_rewards,
    validate_development_seed_plan,
)
from ppo2 import PPO_Supervised, compute_gae
from seed_integrity import DEVELOPMENT_SEED_LIMIT
from trpo_selfplay import masked_old_new_kl


DEFAULT_SWEEP = (
    {
        "name": "kl0p2_lr1em6_ent0",
        "learning_rate": 1e-6,
        "kl_coefficient": 0.2,
        "entropy_coefficient": 0.0,
        "target_kl": 5e-4,
    },
    {
        "name": "kl0p2_lr3em6_ent0",
        "learning_rate": 3e-6,
        "kl_coefficient": 0.2,
        "entropy_coefficient": 0.0,
        "target_kl": 1e-3,
    },
    {
        "name": "kl0p5_lr3em6_ent1em3",
        "learning_rate": 3e-6,
        "kl_coefficient": 0.5,
        "entropy_coefficient": 1e-3,
        "target_kl": 1e-3,
    },
)


def configure_projected_ppo(
    champion: Path,
    *,
    learning_rate: float,
    rollout_batch_size: int,
    base_chunk_size: int,
    torch_threads: int,
) -> tuple[PPO_Supervised, torch.nn.Module]:
    """Build PPO whose rollout and update actor are one projected wrapper."""
    if file_sha256(champion) != CHAMPION_SHA256:
        raise RuntimeError("production champion SHA-256 mismatch")
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    ppo = PPO_Supervised(
        use_pretrained=True,
        actor_path=str(champion),
        critic_path="missing_projected_critic.pt",
        actor_opt_path="missing_projected_actor_optimizer.pt",
        critic_opt_path="missing_projected_critic_optimizer.pt",
        old_save=(),
        max_saves=0,
        rollout_device="cpu",
        rollout_batch_size=rollout_batch_size,
        training_actor_device="cpu",
        compile_updates=False,
        past_policy_probability=0.0,
        torch_threads=torch_threads,
    )
    base_actor = ppo.actor
    projected_actor = BatchedSuitAveragedActor(
        base_actor,
        alpha=1.0,
        aggregation="logits",
        player_orbit_mode="none",
        base_chunk_size=base_chunk_size,
    )
    # PPO_Supervised created its optimizer over base_actor.parameters().
    # Replacing all three policy references with the same wrapper preserves
    # those parameter objects while making both behavior and updates projected.
    ppo.actor = projected_actor
    ppo.rollout_actor = projected_actor
    ppo._actor_update_model = projected_actor
    for group in ppo.actor_optim.param_groups:
        group["lr"] = learning_rate
    for group in ppo.critic_optim.param_groups:
        group["lr"] = learning_rate
    optimized = {
        id(parameter)
        for group in ppo.actor_optim.param_groups
        for parameter in group["params"]
    }
    base_parameters = {id(parameter) for parameter in base_actor.parameters()}
    if optimized != base_parameters:
        raise RuntimeError("actor optimizer does not own exactly the base actor")
    if not (
        ppo.actor is ppo.rollout_actor
        and ppo.actor is ppo._actor_update_model
    ):
        raise RuntimeError("projected rollout/update policy identity failed")
    return ppo, base_actor


def _chunked_projected_logits(
    actor: torch.nn.Module,
    observations: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(observations), chunk_size):
        parts.append(
            actor(observations[start : start + chunk_size]) / temperature
        )
    return torch.cat(parts)


def _chunked_values(
    ppo: PPO_Supervised,
    observations: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(observations), chunk_size):
        parts.append(
            ppo.critic(
                observations[start : start + chunk_size].to(ppo.device)
            ).squeeze(-1)
        )
    return torch.cat(parts)


def common_batch_sha256(
    observations: torch.Tensor,
    actions: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    rewards: torch.Tensor,
    lengths: list[int],
) -> str:
    digest = hashlib.sha256()
    for name, tensor in (
        ("observations", observations),
        ("actions", actions),
        ("old_log_probabilities", old_log_probabilities),
        ("rewards", rewards),
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    digest.update(np.asarray(lengths, dtype=np.int64).tobytes())
    return digest.hexdigest()


def warm_projected_critic(
    ppo: PPO_Supervised,
    *,
    games: int,
    epochs: int,
    minibatch_size: int,
    temperature: float,
) -> dict:
    """Warm the critic only, from exact projected terminal self-play."""
    actor_before = module_sha256(ppo.actor.actor)
    data, deal_audit = generate_audited_self_play_data(
        ppo,
        episodes=games,
        temperature=temperature,
        agent_sets=SELF_PLAY_AGENT_SETS,
    )
    observations, actions, _, rewards, lengths = data
    terminal_rewards = terminal_zero_sum_rewards(actions, rewards)
    payoff_audit = assert_two_player_zero_sum(
        terminal_rewards,
        lengths,
        expected_games=ppo.last_rollout_metrics["completed_games"],
    )
    zeros = torch.zeros_like(terminal_rewards)
    _, targets = compute_gae(
        terminal_rewards,
        zeros,
        episode_lengths=lengths,
        gamma_=0.97,
        lam=1.0,
    )
    generator = torch.Generator().manual_seed(86_100_001)
    losses = []
    ppo.critic.train()
    for _ in range(epochs):
        order = torch.randperm(len(observations), generator=generator)
        for start in range(0, len(order), minibatch_size):
            indices = order[start : start + minibatch_size]
            values = ppo.critic(observations[indices]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(values, targets[indices])
            ppo.critic_optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.critic.parameters(), 1.0)
            ppo.critic_optim.step()
            losses.append(float(loss.detach()))
    ppo.critic.eval()
    if module_sha256(ppo.actor.actor) != actor_before:
        raise RuntimeError("critic warm-up changed the projected base actor")
    return {
        "games": games,
        "decisions": len(actions),
        "loss_mean": float(np.mean(losses)),
        "rollout": {**ppo.last_rollout_metrics, **deal_audit},
        "payoff_audit": payoff_audit,
    }


def collect_common_projected_batch(
    ppo: PPO_Supervised,
    *,
    games: int,
    temperature: float,
    diagnostic_chunk_size: int,
) -> dict:
    """Collect one projected-champion batch reused by every sweep candidate."""
    data, deal_audit = generate_audited_self_play_data(
        ppo,
        episodes=games,
        temperature=temperature,
        agent_sets=SELF_PLAY_AGENT_SETS,
    )
    observations, actions, old_log_probs, rewards, lengths = data
    terminal_rewards = terminal_zero_sum_rewards(actions, rewards)
    payoff_audit = assert_two_player_zero_sum(
        terminal_rewards,
        lengths,
        expected_games=ppo.last_rollout_metrics["completed_games"],
    )
    masks = ppo.get_action_masks(observations)
    with torch.inference_mode():
        old_logits = _chunked_projected_logits(
            ppo.actor,
            observations,
            temperature=temperature,
            chunk_size=diagnostic_chunk_size,
        )
        floor = torch.finfo(old_logits.dtype).min
        recomputed = Categorical(
            logits=old_logits.masked_fill(~masks, floor),
            validate_args=False,
        ).log_prob(actions)
        ratio_error = enforce_initial_ratio_invariant(
            (recomputed - old_log_probs).exp()
        )
        old_values = _chunked_values(
            ppo,
            observations,
            chunk_size=diagnostic_chunk_size,
        )
        advantages, returns = compute_gae(
            terminal_rewards,
            old_values,
            episode_lengths=lengths,
            gamma_=0.97,
            lam=0.95,
        )
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)
    return {
        "observations": observations.detach().clone(),
        "actions": actions.detach().clone(),
        "old_log_probabilities": old_log_probs.detach().clone(),
        "old_logits": old_logits.detach().clone(),
        "legal_actions": masks.detach().clone(),
        "advantages": advantages.detach().clone(),
        "returns": returns.detach().clone(),
        "terminal_rewards": terminal_rewards.detach().clone(),
        "episode_lengths": list(lengths),
        "sha256": common_batch_sha256(
            observations,
            actions,
            old_log_probs,
            terminal_rewards,
            lengths,
        ),
        "max_initial_ratio_error": ratio_error,
        "rollout": {**ppo.last_rollout_metrics, **deal_audit},
        "payoff_audit": payoff_audit,
    }


def train_common_batch_candidate(
    ppo: PPO_Supervised,
    base_actor: torch.nn.Module,
    batch: dict,
    config: dict,
    *,
    update_epochs: int,
    minibatch_size: int,
    temperature: float,
    permutation_seed: int,
    diagnostic_chunk_size: int,
) -> dict:
    """Update one projected actor from the immutable common rollout batch."""
    if (
        config["target_kl"] <= 0
        or config["kl_coefficient"] < 0
        or config["entropy_coefficient"] < 0
        or not all(
            np.isfinite(config[key])
            for key in (
                "target_kl",
                "kl_coefficient",
                "entropy_coefficient",
            )
        )
    ):
        raise ValueError("invalid projected PPO regularization configuration")
    observations = batch["observations"]
    actions = batch["actions"]
    old_log_probs = batch["old_log_probabilities"]
    old_logits = batch["old_logits"]
    masks = batch["legal_actions"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    base_before = module_sha256(base_actor)
    projected_before = module_sha256(ppo.actor)
    with torch.inference_mode():
        initial_logits = _chunked_projected_logits(
            ppo.actor,
            observations,
            temperature=temperature,
            chunk_size=diagnostic_chunk_size,
        )
        floor = torch.finfo(initial_logits.dtype).min
        initial_selected = Categorical(
            logits=initial_logits.masked_fill(~masks, floor),
            validate_args=False,
        ).log_prob(actions)
        max_ratio_error = enforce_initial_ratio_invariant(
            (initial_selected - old_log_probs).exp()
        )
        initial_exact_kl = float(
            masked_old_new_kl(old_logits, initial_logits, masks)
        )
    if initial_exact_kl != 0.0:
        raise RuntimeError("candidate did not start from exact behavior policy")

    generator = torch.Generator().manual_seed(permutation_seed)
    epoch_logs = []
    peak_augmented_minibatch = 0
    accepted_epochs = 0
    for epoch in range(update_epochs):
        epoch_base_state = copy.deepcopy(base_actor.state_dict())
        epoch_critic_state = copy.deepcopy(ppo.critic.state_dict())
        epoch_actor_optim_state = copy.deepcopy(ppo.actor_optim.state_dict())
        epoch_critic_optim_state = copy.deepcopy(
            ppo.critic_optim.state_dict()
        )
        order = torch.randperm(len(actions), generator=generator)
        metrics = {
            "objective": [],
            "clipped_surrogate": [],
            "behavior_kl_old_to_new": [],
            "entropy": [],
            "critic_loss": [],
            "clip_fraction": [],
        }
        for start in range(0, len(order), minibatch_size):
            indices = order[start : start + minibatch_size]
            obs = observations[indices]
            legal = masks[indices]
            new_logits = ppo.actor(obs) / temperature
            peak_augmented_minibatch = max(
                peak_augmented_minibatch,
                len(obs) * 24,
            )
            floor = torch.finfo(new_logits.dtype).min
            distribution = Categorical(
                logits=new_logits.masked_fill(~legal, floor),
                validate_args=False,
            )
            new_selected = distribution.log_prob(actions[indices])
            ratios = (new_selected - old_log_probs[indices]).exp()
            unclipped = ratios * advantages[indices]
            clipped = torch.clamp(
                ratios,
                1.0 - ppo.clip,
                1.0 + ppo.clip,
            ) * advantages[indices]
            surrogate = torch.minimum(unclipped, clipped).mean()
            behavior_kl = masked_old_new_kl(
                old_logits[indices],
                new_logits,
                legal,
            )
            entropy = distribution.entropy().mean()
            values = ppo.critic(obs).squeeze(-1)
            critic_loss = torch.nn.functional.mse_loss(
                values,
                returns[indices],
            )
            objective = (
                surrogate
                - config["kl_coefficient"] * behavior_kl
                + config["entropy_coefficient"] * entropy
            )
            loss = -objective + 0.8 * critic_loss
            ppo.actor_optim.zero_grad(set_to_none=True)
            ppo.critic_optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(ppo.critic.parameters(), 1.0)
            ppo.actor_optim.step()
            ppo.critic_optim.step()
            metrics["objective"].append(float(objective.detach()))
            metrics["clipped_surrogate"].append(float(surrogate.detach()))
            metrics["behavior_kl_old_to_new"].append(
                float(behavior_kl.detach())
            )
            metrics["entropy"].append(float(entropy.detach()))
            metrics["critic_loss"].append(float(critic_loss.detach()))
            metrics["clip_fraction"].append(
                float(((ratios - 1).abs() > ppo.clip).float().mean())
            )
        with torch.inference_mode():
            post_epoch_logits = _chunked_projected_logits(
                ppo.actor,
                observations,
                temperature=temperature,
                chunk_size=diagnostic_chunk_size,
            )
            post_epoch_full_kl = float(
                masked_old_new_kl(old_logits, post_epoch_logits, masks)
            )
        rolled_back = post_epoch_full_kl > config["target_kl"]
        epoch_log = {
            "epoch": epoch + 1,
            **{
                key: float(np.mean(values))
                for key, values in metrics.items()
            },
            "post_epoch_full_exact_behavior_kl_old_to_new": (
                post_epoch_full_kl
            ),
            "target_kl": float(config["target_kl"]),
            "accepted": not rolled_back,
            "rolled_back": rolled_back,
        }
        epoch_logs.append(epoch_log)
        if rolled_back:
            base_actor.load_state_dict(epoch_base_state)
            ppo.critic.load_state_dict(epoch_critic_state)
            ppo.actor_optim.load_state_dict(epoch_actor_optim_state)
            ppo.critic_optim.load_state_dict(epoch_critic_optim_state)
            break
        accepted_epochs += 1
    ppo._sync_rollout_actor()
    with torch.inference_mode():
        final_logits = _chunked_projected_logits(
            ppo.actor,
            observations,
            temperature=temperature,
            chunk_size=diagnostic_chunk_size,
        )
        final_kl = float(masked_old_new_kl(old_logits, final_logits, masks))
    base_after = module_sha256(base_actor)
    if final_kl > config["target_kl"]:
        raise RuntimeError("final projected policy exceeds target KL")
    return {
        "config": dict(config),
        "common_batch_sha256": batch["sha256"],
        "max_initial_ratio_error": max_ratio_error,
        "initial_exact_behavior_kl_old_to_new": initial_exact_kl,
        "final_exact_behavior_kl_old_to_new": final_kl,
        "target_kl_satisfied": True,
        "accepted_epochs": accepted_epochs,
        "policy_updated": base_after != base_before,
        "base_actor_sha256_before": base_before,
        "base_actor_sha256_after": base_after,
        "projected_actor_sha256_before": projected_before,
        "update_epochs": epoch_logs,
        "peak_augmented_observations_per_minibatch": (
            peak_augmented_minibatch
        ),
        "projection_statistics": ppo.actor.statistics(),
    }


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
        default=Path("benchmarks/projected_ppo_pilot_dev"),
    )
    parser.add_argument("--seed", type=int, default=84_600_001)
    parser.add_argument("--critic-games", type=int, default=64)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument("--games", type=int, default=64)
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
    validate_development_seed_plan(
        args.seed,
        outer_rounds=1,
        inner_iterations=1,
    )
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
    args.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    seed_everything(args.seed)
    behavior_ppo, behavior_base = configure_projected_ppo(
        args.champion,
        learning_rate=1e-6,
        rollout_batch_size=args.rollout_batch_size,
        base_chunk_size=args.base_chunk_size,
        torch_threads=args.torch_threads,
    )
    champion_actor_sha = module_sha256(behavior_base)
    warmup = warm_projected_critic(
        behavior_ppo,
        games=args.critic_games,
        epochs=args.critic_epochs,
        minibatch_size=args.minibatch_size,
        temperature=args.temperature,
    )
    warm_critic = copy.deepcopy(behavior_ppo.critic.state_dict())
    common = collect_common_projected_batch(
        behavior_ppo,
        games=args.games,
        temperature=args.temperature,
        diagnostic_chunk_size=args.diagnostic_chunk_size,
    )
    if module_sha256(behavior_base) != champion_actor_sha:
        raise RuntimeError("common-data collection changed behavior actor")

    candidates = []
    for candidate_index, config in enumerate(DEFAULT_SWEEP):
        seed_everything(args.seed + 10_000)
        ppo, base_actor = configure_projected_ppo(
            args.champion,
            learning_rate=config["learning_rate"],
            rollout_batch_size=args.rollout_batch_size,
            base_chunk_size=args.base_chunk_size,
            torch_threads=args.torch_threads,
        )
        ppo.critic.load_state_dict(warm_critic)
        ppo.critic.eval()
        result = train_common_batch_candidate(
            ppo,
            base_actor,
            common,
            config,
            update_epochs=args.update_epochs,
            minibatch_size=args.minibatch_size,
            temperature=args.temperature,
            permutation_seed=args.seed + 20_000,
            diagnostic_chunk_size=args.diagnostic_chunk_size,
        )
        if not result["policy_updated"]:
            raise RuntimeError(
                f"{config['name']} retained no policy update within target KL"
            )
        checkpoint_name = f"projected_ppo_{config['name']}.pth"
        checkpoint = args.output_dir / checkpoint_name
        torch.save(base_actor.state_dict(), checkpoint)
        result["checkpoint"] = {
            "path": checkpoint_name,
            "sha256": file_sha256(checkpoint),
            "actor_state_sha256": module_sha256(base_actor),
            "raw_base_requires_24_suit_projection": True,
        }
        result["candidate_index"] = candidate_index
        candidates.append(result)

    public_common = {
        key: value
        for key, value in common.items()
        if not isinstance(value, torch.Tensor)
    }
    payload = {
        "schema_version": 1,
        "algorithm": "exact_24_suit_projected_ppo_common_data_sweep",
        "champion": {
            "path": str(args.champion.resolve()),
            "file_sha256": CHAMPION_SHA256,
            "actor_state_sha256": champion_actor_sha,
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
        },
        "recipe": {
            "critic_games": args.critic_games,
            "critic_epochs": args.critic_epochs,
            "common_games": args.games,
            "update_epochs": args.update_epochs,
            "minibatch_size": args.minibatch_size,
            "temperature": args.temperature,
            "base_chunk_size": args.base_chunk_size,
            "diagnostic_chunk_size": args.diagnostic_chunk_size,
            "rollout_batch_size": args.rollout_batch_size,
            "torch_threads": args.torch_threads,
            "sweep": list(DEFAULT_SWEEP),
        },
        "critic_warmup": warmup,
        "common_rollout": public_common,
        "candidates": candidates,
        "wall_seconds": time.perf_counter() - started,
    }
    manifest = args.output_dir / "projected_ppo_metrics.json"
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "manifest": str(manifest.resolve()),
                "checkpoints": len(candidates),
                "common_batch_sha256": common["sha256"],
                "sealed_outcomes_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
