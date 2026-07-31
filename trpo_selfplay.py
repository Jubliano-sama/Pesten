#!/usr/bin/env python3
"""Development-only true-TRPO pilot for two-player Pesten self-play.

The actor update follows the practical natural-gradient procedure from
Schulman et al., Trust Region Policy Optimization (arXiv:1502.05477):

* optimize the sampled importance-weighted surrogate;
* estimate the Fisher curvature with Hessian-vector products of mean
  KL(old || new);
* solve for the natural-gradient direction with conjugate gradient;
* scale it to the target-KL quadratic trust region; and
* accept only a backtracking step that improves the surrogate and satisfies
  the measured legal-action KL constraint.

Rollouts contain only the existing 121 human-visible features.  All deals and
policy-sampling seeds are explicit development seeds recorded in the manifest.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
from typing import Callable, Iterable

import numpy as np
import torch

from ai_observation import OBSERVATION_SIZE, PASS_ACTION
from anchored_ppo import CHAMPION_SHA256, file_sha256
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    POLICY_RL,
    collect_batched_rollout,
)
from ppo2 import DEFAULT_TORCH_THREADS, StudentNet, build_action_masks, compute_gae
from seed_integrity import DEVELOPMENT_SEED_LIMIT


PAPER_URL = "https://arxiv.org/abs/1502.05477v5"
SELF_PLAY_ACTOR_KEY = "current"
DEFAULT_CHAMPION = Path("ppo_actor_100run3.pth")
DEFAULT_TARGET_KLS = (0.001, 0.005, 0.01)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_development_seeds(seeds: Iterable[int]) -> None:
    for seed in seeds:
        if seed < 0 or seed >= DEVELOPMENT_SEED_LIMIT:
            raise ValueError("TRPO pilot seeds must stay development-only")


def recorded_deal_seeds(seeds: Iterable[int]) -> list[dict[str, int]]:
    """Render seeds in the recursive scanner's key-visible artifact shape."""
    return [{"deal_seed": int(seed)} for seed in seeds]


def module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def trainable_parameters(module: torch.nn.Module) -> tuple[torch.nn.Parameter, ...]:
    return tuple(parameter for parameter in module.parameters() if parameter.requires_grad)


def flat_parameters(parameters: Iterable[torch.nn.Parameter]) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


def set_flat_parameters(
    parameters: Iterable[torch.nn.Parameter],
    values: torch.Tensor,
) -> None:
    parameters = tuple(parameters)
    if sum(parameter.numel() for parameter in parameters) != values.numel():
        raise ValueError("flat parameter vector has the wrong size")
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.copy_(values[offset : offset + count].view_as(parameter))
            offset += count


def flat_gradient(
    output: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    *,
    create_graph: bool,
) -> torch.Tensor:
    parameters = tuple(parameters)
    gradients = torch.autograd.grad(
        output,
        parameters,
        create_graph=create_graph,
        retain_graph=create_graph,
    )
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def masked_old_new_kl(
    old_logits: torch.Tensor,
    new_logits: torch.Tensor,
    legal_actions: torch.Tensor,
) -> torch.Tensor:
    """Return mean KL(old || new) over each observation's legal actions."""
    if old_logits.shape != new_logits.shape:
        raise ValueError("old and new logits must have equal shape")
    if legal_actions.shape != old_logits.shape:
        raise ValueError("legal mask must match logits")
    if legal_actions.dtype != torch.bool:
        raise ValueError("legal mask must be boolean")
    if not bool(legal_actions.any(dim=-1).all()):
        raise ValueError("every observation must have a legal action")
    if not old_logits.dtype.is_floating_point:
        raise ValueError("logits must be floating point")
    floor = torch.finfo(old_logits.dtype).min
    old_log_probabilities = torch.log_softmax(
        old_logits.masked_fill(~legal_actions, floor),
        dim=-1,
    )
    new_log_probabilities = torch.log_softmax(
        new_logits.masked_fill(~legal_actions, floor),
        dim=-1,
    )
    old_probabilities = old_log_probabilities.exp()
    return (
        old_probabilities
        * (old_log_probabilities - new_log_probabilities)
    ).sum(dim=-1).mean()


def conjugate_gradient(
    matrix_vector_product: Callable[[torch.Tensor], torch.Tensor],
    right_hand_side: torch.Tensor,
    *,
    iterations: int,
    residual_tolerance: float,
) -> tuple[torch.Tensor, dict]:
    """Approximately solve Ax=b for symmetric positive-definite A."""
    if iterations < 1 or residual_tolerance < 0:
        raise ValueError("invalid conjugate-gradient configuration")
    solution = torch.zeros_like(right_hand_side)
    residual = right_hand_side.clone()
    direction = residual.clone()
    residual_squared = torch.dot(residual, residual)
    initial_residual = float(torch.sqrt(residual_squared).detach())
    completed = 0
    for iteration in range(iterations):
        product = matrix_vector_product(direction)
        denominator = torch.dot(direction, product)
        if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
            break
        step = residual_squared / denominator
        solution = solution + step * direction
        next_residual = residual - step * product
        next_squared = torch.dot(next_residual, next_residual)
        completed = iteration + 1
        residual = next_residual
        if float(torch.sqrt(next_squared).detach()) <= residual_tolerance:
            residual_squared = next_squared
            break
        direction = next_residual + (next_squared / residual_squared) * direction
        residual_squared = next_squared
    return solution, {
        "iterations": completed,
        "initial_residual_norm": initial_residual,
        "final_residual_norm": float(torch.sqrt(residual_squared).detach()),
    }


def _masked_distribution_log_probabilities(
    logits: torch.Tensor,
    legal_actions: torch.Tensor,
) -> torch.Tensor:
    floor = torch.finfo(logits.dtype).min
    return torch.log_softmax(
        logits.masked_fill(~legal_actions, floor),
        dim=-1,
    )


def trpo_actor_step(
    actor: torch.nn.Module,
    observations: torch.Tensor,
    actions: torch.Tensor,
    behavior_log_probabilities: torch.Tensor,
    advantages: torch.Tensor,
    legal_actions: torch.Tensor,
    *,
    temperature: float,
    target_kl: float,
    conjugate_gradient_iterations: int = 10,
    conjugate_gradient_tolerance: float = 1e-10,
    fisher_damping: float = 0.1,
    backtrack_iterations: int = 10,
    backtrack_coefficient: float = 0.5,
    minimum_improvement_ratio: float = 0.1,
) -> dict:
    """Perform one rollback-safe TRPO actor update."""
    if temperature <= 0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    if target_kl <= 0 or not np.isfinite(target_kl):
        raise ValueError("target_kl must be finite and positive")
    if fisher_damping <= 0 or not np.isfinite(fisher_damping):
        raise ValueError("fisher_damping must be finite and positive")
    if backtrack_iterations < 1 or not 0 < backtrack_coefficient < 1:
        raise ValueError("invalid backtracking configuration")
    if minimum_improvement_ratio < 0:
        raise ValueError("minimum improvement ratio cannot be negative")
    if not (
        len(observations)
        == len(actions)
        == len(behavior_log_probabilities)
        == len(advantages)
        == len(legal_actions)
    ):
        raise ValueError("TRPO batch tensors must have equal leading length")
    if observations.shape[-1] != OBSERVATION_SIZE:
        raise ValueError("TRPO actor may only consume the public observation")
    if legal_actions.dtype != torch.bool:
        raise ValueError("legal mask must be boolean")
    if not bool(legal_actions.gather(1, actions[:, None]).all()):
        raise ValueError("rollout contains an illegal sampled action")

    # Callers may have produced rollout tensors under inference_mode.  Clone
    # the saved scalar targets into ordinary tensors before autograd uses them.
    actions = actions.detach().clone()
    behavior_log_probabilities = behavior_log_probabilities.detach().clone()
    advantages = advantages.detach().clone()
    actor.eval()
    parameters = trainable_parameters(actor)
    original_parameters = flat_parameters(parameters)
    with torch.no_grad():
        old_logits = (actor(observations) / temperature).detach()
        old_log_probabilities = _masked_distribution_log_probabilities(
            old_logits,
            legal_actions,
        ).detach()
        old_selected = old_log_probabilities.gather(
            1,
            actions[:, None],
        ).squeeze(1)
        initial_ratios = torch.exp(
            old_selected - behavior_log_probabilities
        )
        maximum_initial_ratio_error = float(
            initial_ratios.sub(1.0).abs().max()
        )
    if (
        not np.isfinite(maximum_initial_ratio_error)
        or maximum_initial_ratio_error > 0.02
    ):
        raise RuntimeError(
            "rollout behavior and frozen old policy disagree: "
            f"maximum ratio error {maximum_initial_ratio_error:.6f}"
        )

    def surrogate_and_kl() -> tuple[torch.Tensor, torch.Tensor]:
        new_logits = actor(observations) / temperature
        new_log_probabilities = _masked_distribution_log_probabilities(
            new_logits,
            legal_actions,
        )
        selected = new_log_probabilities.gather(1, actions[:, None]).squeeze(1)
        ratios = torch.exp(selected - behavior_log_probabilities)
        surrogate = (ratios * advantages).mean()
        old_probabilities = old_log_probabilities.exp()
        kl = (
            old_probabilities
            * (old_log_probabilities - new_log_probabilities)
        ).sum(dim=-1).mean()
        return surrogate, kl

    surrogate_before_tensor, kl_before_tensor = surrogate_and_kl()
    policy_gradient = flat_gradient(
        surrogate_before_tensor,
        parameters,
        create_graph=False,
    ).detach()
    surrogate_before = float(surrogate_before_tensor.detach())
    kl_before = float(kl_before_tensor.detach())
    gradient_norm = float(policy_gradient.norm())
    if not np.isfinite(gradient_norm) or gradient_norm == 0:
        set_flat_parameters(parameters, original_parameters)
        return {
            "accepted": False,
            "reason": "zero_or_nonfinite_policy_gradient",
            "surrogate_before": surrogate_before,
            "surrogate_after": surrogate_before,
            "kl_before": kl_before,
            "kl_after": kl_before,
            "policy_gradient_norm": gradient_norm,
            "maximum_initial_ratio_error": maximum_initial_ratio_error,
        }

    def fisher_vector_product(vector: torch.Tensor) -> torch.Tensor:
        _, kl = surrogate_and_kl()
        kl_gradient = flat_gradient(kl, parameters, create_graph=True)
        directional_derivative = torch.dot(kl_gradient, vector)
        hessian_vector = flat_gradient(
            directional_derivative,
            parameters,
            create_graph=False,
        ).detach()
        return hessian_vector + fisher_damping * vector

    natural_direction, cg_log = conjugate_gradient(
        fisher_vector_product,
        policy_gradient,
        iterations=conjugate_gradient_iterations,
        residual_tolerance=conjugate_gradient_tolerance,
    )
    fisher_direction = fisher_vector_product(natural_direction)
    quadratic = float(
        (0.5 * torch.dot(natural_direction, fisher_direction)).detach()
    )
    if not np.isfinite(quadratic) or quadratic <= 0:
        set_flat_parameters(parameters, original_parameters)
        return {
            "accepted": False,
            "reason": "nonpositive_or_nonfinite_curvature",
            "surrogate_before": surrogate_before,
            "surrogate_after": surrogate_before,
            "kl_before": kl_before,
            "kl_after": kl_before,
            "policy_gradient_norm": gradient_norm,
            "maximum_initial_ratio_error": maximum_initial_ratio_error,
            "conjugate_gradient": cg_log,
            "quadratic_model": quadratic,
        }
    full_step = natural_direction * np.sqrt(target_kl / quadratic)
    expected_full_improvement = float(torch.dot(policy_gradient, full_step))

    attempts = []
    accepted = False
    accepted_fraction = 0.0
    surrogate_after = surrogate_before
    kl_after = kl_before
    for backtrack_index in range(backtrack_iterations):
        fraction = backtrack_coefficient**backtrack_index
        candidate = original_parameters + fraction * full_step
        set_flat_parameters(parameters, candidate)
        with torch.inference_mode():
            candidate_surrogate_tensor, candidate_kl_tensor = surrogate_and_kl()
        candidate_surrogate = float(candidate_surrogate_tensor)
        candidate_kl = float(candidate_kl_tensor)
        actual_improvement = candidate_surrogate - surrogate_before
        expected_improvement = fraction * expected_full_improvement
        improvement_ratio = (
            actual_improvement / expected_improvement
            if expected_improvement > 0
            else float("-inf")
        )
        finite = (
            np.isfinite(candidate_surrogate)
            and np.isfinite(candidate_kl)
            and np.isfinite(improvement_ratio)
        )
        satisfies = (
            finite
            and actual_improvement > 0
            and improvement_ratio >= minimum_improvement_ratio
            and candidate_kl <= target_kl
        )
        attempts.append(
            {
                "index": backtrack_index,
                "fraction": fraction,
                "surrogate": candidate_surrogate,
                "actual_improvement": actual_improvement,
                "expected_improvement": expected_improvement,
                "improvement_ratio": improvement_ratio,
                "kl_old_to_new": candidate_kl,
                "accepted": satisfies,
            }
        )
        if satisfies:
            accepted = True
            accepted_fraction = fraction
            surrogate_after = candidate_surrogate
            kl_after = candidate_kl
            break
    if not accepted:
        set_flat_parameters(parameters, original_parameters)
        with torch.inference_mode():
            restored_surrogate, restored_kl = surrogate_and_kl()
        surrogate_after = float(restored_surrogate)
        kl_after = float(restored_kl)

    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "line_search_rejected",
        "target_kl": target_kl,
        "surrogate_before": surrogate_before,
        "surrogate_after": surrogate_after,
        "actual_improvement": surrogate_after - surrogate_before,
        "kl_before": kl_before,
        "kl_after": kl_after,
        "policy_gradient_norm": gradient_norm,
        "maximum_initial_ratio_error": maximum_initial_ratio_error,
        "conjugate_gradient": cg_log,
        "fisher_damping": fisher_damping,
        "quadratic_model": quadratic,
        "expected_full_improvement": expected_full_improvement,
        "accepted_step_fraction": accepted_fraction,
        "backtracking": attempts,
        "rollback_verified": (
            accepted
            or torch.equal(flat_parameters(parameters), original_parameters)
        ),
    }


def terminal_zero_sum_rewards(
    actions: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    if actions.shape != rewards.shape:
        raise ValueError("actions and rewards must have equal shape")
    return rewards + 0.1 * (actions == PASS_ACTION).to(rewards.dtype)


def audit_terminal_payoffs(
    rewards: torch.Tensor,
    episode_lengths: list[int],
    *,
    expected_games: int,
    atol: float = 1e-6,
) -> dict:
    populated = len(episode_lengths)
    if not expected_games <= populated <= 2 * expected_games:
        raise RuntimeError("self-play rollout has invalid seat cardinality")
    if any(length <= 0 for length in episode_lengths):
        raise RuntimeError("self-play episodes must be populated")
    offset = 0
    returns = []
    for length in episode_lengths:
        returns.append(rewards[offset : offset + length].sum())
        offset += length
    if offset != len(rewards):
        raise RuntimeError("episode lengths do not cover rewards")
    terminal_returns = torch.stack(returns)
    valid = torch.stack(
        [
            torch.isclose(
                terminal_returns,
                terminal_returns.new_full((), value),
                atol=atol,
                rtol=0,
            )
            for value in (-1.0, 0.0, 1.0)
        ]
    ).any(dim=0)
    if not bool(valid.all()):
        raise RuntimeError("episode return is not terminal win/draw/loss")
    full_pair = populated == 2 * expected_games
    if full_pair:
        pair_sums = terminal_returns.reshape(-1, 2).sum(dim=1)
        if not torch.allclose(
            pair_sums,
            torch.zeros_like(pair_sums),
            atol=atol,
            rtol=0,
        ):
            raise RuntimeError("full self-play seat pairs are not zero-sum")
    return {
        "populated_episodes": populated,
        "implicit_zero_decision_seats": 2 * expected_games - populated,
        "terminal_payoffs_verified": True,
        "full_pair_zero_sum_verified": full_pair,
    }


def monte_carlo_advantages(
    rewards: torch.Tensor,
    episode_lengths: list[int],
    *,
    gamma: float,
) -> torch.Tensor:
    zeros = torch.zeros_like(rewards)
    _, returns = compute_gae(
        rewards,
        zeros,
        episode_lengths=episode_lengths,
        gamma_=gamma,
        lam=1.0,
    )
    return (returns - returns.mean()) / (
        returns.std(unbiased=False) + 1e-8
    )


def collect_explicit_seed_selfplay(
    actor: torch.nn.Module,
    *,
    deal_seeds: list[int],
    policy_sampling_seed: int,
    temperature: float,
    max_turns: int,
) -> tuple:
    validate_development_seeds([*deal_seeds, policy_sampling_seed])
    if len(set(deal_seeds)) != len(deal_seeds):
        raise ValueError("deal seeds must be unique within an iteration")
    seed_everything(policy_sampling_seed)
    games = len(deal_seeds)
    engine = BatchedPesten(
        np.full((games, 2), POLICY_RL, dtype=np.int8),
        np.full((games, 2), SELF_PLAY_ACTOR_KEY, dtype=object),
        np.asarray(deal_seeds, dtype=np.uint32),
        max_turns=max_turns,
    )
    actor.eval()
    rollout = collect_batched_rollout(
        engine,
        {
            SELF_PLAY_ACTOR_KEY: ActorPolicy(
                actor=actor,
                device=torch.device("cpu"),
                temperature=temperature,
                collect_data=True,
            )
        },
        output_device=torch.device("cpu"),
    )
    if rollout.completed_games != games:
        raise RuntimeError("TRPO rollout did not complete every requested game")
    return rollout, engine


def target_kl_slug(value: float) -> str:
    return format(float(value), "g").replace("-", "m").replace(".", "p")


def run_pilot(args: argparse.Namespace) -> dict:
    champion = args.champion.resolve()
    if file_sha256(champion) != CHAMPION_SHA256:
        raise RuntimeError("refusing to train from an unverified champion")
    if not args.target_kls or any(
        value <= 0 or not np.isfinite(value) for value in args.target_kls
    ):
        raise ValueError("target KL values must be finite and positive")
    if min(
        args.iterations,
        args.games,
        args.max_turns,
        args.cg_iterations,
        args.backtrack_iterations,
        args.torch_threads,
    ) < 1:
        raise ValueError("TRPO pilot sizes must be positive")
    maximum_deal_seed = (
        args.deal_seed_start + args.iterations * args.games - 1
    )
    maximum_policy_seed = args.policy_seed_start + args.iterations - 1
    validate_development_seeds(
        [
            args.training_seed,
            args.deal_seed_start,
            maximum_deal_seed,
            args.policy_seed_start,
            maximum_policy_seed,
        ]
    )
    torch.set_num_threads(args.torch_threads)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu_only else "cpu"
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    champion_state = torch.load(
        champion,
        map_location="cpu",
        weights_only=True,
    )

    sweeps = []
    for target_kl in args.target_kls:
        seed_everything(args.training_seed)
        actor = StudentNet().to(device)
        actor.load_state_dict(champion_state)
        actor.eval()
        rollout_actor = copy.deepcopy(actor).to("cpu").eval()
        for parameter in rollout_actor.parameters():
            parameter.requires_grad_(False)
        sweep = {
            "target_kl": float(target_kl),
            "initial_actor_state_sha256": module_sha256(actor),
            "iterations": [],
        }
        for iteration in range(args.iterations):
            started = time.perf_counter()
            first_deal_seed = (
                args.deal_seed_start + iteration * args.games
            )
            deal_seeds = list(
                range(first_deal_seed, first_deal_seed + args.games)
            )
            policy_seed = args.policy_seed_start + iteration
            rollout_actor.load_state_dict(
                {
                    name: value.detach().cpu()
                    for name, value in actor.state_dict().items()
                }
            )
            rollout, engine = collect_explicit_seed_selfplay(
                rollout_actor,
                deal_seeds=deal_seeds,
                policy_sampling_seed=policy_seed,
                temperature=args.temperature,
                max_turns=args.max_turns,
            )
            training_rewards = terminal_zero_sum_rewards(
                rollout.actions,
                rollout.rewards,
            )
            payoff_audit = audit_terminal_payoffs(
                training_rewards,
                rollout.episode_lengths,
                expected_games=rollout.completed_games,
            )
            observations = rollout.observations.to(device)
            actions = rollout.actions.to(device)
            behavior_log_probabilities = rollout.log_probabilities.to(device)
            legal_actions = build_action_masks(observations)
            advantages = monte_carlo_advantages(
                training_rewards.to(device),
                rollout.episode_lengths,
                gamma=args.gamma,
            )
            actor_before = module_sha256(actor)
            step_log = trpo_actor_step(
                actor,
                observations,
                actions,
                behavior_log_probabilities,
                advantages,
                legal_actions,
                temperature=args.temperature,
                target_kl=target_kl,
                conjugate_gradient_iterations=args.cg_iterations,
                conjugate_gradient_tolerance=args.cg_tolerance,
                fisher_damping=args.fisher_damping,
                backtrack_iterations=args.backtrack_iterations,
                backtrack_coefficient=args.backtrack_coefficient,
                minimum_improvement_ratio=args.minimum_improvement_ratio,
            )
            actor_after = module_sha256(actor)
            if step_log["accepted"] and actor_after == actor_before:
                raise RuntimeError("accepted TRPO step did not change actor")
            if not step_log["accepted"] and actor_after != actor_before:
                raise RuntimeError("rejected TRPO step failed to roll back")
            filename = (
                f"trpo_targetkl_{target_kl_slug(target_kl)}"
                f"_iter_{iteration + 1:02d}.pth"
            )
            checkpoint = output_dir / filename
            torch.save(actor.state_dict(), checkpoint)
            sweep["iterations"].append(
                {
                    "target_kl": float(target_kl),
                    "iteration": iteration + 1,
                    "deal_seeds": recorded_deal_seeds(deal_seeds),
                    "policy_sampling_seed": policy_seed,
                    "rollout": {
                        "completed_games": rollout.completed_games,
                        "environment_turns": rollout.environment_turns,
                        "policy_decisions": rollout.policy_decisions,
                        "inference_calls": rollout.inference_calls,
                        "draws": rollout.draws,
                        "maximum_turns_observed": int(
                            engine.turn_counts.max()
                        ),
                        "truncated_games": int(
                            np.count_nonzero(
                                engine.turn_counts >= args.max_turns
                            )
                        ),
                    },
                    "payoff_audit": payoff_audit,
                    "optimization": step_log,
                    "checkpoint": {
                        "path": filename,
                        "sha256": file_sha256(checkpoint),
                        "actor_state_sha256": actor_after,
                    },
                    "wall_seconds": time.perf_counter() - started,
                }
            )
        sweeps.append(sweep)

    payload = {
        "schema_version": 1,
        "method": "true_trpo_shared_policy_selfplay",
        "status": "complete",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "name": "Trust Region Policy Optimization",
            "url": PAPER_URL,
        },
        "champion": {
            "path": str(champion),
            "sha256": CHAMPION_SHA256,
        },
        "objective": {
            "surrogate": "mean(exp(log_pi_new-log_pi_behavior)*advantage)",
            "curvature": "Hessian of mean legal-mask KL(old||new)",
            "solver": "damped conjugate-gradient natural gradient",
            "acceptance": (
                "backtracking surrogate improvement and measured KL <= target"
            ),
        },
        "privacy": {
            "observation_features": OBSERVATION_SIZE,
            "own_hand_plus_public_state_only": True,
            "opponent_card_identities": False,
            "draw_pile_order": False,
            "shared_symmetric_two_player_policy": True,
        },
        "seed_audit": {
            "training_seed": args.training_seed,
            "deal_seed_schedule": (
                "contiguous per iteration, common across target-KL sweeps"
            ),
            "deal_seed_start": args.deal_seed_start,
            "policy_seed_start": args.policy_seed_start,
            "development_seed_limit": DEVELOPMENT_SEED_LIMIT,
            "development_only": True,
            "sealed_outcomes_used": False,
        },
        "recipe": {
            "target_kls": [float(value) for value in args.target_kls],
            "iterations": args.iterations,
            "games_per_iteration": args.games,
            "temperature": args.temperature,
            "gamma": args.gamma,
            "fisher_damping": args.fisher_damping,
            "conjugate_gradient_iterations": args.cg_iterations,
            "conjugate_gradient_tolerance": args.cg_tolerance,
            "backtrack_iterations": args.backtrack_iterations,
            "backtrack_coefficient": args.backtrack_coefficient,
            "minimum_improvement_ratio": args.minimum_improvement_ratio,
            "maximum_turns": args.max_turns,
            "torch_threads": args.torch_threads,
            "device": str(device),
        },
        "sweeps": sweeps,
    }
    manifest = output_dir / "trpo_selfplay_manifest.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, default=DEFAULT_CHAMPION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--target-kls",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_KLS),
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--training-seed", type=int, default=84_600_001)
    parser.add_argument("--deal-seed-start", type=int, default=84_601_000)
    parser.add_argument("--policy-seed-start", type=int, default=84_602_000)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.97)
    parser.add_argument("--fisher-damping", type=float, default=0.1)
    parser.add_argument("--cg-iterations", type=int, default=10)
    parser.add_argument("--cg-tolerance", type=float, default=1e-10)
    parser.add_argument("--backtrack-iterations", type=int, default=10)
    parser.add_argument("--backtrack-coefficient", type=float, default=0.5)
    parser.add_argument("--minimum-improvement-ratio", type=float, default=0.1)
    parser.add_argument("--max-turns", type=int, default=500)
    parser.add_argument(
        "--torch-threads", type=int, default=DEFAULT_TORCH_THREADS
    )
    parser.add_argument("--cpu-only", action="store_true")
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    payload = run_pilot(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "method": payload["method"],
                "sweeps": len(payload["sweeps"]),
                "sealed_outcomes_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
