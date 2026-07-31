#!/usr/bin/env python3
"""NashPG with PPO inner updates for two-player Pesten self-play.

Implements a shared-symmetric-policy specialization of Yu et al.,
arXiv:2510.18183v2, Equations 11-12 and Algorithm 4.  The maximized actor
objective is exactly

    PPO clipped surrogate - alpha * E_o KL(current || reference).

The reference remains frozen for K fresh on-policy data/update iterations and
is refreshed from the current policy only after each outer round.
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

from ai_observation import OBSERVATION_SIZE, PASS_ACTION
from anchored_ppo import (
    CHAMPION_SHA256,
    enforce_initial_ratio_invariant,
    file_sha256,
    monte_carlo_returns,
)
from ppo2 import (
    DEFAULT_TORCH_THREADS,
    PPO_Supervised,
    StudentNet,
    compute_gae,
)
from seed_integrity import DEVELOPMENT_SEED_LIMIT


PAPER = "Yu et al., NashPG, arXiv:2510.18183v2"
PAPER_URL = "https://arxiv.org/abs/2510.18183v2"
SELF_PLAY_AGENT_SETS = [["self", "self"]]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_development_seed(seed: int) -> None:
    if seed < 0 or seed >= DEVELOPMENT_SEED_LIMIT:
        raise ValueError("NashPG pilot seed must stay in development range")


def validate_development_seed_plan(
    seed: int,
    *,
    outer_rounds: int,
    inner_iterations: int,
) -> None:
    """Guard the root and every deterministic derived training seed."""
    if min(outer_rounds, inner_iterations) < 1:
        raise ValueError("seed plan dimensions must be positive")
    derived = (
        seed,
        seed + 10_000,
        seed + (outer_rounds - 1) * 1_000 + inner_iterations - 1,
    )
    for planned_seed in derived:
        validate_development_seed(planned_seed)


def validate_two_player_self_play(
    agent_sets: list[list[str]],
) -> None:
    normalized = [
        [str(agent).strip().lower() for agent in group]
        for group in agent_sets
    ]
    if normalized != SELF_PLAY_AGENT_SETS:
        raise ValueError(
            "NashPG pilot is restricted to one two-player self-play group"
        )


def _numpy_rng_states_equal(left: tuple, right: tuple) -> bool:
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def generate_audited_self_play_data(
    ppo: PPO_Supervised,
    *,
    episodes: int,
    temperature: float,
    agent_sets: list[list[str]],
) -> tuple[tuple, dict]:
    """Collect data and reconstruct the exact NumPy deal-seed stream."""
    validate_two_player_self_play(agent_sets)
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if getattr(ppo, "past_policy_probability", None) != 0.0:
        raise ValueError("audited NashPG rollouts require no past-policy draws")
    before = np.random.get_state()
    replay = np.random.RandomState()
    replay.set_state(before)
    deal_seeds = replay.randint(
        0,
        DEVELOPMENT_SEED_LIMIT,
        size=episodes,
        dtype=np.uint32,
    )
    expected_after = replay.get_state()
    data = ppo.generate_data(
        num_episodes=episodes,
        temperature=temperature,
        agent_sets=agent_sets,
    )
    actual_after = np.random.get_state()
    if not _numpy_rng_states_equal(expected_after, actual_after):
        raise RuntimeError(
            "rollout NumPy RNG consumption did not match one deal seed "
            "per game"
        )
    if ppo.last_rollout_metrics["completed_games"] != episodes:
        raise RuntimeError("rollout did not complete every audited deal")
    return data, {
        "deal_seeds": [
            {"deal_seed": int(deal_seed)}
            for deal_seed in deal_seeds
        ],
        "deal_seed_count": len(deal_seeds),
        "deal_seed_rng_replay_verified": True,
    }


def module_sha256(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def masked_current_reference_kl(
    current_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    legal_actions: torch.Tensor,
) -> torch.Tensor:
    """Return mean KL(current || reference) over legal actions only."""
    if current_logits.shape != reference_logits.shape:
        raise ValueError("current and reference logits must have equal shape")
    if legal_actions.shape != current_logits.shape:
        raise ValueError("legal mask must match the logits")
    if legal_actions.dtype != torch.bool:
        raise ValueError("legal mask must be boolean")
    if not torch.all(legal_actions.any(dim=-1)):
        raise ValueError("every observation must have a legal action")
    floor = torch.finfo(current_logits.dtype).min
    current_log_probs = torch.log_softmax(
        current_logits.masked_fill(~legal_actions, floor), dim=-1
    )
    reference_log_probs = torch.log_softmax(
        reference_logits.masked_fill(~legal_actions, floor), dim=-1
    )
    current_probs = current_log_probs.exp()
    return (
        current_probs * (current_log_probs - reference_log_probs)
    ).sum(dim=-1).mean()


def clipped_nashpg_objective(
    ratios: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip: float,
    alpha: float,
    reference_kl: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the paper objective and its PPO clipped-surrogate component."""
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")
    if clip < 0:
        raise ValueError("clip must be non-negative")
    surrogate = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip, 1.0 + clip) * advantages
    clipped_surrogate = torch.minimum(surrogate, clipped).mean()
    if alpha == 0:
        # Preserve exact PPO behavior even if a diagnostic KL is non-finite.
        return clipped_surrogate, clipped_surrogate
    objective = clipped_surrogate - float(alpha) * reference_kl
    return objective, clipped_surrogate


def terminal_zero_sum_rewards(
    actions: torch.Tensor,
    rewards: torch.Tensor,
) -> torch.Tensor:
    """Remove standard PPO's per-pass shaping, retaining terminal payoff."""
    if actions.shape != rewards.shape:
        raise ValueError("actions and rewards must have equal shape")
    return rewards + 0.1 * (actions == PASS_ACTION).to(rewards.dtype)


def assert_two_player_zero_sum(
    rewards: torch.Tensor,
    episode_lengths: list[int],
    *,
    expected_games: int,
    atol: float = 1e-6,
) -> dict[str, int | bool]:
    """Audit flattened two-seat terminal returns.

    The batched collector omits an episode when a seat made zero policy
    decisions.  In that case the flattened lengths no longer retain enough
    identity information to pair later episodes.  We therefore check exact
    adjacent pairs only for full-cardinality rollouts; otherwise the omitted
    payoff is structurally fixed by the two-player zero-sum environment and
    contributes no policy-gradient sample.
    """
    if expected_games < 1:
        raise ValueError("expected_games must be positive")
    populated_episodes = len(episode_lengths)
    if not expected_games <= populated_episodes <= 2 * expected_games:
        raise ValueError(
            "two-player rollout must contain one or two populated seats "
            "per game"
        )
    if any(length <= 0 for length in episode_lengths):
        raise ValueError("two-player rollout episodes must be populated")
    offset = 0
    returns = []
    for length in episode_lengths:
        returns.append(rewards[offset : offset + length].sum())
        offset += length
    if offset != len(rewards):
        raise ValueError("episode lengths do not cover rewards")
    terminal_returns = torch.stack(returns)
    valid_terminal = torch.stack(
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
    if not bool(valid_terminal.all()):
        raise RuntimeError("episode return is not a terminal win/draw/loss payoff")
    full_pair_audit = populated_episodes == 2 * expected_games
    if full_pair_audit:
        paired = terminal_returns.reshape(-1, 2).sum(dim=1)
        if not torch.allclose(
            paired,
            torch.zeros_like(paired),
            atol=atol,
            rtol=0,
        ):
            raise RuntimeError("self-play payoff is not two-player zero-sum")
    return {
        "populated_episodes": populated_episodes,
        "implicit_zero_decision_seats": (
            2 * expected_games - populated_episodes
        ),
        "full_pair_zero_sum_verified": full_pair_audit,
        "terminal_payoffs_verified": True,
    }


def _critic_values(
    ppo: PPO_Supervised,
    observations: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    values = []
    with torch.inference_mode():
        for start in range(0, len(observations), batch_size):
            values.append(
                ppo.critic(observations[start : start + batch_size]).squeeze(-1)
            )
    return torch.cat(values)


def fit_zero_sum_critic(
    ppo: PPO_Supervised,
    *,
    iterations: int,
    episodes: int,
    epochs: int,
    batch_size: int,
    temperature: float,
    agent_sets: list[list[str]],
    gamma: float = 0.97,
) -> list[dict]:
    """Warm only the critic on terminal zero-sum self-play returns."""
    validate_two_player_self_play(agent_sets)
    if iterations == 0:
        return []
    if min(iterations, episodes, epochs, batch_size) < 1:
        raise ValueError("enabled critic warm-up sizes must be positive")
    actor_before = module_sha256(ppo.actor)
    history = []
    for iteration in range(iterations):
        started = time.perf_counter()
        data, deal_seed_audit = generate_audited_self_play_data(
            ppo,
            episodes=episodes,
            temperature=temperature,
            agent_sets=agent_sets,
        )
        observations, actions, _, rewards, lengths = data
        zero_sum_rewards = terminal_zero_sum_rewards(actions, rewards)
        payoff_audit = assert_two_player_zero_sum(
            zero_sum_rewards,
            lengths,
            expected_games=ppo.last_rollout_metrics["completed_games"],
        )
        critic_observations = observations.to(ppo.device)
        targets = monte_carlo_returns(
            zero_sum_rewards.to(ppo.device), lengths, gamma=gamma
        )
        ppo.critic.eval()
        before = _critic_values(ppo, critic_observations, batch_size)
        generator = torch.Generator(device=ppo.device).manual_seed(
            81_000 + iteration
        )
        losses = []
        ppo.critic.train()
        for _ in range(epochs):
            order = torch.randperm(
                len(targets), generator=generator, device=ppo.device
            )
            for start in range(0, len(order), batch_size):
                indices = order[start : start + batch_size]
                prediction = ppo.critic(
                    critic_observations[indices]
                ).squeeze(-1)
                loss = torch.nn.functional.smooth_l1_loss(
                    prediction, targets[indices]
                )
                ppo.critic_optim.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ppo.critic.parameters(), 1.0)
                ppo.critic_optim.step()
                losses.append(float(loss.detach()))
        ppo.critic.eval()
        after = _critic_values(ppo, critic_observations, batch_size)
        history.append(
            {
                "iteration": iteration + 1,
                "games": episodes,
                "decisions": len(actions),
                "mse_before": float(
                    torch.nn.functional.mse_loss(before, targets)
                ),
                "mse_after": float(
                    torch.nn.functional.mse_loss(after, targets)
                ),
                "mean_training_loss": float(np.mean(losses)),
                "rollout": {
                    **dict(ppo.last_rollout_metrics),
                    **deal_seed_audit,
                },
                "zero_sum_verified": payoff_audit[
                    "full_pair_zero_sum_verified"
                ],
                "structural_two_player_zero_sum": True,
                "payoff_audit": payoff_audit,
                "wall_seconds": time.perf_counter() - started,
            }
        )
    if module_sha256(ppo.actor) != actor_before:
        raise RuntimeError("critic warm-up changed frozen actor parameters")
    return history


def nashpg_inner_iteration(
    ppo: PPO_Supervised,
    reference: StudentNet,
    *,
    alpha: float,
    episodes: int,
    update_epochs: int,
    batch_size: int,
    temperature: float,
    agent_sets: list[list[str]],
    permutation_seed: int,
    gamma: float = 0.97,
    gae_lambda: float = 0.95,
) -> dict:
    """Collect fresh self-play data and perform one regularized PPO step."""
    validate_two_player_self_play(agent_sets)
    if alpha < 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and non-negative")
    if min(episodes, update_epochs, batch_size) < 1:
        raise ValueError("iteration sizes must be positive")
    reference_hash = module_sha256(reference)
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError("reference policy must be frozen")
    started = time.perf_counter()
    data, deal_seed_audit = generate_audited_self_play_data(
        ppo,
        episodes=episodes,
        temperature=temperature,
        agent_sets=agent_sets,
    )
    observations, actions, old_log_probs, rewards, lengths = data
    zero_sum_rewards = terminal_zero_sum_rewards(actions, rewards)
    payoff_audit = assert_two_player_zero_sum(
        zero_sum_rewards,
        lengths,
        expected_games=ppo.last_rollout_metrics["completed_games"],
    )
    critic_observations = observations.to(ppo.device)
    ppo.critic.eval()
    with torch.inference_mode():
        old_values = _critic_values(
            ppo, critic_observations, batch_size
        )
        advantages, returns = compute_gae(
            zero_sum_rewards.to(ppo.device),
            old_values,
            episode_lengths=lengths,
            gamma_=gamma,
            lam=gae_lambda,
        )
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + 1e-8)
    actor_advantages = advantages.to(ppo.actor_device)
    critic_returns = returns.to(ppo.device)
    masks = ppo.get_action_masks(observations)
    generator = torch.Generator(device=ppo.actor_device).manual_seed(
        permutation_seed
    )
    epoch_logs = []
    max_initial_ratio_error = None
    initial_reference_kl = None
    ppo.critic.train()
    for epoch in range(update_epochs):
        order = torch.randperm(
            len(actions), generator=generator, device=ppo.actor_device
        )
        objectives = []
        surrogates = []
        critic_losses = []
        reference_kls = []
        old_policy_kls = []
        clip_fractions = []
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_observations = observations[indices]
            batch_masks = masks[indices]
            current_logits = ppo.actor(batch_observations) / temperature
            with torch.inference_mode():
                reference_logits = (
                    reference(batch_observations) / temperature
                )
            floor = torch.finfo(current_logits.dtype).min
            distribution = Categorical(
                logits=current_logits.masked_fill(~batch_masks, floor),
                validate_args=False,
            )
            new_log_probs = distribution.log_prob(actions[indices])
            ratios = (new_log_probs - old_log_probs[indices]).exp()
            if epoch == 0 and start == 0:
                max_initial_ratio_error = enforce_initial_ratio_invariant(
                    ratios
                )
            reference_kl = masked_current_reference_kl(
                current_logits, reference_logits, batch_masks
            )
            if epoch == 0 and start == 0:
                initial_reference_kl = float(reference_kl.detach())
            objective, clipped_surrogate = clipped_nashpg_objective(
                ratios,
                actor_advantages[indices],
                clip=ppo.clip,
                alpha=alpha,
                reference_kl=reference_kl,
            )
            critic_indices = indices.to(ppo.device)
            values = ppo.critic(
                critic_observations[critic_indices]
            ).squeeze(-1)
            critic_loss = torch.nn.functional.mse_loss(
                values, critic_returns[critic_indices]
            )
            total_loss = -objective + 0.8 * critic_loss
            ppo.actor_optim.zero_grad(set_to_none=True)
            ppo.critic_optim.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(ppo.actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(ppo.critic.parameters(), 1.0)
            ppo.actor_optim.step()
            ppo.critic_optim.step()

            objectives.append(float(objective.detach()))
            surrogates.append(float(clipped_surrogate.detach()))
            critic_losses.append(float(critic_loss.detach()))
            reference_kls.append(float(reference_kl.detach()))
            old_policy_kls.append(
                float((old_log_probs[indices] - new_log_probs).mean().detach())
            )
            clip_fractions.append(
                float(((ratios - 1.0).abs() > ppo.clip).float().mean())
            )
        epoch_logs.append(
            {
                "epoch": epoch + 1,
                "objective": float(np.mean(objectives)),
                "clipped_surrogate": float(np.mean(surrogates)),
                "reference_kl_current_to_reference": float(
                    np.mean(reference_kls)
                ),
                "critic_loss": float(np.mean(critic_losses)),
                "old_policy_approx_kl": float(np.mean(old_policy_kls)),
                "clip_fraction": float(np.mean(clip_fractions)),
            }
        )
    ppo.critic.eval()
    ppo._sync_rollout_actor()
    if module_sha256(reference) != reference_hash:
        raise RuntimeError("reference policy changed during inner iteration")
    return {
        "alpha": float(alpha),
        "games": episodes,
        "decisions": len(actions),
        "rollout": {
            **dict(ppo.last_rollout_metrics),
            **deal_seed_audit,
        },
        "zero_sum_verified": payoff_audit["full_pair_zero_sum_verified"],
        "structural_two_player_zero_sum": True,
        "payoff_audit": payoff_audit,
        "max_initial_ratio_error": max_initial_ratio_error,
        "initial_reference_kl_current_to_reference": initial_reference_kl,
        "reference_sha256": reference_hash,
        "update_epochs": epoch_logs,
        "wall_seconds": time.perf_counter() - started,
    }


def refresh_reference(
    reference: StudentNet,
    current: StudentNet,
) -> str:
    """Refresh the frozen reference from current and return its state hash."""
    reference.load_state_dict(current.state_dict())
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    current_hash = module_sha256(current)
    reference_hash = module_sha256(reference)
    if reference_hash != current_hash:
        raise RuntimeError("reference refresh did not copy current policy")
    return reference_hash


def alpha_slug(alpha: float) -> str:
    return format(float(alpha), "g").replace("-", "m").replace(".", "p")


def make_ppo(
    champion: Path,
    *,
    torch_threads: int,
    rollout_batch_size: int,
    learning_rate: float,
) -> PPO_Supervised:
    ppo = PPO_Supervised(
        use_pretrained=True,
        actor_path=str(champion),
        critic_path="missing_nashpg_critic.pt",
        actor_opt_path="missing_nashpg_actor_optimizer.pt",
        critic_opt_path="missing_nashpg_critic_optimizer.pt",
        old_save=(),
        max_saves=0,
        rollout_device="cpu",
        rollout_batch_size=rollout_batch_size,
        training_actor_device=(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        compile_updates=False,
        past_policy_probability=0.0,
        torch_threads=torch_threads,
    )
    for optimizer in (ppo.actor_optim, ppo.critic_optim):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    ppo.critic.eval()
    return ppo


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--champion", type=Path, default=Path("ppo_actor_100run3.pth")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=84_400_001)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.05, 0.2, 0.5]
    )
    parser.add_argument("--outer-rounds", type=int, default=3)
    parser.add_argument("--inner-iterations", type=int, default=3)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--rollout-batch-size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--critic-iterations", type=int, default=1)
    parser.add_argument("--critic-games", type=int, default=128)
    parser.add_argument("--critic-epochs", type=int, default=1)
    parser.add_argument(
        "--torch-threads", type=int, default=DEFAULT_TORCH_THREADS
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    validate_development_seed_plan(
        args.seed,
        outer_rounds=args.outer_rounds,
        inner_iterations=args.inner_iterations,
    )
    validate_two_player_self_play(SELF_PLAY_AGENT_SETS)
    if not args.alphas or any(
        alpha < 0 or not np.isfinite(alpha) for alpha in args.alphas
    ):
        raise ValueError("alphas must be finite and non-negative")
    if min(
        args.outer_rounds,
        args.inner_iterations,
        args.games,
        args.update_epochs,
        args.batch_size,
        args.rollout_batch_size,
        args.torch_threads,
    ) < 1:
        raise ValueError("enabled NashPG sizes must be positive")
    champion = args.champion.resolve()
    if file_sha256(champion) != CHAMPION_SHA256:
        raise RuntimeError("refusing to train from an unverified champion")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    base = make_ppo(
        champion,
        torch_threads=args.torch_threads,
        rollout_batch_size=args.rollout_batch_size,
        learning_rate=args.learning_rate,
    )
    warmup = fit_zero_sum_critic(
        base,
        iterations=args.critic_iterations,
        episodes=args.critic_games,
        epochs=args.critic_epochs,
        batch_size=args.batch_size,
        temperature=args.temperature,
        agent_sets=SELF_PLAY_AGENT_SETS,
    )
    warm_critic = copy.deepcopy(base.critic.state_dict())
    warm_critic_optimizer = copy.deepcopy(base.critic_optim.state_dict())

    sweep_runs = []
    for alpha in args.alphas:
        # Every coefficient begins from the same policy, critic, optimizer, and
        # RNG state so the sweep is controlled up to policy-induced divergence.
        seed_everything(args.seed + 10_000)
        ppo = make_ppo(
            champion,
            torch_threads=args.torch_threads,
            rollout_batch_size=args.rollout_batch_size,
            learning_rate=args.learning_rate,
        )
        ppo.critic.load_state_dict(warm_critic)
        ppo.critic_optim.load_state_dict(warm_critic_optimizer)
        ppo.critic.eval()
        reference = copy.deepcopy(ppo.actor).eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        run = {
            "alpha": float(alpha),
            "run_seed": args.seed + 10_000,
            "initial_actor_sha256": module_sha256(ppo.actor),
            "outer_rounds": [],
        }
        for outer_index in range(args.outer_rounds):
            reference_before = module_sha256(reference)
            current_before = module_sha256(ppo.actor)
            if reference_before != current_before:
                raise RuntimeError(
                    "outer round must begin at the refreshed reference"
                )
            inner_logs = []
            for inner_index in range(args.inner_iterations):
                inner_logs.append(
                    nashpg_inner_iteration(
                        ppo,
                        reference,
                        alpha=alpha,
                        episodes=args.games,
                        update_epochs=args.update_epochs,
                        batch_size=args.batch_size,
                        temperature=args.temperature,
                        agent_sets=SELF_PLAY_AGENT_SETS,
                        permutation_seed=(
                            args.seed
                            + outer_index * 1_000
                            + inner_index
                        ),
                    )
                )
            filename = (
                f"nashpg_alpha_{alpha_slug(alpha)}"
                f"_outer_{outer_index + 1:02d}.pth"
            )
            checkpoint_path = output_dir / filename
            torch.save(ppo.actor.state_dict(), checkpoint_path)
            checkpoint_sha = file_sha256(checkpoint_path)
            refreshed_hash = refresh_reference(reference, ppo.actor)
            run["outer_rounds"].append(
                {
                    "alpha": float(alpha),
                    "outer_iteration": outer_index + 1,
                    "reference_sha256_before": reference_before,
                    "current_sha256_before": current_before,
                    "inner_iterations": inner_logs,
                    "checkpoint": {
                        "path": filename,
                        "sha256": checkpoint_sha,
                        "actor_state_sha256": module_sha256(ppo.actor),
                    },
                    "reference_refresh": {
                        "performed_after_inner_iterations": (
                            args.inner_iterations
                        ),
                        "reference_sha256_after": refreshed_hash,
                        "matches_current": (
                            refreshed_hash == module_sha256(ppo.actor)
                        ),
                    },
                }
            )
        sweep_runs.append(run)

    payload = {
        "schema_version": 1,
        "algorithm": "nashpg_ppo",
        "paper": {
            "citation": PAPER,
            "url": PAPER_URL,
            "equations": [11, 12],
            "algorithm": 4,
        },
        "objective": {
            "maximize": (
                "PPO clipped surrogate - alpha * "
                "E_o KL(current || reference)"
            ),
            "kl_direction": "current || reference",
            "reference_schedule": (
                "frozen for K inner data/update iterations, "
                "then refreshed from current"
            ),
        },
        "champion": {
            "path": str(champion),
            "sha256": CHAMPION_SHA256,
        },
        "privacy": {
            "players": 2,
            "self_play": True,
            "shared_symmetric_policy": True,
            "observation_features": OBSERVATION_SIZE,
            "own_hand_plus_public_state_only": True,
            "opponent_card_identities": False,
            "draw_pile_order": False,
        },
        "payoff": {
            "terminal_two_player_zero_sum_only": True,
            "standard_per_pass_shaping_removed": True,
            "zero_decision_seat_handling": (
                "The flattened collector omits seats with no policy "
                "decisions. Their unobserved opposite payoff is fixed by "
                "the two-player zero-sum engine and has no gradient sample; "
                "payoff_audit records their count. Exact adjacent-pair sums "
                "are checked only when all seat episodes are populated."
            ),
        },
        "seed_audit": {
            "root_seed": args.seed,
            "development_seed_limit": DEVELOPMENT_SEED_LIMIT,
            "development_only": True,
            "sealed_outcomes_used": False,
            "per_deal_seeds_recorded": True,
            "deal_seed_rng_replay_verified_per_rollout": True,
        },
        "recipe": {
            "alphas": [float(alpha) for alpha in args.alphas],
            "outer_rounds": args.outer_rounds,
            "inner_iterations_per_outer_round": args.inner_iterations,
            "games_per_inner_iteration": args.games,
            "update_epochs": args.update_epochs,
            "batch_size": args.batch_size,
            "rollout_batch_size": args.rollout_batch_size,
            "temperature": args.temperature,
            "learning_rate": args.learning_rate,
            "torch_threads": args.torch_threads,
            "critic_iterations": args.critic_iterations,
            "critic_games": args.critic_games,
            "critic_epochs": args.critic_epochs,
            "agent_sets": SELF_PLAY_AGENT_SETS,
            "past_policy_probability": 0.0,
        },
        "critic_warmup": warmup,
        "sweeps": sweep_runs,
    }
    manifest = output_dir / "nashpg_pilot_metrics.json"
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "checkpoints": sum(
                    len(run["outer_rounds"]) for run in sweep_runs
                ),
                "sealed_outcomes_used": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
