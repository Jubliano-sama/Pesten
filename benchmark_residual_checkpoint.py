#!/usr/bin/env python3
"""Paired development evaluation for a residual temporal checkpoint."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

import game
import numpy as np
import torch

from benchmark_model_zoo import DEFAULT_TEACHER
from residual_temporal_policy import (
    CHECKPOINT_FORMAT,
    RECURRENT_CHECKPOINT_FORMAT,
    ResidualCausalTemporalPolicy,
    ResidualRecurrentTemporalPolicy,
)
from temporal_puffer_agent import TemporalPestenAgent


CHAMPION_PATH = DEFAULT_TEACHER
DEFAULT_OPPONENT_NAMES = (
    "ppo_champion",
    "bayesian_tree",
    "smart",
    "smarter",
    "random",
)


def _validate_opponent_names(
    names: Sequence[str],
    *,
    available: Sequence[str],
) -> tuple[str, ...]:
    normalized = tuple(str(name).strip().lower() for name in names)
    if not normalized or any(not name for name in normalized):
        raise ValueError("at least one opponent must be selected")
    duplicates = tuple(
        name
        for index, name in enumerate(normalized)
        if name in normalized[:index]
    )
    if duplicates:
        raise ValueError(
            "duplicate opponent(s): "
            + ", ".join(dict.fromkeys(duplicates))
        )
    unknown = tuple(name for name in normalized if name not in available)
    if unknown:
        raise ValueError(
            "unknown opponent(s): "
            + ", ".join(unknown)
            + "; choose from "
            + ", ".join(available)
        )
    return normalized


def parse_opponent_names(value: str) -> tuple[str, ...]:
    """Parse the CLI comma-list against the stable built-in registry."""
    try:
        return _validate_opponent_names(
            value.split(","),
            available=DEFAULT_OPPONENT_NAMES,
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def filter_opponents(
    opponents: Mapping[str, Any],
    selected_names: Sequence[str],
) -> dict[str, Any]:
    """Return a requested opponent subset in user-specified order."""
    names = _validate_opponent_names(
        selected_names,
        available=tuple(opponents),
    )
    return {name: opponents[name] for name in names}


@contextmanager
def deterministic_evaluation_runtime(
    device: str,
    *,
    cpu_threads: int = 1,
):
    """Use inference mode and right-size CPU threading for tiny online steps."""
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be at least one")
    use_cpu_threads = torch.device(device).type == "cpu"
    previous_threads = torch.get_num_threads() if use_cpu_threads else None
    if use_cpu_threads and previous_threads != cpu_threads:
        torch.set_num_threads(cpu_threads)
    try:
        with torch.inference_mode():
            yield
    finally:
        if (
            use_cpu_threads
            and previous_threads is not None
            and previous_threads != cpu_threads
        ):
            torch.set_num_threads(previous_threads)


def deterministic_evaluation_agent_factory(
    policy: ResidualCausalTemporalPolicy
    | ResidualRecurrentTemporalPolicy,
    *,
    seed: int = 0,
) -> Callable:
    """Build a game agent without training-only deterministic-act overhead.

    ``TemporalPestenAgent._act`` always constructs a categorical distribution,
    evaluates its log probability and retains a full training trajectory.
    Paired checkpoint evaluation uses argmax actions and consumes none of that
    data.  This adapter keeps the identical DecisionInput/state/action path but
    skips those unused allocations and synchronizing scalar reads.
    """

    class DeterministicEvaluationAgent(TemporalPestenAgent):
        def __init__(self, game_instance) -> None:
            super().__init__(
                game_instance,
                policy,
                deterministic=True,
                seed=seed,
            )

        def _act(self, decision) -> int:
            with torch.inference_mode():
                result = self.policy.step(decision, self.state)
                action = int(torch.argmax(result.logits).item())
            if isinstance(result.state, torch.Tensor):
                self.state = result.state.detach()
            else:
                self.state = tuple(part.detach() for part in result.state)
            # _sync_public_events uses this to distinguish the first decision
            # from later decisions made before the first public history item.
            self.safe_history.append(decision)
            return action

    DeterministicEvaluationAgent.__name__ = (
        "DeterministicResidualEvaluationAgent"
    )
    return DeterministicEvaluationAgent


def detect_residual_checkpoint_format(checkpoint: Path) -> str:
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=True
    )
    checkpoint_format = payload.get("format")
    if checkpoint_format not in {
        CHECKPOINT_FORMAT,
        RECURRENT_CHECKPOINT_FORMAT,
    }:
        raise ValueError(
            "unsupported residual checkpoint format: "
            f"{checkpoint_format!r}"
        )
    return checkpoint_format


def load_residual_checkpoint(
    checkpoint: Path,
    *,
    champion_path: Path = CHAMPION_PATH,
    map_location: str = "cpu",
):
    """Dispatch to the causal or recurrent loader from checkpoint metadata."""
    checkpoint_format = detect_residual_checkpoint_format(checkpoint)
    policy_type = (
        ResidualCausalTemporalPolicy
        if checkpoint_format == CHECKPOINT_FORMAT
        else ResidualRecurrentTemporalPolicy
    )
    policy = policy_type.load_checkpoint(
        checkpoint,
        champion_path=champion_path,
        map_location=map_location,
    )
    policy.loaded_checkpoint_format = checkpoint_format
    return policy


def paired_bootstrap_interval(
    pair_scores: list[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Match the temporal-league paired bootstrap without importing Puffer."""
    rng = np.random.default_rng(seed)
    values = np.asarray(pair_scores, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975)
    )


def evaluate_paired(
    candidate_factory,
    opponent_factory,
    *,
    pairs: int,
    seed: int,
    candidate_name: str,
    opponent_name: str,
) -> dict:
    """Seat-rotated paired evaluation, independent of training dependencies."""
    result, _ = _evaluate_paired_with_scores(
        candidate_factory,
        opponent_factory,
        pairs=pairs,
        seed=seed,
        candidate_name=candidate_name,
        opponent_name=opponent_name,
    )
    return result


def _evaluate_paired_with_scores(
    candidate_factory,
    opponent_factory,
    *,
    pairs: int,
    seed: int,
    candidate_name: str,
    opponent_name: str,
) -> tuple[dict, list[float]]:
    wins = losses = draws = turns = 0
    seat_wins = [0, 0]
    pair_scores = []
    started = time.perf_counter()
    for pair in range(pairs):
        deal_seed = seed + pair
        scores = []
        for candidate_seat in (0, 1):
            np.random.seed(deal_seed)
            random.seed(deal_seed)
            factories = (
                [candidate_factory, opponent_factory]
                if candidate_seat == 0
                else [opponent_factory, candidate_factory]
            )
            match = game.Game(factories)
            game_turns, winner = match.auto_simulate()
            turns += game_turns
            if winner is None:
                draws += 1
                scores.append(0.5)
            elif winner == candidate_seat:
                wins += 1
                seat_wins[candidate_seat] += 1
                scores.append(1.0)
            else:
                losses += 1
                scores.append(0.0)
        pair_scores.append(sum(scores) / 2)
    games = pairs * 2
    ci = paired_bootstrap_interval(pair_scores, seed=seed + 90_000)
    result = {
        "candidate": candidate_name,
        "opponent": opponent_name,
        "pairs": pairs,
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": (wins + 0.5 * draws) / games,
        "paired_bootstrap_ci95": list(ci),
        "wins_as_seat_0": seat_wins[0],
        "wins_as_seat_1": seat_wins[1],
        "average_turns": turns / games,
        "elapsed_seconds": time.perf_counter() - started,
        "gate_passed": ci[0] > 0.5,
        "seed": seed,
    }
    return result, pair_scores


def _paired_delta_summary(
    candidate_scores: list[float],
    base_scores: list[float],
    *,
    seed: int,
    opponent_name: str,
) -> tuple[dict, list[float]]:
    if len(candidate_scores) != len(base_scores):
        raise AssertionError("candidate and base pair counts differ")
    deltas = [
        candidate_score - base_score
        for candidate_score, base_score in zip(
            candidate_scores, base_scores
        )
    ]
    ci = paired_bootstrap_interval(deltas, seed=seed + 190_000)
    candidate_mean = float(np.mean(candidate_scores))
    base_mean = float(np.mean(base_scores))
    delta = float(np.mean(deltas))
    return (
        {
            "opponent": opponent_name,
            "pairs": len(deltas),
            "same_deals_and_rotated_seats": True,
            "candidate_score": candidate_mean,
            "base_score": base_mean,
            "candidate_minus_base_score": delta,
            "paired_bootstrap_ci95": list(ci),
            "delta_gate_passed": ci[0] > 0.0,
            "seed": seed,
        },
        deltas,
    )


def evaluate_candidate_and_base_paired(
    candidate_factory,
    base_factory,
    opponent_factory,
    *,
    pairs: int,
    seed: int,
    candidate_name: str,
    opponent_name: str,
) -> dict:
    """Evaluate residual and frozen base on identical paired deal strata."""
    candidate_result, candidate_scores = _evaluate_paired_with_scores(
        candidate_factory,
        opponent_factory,
        pairs=pairs,
        seed=seed,
        candidate_name=candidate_name,
        opponent_name=opponent_name,
    )
    base_result, base_scores = _evaluate_paired_with_scores(
        base_factory,
        opponent_factory,
        pairs=pairs,
        seed=seed,
        candidate_name="frozen_ppo_base",
        opponent_name=opponent_name,
    )
    comparison, deltas = _paired_delta_summary(
        candidate_scores,
        base_scores,
        seed=seed,
        opponent_name=opponent_name,
    )
    return {
        "candidate_result": candidate_result,
        "base_result": base_result,
        "comparison": comparison,
        "_candidate_pair_scores": candidate_scores,
        "_base_pair_scores": base_scores,
        "_pair_deltas": deltas,
    }


def stratified_paired_bootstrap_interval(
    delta_groups: Mapping[str, list[float]],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Bootstrap within opponents while preserving the evaluated pool mix."""
    if not delta_groups or any(not values for values in delta_groups.values()):
        raise ValueError("every opponent needs at least one paired delta")
    rng = np.random.default_rng(seed)
    total_pairs = sum(len(values) for values in delta_groups.values())
    bootstrap_sums = np.zeros(samples, dtype=np.float64)
    for values in delta_groups.values():
        array = np.asarray(values, dtype=np.float64)
        indices = rng.integers(
            0, len(array), size=(samples, len(array))
        )
        bootstrap_sums += array[indices].sum(axis=1)
    means = bootstrap_sums / total_pairs
    return float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975)
    )


def _default_evaluation_dependencies(
    champion_path: Path,
    device: str,
) -> tuple[Callable[..., dict], Mapping[str, Any]]:
    # Keep checkpoint screening independent of the optional Puffer training
    # environment.  These are the same factories and paired-deal semantics
    # used by temporal_league.evaluate_paired.
    from benchmark_agents import load_ppo_agent
    from hardcodedAgent import HardcodedStrategyAgent
    from randomAgent import AgentRandom
    from smartAgent import Agent as SmartAgent, SmarterAgent

    def tree_factory(game_instance):
        return HardcodedStrategyAgent(
            game_instance,
            belief_samples=8,
            search_depth=3,
            beam_width=3,
        )

    return evaluate_paired, {
        "ppo_champion": load_ppo_agent(champion_path, device),
        "bayesian_tree": tree_factory,
        "smart": SmartAgent,
        "smarter": SmarterAgent,
        "random": AgentRandom,
    }


def evaluate_residual_checkpoint(
    *,
    checkpoint: Path,
    champion_path: Path = CHAMPION_PATH,
    pairs: int,
    seed: int,
    device: str = "cpu",
    cpu_threads: int = 1,
    evaluate_pair: Callable[..., dict] | None = None,
    evaluate_comparison: Callable[..., dict] | None = None,
    opponents: Mapping[str, Any] | None = None,
    opponent_names: Sequence[str] | None = None,
    base_factory: Any | None = None,
) -> dict:
    """Load either residual family and run the same paired evaluation pool."""
    policy = load_residual_checkpoint(
        checkpoint,
        champion_path=champion_path,
        map_location=device,
    ).eval()
    candidate = deterministic_evaluation_agent_factory(
        policy, seed=seed
    )
    if evaluate_pair is None or opponents is None:
        default_evaluate, default_opponents = (
            _default_evaluation_dependencies(champion_path, device)
        )
        evaluate_pair = evaluate_pair or default_evaluate
        opponents = opponents or default_opponents
    opponent_seed_indices = {
        name: index for index, name in enumerate(opponents)
    }
    if evaluate_comparison is None:
        if evaluate_pair is not evaluate_paired:
            raise ValueError(
                "custom evaluate_pair requires evaluate_comparison so "
                "paired candidate-minus-base samples remain available"
            )
        evaluate_comparison = evaluate_candidate_and_base_paired
    if base_factory is None:
        base_factory = opponents.get("ppo_champion")
    if base_factory is None:
        raise ValueError(
            "base_factory or a ppo_champion opponent is required"
        )
    if opponent_names is not None:
        opponents = filter_opponents(opponents, opponent_names)

    comparison_blocks = {}
    with deterministic_evaluation_runtime(
        device, cpu_threads=cpu_threads
    ):
        for name, opponent in opponents.items():
            opponent_pairs = (
                pairs
                if name == "ppo_champion"
                else max(20, pairs // 2)
            )
            comparison_blocks[name] = evaluate_comparison(
                candidate,
                base_factory,
                opponent,
                pairs=opponent_pairs,
                seed=seed + opponent_seed_indices[name] * 100_000,
                candidate_name=policy.architecture,
                opponent_name=name,
            )

    results = {
        name: block["candidate_result"]
        for name, block in comparison_blocks.items()
    }
    base_results = {
        name: block["base_result"]
        for name, block in comparison_blocks.items()
    }
    paired_deltas = {
        name: block["comparison"]
        for name, block in comparison_blocks.items()
    }
    delta_groups = {
        name: block["_pair_deltas"]
        for name, block in comparison_blocks.items()
    }
    candidate_scores = [
        score
        for block in comparison_blocks.values()
        for score in block["_candidate_pair_scores"]
    ]
    base_scores = [
        score
        for block in comparison_blocks.values()
        for score in block["_base_pair_scores"]
    ]
    all_deltas = [
        delta
        for values in delta_groups.values()
        for delta in values
    ]
    combined_ci = stratified_paired_bootstrap_interval(
        delta_groups, seed=seed + 990_000
    )
    combined_pool_delta = {
        "opponents": list(opponents),
        "pairs": len(all_deltas),
        "weighting": "one vote per evaluated paired deal",
        "bootstrap": "stratified within opponent",
        "same_deals_and_rotated_seats": True,
        "candidate_score": float(np.mean(candidate_scores)),
        "base_score": float(np.mean(base_scores)),
        "candidate_minus_base_score": float(np.mean(all_deltas)),
        "paired_bootstrap_ci95": list(combined_ci),
        "delta_gate_passed": combined_ci[0] > 0.0,
        "seed": seed + 990_000,
    }
    return {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_format": policy.loaded_checkpoint_format,
        "policy": {
            "class": type(policy).__name__,
            "architecture": policy.architecture,
            "recurrent_cell": getattr(policy, "cell", None),
        },
        "base_policy": {
            "path": str(champion_path),
            "sha256": policy.champion_sha256,
            "frozen": True,
        },
        "information_boundary": (
            "DecisionInput plus redacted PublicEventToken history only"
        ),
        "paired_deals_and_rotated_seats": True,
        "evaluation_runtime": {
            "deterministic_fast_path": True,
            "records_training_trajectories": False,
            "cpu_threads": (
                cpu_threads if torch.device(device).type == "cpu" else None
            ),
        },
        "development_seed": seed,
        "opponents": list(opponents),
        "results": results,
        "base_results": base_results,
        "paired_candidate_minus_base": paired_deltas,
        "combined_pool_delta": combined_pool_delta,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--champion", type=Path, default=CHAMPION_PATH)
    parser.add_argument("--pairs", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=1,
        help="temporary PyTorch intra-op thread count for CPU evaluation",
    )
    parser.add_argument(
        "--opponents",
        type=parse_opponent_names,
        default=DEFAULT_OPPONENT_NAMES,
        metavar="NAME[,NAME...]",
        help=(
            "evaluation subset in desired order; available: "
            + ", ".join(DEFAULT_OPPONENT_NAMES)
        ),
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    payload = evaluate_residual_checkpoint(
        checkpoint=args.checkpoint,
        champion_path=args.champion,
        pairs=args.pairs,
        seed=args.seed,
        device=args.device,
        cpu_threads=args.cpu_threads,
        opponent_names=args.opponents,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
