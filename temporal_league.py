"""Reproducible information-safe self-play league for native PufferLib 4.

The league is deliberately two-player for the PPO champion gate.  Three- and
four-player evaluation are separate entry points so their metrics cannot be
mixed with the direct gate.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import copy
import hashlib
import math
from pathlib import Path
import random
import time
from typing import Callable, Sequence

import numpy as np
import torch
from torch.distributions import Categorical

import card
import game
from ai_observation import (
    OBSERVATION_SIZE as LEGACY_OBSERVATION_SIZE,
    populate_game_context,
)
from benchmark_agents import (
    load_ppo_agent,
)
from hardcodedAgent import HardcodedStrategyAgent
from ppo2 import StudentNet, build_action_masks
from randomAgent import AgentRandom
from smartAgent import Agent as SmartAgent, SmarterAgent
from temporal_puffer_agent import (
    TemporalPestenAgent,
    TrajectoryStep,
    temporal_agent_factory,
)
from temporal_puffer_v4 import NativePufferTemporalPolicy
from temporal_training import ppo_update


CHAMPION_PATH = Path("ppo_actor_100run3.pth")
CHAMPION_SHA256 = (
    "aac589b3c7300895604e6c77755631ab4a31c8bf9810768448ab2f3ff5bbe186"
)
LEAGUE_MIXTURE = (
    ("current_selfplay", 4),
    ("historical_temporal", 2),
    ("ppo_champion", 2),
    ("bayesian_tree", 1),
    ("smart_rule", 1),
)
LEAGUE_OPPONENT_TYPES = (
    "current_selfplay",
    "historical_temporal",
    "ppo_champion",
    "bayesian_tree",
    "smart_rule",
    "smarter_rule",
    "random_rule",
)


@dataclass(frozen=True, slots=True)
class ImitationStep:
    decision: object
    action: int
    teacher_logits: tuple[float, ...] = ()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_champion(path: Path = CHAMPION_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"PPO champion checkpoint missing: {path}")
    digest = file_sha256(path)
    if digest != CHAMPION_SHA256:
        raise RuntimeError(
            f"PPO champion hash changed: expected {CHAMPION_SHA256}, got {digest}"
        )
    state = torch.load(path, map_location="cpu", weights_only=True)
    first = state["model.layers.0.weight"]
    final = state["model.final_layer.weight"]
    if tuple(first.shape) != (256, 121) or final.shape[0] != 55:
        raise RuntimeError("PPO champion observation/action contract changed")
    return {
        "path": str(path),
        "sha256": digest,
        "architecture": "ppo2.StudentNet(DenseSkipNet 8x256)",
        "observation_size": 121,
        "action_count": 55,
        "critic_checkpoint_available": False,
    }


def tree_factory(game_instance):
    return HardcodedStrategyAgent(
        game_instance,
        belief_samples=8,
        search_depth=3,
        beam_width=3,
    )


def _legacy_observation(agent: TemporalPestenAgent, change_suit: bool):
    observation = torch.zeros(LEGACY_OBSERVATION_SIZE, dtype=torch.float32)
    for held in agent.mydeck.cards:
        observation[held.number] += 1
        if agent.game.penalty_amount > 0:
            legal = held.truenumber == agent.game.current_true_number
        else:
            legal = held.compatible(
                agent.game.current_sort, agent.game.current_true_number
            )
        if legal:
            observation[held.number + 50] += 1
    observer = agent.game.players.index(agent)
    populate_game_context(
        observation,
        sorts_played=agent.game.sorts_played,
        hand_sizes=[
            player.mydeck.cardCount() for player in agent.game.players
        ],
        player_index=observer,
        direction=agent.game.direction,
        is_change_sort=change_suit,
    )
    return observation


def teacher_recorder_factory(
    policy: NativePufferTemporalPolicy,
    teacher: StudentNet,
):
    class TeacherRecorder(TemporalPestenAgent):
        def __init__(self, game_instance):
            super().__init__(
                game_instance, policy, deterministic=True, seed=0
            )
            self.imitation: list[ImitationStep] = []

        def _act(self, decision):
            change_suit = bool(decision.observation.values[93])
            legacy = _legacy_observation(self, change_suit)
            with torch.no_grad():
                logits = teacher(legacy.unsqueeze(0)).squeeze(0)
                mask = build_action_masks(legacy)
                masked_logits = logits.masked_fill(
                    ~mask, torch.finfo(logits.dtype).min
                )
                action = int(
                    torch.argmax(masked_logits).item()
                )
            self.safe_history.append(decision)
            self.imitation.append(
                ImitationStep(
                    decision,
                    action,
                    tuple(
                        float(value)
                        for value in masked_logits.detach().cpu()
                    ),
                )
            )
            return action

    return TeacherRecorder


def collect_teacher_episodes(
    policy: NativePufferTemporalPolicy,
    teacher: StudentNet,
    *,
    games: int,
    seed: int,
) -> list[tuple[ImitationStep, ...]]:
    recorder = teacher_recorder_factory(policy, teacher)
    opponents = (SmartAgent, tree_factory, AgentRandom)
    episodes = []
    for index in range(games):
        deal_seed = seed + index
        np.random.seed(deal_seed)
        random.seed(deal_seed)
        opponent = opponents[index % len(opponents)]
        seat = index % 2
        factories = (
            [recorder, opponent] if seat == 0 else [opponent, recorder]
        )
        match = game.Game(factories)
        match.auto_simulate()
        episodes.append(tuple(match.players[seat].imitation))
    return episodes


def imitation_metrics(
    policy: NativePufferTemporalPolicy,
    episodes: Sequence[Sequence[ImitationStep]],
) -> dict[str, float]:
    policy.eval()
    losses, correct, samples = [], 0, 0
    with torch.no_grad():
        for start in range(0, len(episodes), 16):
            logits, targets = _batched_imitation_logits(
                policy, episodes[start : start + 16]
            )
            batch_losses = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            losses.extend(batch_losses)
            correct += int((logits.argmax(dim=-1) == targets).sum().item())
            samples += targets.numel()
    return {
        "cross_entropy": float(torch.stack(losses).mean()) if losses else math.nan,
        "accuracy": correct / samples if samples else math.nan,
        "samples": samples,
    }


def imitation_update(
    policy: NativePufferTemporalPolicy,
    episodes: Sequence[Sequence[ImitationStep]],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    seed: int,
    batch_episodes: int = 8,
) -> list[dict]:
    rng = random.Random(seed)
    logs = []
    for epoch in range(epochs):
        order = list(range(len(episodes)))
        rng.shuffle(order)
        epoch_losses = []
        epoch_correct = 0
        epoch_samples = 0
        for start in range(0, len(order), batch_episodes):
            optimizer.zero_grad(set_to_none=True)
            batch = [
                episodes[episode_index]
                for episode_index in order[start : start + batch_episodes]
            ]
            logits, targets = _batched_imitation_logits(policy, batch)
            if targets.numel():
                losses = torch.nn.functional.cross_entropy(
                    logits, targets, reduction="none"
                )
                epoch_losses.extend(losses.detach().cpu().tolist())
                epoch_correct += int(
                    (logits.argmax(dim=-1) == targets).sum().item()
                )
                epoch_samples += targets.numel()
                losses.mean().backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
        logs.append(
            {
                "epoch": epoch,
                "cross_entropy": sum(epoch_losses) / len(epoch_losses),
                "accuracy": epoch_correct / epoch_samples,
                "samples": epoch_samples,
            }
        )
    return logs


def _imitation_metadata(
    episodes: Sequence[Sequence[ImitationStep]],
    *,
    device: torch.device,
    dtype: torch.dtype,
    suit_weight: float,
    many_legal_weight: float,
    early_weight: float,
    large_hand_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    teacher_logits = []
    weights = []
    suit_modes = []
    for episode in episodes:
        for decision_index, sample in enumerate(episode):
            if len(sample.teacher_logits) != 55:
                raise ValueError(
                    "targeted distillation requires frozen teacher logits"
                )
            observation = sample.decision.observation.values
            suit_mode = bool(observation[93])
            weight = 1.0
            if suit_mode:
                weight *= suit_weight
            if sum(sample.decision.legal_actions) >= 4:
                weight *= many_legal_weight
            if decision_index < 5:
                weight *= early_weight
            if observation[91] >= 7:
                weight *= large_hand_weight
            teacher_logits.append(sample.teacher_logits)
            weights.append(weight)
            suit_modes.append(suit_mode)
    return (
        torch.tensor(teacher_logits, device=device, dtype=dtype),
        torch.tensor(weights, device=device, dtype=dtype),
        torch.tensor(suit_modes, device=device, dtype=torch.bool),
    )


def imitation_update_targeted(
    policy: NativePufferTemporalPolicy,
    episodes: Sequence[Sequence[ImitationStep]],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int,
    seed: int,
    batch_episodes: int = 16,
    distillation_weight: float = 0.5,
    temperature: float = 1.0,
    mode_balanced: bool = True,
    suit_weight: float = 2.0,
    many_legal_weight: float = 1.5,
    early_weight: float = 1.25,
    large_hand_weight: float = 1.5,
) -> list[dict]:
    """Train on safe strata and the PPO's full masked action distribution."""
    if not 0 <= distillation_weight <= 1:
        raise ValueError("distillation_weight must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    rng = random.Random(seed)
    logs = []
    for epoch in range(epochs):
        order = list(range(len(episodes)))
        rng.shuffle(order)
        objective_sum = cross_entropy_sum = kl_sum = 0.0
        correct = samples = suit_correct = suit_samples = 0
        optimizer_steps = 0
        for start in range(0, len(order), batch_episodes):
            batch = [
                episodes[episode_index]
                for episode_index in order[
                    start : start + batch_episodes
                ]
            ]
            optimizer.zero_grad(set_to_none=True)
            logits, targets = _batched_imitation_logits(policy, batch)
            if not targets.numel():
                continue
            teacher, weights, suit_modes = _imitation_metadata(
                batch,
                device=logits.device,
                dtype=logits.dtype,
                suit_weight=suit_weight,
                many_legal_weight=many_legal_weight,
                early_weight=early_weight,
                large_hand_weight=large_hand_weight,
            )
            cross_entropy = torch.nn.functional.cross_entropy(
                logits, targets, reduction="none"
            )
            student_log_probabilities = torch.nn.functional.log_softmax(
                logits / temperature, dim=-1
            )
            teacher_probabilities = torch.nn.functional.softmax(
                teacher / temperature, dim=-1
            )
            teacher_log_probabilities = torch.nn.functional.log_softmax(
                teacher / temperature, dim=-1
            )
            distillation_kl = (
                teacher_probabilities
                * (
                    teacher_log_probabilities
                    - student_log_probabilities
                )
            ).sum(dim=-1) * (temperature * temperature)
            per_sample = (
                (1.0 - distillation_weight) * cross_entropy
                + distillation_weight * distillation_kl
            )
            weighted = per_sample * weights
            if (
                mode_balanced
                and bool(suit_modes.any())
                and bool((~suit_modes).any())
            ):
                objective = 0.5 * (
                    weighted[suit_modes].mean()
                    + weighted[~suit_modes].mean()
                )
            else:
                objective = weighted.mean()
            objective.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            optimizer_steps += 1

            batch_samples = targets.numel()
            predictions = logits.argmax(dim=-1)
            objective_sum += float(
                objective.detach().cpu()
            ) * batch_samples
            cross_entropy_sum += float(
                cross_entropy.detach().sum().cpu()
            )
            kl_sum += float(distillation_kl.detach().sum().cpu())
            correct += int((predictions == targets).sum().item())
            samples += batch_samples
            suit_correct += int(
                (
                    (predictions == targets) & suit_modes
                ).sum().item()
            )
            suit_samples += int(suit_modes.sum().item())
        logs.append(
            {
                "epoch": epoch,
                "objective": objective_sum / samples,
                "cross_entropy": cross_entropy_sum / samples,
                "distillation_kl": kl_sum / samples,
                "accuracy": correct / samples,
                "suit_accuracy": (
                    suit_correct / suit_samples
                    if suit_samples
                    else math.nan
                ),
                "samples": samples,
                "optimizer_steps": optimizer_steps,
                "temperature": temperature,
                "distillation_weight": distillation_weight,
                "mode_balanced": mode_balanced,
            }
        )
    return logs


def _batched_imitation_logits(
    policy: NativePufferTemporalPolicy,
    episodes: Sequence[Sequence[ImitationStep]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use PufferLib v4's native parallel MinGRU training path."""
    parameter = next(policy.parameters())
    packed_rows = []
    decision_indices = []
    legal_masks = []
    targets = []
    for episode in episodes:
        rows = []
        indices = []
        masks = []
        actions = []
        for sample in episode:
            rows.extend(policy._event_features(sample.decision))
            rows.append(policy._decision_features(sample.decision))
            indices.append(len(rows) - 1)
            masks.append(sample.decision.legal_actions)
            actions.append(sample.action)
        packed_rows.append(rows)
        decision_indices.append(indices)
        legal_masks.append(masks)
        targets.append(actions)
    max_length = max((len(rows) for rows in packed_rows), default=0)
    if max_length == 0:
        empty_logits = torch.empty(
            (0, 55), device=parameter.device, dtype=parameter.dtype
        )
        empty_targets = torch.empty(
            (0,), device=parameter.device, dtype=torch.long
        )
        return empty_logits, empty_targets
    token_batch = torch.zeros(
        len(episodes),
        max_length,
        100,
        device=parameter.device,
        dtype=parameter.dtype,
    )
    for batch_index, rows in enumerate(packed_rows):
        if rows:
            token_batch[batch_index, : len(rows)] = torch.tensor(
                rows, device=parameter.device, dtype=parameter.dtype
            )
    all_logits, _ = policy.forward_native_sequence(token_batch)
    all_logits = all_logits.reshape(len(episodes), max_length, 55)
    chosen_logits = []
    flat_targets = []
    for batch_index, indices in enumerate(decision_indices):
        if not indices:
            continue
        index_tensor = torch.tensor(
            indices, device=parameter.device, dtype=torch.long
        )
        logits = all_logits[batch_index, index_tensor]
        legal = torch.tensor(
            legal_masks[batch_index],
            device=parameter.device,
            dtype=torch.bool,
        )
        logits = logits.masked_fill(
            ~legal, torch.finfo(logits.dtype).min
        )
        chosen_logits.append(logits)
        flat_targets.extend(targets[batch_index])
    return (
        torch.cat(chosen_logits, dim=0),
        torch.tensor(
            flat_targets, device=parameter.device, dtype=torch.long
        ),
    )


def _snapshot(policy: NativePufferTemporalPolicy):
    snapshot = NativePufferTemporalPolicy(
        hidden_size=policy.hidden_size,
        num_layers=policy.num_layers,
        mode_conditioned_heads=getattr(
            policy, "mode_conditioned_heads", False
        ),
    )
    snapshot.load_state_dict(copy.deepcopy(policy.state_dict()))
    snapshot.eval()
    for parameter in snapshot.parameters():
        parameter.requires_grad_(False)
    return snapshot


def validate_league_mixture(
    mixture: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    """Validate a deterministic public-policy league weight specification."""
    normalized = tuple(mixture)
    names = [name for name, _ in normalized]
    if len(names) != len(set(names)):
        raise ValueError("league mixture contains a duplicate opponent type")
    unknown = sorted(set(names) - set(LEAGUE_OPPONENT_TYPES))
    if unknown:
        raise ValueError(
            f"unknown league opponent types: {', '.join(unknown)}"
        )
    if any(
        isinstance(weight, bool)
        or not isinstance(weight, int)
        or weight < 0
        for _, weight in normalized
    ):
        raise ValueError("league weights must be non-negative integers")
    if not any(weight for _, weight in normalized):
        raise ValueError("league mixture must have at least one positive weight")
    return normalized


def league_schedule(
    mixture: Sequence[tuple[str, int]] = LEAGUE_MIXTURE,
) -> tuple[str, ...]:
    normalized = validate_league_mixture(mixture)
    return tuple(
        name
        for name, count in normalized
        for _ in range(count)
    )


def collect_league_batch(
    policy: NativePufferTemporalPolicy,
    champion_factory,
    historical: Sequence[NativePufferTemporalPolicy],
    *,
    games: int,
    seed: int,
    schedule_offset: int = 0,
    mixture: Sequence[tuple[str, int]] = LEAGUE_MIXTURE,
    player_counts: Sequence[int] | None = None,
) -> tuple[list[tuple[Sequence[TrajectoryStep], bool]], dict]:
    schedule = league_schedule(mixture)
    player_count_schedule = tuple(player_counts or (2,))
    if not player_count_schedule or any(
        isinstance(count, bool) or not isinstance(count, int) or count < 2
        for count in player_count_schedule
    ):
        raise ValueError("player_counts must contain integers >= 2")
    episodes = []
    contributions = Counter()
    game_counts = Counter()
    player_counts_seen = Counter()
    deck_count_by_player_count = {}
    league_fallbacks = Counter()
    total_turns = 0
    for index in range(games):
        requested_opponent_type = schedule[
            (schedule_offset + index) % len(schedule)
        ]
        player_count = player_count_schedule[
            (schedule_offset + index) % len(player_count_schedule)
        ]
        opponent_type = requested_opponent_type
        if opponent_type == "bayesian_tree" and player_count >= 7:
            # The tree's hidden-card sampler models one physical deck.  Keep
            # seven/eight-player two-deck games valid by using the stronger
            # public-rule opponent in that narrow case.
            opponent_type = "smarter_rule"
            league_fallbacks["bayesian_tree_multideck_to_smarter_rule"] += 1
        deal_seed = seed + index
        np.random.seed(deal_seed)
        random.seed(deal_seed)
        current_zero = temporal_agent_factory(
            policy, deterministic=False, seed=deal_seed * 2
        )
        current_seat = index % player_count
        if opponent_type == "current_selfplay":
            factories = [
                (
                    current_zero
                    if seat == 0
                    else temporal_agent_factory(
                        policy,
                        deterministic=False,
                        seed=deal_seed * 2 + seat,
                    )
                )
                for seat in range(player_count)
            ]
            current_seats = tuple(range(player_count))
        else:
            if opponent_type == "historical_temporal":
                frozen = historical[index % len(historical)]
                opponent = temporal_agent_factory(
                    frozen, deterministic=False, seed=deal_seed + 99
                )
            elif opponent_type == "ppo_champion":
                opponent = champion_factory
            elif opponent_type == "bayesian_tree":
                opponent = tree_factory
            elif opponent_type == "smart_rule":
                opponent = SmartAgent
            elif opponent_type == "smarter_rule":
                opponent = SmarterAgent
            elif opponent_type == "random_rule":
                opponent = AgentRandom
            else:  # Defensive guard if schedule validation changes.
                raise RuntimeError(
                    f"unsupported league opponent: {opponent_type}"
                )
            factories = [
                current_zero if seat == current_seat else opponent
                for seat in range(player_count)
            ]
            current_seats = (current_seat,)

        match = game.Game(factories)
        turns, winner = match.auto_simulate()
        total_turns += turns
        game_counts[opponent_type] += 1
        player_counts_seen[str(player_count)] += 1
        observed_deck_count = int(match.deck_count)
        prior_deck_count = deck_count_by_player_count.setdefault(
            str(player_count), observed_deck_count
        )
        if prior_deck_count != observed_deck_count:
            raise RuntimeError(
                "deck count changed within a player-count batch: "
                f"players={player_count}, "
                f"first={prior_deck_count}, current={observed_deck_count}"
            )
        for seat in current_seats:
            trajectory = tuple(match.players[seat].trajectory)
            episodes.append((trajectory, winner == seat))
            contributions[opponent_type] += len(trajectory)
    metadata = {
        "games": dict(game_counts),
        "decisions": dict(contributions),
        "total_turns": total_turns,
        "trajectories": len(episodes),
    }
    if player_counts is not None:
        metadata["player_counts"] = dict(player_counts_seen)
        metadata["deck_count_by_player_count"] = deck_count_by_player_count
        if league_fallbacks:
            metadata["league_fallbacks"] = dict(league_fallbacks)
    return episodes, metadata


def paired_bootstrap_interval(
    pair_scores: Sequence[float],
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(pair_scores, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate_paired(
    candidate_factory,
    opponent_factory,
    *,
    pairs: int,
    seed: int,
    candidate_name: str,
    opponent_name: str,
) -> dict:
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
    return {
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
