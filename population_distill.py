#!/usr/bin/env python3
"""Privacy-safe population distillation for the 121-feature Pesten policy.

The frozen PPO teacher, every heuristic expert, and the student receive only the
acting player's own cards plus public counters.  Rollout storage contains only
that same observation.  Opponent card identities and draw-pile order are never
arguments to any labelling or policy function in this module.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import card
import deck
from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_END,
    COMPATIBLE_START,
    HAND_END,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PASS_ACTION,
    PLAYER_HAND_SIZES_END,
    PLAYER_HAND_SIZES_START,
    SORTS_PLAYED_START,
    SUIT_ACTION_END,
    SUIT_ACTION_START,
    populate_game_context,
)
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    POLICY_FIRST_LEGAL,
    POLICY_RANDOM,
    POLICY_RL,
    POLICY_SMART,
    collect_batched_rollout,
)
from ppo2 import StudentNet as FrozenPPOArchitecture
from ppo2 import build_action_masks


SEALED_FINAL_SEED_START = 92_000_000
DEFAULT_TRAIN_SEED = 81_000_000
ARCHITECTURES = {
    "compact": (128, 128),
    "wide": (256, 256, 128),
}


@dataclass(frozen=True)
class DistillConfig:
    seed: int = 731
    epochs: int = 5
    batch_size: int = 512
    learning_rate: float = 8e-4
    weight_decay: float = 1e-5
    teacher_weight: float = 0.70
    teacher_temperature: float = 0.85
    expert_temperature: float = 0.45


class PopulationStudent(nn.Module):
    """Small feed-forward student; no environment reference is retained."""

    def __init__(self, architecture: str = "compact") -> None:
        super().__init__()
        if architecture not in ARCHITECTURES:
            raise ValueError(f"unknown architecture {architecture!r}")
        widths = ARCHITECTURES[architecture]
        layers: list[nn.Module] = []
        input_width = OBSERVATION_SIZE
        for width in widths:
            layers.extend((nn.Linear(input_width, width), nn.LayerNorm(width), nn.SiLU()))
            input_width = width
        layers.append(nn.Linear(input_width, ACTION_COUNT))
        self.network = nn.Sequential(*layers)
        self.architecture = architecture

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


def validate_development_seed(seed: int, games: int = 1) -> None:
    """Reject the sealed final-evaluation namespace."""
    if seed < 0 or games < 1:
        raise ValueError("seed must be non-negative and games must be positive")
    if seed + games >= SEALED_FINAL_SEED_START:
        raise ValueError(
            f"development seeds must stay below {SEALED_FINAL_SEED_START:,}"
        )


def load_frozen_ppo(path: Path, device: torch.device) -> nn.Module:
    actor = FrozenPPOArchitecture().to(device)
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    actor.load_state_dict(state)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def build_public_observation(
    *,
    own_cards: Sequence[int],
    current_sort: str,
    current_rank: int,
    penalty_amount: int,
    public_sorts_played: Sequence[int],
    public_hand_sizes: Sequence[int],
    player_index: int,
    direction: int,
    change_suit_mode: bool = False,
) -> torch.Tensor:
    """Construct the legacy policy view from own cards and public facts only."""
    observation = torch.zeros(OBSERVATION_SIZE, dtype=torch.float32)
    for number in own_cards:
        if not 0 <= number < HAND_END:
            raise ValueError(f"invalid private own-card id {number}")
        observation[number] += 1
        held = card.Card(number)
        playable = (
            held.truenumber == current_rank
            if penalty_amount > 0
            else held.compatible(current_sort, current_rank)
        )
        if playable:
            observation[COMPATIBLE_START + number] += 1
    populate_game_context(
        observation,
        sorts_played=public_sorts_played,
        hand_sizes=public_hand_sizes,
        player_index=player_index,
        direction=direction,
        is_change_sort=change_suit_mode,
    )
    return observation


def legal_action_mask(observations: torch.Tensor) -> torch.Tensor:
    return build_action_masks(observations).bool()


def _card_tables(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ranks = torch.tensor(
        [card.Card(number).truenumber for number in range(HAND_END)],
        dtype=torch.long,
        device=device,
    )
    suits = torch.tensor(
        [card.sorts.index(card.Card(number).sort) for number in range(HAND_END)],
        dtype=torch.long,
        device=device,
    )
    return ranks, suits


def public_heuristic_logits(observations: torch.Tensor) -> torch.Tensor:
    """Return four diverse expert score tables using observation columns only."""
    if observations.ndim == 1:
        observations = observations.unsqueeze(0)
    if observations.shape[-1] != OBSERVATION_SIZE:
        raise ValueError("unexpected observation width")
    device = observations.device
    batch = observations.shape[0]
    ranks, suits = _card_tables(device)
    counts = observations[:, :HAND_END]
    hand_size = counts.sum(dim=-1)
    legal_cards = observations[:, COMPATIBLE_START:COMPATIBLE_END] > 0
    suit_history = observations[
        :, SORTS_PLAYED_START:SORTS_PLAYED_START + 5
    ]
    hand_slots = observations[
        :, PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_END
    ]
    opponent_slots = hand_slots[:, 1:]
    opponent_positive = torch.where(
        opponent_slots > 0,
        opponent_slots,
        torch.full_like(opponent_slots, 99),
    )
    nearest_threat = opponent_positive.min(dim=-1).values
    threat = (nearest_threat <= 2).float()
    suit_counts = torch.stack(
        [
            counts[:, suits == suit_index].sum(dim=-1)
            for suit_index in range(5)
        ],
        dim=-1,
    )

    expert_scores = torch.full(
        (4, batch, ACTION_COUNT),
        -8.0,
        dtype=observations.dtype,
        device=device,
    )
    special = torch.isin(
        ranks, torch.tensor((0, 1, 2, 7, 8, 13), device=device)
    ).float()
    attack = (
        (ranks == 0).float() * 2.8
        + (ranks == 2).float() * 1.6
        + (ranks == 8).float() * 1.3
        + (ranks == 7).float() * 1.0
        + (ranks == 1).float() * 0.5
        + (ranks == 13).float() * 0.4
    )
    duplicate_bonus = counts.clamp(max=3) * 0.18
    suit_public = suit_history[:, suits]
    suit_private = suit_counts[:, suits]

    # Shed cards quickly while respecting the special-card finish rule.
    expert_scores[0, :, :HAND_END] = 2.0 + duplicate_bonus
    expert_scores[0, :, :HAND_END] += 0.12 * suit_private
    expert_scores[0, :, :HAND_END] += 0.35 * attack

    # Pressure a publicly visible near-empty opponent.
    expert_scores[1, :, :HAND_END] = (
        1.0 + duplicate_bonus + threat[:, None] * attack
    )

    # Conserve flexible cards until they are tactically useful.
    expert_scores[2, :, :HAND_END] = (
        1.8 + 0.12 * suit_private - 0.50 * special
        + threat[:, None] * 0.8 * attack
    )

    # Prefer publicly depleted suits while retaining own suit concentration.
    expert_scores[3, :, :HAND_END] = (
        1.2 + 0.10 * suit_public + 0.16 * suit_private + 0.15 * attack
    )

    for expert in range(4):
        expert_scores[expert, :, PASS_ACTION] = -1.5

    one_card = hand_size <= 1.0
    final_special = one_card[:, None] & special.bool()[None, :]
    expert_scores[:, :, :HAND_END] = torch.where(
        final_special.unsqueeze(0),
        torch.full_like(expert_scores[:, :, :HAND_END], -12.0),
        expert_scores[:, :, :HAND_END],
    )
    last_card_is_special = one_card & ((counts * special).sum(dim=-1) > 0)
    expert_scores[:, :, PASS_ACTION] = torch.where(
        last_card_is_special,
        torch.full_like(hand_size, 4.0),
        expert_scores[:, :, PASS_ACTION],
    )

    suit_mode = observations[:, MODE_INDEX] != 0
    if torch.any(suit_mode):
        suit_choice_scores = suit_counts[:, :4] + 0.08 * suit_history[:, :4]
        expert_scores[:, suit_mode] = -8.0
        for expert in range(4):
            expert_scores[
                expert, suit_mode, SUIT_ACTION_START:SUIT_ACTION_END
            ] = suit_choice_scores[suit_mode]

    # Scores outside the mask are irrelevant, but setting them low makes the
    # expert tensors independently inspectable.
    mask = legal_action_mask(observations)
    expert_scores = expert_scores.masked_fill(~mask.unsqueeze(0), -1e9)
    return expert_scores


def _masked_distribution(
    logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    return torch.softmax(masked / temperature, dim=-1)


@torch.no_grad()
def population_targets(
    observations: torch.Tensor,
    frozen_teacher: nn.Module,
    *,
    teacher_weight: float = 0.70,
    teacher_temperature: float = 0.85,
    expert_temperature: float = 0.45,
) -> torch.Tensor:
    """Blend a frozen PPO distribution with four legal heuristic experts."""
    if not 0.0 <= teacher_weight <= 1.0:
        raise ValueError("teacher_weight must be in [0, 1]")
    mask = legal_action_mask(observations)
    teacher = _masked_distribution(
        frozen_teacher(observations),
        mask,
        teacher_temperature,
    )
    experts = public_heuristic_logits(observations)
    expert_mask = mask.unsqueeze(0).expand_as(experts)
    expert_probs = _masked_distribution(
        experts, expert_mask, expert_temperature
    ).mean(dim=0)
    target = teacher_weight * teacher + (1.0 - teacher_weight) * expert_probs
    target = target.masked_fill(~mask, 0.0)
    return target / target.sum(dim=-1, keepdim=True)


def _make_rollout_layout(
    games: int,
) -> tuple[np.ndarray, np.ndarray]:
    kinds = np.full((games, 2), POLICY_SMART, dtype=np.int8)
    keys = np.full((games, 2), None, dtype=object)
    opponents = (POLICY_SMART, POLICY_RANDOM, POLICY_FIRST_LEGAL, POLICY_RL)
    for game_index in range(games):
        learner_seat = game_index % 2
        opponent_seat = 1 - learner_seat
        kinds[game_index, learner_seat] = POLICY_RL
        keys[game_index, learner_seat] = "collector"
        opponent = opponents[game_index % len(opponents)]
        kinds[game_index, opponent_seat] = opponent
        if opponent == POLICY_RL:
            keys[game_index, opponent_seat] = "collector"
    return kinds, keys


def collect_public_observations(
    actor: nn.Module,
    *,
    games: int,
    seed: int,
    device: torch.device,
    greedy: bool = False,
    max_turns: int = 500,
) -> tuple[torch.Tensor, dict]:
    """Collect reachable safe observations against a broad fixed population."""
    validate_development_seed(seed, games)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    kinds, keys = _make_rollout_layout(games)
    engine = BatchedPesten(
        kinds,
        keys,
        np.arange(seed, seed + games, dtype=np.uint32),
        max_turns=max_turns,
    )
    rollout = collect_batched_rollout(
        engine,
        {
            "collector": ActorPolicy(
                actor=actor,
                device=device,
                temperature=0.85,
                collect_data=True,
                greedy=greedy,
            )
        },
        output_device=torch.device("cpu"),
    )
    metadata = {
        "games": games,
        "seed": seed,
        "observations": len(rollout.observations),
        "environment_turns": rollout.environment_turns,
        "completed_games": rollout.completed_games,
        "policy_decisions": rollout.policy_decisions,
        "inference_calls": rollout.inference_calls,
    }
    return rollout.observations, metadata


def _precompute_targets(
    observations: torch.Tensor,
    teacher: nn.Module,
    config: DistillConfig,
    device: torch.device,
) -> torch.Tensor:
    parts = []
    for start in range(0, len(observations), 2048):
        batch = observations[start:start + 2048].to(device)
        parts.append(
            population_targets(
                batch,
                teacher,
                teacher_weight=config.teacher_weight,
                teacher_temperature=config.teacher_temperature,
                expert_temperature=config.expert_temperature,
            ).cpu()
        )
    return torch.cat(parts)


def evaluate_distillation(
    student: nn.Module,
    observations: torch.Tensor,
    targets: torch.Tensor,
    teacher: nn.Module,
    device: torch.device,
) -> dict:
    student.eval()
    total_loss = total_population = total_teacher = total_illegal = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(observations), 2048):
            batch = observations[start:start + 2048].to(device)
            target = targets[start:start + 2048].to(device)
            mask = legal_action_mask(batch)
            student_logits = student(batch).masked_fill(
                ~mask, torch.finfo(batch.dtype).min
            )
            teacher_logits = teacher(batch).masked_fill(
                ~mask, torch.finfo(batch.dtype).min
            )
            log_probs = torch.log_softmax(student_logits, dim=-1)
            batch_count = len(batch)
            total_loss += float((-(target * log_probs).sum(-1)).sum().item())
            student_actions = student_logits.argmax(-1)
            total_population += float(
                (student_actions == target.argmax(-1)).sum().item()
            )
            total_teacher += float(
                (student_actions == teacher_logits.argmax(-1)).sum().item()
            )
            total_illegal += float(
                (~mask.gather(1, student_actions[:, None])).sum().item()
            )
            count += batch_count
    return {
        "samples": count,
        "cross_entropy": total_loss / max(1, count),
        "population_action_agreement": total_population / max(1, count),
        "teacher_action_agreement": total_teacher / max(1, count),
        "illegal_actions": int(total_illegal),
    }


def distill_student(
    observations: torch.Tensor,
    frozen_teacher: nn.Module,
    *,
    architecture: str,
    config: DistillConfig,
    device: torch.device,
) -> tuple[PopulationStudent, dict]:
    """Train one deterministic candidate and return held-out metrics."""
    if len(observations) < 8:
        raise ValueError("at least eight observations are required")
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    permutation = torch.randperm(len(observations), generator=generator)
    validation_count = max(1, len(observations) // 10)
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    targets = _precompute_targets(
        observations, frozen_teacher, config, device
    )
    student = PopulationStudent(architecture).to(device)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    started = time.perf_counter()
    curve = []
    for epoch in range(config.epochs):
        student.train()
        epoch_generator = torch.Generator().manual_seed(
            config.seed + epoch + 1
        )
        order = training_indices[
            torch.randperm(len(training_indices), generator=epoch_generator)
        ]
        loss_sum = 0.0
        seen = 0
        for start in range(0, len(order), config.batch_size):
            indices = order[start:start + config.batch_size]
            batch = observations[indices].to(device)
            target = targets[indices].to(device)
            mask = legal_action_mask(batch)
            logits = student(batch).masked_fill(
                ~mask, torch.finfo(batch.dtype).min
            )
            loss = (-(target * torch.log_softmax(logits, dim=-1)).sum(-1)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch)
            seen += len(batch)
        curve.append({"epoch": epoch + 1, "cross_entropy": loss_sum / seen})
    metrics = evaluate_distillation(
        student,
        observations[validation_indices],
        targets[validation_indices],
        frozen_teacher,
        device,
    )
    metrics.update(
        {
            "architecture": architecture,
            "parameters": sum(p.numel() for p in student.parameters()),
            "train_samples": len(training_indices),
            "wall_seconds": time.perf_counter() - started,
            "curve": curve,
        }
    )
    return student, metrics


class PopulationStudentAgent:
    """Legacy-game adapter; only this adapter sees explicitly public fields."""

    def __init__(self, game_instance, actor: nn.Module, device: torch.device):
        self.game = game_instance
        self.actor = actor
        self.device = device
        self.mydeck = deck.Deck([])
        self.type = "population_distilled"

    def _observation(self, change_suit_mode: bool = False) -> torch.Tensor:
        player_index = self.game.players.index(self)
        return build_public_observation(
            own_cards=tuple(held.number for held in self.mydeck.cards),
            current_sort=self.game.current_sort,
            current_rank=self.game.current_true_number,
            penalty_amount=self.game.penalty_amount,
            public_sorts_played=tuple(self.game.sorts_played),
            public_hand_sizes=tuple(
                player.mydeck.cardCount() for player in self.game.players
            ),
            player_index=player_index,
            direction=self.game.direction,
            change_suit_mode=change_suit_mode,
        )

    def _action(self, change_suit_mode: bool = False) -> int:
        observation = self._observation(change_suit_mode).to(self.device)
        with torch.no_grad():
            mask = legal_action_mask(observation)
            logits = self.actor(observation.unsqueeze(0)).squeeze(0)
            return int(
                logits.masked_fill(
                    ~mask, torch.finfo(logits.dtype).min
                ).argmax().item()
            )

    def playCard(self, current_sort, current_true_number):
        del current_sort, current_true_number
        action = self._action()
        return None if action == PASS_ACTION else card.Card(action)

    def changeSort(self):
        return card.sorts[self._action(True) - SUIT_ACTION_START]

    def addCard(self, held) -> None:
        if held is not None:
            self.mydeck.cards.append(held)

    def remove(self, held) -> None:
        for own in self.mydeck.cards:
            if own.number == held.number:
                self.mydeck.cards.remove(own)
                return


def population_agent_factory(actor: nn.Module, device: torch.device):
    def factory(game_instance):
        return PopulationStudentAgent(game_instance, actor, device)

    return factory


def save_checkpoint(
    path: Path,
    student: PopulationStudent,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "population-distill-v1",
            "observation_schema": "legacy-public-own-hand-121-v1",
            "architecture": student.architecture,
            "state_dict": student.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_population_student(
    path: Path, device: torch.device
) -> tuple[PopulationStudent, dict]:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("format") != "population-distill-v1":
        raise ValueError("unsupported population checkpoint")
    student = PopulationStudent(payload["architecture"]).to(device)
    student.load_state_dict(payload["state_dict"])
    student.eval()
    return student, payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher", type=Path, default=Path("ppo_actor_100run3.pth")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("population_student.pt")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("benchmarks/population_distill_training.json"),
    )
    parser.add_argument("--games", type=int, default=384)
    parser.add_argument("--dagger-games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=DEFAULT_TRAIN_SEED)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--teacher-weight", type=float, default=0.70)
    parser.add_argument("--teacher-temperature", type=float, default=0.85)
    parser.add_argument("--expert-temperature", type=float, default=0.45)
    parser.add_argument("--architectures", default="compact,wide")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_development_seed(args.seed, args.games + args.dagger_games + 10_000)
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    teacher = load_frozen_ppo(args.teacher, device)
    started = time.perf_counter()
    observations, initial_rollout = collect_public_observations(
        teacher,
        games=args.games,
        seed=args.seed,
        device=device,
    )
    config = DistillConfig(
        seed=args.seed % 1_000_000,
        epochs=args.epochs,
        teacher_weight=args.teacher_weight,
        teacher_temperature=args.teacher_temperature,
        expert_temperature=args.expert_temperature,
    )
    candidates = []
    models = {}
    for architecture in args.architectures.split(","):
        architecture = architecture.strip()
        model, metrics = distill_student(
            observations,
            teacher,
            architecture=architecture,
            config=config,
            device=device,
        )
        candidates.append(metrics)
        models[architecture] = model
    selected = max(
        candidates,
        key=lambda item: (
            item["teacher_action_agreement"],
            item["population_action_agreement"],
            -item["parameters"],
        ),
    )
    selected_model = models[selected["architecture"]]

    dagger_rollout = None
    if args.dagger_games:
        dagger_observations, dagger_rollout = collect_public_observations(
            selected_model,
            games=args.dagger_games,
            seed=args.seed + args.games + 1000,
            device=device,
        )
        observations = torch.cat((observations, dagger_observations), dim=0)
        selected_model, selected = distill_student(
            observations,
            teacher,
            architecture=selected["architecture"],
            config=config,
            device=device,
        )

    report = {
        "schema_version": 1,
        "method": "frozen PPO plus four public heuristic experts and DAgger",
        "privacy_boundary": (
            "121 features: own cards plus public counters; no opponent-card "
            "identity or draw-pile order"
        ),
        "teacher": str(args.teacher),
        "device": str(device),
        "seed": args.seed,
        "sealed_seed_start": SEALED_FINAL_SEED_START,
        "initial_rollout": initial_rollout,
        "dagger_rollout": dagger_rollout,
        "config": asdict(config),
        "candidate_metrics_before_dagger": candidates,
        "selected_metrics": selected,
        "total_observations": len(observations),
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint": str(args.output),
    }
    save_checkpoint(args.output, selected_model, report)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
