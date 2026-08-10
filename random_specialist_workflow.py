"""Phase-zero workflow for an independently trained random-opponent specialist.

This module deliberately stops before candidate training.  It establishes a
precommitted, locally seeded uniform-random opponent, proves scalar/batched
parity for that contract, measures the recurrent lane on the real CUDA path,
and freezes the later campaign rules.  The legacy ``AgentRandom`` remains a
separate smoke-only reference and is never silently substituted for the
audited opponent.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import card
import deck
import game
from batched_pesten import PHASE_CARD, PHASE_DRAW_RETRY, PHASE_SUIT
from randomAgent import AgentRandom
from specialist_seed_registry import (
    AllocationRequest,
    CONSUMED_LEGACY_SEALED_INTERVAL,
    SeedInterval,
    SeedRegistry,
)
from temporal_batched_speed import (
    AUDITED_UNIFORM_RANDOM_CONTRACT,
    BatchRolloutStats,
    collect_league_batch_batched,
    ordered_legal_actions,
    sample_audited_uniform_action,
)
from temporal_observation import (
    PASS_ACTION,
    SUIT_ACTION_START,
    DecisionInput,
    assert_policy_schema_is_hidden_state_free,
    build_draw_retry_legal_actions,
    build_legal_actions,
    serialize_safe_history,
)
from temporal_puffer_agent import PolicyStep, TrajectoryStep, temporal_agent_factory


CAMPAIGN_ID = "random_specialist_20260802"
CAMPAIGN_FORMAT = "pesten-random-specialist-frozen-campaign-v1"
CALIBRATION_FORMAT = "pesten-random-specialist-calibration-v1"
DEFAULT_OUTPUT_DIR = Path("benchmarks") / CAMPAIGN_ID
REGISTRY_NAME = "seed_registry.jsonl"
CALIBRATION_NAME = "calibration.json"
FROZEN_MANIFEST_NAME = "frozen_campaign.json"

PARITY_GAMES = 8
LEGACY_SMOKE_GAMES = 8
CALIBRATION_GAMES = 32
CALIBRATION_MAX_TURNS = 1_000
DEVELOPMENT_PAIRS_PER_CHECKPOINT = 250
CONFIRMATION_PAIRS = 2_500
FINAL_AUDIT_PAIRS = 5_000
FINAL_AUDIT_BOOTSTRAP_SAMPLES = 100_000

SOURCE_FILES = (
    "temporal_batched_speed.py",
    "random_specialist_workflow.py",
    "specialist_seed_registry.py",
    "batched_pesten.py",
    "game.py",
    "deck.py",
    "randomAgent.py",
    "temporal_observation.py",
    "temporal_puffer_agent.py",
    "temporal_puffer_v4.py",
    "temporal_training.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def campaign_source_hashes(root: Path | None = None) -> dict[str, str]:
    base = Path.cwd() if root is None else Path(root)
    return {
        name: file_sha256(base / name)
        for name in SOURCE_FILES
    }


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a frozen/calibration record once, durably, without overwrite."""

    rendered = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # An exclusive creation itself makes a purportedly frozen artifact
        # non-retryable until a human has inspected it.  Do not delete it.
        raise


def _state_sha256(state: Any) -> str:
    """Hash policy state for reset diagnostics without serializing it as input."""

    parts = (state,) if isinstance(state, torch.Tensor) else tuple(state)
    digest = hashlib.sha256()
    for part in parts:
        value = part.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _normalise_public_history(history: Sequence[Mapping[str, Any]]) -> tuple[tuple, ...]:
    result = []
    for item in history:
        action = item.get("action")
        player = int(item.get("player", -1))
        if action == "play":
            result.append(("play", player, int(item["card"])))
        elif action == "choose_sort":
            result.append(
                ("choose_sort", player, card.sorts.index(str(item["sort"])))
            )
    return tuple(result)


def _trajectory_rewards(
    trajectory: Sequence[TrajectoryStep],
    *,
    won: bool,
) -> list[float]:
    """The reward-only lane's immediate reward sequence for one trajectory."""

    rewards = [-0.01] * len(trajectory)
    if rewards:
        rewards[-1] += 1.0 if won else -1.0
    return rewards


def uniform_action_kernel(legal_actions: Sequence[bool]) -> dict[int, float]:
    """Return the exact uniform action kernel defined by the audit contract."""

    actions = ordered_legal_actions(legal_actions)
    probability = 1.0 / len(actions)
    return {action: probability for action in actions}


def audited_uniform_contract() -> dict[str, Any]:
    """Return the immutable scalar/batched opponent contract description."""

    return {
        **AUDITED_UNIFORM_RANDOM_CONTRACT,
        "name": "audited uniform-random opponent",
        "legacy_agent_random": False,
        "deal_rng_independence": (
            "Each game receives a separately registered opponent-RNG seed; "
            "the sampler owns a local RNG and never advances deck RNG state."
        ),
        "explicit_reset": (
            "The scalar adapter reinstantiates its local RNG at every Game "
            "reset; the batched collector creates one local RNG per game."
        ),
    }


def audited_uniform_agent_factory(
    opponent_seed: int,
    *,
    action_trace: list[dict] | None = None,
):
    """Build the scalar mirror of the collector's audited random opponent.

    The adapter accesses its own hand only to construct an action mask.  It
    is an environment opponent, not a neural policy, and it neither receives
    nor writes a ``DecisionInput``.
    """

    if not 0 <= int(opponent_seed) < 2**32:
        raise ValueError("audited opponent seed must be uint32-compatible")
    fixed_seed = int(opponent_seed)

    class AuditedUniformRandomAgent:
        type = "audited_uniform_random"
        contract_id = AUDITED_UNIFORM_RANDOM_CONTRACT["id"]

        def __init__(self, game_instance) -> None:
            self.game = game_instance
            self.mydeck = deck.Deck([])
            self._rng_seed = fixed_seed
            self._rng = np.random.RandomState(fixed_seed)
            self._episode_marker: tuple[int, int] | None = None
            self._last_pass_turn: int | None = None
            self._retry_card: int | None = None
            self._retry_turn: int | None = None
            self.action_history: list[dict] = []

        def reset_episode(self) -> None:
            self._rng = np.random.RandomState(self._rng_seed)
            self._last_pass_turn = None
            self._retry_card = None
            self._retry_turn = None
            self.action_history.clear()

        def _ensure_episode(self) -> None:
            marker = (id(self.game.grab_deck), id(self.game.game_deck))
            if marker == self._episode_marker:
                return
            self._episode_marker = marker
            self.reset_episode()

        def _retry_card_for_current_turn(self) -> int | None:
            if self._retry_turn != self.game.turn_count:
                self._retry_card = None
                self._retry_turn = None
            return self._retry_card

        def _legal_actions(self, *, change_suit_mode: bool = False) -> tuple[bool, ...]:
            self._ensure_episode()
            if change_suit_mode:
                return build_legal_actions(
                    own_cards=(),
                    current_suit=self.game.current_sort,
                    current_rank=self.game.current_true_number,
                    penalty_amount=self.game.penalty_amount,
                    change_suit_mode=True,
                )
            retry_card = self._retry_card_for_current_turn()
            if retry_card is not None:
                return build_draw_retry_legal_actions(retry_card)
            return build_legal_actions(
                own_cards=tuple(value.number for value in self.mydeck.cards),
                current_suit=self.game.current_sort,
                current_rank=self.game.current_true_number,
                penalty_amount=self.game.penalty_amount,
            )

        def _record(
            self,
            *,
            phase: int,
            action: int,
            legal_actions: Sequence[bool],
        ) -> None:
            try:
                seat = self.game.players.index(self)
            except ValueError:
                seat = -1
            record = {
                "seat": seat,
                "role": "audited_uniform_random",
                "phase": phase,
                "action": int(action),
                "legal_actions": list(legal_actions),
            }
            self.action_history.append(record)
            if action_trace is not None:
                action_trace.append(record)

        def playCard(self, current_sort, current_true_number):
            del current_sort, current_true_number
            retry_active = self._retry_card_for_current_turn() is not None
            legal_actions = self._legal_actions()
            action = sample_audited_uniform_action(legal_actions, self._rng)
            self._record(
                phase=PHASE_DRAW_RETRY if retry_active else PHASE_CARD,
                action=action,
                legal_actions=legal_actions,
            )
            if action == PASS_ACTION:
                self._last_pass_turn = (
                    None
                    if retry_active or self.game.penalty_amount > 0
                    else self.game.turn_count
                )
                self._retry_card = None
                self._retry_turn = None
                return None
            self._last_pass_turn = None
            self._retry_card = None
            self._retry_turn = None
            return card.Card(action)

        def changeSort(self):
            legal_actions = self._legal_actions(change_suit_mode=True)
            action = sample_audited_uniform_action(legal_actions, self._rng)
            self._record(
                phase=PHASE_SUIT,
                action=action,
                legal_actions=legal_actions,
            )
            return card.sorts[action - SUIT_ACTION_START]

        def addCard(self, drawn_card) -> None:
            if drawn_card is None:
                return
            # ``Game.reset`` clears hands before dealing.  Reset the local RNG
            # at that game boundary, not after any prior game history.
            if (
                not self.mydeck.cards
                and self.game.turn_count == 0
                and not self.game.public_history
            ):
                marker = (id(self.game.grab_deck), id(self.game.game_deck))
                if marker != self._episode_marker:
                    self._episode_marker = marker
                    self.reset_episode()
            self.mydeck.cards.append(drawn_card)
            if (
                self._last_pass_turn == self.game.turn_count
                and self.game.penalty_amount == 0
            ):
                self._retry_card = int(drawn_card.number)
                self._retry_turn = self.game.turn_count
                self._last_pass_turn = None

        def remove(self, played_card) -> None:
            for own_card in self.mydeck.cards:
                if own_card.number == played_card.number:
                    self.mydeck.cards.remove(own_card)
                    return

    return AuditedUniformRandomAgent


def _trace_temporal_factory(
    policy: torch.nn.Module,
    *,
    seed: int,
    action_trace: list[dict],
):
    """Wrap the existing safe temporal adapter only to record comparisons."""

    base = temporal_agent_factory(policy, deterministic=False, seed=seed)

    class TraceTemporalAgent(base):
        def _record_last_action(self, phase: int) -> None:
            step = self.trajectory[-1]
            action_trace.append(
                {
                    "seat": self.game.players.index(self),
                    "role": "current",
                    "phase": phase,
                    "action": int(step.action),
                    "legal_actions": list(step.decision.legal_actions),
                }
            )

        def playCard(self, current_sort, current_true_number):
            retry_active = self._retry_card_for_current_turn() is not None
            result = super().playCard(current_sort, current_true_number)
            self._record_last_action(
                PHASE_DRAW_RETRY if retry_active else PHASE_CARD
            )
            return result

        def changeSort(self):
            result = super().changeSort()
            self._record_last_action(PHASE_SUIT)
            return result

    return TraceTemporalAgent


def _initial_scalar_deal(match: game.Game) -> dict[str, Any]:
    assert match.game_deck is not None
    return {
        "hands": [
            [held.number for held in player.mydeck.cards]
            for player in match.players
        ],
        "opening_card": int(match.game_deck.topCard().number),
    }


def collect_scalar_uniform_trace(
    policy: torch.nn.Module,
    *,
    absolute_game_index: int,
    deal_seed: int,
    opponent_seed: int,
    max_turns: int,
) -> dict[str, Any]:
    """Run one scalar Game against the audited scalar opponent mirror."""

    action_trace: list[dict] = []
    np.random.seed(deal_seed)
    random.seed(deal_seed)
    current_factory = _trace_temporal_factory(
        policy,
        seed=deal_seed * 2,
        action_trace=action_trace,
    )
    opponent_factory = audited_uniform_agent_factory(
        opponent_seed,
        action_trace=action_trace,
    )
    current_seat = absolute_game_index % 2
    factories = (
        [current_factory, opponent_factory]
        if current_seat == 0
        else [opponent_factory, current_factory]
    )
    match = game.Game(factories)
    match.reset()
    initial_deal = _initial_scalar_deal(match)
    candidate = match.players[current_seat]
    initial_state_sha256 = _state_sha256(candidate.state)
    while not match.game_over and match.turn_count < max_turns:
        match.simulate_turn()
    winner = -1 if match.winner is None else int(match.winner)
    terminal_turn_count = int(match.turn_count)
    trajectory = tuple(candidate.trajectory)
    won = winner == current_seat
    public_events = _normalise_public_history(match.public_history)
    final_state_sha256 = _state_sha256(candidate.state)
    safe_history_sha256 = _sha256_bytes(
        serialize_safe_history(candidate.safe_history)
    )

    # Exercise the same agent instance through another Game.reset.  This is
    # stronger than merely constructing a second agent: leakage across games
    # would leave a different state hash here.
    match.reset()
    reset_state_sha256 = _state_sha256(candidate.state)
    return {
        "absolute_game_index": absolute_game_index,
        "deal_seed": int(deal_seed),
        "opponent_rng_seed": int(opponent_seed),
        "initial_deal": initial_deal,
        "actions": action_trace,
        "winner": winner,
        "turn_count": terminal_turn_count,
        "public_events": public_events,
        "trajectory_actions": [int(step.action) for step in trajectory],
        "trajectory_legal_actions": [
            list(step.decision.legal_actions) for step in trajectory
        ],
        "trajectory_rewards": _trajectory_rewards(trajectory, won=won),
        "safe_history_sha256": safe_history_sha256,
        "recurrent": {
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "reset_state_sha256": reset_state_sha256,
            "reset_verified": reset_state_sha256 == initial_state_sha256,
            "decision_count": len(trajectory),
        },
    }


def _batch_trace_with_rewards(
    trace: Mapping[str, Any],
    episode: tuple[Sequence[TrajectoryStep], bool],
) -> dict[str, Any]:
    trajectory, won = episode
    recurrent = list(trace.get("recurrent", ()))
    if len(recurrent) != 1:
        raise RuntimeError("random parity expects exactly one recurrent seat")
    record = recurrent[0]
    return {
        "absolute_game_index": int(trace["absolute_game_index"]),
        "deal_seed": int(trace["deal_seed"]),
        "opponent_rng_seed": int(trace["opponent_rng_seed"]),
        "initial_deal": trace["initial_deal"],
        "actions": list(trace["actions"]),
        "winner": int(trace["winner"]),
        "turn_count": int(trace["turn_count"]),
        "public_events": tuple(trace["public_events"]),
        "trajectory_actions": [int(step.action) for step in trajectory],
        "trajectory_legal_actions": [
            list(step.decision.legal_actions) for step in trajectory
        ],
        "trajectory_rewards": _trajectory_rewards(trajectory, won=bool(won)),
        "safe_history_sha256": _sha256_bytes(
            serialize_safe_history(
                [step.decision for step in trajectory]
            )
        ),
        "recurrent": {
            "initial_state_sha256": record["initial_state_sha256"],
            "final_state_sha256": record["final_state_sha256"],
            # Every independently allocated vector game is initialized from
            # a fresh policy state.  The scalar side additionally checks an
            # in-place Game.reset on the same agent instance.
            "reset_verified": True,
            "decision_count": int(record["decision_count"]),
        },
    }


PARITY_FIELDS = (
    "initial_deal",
    "actions",
    "winner",
    "turn_count",
    "public_events",
    "trajectory_actions",
    "trajectory_legal_actions",
    "trajectory_rewards",
    "safe_history_sha256",
)


def _parity_projection(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Canonical trace subset whose digest is meaningful across adapters."""

    recurrent = trace["recurrent"]
    return {
        **{field: trace[field] for field in PARITY_FIELDS},
        "recurrent": {
            "initial_state_sha256": recurrent["initial_state_sha256"],
            "final_state_sha256": recurrent["final_state_sha256"],
            "decision_count": recurrent["decision_count"],
        },
    }


def run_fixed_random_parity_gate(
    policy: torch.nn.Module,
    *,
    deal_seeds: Sequence[int],
    opponent_seeds: Sequence[int],
    max_turns: int = CALIBRATION_MAX_TURNS,
) -> dict[str, Any]:
    """Compare scalar and batched traces for the audited opponent contract.

    The deal range must be consecutive because the public collector's seed
    interface is ``seed + absolute_game_index``.  Opponent RNG seeds remain a
    separate explicit sequence and are never derived from policy inputs.
    """

    deal_seeds = tuple(int(value) for value in deal_seeds)
    opponent_seeds = tuple(int(value) for value in opponent_seeds)
    if not deal_seeds or len(deal_seeds) != len(opponent_seeds):
        raise ValueError("parity gate needs equally sized non-empty seed banks")
    if deal_seeds != tuple(
        range(deal_seeds[0], deal_seeds[0] + len(deal_seeds))
    ):
        raise ValueError("parity deal seeds must be one consecutive interval")
    if any(not 0 <= value < 2**32 for value in (*deal_seeds, *opponent_seeds)):
        raise ValueError("parity seeds must be uint32-compatible")

    policy.eval()
    scalar = [
        collect_scalar_uniform_trace(
            policy,
            absolute_game_index=index,
            deal_seed=deal_seed,
            opponent_seed=opponent_seeds[index],
            max_turns=max_turns,
        )
        for index, deal_seed in enumerate(deal_seeds)
    ]
    batched_raw: list[dict] = []
    batched_episodes, metadata = collect_league_batch_batched(
        policy,
        None,
        (),
        games=len(deal_seeds),
        seed=deal_seeds[0],
        schedule=("random",),
        random_opponent_seeds=opponent_seeds,
        max_turns=max_turns,
        trace_sink=batched_raw,
    )
    if len(batched_raw) != len(deal_seeds) or len(batched_episodes) != len(deal_seeds):
        raise RuntimeError("batched random parity trace omitted a game")
    raw_by_index = {
        int(trace["absolute_game_index"]): trace for trace in batched_raw
    }
    batched = [
        _batch_trace_with_rewards(raw_by_index[index], batched_episodes[index])
        for index in range(len(deal_seeds))
    ]

    cases = []
    for scalar_trace, batched_trace in zip(scalar, batched):
        mismatches = [
            field
            for field in PARITY_FIELDS
            if scalar_trace[field] != batched_trace[field]
        ]
        scalar_recurrent = scalar_trace["recurrent"]
        batched_recurrent = batched_trace["recurrent"]
        reset_matches = (
            scalar_recurrent["reset_verified"]
            and batched_recurrent["reset_verified"]
            and scalar_recurrent["initial_state_sha256"]
            == batched_recurrent["initial_state_sha256"]
            and scalar_recurrent["final_state_sha256"]
            == batched_recurrent["final_state_sha256"]
            and scalar_recurrent["decision_count"]
            == batched_recurrent["decision_count"]
        )
        if not reset_matches:
            mismatches.append("recurrent_reset_or_state")
        cases.append(
            {
                "absolute_game_index": scalar_trace["absolute_game_index"],
                "deal_seed": scalar_trace["deal_seed"],
                "opponent_rng_seed": scalar_trace["opponent_rng_seed"],
                "passed": not mismatches,
                "mismatches": mismatches,
                "scalar_trace_sha256": _sha256_bytes(
                    json.dumps(
                        _parity_projection(scalar_trace),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
                "batched_trace_sha256": _sha256_bytes(
                    json.dumps(
                        _parity_projection(batched_trace),
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ),
                "deal": scalar_trace["initial_deal"],
                "turn_count": scalar_trace["turn_count"],
                "winner": scalar_trace["winner"],
                "recurrent_reset_verified": reset_matches,
            }
        )
    return {
        "name": "audited_uniform_random_scalar_batched_exact_parity",
        "opponent": audited_uniform_contract(),
        "metadata": metadata,
        "checked_fields": [*PARITY_FIELDS, "recurrent_reset_or_state"],
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def _uniform_distribution_guard(seed: int) -> dict[str, Any]:
    """Empirically guard the exact mathematical uniform-action kernel."""

    legal = tuple(index in (3, 11, 23, PASS_ACTION) for index in range(55))
    kernel = uniform_action_kernel(legal)
    rng = np.random.RandomState(seed)
    draws = 20_000
    counts = {action: 0 for action in kernel}
    for _ in range(draws):
        counts[sample_audited_uniform_action(legal, rng)] += 1
    empirical = {action: count / draws for action, count in counts.items()}
    maximum_absolute_error = max(
        abs(empirical[action] - probability)
        for action, probability in kernel.items()
    )
    return {
        "legal_actions": list(kernel),
        "exact_kernel": kernel,
        "draws": draws,
        "counts": counts,
        "empirical_kernel": empirical,
        "maximum_absolute_error": maximum_absolute_error,
        # The expected one-sigma binomial deviation here is about 0.0031; a
        # 0.02 guard leaves a broad deterministic safety margin while still
        # catching a biased or missing action branch.
        "passed": maximum_absolute_error <= 0.02,
    }


def _legacy_agent_random_smoke(
    *,
    deal_seeds: Sequence[int],
    opponent_seeds: Sequence[int],
    max_turns: int,
) -> dict[str, Any]:
    """Exercise legacy AgentRandom separately without claiming trace parity."""

    completed = 0
    turns = []
    for index, deal_seed in enumerate(deal_seeds):
        np.random.seed(int(deal_seed))
        random.seed(int(deal_seed))
        audited = audited_uniform_agent_factory(int(opponent_seeds[index]))
        factories = (
            [audited, AgentRandom]
            if index % 2 == 0
            else [AgentRandom, audited]
        )
        match = game.Game(factories)
        game_turns, _ = match.auto_simulate(max_turns=max_turns)
        completed += int(match.game_over or game_turns >= max_turns)
        turns.append(int(game_turns))
    return {
        "name": "legacy_AgentRandom_smoke_only",
        "legacy_agent_random_source": "randomAgent.AgentRandom",
        "games": len(deal_seeds),
        "completed_or_explicitly_truncated": completed,
        "maximum_turns": max(turns, default=0),
        "comparison_is_not_exact_parity": True,
        "comparison_is_not_strength_or_selection_evidence": True,
        "passed": completed == len(deal_seeds),
    }


def _finite_metrics(value: Mapping[str, Any]) -> bool:
    for child in value.values():
        if isinstance(child, Mapping):
            if not _finite_metrics(child):
                return False
        elif isinstance(child, float) and not math.isfinite(child):
            return False
    return True


def _native_runtime_facts(device: torch.device) -> dict[str, Any]:
    facts = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        facts.update(
            {
                "cuda_device_index": device.index,
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_total_memory_bytes": int(properties.total_memory),
                "cuda_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
            }
        )
    return facts


def _load_native_temporal_policy():
    """Keep the non-Puffer test environment importable until calibration."""

    from temporal_puffer_v4 import (  # pylint: disable=import-outside-toplevel
        NativePufferTemporalPolicy,
        PUFFERLIB_REVISION,
    )
    import pufferlib  # pylint: disable=import-outside-toplevel

    return NativePufferTemporalPolicy, PUFFERLIB_REVISION, pufferlib


class _PhaseZeroTraceBackend:
    """Tiny deterministic recurrent backend used only for parity evidence."""

    def __init__(self, owner: "PhaseZeroTracePolicy") -> None:
        self.owner = owner

    def forward_eval(
        self,
        tokens: torch.Tensor,
        state: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        if len(state) != 1:
            raise ValueError("phase-zero trace policy expects one state tensor")
        batch_size = tokens.shape[0]
        old = state[0]
        if old.shape != (1, batch_size, 2):
            raise ValueError("phase-zero trace state has an unexpected shape")
        values = tokens.reshape(batch_size, -1).sum(dim=1)
        delta = values.reshape(1, batch_size, 1)
        next_state = old + torch.cat((delta, delta * 0.5), dim=-1)
        action_axis = torch.arange(
            55,
            device=tokens.device,
            dtype=tokens.dtype,
        ).unsqueeze(0)
        logits = action_axis * 0.001 + next_state[0, :, :1] * 0.0001
        critic = next_state[0, :, 1:2]
        return logits, critic, (next_state,)


class PhaseZeroTracePolicy(torch.nn.Module):
    """Safe recurrent fixture that makes reset leakage observable in parity."""

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.puffer_policy = _PhaseZeroTraceBackend(self)

    def initial_state(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, ...]:
        parameter = self.anchor
        return (
            torch.zeros(
                (1, batch_size, 2),
                device=device or parameter.device,
                dtype=parameter.dtype,
            ),
        )

    @staticmethod
    def _event_features(decision: DecisionInput) -> list[tuple[float, float]]:
        return [
            (
                float(sum(event.as_tuple())),
                float(event.count),
            )
            for event in decision.public_events
        ]

    @staticmethod
    def _decision_features(decision: DecisionInput) -> tuple[float, float]:
        return (
            float(sum(decision.observation.values)),
            float(sum(decision.legal_actions)),
        )

    def step(
        self,
        decision: DecisionInput,
        state: tuple[torch.Tensor, ...],
    ) -> PolicyStep:
        if not isinstance(decision, DecisionInput):
            raise TypeError("phase-zero trace policy accepts only DecisionInput")
        parameter = self.anchor
        next_state = state
        for row in self._event_features(decision):
            token = torch.tensor(
                row,
                device=parameter.device,
                dtype=parameter.dtype,
            ).unsqueeze(0)
            _, _, next_state = self.puffer_policy.forward_eval(token, next_state)
        token = torch.tensor(
            self._decision_features(decision),
            device=parameter.device,
            dtype=parameter.dtype,
        ).unsqueeze(0)
        logits, values, next_state = self.puffer_policy.forward_eval(token, next_state)
        legal = torch.tensor(
            decision.legal_actions,
            device=parameter.device,
            dtype=torch.bool,
        ).unsqueeze(0)
        return PolicyStep(
            logits=logits.masked_fill(~legal, torch.finfo(logits.dtype).min)[0],
            value=values.reshape(-1)[0],
            state=next_state,
        )


def _privacy_boundary_report() -> dict[str, Any]:
    """Freeze the information boundary before any calibration rollout."""

    assert_policy_schema_is_hidden_state_free()
    safe_schema_hash = _sha256_bytes(
        json.dumps(
            {
                "decision_input_only": True,
                "policy_inputs": [
                    "DecisionInput.observation",
                    "DecisionInput.legal_actions",
                    "DecisionInput.public_events",
                ],
                "forbidden": [
                    "opponent_card_identity",
                    "draw_pile_order",
                    "privileged_critic",
                    "ppo_checkpoint",
                    "imitation_data",
                    "teacher_logits",
                    "parent_weights",
                    "opponent_rng_seed",
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    return {
        "passed": True,
        "safe_schema_sha256": safe_schema_hash,
        "native_policy_input_boundary": "DecisionInput plus public event tokens only",
        "prohibited_training_sources": [
            "PPO checkpoint",
            "imitation data",
            "teacher logits",
            "parent weights",
            "privileged critic",
        ],
    }


def _check_native_decision_input_only(policy: torch.nn.Module) -> bool:
    """Assert the native policy rejects an object outside the safe boundary."""

    try:
        policy.step(object(), policy.initial_state())
    except TypeError:
        return True
    return False


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(child) for child in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return True


def _calibration_registry_requests() -> tuple[AllocationRequest, ...]:
    total_games = PARITY_GAMES + LEGACY_SMOKE_GAMES + CALIBRATION_GAMES
    return (
        AllocationRequest(
            "calibration",
            total_games,
            "phase-zero scalar/batched parity, legacy smoke, and recurrent calibration deals",
        ),
        AllocationRequest(
            "calibration_opponent_rng",
            total_games,
            "separate local audited-opponent RNG seeds for every calibration deal",
        ),
        AllocationRequest(
            "calibration_contract_rng",
            1,
            "uniform action-kernel distributional guard RNG seed",
        ),
        AllocationRequest(
            "calibration_initialization",
            1,
            "discarded random initialization for recurrent calibration only",
        ),
        AllocationRequest(
            "calibration_update_rng",
            1,
            "discarded recurrent calibration PPO minibatch order seed",
        ),
    )


def _interval_values(interval: SeedInterval) -> tuple[int, ...]:
    return tuple(range(interval.start_inclusive, interval.end_exclusive))


def run_calibration(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Run one dedicated phase-zero calibration, never candidate training."""

    output_dir = Path(output_dir)
    calibration_path = output_dir / CALIBRATION_NAME
    if calibration_path.exists():
        raise FileExistsError(
            f"calibration artifact already exists and its seed bank is burned: {calibration_path}"
        )
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable to this process")
    if device.type != "cuda":
        raise RuntimeError(
            "phase-zero calibration requires CUDA update evidence; refuse a CPU substitute"
        )

    NativePufferTemporalPolicy, puffer_revision, pufferlib = _load_native_temporal_policy()
    registry = SeedRegistry(output_dir / REGISTRY_NAME)
    allocations = registry.allocate_many(_calibration_registry_requests())
    deal_seeds = _interval_values(allocations["calibration"])
    opponent_seeds = _interval_values(allocations["calibration_opponent_rng"])
    parity_deals = deal_seeds[:PARITY_GAMES]
    parity_opponents = opponent_seeds[:PARITY_GAMES]
    legacy_offset = PARITY_GAMES
    calibration_offset = PARITY_GAMES + LEGACY_SMOKE_GAMES

    parity_policy = PhaseZeroTracePolicy().eval()
    parity = run_fixed_random_parity_gate(
        parity_policy,
        deal_seeds=parity_deals,
        opponent_seeds=parity_opponents,
        max_turns=CALIBRATION_MAX_TURNS,
    )
    uniform_guard = _uniform_distribution_guard(
        allocations["calibration_contract_rng"].start_inclusive
    )
    legacy_smoke = _legacy_agent_random_smoke(
        deal_seeds=deal_seeds[
            legacy_offset : legacy_offset + LEGACY_SMOKE_GAMES
        ],
        opponent_seeds=opponent_seeds[
            legacy_offset : legacy_offset + LEGACY_SMOKE_GAMES
        ],
        max_turns=CALIBRATION_MAX_TURNS,
    )
    privacy = _privacy_boundary_report()

    initialization_seed = allocations["calibration_initialization"].start_inclusive
    update_seed = allocations["calibration_update_rng"].start_inclusive
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    random.seed(initialization_seed)
    policy = NativePufferTemporalPolicy(
        hidden_size=96,
        num_layers=1,
        mode_conditioned_heads=False,
    ).to(device)
    native_decision_input_only = _check_native_decision_input_only(policy)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    rollout_stats = BatchRolloutStats()
    rollout_started = time.perf_counter()
    policy.eval()
    episodes, contribution = collect_league_batch_batched(
        policy,
        None,
        (),
        games=CALIBRATION_GAMES,
        seed=deal_seeds[calibration_offset],
        schedule=("random",),
        random_opponent_seeds=opponent_seeds[
            calibration_offset : calibration_offset + CALIBRATION_GAMES
        ],
        max_turns=CALIBRATION_MAX_TURNS,
        stats=rollout_stats,
    )
    torch.cuda.synchronize(device)
    rollout_seconds = time.perf_counter() - rollout_started
    legal_actions = all(
        step.decision.legal_actions[step.action]
        for trajectory, _ in episodes
        for step in trajectory
    )
    policy.train()
    from temporal_training import (  # pylint: disable=import-outside-toplevel
        ppo_update_batched_epochs,
    )

    torch.cuda.synchronize(device)
    update_started = time.perf_counter()
    update_metrics = ppo_update_batched_epochs(
        policy,
        episodes,
        optimizer,
        update_epochs=1,
        minibatch_trajectories=8,
        seed=update_seed,
        target_kl=None,
        clip_ratio=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        gamma=0.97,
        gae_lambda=0.95,
        normalize_advantages=True,
    )
    torch.cuda.synchronize(device)
    update_seconds = time.perf_counter() - update_started
    end_to_end_seconds = rollout_seconds + update_seconds
    finite_parameters = all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in policy.parameters()
    )
    finite_metrics = _all_finite(update_metrics)
    runtime = _native_runtime_facts(device)
    runtime.update(
        {
            "pufferlib": str(pufferlib.__version__),
            "pufferlib_revision": puffer_revision,
            "update_parameter_device": str(next(policy.parameters()).device),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cpu_responsibilities": [
                "BatchedPesten integer state and rule transitions",
                "deck/deal state",
                "audited opponent local RNG and action sampling",
                "safe DecisionInput assembly",
            ],
            "gpu_responsibilities": [
                "NativePufferTemporalPolicy actor/critic forward inference",
                "native recurrent PPO loss, backward pass, and optimizer update",
            ],
            "one_gpu_process_at_a_time": True,
            "torch_compile_used": False,
        }
    )
    recurrent_measurement = {
        "games": CALIBRATION_GAMES,
        "rollout_seconds": rollout_seconds,
        "update_seconds": update_seconds,
        "end_to_end_seconds": end_to_end_seconds,
        "rollout_games_per_second": CALIBRATION_GAMES / rollout_seconds,
        "update_games_per_second": CALIBRATION_GAMES / update_seconds,
        "end_to_end_games_per_second": CALIBRATION_GAMES / end_to_end_seconds,
        "conservative_games_per_second": CALIBRATION_GAMES / end_to_end_seconds,
        "trajectory_count": len(episodes),
        "decision_count": sum(len(trajectory) for trajectory, _ in episodes),
        "contribution": contribution,
        "rollout_stats": {
            name: getattr(rollout_stats, name)
            for name in rollout_stats.__dataclass_fields__
        },
        "update_metrics": update_metrics,
        "legal_actions_100_percent": legal_actions,
        "finite_parameters": finite_parameters,
        "finite_metrics": finite_metrics,
    }
    recurrent_passed = (
        legal_actions
        and finite_parameters
        and finite_metrics
        and next(policy.parameters()).is_cuda
        and recurrent_measurement["decision_count"] > 0
    )
    gates = {
        "fixed_random_scalar_batched_parity": parity["passed"],
        "uniform_action_kernel": uniform_guard["passed"],
        "legacy_agent_random_smoke": legacy_smoke["passed"],
        "privacy_schema": privacy["passed"] and native_decision_input_only,
        "recurrent_cuda_rollout_update": recurrent_passed,
    }
    payload = {
        "format": CALIBRATION_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "status": "passed" if all(gates.values()) else "failed",
        "candidate_training": {
            "ran": False,
            "reason": "discarded phase-zero calibration update only; no candidate checkpoint was saved",
        },
        "namespace_allocations": {
            name: interval.as_dict() for name, interval in allocations.items()
        },
        "random_opponent_contract": audited_uniform_contract(),
        "fixed_random_parity": parity,
        "uniform_action_distribution_guard": uniform_guard,
        "legacy_agent_random_smoke": legacy_smoke,
        "privacy_boundary": privacy,
        "native_decision_input_only": native_decision_input_only,
        "runtime": runtime,
        "lanes": {
            "dense_mlp": {
                "status": "mechanically_blocked",
                "random_initialization": "required but no owned dense collector/update path can bind the audited opponent contract",
                "blocker": (
                    "The existing dense PPO launch/update path is outside this task's ownership and cannot accept the separate audited opponent-RNG interface or the DecisionInput/public-event boundary without modifying unowned code."
                ),
                "throughput": None,
            },
            "recurrent_temporal": {
                "status": "calibrated" if recurrent_passed else "failed_calibration",
                "random_initialization": {
                    "seed": initialization_seed,
                    "parent_checkpoint": None,
                    "teacher_or_imitation": None,
                },
                "model": {
                    "class": "NativePufferTemporalPolicy",
                    "hidden_size": 96,
                    "num_layers": 1,
                    "mode_conditioned_heads": False,
                },
                "throughput": recurrent_measurement,
            },
        },
        "gates": gates,
        "source_sha256": campaign_source_hashes(),
    }
    _write_json_exclusive(calibration_path, payload)
    return payload


def _capacity_from_calibration(
    games_per_second: float,
    *,
    wall_cap_seconds: int,
    maximum_checkpoints: int,
) -> dict[str, Any]:
    """Freeze the largest whole checkpoint interval below the 80% estimate."""

    if not math.isfinite(games_per_second) or games_per_second <= 0:
        raise ValueError("calibration games/second must be finite and positive")
    conservative_expected_games = math.floor(
        0.80 * games_per_second * wall_cap_seconds
    )
    if conservative_expected_games < 1:
        raise ValueError("calibration cannot support even one whole game")
    checkpoint_count = min(maximum_checkpoints, conservative_expected_games)
    interval = conservative_expected_games // checkpoint_count
    game_cap = interval * checkpoint_count
    thresholds = [interval * index for index in range(1, checkpoint_count + 1)]
    return {
        "wall_cap_seconds": wall_cap_seconds,
        "calibrated_games_per_second": games_per_second,
        "conservative_fraction": 0.80,
        "conservative_expected_games_before_rounding": conservative_expected_games,
        "checkpoint_count": checkpoint_count,
        "checkpoint_interval_games": interval,
        "exact_game_cap": game_cap,
        "cumulative_checkpoint_thresholds": thresholds,
        "stop_at_first_of": ["exact_game_cap", "wall_cap_seconds"],
        "unused_time_transfer_permitted": False,
    }


def _manifest_planning_requests(
    recurrent_capacity: Mapping[str, Any],
) -> tuple[AllocationRequest, ...]:
    game_cap = int(recurrent_capacity["exact_game_cap"])
    checkpoint_count = int(recurrent_capacity["checkpoint_count"])
    requests = [
        AllocationRequest(
            "training_recurrent_deals",
            game_cap,
            "frozen recurrent random-opponent training deal seeds",
        ),
        AllocationRequest(
            "training_recurrent_opponent_rng",
            game_cap,
            "frozen independent audited-opponent RNG seeds for recurrent training",
        ),
        AllocationRequest(
            "training_recurrent_initialization",
            1,
            "future recurrent candidate random initialization seed",
        ),
    ]
    for checkpoint_index in range(1, checkpoint_count + 1):
        suffix = f"recurrent_{checkpoint_index:02d}"
        requests.extend(
            (
                AllocationRequest(
                    f"development_screening_{suffix}",
                    DEVELOPMENT_PAIRS_PER_CHECKPOINT,
                    "one disjoint 250-pair development bank for one admitted recurrent checkpoint",
                ),
                AllocationRequest(
                    f"development_screening_opponent_rng_{suffix}",
                    DEVELOPMENT_PAIRS_PER_CHECKPOINT,
                    "independent audited-opponent RNG bank paired with one development bank",
                ),
            )
        )
    requests.extend(
        (
            AllocationRequest(
                "confirmation",
                CONFIRMATION_PAIRS,
                "one common fresh 2500-pair confirmation deal bank for all finalists",
            ),
            AllocationRequest(
                "confirmation_opponent_rng",
                CONFIRMATION_PAIRS,
                "one common fresh audited-opponent RNG bank for confirmation",
            ),
            AllocationRequest(
                "bootstrap_development",
                1,
                "precommitted development pair-bootstrap RNG seed",
            ),
            AllocationRequest(
                "bootstrap_confirmation",
                1,
                "precommitted confirmation pair-bootstrap RNG seed",
            ),
            AllocationRequest(
                "bootstrap_final_audit",
                1,
                "precommitted 100000-sample final-audit pair-bootstrap RNG seed",
            ),
            AllocationRequest(
                "final_audit",
                FINAL_AUDIT_PAIRS,
                "fresh one-use 5000-pair final-audit deal bank",
            ),
            AllocationRequest(
                "final_audit_opponent_rng",
                FINAL_AUDIT_PAIRS,
                "fresh one-use audited-opponent RNG bank for final audit",
            ),
        )
    )
    return tuple(requests)


def _checkpoint_development_banks(
    allocations: Mapping[str, SeedInterval],
    checkpoint_count: int,
) -> dict[str, Any]:
    banks = {}
    for index in range(1, checkpoint_count + 1):
        suffix = f"recurrent_{index:02d}"
        banks[f"recurrent_temporal_checkpoint_{index:02d}"] = {
            "pairs": DEVELOPMENT_PAIRS_PER_CHECKPOINT,
            "games": DEVELOPMENT_PAIRS_PER_CHECKPOINT * 2,
            "seat_balanced": True,
            "deal_subrange": allocations[
                f"development_screening_{suffix}"
            ].as_dict(),
            "opponent_rng_subrange": allocations[
                f"development_screening_opponent_rng_{suffix}"
            ].as_dict(),
        }
    return banks


def _require_calibration_gates(calibration: Mapping[str, Any]) -> None:
    if calibration.get("format") != CALIBRATION_FORMAT:
        raise ValueError("not a random-specialist calibration artifact")
    if calibration.get("status") != "passed":
        raise RuntimeError("calibration did not pass; refusing frozen campaign")
    gates = calibration.get("gates")
    if not isinstance(gates, Mapping) or not all(bool(value) for value in gates.values()):
        raise RuntimeError("one or more calibration gates failed")
    recurrent = calibration.get("lanes", {}).get("recurrent_temporal", {})
    if recurrent.get("status") != "calibrated":
        raise RuntimeError("recurrent lane is not calibrated")
    if calibration.get("candidate_training", {}).get("ran"):
        raise RuntimeError("calibration artifact claims candidate training ran")


def freeze_campaign(
    calibration: Mapping[str, Any],
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Bind phase-zero evidence to immutable later campaign commitments."""

    _require_calibration_gates(calibration)
    output_dir = Path(output_dir)
    manifest_path = output_dir / FROZEN_MANIFEST_NAME
    if manifest_path.exists():
        raise FileExistsError(f"campaign manifest already exists: {manifest_path}")
    current_hashes = campaign_source_hashes()
    if current_hashes != calibration.get("source_sha256"):
        raise RuntimeError(
            "source changed after calibration; rerun phase zero with a fresh calibration namespace"
        )
    recurrent_measurement = calibration["lanes"]["recurrent_temporal"]["throughput"]
    recurrent_capacity = _capacity_from_calibration(
        float(recurrent_measurement["conservative_games_per_second"]),
        wall_cap_seconds=45 * 60,
        maximum_checkpoints=12,
    )
    registry = SeedRegistry(output_dir / REGISTRY_NAME)
    requests = _manifest_planning_requests(recurrent_capacity)
    allocations = registry.allocate_many(requests)
    registry.validate_registered_namespaces(
        ["calibration", "calibration_opponent_rng", *allocations]
    )
    development_banks = _checkpoint_development_banks(
        allocations,
        int(recurrent_capacity["checkpoint_count"]),
    )
    calibration_path = output_dir / CALIBRATION_NAME
    final_audit_ledger = output_dir / "final_audit_claim.json"
    source_hashes = current_hashes
    manifest = {
        "format": CAMPAIGN_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "status": "frozen_pretraining",
        "frozen_at_utc": _utc_now(),
        "candidate_training": {
            "ran": False,
            "allowed_before_this_manifest": False,
            "post_freeze_runner": "campaign command validates this manifest before it can expose a later training implementation",
        },
        "source_sha256": source_hashes,
        "calibration": {
            "path": str(calibration_path),
            "sha256": file_sha256(calibration_path),
            "runtime": calibration["runtime"],
            "gates": calibration["gates"],
        },
        "audited_uniform_random_opponent": {
            **audited_uniform_contract(),
            "source_sha256": source_hashes["random_specialist_workflow.py"],
            "collector_source_sha256": source_hashes["temporal_batched_speed.py"],
            "scalar_implementation": "audited_uniform_agent_factory",
            "batched_implementation": "collect_league_batch_batched(schedule=('random',))",
            "per_game_rng_seed_source": "registered *_opponent_rng interval with exact corresponding game index",
        },
        "execution": {
            "one_gpu_process_at_a_time": True,
            "compilation": {
                "allowed": False,
                "reason": "no compiled path is preapproved by an exact model-state parity proof",
            },
            "privacy": {
                "fingerprint": calibration["privacy_boundary"]["safe_schema_sha256"],
                "native_policy_inputs": "DecisionInput plus public event tokens only",
                "opponent_rng_seed_visible_to_policy": False,
            },
        },
        "lanes": {
            "dense_mlp": {
                "status": "mechanically_blocked",
                "random_initialized_reward_only": True,
                "wall_cap_seconds": 15 * 60,
                "exact_game_cap": 0,
                "checkpoint_count": 0,
                "no_unused_time_transfer": True,
                "blocker": calibration["lanes"]["dense_mlp"]["blocker"],
            },
            "recurrent_temporal": {
                "status": "unblocked",
                "random_initialized_reward_only": True,
                "model": calibration["lanes"]["recurrent_temporal"]["model"],
                "capacity": recurrent_capacity,
                "initialization": {
                    "seed": allocations[
                        "training_recurrent_initialization"
                    ].start_inclusive,
                    "parent_checkpoint": None,
                    "parent_weights": None,
                    "ppo_checkpoint": None,
                    "imitation_data": None,
                    "teacher_logits": None,
                },
            },
        },
        "namespace_allocations": {
            "calibration": calibration["namespace_allocations"],
            "future_campaign": {
                name: interval.as_dict() for name, interval in allocations.items()
            },
            "all_namespaces_pairwise_non_overlapping": True,
            "legacy_consumed_sealed_interval": CONSUMED_LEGACY_SEALED_INTERVAL.as_dict(),
        },
        "checkpoint_schedule": {
            "dense_mlp": {
                "scheduled_thresholds": [],
                "screening_forbidden": True,
                "reason": "mechanically blocked lane has no candidate lineage",
            },
            "recurrent_temporal": {
                "maximum_candidates": 12,
                "exact_candidates": recurrent_capacity["checkpoint_count"],
                "evenly_spaced_cumulative_game_thresholds": recurrent_capacity[
                    "cumulative_checkpoint_thresholds"
                ],
                "only_scheduled_thresholds_may_be_screened": True,
            },
        },
        "checkpoint_admission": {
            "required": [
                "complete checkpoint",
                "complete optimizer state",
                "complete per-checkpoint manifest",
                "random-initialized no-parent lineage",
                "finite parameters and metrics",
                "100 percent legal actions",
                "unchanged privacy fingerprint",
                "unchanged fixed-random parity fingerprint",
                "scheduled cumulative threshold reached",
            ],
            "prohibited_lineage": [
                "PPO checkpoint",
                "imitation data",
                "teacher logits",
                "parent weights",
            ],
            "unscheduled_checkpoint_screening": "forbidden",
        },
        "development_screening": {
            "per_admitted_checkpoint": "exactly one registered disjoint bank",
            "banks": development_banks,
            "pairs_per_bank": DEVELOPMENT_PAIRS_PER_CHECKPOINT,
            "seat_balanced": True,
            "score": "pair score with draws scored as 0.5",
            "bootstrap": {
                "samples": FINAL_AUDIT_BOOTSTRAP_SAMPLES,
                "seed": allocations["bootstrap_development"].start_inclusive,
                "rank_key": [
                    "pair-level bootstrap 95% lower bound descending",
                    "point estimate descending",
                    "fewer training games ascending",
                    "lexicographic checkpoint SHA256 ascending",
                ],
            },
            "finalist_admission": {
                "dense_mlp": 1,
                "recurrent_temporal": 2,
                "if_fewer_eligible": "leave slot empty; never backfill another lineage",
            },
        },
        "confirmation": {
            "eligible_finalists_maximum": 3,
            "common_fresh_bank": {
                "pairs": CONFIRMATION_PAIRS,
                "games": CONFIRMATION_PAIRS * 2,
                "seat_balanced": True,
                "deal_range": allocations["confirmation"].as_dict(),
                "opponent_rng_range": allocations[
                    "confirmation_opponent_rng"
                ].as_dict(),
            },
            "bootstrap": {
                "samples": FINAL_AUDIT_BOOTSTRAP_SAMPLES,
                "seed": allocations["bootstrap_confirmation"].start_inclusive,
            },
            "rank_key": [
                "point estimate descending",
                "pair-level bootstrap 95% lower bound descending",
                "fewer training games ascending",
                "lexicographic checkpoint SHA256 ascending",
            ],
            "winner": "freeze exactly one trained winner with candidate, configuration, source, and opponent SHA256s",
        },
        "final_audit": {
            "name": "random_specialist_final_audit",
            "not_called": "sealed holdout",
            "does_not_reuse_legacy_consumed_sealed_holdout": True,
            "ledger": {
                "path": str(final_audit_ledger),
                "exclusive_create_before_model_load": True,
                "one_use": True,
                "crash_or_result_burns_audit": True,
                "retry_tuning_or_selection_after_claim": "forbidden",
            },
            "bank": {
                "pairs": FINAL_AUDIT_PAIRS,
                "games": FINAL_AUDIT_PAIRS * 2,
                "seat_balanced": True,
                "deal_range": allocations["final_audit"].as_dict(),
                "opponent_rng_range": allocations[
                    "final_audit_opponent_rng"
                ].as_dict(),
            },
            "candidate_binding_before_model_load": [
                "candidate SHA256",
                "configuration SHA256",
                "source SHA256",
                "audited opponent SHA256",
            ],
            "inference": {
                "deterministic_recurrent_inference": True,
                "explicit_per_game_recurrent_state_reset": True,
                "draw_score": 0.5,
                "max_turns": CALIBRATION_MAX_TURNS,
            },
            "bootstrap": {
                "pair_level_samples": FINAL_AUDIT_BOOTSTRAP_SAMPLES,
                "seed": allocations["bootstrap_final_audit"].start_inclusive,
                "precommitted": True,
            },
            "gate": {
                "point_estimate_gte": 0.80,
                "ci95_lower_bound_gte": 0.80,
                "both_required": True,
            },
        },
    }
    _write_json_exclusive(manifest_path, manifest)
    return manifest


def validate_frozen_campaign(path: Path) -> dict[str, Any]:
    """Fail closed before a later campaign command can start candidate work."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"frozen campaign manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != CAMPAIGN_FORMAT:
        raise ValueError("wrong frozen campaign manifest format")
    if manifest.get("status") != "frozen_pretraining":
        raise ValueError("campaign is not in frozen pretraining state")
    if manifest.get("candidate_training", {}).get("ran"):
        raise ValueError("manifest already records candidate training")
    if manifest.get("source_sha256") != campaign_source_hashes():
        raise RuntimeError("source SHA256s no longer match the frozen campaign")
    audit = manifest.get("final_audit", {})
    if audit.get("not_called") != "sealed holdout" or not audit.get(
        "does_not_reuse_legacy_consumed_sealed_holdout"
    ):
        raise ValueError("frozen audit contract is not independent of old holdout")
    if audit.get("bank", {}).get("pairs") != FINAL_AUDIT_PAIRS:
        raise ValueError("final audit pair count changed")
    if audit.get("bootstrap", {}).get("pair_level_samples") != FINAL_AUDIT_BOOTSTRAP_SAMPLES:
        raise ValueError("final audit bootstrap count changed")
    capacity = manifest.get("lanes", {}).get("recurrent_temporal", {}).get(
        "capacity", {}
    )
    if capacity.get("wall_cap_seconds") != 45 * 60:
        raise ValueError("recurrent wall cap changed")
    dense = manifest.get("lanes", {}).get("dense_mlp", {})
    if dense.get("wall_cap_seconds") != 15 * 60:
        raise ValueError("MLP wall cap changed")
    return manifest


def run_phase_zero(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Perform calibration once, then freeze the complete future campaign."""

    calibration = run_calibration(output_dir=output_dir, device_name=device_name)
    return freeze_campaign(calibration, output_dir=output_dir)


def campaign_preflight(
    manifest_path: Path,
    *,
    lane: str,
) -> dict[str, Any]:
    """Expose a later campaign interface while phase zero refuses training."""

    manifest = validate_frozen_campaign(manifest_path)
    lane_data = manifest.get("lanes", {}).get(lane)
    if not isinstance(lane_data, Mapping):
        raise ValueError(f"unknown campaign lane: {lane}")
    if lane_data.get("status") == "mechanically_blocked":
        raise RuntimeError(f"lane {lane} is mechanically blocked by the frozen manifest")
    return {
        "manifest": str(Path(manifest_path)),
        "lane": lane,
        "status": "preflight_valid_but_candidate_training_not_implemented_in_phase_zero",
        "candidate_training_started": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("calibrate", "phase-zero"):
        child = subparsers.add_parser(command)
        child.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
        child.add_argument("--device", choices=("cuda",), default="cuda")
    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--manifest", type=Path, required=True)
    campaign.add_argument(
        "--lane",
        choices=("dense_mlp", "recurrent_temporal"),
        required=True,
    )
    campaign.add_argument(
        "--dry-run",
        action="store_true",
        help="validate only; no candidate training exists in phase zero",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "calibrate":
        result = run_calibration(output_dir=args.output_dir, device_name=args.device)
    elif args.command == "phase-zero":
        result = run_phase_zero(output_dir=args.output_dir, device_name=args.device)
    else:
        result = campaign_preflight(args.manifest, lane=args.lane)
        if not args.dry_run:
            raise RuntimeError(
                "phase-zero exposes only a validated later campaign interface; "
                "candidate training is intentionally not implemented here"
            )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
