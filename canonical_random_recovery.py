"""Fail-closed recovery campaign against the canonical ``AgentRandom``.

The earlier random-specialist result used an independently seeded audited
uniform opponent.  That is a useful diagnostic contract, but it is not the
repository's canonical :class:`randomAgent.AgentRandom`.  This module freezes
the replacement campaign without training it: it defines the coupled RNG
contract, proves scalar/batched identity before any rollout can be admitted,
and writes only immutable phase-zero evidence into its own namespace.

The key compatibility detail is deliberate coupling.  For each game,
``np.random.seed(deal_seed)`` drives the scalar deal, AgentRandom's hand
shuffles, and discard recycling.  ``random.seed(deal_seed)`` drives only its
``changeSort`` choice.  The batched mirror uses the matching per-game
``RandomState`` already owned by :class:`batched_pesten.BatchedPesten`, plus a
per-game ``random.Random(deal_seed)`` for suit changes.  There is no opponent
RNG bank in this campaign.
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
import game
from batched_pesten import PHASE_CARD, PHASE_DRAW_RETRY, PHASE_SUIT
from randomAgent import AgentRandom
from random_specialist_workflow import PhaseZeroTracePolicy
from specialist_seed_registry import (
    AllocationRequest,
    CONSUMED_LEGACY_SEALED_INTERVAL,
    SeedInterval,
    SeedRegistry,
    scan_artifact_seed_intervals,
)
from temporal_batched_speed import (
    CANONICAL_RANDOM_AGENT_CONTRACT,
    collect_league_batch_batched,
)
from temporal_observation import (
    PASS_ACTION,
    SUIT_ACTION_START,
    DecisionInput,
    assert_policy_schema_is_hidden_state_free,
    build_legal_actions,
    serialize_safe_history,
)
from temporal_puffer_agent import TrajectoryStep, temporal_agent_factory


CAMPAIGN_ID = "canonical_random_recovery_20260802_v1"
CAMPAIGN_FORMAT = "pesten-canonical-random-recovery-frozen-campaign-v1"
CALIBRATION_FORMAT = "pesten-canonical-random-recovery-calibration-v1"
DEFAULT_OUTPUT_DIR = Path("benchmarks") / CAMPAIGN_ID
DEFAULT_TRAINING_DIR = Path("live_training") / CAMPAIGN_ID
REGISTRY_NAME = "seed_registry.jsonl"
FROZEN_MANIFEST_NAME = "frozen_campaign.json"
COLLISION_SCAN_NAME = "collision_scan.json"
CALIBRATION_NAME = "calibration.json"
SMOKE_NAME = "native_calibration_smoke.json"

# The independently reproduced contiguous candidate gap is intentionally
# bounded by occupied seeds on both sides.  It is below the sealed legacy
# holdout and carries enough space for every phase-zero and future campaign
# allocation.  ``build_collision_scan_evidence`` refuses to use it if a live
# scan no longer agrees.
CAMPAIGN_GAP = SeedInterval(
    4_028_451_531,
    4_029_882_669,
    "recovery_candidate_gap",
    "full historical artifact collision scan 2026-08-02",
)

PARITY_GAMES = 8
GPU_SMOKE_GAMES = 8
CALIBRATION_INITIALIZATION_COUNT = 1
TRAINING_GAMES = 150_000
DEVELOPMENT_CHECKPOINTS = 6
DEVELOPMENT_PAIRS_PER_CHECKPOINT = 500
CONFIRMATION_PAIRS = 2_500
FINAL_AUDIT_PAIRS = 5_000
BOOTSTRAP_SAMPLES = 100_000
PPO_EPOCHS = 4
CHECKPOINT_GAMES = tuple(range(25_000, TRAINING_GAMES + 1, 25_000))
WALL_CLOCK_CAP_SECONDS = 5 * 60 * 60
MAX_TURNS = 1_000
TRAINING_MICROBATCH_GAMES = 128
CHECKPOINT_FORMAT = "pesten-canonical-random-recovery-checkpoint-v1"
TRAINING_RESULT_FORMAT = "pesten-canonical-random-recovery-training-result-v1"
WINNER_FORMAT = "pesten-canonical-random-recovery-frozen-winner-v1"
CONFIRMATION_FAILED_FORMAT = "pesten-canonical-random-recovery-confirmation-failed-v1"
AUDIT_CLAIM_FORMAT = "pesten-canonical-random-recovery-final-audit-claim-v1"
AUDIT_RESULT_FORMAT = "pesten-canonical-random-recovery-final-audit-result-v1"

PARENT_CHECKPOINT = Path(
    "live_training/random_specialist_20260802/checkpoint_10845.pt"
)
PARENT_CHECKPOINT_SHA256 = (
    "58fcc1705f6481adb6f7aff7b28959d9554640467635bd60a0a027d1b99e8948"
)
PARENT_MANIFEST = Path(
    "live_training/random_specialist_20260802/checkpoint_10845.json"
)
PARENT_MANIFEST_SHA256 = (
    "47b3f6d4d6e970555d5ae0b817ab42c9bd447f1da435f76cdf4efb5c6a30c8d0"
)
OLD_DIAGNOSTIC = Path(
    "live_training/measurements/puffer_cycle_0005_vs_old_random.json"
)
OLD_DIAGNOSTIC_SHA256 = (
    "021b0f7e509b1193062eee4ec5a0bb7f6276d8de9487d9cd4ebddfd43666539e"
)
OLD_DIAGNOSTIC_DEAL_INTERVAL = SeedInterval(
    81_000_001,
    81_000_251,
    "diagnostic_old_random_deals",
    "prior noncanonical random-specialist diagnostic allocation",
)

SOURCE_FILES = (
    "canonical_random_recovery.py",
    "temporal_batched_speed.py",
    "batched_pesten.py",
    "game.py",
    "deck.py",
    "randomAgent.py",
    "temporal_observation.py",
    "temporal_puffer_agent.py",
    "temporal_puffer_v4.py",
    "specialist_seed_registry.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    """Return the exact content digest, failing instead of accepting drift."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Create one immutable JSON record; never overwrite evidence."""

    rendered = json.dumps(
        payload, indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partial/failed exclusive creation must remain visible for audit.
        raise


def campaign_source_hashes(root: Path | None = None) -> dict[str, str]:
    base = Path.cwd() if root is None else Path(root)
    return {name: file_sha256(base / name) for name in SOURCE_FILES}


def canonical_random_contract() -> dict[str, Any]:
    """Immutable scalar/batched contract for the real AgentRandom opponent."""

    return {
        **CANONICAL_RANDOM_AGENT_CONTRACT,
        "name": "canonical randomAgent.AgentRandom",
        "opponent_kind": "randomAgent.AgentRandom",
        "deal_rng_scalar": "np.random.seed(deal_seed)",
        "deal_rng_batched": "BatchedPesten.rngs[game_index] RandomState(deal_seed)",
        "shared_numpy_stream": [
            "initial deck shuffle and deal",
            "AgentRandom hand shuffle before every playCard call",
            "discard-pile recycle shuffle",
        ],
        "suit_rng_scalar": "random.seed(deal_seed); randomAgent.randint(0, 3)",
        "suit_rng_batched": "random.Random(deal_seed).randint(0, 3)",
        "opponent_rng_bank": False,
        "invalid_audited_uniform_substitution": True,
    }


def _state_sha256(state: Any) -> str:
    parts = (state,) if isinstance(state, torch.Tensor) else tuple(state)
    digest = hashlib.sha256()
    for part in parts:
        value = part.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _initial_scalar_deal(match: game.Game) -> dict[str, Any]:
    assert match.game_deck is not None
    return {
        "hands": [
            [held.number for held in player.mydeck.cards]
            for player in match.players
        ],
        "opening_card": int(match.game_deck.topCard().number),
    }


def _normalise_public_history(
    history: Sequence[Mapping[str, Any]],
) -> tuple[tuple, ...]:
    public_events: list[tuple] = []
    for item in history:
        action = item.get("action")
        player = int(item.get("player", -1))
        if action == "play":
            public_events.append(("play", player, int(item["card"])))
        elif action == "choose_sort":
            public_events.append(
                ("choose_sort", player, card.sorts.index(str(item["sort"])))
            )
    return tuple(public_events)


def _trajectory_rewards(
    trajectory: Sequence[TrajectoryStep], *, won: bool
) -> list[float]:
    """The reward-only contract: every non-win terminal outcome is zero win."""

    rewards = [-0.01] * len(trajectory)
    if rewards:
        rewards[-1] += 1.0 if won else -1.0
    return rewards


def _canonical_trace_temporal_factory(
    policy: torch.nn.Module,
    *,
    seed: int,
    deterministic: bool,
    action_trace: list[dict[str, Any]],
):
    """Record safe recurrent decisions without changing the scalar adapter."""

    base = temporal_agent_factory(policy, deterministic=deterministic, seed=seed)

    class TraceTemporalAgent(base):
        def _record_last_action(self, phase: int) -> None:
            step = self.trajectory[-1]
            action_trace.append(
                {
                    "seat": self.game.players.index(self),
                    "role": "current",
                    "phase": int(phase),
                    "action": int(step.action),
                    "penalty_amount": int(self.game.penalty_amount),
                    "current_sort": card.sorts.index(self.game.current_sort),
                    "current_rank": int(self.game.current_true_number),
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


def canonical_random_agent_factory(
    *, action_trace: list[dict[str, Any]]
):
    """Trace the real AgentRandom while preserving every random draw it makes.

    This is intentionally a subclass of :class:`AgentRandom`, not an
    independently written random policy.  ``super().playCard`` owns the
    global-NumPy in-place deck shuffle, and ``super().changeSort`` owns the
    module-level Python ``randint`` call.  The factory does not seed either
    source; the scalar game driver does that once per deal.
    """

    class TraceCanonicalRandomAgent(AgentRandom):
        type = "canonical_random"
        contract_id = CANONICAL_RANDOM_AGENT_CONTRACT["id"]

        def __init__(self, game_instance) -> None:
            super().__init__()
            self.game = game_instance
            self._last_pass_turn: int | None = None
            self._retry_card: int | None = None
            self._retry_turn: int | None = None

        def _retry_card_for_current_turn(self) -> int | None:
            if self._retry_turn != self.game.turn_count:
                self._retry_card = None
                self._retry_turn = None
            return self._retry_card

        def _record(
            self,
            *,
            phase: int,
            action: int,
            legal_actions: Sequence[bool] | None = None,
            opponent_hand_order: Sequence[int] | None = None,
        ) -> None:
            event: dict[str, Any] = {
                "seat": self.game.players.index(self),
                "role": "canonical_random",
                "phase": int(phase),
                "action": int(action),
                "penalty_amount": int(self.game.penalty_amount),
                "current_sort": card.sorts.index(self.game.current_sort),
                "current_rank": int(self.game.current_true_number),
            }
            if legal_actions is not None:
                event["legal_actions"] = list(legal_actions)
            if opponent_hand_order is not None:
                event["opponent_hand_order"] = list(opponent_hand_order)
            action_trace.append(event)

        def playCard(self, sort, truenumber):
            retry_active = self._retry_card_for_current_turn() is not None
            # Do not duplicate the algorithm: this advances the exact global
            # NumPy stream used by deck.Deck.shuffle.
            played = super().playCard(sort, truenumber)
            action = PASS_ACTION if played is None else int(played.number)
            self._record(
                phase=PHASE_DRAW_RETRY if retry_active else PHASE_CARD,
                action=action,
                opponent_hand_order=[held.number for held in self.mydeck.cards],
            )
            if played is None:
                self._last_pass_turn = (
                    None
                    if retry_active or self.game.penalty_amount > 0
                    else self.game.turn_count
                )
                self._retry_card = None
                self._retry_turn = None
            else:
                self._last_pass_turn = None
                self._retry_card = None
                self._retry_turn = None
            return played

        def changeSort(self):
            # ``AgentRandom.changeSort`` calls its imported ``randint``;
            # random.seed(deal_seed) in the scalar driver is its only source.
            chosen = super().changeSort()
            suit_action = SUIT_ACTION_START + card.sorts.index(chosen)
            legal = build_legal_actions(
                own_cards=(),
                current_suit=self.game.current_sort,
                current_rank=self.game.current_true_number,
                penalty_amount=self.game.penalty_amount,
                change_suit_mode=True,
            )
            self._record(
                phase=PHASE_SUIT,
                action=suit_action,
                legal_actions=legal,
            )
            return chosen

        def addCard(self, drawn_card) -> None:
            super().addCard(drawn_card)
            if drawn_card is not None and (
                self._last_pass_turn == self.game.turn_count
                and self.game.penalty_amount == 0
            ):
                self._retry_card = int(drawn_card.number)
                self._retry_turn = self.game.turn_count
                self._last_pass_turn = None

    return TraceCanonicalRandomAgent


def collect_scalar_canonical_trace(
    policy: torch.nn.Module,
    *,
    absolute_game_index: int,
    deal_seed: int,
    max_turns: int = MAX_TURNS,
    deterministic: bool = False,
) -> dict[str, Any]:
    """Run one scalar canonical-RNG game and return its complete parity trace."""

    if not 0 <= int(deal_seed) < 2**32:
        raise ValueError("canonical deal seed must be uint32-compatible")
    action_trace: list[dict[str, Any]] = []
    # This is the scalar source of truth.  Do this before instantiating/resetting
    # the game, and deliberately do not make an opponent-specific RNG object.
    np.random.seed(int(deal_seed))
    random.seed(int(deal_seed))
    current_factory = _canonical_trace_temporal_factory(
        policy,
        seed=int(deal_seed) * 2,
        deterministic=deterministic,
        action_trace=action_trace,
    )
    opponent_factory = canonical_random_agent_factory(action_trace=action_trace)
    current_seat = int(absolute_game_index) % 2
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
    while not match.game_over and match.turn_count < int(max_turns):
        match.simulate_turn()
    winner = -1 if match.winner is None else int(match.winner)
    trajectory = tuple(candidate.trajectory)
    final_state_sha256 = _state_sha256(candidate.state)
    trace = {
        "absolute_game_index": int(absolute_game_index),
        "deal_seed": int(deal_seed),
        "canonical_random_rng": {
            "deal_seed_shared_with_engine": int(deal_seed),
            "suit_rng": "random.Random(deal_seed).randint(0, 3)",
            "independent_opponent_rng": False,
        },
        "initial_deal": initial_deal,
        "actions": action_trace,
        "winner": winner,
        "turn_count": int(match.turn_count),
        "public_events": _normalise_public_history(match.public_history),
        "trajectory_actions": [int(step.action) for step in trajectory],
        "trajectory_legal_actions": [
            list(step.decision.legal_actions) for step in trajectory
        ],
        "trajectory_rewards": _trajectory_rewards(
            trajectory, won=winner == current_seat
        ),
        "safe_history_sha256": _sha256_bytes(
            serialize_safe_history(candidate.safe_history)
        ),
        "recurrent": {
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": final_state_sha256,
            "decision_count": len(trajectory),
        },
    }
    # Explicitly exercise the same adapter across reset.  A reconstructed
    # initial state is not enough evidence against recurrent leakage.
    match.reset()
    reset_state_sha256 = _state_sha256(candidate.state)
    trace["recurrent"]["reset_state_sha256"] = reset_state_sha256
    trace["recurrent"]["reset_verified"] = (
        reset_state_sha256 == initial_state_sha256
    )
    return trace


def _batch_trace_with_rewards(
    trace: Mapping[str, Any],
    episode: tuple[Sequence[TrajectoryStep], bool],
) -> dict[str, Any]:
    trajectory, won = episode
    recurrent = list(trace.get("recurrent", ()))
    if len(recurrent) != 1:
        raise RuntimeError("canonical random parity expects one recurrent seat")
    record = recurrent[0]
    return {
        "absolute_game_index": int(trace["absolute_game_index"]),
        "deal_seed": int(trace["deal_seed"]),
        "canonical_random_rng": dict(trace["canonical_random_rng"]),
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
            serialize_safe_history([step.decision for step in trajectory])
        ),
        "recurrent": {
            "initial_state_sha256": record["initial_state_sha256"],
            "final_state_sha256": record["final_state_sha256"],
            "decision_count": int(record["decision_count"]),
            # Each BatchedPesten vector is initialized independently.  The
            # scalar half additionally proves a same-instance reset above.
            "reset_verified": True,
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
    recurrent = trace["recurrent"]
    return {
        **{field: trace[field] for field in PARITY_FIELDS},
        "canonical_random_rng": trace["canonical_random_rng"],
        "recurrent": {
            "initial_state_sha256": recurrent["initial_state_sha256"],
            "final_state_sha256": recurrent["final_state_sha256"],
            "decision_count": recurrent["decision_count"],
        },
    }


def run_canonical_random_parity_gate(
    policy: torch.nn.Module,
    *,
    deal_seeds: Sequence[int],
    max_turns: int = MAX_TURNS,
) -> dict[str, Any]:
    """Require full scalar/batched identity for both seat rotations.

    This is an admission gate, not an approximate differential test.  A
    mismatch in initial hands, hand shuffle order, actions, suit choice,
    penalty/retry context, public events, safe trajectory, or recurrent reset
    state fails closed before any training or screening call may proceed.
    """

    deal_seeds = tuple(int(value) for value in deal_seeds)
    if not deal_seeds:
        raise ValueError("canonical parity gate needs at least one deal seed")
    if deal_seeds != tuple(range(deal_seeds[0], deal_seeds[0] + len(deal_seeds))):
        raise ValueError("canonical parity deal seeds must be a consecutive range")
    if any(not 0 <= value < 2**32 for value in deal_seeds):
        raise ValueError("canonical parity deal seeds must be uint32-compatible")
    policy.eval()
    scalar = [
        collect_scalar_canonical_trace(
            policy,
            absolute_game_index=index,
            deal_seed=deal_seed,
            max_turns=max_turns,
        )
        for index, deal_seed in enumerate(deal_seeds)
    ]
    batched_raw: list[dict[str, Any]] = []
    batched_episodes, metadata = collect_league_batch_batched(
        policy,
        None,
        (),
        games=len(deal_seeds),
        seed=deal_seeds[0],
        schedule=("canonical_random",),
        max_turns=max_turns,
        trace_sink=batched_raw,
    )
    if len(batched_raw) != len(deal_seeds) or len(batched_episodes) != len(deal_seeds):
        raise RuntimeError("batched canonical parity trace omitted a game")
    raw_by_index = {
        int(item["absolute_game_index"]): item for item in batched_raw
    }
    batched = [
        _batch_trace_with_rewards(raw_by_index[index], batched_episodes[index])
        for index in range(len(deal_seeds))
    ]
    cases: list[dict[str, Any]] = []
    for scalar_trace, batched_trace in zip(scalar, batched):
        mismatches = [
            field
            for field in PARITY_FIELDS
            if scalar_trace[field] != batched_trace[field]
        ]
        scalar_recurrent = scalar_trace["recurrent"]
        batched_recurrent = batched_trace["recurrent"]
        reset_matches = (
            bool(scalar_recurrent["reset_verified"])
            and bool(batched_recurrent["reset_verified"])
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
                "passed": not mismatches,
                "mismatches": mismatches,
                "scalar_trace_sha256": _sha256_bytes(
                    _canonical_json(_parity_projection(scalar_trace))
                ),
                "batched_trace_sha256": _sha256_bytes(
                    _canonical_json(_parity_projection(batched_trace))
                ),
                "initial_deal": scalar_trace["initial_deal"],
                "turn_count": scalar_trace["turn_count"],
                "winner": scalar_trace["winner"],
                "recurrent_reset_verified": reset_matches,
            }
        )
    return {
        "name": "canonical_AgentRandom_scalar_batched_exact_parity",
        "opponent": canonical_random_contract(),
        "metadata": metadata,
        "checked_fields": [*PARITY_FIELDS, "recurrent_reset_or_state"],
        "passed": all(case["passed"] for case in cases),
        "cases": cases,
    }


def require_canonical_parity(
    policy: torch.nn.Module, *, deal_seeds: Sequence[int], max_turns: int = MAX_TURNS
) -> dict[str, Any]:
    """Run and enforce the exact parity gate for all later campaign phases."""

    result = run_canonical_random_parity_gate(
        policy, deal_seeds=deal_seeds, max_turns=max_turns
    )
    if not result["passed"]:
        failed = [
            str(case["deal_seed"])
            for case in result["cases"]
            if not case["passed"]
        ]
        raise RuntimeError(
            "canonical AgentRandom scalar/batched parity failed for deal seeds "
            + ", ".join(failed)
        )
    return result


def _interval_payload(interval: SeedInterval) -> dict[str, Any]:
    return {
        "start_inclusive": int(interval.start_inclusive),
        "end_exclusive": int(interval.end_exclusive),
        "count": int(interval.count),
        "namespace": interval.namespace,
        "source": interval.source,
    }


def _merge_intervals(intervals: Iterable[SeedInterval]) -> list[tuple[int, int]]:
    ordered = sorted(
        (int(item.start_inclusive), int(item.end_exclusive))
        for item in intervals
    )
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def _artifact_content_hashes(roots: Sequence[Path]) -> dict[str, str]:
    """Hash every JSON/JSONL input used by the full collision scan."""

    result: dict[str, str] = {}
    cwd = Path.cwd().resolve()
    for root in roots:
        resolved = Path(root).resolve()
        if not resolved.exists():
            continue
        for path in sorted(resolved.rglob("*")):
            if not path.is_file() or path.suffix not in {".json", ".jsonl"}:
                continue
            try:
                label = str(path.relative_to(cwd))
            except ValueError:
                label = str(path)
            result[label] = file_sha256(path)
    return result


def build_collision_scan_evidence(
    *,
    roots: Sequence[Path] = (Path("benchmarks"), Path("live_training")),
    candidate_gap: SeedInterval = CAMPAIGN_GAP,
) -> dict[str, Any]:
    """Bind a full historical scan and prove the chosen gap is still clean.

    The artifact-content map plus the canonical ordered interval fingerprint is
    intentionally more useful than writing hundreds of thousands of copied
    intervals into a new benchmark artifact: a reviewer can reproduce every
    scanned input and every interval in the fingerprint exactly.
    """

    intervals = scan_artifact_seed_intervals(roots)
    occupied = [CONSUMED_LEGACY_SEALED_INTERVAL, *intervals]
    collisions = [
        item for item in occupied if candidate_gap.overlaps(item)
    ]
    if collisions:
        details = "; ".join(
            f"{item.namespace} [{item.start_inclusive},{item.end_exclusive})"
            for item in collisions[:4]
        )
        raise RuntimeError(
            "canonical recovery candidate gap is no longer clean: " + details
        )
    merged = _merge_intervals(occupied)
    prior = [item for item in merged if item[1] <= candidate_gap.start_inclusive]
    later = [item for item in merged if item[0] >= candidate_gap.end_exclusive]
    left = prior[-1] if prior else None
    right = later[0] if later else None
    if left is None or right is None:
        raise RuntimeError("candidate gap lacks bounded collision-scan neighbors")
    if left[1] != candidate_gap.start_inclusive or right[0] != candidate_gap.end_exclusive:
        raise RuntimeError(
            "candidate gap boundaries no longer match the full collision scan"
        )
    ordered_payload = [_interval_payload(item) for item in sorted(
        occupied,
        key=lambda item: (
            item.start_inclusive,
            item.end_exclusive,
            item.namespace,
            item.source,
        ),
    )]
    artifact_hashes = _artifact_content_hashes(roots)
    return {
        "format": "pesten-canonical-random-full-seed-collision-scan-v1",
        "scanned_at_utc": _utc_now(),
        "roots": [str(Path(root)) for root in roots],
        "artifact_count": len(artifact_hashes),
        "artifact_content_sha256": artifact_hashes,
        "artifact_content_fingerprint_sha256": _sha256_bytes(
            _canonical_json(artifact_hashes)
        ),
        "interval_count": len(intervals),
        "occupied_interval_count_including_sealed": len(occupied),
        "merged_interval_count": len(merged),
        "full_ordered_interval_fingerprint_sha256": _sha256_bytes(
            _canonical_json(ordered_payload)
        ),
        "candidate_gap": _interval_payload(candidate_gap),
        "left_neighbor": {
            "start_inclusive": left[0],
            "end_exclusive": left[1],
        },
        "right_neighbor": {
            "start_inclusive": right[0],
            "end_exclusive": right[1],
        },
        "candidate_gap_is_clean": True,
        "sealed_holdout": _interval_payload(CONSUMED_LEGACY_SEALED_INTERVAL),
    }


def campaign_allocation_requests() -> tuple[AllocationRequest, ...]:
    """Return the complete, contiguous no-opponent-bank campaign ledger."""

    requests: list[tuple[str, int, str]] = [
        (
            "calibration_parity_deals",
            PARITY_GAMES,
            "scalar/batched canonical AgentRandom exact-parity calibration only",
        ),
        (
            "calibration_gpu_smoke_deals",
            GPU_SMOKE_GAMES,
            "native Puffer canonical AgentRandom throughput smoke only",
        ),
        (
            "calibration_initialization",
            CALIBRATION_INITIALIZATION_COUNT,
            "discarded calibration-only model initialization seed",
        ),
        (
            "training_canonical_deals",
            TRAINING_GAMES,
            "future canonical AgentRandom reward-only training deals",
        ),
    ]
    requests.extend(
        (
            f"development_screening_canonical_{index:02d}",
            DEVELOPMENT_PAIRS_PER_CHECKPOINT,
            "one-use canonical AgentRandom seat-rotated development pair bank "
            f"for checkpoint {index:02d}",
        )
        for index in range(1, DEVELOPMENT_CHECKPOINTS + 1)
    )
    requests.extend(
        [
            (
                "confirmation_canonical_deals",
                CONFIRMATION_PAIRS,
                "one common fresh canonical AgentRandom confirmation pair bank",
            ),
            (
                "bootstrap_development",
                1,
                "development pair-bootstrap RandomState seed",
            ),
            (
                "bootstrap_confirmation",
                1,
                "confirmation pair-bootstrap RandomState seed",
            ),
            (
                "bootstrap_final_audit",
                1,
                "100000-resample final-audit pair-bootstrap RandomState seed",
            ),
            (
                "final_audit_canonical_deals",
                FINAL_AUDIT_PAIRS,
                "fresh one-use canonical AgentRandom final-audit pair bank",
            ),
        ]
    )
    offset = CAMPAIGN_GAP.start_inclusive
    allocated: list[AllocationRequest] = []
    for namespace, count, purpose in requests:
        allocated.append(
            AllocationRequest(
                namespace,
                count,
                purpose,
                start_inclusive=offset,
            )
        )
        offset += count
    if offset > CAMPAIGN_GAP.end_exclusive:
        raise RuntimeError("campaign allocation does not fit the verified gap")
    if offset - CAMPAIGN_GAP.start_inclusive != 160_520:
        raise AssertionError("canonical recovery allocation total drifted")
    return tuple(allocated)


def planned_intervals() -> dict[str, SeedInterval]:
    """Materialize the fixed plan without touching any registry or artifact."""

    return {
        request.namespace: SeedInterval(
            int(request.start_inclusive),
            int(request.start_inclusive) + request.count,
            request.namespace,
            "canonical recovery fixed allocation plan",
        )
        for request in campaign_allocation_requests()
    }


def _retired_invalid_registry_records() -> list[dict[str, Any]]:
    """Copy every old random-specialist allocation as prohibited, not reusable."""

    legacy_path = Path("benchmarks/random_specialist_20260802/seed_registry.jsonl")
    if not legacy_path.exists():
        raise FileNotFoundError(
            "cannot retire invalid campaign allocations; legacy registry is missing"
        )
    records: list[dict[str, Any]] = [
        {
            "schema_version": 1,
            "record_type": "prohibited_retired_allocation",
            "campaign": CAMPAIGN_ID,
            "namespace": "retired_diagnostic_old_random_000",
            "purpose": (
                "diagnostic-only legacy old-random measurement; consumed and "
                "prohibited for canonical recovery"
            ),
            "interval": OLD_DIAGNOSTIC_DEAL_INTERVAL.as_dict(),
            "source_path": str(OLD_DIAGNOSTIC),
            "source_sha256": OLD_DIAGNOSTIC_SHA256,
            "append_only": True,
        }
    ]
    for index, old_record in enumerate(
        SeedRegistry(legacy_path).read_records(), start=1
    ):
        if old_record.get("record_type") != "seed_allocation":
            continue
        interval = old_record.get("interval")
        if not isinstance(interval, Mapping):
            raise ValueError("legacy seed allocation record lacks an interval")
        records.append(
            {
                "schema_version": 1,
                "record_type": "prohibited_retired_allocation",
                "campaign": CAMPAIGN_ID,
                "namespace": f"retired_invalid_random_specialist_{index:03d}",
                "retired_namespace": old_record.get("namespace"),
                "purpose": (
                    "invalid audited-uniform random-specialist allocation; "
                    "prohibited from canonical AgentRandom recovery"
                ),
                "interval": dict(interval),
                "source_path": str(legacy_path),
                "append_only": True,
            }
        )
    return records


def _write_jsonl_exclusive(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n"
        for record in records
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise


def validate_parent_warm_start() -> dict[str, Any]:
    """Bind the permitted parent weights and explicitly reject its opponent."""

    checkpoint_sha = file_sha256(PARENT_CHECKPOINT)
    manifest_sha = file_sha256(PARENT_MANIFEST)
    diagnostic_sha = file_sha256(OLD_DIAGNOSTIC)
    if checkpoint_sha != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("approved parent checkpoint SHA256 changed")
    if manifest_sha != PARENT_MANIFEST_SHA256:
        raise RuntimeError("approved parent checkpoint manifest SHA256 changed")
    if diagnostic_sha != OLD_DIAGNOSTIC_SHA256:
        raise RuntimeError("old-random diagnostic SHA256 changed")
    diagnostic = json.loads(OLD_DIAGNOSTIC.read_text(encoding="utf-8"))
    old_random_score = float(diagnostic["results"]["random"]["score"])
    if old_random_score != 0.632:
        raise RuntimeError("expected 63.2 percent invalid-opponent diagnostic drifted")
    return {
        "warm_start": {
            "checkpoint_path": str(PARENT_CHECKPOINT),
            "checkpoint_sha256": checkpoint_sha,
            "parent_manifest_path": str(PARENT_MANIFEST),
            "parent_manifest_sha256": manifest_sha,
            "load": "model_state_dict only",
            "optimizer": "fresh optimizer; parent optimizer_state_dict prohibited",
        },
        "lineage": {
            "random_initialized": False,
            "reward_only_parent": True,
            "source_opponent": "audited uniform random",
            "source_opponent_invalid_for_canonical_selection": True,
            "parent_is_not_promotable_for_canonical_random": True,
        },
        "invalid_run_attestation": {
            "old_random_specialist_development_score": 0.988,
            "old_random_specialist_development_score_interpretation": (
                "98.8 percent against noncanonical audited-uniform random; "
                "not canonical AgentRandom evidence and not promotion evidence"
            ),
            "diagnostic_record": {
                "path": str(OLD_DIAGNOSTIC),
                "sha256": diagnostic_sha,
                "score": old_random_score,
                "percentage": "63.2%",
                "interpretation": (
                    "existing old-random diagnostic only; not reused as a "
                    "canonical development, confirmation, or audit result"
                ),
            },
        },
    }


def load_warm_start_weights_only(
    policy: torch.nn.Module,
    *,
    checkpoint_path: Path = PARENT_CHECKPOINT,
) -> dict[str, Any]:
    """Load approved model weights and intentionally never touch optimizer state."""

    expected_path = Path(checkpoint_path)
    if expected_path.resolve() != PARENT_CHECKPOINT.resolve():
        raise ValueError("canonical recovery permits only the approved parent checkpoint")
    if file_sha256(expected_path) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("approved warm-start checkpoint digest changed")
    checkpoint = torch.load(expected_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError("parent checkpoint lacks model_state_dict")
    # Do not even read ``optimizer_state_dict`` beyond its presence check: a
    # caller receives a fresh optimizer separately and cannot inherit moments.
    policy.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return {
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "model_weights_loaded": True,
        "optimizer_state_loaded": False,
        "optimizer_reset_required": True,
    }


def raw_win_rate(pair_scores: Sequence[float]) -> float:
    """Compute the raw binary-win average: draws score exactly zero."""

    values = np.asarray(pair_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("raw win rate needs a non-empty one-dimensional score vector")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("raw win scores must be finite values in [0, 1]")
    return float(values.mean())


def pair_mean_from_rotated_binary_wins(
    wins_as_seat_zero: Sequence[int | bool],
    wins_as_seat_one: Sequence[int | bool],
) -> np.ndarray:
    """Make pair scores from two binary wins; a draw is deliberately zero."""

    first = np.asarray(wins_as_seat_zero, dtype=np.float64)
    second = np.asarray(wins_as_seat_one, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or not len(first):
        raise ValueError("seat-rotated binary wins must be equal non-empty vectors")
    if np.any(~np.isin(first, (0.0, 1.0))) or np.any(~np.isin(second, (0.0, 1.0))):
        raise ValueError("draws must be encoded as 0, never 0.5")
    return (first + second) / 2.0


def pair_bootstrap_lower_bound(
    pair_scores: Sequence[float],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
    confidence: float = 0.95,
    chunk_size: int = 64,
) -> float:
    """Deterministic pair-level bootstrap lower confidence bound."""

    values = np.asarray(pair_scores, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("bootstrap needs one or more pair scores")
    if samples < 1 or chunk_size < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap parameters")
    rng = np.random.RandomState(int(seed))
    means = np.empty(samples, dtype=np.float64)
    offset = 0
    while offset < samples:
        size = min(chunk_size, samples - offset)
        indices = rng.randint(0, len(values), size=(size, len(values)))
        means[offset : offset + size] = values[indices].mean(axis=1)
        offset += size
    return float(np.quantile(means, (1.0 - confidence) / 2.0, method="linear"))


def privacy_boundary_report() -> dict[str, Any]:
    """Freeze the same DecisionInput-only boundary before any rollout."""

    assert_policy_schema_is_hidden_state_free()
    projection = {
        "policy_input": "DecisionInput only",
        "allowed": ["private observation", "legal actions", "public events"],
        "forbidden": [
            "opponent hand identity",
            "draw pile order",
            "opponent RNG state",
            "privileged critic",
            "teacher logits",
            "imitation data",
        ],
    }
    return {
        "passed": True,
        "privacy_fingerprint_sha256": _sha256_bytes(_canonical_json(projection)),
        **projection,
    }


def _intervals_payload(intervals: Mapping[str, SeedInterval]) -> dict[str, Any]:
    return {
        name: interval.as_dict()
        for name, interval in sorted(intervals.items())
    }


def _planned_development_banks(
    intervals: Mapping[str, SeedInterval],
) -> dict[str, Any]:
    return {
        f"checkpoint_{index:02d}": {
            "pair_interval": intervals[
                f"development_screening_canonical_{index:02d}"
            ].as_dict(),
            "pairs": DEVELOPMENT_PAIRS_PER_CHECKPOINT,
            "games": DEVELOPMENT_PAIRS_PER_CHECKPOINT * 2,
            "seat_balanced": True,
            "one_use": True,
        }
        for index in range(1, DEVELOPMENT_CHECKPOINTS + 1)
    }


def _campaign_manifest_payload(
    *,
    intervals: Mapping[str, SeedInterval],
    collision_scan: Mapping[str, Any],
    parent: Mapping[str, Any],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Produce the immutable plan that later execution must not reinterpret."""

    output = Path(output_dir)
    return {
        "format": CAMPAIGN_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "source_sha256": campaign_source_hashes(),
        "collision_scan": {
            "path": str(output / COLLISION_SCAN_NAME),
            "full_ordered_interval_fingerprint_sha256": collision_scan[
                "full_ordered_interval_fingerprint_sha256"
            ],
            "artifact_content_fingerprint_sha256": collision_scan[
                "artifact_content_fingerprint_sha256"
            ],
            "candidate_gap": collision_scan["candidate_gap"],
            "candidate_gap_is_clean": True,
        },
        "seed_registry": {
            "path": str(output / REGISTRY_NAME),
            "allocation_order": list(intervals),
            "intervals": _intervals_payload(intervals),
            "opponent_rng_bank": "prohibited",
            "retired_invalid_campaign_ranges": "prohibited in registry",
            "old_diagnostic_deal_interval": OLD_DIAGNOSTIC_DEAL_INTERVAL.as_dict(),
        },
        "canonical_random_opponent": canonical_random_contract(),
        "calibration": {
            "parity_deals": intervals["calibration_parity_deals"].as_dict(),
            "gpu_smoke_deals": intervals["calibration_gpu_smoke_deals"].as_dict(),
            "initialization": intervals["calibration_initialization"].as_dict(),
            "only_permitted_phase_zero_rollouts": True,
            "parity_required_before_training": True,
            "native_smoke_requires_cuda": True,
        },
        "training": {
            "status": "precommitted_not_started",
            "deal_interval": intervals["training_canonical_deals"].as_dict(),
            "games": TRAINING_GAMES,
            "microbatch_games": TRAINING_MICROBATCH_GAMES,
            "opponent_schedule": ["canonical_random"],
            "opponent_rng_bank": False,
            "reward_only": True,
            "model": {
                "class": "NativePufferTemporalPolicy",
                "hidden_size": 96,
                "num_layers": 1,
            },
            "ppo_epochs": PPO_EPOCHS,
            "optimizer": {
                "class": "torch.optim.Adam",
                "learning_rate": 0.0003,
                "parent_optimizer_state": "prohibited",
                "fresh_state_required": True,
            },
            "scheduled_checkpoints_games": list(CHECKPOINT_GAMES),
            "wall_clock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
            "adaptation_after_start": "forbidden",
            "restart_policy": "fail closed; exclusive output directory",
            "default_output_dir": str(DEFAULT_TRAINING_DIR),
            "warm_start": parent["warm_start"],
            "lineage": parent["lineage"],
        },
        "development": {
            "banks": _planned_development_banks(intervals),
            "rank_one_raw_win": True,
            "rank_tiebreakers": [
                "higher pair-bootstrap lower 95% confidence bound",
                "fewer training games",
                "lexicographically smaller checkpoint SHA256",
            ],
            "freeze_exactly_one_development_winner": True,
            "bootstrap_seed_interval": intervals["bootstrap_development"].as_dict(),
        },
        "confirmation": {
            "pair_interval": intervals["confirmation_canonical_deals"].as_dict(),
            "pairs": CONFIRMATION_PAIRS,
            "games": CONFIRMATION_PAIRS * 2,
            "seat_balanced": True,
            "draw_score": 0,
            "pair_score": "mean of two binary candidate win indicators",
            "bootstrap_seed_interval": intervals["bootstrap_confirmation"].as_dict(),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "one_use": True,
            "required_raw_win_rate": 0.80,
            "required_lower_95_confidence_bound": 0.80,
            "on_failure": "exclusive confirmation_failed.json and no frozen_winner",
        },
        "final_audit": {
            "pair_interval": intervals["final_audit_canonical_deals"].as_dict(),
            "pairs": FINAL_AUDIT_PAIRS,
            "games": FINAL_AUDIT_PAIRS * 2,
            "seat_balanced": True,
            "draw_score": 0,
            "pair_score": "mean of two binary candidate win indicators",
            "bootstrap_seed_interval": intervals["bootstrap_final_audit"].as_dict(),
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "required_raw_win_rate": 0.80,
            "required_lower_95_confidence_bound": 0.80,
            "one_use": True,
            "claim_path": str(output / "final_audit_claim.json"),
            "result_path": str(output / "final_audit_result.json"),
        },
        "bootstrap": {
            "development_seed_interval": intervals["bootstrap_development"].as_dict(),
            "confirmation_seed_interval": intervals["bootstrap_confirmation"].as_dict(),
            "audit_seed_interval": intervals["bootstrap_final_audit"].as_dict(),
            "algorithm": {
                "rng": "numpy.random.RandomState MT19937",
                "samples": BOOTSTRAP_SAMPLES,
                "unit": "pair mean of two binary wins",
                "draw_score": 0,
                "lower_quantile": 0.025,
            },
        },
        "invalid_run_attestation": parent["invalid_run_attestation"],
        "candidate_training": {
            "started": False,
            "checkpoint_created": False,
            "promotable": False,
        },
    }


def _assert_output_is_fresh(output_dir: Path) -> None:
    if Path(output_dir).exists():
        raise FileExistsError(
            f"canonical recovery namespace already exists: {output_dir}"
        )


def bootstrap_campaign(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Seal the canonical campaign plan without starting candidate training.

    ``dry_run`` is intentionally side-effect free: it neither scans artifact
    banks nor instantiates policies nor creates a directory.  That gives CI a
    way to inspect the fixed plan without consuming seeds or touching a GPU.
    """

    plan = planned_intervals()
    if dry_run:
        return {
            "dry_run": True,
            "campaign_id": CAMPAIGN_ID,
            "output_dir": str(output_dir),
            "planned_intervals": _intervals_payload(plan),
            "total_allocated_seeds": sum(item.count for item in plan.values()),
            "opponent_rng_bank": False,
            "candidate_training_started": False,
        }
    output_dir = Path(output_dir)
    _assert_output_is_fresh(output_dir)
    parent = validate_parent_warm_start()
    # The gate executes before any output/ledger mutation, so a mismatching
    # seam cannot consume a campaign range or appear as a valid calibration.
    parity = require_canonical_parity(
        PhaseZeroTracePolicy().eval(),
        deal_seeds=range(
            plan["calibration_parity_deals"].start_inclusive,
            plan["calibration_parity_deals"].end_exclusive,
        ),
    )
    privacy = privacy_boundary_report()
    collision_scan = build_collision_scan_evidence()
    registry_path = output_dir / REGISTRY_NAME
    _write_jsonl_exclusive(registry_path, _retired_invalid_registry_records())
    registry = SeedRegistry(
        registry_path,
        artifact_roots=(Path("benchmarks"), Path("live_training")),
    )
    allocated = registry.allocate_many(
        campaign_allocation_requests(),
        preferred_start=CAMPAIGN_GAP.start_inclusive,
    )
    if _intervals_payload(allocated) != _intervals_payload(plan):
        raise RuntimeError("registry allocation differs from fixed campaign plan")
    collision_payload = {
        **collision_scan,
        "campaign_id": CAMPAIGN_ID,
        "reserved_intervals": _intervals_payload(allocated),
        "reserved_interval_fingerprint_sha256": _sha256_bytes(
            _canonical_json(_intervals_payload(allocated))
        ),
    }
    _write_json_exclusive(output_dir / COLLISION_SCAN_NAME, collision_payload)
    manifest = _campaign_manifest_payload(
        intervals=allocated,
        collision_scan=collision_payload,
        parent=parent,
        output_dir=output_dir,
    )
    manifest["phase_zero_bootstrap"] = {
        "parity": parity,
        "privacy": privacy,
        "passed": bool(parity["passed"] and privacy["passed"]),
    }
    _write_json_exclusive(output_dir / FROZEN_MANIFEST_NAME, manifest)
    return {
        "dry_run": False,
        "output_dir": str(output_dir),
        "registry_path": str(registry_path),
        "manifest_path": str(output_dir / FROZEN_MANIFEST_NAME),
        "collision_scan_path": str(output_dir / COLLISION_SCAN_NAME),
        "parity_passed": parity["passed"],
        "total_allocated_seeds": sum(item.count for item in allocated.values()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object JSON: {path}")
    return value


def load_frozen_manifest(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    manifest = _read_json(Path(output_dir) / FROZEN_MANIFEST_NAME)
    if manifest.get("format") != CAMPAIGN_FORMAT or manifest.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("wrong or malformed canonical recovery manifest")
    if manifest.get("canonical_random_opponent", {}).get("opponent_rng_bank") is not False:
        raise RuntimeError("canonical recovery manifest permits an opponent RNG bank")
    if manifest.get("training", {}).get("opponent_schedule") != ["canonical_random"]:
        raise RuntimeError("canonical recovery manifest drifted from AgentRandom")
    return manifest


def _assert_manifest_source_hashes(manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("source_sha256")
    if not isinstance(expected, Mapping):
        raise RuntimeError("frozen manifest lacks source hashes")
    observed = campaign_source_hashes()
    if dict(expected) != observed:
        raise RuntimeError("canonical recovery source changed after plan freeze")


def run_phase_zero_calibration(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    policy: torch.nn.Module | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write parity/privacy calibration using only its dedicated deal range."""

    if dry_run:
        return {
            "dry_run": True,
            "would_read": str(Path(output_dir) / FROZEN_MANIFEST_NAME),
            "would_write": str(Path(output_dir) / CALIBRATION_NAME),
            "would_use_only": "calibration_parity_deals",
            "candidate_training_started": False,
        }
    manifest = load_frozen_manifest(output_dir)
    _assert_manifest_source_hashes(manifest)
    target = Path(output_dir) / CALIBRATION_NAME
    if target.exists():
        raise FileExistsError(f"immutable calibration already exists: {target}")
    # Bootstrap has already executed this dedicated parity bank before
    # allocating/writing the campaign.  Materialize that immutable evidence
    # exactly once instead of consuming the same calibration deals again.
    del policy
    interval = manifest["calibration"]["parity_deals"]
    phase_zero = manifest.get("phase_zero_bootstrap", {})
    parity = phase_zero.get("parity")
    privacy = phase_zero.get("privacy")
    if not isinstance(parity, Mapping) or not isinstance(privacy, Mapping):
        raise RuntimeError("frozen phase-zero parity/privacy evidence is missing")
    if not parity.get("passed") or not privacy.get("passed"):
        raise RuntimeError("frozen phase-zero parity/privacy evidence failed")
    payload = {
        "format": CALIBRATION_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "deal_interval": interval,
        "canonical_parity": parity,
        "privacy": privacy,
        "gates": {
            "canonical_scalar_batched_parity": bool(parity["passed"]),
            "decision_input_only": bool(privacy["passed"]),
            "calibration_only": True,
            "candidate_training_started": False,
        },
        "passed": bool(parity["passed"] and privacy["passed"]),
    }
    _write_json_exclusive(target, payload)
    return payload


def execute_native_calibration_smoke(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR, dry_run: bool = False
) -> dict[str, Any]:
    """Native-runtime CLI entrypoint for the GPU-only smoke.

    Imports stay local so ordinary test interpreters do not need PufferLib 4.
    The allocated initialization seed is consumed only to initialize this
    disposable smoke model; no optimizer is constructed and no checkpoint is
    written.  On a host without CUDA the function fails before creating the
    smoke artifact.
    """

    if dry_run:
        return run_native_calibration_smoke(
            torch.nn.Linear(1, 1), output_dir=output_dir, dry_run=True
        )
    manifest = load_frozen_manifest(output_dir)
    initialization = manifest["calibration"]["initialization"]
    if int(initialization["count"]) != 1:
        raise RuntimeError("native smoke initialization allocation is malformed")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable; canonical native smoke was not created"
        )
    from temporal_puffer_v4 import NativePufferTemporalPolicy

    torch.manual_seed(int(initialization["start_inclusive"]))
    policy = NativePufferTemporalPolicy(hidden_size=96, num_layers=1).to("cuda")
    return run_native_calibration_smoke(policy, output_dir=output_dir)


def run_native_calibration_smoke(
    policy: torch.nn.Module,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Measure native recurrent rollouts only; it performs no PPO update."""

    if dry_run:
        return {
            "dry_run": True,
            "would_write": str(Path(output_dir) / SMOKE_NAME),
            "would_use_only": "calibration_gpu_smoke_deals",
            "optimizer_steps": 0,
            "candidate_training_started": False,
        }
    manifest = load_frozen_manifest(output_dir)
    _assert_manifest_source_hashes(manifest)
    target = Path(output_dir) / SMOKE_NAME
    if target.exists():
        raise FileExistsError(f"immutable native smoke already exists: {target}")
    if not bool(getattr(policy, "native_pufferlib", False)):
        raise TypeError("native calibration smoke requires NativePufferTemporalPolicy")
    parameter = next(policy.parameters())
    if parameter.device.type != "cuda":
        raise RuntimeError("native calibration smoke requires a CUDA-resident policy")
    interval = manifest["calibration"]["gpu_smoke_deals"]
    started = time.perf_counter()
    episodes, metadata = collect_league_batch_batched(
        policy.eval(),
        None,
        (),
        games=int(interval["count"]),
        seed=int(interval["start_inclusive"]),
        schedule=("canonical_random",),
        max_turns=MAX_TURNS,
    )
    elapsed = time.perf_counter() - started
    legal = all(
        step.decision.legal_actions[step.action]
        for trajectory, _ in episodes
        for step in trajectory
    )
    payload = {
        "format": "pesten-canonical-random-native-calibration-smoke-v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "deal_interval": interval,
        "opponent": canonical_random_contract(),
        "model": {
            "class": type(policy).__name__,
            "native_pufferlib": True,
            "hidden_size": getattr(policy, "hidden_size", None),
            "num_layers": getattr(policy, "num_layers", None),
            "device": str(parameter.device),
        },
        "rollout": {
            "episodes": len(episodes),
            "metadata": metadata,
            "elapsed_seconds": elapsed,
            "games_per_second": float(len(episodes) / elapsed) if elapsed else math.inf,
            "all_actions_legal": legal,
            "optimizer_steps": 0,
        },
        "candidate_training_started": False,
        "passed": bool(episodes and legal),
    }
    _write_json_exclusive(target, payload)
    return payload


def canonical_evaluator_mapping() -> dict[str, Any]:
    """The only evaluator admissible for development, confirmation and audit."""

    return {
        "implementation": "scalar game.Game + randomAgent.AgentRandom",
        "opponent": "randomAgent.AgentRandom",
        "deal_rng": "np.random.seed(deal_seed) once before each rotated game",
        "hand_and_recycle_rng": "same global NumPy stream as deal",
        "suit_rng": "random.seed(deal_seed) once before each rotated game",
        "seat_order": ["candidate_seat_0", "candidate_seat_1"],
        "pair_deal_seed": "same deal_seed reused for both seat rotations",
        "candidate_score": "binary 1 only for candidate winner; draw is 0",
        "pair_score": "mean of two binary seat-rotated candidate scores",
        "opponent_rng_bank": False,
    }


def evaluate_canonical_seat_balanced_pairs(
    policy: torch.nn.Module,
    *,
    deal_seeds: Sequence[int],
    max_turns: int = MAX_TURNS,
) -> dict[str, Any]:
    """Evaluate a candidate on a one-use canonical pair bank in fixed order.

    The function is intentionally scalar.  The batched collector is proven
    equivalent by the admission gate and used for rollouts, while evaluation
    keeps the canonical AgentRandom call path itself directly inspectable.
    """

    seeds = tuple(int(seed) for seed in deal_seeds)
    if not seeds or any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("canonical evaluator needs non-empty uint32 deal seeds")
    was_training = bool(policy.training)
    policy.eval()
    try:
        rows: list[dict[str, Any]] = []
        pair_scores: list[float] = []
        outcomes = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "wins_as_seat_0": 0,
            "wins_as_seat_1": 0,
            "losses_as_seat_0": 0,
            "losses_as_seat_1": 0,
            "draws_as_seat_0": 0,
            "draws_as_seat_1": 0,
        }
        for pair_index, deal_seed in enumerate(seeds):
            first = collect_scalar_canonical_trace(
            policy,
            absolute_game_index=0,
            deal_seed=deal_seed,
            max_turns=max_turns,
            deterministic=True,
            )
            second = collect_scalar_canonical_trace(
            policy,
            absolute_game_index=1,
            deal_seed=deal_seed,
            max_turns=max_turns,
            deterministic=True,
            )
            first_win = int(first["winner"] == 0)
            second_win = int(second["winner"] == 1)
            first_outcome = (
                "win" if first_win else "draw" if first["winner"] == -1 else "loss"
            )
            second_outcome = (
                "win" if second_win else "draw" if second["winner"] == -1 else "loss"
            )
            plural = {"win": "wins", "loss": "losses", "draw": "draws"}
            for seat, outcome in ((0, first_outcome), (1, second_outcome)):
                outcomes[plural[outcome]] += 1
                outcomes[f"{plural[outcome]}_as_seat_{seat}"] += 1
            pair_score = float((first_win + second_win) / 2.0)
            pair_scores.append(pair_score)
            rows.append(
                {
                    "pair_index": pair_index,
                    "deal_seed": deal_seed,
                    "seat_0": {
                        "candidate_seat": 0,
                        "winner": first["winner"],
                        "outcome": first_outcome,
                        "candidate_binary_win": first_win,
                        "turn_count": first["turn_count"],
                        "trace_sha256": _sha256_bytes(
                            _canonical_json(_parity_projection(first))
                        ),
                    },
                    "seat_1": {
                        "candidate_seat": 1,
                        "winner": second["winner"],
                        "outcome": second_outcome,
                        "candidate_binary_win": second_win,
                        "turn_count": second["turn_count"],
                        "trace_sha256": _sha256_bytes(
                            _canonical_json(_parity_projection(second))
                        ),
                    },
                    "pair_score": pair_score,
                }
            )
    finally:
        policy.train(was_training)
    pair_values = np.asarray(pair_scores, dtype=np.float64)
    return {
        "format": "pesten-canonical-random-seat-balanced-evaluation-v1",
        "evaluator": canonical_evaluator_mapping(),
        "pairs": len(rows),
        "games": len(rows) * 2,
        "rows": rows,
        "pair_scores": pair_values.tolist(),
        "raw_win_rate": raw_win_rate(pair_values),
        "draw_score": 0,
        "outcomes": outcomes,
    }


def rank_development_checkpoints(
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rank once by raw win then the frozen, deterministic tie breakers."""

    normalized: list[dict[str, Any]] = []
    for candidate in candidates:
        required = {
            "checkpoint_sha256",
            "training_games",
            "pair_scores",
            "bootstrap_lower_95",
        }
        missing = required - set(candidate)
        if missing:
            raise ValueError("development candidate lacks fields: " + ", ".join(sorted(missing)))
        scores = tuple(float(item) for item in candidate["pair_scores"])
        normalized.append(
            {
                **dict(candidate),
                "raw_win_rate": raw_win_rate(scores),
                "pair_scores": list(scores),
            }
        )
    return sorted(
        normalized,
        key=lambda item: (
            -float(item["raw_win_rate"]),
            -float(item["bootstrap_lower_95"]),
            int(item["training_games"]),
            str(item["checkpoint_sha256"]),
        ),
    )


def canonical_final_audit_paths(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    output = Path(output_dir)
    return output / "final_audit_claim.json", output / "final_audit_result.json"


def audit_claim_order() -> list[str]:
    return [
        "validate frozen manifest, commitment, winner and exact canonical paths",
        "verify claim and result paths are absent",
        "exclusive-create one-use canonical final-audit claim",
        "load selected candidate weights after claim only",
        "iterate registered final-audit deal seeds in ascending order",
        "evaluate candidate seat 0 then candidate seat 1 for every deal seed",
        "compute raw binary win rate and pair bootstrap lower 95% bound",
        "exclusive-create canonical final-audit result",
    ]


def create_final_audit_claim(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    winner_sha256: str,
    commitment_sha256: str | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Burn the final bank before candidate model loading or seed iteration."""

    manifest = load_frozen_manifest(output_dir)
    claim_path, result_path = canonical_final_audit_paths(output_dir)
    committed = manifest["final_audit"]
    if str(claim_path) != committed["claim_path"] or str(result_path) != committed["result_path"]:
        raise RuntimeError("final audit paths differ from frozen canonical paths")
    if claim_path.exists() or result_path.exists():
        raise FileExistsError("one-use final audit claim or result already exists")
    interval = committed["pair_interval"]
    payload = {
        "format": AUDIT_CLAIM_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "winner_checkpoint_sha256": str(winner_sha256),
        "pair_interval": interval,
        "pair_count": int(committed["pairs"]),
        "order": audit_claim_order(),
        "claim_before_model_load": True,
        "claim_before_final_seed_iteration": True,
        "opponent": canonical_random_contract(),
    }
    if commitment_sha256 is not None:
        payload["commitment_sha256"] = str(commitment_sha256)
    if manifest_sha256 is not None:
        payload["manifest_sha256"] = str(manifest_sha256)
    _write_json_exclusive(claim_path, payload)
    return payload


def _load_calibration_admission(output_dir: Path) -> dict[str, Any]:
    payload = _read_json(Path(output_dir) / CALIBRATION_NAME)
    if payload.get("format") != CALIBRATION_FORMAT or not payload.get("passed"):
        raise RuntimeError("canonical parity/privacy calibration is missing or failed")
    gates = payload.get("gates", {})
    if not all(
        gates.get(name) is True
        for name in (
            "canonical_scalar_batched_parity",
            "decision_input_only",
            "calibration_only",
            "candidate_training_started",
        )
        if name != "candidate_training_started"
    ):
        raise RuntimeError("canonical calibration admission gates are incomplete")
    if gates.get("candidate_training_started") is not False:
        raise RuntimeError("calibration incorrectly claims candidate training")
    return payload


def build_implementation_commitment(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Bind calibration and native smoke evidence before any training wrapper."""

    if dry_run:
        return {
            "dry_run": True,
            "would_read": [
                str(Path(output_dir) / FROZEN_MANIFEST_NAME),
                str(Path(output_dir) / CALIBRATION_NAME),
                str(Path(output_dir) / SMOKE_NAME),
            ],
            "would_write": str(Path(output_dir) / "implementation_commitment.json"),
            "candidate_training_started": False,
        }
    manifest = load_frozen_manifest(output_dir)
    _assert_manifest_source_hashes(manifest)
    calibration = _load_calibration_admission(output_dir)
    smoke = _read_json(Path(output_dir) / SMOKE_NAME)
    if not smoke.get("passed") or smoke.get("candidate_training_started") is not False:
        raise RuntimeError("native calibration smoke is missing, failed, or not phase-zero only")
    target = Path(output_dir) / "implementation_commitment.json"
    if target.exists():
        raise FileExistsError("canonical implementation commitment already exists")
    payload = {
        "format": "pesten-canonical-random-recovery-implementation-commitment-v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "manifest_path": str(Path(output_dir) / FROZEN_MANIFEST_NAME),
        "manifest_sha256": file_sha256(Path(output_dir) / FROZEN_MANIFEST_NAME),
        "calibration_path": str(Path(output_dir) / CALIBRATION_NAME),
        "calibration_sha256": file_sha256(Path(output_dir) / CALIBRATION_NAME),
        "native_smoke_path": str(Path(output_dir) / SMOKE_NAME),
        "native_smoke_sha256": file_sha256(Path(output_dir) / SMOKE_NAME),
        "source_sha256": campaign_source_hashes(),
        "admission": {
            "canonical_scalar_batched_parity": True,
            "privacy_boundary": True,
            "native_cuda_smoke": True,
            "source_unchanged": True,
            "opponent_rng_bank": False,
            "candidate_training_started": False,
        },
        "training": manifest["training"],
        "development": manifest["development"],
        "confirmation": manifest["confirmation"],
        "final_audit": manifest["final_audit"],
        "candidate_training": {
            "started": False,
            "checkpoint_created": False,
            "promotable": False,
        },
        "calibration_summary": {
            "parity_passed": calibration["canonical_parity"]["passed"],
            "native_smoke_passed": smoke["passed"],
        },
    }
    _write_json_exclusive(target, payload)
    return payload


def validate_pretraining_admission(
    *, output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Fail closed before any candidate training, bank evaluation or checkpoint."""

    manifest = load_frozen_manifest(output_dir)
    _assert_manifest_source_hashes(manifest)
    calibration = _load_calibration_admission(output_dir)
    commitment = _read_json(Path(output_dir) / "implementation_commitment.json")
    if commitment.get("campaign_id") != CAMPAIGN_ID:
        raise RuntimeError("wrong canonical implementation commitment")
    if commitment.get("source_sha256") != campaign_source_hashes():
        raise RuntimeError("commitment source fingerprint drifted")
    admission = commitment.get("admission", {})
    required = (
        "canonical_scalar_batched_parity",
        "privacy_boundary",
        "native_cuda_smoke",
        "source_unchanged",
    )
    if not all(admission.get(key) is True for key in required):
        raise RuntimeError("canonical pretraining admission is not complete")
    if admission.get("opponent_rng_bank") is not False:
        raise RuntimeError("canonical pretraining admission permits an opponent RNG bank")
    if commitment.get("candidate_training", {}).get("started"):
        raise RuntimeError("commitment cannot be created after training starts")
    return {
        "passed": True,
        "manifest_sha256": file_sha256(Path(output_dir) / FROZEN_MANIFEST_NAME),
        "calibration_sha256": file_sha256(Path(output_dir) / CALIBRATION_NAME),
        "canonical_parity_passed": calibration["canonical_parity"]["passed"],
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return True


def _interval_range(interval: Mapping[str, Any], *, label: str) -> range:
    start = int(interval["start_inclusive"])
    end = int(interval["end_exclusive"])
    if not 0 <= start < end <= 2**32:
        raise ValueError(f"{label} is not a non-empty uint32 interval")
    return range(start, end)


def _load_implementation_commitment(output_dir: Path) -> tuple[dict[str, Any], Path]:
    path = Path(output_dir) / "implementation_commitment.json"
    commitment = _read_json(path)
    if (
        commitment.get("format")
        != "pesten-canonical-random-recovery-implementation-commitment-v1"
        or commitment.get("campaign_id") != CAMPAIGN_ID
    ):
        raise RuntimeError("wrong or malformed canonical implementation commitment")
    return commitment, path


def _assert_canonical_seed_manifest(manifest: Mapping[str, Any]) -> None:
    registry = manifest.get("seed_registry", {})
    intervals = registry.get("intervals", {}) if isinstance(registry, Mapping) else {}
    if not isinstance(intervals, Mapping):
        raise RuntimeError("canonical seed registry intervals are malformed")
    forbidden = [name for name in intervals if "opponent_rng" in str(name)]
    if forbidden:
        raise RuntimeError("canonical campaign contains prohibited opponent RNG bank")
    expected = planned_intervals()
    if set(intervals) != set(expected):
        raise RuntimeError("canonical seed namespace set drifted")
    for name, interval in expected.items():
        if dict(intervals[name]) != interval.as_dict():
            raise RuntimeError(f"canonical seed interval drifted: {name}")
    if any(
        interval.end_exclusive
        > CONSUMED_LEGACY_SEALED_INTERVAL.start_inclusive
        for interval in expected.values()
    ):
        raise RuntimeError("canonical campaign enters sealed legacy holdout")


def training_microbatch_plan(
    *,
    games: int = TRAINING_GAMES,
    microbatch_games: int = TRAINING_MICROBATCH_GAMES,
    checkpoints: Sequence[int] = CHECKPOINT_GAMES,
) -> list[dict[str, int]]:
    """Return the exact deterministic microbatch/checkpoint consumption plan."""

    if games != TRAINING_GAMES:
        raise ValueError("canonical recovery game cap is fixed at 150000")
    if microbatch_games < 1:
        raise ValueError("microbatch size must be positive")
    thresholds = tuple(int(value) for value in checkpoints)
    if thresholds != CHECKPOINT_GAMES:
        raise ValueError("canonical recovery checkpoint thresholds are immutable")
    completed = 0
    rows: list[dict[str, int]] = []
    for threshold in thresholds:
        while completed < threshold:
            size = min(microbatch_games, threshold - completed)
            rows.append(
                {
                    "start_offset": completed,
                    "end_offset": completed + size,
                    "games": size,
                    "checkpoint_after": int(completed + size == threshold),
                }
            )
            completed += size
    if completed != games or sum(row["games"] for row in rows) != games:
        raise RuntimeError("canonical training microbatch plan is not exact")
    return rows


def training_budget_contract() -> dict[str, Any]:
    rows = training_microbatch_plan()
    return {
        "games": TRAINING_GAMES,
        "microbatch_games": TRAINING_MICROBATCH_GAMES,
        "microbatch_count": len(rows),
        "last_microbatch_games": rows[-1]["games"],
        "scheduled_checkpoints_games": list(CHECKPOINT_GAMES),
        "checkpoint_count": len(CHECKPOINT_GAMES),
        "ppo_epochs": PPO_EPOCHS,
        "wall_clock_cap_seconds": WALL_CLOCK_CAP_SECONDS,
        "wall_clock_enforcement": "before and after each rollout/update/checkpoint",
        "adaptation_after_start": "forbidden",
    }


def _new_native_policy(*, device: torch.device) -> torch.nn.Module:
    """Instantiate only the committed native 96x1 Puffer policy."""

    from temporal_puffer_v4 import NativePufferTemporalPolicy

    policy = NativePufferTemporalPolicy(
        hidden_size=96,
        num_layers=1,
        mode_conditioned_heads=False,
    ).to(device)
    checks = {
        "native_pufferlib": True,
        "hidden_size": 96,
        "num_layers": 1,
        "mode_conditioned_heads": False,
    }
    for name, expected in checks.items():
        if getattr(policy, name, None) != expected:
            raise RuntimeError(f"native policy {name} does not match frozen contract")
    return policy


def _fresh_optimizer_after_warm_start(
    policy: torch.nn.Module,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Create a new optimizer and prove no parent moments were inherited."""

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    state = optimizer.state_dict()
    if state["state"]:
        raise RuntimeError("fresh canonical optimizer unexpectedly has state")
    return optimizer, {
        "class": "torch.optim.Adam",
        "learning_rate": 0.0003,
        "parent_optimizer_state_loaded": False,
        "fresh_optimizer_state_entries_before_first_update": 0,
        "reset_verified": True,
    }


def _checkpoint_lineage(
    *,
    commitment_sha256: str,
    manifest_sha256: str,
    cumulative_games: int,
    metrics: Mapping[str, Any],
    training_seconds: float,
    optimizer_reset: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "random_initialized": False,
        "warm_start_parent_checkpoint": str(PARENT_CHECKPOINT),
        "warm_start_parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "warm_start_parent_manifest_sha256": PARENT_MANIFEST_SHA256,
        "parent_optimizer_state_loaded": False,
        "optimizer_reset": dict(optimizer_reset),
        "reward_only": True,
        "source_opponent_invalid_for_canonical_selection": True,
        "cumulative_training_games": int(cumulative_games),
        "training_wall_seconds": float(training_seconds),
        "metrics": dict(metrics),
        "commitment_sha256": commitment_sha256,
        "manifest_sha256": manifest_sha256,
    }


def _checkpoint_admission(
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cumulative_games: int,
    metrics: Mapping[str, Any],
    all_legal_actions: bool,
    optimizer_reset: Mapping[str, Any],
) -> dict[str, Any]:
    scheduled = int(cumulative_games) in CHECKPOINT_GAMES
    finite = _all_finite(metrics) and all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in policy.parameters()
    )
    optimizer_state = optimizer.state_dict()
    gates = {
        "scheduled_cumulative_games": scheduled,
        "complete_model_state": bool(policy.state_dict()),
        "complete_optimizer_state": bool(optimizer_state),
        "finite_parameters_and_metrics": finite,
        "all_collected_actions_legal": bool(all_legal_actions),
        "weights_only_warm_start": (
            optimizer_reset.get("parent_optimizer_state_loaded") is False
            and optimizer_reset.get("reset_verified") is True
        ),
        "canonical_opponent_only": True,
        "reward_only": True,
    }
    return {"gates": gates, "eligible": all(gates.values())}


def _fsync_path(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _write_checkpoint_exclusive(
    *,
    training_dir: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cumulative_games: int,
    metrics: Mapping[str, Any],
    training_seconds: float,
    commitment_sha256: str,
    manifest_sha256: str,
    optimizer_reset: Mapping[str, Any],
    all_legal_actions: bool,
) -> tuple[Path, dict[str, Any]]:
    """Publish a complete canonical checkpoint or leave terminal evidence only."""

    checkpoint_path = training_dir / f"checkpoint_{cumulative_games:06d}.pt"
    manifest_path = training_dir / f"checkpoint_{cumulative_games:06d}.json"
    pending_checkpoint = checkpoint_path.with_suffix(".pt.pending")
    pending_manifest = manifest_path.with_suffix(".json.pending")
    if any(path.exists() for path in (checkpoint_path, manifest_path, pending_checkpoint, pending_manifest)):
        raise FileExistsError("checkpoint publication path is already occupied")
    admission = _checkpoint_admission(
        policy=policy,
        optimizer=optimizer,
        cumulative_games=cumulative_games,
        metrics=metrics,
        all_legal_actions=all_legal_actions,
        optimizer_reset=optimizer_reset,
    )
    if not admission["eligible"]:
        raise RuntimeError("checkpoint admission failed")
    lineage = _checkpoint_lineage(
        commitment_sha256=commitment_sha256,
        manifest_sha256=manifest_sha256,
        cumulative_games=cumulative_games,
        metrics=metrics,
        training_seconds=training_seconds,
        optimizer_reset=optimizer_reset,
    )
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lineage": lineage,
        "configuration": {
            "model": "NativePufferTemporalPolicy 96x1",
            "reward_only": True,
            "opponent_schedule": ["canonical_random"],
            "ppo_epochs": PPO_EPOCHS,
        },
    }
    torch.save(payload, pending_checkpoint)
    _fsync_path(pending_checkpoint)
    checkpoint_sha = file_sha256(pending_checkpoint)
    record = {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "cumulative_training_games": int(cumulative_games),
        "complete_model_state": True,
        "complete_optimizer_state": True,
        "lineage": lineage,
        "admission": admission,
        "source_sha256": campaign_source_hashes(),
        "commitment_sha256": commitment_sha256,
        "manifest_sha256": manifest_sha256,
    }
    _write_json_exclusive(pending_manifest, record)
    os.replace(pending_checkpoint, checkpoint_path)
    os.replace(pending_manifest, manifest_path)
    return checkpoint_path, record


def _load_canonical_checkpoint_policy(
    *, checkpoint_path: Path, device: torch.device, commitment_sha256: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a selected candidate only after all pre-claim checks succeed."""

    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise RuntimeError("candidate checkpoint format is not canonical recovery")
    lineage = payload.get("lineage", {})
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("commitment_sha256") != commitment_sha256
        or lineage.get("parent_optimizer_state_loaded") is not False
        or lineage.get("random_initialized") is not False
    ):
        raise RuntimeError("candidate checkpoint lineage is not approved canonical warm start")
    policy = _new_native_policy(device=device)
    policy.load_state_dict(payload["model_state_dict"], strict=True)
    return policy, dict(payload)


def _bootstrap_summary(
    evaluation: Mapping[str, Any], *, seed: int
) -> dict[str, Any]:
    pair_scores = tuple(float(value) for value in evaluation["pair_scores"])
    return {
        "seed": int(seed),
        "samples": BOOTSTRAP_SAMPLES,
        "raw_win_rate": raw_win_rate(pair_scores),
        "lower_95": pair_bootstrap_lower_bound(
            pair_scores, seed=int(seed), samples=BOOTSTRAP_SAMPLES
        ),
        "draw_score": 0,
        "unit": "pair mean of two binary wins",
    }


def _development_output_path(training_dir: Path, checkpoint_index: int) -> Path:
    if not 1 <= int(checkpoint_index) <= DEVELOPMENT_CHECKPOINTS:
        raise ValueError("development checkpoint index is out of range")
    return training_dir / f"development_screen_{checkpoint_index:02d}.json"


def _evaluate_development_checkpoint(
    *,
    training_dir: Path,
    checkpoint_index: int,
    checkpoint_path: Path,
    checkpoint_record: Mapping[str, Any],
    manifest: Mapping[str, Any],
    device: torch.device,
    commitment_sha256: str,
) -> dict[str, Any]:
    """Consume exactly one preassigned 500-pair development bank once."""

    output_path = _development_output_path(training_dir, checkpoint_index)
    if output_path.exists():
        raise FileExistsError("development bank was already consumed")
    if checkpoint_record.get("admission", {}).get("eligible") is not True:
        raise RuntimeError("ineligible checkpoint cannot consume a development bank")
    bank_name = f"development_screening_canonical_{checkpoint_index:02d}"
    interval = manifest["seed_registry"]["intervals"][bank_name]
    deal_seeds = _interval_range(interval, label=bank_name)
    if len(deal_seeds) != DEVELOPMENT_PAIRS_PER_CHECKPOINT:
        raise RuntimeError("development bank pair count drifted")
    policy, _ = _load_canonical_checkpoint_policy(
        checkpoint_path=checkpoint_path,
        device=device,
        commitment_sha256=commitment_sha256,
    )
    evaluation = evaluate_canonical_seat_balanced_pairs(
        policy, deal_seeds=deal_seeds
    )
    bootstrap_seed = int(
        manifest["development"]["bootstrap_seed_interval"]["start_inclusive"]
    )
    payload = {
        "format": "pesten-canonical-random-recovery-development-v1",
        "checkpoint_index": int(checkpoint_index),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_record["checkpoint_sha256"],
        "training_games": checkpoint_record["cumulative_training_games"],
        "bank_name": bank_name,
        "pair_interval": dict(interval),
        "evaluation": evaluation,
        "bootstrap": _bootstrap_summary(evaluation, seed=bootstrap_seed),
        "one_use": True,
        "draw_score": 0,
    }
    _write_json_exclusive(output_path, payload)
    return payload


def _rank_development_results(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    candidates = [
        {
            "checkpoint_path": result["checkpoint_path"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "training_games": result["training_games"],
            "pair_scores": result["evaluation"]["pair_scores"],
            "bootstrap_lower_95": result["bootstrap"]["lower_95"],
            "development_checkpoint_index": result["checkpoint_index"],
        }
        for result in results
    ]
    return rank_development_checkpoints(candidates)


def _freeze_development_winner(
    *, training_dir: Path, ranked: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not ranked:
        raise RuntimeError("cannot freeze a development winner without checkpoints")
    path = training_dir / "development_winner.json"
    if path.exists():
        raise FileExistsError("development winner is already frozen")
    selected = dict(ranked[0])
    payload = {
        "format": "pesten-canonical-random-recovery-development-winner-v1",
        "selected_at_utc": _utc_now(),
        "selection": selected,
        "rank_order": [
            "raw win rate descending",
            "pair bootstrap lower 95% descending",
            "fewer training games ascending",
            "checkpoint SHA256 ascending",
        ],
        "ranked_checkpoint_sha256s": [
            str(item["checkpoint_sha256"]) for item in ranked
        ],
        "exactly_one_selected": True,
        "confirmation_not_run": True,
    }
    _write_json_exclusive(path, payload)
    return payload


def _run_confirmation(
    *,
    training_dir: Path,
    development_winner: Mapping[str, Any],
    manifest: Mapping[str, Any],
    device: torch.device,
    commitment_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Run exactly one 2500-pair confirmation and terminally accept/reject it."""

    confirmation_path = training_dir / "confirmation.json"
    failure_path = training_dir / "confirmation_failed.json"
    frozen_path = training_dir / "frozen_winner.json"
    if any(path.exists() for path in (confirmation_path, failure_path, frozen_path)):
        raise FileExistsError("confirmation path is already terminally occupied")
    selected = development_winner.get("selection")
    if not isinstance(selected, Mapping):
        raise RuntimeError("development winner selection is malformed")
    interval = manifest["confirmation"]["pair_interval"]
    deal_seeds = _interval_range(interval, label="confirmation_canonical_deals")
    if len(deal_seeds) != CONFIRMATION_PAIRS:
        raise RuntimeError("confirmation pair count drifted")
    checkpoint_path = Path(str(selected["checkpoint_path"]))
    policy, _ = _load_canonical_checkpoint_policy(
        checkpoint_path=checkpoint_path,
        device=device,
        commitment_sha256=commitment_sha256,
    )
    evaluation = evaluate_canonical_seat_balanced_pairs(
        policy, deal_seeds=deal_seeds
    )
    bootstrap_seed = int(
        manifest["confirmation"]["bootstrap_seed_interval"]["start_inclusive"]
    )
    bootstrap = _bootstrap_summary(evaluation, seed=bootstrap_seed)
    passed = bool(
        bootstrap["raw_win_rate"]
        >= float(manifest["confirmation"]["required_raw_win_rate"])
        and bootstrap["lower_95"]
        >= float(manifest["confirmation"]["required_lower_95_confidence_bound"])
    )
    confirmation = {
        "format": "pesten-canonical-random-recovery-confirmation-v1",
        "created_at_utc": _utc_now(),
        "commitment_sha256": commitment_sha256,
        "manifest_sha256": manifest_sha256,
        "selected_checkpoint": dict(selected),
        "pair_interval": dict(interval),
        "evaluation": evaluation,
        "bootstrap": bootstrap,
        "gate": {
            "raw_win_rate_gte": float(manifest["confirmation"]["required_raw_win_rate"]),
            "lower_95_gte": float(manifest["confirmation"]["required_lower_95_confidence_bound"]),
        },
        "passed": passed,
        "draw_score": 0,
        "one_use": True,
    }
    _write_json_exclusive(confirmation_path, confirmation)
    if not passed:
        failure = {
            "format": CONFIRMATION_FAILED_FORMAT,
            "created_at_utc": _utc_now(),
            "commitment_sha256": commitment_sha256,
            "manifest_sha256": manifest_sha256,
            "confirmation_path": str(confirmation_path),
            "confirmation_sha256": file_sha256(confirmation_path),
            "reason": "canonical confirmation raw win or lower 95% threshold not met",
            "raw_win_rate": bootstrap["raw_win_rate"],
            "lower_95": bootstrap["lower_95"],
            "audit_forbidden": True,
            "frozen_winner_created": False,
        }
        _write_json_exclusive(failure_path, failure)
        return {"confirmation": confirmation, "frozen_winner": None, "failed": failure}
    winner = {
        "format": WINNER_FORMAT,
        "created_at_utc": _utc_now(),
        "commitment_sha256": commitment_sha256,
        "manifest_sha256": manifest_sha256,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "training_games": selected["training_games"],
        "confirmation_path": str(confirmation_path),
        "confirmation_sha256": file_sha256(confirmation_path),
        "confirmation": confirmation,
        "canonical_opponent": canonical_random_contract(),
        "final_audit_not_run": True,
        "final_audit_seed_bank_touched": False,
    }
    _write_json_exclusive(frozen_path, winner)
    return {"confirmation": confirmation, "frozen_winner": winner, "failed": None}


def _training_preflight(
    *, output_dir: Path, training_dir: Path, device_name: str
) -> tuple[dict[str, Any], dict[str, Any], Path, str, str, torch.device]:
    """Validate every mutable precondition before creating training output."""

    manifest = load_frozen_manifest(output_dir)
    _assert_manifest_source_hashes(manifest)
    _assert_canonical_seed_manifest(manifest)
    admission = validate_pretraining_admission(output_dir=output_dir)
    commitment, commitment_path = _load_implementation_commitment(output_dir)
    if commitment.get("training") != manifest.get("training"):
        raise RuntimeError("implementation commitment training contract drifted")
    if commitment.get("development") != manifest.get("development"):
        raise RuntimeError("implementation commitment development contract drifted")
    if commitment.get("confirmation") != manifest.get("confirmation"):
        raise RuntimeError("implementation commitment confirmation contract drifted")
    if commitment.get("final_audit") != manifest.get("final_audit"):
        raise RuntimeError("implementation commitment final-audit contract drifted")
    if Path(training_dir).exists():
        raise FileExistsError(
            f"canonical training output already exists and is not restartable: {training_dir}"
        )
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("canonical recovery training requires one CUDA native-Puffer process")
    return (
        manifest,
        commitment,
        commitment_path,
        file_sha256(commitment_path),
        admission["manifest_sha256"],
        device,
    )


def _cuda_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_canonical_campaign(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    device_name: str = "cuda",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run the precommitted 150,000-game canonical recovery campaign.

    The non-dry path is purposefully non-resumable: any existing output is a
    consumed/partial execution record, so a later operator cannot silently
    replay development, confirmation, or training seeds under a changed state.
    """

    if dry_run:
        return {
            "dry_run": True,
            "candidate_training_started": False,
            "output_dir": str(training_dir),
            "requires": [
                "successful canonical phase-zero calibration",
                "successful native CUDA smoke",
                "immutable implementation commitment",
                "CUDA NativePufferTemporalPolicy 96x1",
            ],
            "budget": training_budget_contract(),
            "opponent_schedule": ["canonical_random"],
            "opponent_rng_bank": False,
            "confirmation_integrated": True,
            "final_audit_seed_bank_touched": False,
        }

    training_dir = Path(training_dir)
    (
        manifest,
        commitment,
        commitment_path,
        commitment_sha256,
        manifest_sha256,
        device,
    ) = _training_preflight(
        output_dir=Path(output_dir),
        training_dir=training_dir,
        device_name=device_name,
    )
    del commitment  # The immutable on-disk SHA is the checkpoint binding.
    initialization = manifest["calibration"]["initialization"]
    initialization_seed = int(initialization["start_inclusive"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    random.seed(initialization_seed)
    # Do all expensive/type-sensitive preflight before any output mutation.
    policy = _new_native_policy(device=device)
    warm_start = load_warm_start_weights_only(policy)
    optimizer, optimizer_reset = _fresh_optimizer_after_warm_start(policy)
    if warm_start["optimizer_state_loaded"] or not optimizer_reset["reset_verified"]:
        raise RuntimeError("canonical warm start did not prove optimizer reset")
    if len(optimizer.state) != 0:
        raise RuntimeError("fresh optimizer contains inherited state before training")

    from temporal_batched_speed import collect_league_batch_batched
    from temporal_training import ppo_update_batched_epochs

    training_interval = manifest["training"]["deal_interval"]
    deal_range = _interval_range(training_interval, label="training_canonical_deals")
    plan = training_microbatch_plan(
        games=int(manifest["training"]["games"]),
        microbatch_games=int(manifest["training"]["microbatch_games"]),
        checkpoints=tuple(manifest["training"]["scheduled_checkpoints_games"]),
    )
    if len(deal_range) != TRAINING_GAMES:
        raise RuntimeError("canonical training seed range does not contain 150000 games")
    if manifest["training"].get("opponent_schedule") != ["canonical_random"]:
        raise RuntimeError("canonical training schedule drifted")
    if manifest["training"].get("opponent_rng_bank") is not False:
        raise RuntimeError("canonical training permits prohibited opponent RNG bank")

    training_dir.mkdir(parents=True, exist_ok=False)
    preflight = {
        "format": "pesten-canonical-random-recovery-training-start-v1",
        "created_at_utc": _utc_now(),
        "commitment_path": str(commitment_path),
        "commitment_sha256": commitment_sha256,
        "manifest_sha256": manifest_sha256,
        "device": str(device),
        "budget": training_budget_contract(),
        "warm_start": warm_start,
        "optimizer_reset": optimizer_reset,
        "candidate_training_started": True,
        "final_audit_seed_bank_touched": False,
        "restart_policy": "fail closed; this directory makes the run non-retryable",
    }
    _write_json_exclusive(training_dir / "training_started.json", preflight)

    started = time.perf_counter()
    training_seconds = 0.0
    completed = 0
    checkpoint_records: list[tuple[Path, dict[str, Any]]] = []
    development_results: list[dict[str, Any]] = []
    all_legal_actions = True
    last_metrics: Mapping[str, Any] = {}
    stop_reason = "exact_game_cap_reached"

    for batch_index, batch in enumerate(plan, start=1):
        _cuda_sync(device)
        training_seconds = time.perf_counter() - started
        if training_seconds >= WALL_CLOCK_CAP_SECONDS:
            stop_reason = "wall_clock_cap_before_microbatch"
            break
        start_offset = int(batch["start_offset"])
        end_offset = int(batch["end_offset"])
        seeds = range(deal_range.start + start_offset, deal_range.start + end_offset)
        if len(seeds) != int(batch["games"]):
            raise RuntimeError("canonical training microbatch seed slicing drifted")
        policy.eval()
        _cuda_sync(device)
        rollout_started = time.perf_counter()
        episodes, metadata = collect_league_batch_batched(
            policy,
            None,
            (),
            games=len(seeds),
            seed=seeds.start,
            schedule=("canonical_random",),
            max_turns=MAX_TURNS,
        )
        _cuda_sync(device)
        rollout_seconds = time.perf_counter() - rollout_started
        if metadata.get("games") != {"canonical_random": len(seeds)}:
            raise RuntimeError("training collector constructed a noncanonical opponent")
        legal_actions = all(
            step.decision.legal_actions[step.action]
            for trajectory, _ in episodes
            for step in trajectory
        )
        if not legal_actions:
            raise RuntimeError("canonical training collector produced an illegal action")
        all_legal_actions = all_legal_actions and legal_actions
        policy.train()
        _cuda_sync(device)
        update_started = time.perf_counter()
        last_metrics = ppo_update_batched_epochs(
            policy,
            episodes,
            optimizer,
            update_epochs=PPO_EPOCHS,
            minibatch_trajectories=16,
            seed=(initialization_seed << 32) | start_offset,
            target_kl=0.02,
            clip_ratio=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
            gamma=0.97,
            gae_lambda=0.95,
            normalize_advantages=True,
        )
        _cuda_sync(device)
        update_seconds = time.perf_counter() - update_started
        if not _all_finite(last_metrics) or not all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in policy.parameters()
        ):
            raise RuntimeError("canonical training generated non-finite metrics or weights")
        completed = end_offset
        _cuda_sync(device)
        training_seconds = time.perf_counter() - started
        _write_json_exclusive(
            training_dir / f"microbatch_{completed:06d}.json",
            {
                "batch_index": batch_index,
                "cumulative_games": completed,
                "games": len(seeds),
                "deal_seed_interval": {
                    "start_inclusive": seeds.start,
                    "end_exclusive": seeds.stop,
                },
                "opponent_schedule": ["canonical_random"],
                "opponent_rng_bank": False,
                "rollout_seconds": rollout_seconds,
                "update_seconds": update_seconds,
                "training_wall_seconds": training_seconds,
                "metrics": last_metrics,
                "legal_actions_verified": legal_actions,
                "checkpoint_after": bool(batch["checkpoint_after"]),
            },
        )
        if training_seconds > WALL_CLOCK_CAP_SECONDS:
            stop_reason = "wall_clock_cap_after_microbatch"
            break
        if not batch["checkpoint_after"]:
            continue

        # Revalidate immutable phase-zero evidence before publishing or
        # screening every scheduled checkpoint.
        validate_pretraining_admission(output_dir=Path(output_dir))
        _cuda_sync(device)
        checkpoint_started = time.perf_counter()
        checkpoint_path, checkpoint_record = _write_checkpoint_exclusive(
            training_dir=training_dir,
            policy=policy,
            optimizer=optimizer,
            cumulative_games=completed,
            metrics=last_metrics,
            training_seconds=training_seconds,
            commitment_sha256=commitment_sha256,
            manifest_sha256=manifest_sha256,
            optimizer_reset=optimizer_reset,
            all_legal_actions=all_legal_actions,
        )
        _cuda_sync(device)
        training_seconds = time.perf_counter() - started
        if training_seconds > WALL_CLOCK_CAP_SECONDS:
            stop_reason = "wall_clock_cap_during_checkpoint"
            break
        checkpoint_records.append((checkpoint_path, checkpoint_record))
        development_results.append(
            _evaluate_development_checkpoint(
                training_dir=training_dir,
                checkpoint_index=len(checkpoint_records),
                checkpoint_path=checkpoint_path,
                checkpoint_record=checkpoint_record,
                manifest=manifest,
                device=device,
                commitment_sha256=commitment_sha256,
            )
        )

    if completed != TRAINING_GAMES or len(checkpoint_records) != len(CHECKPOINT_GAMES):
        incomplete = {
            "format": "pesten-canonical-random-recovery-training-incomplete-v1",
            "created_at_utc": _utc_now(),
            "stop_reason": stop_reason,
            "completed_games": completed,
            "required_games": TRAINING_GAMES,
            "checkpoint_count": len(checkpoint_records),
            "required_checkpoint_count": len(CHECKPOINT_GAMES),
            "training_wall_seconds": training_seconds,
            "confirmation_started": False,
            "final_audit_seed_bank_touched": False,
            "retry_permitted": False,
        }
        _write_json_exclusive(training_dir / "training_incomplete.json", incomplete)
        _write_json_exclusive(
            training_dir / "training_result.json",
            {**incomplete, "format": TRAINING_RESULT_FORMAT, "completed": False},
        )
        return incomplete

    ranked = _rank_development_results(development_results)
    development_winner = _freeze_development_winner(
        training_dir=training_dir, ranked=ranked
    )
    confirmation_result = _run_confirmation(
        training_dir=training_dir,
        development_winner=development_winner,
        manifest=manifest,
        device=device,
        commitment_sha256=commitment_sha256,
        manifest_sha256=manifest_sha256,
    )
    final = {
        "format": TRAINING_RESULT_FORMAT,
        "created_at_utc": _utc_now(),
        "completed": True,
        "completed_games": completed,
        "training_wall_seconds": training_seconds,
        "checkpoint_count": len(checkpoint_records),
        "development_screens": len(development_results),
        "development_winner_path": str(training_dir / "development_winner.json"),
        "confirmation_path": str(training_dir / "confirmation.json"),
        "confirmation_passed": bool(confirmation_result["confirmation"]["passed"]),
        "frozen_winner_path": (
            str(training_dir / "frozen_winner.json")
            if confirmation_result["frozen_winner"] is not None
            else None
        ),
        "confirmation_failed_path": (
            str(training_dir / "confirmation_failed.json")
            if confirmation_result["failed"] is not None
            else None
        ),
        "final_audit_seed_bank_touched": False,
        "opponent_rng_bank": False,
    }
    _write_json_exclusive(training_dir / "training_result.json", final)
    return final


def _preflight_final_audit(
    *, output_dir: Path, training_dir: Path, device_name: str
) -> tuple[dict[str, Any], str, str, dict[str, Any], Path, dict[str, Any], torch.device]:
    """Validate commitment, confirmation, checkpoint and CUDA before claim."""

    manifest = load_frozen_manifest(output_dir)
    _assert_manifest_source_hashes(manifest)
    _assert_canonical_seed_manifest(manifest)
    validate_pretraining_admission(output_dir=output_dir)
    commitment, commitment_path = _load_implementation_commitment(output_dir)
    commitment_sha256 = file_sha256(commitment_path)
    manifest_sha256 = file_sha256(Path(output_dir) / FROZEN_MANIFEST_NAME)
    failure_path = Path(training_dir) / "confirmation_failed.json"
    if failure_path.exists():
        raise RuntimeError("canonical confirmation failed; final audit is forbidden")
    winner_path = Path(training_dir) / "frozen_winner.json"
    winner = _read_json(winner_path)
    if winner.get("format") != WINNER_FORMAT:
        raise RuntimeError("canonical frozen winner is missing or malformed")
    if (
        winner.get("commitment_sha256") != commitment_sha256
        or winner.get("manifest_sha256") != manifest_sha256
        or winner.get("final_audit_not_run") is not True
        or winner.get("final_audit_seed_bank_touched") is not False
    ):
        raise RuntimeError("canonical frozen winner binding is invalid for final audit")
    confirmation = winner.get("confirmation", {})
    if not isinstance(confirmation, Mapping) or confirmation.get("passed") is not True:
        raise RuntimeError("only a passing canonical confirmation may enter audit")
    checkpoint_path = Path(str(winner.get("checkpoint_path", "")))
    if not checkpoint_path.is_file():
        raise FileNotFoundError("frozen winner checkpoint is missing")
    if file_sha256(checkpoint_path) != winner.get("checkpoint_sha256"):
        raise RuntimeError("frozen winner checkpoint digest changed")
    checkpoint_record = _read_json(checkpoint_path.with_suffix(".json"))
    if (
        checkpoint_record.get("format") != CHECKPOINT_FORMAT
        or checkpoint_record.get("checkpoint_sha256") != winner.get("checkpoint_sha256")
        or checkpoint_record.get("commitment_sha256") != commitment_sha256
        or checkpoint_record.get("admission", {}).get("eligible") is not True
    ):
        raise RuntimeError("frozen winner checkpoint admission binding is invalid")
    claim_path, result_path = canonical_final_audit_paths(output_dir)
    if claim_path.exists() or result_path.exists():
        raise FileExistsError("canonical final audit is one-use and cannot retry")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("canonical final audit requires CUDA before claim")
    return (
        manifest,
        commitment_sha256,
        manifest_sha256,
        winner,
        checkpoint_path,
        checkpoint_record,
        device,
    )


def audit_canonical_winner(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    device_name: str = "cuda",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute the one-use canonical 5,000-pair audit after confirmation.

    The final interval is deliberately converted to a seed iterator only after
    the durable claim has been created.  Any post-claim failure burns the
    audit bank and is non-retryable by design.
    """

    if dry_run:
        claim_path, result_path = canonical_final_audit_paths(output_dir)
        return {
            "dry_run": True,
            "candidate_model_loaded": False,
            "final_seed_iterator_constructed": False,
            "claim_path": str(claim_path),
            "result_path": str(result_path),
            "order": audit_claim_order(),
            "requires_passing_confirmation": True,
            "opponent_rng_bank": False,
        }
    (
        manifest,
        commitment_sha256,
        manifest_sha256,
        winner,
        checkpoint_path,
        _checkpoint_record,
        device,
    ) = _preflight_final_audit(
        output_dir=Path(output_dir),
        training_dir=Path(training_dir),
        device_name=device_name,
    )
    # This creates the immutable claim before model load or final seed range.
    claim = create_final_audit_claim(
        output_dir=Path(output_dir),
        winner_sha256=str(winner["checkpoint_sha256"]),
        commitment_sha256=commitment_sha256,
        manifest_sha256=manifest_sha256,
    )
    claim_path, result_path = canonical_final_audit_paths(output_dir)
    # Bind the preflight facts after the claim is durable without changing its
    # order guarantee.  Claim content contains the candidate hash; these facts
    # are independently bound by result and checkpoint records.
    if claim.get("candidate_checkpoint_sha256") != winner.get("checkpoint_sha256"):
        raise RuntimeError("final audit claim candidate binding drifted")
    if (
        claim.get("commitment_sha256") != commitment_sha256
        or claim.get("manifest_sha256") != manifest_sha256
    ):
        raise RuntimeError("final audit claim preflight binding drifted")
    policy, checkpoint_payload = _load_canonical_checkpoint_policy(
        checkpoint_path=checkpoint_path,
        device=device,
        commitment_sha256=commitment_sha256,
    )
    # Do not construct this iterator before the preceding exclusive claim.
    final_interval = manifest["final_audit"]["pair_interval"]
    final_deal_seeds = _interval_range(
        final_interval, label="final_audit_canonical_deals"
    )
    if len(final_deal_seeds) != FINAL_AUDIT_PAIRS:
        raise RuntimeError("canonical final audit pair count drifted")
    evaluation = evaluate_canonical_seat_balanced_pairs(
        policy, deal_seeds=final_deal_seeds
    )
    bootstrap_seed = int(
        manifest["final_audit"]["bootstrap_seed_interval"]["start_inclusive"]
    )
    bootstrap = _bootstrap_summary(evaluation, seed=bootstrap_seed)
    passed = bool(
        bootstrap["raw_win_rate"]
        >= float(manifest["final_audit"]["required_raw_win_rate"])
        and bootstrap["lower_95"]
        >= float(manifest["final_audit"]["required_lower_95_confidence_bound"])
    )
    result = {
        "format": AUDIT_RESULT_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "completed_at_utc": _utc_now(),
        "claim_path": str(claim_path),
        "claim_sha256": file_sha256(claim_path),
        "commitment_sha256": commitment_sha256,
        "manifest_sha256": manifest_sha256,
        "checkpoint_sha256": winner["checkpoint_sha256"],
        "checkpoint_lineage": checkpoint_payload["lineage"],
        "pair_interval": dict(final_interval),
        "evaluation": evaluation,
        "wins": evaluation["outcomes"]["wins"],
        "losses": evaluation["outcomes"]["losses"],
        "draws": evaluation["outcomes"]["draws"],
        "seat_results": {
            "seat_0": {
                "wins": evaluation["outcomes"]["wins_as_seat_0"],
                "losses": evaluation["outcomes"]["losses_as_seat_0"],
                "draws": evaluation["outcomes"]["draws_as_seat_0"],
            },
            "seat_1": {
                "wins": evaluation["outcomes"]["wins_as_seat_1"],
                "losses": evaluation["outcomes"]["losses_as_seat_1"],
                "draws": evaluation["outcomes"]["draws_as_seat_1"],
            },
        },
        "bootstrap": bootstrap,
        "gate": {
            "raw_win_rate_gte": float(manifest["final_audit"]["required_raw_win_rate"]),
            "lower_95_gte": float(manifest["final_audit"]["required_lower_95_confidence_bound"]),
        },
        "passed": passed,
        "one_use": True,
        "draw_score": 0,
    }
    _write_json_exclusive(result_path, result)
    return result


def canonical_training_wrapper(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    training_dir: Path = DEFAULT_TRAINING_DIR,
    device_name: str = "cuda",
    dry_run: bool = True,
) -> dict[str, Any]:
    """Compatibility wrapper for the reviewed train execution surface."""

    return train_canonical_campaign(
        output_dir=output_dir,
        training_dir=training_dir,
        device_name=device_name,
        dry_run=dry_run,
    )


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bootstrap")
    commands.add_parser("calibrate")
    commands.add_parser("native-smoke")
    commands.add_parser("precommit")
    commands.add_parser("train")
    commands.add_parser("audit")
    commands.add_parser("plan")
    for name in ("bootstrap", "calibrate", "native-smoke", "precommit", "plan"):
        commands.choices[name].add_argument("--dry-run", action="store_true")
        commands.choices[name].add_argument(
            "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
        )
    for name in ("train", "audit"):
        commands.choices[name].add_argument("--dry-run", action="store_true")
        commands.choices[name].add_argument(
            "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR
        )
        commands.choices[name].add_argument(
            "--training-dir", type=Path, default=DEFAULT_TRAINING_DIR
        )
        commands.choices[name].add_argument(
            "--device", choices=("cuda",), default="cuda"
        )
    args = parser.parse_args(argv)
    if args.command == "bootstrap":
        result = bootstrap_campaign(output_dir=args.output_dir, dry_run=args.dry_run)
    elif args.command == "calibrate":
        result = run_phase_zero_calibration(
            output_dir=args.output_dir, dry_run=args.dry_run
        )
    elif args.command == "native-smoke":
        result = execute_native_calibration_smoke(
            output_dir=args.output_dir, dry_run=args.dry_run
        )
    elif args.command == "precommit":
        result = build_implementation_commitment(
            output_dir=args.output_dir, dry_run=args.dry_run
        )
    elif args.command == "train":
        result = train_canonical_campaign(
            output_dir=args.output_dir,
            training_dir=args.training_dir,
            device_name=args.device,
            dry_run=args.dry_run,
        )
    elif args.command == "audit":
        result = audit_canonical_winner(
            output_dir=args.output_dir,
            training_dir=args.training_dir,
            device_name=args.device,
            dry_run=args.dry_run,
        )
    else:
        result = canonical_training_wrapper(
            output_dir=args.output_dir,
            training_dir=DEFAULT_TRAINING_DIR,
            dry_run=True,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
