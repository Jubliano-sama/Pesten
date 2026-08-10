"""Immutable runner for the frozen random-specialist campaign.

The phase-zero workflow intentionally stopped before candidate training.  This
module is the only later execution surface: it binds the frozen plan to an
implementation commitment, trains the one admitted reward-only lane, evaluates
only the prescribed banks, and burns the independent final audit before a
candidate checkpoint is loaded.

It deliberately keeps PufferLib imports lazy.  That lets the ordinary project
test environment inspect the immutable protocol while native execution remains
confined to ``puffer4-native/.venv``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

import game
from random_specialist_workflow import (
    CAMPAIGN_ID,
    CALIBRATION_MAX_TURNS,
    FINAL_AUDIT_BOOTSTRAP_SAMPLES,
    SOURCE_FILES,
    audited_uniform_agent_factory,
    audited_uniform_contract,
    campaign_source_hashes,
    file_sha256,
    validate_frozen_campaign,
)
from temporal_observation import serialize_safe_history
from temporal_puffer_agent import temporal_agent_factory


LEGACY_V1_COMMITMENT_FORMAT = "pesten-random-specialist-implementation-commitment-v1"
LEGACY_V2_COMMITMENT_FORMAT = "pesten-random-specialist-implementation-commitment-v2"
COMMITMENT_FORMAT = "pesten-random-specialist-implementation-commitment-v3"
SMOKE_FORMAT = "pesten-random-specialist-runner-calibration-smoke-v1"
EVALUATION_FORMAT = "pesten-random-specialist-paired-evaluation-v1"
WINNER_FORMAT = "pesten-random-specialist-frozen-winner-v1"
AUDIT_CLAIM_FORMAT = "pesten-random-specialist-final-audit-claim-v1"
AUDIT_RESULT_FORMAT = "pesten-random-specialist-final-audit-result-v1"
CHECKPOINT_FORMAT = "pesten-random-specialist-recurrent-checkpoint-v1"

RUNNER_SOURCE_FILES = ("random_specialist_campaign.py", *SOURCE_FILES)
MICROBATCH_GAMES = 128
BOOTSTRAP_SAMPLES = FINAL_AUDIT_BOOTSTRAP_SAMPLES
BOOTSTRAP_CHUNK_SIZE = 64
BOOTSTRAP_QUANTILES = (0.025, 0.975)
FROZEN_CALIBRATION_SHA256 = "73c7437074d44a96a60603643604f95c454be4dd2208f0097abdf60b44c49ce7"
FROZEN_PRIVACY_FINGERPRINT = "7f9bd563857ccd91330bb340fb721889ac7506d75599d97f799220e7c49bc2a0"
FIXED_RANDOM_PARITY_FINGERPRINT_SCHEMA = (
    "pesten-random-specialist-fixed-random-parity-fingerprint-v1"
)
FIXED_RANDOM_PARITY_CANONICALIZATION = (
    "UTF-8 json.dumps(sort_keys=True,separators=(',', ':'),allow_nan=False) "
    "over {'schema', 'fixed_random_parity'}"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value))


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Durably publish one immutable JSON record without replacement."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # A partially created immutable record is evidence, not a retryable
        # scratch file.  Do not delete it behind the operator's back.
        raise


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _source_hashes() -> dict[str, str]:
    root = Path.cwd()
    return {name: file_sha256(root / name) for name in RUNNER_SOURCE_FILES}


def _state_sha256(state: Any) -> str:
    """Hash recurrent state for reset evidence without exposing policy inputs."""

    parts = (state,) if isinstance(state, torch.Tensor) else tuple(state)
    digest = hashlib.sha256()
    for part in parts:
        value = part.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_all_finite(child) for child in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return True


def _require_uint32_seeds(values: Iterable[int], *, label: str) -> tuple[int, ...]:
    resolved = tuple(int(value) for value in values)
    if not resolved:
        raise ValueError(f"{label} must not be empty")
    if any(value < 0 or value >= 2**32 for value in resolved):
        raise ValueError(f"{label} must contain uint32-compatible seeds")
    return resolved


def current_policy_rollout_seed(deal_seed: int) -> int:
    """The collector's fixed, public stochastic-current-policy seed mapping.

    ``collect_league_batch_batched`` uses this formula when it instantiates the
    current policy's CPU ``torch.Generator``.  It is derived from the
    registered training deal member, is never passed through ``DecisionInput``,
    and remains separate from the independently registered opponent-RNG bank.
    """

    if not 0 <= int(deal_seed) < 2**32:
        raise ValueError("deal seed must be uint32-compatible")
    return int(deal_seed) * 2


def update_rng_seed(initialization_seed: int, cumulative_games: int) -> int:
    """Domain-separated deterministic update-shuffle seed under init root."""

    if not 0 <= int(initialization_seed) < 2**32:
        raise ValueError("initialization seed must be uint32-compatible")
    if cumulative_games < 0:
        raise ValueError("cumulative games must be non-negative")
    return (int(initialization_seed) << 32) | int(cumulative_games)


def bootstrap_algorithm() -> dict[str, Any]:
    """The exact pair-level bootstrap, frozen identically for every phase."""

    return {
        "unit": "pair score",
        "pair_score": "mean of the two seat-rotated candidate game scores",
        "draw_score": 0.5,
        "rng_class": "numpy.random.RandomState",
        "rng_algorithm": "MT19937",
        "sampling": "rng.randint(0, pair_count, size=(chunk, pair_count))",
        "with_replacement": True,
        "samples": BOOTSTRAP_SAMPLES,
        "chunk_size": BOOTSTRAP_CHUNK_SIZE,
        "dtype": "float64",
        "quantiles": list(BOOTSTRAP_QUANTILES),
        "quantile_method": "numpy.quantile(method='linear')",
    }


def bootstrap_pair_scores(
    pair_scores: Sequence[float],
    *,
    seed: int,
) -> dict[str, Any]:
    """Run the committed bootstrap without materialising index mega-arrays."""

    scores = np.asarray(tuple(float(value) for value in pair_scores), dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0:
        raise ValueError("pair bootstrap requires a non-empty one-dimensional score list")
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("pair scores must be finite values in [0, 1]")
    if not 0 <= int(seed) < 2**32:
        raise ValueError("bootstrap seed must be uint32-compatible")

    algorithm = bootstrap_algorithm()
    rng = np.random.RandomState(int(seed))
    samples = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    offset = 0
    while offset < BOOTSTRAP_SAMPLES:
        count = min(BOOTSTRAP_CHUNK_SIZE, BOOTSTRAP_SAMPLES - offset)
        indices = rng.randint(0, scores.size, size=(count, scores.size))
        samples[offset : offset + count] = scores[indices].mean(
            axis=1, dtype=np.float64
        )
        offset += count
    lower, upper = np.quantile(
        samples,
        BOOTSTRAP_QUANTILES,
        method="linear",
    )
    return {
        "algorithm": algorithm,
        "seed": int(seed),
        "pair_count": int(scores.size),
        "point_estimate": float(scores.mean(dtype=np.float64)),
        "ci95": {"lower": float(lower), "upper": float(upper)},
        "bootstrap_means_sha256": _sha256_bytes(samples.tobytes()),
    }


def _trace_temporal_factory(
    policy: torch.nn.Module,
    *,
    deterministic: bool,
    seed: int,
    action_trace: list[dict[str, Any]],
):
    """Add environment-side action accounting to the safe temporal adapter."""

    base = temporal_agent_factory(policy, deterministic=deterministic, seed=seed)

    class TraceTemporalAgent(base):
        def _record_last_action(self, phase: str) -> None:
            step = self.trajectory[-1]
            action_trace.append(
                {
                    "seat": int(self.game.players.index(self)),
                    "role": "candidate",
                    "phase": phase,
                    "action": int(step.action),
                    "legal_actions": list(step.decision.legal_actions),
                }
            )

        def playCard(self, current_sort, current_true_number):
            retry_active = self._retry_card_for_current_turn() is not None
            result = super().playCard(current_sort, current_true_number)
            self._record_last_action("draw_retry" if retry_active else "card")
            return result

        def changeSort(self):
            result = super().changeSort()
            self._record_last_action("suit")
            return result

    return TraceTemporalAgent


def _initial_deal_hash(match: game.Game) -> str:
    assert match.game_deck is not None
    projection = {
        "hands": [
            [int(held.number) for held in player.mydeck.cards]
            for player in match.players
        ],
        "opening_card": int(match.game_deck.topCard().number),
    }
    return _json_sha256(projection)


def _public_history_hash(match: game.Game) -> str:
    rows = []
    for item in match.public_history:
        action = item.get("action")
        row: dict[str, Any] = {"action": action, "player": int(item.get("player", -1))}
        if action == "play":
            row["card"] = int(item["card"])
            row["sort"] = str(item["sort"])
        elif action == "choose_sort":
            row["sort"] = str(item["sort"])
        rows.append(row)
    return _json_sha256(rows)


def _score_outcome(winner: int | None, candidate_seat: int) -> tuple[str, float]:
    if winner is None:
        return "draw", 0.5
    if int(winner) == int(candidate_seat):
        return "win", 1.0
    return "loss", 0.0


def _run_scalar_game(
    policy: torch.nn.Module,
    *,
    deal_seed: int,
    opponent_seed: int,
    candidate_seat: int,
    max_turns: int,
) -> dict[str, Any]:
    """Run one deterministic scalar game and retain compact audit evidence."""

    if candidate_seat not in (0, 1):
        raise ValueError("candidate seat must be 0 or 1")
    if max_turns != CALIBRATION_MAX_TURNS:
        raise ValueError("the frozen evaluator always uses max_turns=1000")
    _require_uint32_seeds((deal_seed, opponent_seed), label="scalar game seeds")

    # Game/deck randomness and the audited opponent's local RNG are separate.
    np.random.seed(int(deal_seed))
    random.seed(int(deal_seed))
    action_trace: list[dict[str, Any]] = []
    candidate_factory = _trace_temporal_factory(
        policy,
        deterministic=True,
        seed=current_policy_rollout_seed(int(deal_seed)),
        action_trace=action_trace,
    )
    opponent_factory = audited_uniform_agent_factory(
        int(opponent_seed), action_trace=action_trace
    )
    factories = (
        [candidate_factory, opponent_factory]
        if candidate_seat == 0
        else [opponent_factory, candidate_factory]
    )
    match = game.Game(factories)
    match.reset()
    candidate = match.players[candidate_seat]
    opponent = match.players[1 - candidate_seat]
    candidate_initial_state = _state_sha256(candidate.state)
    initial_deal_sha256 = _initial_deal_hash(match)

    while not match.game_over and match.turn_count < max_turns:
        match.simulate_turn()

    candidate_actions = [
        record for record in action_trace if record["role"] == "candidate"
    ]
    opponent_actions = [
        record
        for record in action_trace
        if record["role"] == "audited_uniform_random"
    ]
    all_actions = (*candidate_actions, *opponent_actions)
    legal_actions = all(
        bool(record["legal_actions"][int(record["action"])])
        for record in all_actions
    )
    if not legal_actions:
        raise RuntimeError("scalar evaluator observed an illegal action")
    outcome, score = _score_outcome(match.winner, candidate_seat)
    candidate_final_state = _state_sha256(candidate.state)
    safe_history_sha256 = _sha256_bytes(
        serialize_safe_history(candidate.safe_history)
    )
    turns = int(match.turn_count)
    game_completed = bool(match.game_over)
    public_history_sha256 = _public_history_hash(match)

    # Reuse the same Game and same temporal agent to make state leakage visible.
    match.reset()
    candidate_reset_state = _state_sha256(candidate.state)
    opponent_history_cleared = not getattr(opponent, "action_history", ())
    reset_verified = (
        candidate_reset_state == candidate_initial_state and opponent_history_cleared
    )
    if not reset_verified:
        raise RuntimeError("scalar evaluator recurrent/opponent reset failed")

    compact_trace = {
        "candidate_actions": candidate_actions,
        "opponent_actions": opponent_actions,
        "public_history_sha256": public_history_sha256,
    }
    return {
        "deal_seed": int(deal_seed),
        "opponent_rng_seed": int(opponent_seed),
        "candidate_policy_rng_seed": current_policy_rollout_seed(int(deal_seed)),
        "candidate_seat": int(candidate_seat),
        "outcome": outcome,
        "score": score,
        "turn_count": turns,
        "game_completed": game_completed,
        "turn_cap": max_turns,
        "initial_deal_sha256": initial_deal_sha256,
        "public_history_sha256": public_history_sha256,
        "candidate": {
            "actions": len(candidate_actions),
            "legal_actions": len(candidate_actions),
            "actions_sha256": _json_sha256(candidate_actions),
            "safe_history_sha256": safe_history_sha256,
            "initial_state_sha256": candidate_initial_state,
            "final_state_sha256": candidate_final_state,
            "reset_state_sha256": candidate_reset_state,
            "reset_verified": True,
        },
        "opponent": {
            "actions": len(opponent_actions),
            "legal_actions": len(opponent_actions),
            "actions_sha256": _json_sha256(opponent_actions),
            "rng_reset_verified": bool(opponent_history_cleared),
        },
        "legal_actions_verified": True,
        "action_trace_sha256": _json_sha256(compact_trace),
    }


def evaluator_mapping() -> dict[str, Any]:
    """Immutable pairing and reset mapping shared by development/audit phases."""

    return {
        "implementation": "scalar Game + temporal_agent_factory + audited_uniform_agent_factory",
        "candidate_policy": "NativePufferTemporalPolicy",
        "opponent": "audited_uniform_agent_factory",
        "pair_index_mapping": {
            "deal_seed": "deal_range.start_inclusive + pair_index",
            "opponent_rng_seed": "opponent_rng_range.start_inclusive + pair_index",
            "games": [
                {
                    "candidate_seat": 0,
                    "deal_seed": "same pair deal_seed",
                    "opponent_rng_seed": "same pair opponent_rng_seed",
                },
                {
                    "candidate_seat": 1,
                    "deal_seed": "same pair deal_seed",
                    "opponent_rng_seed": "same pair opponent_rng_seed",
                },
            ],
        },
        "candidate_inference": "deterministic argmax over legal actions",
        "candidate_policy_rng_derivation": "2 * registered deal_seed",
        "recurrent_reset": "new Game per rotated game plus same-instance Game.reset hash check",
        "opponent_reset": "audited local MT19937 reset on every Game.reset",
        "draw_score": 0.5,
        "max_turns": CALIBRATION_MAX_TURNS,
        "recording": {
            "per_game": [
                "deal/opponent RNG seeds",
                "candidate seat/outcome/score/turn count",
                "candidate and opponent action/legal counters and digests",
                "public/deal/safe-history hashes",
                "initial/final/reset recurrent hashes",
            ],
            "per_pair": "both rotations, common seed assertion, pair score",
        },
    }


def evaluate_seat_balanced_pairs(
    policy: torch.nn.Module,
    *,
    deal_seeds: Sequence[int],
    opponent_seeds: Sequence[int],
    phase: str,
    require_native_policy: bool = False,
) -> dict[str, Any]:
    """Evaluate every pair twice, with exactly the same deal/RNG in both seats."""

    deals = _require_uint32_seeds(deal_seeds, label="pair deal seeds")
    opponents = _require_uint32_seeds(opponent_seeds, label="pair opponent RNG seeds")
    if len(deals) != len(opponents):
        raise ValueError("deal and opponent RNG pair banks must have equal length")
    if require_native_policy and not bool(getattr(policy, "native_pufferlib", False)):
        raise TypeError("campaign evaluation requires NativePufferTemporalPolicy")

    was_training = bool(policy.training)
    policy.eval()
    pairs: list[dict[str, Any]] = []
    try:
        for pair_index, (deal_seed, opponent_seed) in enumerate(zip(deals, opponents)):
            first = _run_scalar_game(
                policy,
                deal_seed=deal_seed,
                opponent_seed=opponent_seed,
                candidate_seat=0,
                max_turns=CALIBRATION_MAX_TURNS,
            )
            second = _run_scalar_game(
                policy,
                deal_seed=deal_seed,
                opponent_seed=opponent_seed,
                candidate_seat=1,
                max_turns=CALIBRATION_MAX_TURNS,
            )
            if (
                first["deal_seed"] != second["deal_seed"]
                or first["opponent_rng_seed"] != second["opponent_rng_seed"]
                or [first["candidate_seat"], second["candidate_seat"]] != [0, 1]
            ):
                raise RuntimeError("seat-balanced evaluator pairing drifted")
            pair_score = (float(first["score"]) + float(second["score"])) / 2.0
            pair = {
                "pair_index": pair_index,
                "deal_seed": deal_seed,
                "opponent_rng_seed": opponent_seed,
                "candidate_seats": [0, 1],
                "same_deal_and_opponent_rng_each_seat": True,
                "pair_score": pair_score,
                "games": [first, second],
            }
            pair["pair_trace_sha256"] = _json_sha256(pair)
            pairs.append(pair)
    finally:
        policy.train(was_training)

    pair_scores = [float(pair["pair_score"]) for pair in pairs]
    result = {
        "format": EVALUATION_FORMAT,
        "phase": str(phase),
        "evaluator_mapping": evaluator_mapping(),
        "pair_count": len(pairs),
        "game_count": len(pairs) * 2,
        "point_estimate": float(np.mean(np.asarray(pair_scores, dtype=np.float64))),
        "all_legal_actions_verified": all(
            game_result["legal_actions_verified"]
            for pair in pairs
            for game_result in pair["games"]
        ),
        "all_recurrent_resets_verified": all(
            game_result["candidate"]["reset_verified"]
            for pair in pairs
            for game_result in pair["games"]
        ),
        "all_opponent_resets_verified": all(
            game_result["opponent"]["rng_reset_verified"]
            for pair in pairs
            for game_result in pair["games"]
        ),
        "pair_scores": pair_scores,
        "pairs": pairs,
    }
    result["evaluation_sha256"] = _json_sha256(
        {name: value for name, value in result.items() if name != "evaluation_sha256"}
    )
    return result


def _interval_values(interval: Mapping[str, Any]) -> range:
    start = int(interval["start_inclusive"])
    end = int(interval["end_exclusive"])
    if not 0 <= start < end <= 2**32:
        raise ValueError("invalid frozen uint32 interval")
    return range(start, end)


def _manifest_interval(manifest: Mapping[str, Any], namespace: str) -> Mapping[str, Any]:
    allocations = manifest.get("namespace_allocations", {})
    future = allocations.get("future_campaign", {})
    calibration = allocations.get("calibration", {})
    source = future if namespace in future else calibration
    interval = source.get(namespace)
    if not isinstance(interval, Mapping):
        raise ValueError(f"frozen manifest lacks namespace {namespace!r}")
    return interval


def _registry_records(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        record = json.loads(raw_line)
        if record.get("record_type") != "seed_allocation":
            continue
        namespace = str(record.get("namespace", ""))
        if not namespace or namespace in records:
            raise ValueError("registry has missing or duplicate allocation namespaces")
        records[namespace] = record
    if not records:
        raise ValueError("seed registry has no allocation records")
    return records


def _expected_namespace_intervals(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    allocations = manifest.get("namespace_allocations", {})
    merged: dict[str, Mapping[str, Any]] = {}
    for group_name in ("calibration", "future_campaign"):
        group = allocations.get(group_name, {})
        if not isinstance(group, Mapping):
            raise ValueError(f"frozen manifest has invalid {group_name} allocations")
        for namespace, interval in group.items():
            if namespace in merged or not isinstance(interval, Mapping):
                raise ValueError("frozen allocation namespace is malformed")
            merged[str(namespace)] = interval
    return merged


def _namespace_snapshot(
    manifest: Mapping[str, Any], registry_path: Path
) -> dict[str, Any]:
    records = _registry_records(registry_path)
    expected = _expected_namespace_intervals(manifest)
    result: dict[str, Any] = {}
    for namespace, interval in sorted(expected.items()):
        record = records.get(namespace)
        if record is None:
            raise ValueError(f"registry lacks committed namespace {namespace}")
        if record.get("interval") != dict(interval):
            raise ValueError(f"registry interval drifted for {namespace}")
        result[namespace] = {
            "record": record,
            "record_sha256": _json_sha256(record),
        }
    return result


def fixed_random_parity_fingerprint(
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    """Canonical, versioned digest of the frozen scalar/batched parity proof."""

    if not isinstance(parity, Mapping):
        raise ValueError("fixed-random parity evidence must be a JSON object")
    record = {
        "schema": FIXED_RANDOM_PARITY_FINGERPRINT_SCHEMA,
        "fixed_random_parity": dict(parity),
    }
    return {
        "schema": FIXED_RANDOM_PARITY_FINGERPRINT_SCHEMA,
        "canonicalization": FIXED_RANDOM_PARITY_CANONICALIZATION,
        "sha256": _json_sha256(record),
    }


def _validate_fixed_random_parity(parity: Mapping[str, Any]) -> dict[str, Any]:
    if parity.get("passed") is not True:
        raise RuntimeError("fixed-random scalar/batched parity evidence did not pass")
    cases = parity.get("cases")
    if not isinstance(cases, list) or not cases:
        raise RuntimeError("fixed-random parity evidence has no cases")
    for case in cases:
        if not isinstance(case, Mapping) or case.get("passed") is not True:
            raise RuntimeError("fixed-random parity case did not pass")
        if case.get("mismatches") not in ([], ()):
            raise RuntimeError("fixed-random parity case records mismatches")
        if case.get("scalar_trace_sha256") != case.get("batched_trace_sha256"):
            raise RuntimeError("fixed-random parity trace hashes differ")
        if case.get("recurrent_reset_verified") is not True:
            raise RuntimeError("fixed-random parity recurrent reset proof failed")
    return fixed_random_parity_fingerprint(parity)


def calibration_admission_evidence(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Read and verify calibration evidence required before any checkpoint work."""

    calibration_ref = manifest.get("calibration", {})
    if not isinstance(calibration_ref, Mapping):
        raise RuntimeError("frozen manifest lacks calibration reference")
    calibration_path_value = calibration_ref.get("path")
    expected_sha256 = calibration_ref.get("sha256")
    if not isinstance(calibration_path_value, str) or not isinstance(expected_sha256, str):
        raise RuntimeError("frozen calibration path/SHA is malformed")
    if expected_sha256 != FROZEN_CALIBRATION_SHA256:
        raise RuntimeError("frozen calibration SHA differs from the approved phase-zero artifact")
    calibration_path = Path(calibration_path_value)
    if file_sha256(calibration_path) != expected_sha256:
        raise RuntimeError("calibration artifact SHA changed before pretraining")
    calibration = _read_json(calibration_path)
    if calibration.get("status") != "passed":
        raise RuntimeError("calibration artifact is not passed")
    expected_gates = calibration_ref.get("gates")
    actual_gates = calibration.get("gates")
    if not isinstance(expected_gates, Mapping) or not isinstance(actual_gates, Mapping):
        raise RuntimeError("calibration gates are malformed")
    if dict(actual_gates) != dict(expected_gates) or not actual_gates or not all(
        bool(value) for value in actual_gates.values()
    ):
        raise RuntimeError("one or more required frozen calibration gates failed")
    privacy = calibration.get("privacy_boundary", {})
    execution = manifest.get("execution", {})
    if not isinstance(privacy, Mapping) or not isinstance(execution, Mapping):
        raise RuntimeError("calibration privacy evidence is malformed")
    execution_privacy = execution.get("privacy", {})
    if not isinstance(execution_privacy, Mapping):
        raise RuntimeError("frozen execution privacy contract is malformed")
    privacy_fingerprint = privacy.get("safe_schema_sha256")
    frozen_privacy_fingerprint = execution_privacy.get("fingerprint")
    if (
        privacy.get("passed") is not True
        or privacy_fingerprint != frozen_privacy_fingerprint
        or privacy_fingerprint != FROZEN_PRIVACY_FINGERPRINT
    ):
        raise RuntimeError("calibration privacy fingerprint does not match frozen contract")
    parity = calibration.get("fixed_random_parity")
    parity_fingerprint = _validate_fixed_random_parity(parity)
    return {
        "calibration": {
            "path": str(calibration_path),
            "sha256": expected_sha256,
            "required_gates": dict(expected_gates),
            "all_required_gates_true": True,
        },
        "privacy": {
            "fingerprint": privacy_fingerprint,
            "frozen_fingerprint": frozen_privacy_fingerprint,
            "matches": True,
        },
        "fixed_random_parity": {
            **parity_fingerprint,
            "evidence_passed": True,
            "case_count": len(parity["cases"]),
        },
    }


def validate_pretraining_admission(
    commitment: Mapping[str, Any], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail closed on calibration, privacy, or parity drift before training."""

    evidence = calibration_admission_evidence(manifest)
    configured = commitment.get("configuration", {}).get("calibration_admission")
    if configured != evidence:
        raise RuntimeError("committed calibration admission evidence drifted")
    expected_admission = manifest.get("checkpoint_admission")
    if commitment.get("configuration", {}).get("checkpoint_admission") != expected_admission:
        raise RuntimeError("committed checkpoint admission contract drifted")
    return evidence


def _training_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    lane = manifest["lanes"]["recurrent_temporal"]
    capacity = lane["capacity"]
    thresholds = manifest["checkpoint_schedule"]["recurrent_temporal"][
        "evenly_spaced_cumulative_game_thresholds"
    ]
    if list(thresholds) != list(capacity["cumulative_checkpoint_thresholds"]):
        raise ValueError("checkpoint schedule does not match frozen capacity")
    if int(capacity["exact_game_cap"]) != 26_028 or list(thresholds)[-1] != 26_028:
        raise ValueError("unexpected frozen recurrent game cap")
    if int(capacity["checkpoint_interval_games"]) != 2_169 or len(thresholds) != 12:
        raise ValueError("unexpected frozen recurrent checkpoint schedule")
    return {
        "lane": "recurrent_temporal",
        "model": dict(lane["model"]),
        "reward_only": True,
        "prohibited_lineage": list(manifest["checkpoint_admission"]["prohibited_lineage"]),
        "initialization": dict(lane["initialization"]),
        "collector": {
            "implementation": "collect_league_batch_batched",
            "schedule": ["random"],
            "opponent": "audited_uniform_agent_factory contract via random collector role",
            "max_turns": CALIBRATION_MAX_TURNS,
            "microbatch_games": MICROBATCH_GAMES,
            "stochastic_current_policy_rng": {
                "derivation": "2 * registered training_recurrent_deals member",
                "torch_generator": "CPU torch.Generator.manual_seed",
                "never_policy_input": True,
                "distinct_from_opponent_rng_namespace": "training_recurrent_opponent_rng",
            },
        },
        "optimizer": {
            "class": "torch.optim.Adam",
            "learning_rate": 0.0003,
            "gamma": 0.97,
            "gae_lambda": 0.95,
            "clip_ratio": 0.2,
            "target_kl": 0.02,
            "entropy_coefficient": 0.01,
            "value_coefficient": 0.5,
            "normalize_advantages": True,
            "ppo_epochs": 4,
            "minibatch_trajectories": 16,
            "update_rng_derivation": "(training_recurrent_initialization_seed << 32) | cumulative_games_before_update",
        },
        "capacity": dict(capacity),
        "checkpoint_thresholds": [int(value) for value in thresholds],
        "execution": {
            "device": "cuda only",
            "one_gpu_process_at_a_time": True,
            "torch_compile": False,
            "wall_time_counts": ["rollout", "update", "checkpoint serialization"],
            "development_and_confirmation_excluded_from_wall_time": True,
            "partial_interval_checkpoint_eligible": False,
        },
    }


def _commitment_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    final_audit = dict(manifest["final_audit"])
    ledger = final_audit.get("ledger", {})
    ledger_path = ledger.get("path") if isinstance(ledger, Mapping) else None
    if not isinstance(ledger_path, str) or not ledger_path:
        raise ValueError("frozen final audit ledger path is missing")
    final_audit["result"] = {
        "path": str(Path(ledger_path).with_name("final_audit_result.json")),
        "exclusive_create": True,
        "canonical": True,
    }
    final_audit["paths_must_resolve_exactly"] = True
    return {
        "training": _training_config(manifest),
        # Preserve this frozen contract byte-for-byte in the committed config;
        # checkpoint manifests use the same human-readable gate names.
        "checkpoint_admission": dict(manifest["checkpoint_admission"]),
        "calibration_admission": calibration_admission_evidence(manifest),
        "evaluator": evaluator_mapping(),
        "bootstrap": bootstrap_algorithm(),
        "development": dict(manifest["development_screening"]),
        "confirmation": dict(manifest["confirmation"]),
        "final_audit": final_audit,
    }


def _failed_smoke_reason(smoke: Mapping[str, Any]) -> str:
    """Explain the retained failed smoke without pretending it was accepted."""

    reset = smoke.get("mapping_reset_repeat", {})
    if (
        isinstance(reset, Mapping)
        and reset.get("repeat_evaluation_sha256_equal") is False
        and smoke.get("evaluation_sha256") != smoke.get("repeat_evaluation_sha256")
    ):
        return (
            "pre-fix repeat digest mismatch: the two otherwise identical "
            "evaluations used distinct phase labels that were included in the digest"
        )
    return "retained failed calibration-only smoke; see immutable artifact gates"


def _validate_failed_smoke(smoke_path: Path) -> dict[str, Any]:
    smoke = _read_json(smoke_path)
    if smoke.get("format") != SMOKE_FORMAT or smoke.get("passed") is not False:
        raise RuntimeError("retained failed smoke is missing or not a failed smoke record")
    if smoke.get("candidate_training_started") or smoke.get("candidate_checkpoint_created"):
        raise RuntimeError("retained failed smoke is not calibration-only")
    return smoke


def _validate_successful_recheck(
    smoke_path: Path,
    *,
    manifest_sha256: str,
    registry_sha256: str,
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    smoke = _read_json(smoke_path)
    if smoke.get("format") != SMOKE_FORMAT or not smoke.get("passed"):
        raise RuntimeError("calibration-only runner recheck is absent or failed")
    if smoke.get("candidate_training_started") or smoke.get("candidate_checkpoint_created"):
        raise RuntimeError("runner smoke is not calibration-only")
    if smoke.get("frozen_manifest_sha256") != manifest_sha256:
        raise RuntimeError("runner smoke used a different frozen manifest")
    if smoke.get("registry_sha256") != registry_sha256:
        raise RuntimeError("runner smoke used a different seed registry")
    if smoke.get("source_sha256") != dict(source_sha256):
        raise RuntimeError("runner source changed after calibration recheck")
    gates = smoke.get("gates")
    if not isinstance(gates, Mapping) or not gates or not all(bool(value) for value in gates.values()):
        raise RuntimeError("calibration-only runner recheck lacks passing gates")
    return smoke


def _superseded_v2_binding(path: Path) -> dict[str, Any]:
    """Bind immutable v2 history without accepting it as executable v3."""

    path = Path(path)
    legacy = _read_json(path)
    if legacy.get("format") != LEGACY_V2_COMMITMENT_FORMAT:
        raise RuntimeError("superseded commitment is not the expected v2 artifact")
    projection = dict(legacy)
    legacy_digest = projection.pop("commitment_sha256", None)
    if legacy_digest != _json_sha256(projection):
        raise RuntimeError("superseded v2 commitment digest mismatch")
    if legacy.get("candidate_training", {}).get("started"):
        raise RuntimeError("superseded v2 unexpectedly records candidate training")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "commitment_sha256": legacy_digest,
        "reason": "v2 omitted explicit checkpoint admission parity/privacy gates",
        "executable": False,
    }


def build_implementation_commitment(
    *,
    manifest_path: Path,
    registry_path: Path,
    smoke_path: Path,
    failed_smoke_path: Path | None = None,
    superseded_v2_path: Path | None = None,
) -> dict[str, Any]:
    """Build but do not write the immutable pretraining commitment payload."""

    manifest_path = Path(manifest_path)
    registry_path = Path(registry_path)
    smoke_path = Path(smoke_path)
    failed_smoke_path = (
        smoke_path.with_name("runner_calibration_smoke.json")
        if failed_smoke_path is None
        else Path(failed_smoke_path)
    )
    superseded_v2_path = (
        smoke_path.with_name("implementation_commitment_v2.json")
        if superseded_v2_path is None
        else Path(superseded_v2_path)
    )
    manifest = validate_frozen_campaign(manifest_path)
    manifest_sha256 = file_sha256(manifest_path)
    registry_sha256 = file_sha256(registry_path)
    sources = _source_hashes()
    # The phase-zero dependency subset is already validated by the frozen
    # manifest; ensure the runner only adds, never replaces, that subset.
    if {name: sources[name] for name in SOURCE_FILES} != manifest["source_sha256"]:
        raise RuntimeError("frozen source dependency changed before commitment")
    smoke = _validate_successful_recheck(
        smoke_path,
        manifest_sha256=manifest_sha256,
        registry_sha256=registry_sha256,
        source_sha256=sources,
    )
    failed_smoke = _validate_failed_smoke(failed_smoke_path)
    supersedes = _superseded_v2_binding(superseded_v2_path)
    namespaces = _namespace_snapshot(manifest, registry_path)
    opponent = dict(manifest["audited_uniform_random_opponent"])
    config = _commitment_config(manifest)
    commitment = {
        "format": COMMITMENT_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "status": "pretraining_implementation_committed",
        "created_at_utc": _utc_now(),
        "candidate_training": {
            "started": False,
            "checkpoint_created": False,
            "precommit_only": True,
        },
        "supersedes": supersedes,
        "frozen_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "seed_registry": {
            "path": str(registry_path),
            "sha256": registry_sha256,
            "namespace_records": namespaces,
        },
        "source_sha256": sources,
        "opponent_contract": opponent,
        "opponent_contract_sha256": _json_sha256(opponent),
        "configuration": config,
        "configuration_sha256": _json_sha256(config),
        "calibration_smoke": {
            "successful_recheck": {
                "path": str(smoke_path),
                "sha256": file_sha256(smoke_path),
                "evaluation_sha256": smoke["evaluation_sha256"],
                "gates": dict(smoke["gates"]),
                "passed": True,
                "acceptance_authority": True,
            },
            "retained_failed_smoke": {
                "path": str(failed_smoke_path),
                "sha256": file_sha256(failed_smoke_path),
                "evaluation_sha256": failed_smoke.get("evaluation_sha256"),
                "repeat_evaluation_sha256": failed_smoke.get("repeat_evaluation_sha256"),
                "reason": _failed_smoke_reason(failed_smoke),
                "passed": False,
                "acceptance_authority": False,
            },
        },
        "artifact_protocol": {
            "checkpoint": CHECKPOINT_FORMAT,
            "winner": WINNER_FORMAT,
            "audit_claim": AUDIT_CLAIM_FORMAT,
            "audit_result": AUDIT_RESULT_FORMAT,
            "exclusive_create": True,
            "audit_claim_before_model_load_or_final_seed_iteration": True,
            "audit_paths_must_match_committed_canonical_paths": True,
        },
    }
    commitment["commitment_sha256"] = _json_sha256(commitment)
    return commitment


def validate_implementation_commitment(path: Path) -> dict[str, Any]:
    """Fail closed if any bound source, registry, manifest, or config drifts."""

    path = Path(path)
    commitment = _read_json(path)
    if commitment.get("format") != COMMITMENT_FORMAT:
        raise ValueError("wrong implementation commitment format")
    stored = commitment.get("commitment_sha256")
    projection = dict(commitment)
    projection.pop("commitment_sha256", None)
    if stored != _json_sha256(projection):
        raise RuntimeError("implementation commitment digest mismatch")
    if commitment.get("candidate_training", {}).get("started"):
        raise RuntimeError("pretraining commitment must not claim training started")
    supersedes = commitment.get("supersedes", {})
    superseded_path = Path(supersedes.get("path", ""))
    if not superseded_path.is_file():
        raise RuntimeError("superseded v2 commitment is missing")
    if _superseded_v2_binding(superseded_path) != supersedes:
        raise RuntimeError("superseded v2 commitment binding changed after v3")
    manifest_path = Path(commitment["frozen_manifest"]["path"])
    registry_path = Path(commitment["seed_registry"]["path"])
    if file_sha256(manifest_path) != commitment["frozen_manifest"]["sha256"]:
        raise RuntimeError("frozen manifest changed after commitment")
    if file_sha256(registry_path) != commitment["seed_registry"]["sha256"]:
        raise RuntimeError("seed registry changed after commitment")
    manifest = validate_frozen_campaign(manifest_path)
    sources = _source_hashes()
    if sources != commitment.get("source_sha256"):
        raise RuntimeError("source dependency changed after commitment")
    if {name: sources[name] for name in SOURCE_FILES} != manifest["source_sha256"]:
        raise RuntimeError("phase-zero source binding changed after commitment")
    namespaces = _namespace_snapshot(manifest, registry_path)
    if namespaces != commitment["seed_registry"].get("namespace_records"):
        raise RuntimeError("registered namespace records changed after commitment")
    config = _commitment_config(manifest)
    if config != commitment.get("configuration"):
        raise RuntimeError("committed configuration no longer matches frozen manifest")
    if _json_sha256(config) != commitment.get("configuration_sha256"):
        raise RuntimeError("committed configuration digest mismatch")
    validate_pretraining_admission(commitment, manifest)
    opponent = dict(manifest["audited_uniform_random_opponent"])
    if opponent != commitment.get("opponent_contract"):
        raise RuntimeError("audited opponent contract changed after commitment")
    if _json_sha256(opponent) != commitment.get("opponent_contract_sha256"):
        raise RuntimeError("opponent contract digest mismatch")
    smoke_binding = commitment.get("calibration_smoke", {})
    successful = smoke_binding.get("successful_recheck", {})
    failed = smoke_binding.get("retained_failed_smoke", {})
    successful_path = Path(successful.get("path", ""))
    failed_path = Path(failed.get("path", ""))
    if not successful_path.is_file() or not failed_path.is_file():
        raise RuntimeError("one or both committed calibration smoke artifacts are missing")
    if file_sha256(successful_path) != successful.get("sha256"):
        raise RuntimeError("successful calibration recheck changed after commitment")
    if file_sha256(failed_path) != failed.get("sha256"):
        raise RuntimeError("retained failed smoke changed after commitment")
    _validate_successful_recheck(
        successful_path,
        manifest_sha256=commitment["frozen_manifest"]["sha256"],
        registry_sha256=commitment["seed_registry"]["sha256"],
        source_sha256=sources,
    )
    failed_record = _validate_failed_smoke(failed_path)
    if failed.get("reason") != _failed_smoke_reason(failed_record):
        raise RuntimeError("retained failed smoke reason binding mismatch")
    if successful.get("acceptance_authority") is not True or failed.get(
        "acceptance_authority"
    ) is not False:
        raise RuntimeError("smoke acceptance authority is not fail-closed")
    return commitment


def audit_claim_order() -> list[str]:
    """Publicly testable ordering for a one-use audit execution."""

    return [
        "validate commitment and winner hashes without checkpoint load",
        "verify audit output and claim paths do not exist",
        "exclusive-create one-use claim",
        "load candidate checkpoint",
        "construct final pair iterator and evaluate outcomes",
        "exclusive-create final audit result",
    ]


def canonical_final_audit_paths(
    commitment: Mapping[str, Any],
) -> tuple[Path, Path]:
    """Resolve the sole allowed claim/result targets from the v2 commitment."""

    final_audit = commitment["configuration"]["final_audit"]
    ledger = final_audit.get("ledger", {})
    result = final_audit.get("result", {})
    claim_value = ledger.get("path") if isinstance(ledger, Mapping) else None
    result_value = result.get("path") if isinstance(result, Mapping) else None
    if not isinstance(claim_value, str) or not isinstance(result_value, str):
        raise RuntimeError("v2 commitment lacks canonical final audit paths")
    claim_path = Path(claim_value).resolve()
    result_path = Path(result_value).resolve()
    if claim_path == result_path:
        raise RuntimeError("canonical final audit claim and result paths collide")
    return claim_path, result_path


def assert_canonical_final_audit_paths(
    *,
    commitment: Mapping[str, Any],
    claim_path: Path,
    output_path: Path,
) -> tuple[Path, Path]:
    """Reject alternate audit paths before a claim, model load, or seed iterator."""

    expected_claim, expected_result = canonical_final_audit_paths(commitment)
    received_claim = Path(claim_path).resolve()
    received_result = Path(output_path).resolve()
    if received_claim != expected_claim:
        raise ValueError(
            "audit --claim must resolve exactly to the committed frozen ledger path"
        )
    if received_result != expected_result:
        raise ValueError(
            "audit --output must resolve exactly to the committed final result path"
        )
    return expected_claim, expected_result


def _load_native_policy(*, device: torch.device) -> torch.nn.Module:
    """Load the pinned native policy only in an actual native operation."""

    from temporal_puffer_v4 import NativePufferTemporalPolicy  # pylint: disable=import-outside-toplevel

    policy = NativePufferTemporalPolicy(
        hidden_size=96,
        num_layers=1,
        mode_conditioned_heads=False,
    ).to(device)
    if not bool(getattr(policy, "native_pufferlib", False)):
        raise RuntimeError("native PufferLib policy integration is unavailable")
    return policy


def _assert_frozen_model(policy: torch.nn.Module, commitment: Mapping[str, Any]) -> None:
    model = commitment["configuration"]["training"]["model"]
    checks = {
        "hidden_size": int(model["hidden_size"]),
        "num_layers": int(model["num_layers"]),
        "mode_conditioned_heads": bool(model["mode_conditioned_heads"]),
    }
    for name, expected in checks.items():
        if getattr(policy, name) != expected:
            raise RuntimeError(f"native policy {name} does not match commitment")


def run_calibration_smoke(
    *,
    manifest_path: Path,
    registry_path: Path,
    output_path: Path,
    device_name: str = "cpu",
    pair_count: int = 2,
) -> dict[str, Any]:
    """Exercise only calibration seeds; no candidate/checkpoint is created."""

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"runner smoke output already exists: {output_path}")
    manifest_path = Path(manifest_path)
    registry_path = Path(registry_path)
    manifest = validate_frozen_campaign(manifest_path)
    admission_evidence = calibration_admission_evidence(manifest)
    if pair_count < 1:
        raise ValueError("smoke pair count must be positive")
    deal_bank = _interval_values(_manifest_interval(manifest, "calibration"))
    opponent_bank = _interval_values(
        _manifest_interval(manifest, "calibration_opponent_rng")
    )
    if pair_count > len(deal_bank) or pair_count > len(opponent_bank):
        raise ValueError("requested smoke pairs exceed the calibration-only bank")
    bootstrap_seed = _interval_values(
        _manifest_interval(manifest, "calibration_contract_rng")
    ).start
    initialization_seed = _interval_values(
        _manifest_interval(manifest, "calibration_initialization")
    ).start
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke requested but unavailable")
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    random.seed(initialization_seed)
    torch.use_deterministic_algorithms(True)
    policy = _load_native_policy(device=device)
    _assert_frozen_model(policy, {"configuration": {"training": _training_config(manifest)}})
    deals = tuple(deal_bank[:pair_count])
    opponents = tuple(opponent_bank[:pair_count])
    first = evaluate_seat_balanced_pairs(
        policy,
        deal_seeds=deals,
        opponent_seeds=opponents,
        # The phase name is part of the evidence digest, so repeat the exact
        # same evaluator request rather than merely the same seed pair.
        phase="calibration_runner_smoke",
        require_native_policy=True,
    )
    second = evaluate_seat_balanced_pairs(
        policy,
        deal_seeds=deals,
        opponent_seeds=opponents,
        phase="calibration_runner_smoke",
        require_native_policy=True,
    )
    bootstrap = bootstrap_pair_scores(first["pair_scores"], seed=bootstrap_seed)
    repeated = first["evaluation_sha256"] == second["evaluation_sha256"]
    mapping_ok = all(
        pair["same_deal_and_opponent_rng_each_seat"]
        and pair["candidate_seats"] == [0, 1]
        for pair in first["pairs"]
    )
    calibration_only = all(
        deal_seed in deal_bank and opponent_seed in opponent_bank
        for deal_seed, opponent_seed in zip(deals, opponents)
    ) and bootstrap_seed in _interval_values(
        _manifest_interval(manifest, "calibration_contract_rng")
    )
    gates = {
        "calibration_only": calibration_only,
        "pair_mapping": mapping_ok,
        "legal_actions": first["all_legal_actions_verified"],
        "recurrent_resets": first["all_recurrent_resets_verified"],
        "opponent_resets": first["all_opponent_resets_verified"],
        "exact_repeat_digest": repeated,
        "native_puffer_policy": bool(getattr(policy, "native_pufferlib", False)),
        "claim_before_model_and_final_seed_iteration": (
            audit_claim_order().index("exclusive-create one-use claim")
            < audit_claim_order().index("load candidate checkpoint")
            < audit_claim_order().index("construct final pair iterator and evaluate outcomes")
        ),
        "calibration_artifact_sha": (
            admission_evidence["calibration"]["sha256"] == FROZEN_CALIBRATION_SHA256
        ),
        "privacy_fingerprint": admission_evidence["privacy"]["matches"],
        "fixed_random_parity_fingerprint": admission_evidence[
            "fixed_random_parity"
        ]["evidence_passed"],
    }
    passed = all(gates.values())
    sources = _source_hashes()
    payload = {
        "format": SMOKE_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "created_at_utc": _utc_now(),
        "passed": passed,
        "candidate_training_started": False,
        "candidate_checkpoint_created": False,
        "calibration_only": calibration_only,
        "gates": gates,
        "frozen_manifest_sha256": file_sha256(manifest_path),
        "registry_sha256": file_sha256(registry_path),
        "source_sha256": sources,
        "calibration_admission": admission_evidence,
        "native_policy": {
            "class": type(policy).__name__,
            "device": str(next(policy.parameters()).device),
            "native_pufferlib": bool(getattr(policy, "native_pufferlib", False)),
        },
        "seed_use": {
            "deal_seeds": list(deals),
            "opponent_rng_seeds": list(opponents),
            "bootstrap_seed": bootstrap_seed,
            "initialization_seed": initialization_seed,
            "no_future_campaign_seed_used": True,
        },
        "mapping_reset_repeat": {
            "mapping_verified": mapping_ok,
            "recurrent_resets_verified": first["all_recurrent_resets_verified"],
            "opponent_resets_verified": first["all_opponent_resets_verified"],
            "legal_actions_verified": first["all_legal_actions_verified"],
            "repeat_evaluation_sha256_equal": repeated,
            "claim_order": audit_claim_order(),
            "claim_not_created": True,
        },
        "bootstrap": bootstrap,
        "evaluation_sha256": first["evaluation_sha256"],
        "repeat_evaluation_sha256": second["evaluation_sha256"],
    }
    _write_json_exclusive(output_path, payload)
    return payload


def _projection_before_training(training: Mapping[str, Any]) -> dict[str, Any]:
    capacity = training["capacity"]
    cap = int(capacity["exact_game_cap"])
    wall = float(capacity["wall_cap_seconds"])
    conservative_rate = float(capacity["calibrated_games_per_second"]) * float(
        capacity["conservative_fraction"]
    )
    batch_sizes = []
    completed = 0
    for threshold in training["checkpoint_thresholds"]:
        while completed < int(threshold):
            size = min(MICROBATCH_GAMES, int(threshold) - completed)
            batch_sizes.append(size)
            completed += size
    if completed != cap:
        raise RuntimeError("microbatch plan does not reach exact committed cap")
    projected_seconds = sum(batch_sizes) / conservative_rate
    return {
        "microbatch_sizes": batch_sizes,
        "completed_games": completed,
        "conservative_games_per_second": conservative_rate,
        "projected_training_seconds": projected_seconds,
        "wall_cap_seconds": wall,
        "within_cap": projected_seconds <= wall,
    }


def _checkpoint_lineage(
    *,
    commitment: Mapping[str, Any],
    cumulative_games: int,
    metrics: Mapping[str, Any],
    training_seconds: float,
) -> dict[str, Any]:
    return {
        "random_initialized": True,
        "parent_checkpoint": None,
        "parent_weights": None,
        "ppo_checkpoint": None,
        "imitation_data": None,
        "teacher_logits": None,
        "reward_only": True,
        "cumulative_training_games": int(cumulative_games),
        "training_wall_seconds_excluding_evaluation": training_seconds,
        "metrics": dict(metrics),
        "commitment_sha256": commitment["commitment_sha256"],
        "configuration_sha256": commitment["configuration_sha256"],
        "source_sha256": commitment["source_sha256"],
        "opponent_contract_sha256": commitment["opponent_contract_sha256"],
    }


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def _checkpoint_has_random_no_parent_lineage(lineage: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint explicitly proves fresh random lineage."""

    return bool(
        lineage.get("random_initialized") is True
        and lineage.get("parent_checkpoint") is None
        and lineage.get("parent_weights") is None
        and lineage.get("ppo_checkpoint") is None
        and lineage.get("imitation_data") is None
        and lineage.get("teacher_logits") is None
        and lineage.get("reward_only") is True
    )


def _checkpoint_admission_record(
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    commitment: Mapping[str, Any],
    cumulative_games: int,
    metrics: Mapping[str, Any],
    lineage: Mapping[str, Any],
    all_legal_actions: bool,
    pretraining_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the complete v3 checkpoint gate record before publication."""

    contract = commitment.get("configuration", {}).get("checkpoint_admission", {})
    required = contract.get("required") if isinstance(contract, Mapping) else None
    if not isinstance(required, list) or not required:
        raise RuntimeError("committed checkpoint admission contract is malformed")
    if len(required) != len(set(required)):
        raise RuntimeError("committed checkpoint admission gate names are duplicated")
    calibration = commitment.get("configuration", {}).get("calibration_admission", {})
    if not isinstance(calibration, Mapping):
        raise RuntimeError("committed calibration admission evidence is malformed")
    privacy = calibration.get("privacy", {})
    parity = calibration.get("fixed_random_parity", {})
    observed_privacy = pretraining_evidence.get("privacy", {})
    observed_parity = pretraining_evidence.get("fixed_random_parity", {})
    if not all(
        isinstance(value, Mapping)
        for value in (privacy, parity, observed_privacy, observed_parity)
    ):
        raise RuntimeError("checkpoint admission calibration evidence is malformed")
    expected_thresholds = {
        int(value)
        for value in commitment.get("configuration", {})
        .get("training", {})
        .get("checkpoint_thresholds", [])
    }
    if not expected_thresholds:
        raise RuntimeError("committed checkpoint schedule is malformed")
    model_state = policy.state_dict()
    optimizer_state = optimizer.state_dict()
    finite = _all_finite(metrics) and all(
        bool(torch.isfinite(parameter).all().item())
        for parameter in policy.parameters()
    )
    privacy_matches = (
        observed_privacy.get("matches") is True
        and observed_privacy.get("fingerprint") == privacy.get("fingerprint")
        and observed_privacy.get("fingerprint") == FROZEN_PRIVACY_FINGERPRINT
    )
    parity_matches = (
        observed_parity.get("evidence_passed") is True
        and observed_parity.get("schema") == parity.get("schema")
        and observed_parity.get("canonicalization") == parity.get("canonicalization")
        and observed_parity.get("sha256") == parity.get("sha256")
    )
    gates = {
        "complete checkpoint": bool(model_state),
        "complete optimizer state": bool(optimizer_state),
        "complete per-checkpoint manifest": True,
        "random-initialized no-parent lineage": _checkpoint_has_random_no_parent_lineage(
            lineage
        ),
        "finite parameters and metrics": finite,
        "100 percent legal actions": bool(all_legal_actions),
        "unchanged privacy fingerprint": privacy_matches,
        "unchanged fixed-random parity fingerprint": parity_matches,
        "scheduled cumulative threshold reached": int(cumulative_games)
        in expected_thresholds,
    }
    if set(gates) != set(required):
        raise RuntimeError("checkpoint admission record does not cover frozen contract")
    return {
        "required": list(required),
        "gates": gates,
        "eligible": all(gates.values()),
        "calibration_admission": {
            "privacy_fingerprint": observed_privacy.get("fingerprint"),
            "fixed_random_parity": {
                "schema": observed_parity.get("schema"),
                "canonicalization": observed_parity.get("canonicalization"),
                "sha256": observed_parity.get("sha256"),
                "evidence_passed": observed_parity.get("evidence_passed"),
            },
        },
    }


def _assert_checkpoint_admission(
    checkpoint_manifest: Mapping[str, Any], commitment: Mapping[str, Any]
) -> None:
    """Refuse publication or screening unless every v3 gate remains true."""

    contract = commitment.get("configuration", {}).get("checkpoint_admission", {})
    required = contract.get("required") if isinstance(contract, Mapping) else None
    admission = checkpoint_manifest.get("admission")
    if not isinstance(required, list) or not isinstance(admission, Mapping):
        raise RuntimeError("checkpoint admission manifest is malformed")
    gates = admission.get("gates")
    if admission.get("required") != required or not isinstance(gates, Mapping):
        raise RuntimeError("checkpoint admission contract drifted")
    if set(gates) != set(required):
        raise RuntimeError("checkpoint admission gates do not match frozen contract")
    if admission.get("eligible") is not True or any(
        gates.get(name) is not True for name in required
    ):
        raise RuntimeError("checkpoint is ineligible for publication or screening")
    if checkpoint_manifest.get("complete_model_state") is not True or checkpoint_manifest.get(
        "complete_optimizer_state"
    ) is not True:
        raise RuntimeError("checkpoint manifest does not prove complete model and optimizer")
    lineage = checkpoint_manifest.get("lineage")
    if not isinstance(lineage, Mapping) or not _checkpoint_has_random_no_parent_lineage(
        lineage
    ):
        raise RuntimeError("checkpoint lineage is not fresh random initialization")
    thresholds = {
        int(value)
        for value in commitment.get("configuration", {})
        .get("training", {})
        .get("checkpoint_thresholds", [])
    }
    if int(checkpoint_manifest.get("cumulative_training_games", -1)) not in thresholds:
        raise RuntimeError("checkpoint cumulative games is not a scheduled threshold")
    calibration = commitment.get("configuration", {}).get("calibration_admission", {})
    expected_privacy = calibration.get("privacy", {}) if isinstance(calibration, Mapping) else {}
    expected_parity = (
        calibration.get("fixed_random_parity", {}) if isinstance(calibration, Mapping) else {}
    )
    recorded = admission.get("calibration_admission")
    if not isinstance(recorded, Mapping):
        raise RuntimeError("checkpoint lacks calibration admission fingerprint")
    recorded_parity = recorded.get("fixed_random_parity")
    if (
        recorded.get("privacy_fingerprint") != expected_privacy.get("fingerprint")
        or not isinstance(recorded_parity, Mapping)
        or recorded_parity.get("schema") != expected_parity.get("schema")
        or recorded_parity.get("canonicalization")
        != expected_parity.get("canonicalization")
        or recorded_parity.get("sha256") != expected_parity.get("sha256")
        or recorded_parity.get("evidence_passed") is not True
    ):
        raise RuntimeError("checkpoint admission privacy/parity fingerprint drifted")


def _stage_checkpoint(
    *,
    output_dir: Path,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    commitment: Mapping[str, Any],
    cumulative_games: int,
    metrics: Mapping[str, Any],
    training_seconds: float,
    admission: Mapping[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Serialize model+optimizer and complete manifest before publication."""

    checkpoint_path = output_dir / f"checkpoint_{cumulative_games:05d}.pt"
    manifest_path = output_dir / f"checkpoint_{cumulative_games:05d}.json"
    pending_checkpoint = output_dir / f".checkpoint_{cumulative_games:05d}.pt.pending"
    pending_manifest = output_dir / f".checkpoint_{cumulative_games:05d}.json.pending"
    if any(path.exists() for path in (checkpoint_path, manifest_path, pending_checkpoint, pending_manifest)):
        raise FileExistsError("checkpoint publication path already exists")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "model_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "lineage": _checkpoint_lineage(
            commitment=commitment,
            cumulative_games=cumulative_games,
            metrics=metrics,
            training_seconds=training_seconds,
        ),
        "configuration": commitment["configuration"]["training"],
    }
    torch.save(payload, pending_checkpoint)
    _fsync_file(pending_checkpoint)
    checkpoint_sha256 = file_sha256(pending_checkpoint)
    checkpoint_manifest = {
        "format": CHECKPOINT_FORMAT,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "cumulative_training_games": int(cumulative_games),
        "complete_model_state": True,
        "complete_optimizer_state": True,
        "lineage": payload["lineage"],
        "configuration_sha256": commitment["configuration_sha256"],
        "source_sha256": commitment["source_sha256"],
        "commitment_sha256": commitment["commitment_sha256"],
        "opponent_contract_sha256": commitment["opponent_contract_sha256"],
        "admission": dict(admission),
    }
    _write_json_exclusive(pending_manifest, checkpoint_manifest)
    _assert_checkpoint_admission(_read_json(pending_manifest), commitment)
    return pending_checkpoint, pending_manifest, checkpoint_manifest


def _publish_staged_checkpoint(
    pending_checkpoint: Path,
    pending_manifest: Path,
    checkpoint_manifest: Mapping[str, Any],
    *,
    commitment: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[Path, Path]:
    # Re-read current calibration evidence immediately before the irreversible
    # rename.  A privacy/parity drift therefore cannot publish or screen a
    # checkpoint, even if it occurred after staging.
    validate_pretraining_admission(commitment, manifest)
    staged_manifest = _read_json(pending_manifest)
    if staged_manifest != dict(checkpoint_manifest):
        raise RuntimeError("pending checkpoint manifest changed before publication")
    _assert_checkpoint_admission(staged_manifest, commitment)
    if file_sha256(pending_checkpoint) != checkpoint_manifest.get("checkpoint_sha256"):
        raise RuntimeError("pending checkpoint digest changed before publication")
    checkpoint_path = Path(checkpoint_manifest["checkpoint_path"])
    manifest_path = checkpoint_path.with_suffix(".json")
    if checkpoint_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to replace an existing checkpoint")
    os.replace(pending_checkpoint, checkpoint_path)
    os.replace(pending_manifest, manifest_path)
    return checkpoint_path, manifest_path


def _load_checkpoint_policy(
    *,
    checkpoint_path: Path,
    commitment: Mapping[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    policy = _load_native_policy(device=device)
    _assert_frozen_model(policy, commitment)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("candidate checkpoint has wrong format")
    if payload.get("lineage", {}).get("commitment_sha256") != commitment["commitment_sha256"]:
        raise RuntimeError("candidate checkpoint commitment binding mismatch")
    policy.load_state_dict(payload["model_state_dict"])
    return policy


def _development_bank(manifest: Mapping[str, Any], checkpoint_index: int) -> tuple[range, range]:
    name = f"recurrent_temporal_checkpoint_{checkpoint_index:02d}"
    bank = manifest["development_screening"]["banks"].get(name)
    if not isinstance(bank, Mapping):
        raise RuntimeError("missing committed checkpoint development bank")
    return (
        _interval_values(bank["deal_subrange"]),
        _interval_values(bank["opponent_rng_subrange"]),
    )


def _evaluate_checkpoint_development(
    *,
    checkpoint_path: Path,
    checkpoint_manifest: Mapping[str, Any],
    commitment: Mapping[str, Any],
    manifest: Mapping[str, Any],
    checkpoint_index: int,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    # Do this before resolving even the development bank.  An ineligible
    # checkpoint must not consume a screening opportunity.
    _assert_checkpoint_admission(checkpoint_manifest, commitment)
    if checkpoint_manifest.get("checkpoint_path") != str(checkpoint_path):
        raise RuntimeError("checkpoint manifest path does not match development input")
    stored_manifest = _read_json(checkpoint_path.with_suffix(".json"))
    if stored_manifest != dict(checkpoint_manifest):
        raise RuntimeError("checkpoint manifest changed before development screening")
    _assert_checkpoint_admission(stored_manifest, commitment)
    deals, opponents = _development_bank(manifest, checkpoint_index)
    policy = _load_checkpoint_policy(
        checkpoint_path=checkpoint_path,
        commitment=commitment,
        device=device,
    )
    evaluation = evaluate_seat_balanced_pairs(
        policy,
        deal_seeds=deals,
        opponent_seeds=opponents,
        phase=f"development_screening_recurrent_{checkpoint_index:02d}",
        require_native_policy=True,
    )
    seed = int(manifest["development_screening"]["bootstrap"]["seed"])
    result = {
        "checkpoint_index": checkpoint_index,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_manifest_path": str(checkpoint_path.with_suffix(".json")),
        "checkpoint_sha256": checkpoint_manifest["checkpoint_sha256"],
        "training_games": checkpoint_manifest["cumulative_training_games"],
        "admission": dict(checkpoint_manifest["admission"]),
        "evaluation": evaluation,
        "bootstrap": bootstrap_pair_scores(evaluation["pair_scores"], seed=seed),
    }
    _write_json_exclusive(
        output_dir / f"development_screen_{checkpoint_index:02d}.json", result
    )
    return result


def _rank_development(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(result) for result in results),
        key=lambda result: (
            -float(result["bootstrap"]["ci95"]["lower"]),
            -float(result["bootstrap"]["point_estimate"]),
            int(result["training_games"]),
            str(result["checkpoint_sha256"]),
        ),
    )


def _confirm_finalists(
    *,
    finalists: Sequence[Mapping[str, Any]],
    commitment: Mapping[str, Any],
    manifest: Mapping[str, Any],
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, Any]]:
    # Validate every finalist before constructing the fresh confirmation
    # iterator.  Confirmation inherits the exact admission gate from the
    # checkpoint manifest rather than trusting development result fields.
    admitted_finalists: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for finalist in finalists:
        checkpoint_path = Path(finalist["checkpoint_path"])
        manifest_value = finalist.get("checkpoint_manifest_path")
        checkpoint_manifest_path = (
            Path(manifest_value)
            if isinstance(manifest_value, str)
            else checkpoint_path.with_suffix(".json")
        )
        checkpoint_manifest = _read_json(checkpoint_manifest_path)
        if checkpoint_manifest.get("checkpoint_path") != str(checkpoint_path):
            raise RuntimeError("confirmation checkpoint manifest path drifted")
        if checkpoint_manifest.get("checkpoint_sha256") != finalist.get(
            "checkpoint_sha256"
        ):
            raise RuntimeError("confirmation checkpoint manifest digest drifted")
        _assert_checkpoint_admission(checkpoint_manifest, commitment)
        admitted_finalists.append(
            (dict(finalist), checkpoint_manifest, checkpoint_manifest_path)
        )
    bank = manifest["confirmation"]["common_fresh_bank"]
    deals = _interval_values(bank["deal_range"])
    opponents = _interval_values(bank["opponent_rng_range"])
    bootstrap_seed = int(manifest["confirmation"]["bootstrap"]["seed"])
    results = []
    for finalist_index, (finalist, checkpoint_manifest, checkpoint_manifest_path) in enumerate(
        admitted_finalists, start=1
    ):
        policy = _load_checkpoint_policy(
            checkpoint_path=Path(finalist["checkpoint_path"]),
            commitment=commitment,
            device=device,
        )
        evaluation = evaluate_seat_balanced_pairs(
            policy,
            deal_seeds=deals,
            opponent_seeds=opponents,
            phase=f"confirmation_finalist_{finalist_index:02d}",
            require_native_policy=True,
        )
        result = {
            "finalist_index": finalist_index,
            "checkpoint_path": finalist["checkpoint_path"],
            "checkpoint_sha256": finalist["checkpoint_sha256"],
            "training_games": finalist["training_games"],
            "checkpoint_manifest_path": str(checkpoint_manifest_path),
            "admission": dict(checkpoint_manifest["admission"]),
            "evaluation": evaluation,
            "bootstrap": bootstrap_pair_scores(
                evaluation["pair_scores"], seed=bootstrap_seed
            ),
        }
        _write_json_exclusive(
            output_dir / f"confirmation_finalist_{finalist_index:02d}.json", result
        )
        results.append(result)
    return results


def _rank_confirmation(results: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(result) for result in results),
        key=lambda result: (
            -float(result["bootstrap"]["point_estimate"]),
            -float(result["bootstrap"]["ci95"]["lower"]),
            int(result["training_games"]),
            str(result["checkpoint_sha256"]),
        ),
    )


def train_campaign(
    *,
    commitment_path: Path,
    output_dir: Path,
    device_name: str = "cuda",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run exactly the committed recurrent lane, or perform a no-write preflight."""

    commitment = validate_implementation_commitment(commitment_path)
    manifest_path = Path(commitment["frozen_manifest"]["path"])
    manifest = validate_frozen_campaign(manifest_path)
    pretraining_evidence = validate_pretraining_admission(commitment, manifest)
    training = commitment["configuration"]["training"]
    projection = _projection_before_training(training)
    if not projection["within_cap"]:
        raise RuntimeError(
            "conservative completed-microbatch projection exceeds frozen 45 minute cap"
        )
    output_dir = Path(output_dir)
    preflight = {
        "commitment": str(commitment_path),
        "output_dir": str(output_dir),
        "candidate_training_started": False,
        "candidate_checkpoint_created": False,
        "dry_run": bool(dry_run),
        "projection": projection,
        "calibration_admission": pretraining_evidence,
        "final_audit_seed_bank_touched": False,
    }
    if dry_run:
        return preflight
    if output_dir.exists():
        raise FileExistsError(f"training output directory already exists: {output_dir}")
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("committed training requires one CUDA native-Puffer process")

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json_exclusive(
        output_dir / "training_started.json",
        {
            **preflight,
            "candidate_training_started": True,
            "started_at_utc": _utc_now(),
            "device": str(device),
            "torch_compile_used": False,
        },
    )
    torch.use_deterministic_algorithms(True)
    initialization_seed = int(training["initialization"]["seed"])
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    random.seed(initialization_seed)
    policy = _load_native_policy(device=device)
    _assert_frozen_model(policy, commitment)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.0003)
    from temporal_batched_speed import collect_league_batch_batched  # pylint: disable=import-outside-toplevel
    from temporal_training import ppo_update_batched_epochs  # pylint: disable=import-outside-toplevel

    deal_range = _interval_values(_manifest_interval(manifest, "training_recurrent_deals"))
    opponent_range = _interval_values(
        _manifest_interval(manifest, "training_recurrent_opponent_rng")
    )
    thresholds = [int(value) for value in training["checkpoint_thresholds"]]
    if len(deal_range) != thresholds[-1] or len(opponent_range) != thresholds[-1]:
        raise RuntimeError("training seed ranges do not match committed game cap")

    training_seconds = 0.0
    completed = 0
    checkpoint_index = 0
    development_results: list[dict[str, Any]] = []
    all_legal_actions = True
    stop_reason = "exact_game_cap_reached"
    while completed < thresholds[-1]:
        threshold = thresholds[checkpoint_index]
        while completed < threshold:
            if training_seconds >= float(training["capacity"]["wall_cap_seconds"]):
                stop_reason = "wall_cap_before_complete_interval"
                break
            batch_size = min(MICROBATCH_GAMES, threshold - completed)
            batch_deals = deal_range[completed : completed + batch_size]
            batch_opponents = opponent_range[completed : completed + batch_size]
            if len(batch_deals) != batch_size or len(batch_opponents) != batch_size:
                raise RuntimeError("training seed slicing drifted")
            policy.eval()
            started = time.perf_counter()
            episodes, contribution = collect_league_batch_batched(
                policy,
                None,
                (),
                games=batch_size,
                seed=batch_deals.start,
                schedule=("random",),
                random_opponent_seeds=tuple(batch_opponents),
                max_turns=CALIBRATION_MAX_TURNS,
            )
            rollout_seconds = time.perf_counter() - started
            legal_actions = all(
                step.decision.legal_actions[step.action]
                for trajectory, _ in episodes
                for step in trajectory
            )
            if not legal_actions:
                raise RuntimeError("training collector produced an illegal action")
            all_legal_actions = all_legal_actions and legal_actions
            policy.train()
            started = time.perf_counter()
            metrics = ppo_update_batched_epochs(
                policy,
                episodes,
                optimizer,
                update_epochs=4,
                minibatch_trajectories=16,
                seed=update_rng_seed(initialization_seed, completed),
                target_kl=0.02,
                clip_ratio=0.2,
                value_coefficient=0.5,
                entropy_coefficient=0.01,
                gamma=0.97,
                gae_lambda=0.95,
                normalize_advantages=True,
            )
            update_seconds = time.perf_counter() - started
            if not _all_finite(metrics) or not all(
                bool(torch.isfinite(parameter).all().item())
                for parameter in policy.parameters()
            ):
                raise RuntimeError("non-finite training parameters or metrics")
            completed += batch_size
            training_seconds += rollout_seconds + update_seconds
            _write_json_exclusive(
                output_dir / f"microbatch_{completed:05d}.json",
                {
                    "cumulative_games": completed,
                    "games": batch_size,
                    "deal_seed_range": {
                        "start_inclusive": batch_deals.start,
                        "end_exclusive": batch_deals.stop,
                    },
                    "opponent_rng_range": {
                        "start_inclusive": batch_opponents.start,
                        "end_exclusive": batch_opponents.stop,
                    },
                    "current_policy_rng_derivation": "2 * deal_seed",
                    "update_rng_seed": update_rng_seed(initialization_seed, completed - batch_size),
                    "contribution": contribution,
                    "rollout_seconds": rollout_seconds,
                    "update_seconds": update_seconds,
                    "training_wall_seconds_excluding_evaluation": training_seconds,
                    "metrics": metrics,
                    "legal_actions_verified": legal_actions,
                },
            )
            if training_seconds >= float(training["capacity"]["wall_cap_seconds"]):
                stop_reason = "wall_cap_after_microbatch"
                break
        if completed != threshold:
            break

        # Revalidate the whole frozen calibration artifact before every
        # scheduled checkpoint.  This is intentionally before staging so a
        # privacy/parity mismatch cannot create, publish, or screen it.
        pretraining_evidence = validate_pretraining_admission(commitment, manifest)
        checkpoint_started = time.perf_counter()
        lineage = _checkpoint_lineage(
            commitment=commitment,
            cumulative_games=completed,
            metrics=metrics,
            training_seconds=training_seconds,
        )
        admission = _checkpoint_admission_record(
            policy=policy,
            optimizer=optimizer,
            commitment=commitment,
            cumulative_games=completed,
            metrics=metrics,
            lineage=lineage,
            all_legal_actions=all_legal_actions,
            pretraining_evidence=pretraining_evidence,
        )
        if admission["eligible"] is not True:
            raise RuntimeError("checkpoint admission gates failed before staging")
        pending_checkpoint, pending_manifest, checkpoint_manifest = _stage_checkpoint(
            output_dir=output_dir,
            policy=policy,
            optimizer=optimizer,
            commitment=commitment,
            cumulative_games=completed,
            metrics=metrics,
            training_seconds=training_seconds,
            admission=admission,
        )
        training_seconds += time.perf_counter() - checkpoint_started
        if training_seconds > float(training["capacity"]["wall_cap_seconds"]):
            stop_reason = "wall_cap_during_checkpoint_serialization"
            _write_json_exclusive(
                output_dir / f"checkpoint_{completed:05d}_ineligible.json",
                {
                    "reason": stop_reason,
                    "cumulative_games": completed,
                    "pending_checkpoint": str(pending_checkpoint),
                    "pending_manifest": str(pending_manifest),
                    "training_wall_seconds_excluding_evaluation": training_seconds,
                    "eligible": False,
                },
            )
            break
        checkpoint_path, _ = _publish_staged_checkpoint(
            pending_checkpoint,
            pending_manifest,
            checkpoint_manifest,
            commitment=commitment,
            manifest=manifest,
        )
        checkpoint_index += 1
        # Development time is explicitly not part of the frozen 45 minute
        # training cap.  It uses precisely one bank for this scheduled point.
        development_results.append(
            _evaluate_checkpoint_development(
                checkpoint_path=checkpoint_path,
                checkpoint_manifest=checkpoint_manifest,
                commitment=commitment,
                manifest=manifest,
                checkpoint_index=checkpoint_index,
                device=device,
                output_dir=output_dir,
            )
        )
        if training_seconds >= float(training["capacity"]["wall_cap_seconds"]):
            stop_reason = "wall_cap_after_complete_checkpoint"
            break

    ranked_development = _rank_development(development_results)
    finalist_limit = int(
        manifest["development_screening"]["finalist_admission"]["recurrent_temporal"]
    )
    finalists = ranked_development[:finalist_limit]
    confirmations = _confirm_finalists(
        finalists=finalists,
        commitment=commitment,
        manifest=manifest,
        device=device,
        output_dir=output_dir,
    ) if finalists else []
    ranked_confirmation = _rank_confirmation(confirmations)
    winner_path: str | None = None
    if ranked_confirmation:
        winner = ranked_confirmation[0]
        winner_payload = {
            "format": WINNER_FORMAT,
            "campaign_id": CAMPAIGN_ID,
            "created_at_utc": _utc_now(),
            "commitment_sha256": commitment["commitment_sha256"],
            "configuration_sha256": commitment["configuration_sha256"],
            "source_sha256": commitment["source_sha256"],
            "opponent_contract_sha256": commitment["opponent_contract_sha256"],
            "candidate_checkpoint_path": winner["checkpoint_path"],
            "candidate_checkpoint_sha256": winner["checkpoint_sha256"],
            "training_games": winner["training_games"],
            "confirmation": winner,
            "final_audit_not_run": True,
            "final_audit_seed_bank_touched": False,
        }
        frozen_winner_path = output_dir / "frozen_winner.json"
        _write_json_exclusive(frozen_winner_path, winner_payload)
        winner_path = str(frozen_winner_path)
    result = {
        "commitment": str(commitment_path),
        "output_dir": str(output_dir),
        "candidate_training_started": True,
        "candidate_checkpoint_created": bool(development_results),
        "completed_games": completed,
        "complete_checkpoints": len(development_results),
        "training_wall_seconds_excluding_evaluation": training_seconds,
        "stop_reason": stop_reason,
        "finalists": len(finalists),
        "winner": winner_path,
        "final_audit_seed_bank_touched": False,
    }
    _write_json_exclusive(output_dir / "training_result.json", result)
    return result


def _validate_winner_without_checkpoint_load(
    *,
    commitment: Mapping[str, Any],
    winner_path: Path,
) -> tuple[dict[str, Any], Path]:
    winner = _read_json(winner_path)
    if winner.get("format") != WINNER_FORMAT:
        raise ValueError("wrong frozen winner format")
    for name in (
        "commitment_sha256",
        "configuration_sha256",
        "source_sha256",
        "opponent_contract_sha256",
    ):
        if winner.get(name) != commitment.get(name):
            raise RuntimeError(f"winner binding mismatch: {name}")
    if winner.get("final_audit_not_run") is not True or winner.get(
        "final_audit_seed_bank_touched"
    ):
        raise RuntimeError("winner artifact already touches final audit state")
    checkpoint_path = Path(winner["candidate_checkpoint_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError("winner candidate checkpoint is missing")
    if file_sha256(checkpoint_path) != winner.get("candidate_checkpoint_sha256"):
        raise RuntimeError("winner candidate checkpoint digest mismatch")
    return winner, checkpoint_path


def audit_winner(
    *,
    commitment_path: Path,
    winner_path: Path,
    output_path: Path,
    claim_path: Path,
    device_name: str = "cuda",
) -> dict[str, Any]:
    """Burn a one-use final audit before model load or final-seed iteration."""

    commitment = validate_implementation_commitment(commitment_path)
    canonical_claim_path, canonical_output_path = assert_canonical_final_audit_paths(
        commitment=commitment,
        claim_path=Path(claim_path),
        output_path=Path(output_path),
    )
    winner, checkpoint_path = _validate_winner_without_checkpoint_load(
        commitment=commitment,
        winner_path=Path(winner_path),
    )
    output_path = canonical_output_path
    claim_path = canonical_claim_path
    if output_path.exists() or claim_path.exists():
        raise FileExistsError("final audit output or one-use claim already exists")
    # Do not construct a range/list of final seeds before this durable claim.
    claim = {
        "format": AUDIT_CLAIM_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "claimed_at_utc": _utc_now(),
        "claim_burns_audit_even_on_failure": True,
        "execution_order": audit_claim_order(),
        "pre_model_load_validation_complete": True,
        "commitment_sha256": commitment["commitment_sha256"],
        "configuration_sha256": commitment["configuration_sha256"],
        "source_sha256": commitment["source_sha256"],
        "opponent_contract_sha256": commitment["opponent_contract_sha256"],
        "candidate_checkpoint_sha256": winner["candidate_checkpoint_sha256"],
        "final_audit_ranges": commitment["configuration"]["final_audit"]["bank"],
        "bootstrap_seed": commitment["configuration"]["final_audit"]["bootstrap"]["seed"],
    }
    _write_json_exclusive(claim_path, claim)

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("final audit requires the native CUDA Puffer path")
    policy = _load_checkpoint_policy(
        checkpoint_path=checkpoint_path,
        commitment=commitment,
        device=device,
    )
    audit_bank = commitment["configuration"]["final_audit"]["bank"]
    # The final pair iterators and outcomes are intentionally created only here.
    evaluation = evaluate_seat_balanced_pairs(
        policy,
        deal_seeds=_interval_values(audit_bank["deal_range"]),
        opponent_seeds=_interval_values(audit_bank["opponent_rng_range"]),
        phase="final_audit_one_use",
        require_native_policy=True,
    )
    bootstrap = bootstrap_pair_scores(
        evaluation["pair_scores"],
        seed=int(commitment["configuration"]["final_audit"]["bootstrap"]["seed"]),
    )
    gate = commitment["configuration"]["final_audit"]["gate"]
    passed = bool(
        bootstrap["point_estimate"] >= float(gate["point_estimate_gte"])
        and bootstrap["ci95"]["lower"] >= float(gate["ci95_lower_bound_gte"])
    )
    result = {
        "format": AUDIT_RESULT_FORMAT,
        "campaign_id": CAMPAIGN_ID,
        "completed_at_utc": _utc_now(),
        "claim_path": str(claim_path),
        "claim_sha256": file_sha256(claim_path),
        "commitment_sha256": commitment["commitment_sha256"],
        "candidate_checkpoint_sha256": winner["candidate_checkpoint_sha256"],
        "evaluation": evaluation,
        "bootstrap": bootstrap,
        "gate": gate,
        "passed": passed,
    }
    _write_json_exclusive(output_path, result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--manifest", type=Path, required=True)
    smoke.add_argument("--registry", type=Path, required=True)
    smoke.add_argument("--output", type=Path, required=True)
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke.add_argument("--pairs", type=int, default=2)

    precommit = subparsers.add_parser("precommit")
    precommit.add_argument("--manifest", type=Path, required=True)
    precommit.add_argument("--registry", type=Path, required=True)
    precommit.add_argument("--output", type=Path, required=True)
    precommit.add_argument("--smoke", type=Path)
    precommit.add_argument("--failed-smoke", type=Path)
    precommit.add_argument("--supersedes", type=Path)

    train = subparsers.add_parser("train")
    train.add_argument("--commitment", type=Path, required=True)
    train.add_argument(
        "--output-dir", type=Path, default=Path("live_training") / CAMPAIGN_ID
    )
    train.add_argument("--device", choices=("cuda",), default="cuda")
    train.add_argument("--dry-run", action="store_true")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--commitment", type=Path, required=True)
    audit.add_argument("--winner", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--claim", type=Path, required=True)
    audit.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "smoke":
        result = run_calibration_smoke(
            manifest_path=args.manifest,
            registry_path=args.registry,
            output_path=args.output,
            device_name=args.device,
            pair_count=args.pairs,
        )
    elif args.command == "precommit":
        smoke_path = args.smoke or args.manifest.with_name(
            "runner_calibration_smoke_recheck_v3.json"
        )
        failed_smoke_path = args.failed_smoke or args.manifest.with_name(
            "runner_calibration_smoke.json"
        )
        superseded_v2_path = args.supersedes or args.manifest.with_name(
            "implementation_commitment_v2.json"
        )
        if args.output.exists():
            raise FileExistsError(f"implementation commitment already exists: {args.output}")
        result = build_implementation_commitment(
            manifest_path=args.manifest,
            registry_path=args.registry,
            smoke_path=smoke_path,
            failed_smoke_path=failed_smoke_path,
            superseded_v2_path=superseded_v2_path,
        )
        _write_json_exclusive(args.output, result)
    elif args.command == "train":
        result = train_campaign(
            commitment_path=args.commitment,
            output_dir=args.output_dir,
            device_name=args.device,
            dry_run=args.dry_run,
        )
    else:
        result = audit_winner(
            commitment_path=args.commitment,
            winner_path=args.winner,
            output_path=args.output,
            claim_path=args.claim,
            device_name=args.device,
        )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
