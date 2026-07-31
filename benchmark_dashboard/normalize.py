"""Normalize legacy and schema-v1 Pesten benchmark JSON files.

The normalizer deliberately uses only the Python standard library so dashboard
generation does not add a project dependency. Source-provided rates and
confidence intervals are reconciled before rows are accepted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
DATE_IN_NAME = re.compile(r"(20\d{6})")
RUN_KINDS = {"training", "tournament", "microbenchmark", "evaluation"}
RUN_STATUSES = {"complete", "incomplete", "failed"}


class BenchmarkInputError(ValueError):
    """Raised when a benchmark file is malformed or unsupported."""


class MetricReconciliationError(BenchmarkInputError):
    """Raised when source totals and derived metrics materially disagree."""


class UnsupportedBenchmarkShapeError(BenchmarkInputError):
    """Raised only when valid JSON is outside every dashboard input schema."""


@dataclass(frozen=True)
class NormalizationResult:
    runs: list[dict[str, Any]]
    source_files: list[dict[str, Any]]
    warnings: list[str]


def _number(value: Any, path: str, *, allow_zero: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchmarkInputError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BenchmarkInputError(f"{path} must be finite")
    if result < 0 or (not allow_zero and result == 0):
        comparator = "positive" if not allow_zero else "non-negative"
        raise BenchmarkInputError(f"{path} must be {comparator}")
    return result


def _integer(value: Any, path: str, *, allow_zero: bool = True) -> int:
    number = _number(value, path, allow_zero=allow_zero)
    if not number.is_integer():
        raise BenchmarkInputError(f"{path} must be an integer")
    return int(number)


def _close(actual: float, expected: float, *, rtol: float = 0.005, atol: float = 1e-9) -> bool:
    return abs(actual - expected) <= max(atol, rtol * max(abs(actual), abs(expected)))


def _reconcile_rate(
    *,
    count: float,
    seconds: float,
    stated: Any,
    path: str,
) -> float:
    if seconds <= 0:
        raise BenchmarkInputError(f"{path} denominator seconds must be positive")
    computed = count / seconds
    if stated is not None:
        stated_number = _number(stated, path)
        if not _close(stated_number, computed):
            raise MetricReconciliationError(
                f"{path}={stated_number:.9g} does not reconcile with "
                f"count/seconds={computed:.9g}"
            )
    return computed


def wilson_interval(wins: int, games: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson score interval for a binomial rate."""
    if games <= 0:
        raise BenchmarkInputError("Wilson interval requires games > 0")
    p = wins / games
    z2 = z * z
    denominator = 1 + z2 / games
    center = (p + z2 / (2 * games)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z2 / (4 * games)) / games) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _source_date(path: Path, payload: dict[str, Any]) -> str:
    recorded = payload.get("recorded_at")
    if isinstance(recorded, str):
        try:
            return date.fromisoformat(recorded[:10]).isoformat()
        except ValueError as exc:
            raise BenchmarkInputError(f"{path}: recorded_at is not an ISO date") from exc
    bundled_runs = payload.get("runs")
    if isinstance(bundled_runs, list):
        bundled_dates = [
            str(run.get("recorded_at"))[:10]
            for run in bundled_runs
            if isinstance(run, dict) and run.get("recorded_at") is not None
        ]
        if bundled_dates:
            try:
                return max(date.fromisoformat(value).isoformat() for value in bundled_dates)
            except ValueError as exc:
                raise BenchmarkInputError(
                    f"{path}: bundled run recorded_at is not an ISO date"
                ) from exc
    match = DATE_IN_NAME.search(path.name)
    if match:
        raw = match.group(1)
        return date(int(raw[:4]), int(raw[4:6]), int(raw[6:8])).isoformat()
    raise BenchmarkInputError(
        f"{path}: missing recorded_at and no YYYYMMDD date in file name"
    )


def _relative_source(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_inputs(input_dir: Path) -> list[Path]:
    """Discover top-level benchmark JSON files, excluding generated artifacts."""
    return sorted(
        path
        for path in input_dir.glob("*.json")
        if path.is_file() and path.name not in {"artifact.json", "normalized_runs.json"}
    )


def _source_record(path: Path, root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": _relative_source(path, root),
        "sha256": _sha256(path),
        "recorded_at": _source_date(path, payload),
    }


def _base_run(
    *,
    run_id: str,
    run_kind: str,
    status: str,
    recorded_at: str,
    source: dict[str, Any],
    player_count: int | None,
    seed: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_kind": run_kind,
        "status": status,
        "recorded_at": recorded_at,
        "game": {
            "player_count": player_count,
            "observation_version": None,
            "rules_version": None,
        },
        "models": [],
        "performance": {},
        "outcome": None,
        "timing": [],
        "resources": {
            "cpu_utilization_pct": None,
            "gpu_utilization_pct": None,
            "peak_cpu_memory_mib": None,
            "peak_gpu_memory_mib": None,
        },
        "guardrails": [],
        "learning_curve": [],
        "provenance": {
            "source_path": source["path"],
            "source_sha256": source["sha256"],
            "git_sha": None,
            "seed": seed,
            "hardware": None,
            "recipe": None,
            "checkpoint_sha256": None,
        },
        "notes": [],
    }


def _performance_fields(
    metrics: dict[str, Any],
    path: str,
    *,
    games_key: str = "episodes",
    decisions_key: str = "samples",
) -> dict[str, Any]:
    games = _integer(metrics[games_key], f"{path}.{games_key}", allow_zero=False)
    decisions = _integer(metrics[decisions_key], f"{path}.{decisions_key}", allow_zero=False)
    total = _number(metrics["total_seconds"], f"{path}.total_seconds", allow_zero=False)
    games_per_second = _reconcile_rate(
        count=games,
        seconds=total,
        stated=metrics.get("episodes_per_second"),
        path=f"{path}.episodes_per_second",
    )
    decisions_per_second = _reconcile_rate(
        count=decisions,
        seconds=total,
        stated=metrics.get("samples_per_second"),
        path=f"{path}.samples_per_second",
    )
    return {
        "games": games,
        "environment_turns": None,
        "policy_decisions": decisions,
        "wall_time_seconds": total,
        "games_per_second": games_per_second,
        "rollout_games_per_second": (
            games / _number(metrics["rollout_seconds"], f"{path}.rollout_seconds", allow_zero=False)
            if metrics.get("rollout_seconds") is not None
            else None
        ),
        "environment_turns_per_second": None,
        "rollout_environment_turns_per_second": None,
        "policy_decisions_per_second": decisions_per_second,
        "collected_policy_decisions_per_second": (
            _reconcile_rate(
                count=decisions,
                seconds=_number(
                    metrics["rollout_seconds"],
                    f"{path}.rollout_seconds",
                    allow_zero=False,
                ),
                stated=metrics.get("rollout_samples_per_second"),
                path=f"{path}.rollout_samples_per_second",
            )
            if metrics.get("rollout_seconds") is not None
            else None
        ),
        "update_time_seconds": (
            _number(metrics["update_seconds"], f"{path}.update_seconds")
            if metrics.get("update_seconds") is not None
            else None
        ),
        "total_time_seconds": total,
    }


def _timing_rows(metrics: dict[str, Any], path: str) -> list[dict[str, Any]]:
    phase_fields = {
        "rollout_seconds": "Rollout",
        "gae_seconds": "GAE",
        "mask_seconds": "Action masking",
        "prep_seconds": "Batch preparation",
        "update_seconds": "Policy update",
        "non_rollout_seconds": "Non-rollout",
    }
    rows = []
    for field, label in phase_fields.items():
        if metrics.get(field) is not None:
            rows.append(
                {
                    "phase": label,
                    "seconds": _number(metrics[field], f"{path}.{field}"),
                }
            )
    return rows


def _infer_player_count(payload: dict[str, Any]) -> int | None:
    sets = payload.get("workload", {}).get("agent_sets")
    if isinstance(sets, list) and sets and isinstance(sets[0], list):
        lengths = {len(item) for item in sets if isinstance(item, list)}
        if len(lengths) == 1:
            return lengths.pop()
    return None


def _normalize_training(
    payload: dict[str, Any],
    path: Path,
    root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    source = _source_record(path, root, payload)
    recorded_at = source["recorded_at"]
    environment = payload.get("environment", {})
    workload = payload.get("workload", {})
    player_count = _infer_player_count(payload)
    common_hardware = {
        "python": environment.get("python"),
        "torch": environment.get("torch"),
        "gpu": environment.get("gpu"),
        "gpu_memory_mib": environment.get("gpu_memory_mib"),
    }
    common_seed = workload.get("seed")
    if common_seed is not None:
        common_seed = _integer(common_seed, f"{path}.workload.seed")
    runs: list[dict[str, Any]] = []
    warnings = [
        f"{source['path']}: legacy 'samples' mapped to collected policy decisions; "
        "environment turns were not separately reported."
    ]

    def add_run(
        run_id: str,
        label: str,
        stage: str,
        metrics: dict[str, Any],
        *,
        seed: int | None,
        status: str,
        notes: Iterable[str] = (),
    ) -> None:
        run = _base_run(
            run_id=run_id,
            run_kind="training",
            status=status,
            recorded_at=recorded_at,
            source=source,
            player_count=player_count,
            seed=seed,
        )
        run["label"] = label
        run["stage"] = stage
        run["models"] = [
            {"id": model_id, "role": "training participant"}
            for model_id in sorted(
                {
                    str(agent)
                    for agent_set in workload.get("agent_sets", [])
                    if isinstance(agent_set, list)
                    for agent in agent_set
                }
            )
        ]
        run["performance"] = _performance_fields(metrics, f"{path}:{run_id}")
        run["timing"] = _timing_rows(metrics, f"{path}:{run_id}")
        run["resources"]["peak_gpu_memory_mib"] = metrics.get("peak_cuda_mib")
        run["provenance"]["hardware"] = common_hardware
        run["provenance"]["recipe"] = {
            key: workload.get(key)
            for key in (
                "pretrained",
                "historical_checkpoints",
                "policy_updates",
                "temperature",
                "penalty_aware_action_masks",
            )
            if key in workload
        }
        run["notes"].extend(str(note) for note in notes if note)
        runs.append(run)

    baseline = payload.get("baseline")
    if not isinstance(baseline, dict):
        raise BenchmarkInputError(f"{path}: baseline must be an object")
    for device in ("cpu", "cuda"):
        metrics = baseline.get(device)
        if isinstance(metrics, dict):
            add_run(
                f"training-legacy-baseline-{device}-{recorded_at}",
                f"Original baseline ({device.upper()})",
                "baseline",
                metrics,
                seed=common_seed,
                status="complete",
                notes=(
                    baseline.get("correctness_caveat"),
                    f"Revision: {baseline.get('revision')}" if baseline.get("revision") else None,
                ),
            )

    interim = payload.get("optimized_interim", {})
    if isinstance(interim, dict):
        for key, label in (
            ("cpu_rollout_cuda_training", "Interim CPU rollout + CUDA training"),
            ("cuda_rollout_cuda_training", "Interim CUDA rollout + CUDA training"),
        ):
            metrics = interim.get(key)
            if isinstance(metrics, dict):
                add_run(
                    f"training-interim-{key.replace('_', '-')}-{recorded_at}",
                    label,
                    "interim",
                    metrics,
                    seed=common_seed,
                    status="incomplete",
                    notes=(interim.get("correctness_caveat"),),
                )

    final = payload.get("final")
    if not isinstance(final, dict) or not isinstance(final.get("runs"), list):
        raise BenchmarkInputError(f"{path}: final.runs must be an array")
    for index, metrics in enumerate(final["runs"]):
        if not isinstance(metrics, dict):
            raise BenchmarkInputError(f"{path}: final.runs[{index}] must be an object")
        seed = _integer(metrics.get("seed"), f"{path}.final.runs[{index}].seed")
        add_run(
            f"training-final-seed-{seed}",
            f"Final recipe · seed {seed}",
            "final",
            metrics,
            seed=seed,
            status="complete",
            notes=(
                final.get("correctness"),
                final.get("comparability_caveat"),
                f"Architecture: {final.get('architecture')}" if final.get("architecture") else None,
            ),
        )

    # Rollout-only thread sweep rows are normalized separately because they do
    # not include update or total training time.
    sweep = payload.get("rollout_cpu_thread_sweep", [])
    if not isinstance(sweep, list):
        raise BenchmarkInputError(f"{path}: rollout_cpu_thread_sweep must be an array")
    for index, metrics in enumerate(sweep):
        if not isinstance(metrics, dict):
            raise BenchmarkInputError(f"{path}: rollout_cpu_thread_sweep[{index}] must be an object")
        threads = _integer(metrics.get("threads"), f"{path}.rollout_cpu_thread_sweep[{index}].threads")
        episodes = _integer(metrics.get("episodes"), f"{path}.rollout_cpu_thread_sweep[{index}].episodes", allow_zero=False)
        samples = _integer(metrics.get("samples"), f"{path}.rollout_cpu_thread_sweep[{index}].samples", allow_zero=False)
        seconds = _number(metrics.get("seconds"), f"{path}.rollout_cpu_thread_sweep[{index}].seconds", allow_zero=False)
        games_rate = _reconcile_rate(
            count=episodes,
            seconds=seconds,
            stated=metrics.get("episodes_per_second"),
            path=f"{path}.rollout_cpu_thread_sweep[{index}].episodes_per_second",
        )
        decisions_rate = _reconcile_rate(
            count=samples,
            seconds=seconds,
            stated=metrics.get("samples_per_second"),
            path=f"{path}.rollout_cpu_thread_sweep[{index}].samples_per_second",
        )
        run = _base_run(
            run_id=f"rollout-thread-sweep-{threads}-{recorded_at}",
            run_kind="microbenchmark",
            status="complete",
            recorded_at=recorded_at,
            source=source,
            player_count=player_count,
            seed=common_seed,
        )
        run["label"] = f"CPU rollout · {threads} thread{'s' if threads != 1 else ''}"
        run["stage"] = "thread_sweep"
        run["performance"] = {
            "games": episodes,
            "environment_turns": None,
            "policy_decisions": samples,
            "wall_time_seconds": seconds,
            "games_per_second": games_rate,
            "rollout_games_per_second": games_rate,
            "environment_turns_per_second": None,
            "rollout_environment_turns_per_second": None,
            "policy_decisions_per_second": decisions_rate,
            "collected_policy_decisions_per_second": decisions_rate,
            "update_time_seconds": None,
            "total_time_seconds": None,
        }
        run["timing"] = [{"phase": "Rollout", "seconds": seconds}]
        run["provenance"]["hardware"] = common_hardware
        run["provenance"]["recipe"] = {"cpu_threads": threads}
        runs.append(run)

    ratio = payload.get("initial_ratio_integrity")
    if isinstance(ratio, dict):
        for run in runs:
            if run.get("stage") != "final":
                continue
            run["guardrails"].append(
                {
                    "name": "Unchanged-policy ratio integrity",
                    "status": "observed",
                    "value": ratio,
                    "threshold": 0.02,
                    "notes": final.get("correctness"),
                }
            )
    return runs, warnings


def _normalize_tournament(
    payload: dict[str, Any],
    path: Path,
    root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    source = _source_record(path, root, payload)
    result = payload.get("result")
    if not isinstance(result, dict):
        raise BenchmarkInputError(f"{path}: result must be an object")
    agent = result.get("agent")
    opponent = result.get("opponent")
    if not isinstance(agent, str) or not agent:
        raise BenchmarkInputError(f"{path}: result.agent must be a non-empty string")
    if not isinstance(opponent, str) or not opponent:
        raise BenchmarkInputError(f"{path}: result.opponent must be a non-empty string")
    games = _integer(result.get("games"), f"{path}.result.games", allow_zero=False)
    wins = _integer(result.get("wins"), f"{path}.result.wins")
    losses = _integer(result.get("losses"), f"{path}.result.losses")
    draws = _integer(result.get("draws", 0), f"{path}.result.draws")
    if wins + losses + draws != games:
        raise MetricReconciliationError(
            f"{path}: wins + losses + draws ({wins + losses + draws}) != games ({games})"
        )
    win_rate = wins / games
    if result.get("win_rate") is not None and not _close(
        _number(result["win_rate"], f"{path}.result.win_rate"),
        win_rate,
        rtol=1e-9,
    ):
        raise MetricReconciliationError(f"{path}: result.win_rate does not equal wins/games")
    ci_low, ci_high = wilson_interval(wins, games)
    stated_ci = result.get("win_rate_ci95")
    if stated_ci is not None:
        if not isinstance(stated_ci, list) or len(stated_ci) != 2:
            raise BenchmarkInputError(f"{path}: result.win_rate_ci95 must contain two numbers")
        stated_low = _number(stated_ci[0], f"{path}.result.win_rate_ci95[0]")
        stated_high = _number(stated_ci[1], f"{path}.result.win_rate_ci95[1]")
        if not (_close(stated_low, ci_low, rtol=1e-8) and _close(stated_high, ci_high, rtol=1e-8)):
            raise MetricReconciliationError(
                f"{path}: result.win_rate_ci95 does not reconcile with Wilson interval"
            )
    elapsed = _number(
        result.get("elapsed_seconds"),
        f"{path}.result.elapsed_seconds",
        allow_zero=False,
    )
    average_turns = _number(
        result.get("average_turns"),
        f"{path}.result.average_turns",
        allow_zero=False,
    )
    environment_turns = int(round(average_turns * games))
    seed = _integer(result.get("seed"), f"{path}.result.seed")

    seat_keys = (
        "wins_as_seat_0",
        "games_as_seat_0",
        "wins_as_seat_1",
        "games_as_seat_1",
    )
    has_seats = all(key in result for key in seat_keys)
    player_count = 2 if has_seats else None
    warnings = []
    if has_seats:
        seat_games = _integer(result["games_as_seat_0"], f"{path}.result.games_as_seat_0") + _integer(
            result["games_as_seat_1"], f"{path}.result.games_as_seat_1"
        )
        if seat_games != games:
            raise MetricReconciliationError(
                f"{path}: seat game counts ({seat_games}) do not equal games ({games})"
            )
        warnings.append(
            f"{source['path']}: player_count inferred as 2 from complete seat-0/seat-1 counts."
        )

    run = _base_run(
        run_id=f"tournament-{path.stem}",
        run_kind="tournament",
        status="complete",
        recorded_at=source["recorded_at"],
        source=source,
        player_count=player_count,
        seed=seed,
    )
    run["label"] = f"{agent} vs {opponent} · {games:,} games"
    run["stage"] = "evaluation"
    run["models"] = [
        {"id": agent, "role": "evaluated"},
        {"id": opponent, "role": "opponent"},
    ]
    run["performance"] = {
        "games": games,
        "environment_turns": environment_turns,
        "policy_decisions": None,
        "wall_time_seconds": elapsed,
        "games_per_second": games / elapsed,
        "rollout_games_per_second": None,
        "environment_turns_per_second": environment_turns / elapsed,
        "rollout_environment_turns_per_second": None,
        "policy_decisions_per_second": None,
        "collected_policy_decisions_per_second": None,
        "update_time_seconds": None,
        "total_time_seconds": elapsed,
    }
    run["outcome"] = {
        "agent": agent,
        "opponent": opponent,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": games,
        "win_rate": win_rate,
        "win_rate_ci95": [ci_low, ci_high],
        "average_environment_turns": average_turns,
        "seats": (
            [
                {
                    "seat": 0,
                    "wins": _integer(result["wins_as_seat_0"], f"{path}.result.wins_as_seat_0"),
                    "games": _integer(result["games_as_seat_0"], f"{path}.result.games_as_seat_0"),
                },
                {
                    "seat": 1,
                    "wins": _integer(result["wins_as_seat_1"], f"{path}.result.wins_as_seat_1"),
                    "games": _integer(result["games_as_seat_1"], f"{path}.result.games_as_seat_1"),
                },
            ]
            if has_seats
            else []
        ),
    }
    tree_search = payload.get("tree_search")
    if isinstance(tree_search, dict):
        run["provenance"]["recipe"] = deepcopy(tree_search)
    checkpoint = payload.get("ppo_checkpoint")
    if isinstance(checkpoint, dict):
        sha = checkpoint.get("sha256")
        if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha)):
            raise BenchmarkInputError(f"{path}: ppo_checkpoint.sha256 must be a SHA-256 hex digest")
        run["provenance"]["checkpoint_sha256"] = sha
        run["provenance"]["checkpoint_path"] = checkpoint.get("path")
    return [run], warnings


def _normalize_vectorized_training(
    payload: dict[str, Any],
    path: Path,
    root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    source = _source_record(path, root, payload)
    workload = payload.get("workload")
    results = payload.get("results")
    if not isinstance(workload, dict) or not isinstance(results, dict):
        raise BenchmarkInputError(f"{path}: workload and results must be objects")
    player_count = _integer(
        workload.get("players_per_game"),
        f"{path}.workload.players_per_game",
        allow_zero=False,
    )
    if player_count not in (2, 3, 4):
        raise BenchmarkInputError(f"{path}: workload.players_per_game must be 2, 3, or 4")
    environment = payload.get("environment", {})
    git = payload.get("git", {})
    stage = (
        "vectorized_final"
        if path.stem == "training_speed_vectorized_20260731"
        else "vectorized_interim"
    )
    runs: list[dict[str, Any]] = []
    for implementation, thermal_groups in sorted(results.items()):
        if not isinstance(thermal_groups, dict):
            raise BenchmarkInputError(f"{path}: results.{implementation} must be an object")
        for thermal_state, group in sorted(thermal_groups.items()):
            if not isinstance(group, dict) or not isinstance(group.get("runs"), list):
                raise BenchmarkInputError(
                    f"{path}: results.{implementation}.{thermal_state}.runs must be an array"
                )
            for index, metrics in enumerate(group["runs"]):
                metric_path = (
                    f"{path}.results.{implementation}.{thermal_state}.runs[{index}]"
                )
                if not isinstance(metrics, dict):
                    raise BenchmarkInputError(f"{metric_path} must be an object")
                games = _integer(
                    metrics.get("episodes"),
                    f"{metric_path}.episodes",
                    allow_zero=False,
                )
                decisions = _integer(
                    metrics.get("collected_policy_decisions"),
                    f"{metric_path}.collected_policy_decisions",
                    allow_zero=False,
                )
                turns = _integer(
                    metrics.get("environment_turns"),
                    f"{metric_path}.environment_turns",
                    allow_zero=False,
                )
                rollout = _number(
                    metrics.get("rollout_seconds"),
                    f"{metric_path}.rollout_seconds",
                    allow_zero=False,
                )
                total = _number(
                    metrics.get("total_seconds"),
                    f"{metric_path}.total_seconds",
                    allow_zero=False,
                )
                seed = _integer(metrics.get("seed"), f"{metric_path}.seed")
                rollout_games = _reconcile_rate(
                    count=games,
                    seconds=rollout,
                    stated=metrics.get("games_per_second"),
                    path=f"{metric_path}.games_per_second",
                )
                rollout_turns = _reconcile_rate(
                    count=turns,
                    seconds=rollout,
                    stated=metrics.get("environment_turns_per_second"),
                    path=f"{metric_path}.environment_turns_per_second",
                )
                collected_decisions = _reconcile_rate(
                    count=decisions,
                    seconds=rollout,
                    stated=metrics.get("collected_policy_decisions_per_second"),
                    path=f"{metric_path}.collected_policy_decisions_per_second",
                )
                end_to_end_decisions = _reconcile_rate(
                    count=decisions,
                    seconds=total,
                    stated=metrics.get("end_to_end_policy_decisions_per_second"),
                    path=f"{metric_path}.end_to_end_policy_decisions_per_second",
                )
                run = _base_run(
                    run_id=(
                        f"vectorized-training-{path.stem}-{implementation}-"
                        f"{thermal_state}-seed-{seed}"
                    ),
                    run_kind="training",
                    status="complete" if payload.get("status") == "final" else "incomplete",
                    recorded_at=source["recorded_at"],
                    source=source,
                    player_count=player_count,
                    seed=seed,
                )
                run["label"] = (
                    f"{implementation.title()} · {thermal_state} · seed {seed}"
                    f"{' · interim source' if stage == 'vectorized_interim' else ''}"
                )
                run["stage"] = stage
                run["models"] = [
                    {"id": str(model), "role": "training participant"}
                    for agent_set in workload.get("agent_sets", [])
                    if isinstance(agent_set, list)
                    for model in agent_set
                ]
                run["performance"] = {
                    "games": games,
                    "environment_turns": turns,
                    "policy_decisions": decisions,
                    "wall_time_seconds": total,
                    "games_per_second": games / total,
                    "rollout_games_per_second": rollout_games,
                    "environment_turns_per_second": turns / total,
                    "rollout_environment_turns_per_second": rollout_turns,
                    "policy_decisions_per_second": end_to_end_decisions,
                    "collected_policy_decisions_per_second": collected_decisions,
                    "update_time_seconds": _number(
                        metrics.get("update_seconds"),
                        f"{metric_path}.update_seconds",
                    ),
                    "total_time_seconds": total,
                }
                run["timing"] = _timing_rows(metrics, metric_path)
                run["resources"] = {
                    "cpu_utilization_pct": None,
                    "gpu_utilization_pct": None,
                    "peak_cpu_memory_mib": metrics.get("rss_peak_mib"),
                    "peak_gpu_memory_mib": metrics.get("cuda_peak_allocated_mib"),
                }
                run["guardrails"].append(
                    {
                        "name": "First-update unchanged-policy ratio error",
                        "status": "observed",
                        "value": metrics.get("first_update_max_ratio_error"),
                        "notes": "Source reports the maximum first-update ratio error but no pass/fail threshold.",
                    }
                )
                run["provenance"]["hardware"] = environment
                run["provenance"]["git_sha"] = (
                    git.get("revision") if isinstance(git, dict) else None
                )
                run["provenance"]["worktree_dirty"] = (
                    git.get("worktree_dirty") if isinstance(git, dict) else None
                )
                recipe = {
                    key: workload.get(key)
                    for key in (
                        "actor",
                        "critic",
                        "rollout_device",
                        "vector_game_batch_size",
                        "observation_features",
                        "actions",
                        "policy_updates",
                        "temperature",
                    )
                    if key in workload
                }
                recipe.update(
                    {
                        "implementation": implementation,
                        "thermal_state": thermal_state,
                    }
                )
                run["provenance"]["recipe"] = recipe
                run["notes"].extend(
                    str(note) for note in payload.get("semantic_caveats", [])
                )
                runs.append(run)
    return runs, []


def _normalize_puffer_smoke(
    payload: dict[str, Any],
    path: Path,
    root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    source = _source_record(path, root, payload)
    rollout = payload.get("rollout")
    training = payload.get("training")
    model = payload.get("model")
    integration = payload.get("integration")
    if not all(isinstance(item, dict) for item in (rollout, training, model, integration)):
        raise BenchmarkInputError(
            f"{path}: rollout, training, model, and integration must be objects"
        )
    elapsed = _number(
        payload.get("elapsed_seconds"), f"{path}.elapsed_seconds", allow_zero=False
    )
    games = _integer(rollout.get("episodes"), f"{path}.rollout.episodes", allow_zero=False)
    turns = _integer(rollout.get("turns"), f"{path}.rollout.turns", allow_zero=False)
    decisions = _integer(
        rollout.get("decisions"), f"{path}.rollout.decisions", allow_zero=False
    )
    seed = _integer(payload.get("seed"), f"{path}.seed")
    native = bool(integration.get("native_pufferlib"))
    run = _base_run(
        run_id=f"puffer-smoke-{path.stem}",
        run_kind="evaluation",
        status="complete" if native else "incomplete",
        recorded_at=source["recorded_at"],
        source=source,
        player_count=None,
        seed=seed,
    )
    run["label"] = f"{model.get('name', 'Puffer-style model')} smoke test"
    run["stage"] = "integration_smoke"
    run["models"] = [
        {
            "id": str(model.get("name", "puffer_style_temporal_net")),
            "role": "integration candidate",
        }
    ]
    run["performance"] = {
        "games": games,
        "environment_turns": turns,
        "policy_decisions": decisions,
        "wall_time_seconds": elapsed,
        "games_per_second": games / elapsed,
        "rollout_games_per_second": None,
        "environment_turns_per_second": turns / elapsed,
        "rollout_environment_turns_per_second": None,
        "policy_decisions_per_second": decisions / elapsed,
        "collected_policy_decisions_per_second": None,
        "update_time_seconds": None,
        "total_time_seconds": elapsed,
    }
    run["guardrails"] = [
        {
            "name": "Training parameters changed",
            "status": "passed" if training.get("parameters_changed") is True else "failed",
            "value": training.get("parameters_changed"),
            "notes": "Smoke-run source verdict.",
        },
        {
            "name": "Privileged critic disabled",
            "status": "observed",
            "value": model.get("critic_privileged"),
            "notes": "Configuration observation; not a complete information-leakage test.",
        },
    ]
    run["provenance"]["recipe"] = {
        "model": model,
        "training": {
            key: training.get(key)
            for key in ("samples", "loss", "policy_loss", "value_loss", "entropy")
            if key in training
        },
        "integration": integration,
    }
    run["notes"].append(str(integration.get("status", "Integration status not reported")))
    return [run], []


def validate_normalized_run(run: dict[str, Any], path: str = "run") -> None:
    if run.get("schema_version") != SCHEMA_VERSION:
        raise BenchmarkInputError(f"{path}.schema_version must be {SCHEMA_VERSION}")
    if not isinstance(run.get("run_id"), str) or not run["run_id"]:
        raise BenchmarkInputError(f"{path}.run_id must be a non-empty string")
    if run.get("run_kind") not in RUN_KINDS:
        raise BenchmarkInputError(f"{path}.run_kind must be one of {sorted(RUN_KINDS)}")
    if run.get("status") not in RUN_STATUSES:
        raise BenchmarkInputError(f"{path}.status must be one of {sorted(RUN_STATUSES)}")
    try:
        date.fromisoformat(str(run.get("recorded_at"))[:10])
    except ValueError as exc:
        raise BenchmarkInputError(f"{path}.recorded_at must be an ISO date") from exc
    game = run.get("game")
    if not isinstance(game, dict):
        raise BenchmarkInputError(f"{path}.game must be an object")
    player_count = game.get("player_count")
    if player_count is not None and player_count not in (2, 3, 4):
        raise BenchmarkInputError(f"{path}.game.player_count must be 2, 3, 4, or null")
    performance = run.get("performance", {})
    if not isinstance(performance, dict):
        raise BenchmarkInputError(f"{path}.performance must be an object")
    if run["status"] == "complete" and run["run_kind"] in {"training", "tournament"}:
        if performance.get("wall_time_seconds") is None:
            raise BenchmarkInputError(f"{path}.performance.wall_time_seconds is required for complete runs")
    if performance.get("games") is not None and performance.get("wall_time_seconds") is not None:
        games = _number(performance["games"], f"{path}.performance.games")
        seconds = _number(
            performance["wall_time_seconds"],
            f"{path}.performance.wall_time_seconds",
            allow_zero=False,
        )
        stated = performance.get("games_per_second")
        if stated is not None and not _close(
            _number(stated, f"{path}.performance.games_per_second"),
            games / seconds,
        ):
            raise MetricReconciliationError(
                f"{path}.performance.games_per_second does not reconcile with games/wall_time_seconds"
            )
    outcome = run.get("outcome")
    if outcome is not None:
        if not isinstance(outcome, dict):
            raise BenchmarkInputError(f"{path}.outcome must be an object or null")
        games = _integer(outcome.get("games"), f"{path}.outcome.games", allow_zero=False)
        wins = _integer(outcome.get("wins"), f"{path}.outcome.wins")
        losses = _integer(outcome.get("losses"), f"{path}.outcome.losses")
        draws = _integer(outcome.get("draws", 0), f"{path}.outcome.draws")
        if wins + losses + draws != games:
            raise MetricReconciliationError(f"{path}.outcome counts do not add to games")


def _normalize_schema_v1(
    payload: dict[str, Any],
    path: Path,
    root: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    source = _source_record(path, root, payload)
    raw_runs = payload.get("runs") if isinstance(payload.get("runs"), list) else [payload]
    runs = []
    for index, raw in enumerate(raw_runs):
        if not isinstance(raw, dict):
            raise BenchmarkInputError(f"{path}: runs[{index}] must be an object")
        run = deepcopy(raw)
        validate_normalized_run(run, f"{path}:runs[{index}]")
        provenance = run.setdefault("provenance", {})
        provenance.setdefault("source_path", source["path"])
        provenance.setdefault("source_sha256", source["sha256"])
        run.setdefault("models", [])
        run.setdefault("timing", [])
        run.setdefault("resources", {})
        run.setdefault("guardrails", [])
        run.setdefault("learning_curve", [])
        run.setdefault("notes", [])
        runs.append(run)
    return runs, []


def normalize_file(path: Path, root: Path) -> tuple[list[dict[str, Any]], dict[str, Any], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BenchmarkInputError(
            f"{path}: malformed JSON at line {exc.lineno}, column {exc.colno}"
        ) from exc
    if not isinstance(payload, dict):
        raise BenchmarkInputError(f"{path}: top-level JSON value must be an object")
    if (
        payload.get("schema_version") == SCHEMA_VERSION
        and "run_id" in payload
    ) or isinstance(payload.get("runs"), list):
        runs, warnings = _normalize_schema_v1(payload, path, root)
    elif {
        "elapsed_seconds",
        "integration",
        "model",
        "rollout",
        "training",
    }.issubset(payload):
        runs, warnings = _normalize_puffer_smoke(payload, path, root)
    elif "results" in payload and isinstance(payload.get("workload"), dict):
        runs, warnings = _normalize_vectorized_training(payload, path, root)
    elif "result" in payload:
        runs, warnings = _normalize_tournament(payload, path, root)
    elif "final" in payload and "baseline" in payload:
        runs, warnings = _normalize_training(payload, path, root)
    else:
        raise UnsupportedBenchmarkShapeError(
            f"{path}: unsupported raw benchmark shape; expected schema-v1, "
            "tournament, puffer-smoke, vectorized-training, or "
            "training-performance JSON"
        )
    source = _source_record(path, root, payload)
    for index, run in enumerate(runs):
        validate_normalized_run(run, f"{path}:normalized[{index}]")
    return runs, source, warnings


def load_and_normalize(paths: Iterable[Path], root: Path) -> NormalizationResult:
    runs: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    for path in sorted(Path(item) for item in paths):
        try:
            file_runs, source, file_warnings = normalize_file(path, root)
        except UnsupportedBenchmarkShapeError:
            warnings.append(
                f"{_relative_source(path, root)}: skipped unsupported "
                "autoresearch raw benchmark shape"
            )
            continue
        for run in file_runs:
            if run["run_id"] in seen_ids:
                raise BenchmarkInputError(f"duplicate normalized run_id: {run['run_id']}")
            seen_ids.add(run["run_id"])
            runs.append(run)
        source_files.append(source)
        warnings.extend(file_warnings)
    if not runs:
        raise BenchmarkInputError("no benchmark runs were found")
    return NormalizationResult(runs=runs, source_files=source_files, warnings=warnings)
