"""Build a canonical Data Analytics dashboard artifact from normalized runs."""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


NORMALIZED_SOURCE_ID = "normalized_benchmark_runs"


def _display_model(model_id: str) -> str:
    aliases = {
        "ppo": "PPO",
        "rl": "RL",
        "bayesian_tree_search": "Bayesian tree search",
        "smart": "Smart agent",
        "random": "Random agent",
    }
    return aliases.get(model_id, model_id.replace("_", " ").title())


def _short_sha(value: str | None) -> str:
    return value[:12] if value else "Not reported"


def _iso_generated_at(runs: list[dict[str, Any]]) -> str:
    latest = max(str(run["recorded_at"])[:10] for run in runs)
    return f"{latest}T23:59:59Z"


def _source_specs(
    source_files: list[dict[str, Any]],
    normalized_path: str,
    generated_at: str,
) -> list[dict[str, Any]]:
    sources = [
        {
            "id": NORMALIZED_SOURCE_ID,
            "label": "Normalized benchmark runs",
            "path": normalized_path,
            "query": {
                "sql": "SELECT payload_json FROM benchmark_runs ORDER BY run_id",
                "language": "sql",
                "engine": "sqlite",
                "description": "Actual query used by the generator to reload normalized run payloads before building dashboard datasets.",
                "executed_at": generated_at,
                "tables_used": ["benchmark_runs"],
                "filters": [
                    "Top-level benchmark JSON only; generated dashboard outputs are excluded.",
                    "Largest-game run per unordered pair is used for cross-play ratings.",
                ],
                "metric_definitions": [
                    "Win rate = wins / games; intervals use the Wilson 95% score interval.",
                    "Legacy tournament environment turns = round(average_turns * games).",
                    "Legacy training samples map to collected policy decisions; missing environment turns remain null.",
                    "Bradley-Terry ratings select the largest-game run for each observed unordered pair and use a 1500-centered Elo-style display scale.",
                    "A candidate beats a declared champion only when its 95% win-rate interval is entirely above 50%.",
                ],
            },
        }
    ]
    for index, item in enumerate(source_files):
        sources.append(
            {
                "id": f"input_{index + 1}",
                "label": (
                    f"{Path(item['path']).name} · {item['recorded_at']} · "
                    f"SHA-256 {item['sha256'][:12]}"
                ),
                "path": item["path"],
            }
        )
    return sources


def _training_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        if run["run_kind"] not in {"training", "microbenchmark"}:
            continue
        performance = run.get("performance", {})
        recipe = run.get("provenance", {}).get("recipe") or {}
        rows.append(
            {
                "run_id": run["run_id"],
                "configuration": run.get("label", run["run_id"]),
                "stage": run.get("stage", "unknown"),
                "status": run["status"],
                "player_count": run.get("game", {}).get("player_count"),
                "seed": run.get("provenance", {}).get("seed"),
                "games": performance.get("games"),
                "policy_decisions": performance.get("policy_decisions"),
                "total_seconds": performance.get("total_time_seconds"),
                "games_per_second": performance.get("games_per_second"),
                "policy_decisions_per_second": performance.get(
                    "policy_decisions_per_second"
                ),
                "collected_policy_decisions_per_second": performance.get(
                    "collected_policy_decisions_per_second"
                ),
                "environment_turns_per_second": performance.get(
                    "environment_turns_per_second"
                ),
                "rollout_environment_turns_per_second": performance.get(
                    "rollout_environment_turns_per_second"
                ),
                "rollout_games_per_second": performance.get(
                    "rollout_games_per_second"
                ),
                "update_seconds": performance.get("update_time_seconds"),
                "peak_gpu_memory_mib": run.get("resources", {}).get(
                    "peak_gpu_memory_mib"
                ),
                "peak_cpu_memory_mib": run.get("resources", {}).get(
                    "peak_cpu_memory_mib"
                ),
                "source_path": run.get("provenance", {}).get("source_path"),
                "implementation": (
                    recipe.get("implementation")
                    if isinstance(recipe, dict)
                    else None
                ),
                "thermal_state": (
                    recipe.get("thermal_state")
                    if isinstance(recipe, dict)
                    else None
                ),
            }
        )
    return rows


def _matchup_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        outcome = run.get("outcome")
        if not outcome:
            continue
        ci_low, ci_high = outcome["win_rate_ci95"]
        seats = outcome.get("seats", [])
        seat_0_rate = seats[0]["wins"] / seats[0]["games"] if len(seats) > 0 else None
        seat_1_rate = seats[1]["wins"] / seats[1]["games"] if len(seats) > 1 else None
        rows.append(
            {
                "run_id": run["run_id"],
                "matchup": (
                    f"{_display_model(outcome['agent'])} vs "
                    f"{_display_model(outcome['opponent'])}"
                ),
                "agent": _display_model(outcome["agent"]),
                "agent_id": outcome["agent"],
                "opponent": _display_model(outcome["opponent"]),
                "opponent_id": outcome["opponent"],
                "player_count": run.get("game", {}).get("player_count"),
                "games": outcome["games"],
                "wins": outcome["wins"],
                "losses": outcome["losses"],
                "draws": outcome["draws"],
                "win_rate": outcome["win_rate"],
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "confidence_vs_even": (
                    "Above 50%"
                    if ci_low > 0.5
                    else "Below 50%"
                    if ci_high < 0.5
                    else "Includes 50%"
                ),
                "elapsed_seconds": run["performance"].get("wall_time_seconds"),
                "environment_turns": run["performance"].get("environment_turns"),
                "average_environment_turns": outcome.get(
                    "average_environment_turns"
                ),
                "seat_0_win_rate": seat_0_rate,
                "seat_1_win_rate": seat_1_rate,
                "seat_bias_pp": (
                    (seat_0_rate - seat_1_rate) * 100
                    if seat_0_rate is not None and seat_1_rate is not None
                    else None
                ),
                "seed": run.get("provenance", {}).get("seed"),
                "checkpoint_sha256": run.get("provenance", {}).get(
                    "checkpoint_sha256"
                ),
                "source_path": run.get("provenance", {}).get("source_path"),
            }
        )
    return rows


def _largest_pairwise_rows(matchups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in matchups:
        pair = tuple(sorted((row["agent_id"], row["opponent_id"])))
        current = selected.get(pair)
        if current is None or (row["games"], row["run_id"]) > (
            current["games"],
            current["run_id"],
        ):
            selected[pair] = row
    return sorted(selected.values(), key=lambda row: row["run_id"])


def _crossplay_rows(matchups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in _largest_pairwise_rows(matchups):
        rows.append(
            {
                "model": row["agent"],
                "opponent": row["opponent"],
                "model_id": row["agent_id"],
                "opponent_id": row["opponent_id"],
                "player_count": row["player_count"],
                "games": row["games"],
                "wins": row["wins"],
                "win_rate": row["win_rate"],
                "ci95_low": row["ci95_low"],
                "ci95_high": row["ci95_high"],
                "source_run": row["run_id"],
            }
        )
        rows.append(
            {
                "model": row["opponent"],
                "opponent": row["agent"],
                "model_id": row["opponent_id"],
                "opponent_id": row["agent_id"],
                "player_count": row["player_count"],
                "games": row["games"],
                "wins": row["losses"],
                "win_rate": 1 - row["win_rate"],
                "ci95_low": 1 - row["ci95_high"],
                "ci95_high": 1 - row["ci95_low"],
                "source_run": row["run_id"],
            }
        )
    return rows


def _crossplay_matrix(
    crossplay: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    models = sorted(
        {row["model_id"] for row in crossplay}
        | {row["opponent_id"] for row in crossplay}
    )
    by_pair = {
        (row["model_id"], row["opponent_id"]): row["win_rate"]
        for row in crossplay
    }
    rows = []
    for model in models:
        row: dict[str, Any] = {
            "model": _display_model(model),
            "model_id": model,
            "player_count": 2,
        }
        for index, opponent in enumerate(models):
            row[f"opponent_{index}"] = (
                None if model == opponent else by_pair.get((model, opponent))
            )
        rows.append(row)
    columns = [
        {"field": "model", "label": "Model", "type": "text"},
        {"field": "player_count", "label": "Players", "type": "number"},
    ]
    columns.extend(
        {
            "field": f"opponent_{index}",
            "label": f"vs {_display_model(opponent)}",
            "format": "percent",
        }
        for index, opponent in enumerate(models)
    )
    return rows, columns


def _bradley_terry_rows(matchups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = _largest_pairwise_rows(matchups)
    models = sorted(
        {
            model
            for row in selected
            for model in (row["agent_id"], row["opponent_id"])
        }
    )
    if len(models) < 2:
        return []
    wins = defaultdict(float)
    games = defaultdict(float)
    for row in selected:
        agent = row["agent_id"]
        opponent = row["opponent_id"]
        wins[agent] += row["wins"] + row["draws"] / 2
        wins[opponent] += row["losses"] + row["draws"] / 2
        games[tuple(sorted((agent, opponent)))] += row["games"]
    ability = {model: 1.0 for model in models}
    for _ in range(10_000):
        updated = {}
        for model in models:
            denominator = 0.0
            for opponent in models:
                if opponent == model:
                    continue
                n_ij = games.get(tuple(sorted((model, opponent))), 0.0)
                if n_ij:
                    denominator += n_ij / (ability[model] + ability[opponent])
            updated[model] = wins[model] / denominator if denominator else ability[model]
            updated[model] = max(updated[model], 1e-12)
        geometric_mean = math.exp(
            sum(math.log(value) for value in updated.values()) / len(updated)
        )
        updated = {
            model: value / geometric_mean for model, value in updated.items()
        }
        if max(
            abs(math.log(updated[model] / ability[model])) for model in models
        ) < 1e-12:
            ability = updated
            break
        ability = updated
    rows = []
    for rank, model in enumerate(
        sorted(models, key=lambda item: (-ability[item], item)), start=1
    ):
        connected_games = sum(
            games.get(tuple(sorted((model, opponent))), 0.0)
            for opponent in models
            if opponent != model
        )
        rows.append(
            {
                "rank": rank,
                "model": _display_model(model),
                "model_id": model,
                "player_count": 2,
                "bt_strength": ability[model],
                "rating": 1500 + 400 * math.log10(ability[model]),
                "connected_games": int(connected_games),
                "schedule_note": "Incomplete pairwise schedule; descriptive only",
            }
        )
    return rows


def _explicit_challenge(runs: list[dict[str, Any]]) -> dict[str, Any]:
    champion_ids = {
        model["id"]
        for run in runs
        for model in run.get("models", [])
        if str(model.get("role", "")).lower() == "champion"
    }
    if len(champion_ids) != 1:
        return {
            "champion_status": (
                "Not declared" if not champion_ids else "Ambiguous declarations"
            ),
            "challenge_status": "No eligible declared matchup",
            "time_to_beat_seconds": None,
            "decisions_to_beat": None,
        }
    champion = next(iter(champion_ids))
    eligible = []
    for run in runs:
        outcome = run.get("outcome")
        if not outcome:
            continue
        candidate_ids = {
            model["id"]
            for model in run.get("models", [])
            if str(model.get("role", "")).lower() == "candidate"
        }
        if len(candidate_ids) != 1 or champion not in {
            outcome["agent"],
            outcome["opponent"],
        }:
            continue
        candidate = next(iter(candidate_ids))
        if candidate == outcome["agent"]:
            rate = outcome["win_rate"]
            ci_low, ci_high = outcome["win_rate_ci95"]
        else:
            rate = 1 - outcome["win_rate"]
            ci_low, ci_high = 1 - outcome["win_rate_ci95"][1], 1 - outcome["win_rate_ci95"][0]
        eligible.append((run, candidate, rate, ci_low, ci_high))
    if not eligible:
        return {
            "champion_status": _display_model(champion),
            "challenge_status": "No eligible declared matchup",
            "time_to_beat_seconds": None,
            "decisions_to_beat": None,
        }
    run, candidate, rate, ci_low, ci_high = max(
        eligible, key=lambda item: item[0]["outcome"]["games"]
    )
    if ci_low > 0.5:
        status = f"{_display_model(candidate)} beat champion"
        time_to_beat = run["performance"].get("wall_time_seconds")
        decisions = run["performance"].get("environment_turns")
    elif ci_high < 0.5:
        status = f"{_display_model(candidate)} did not beat champion"
        time_to_beat = None
        decisions = None
    else:
        status = f"{_display_model(candidate)} result is inconclusive"
        time_to_beat = None
        decisions = None
    return {
        "champion_status": _display_model(champion),
        "challenge_status": status,
        "time_to_beat_seconds": time_to_beat,
        "decisions_to_beat": decisions,
        "candidate_win_rate": rate,
        "candidate_ci95_low": ci_low,
        "candidate_ci95_high": ci_high,
    }


def _headline_row(
    runs: list[dict[str, Any]],
    training: list[dict[str, Any]],
    matchups: list[dict[str, Any]],
) -> dict[str, Any]:
    challenge = _explicit_challenge(runs)
    final_rows = [
        row
        for row in training
        if row["stage"] == "vectorized_final"
        and row["status"] == "complete"
        and row["implementation"] == "vectorized"
        and row["thermal_state"] == "warm"
        and row["collected_policy_decisions_per_second"] is not None
    ]
    if not final_rows:
        final_rows = [
            row
            for row in training
            if row["stage"] == "final"
            and row["status"] == "complete"
            and row["collected_policy_decisions_per_second"] is not None
        ]
    largest_matchup = max(matchups, key=lambda row: row["games"], default=None)
    challenge.update(
        {
            "player_count": 2,
            "final_policy_decisions_per_second": (
                statistics.median(
                    row["collected_policy_decisions_per_second"]
                    for row in final_rows
                )
                if final_rows
                else None
            ),
            "final_total_seconds": (
                statistics.median(row["total_seconds"] for row in final_rows)
                if final_rows
                else None
            ),
            "largest_evaluation_seconds": (
                largest_matchup["elapsed_seconds"] if largest_matchup else None
            ),
            "largest_evaluation_turns": (
                largest_matchup["environment_turns"] if largest_matchup else None
            ),
            "largest_evaluation_games": (
                largest_matchup["games"] if largest_matchup else None
            ),
        }
    )
    return challenge


def _time_breakdown_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        if run.get("stage") not in {"final", "vectorized_final"}:
            continue
        for phase in run.get("timing", []):
            rows.append(
                {
                    "run_id": run["run_id"],
                    "seed": str(run.get("provenance", {}).get("seed")),
                    "configuration": run.get("label", run["run_id"]),
                    "phase": phase["phase"],
                    "seconds": phase["seconds"],
                    "player_count": run.get("game", {}).get("player_count"),
                    "total_seconds": run.get("performance", {}).get(
                        "total_time_seconds"
                    ),
                }
            )
    return rows


def _seed_variance_rows(training: list[dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = [
        row
        for row in training
        if row["stage"] == "vectorized_final" and row["thermal_state"] == "warm"
    ]
    selected = preferred or [row for row in training if row["stage"] == "final"]
    return [
        {
            "seed": str(row["seed"]),
            "player_count": row["player_count"],
            "implementation": row["implementation"] or "legacy final",
            "games_per_second": row["games_per_second"],
            "policy_decisions_per_second": row["policy_decisions_per_second"],
            "collected_policy_decisions_per_second": row[
                "collected_policy_decisions_per_second"
            ],
            "total_seconds": row["total_seconds"],
            "policy_decisions": row["policy_decisions"],
        }
        for row in selected
    ]


def _coverage_rows(
    runs: list[dict[str, Any]],
    matchups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    has_learning = any(run.get("learning_curve") for run in runs)
    has_declared_champion = any(
        str(model.get("role", "")).lower() == "champion"
        for run in runs
        for model in run.get("models", [])
    )
    has_ratings = len(_bradley_terry_rows(matchups)) >= 2
    complete_player_counts = {
        row["player_count"] for row in matchups if row["player_count"] is not None
    }
    has_gpu_peak = any(
        run.get("resources", {}).get("peak_gpu_memory_mib") is not None
        for run in runs
    )
    has_utilization = any(
        run.get("resources", {}).get("cpu_utilization_pct") is not None
        or run.get("resources", {}).get("gpu_utilization_pct") is not None
        for run in runs
    )
    definitions = [
        (
            "Declared champion and time-to-beat",
            has_declared_champion,
            "Requires explicit model roles plus an eligible candidate/champion matchup.",
        ),
        (
            "Learning curves",
            has_learning,
            "Requires timestamped or decision-indexed checkpoint observations.",
        ),
        (
            "Win rates with 95% CI",
            bool(matchups),
            f"{len(matchups)} source matchup run(s); Wilson intervals reconciled.",
        ),
        (
            "Cross-play matrix",
            bool(matchups),
            "Observed cells only; missing model pairs remain blank.",
        ),
        (
            "Bradley-Terry ratings",
            has_ratings,
            "Descriptive rating from largest-game run per observed pair; schedule is incomplete.",
        ),
        (
            "Seed variance",
            len(
                {
                    run.get("provenance", {}).get("seed")
                    for run in runs
                    if run.get("stage") == "final"
                }
            )
            >= 2,
            "Final training recipe has multiple seeds.",
        ),
        (
            "Seat bias",
            any(row["seat_bias_pp"] is not None for row in matchups),
            "Seat-specific wins and games are reported in tournament inputs.",
        ),
        (
            "2-player results",
            2 in complete_player_counts,
            "Player count inferred only where complete seat-0/seat-1 counts exist.",
        ),
        ("3-player results", 3 in complete_player_counts, "No 3-player result file."),
        ("4-player results", 4 in complete_player_counts, "No 4-player result file."),
        (
            "CPU/GPU utilization",
            has_utilization,
            "No sampled utilization series or aggregate was reported.",
        ),
        (
            "Peak memory",
            has_gpu_peak,
            "Only source-reported peaks are rendered.",
        ),
        (
            "Illegal-action guardrail",
            any(
                "illegal" in str(guard.get("name", "")).lower()
                for run in runs
                for guard in run.get("guardrails", [])
            ),
            "No illegal-action count or rate was reported.",
        ),
        (
            "Information-leakage guardrail",
            any(
                "leak" in str(guard.get("name", "")).lower()
                for run in runs
                for guard in run.get("guardrails", [])
            ),
            "No benchmark-level information-leakage result was reported.",
        ),
    ]
    return [
        {
            "view": label,
            "status": "Available" if available else "No data",
            "supported": available,
            "evidence": evidence,
        }
        for label, available, evidence in definitions
    ]


def _provenance_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        provenance = run.get("provenance", {})
        recipe = provenance.get("recipe")
        hardware = provenance.get("hardware")
        rows.append(
            {
                "run_id": run["run_id"],
                "kind": run["run_kind"],
                "status": run["status"],
                "recorded_at": run["recorded_at"],
                "player_count": run.get("game", {}).get("player_count"),
                "seed": provenance.get("seed"),
                "checkpoint_hash": _short_sha(
                    provenance.get("checkpoint_sha256")
                ),
                "recipe": (
                    json.dumps(recipe, sort_keys=True, separators=(",", ":"))
                    if recipe
                    else "Not reported"
                ),
                "observation_version": (
                    run.get("game", {}).get("observation_version")
                    or "Not reported"
                ),
                "benchmark_git_sha": provenance.get("git_sha") or "Not reported",
                "hardware": (
                    json.dumps(hardware, sort_keys=True, separators=(",", ":"))
                    if hardware
                    else "Not reported"
                ),
                "source_path": provenance.get("source_path"),
                "source_sha256": provenance.get("source_sha256"),
            }
        )
    return rows


def _guardrail_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observed = []
    for run in runs:
        for guard in run.get("guardrails", []):
            observed.append(
                {
                    "guardrail": guard.get("name"),
                    "status": guard.get("status", "observed"),
                    "run_id": run["run_id"],
                    "evidence": guard.get("notes") or "Source-reported observation",
                }
            )
    observed.extend(
        [
            {
                "guardrail": "Illegal actions",
                "status": "No data",
                "run_id": "—",
                "evidence": "No count, rate, or pass/fail result in current benchmark JSON.",
            },
            {
                "guardrail": "Information leakage",
                "status": "No data",
                "run_id": "—",
                "evidence": "No benchmark-level leakage test result in current benchmark JSON.",
            },
            {
                "guardrail": "Benchmark run git SHA",
                "status": "No data",
                "run_id": "—",
                "evidence": "Current inputs do not report the code revision used for each run.",
            },
        ]
    )
    return observed


def build_artifact(
    runs: list[dict[str, Any]],
    source_files: list[dict[str, Any]],
    *,
    normalized_path: str,
) -> dict[str, Any]:
    generated_at = _iso_generated_at(runs)
    sources = _source_specs(source_files, normalized_path, generated_at)
    training = _training_rows(runs)
    matchups = _matchup_rows(runs)
    crossplay = _crossplay_rows(matchups)
    crossplay_matrix, crossplay_columns = _crossplay_matrix(crossplay)
    ratings = _bradley_terry_rows(matchups)
    time_breakdown = _time_breakdown_rows(runs)
    seed_variance = _seed_variance_rows(training)
    throughput_summary = [
        row for row in training if row["stage"] == "vectorized_final"
    ] or [row for row in training if row["stage"] == "final"]
    coverage = _coverage_rows(runs, matchups)
    provenance = _provenance_rows(runs)
    guardrails = _guardrail_rows(runs)
    headline = [_headline_row(runs, training, matchups)]

    cards = [
        {
            "id": "champion",
            "description": "Explicitly declared champion in the normalized benchmark inputs.",
            "dataset": "headline",
            "sourceId": NORMALIZED_SOURCE_ID,
            "metrics": [{"label": "Declared champion", "field": "champion_status"}],
        },
        {
            "id": "challenge",
            "description": "Candidate result against an explicitly declared champion using a 95% confidence interval.",
            "dataset": "headline",
            "sourceId": NORMALIZED_SOURCE_ID,
            "metrics": [{"label": "Candidate verdict", "field": "challenge_status"}],
        },
        {
            "id": "evaluation_time",
            "description": "Wall time of the largest current tournament run, not time-to-beat unless a champion is declared and beaten.",
            "dataset": "headline",
            "sourceId": NORMALIZED_SOURCE_ID,
            "metrics": [
                {
                    "label": "Largest evaluation wall time",
                    "field": "largest_evaluation_seconds",
                    "format": "number",
                }
            ],
        },
        {
            "id": "evaluation_turns",
            "description": "Derived as source average turns multiplied by games for the largest tournament run.",
            "dataset": "headline",
            "sourceId": NORMALIZED_SOURCE_ID,
            "metrics": [
                {
                    "label": "Largest evaluation env turns",
                    "field": "largest_evaluation_turns",
                    "format": "compact",
                }
            ],
        },
        {
            "id": "training_throughput",
            "description": "Median collected policy decisions per second across final warm-vectorized seeds.",
            "dataset": "headline",
            "sourceId": NORMALIZED_SOURCE_ID,
            "metrics": [
                {
                    "label": "Final vectorized collected decisions/s",
                    "field": "final_policy_decisions_per_second",
                    "format": "compact",
                }
            ],
        },
        {
            "id": "training_time",
            "description": "Median total time across final warm-vectorized runs (128 games per source run).",
            "dataset": "headline",
            "sourceId": NORMALIZED_SOURCE_ID,
            "metrics": [
                {
                    "label": "Final vectorized run time",
                    "field": "final_total_seconds",
                    "format": "number",
                }
            ],
        },
    ]

    charts = [
        {
            "id": "matchup_win_rates",
            "title": "Agent win rates in observed tournament runs",
            "subtitle": "Bars show point estimates; exact Wilson 95% intervals and sample sizes are in the adjacent table.",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "How did each evaluated agent perform against its named opponent?",
            "rationale": "A horizontal bar supports direct comparison of long matchup labels.",
            "dataset": "matchups",
            "sourceId": NORMALIZED_SOURCE_ID,
            "encodings": {
                "x": {"field": "matchup", "type": "nominal", "label": "Matchup"},
                "y": {
                    "field": "win_rate",
                    "type": "quantitative",
                    "label": "Win rate",
                    "format": "percent",
                },
                "tooltip": [
                    {"field": "games", "type": "quantitative", "label": "Games"},
                    {
                        "field": "ci95_low",
                        "type": "quantitative",
                        "label": "95% CI low",
                        "format": "percent",
                    },
                    {
                        "field": "ci95_high",
                        "type": "quantitative",
                        "label": "95% CI high",
                        "format": "percent",
                    },
                    {
                        "field": "elapsed_seconds",
                        "type": "quantitative",
                        "label": "Wall time",
                        "unit": "s",
                    },
                ],
            },
            "yAxisTitle": "Win rate",
            "valueFormat": "percent",
            "layout": "full",
            "palette": {"kind": "sequential", "name": "blue"},
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 0.5,
                    "label": "Even",
                    "color": "neutral",
                    "lineStyle": "dashed",
                }
            ],
            "settings": {"orientation": "horizontal", "sort": "descending"},
        },
        {
            "id": "crossplay_matrix",
            "title": "Observed cross-play win-rate matrix",
            "subtitle": "Only observed cells are shown; the largest-game run represents repeated model pairs.",
            "type": "heatmap",
            "intent": "relationship",
            "question": "Which observed model pairs favor which side?",
            "rationale": "A heatmap makes the sparse pairwise schedule and direction visible.",
            "dataset": "crossplay",
            "sourceId": NORMALIZED_SOURCE_ID,
            "encodings": {
                "x": {"field": "opponent", "type": "nominal", "label": "Opponent"},
                "y": {"field": "model", "type": "nominal", "label": "Model"},
                "color": {
                    "field": "win_rate",
                    "type": "quantitative",
                    "label": "Win rate",
                    "format": "percent",
                },
                "tooltip": [
                    {"field": "games", "type": "quantitative", "label": "Games"},
                    {
                        "field": "ci95_low",
                        "type": "quantitative",
                        "label": "95% CI low",
                        "format": "percent",
                    },
                    {
                        "field": "ci95_high",
                        "type": "quantitative",
                        "label": "95% CI high",
                        "format": "percent",
                    },
                ],
            },
            "valueFormat": "percent",
            "layout": "full",
            "palette": {"kind": "sequential", "name": "blue"},
        },
        {
            "id": "ratings",
            "title": "Bradley–Terry ratings from observed pairings",
            "subtitle": "Descriptive only: several model pairs were never evaluated directly.",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "What ranking is implied by the observed pairwise results?",
            "rationale": "A ranked horizontal bar shows rating order while retaining schedule volume.",
            "dataset": "ratings",
            "sourceId": NORMALIZED_SOURCE_ID,
            "encodings": {
                "x": {"field": "model", "type": "nominal", "label": "Model"},
                "y": {
                    "field": "rating",
                    "type": "quantitative",
                    "label": "Rating",
                    "format": "number",
                },
                "tooltip": [
                    {
                        "field": "connected_games",
                        "type": "quantitative",
                        "label": "Connected games",
                    },
                    {
                        "field": "bt_strength",
                        "type": "quantitative",
                        "label": "BT strength",
                    },
                ],
            },
            "layout": "full",
            "palette": {"kind": "sequential", "name": "blue"},
            "settings": {"orientation": "horizontal", "sort": "descending"},
        },
        {
            "id": "seat_bias",
            "title": "Seat-0 minus seat-1 win-rate difference",
            "subtitle": "Percentage-point difference for the named evaluated agent; positive values favor seat 0.",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "Do evaluated agents perform differently by seat?",
            "rationale": "Signed bars around zero expose direction and magnitude.",
            "dataset": "matchups",
            "sourceId": NORMALIZED_SOURCE_ID,
            "encodings": {
                "x": {"field": "matchup", "type": "nominal", "label": "Matchup"},
                "y": {
                    "field": "seat_bias_pp",
                    "type": "quantitative",
                    "label": "Seat bias",
                    "unit": "pp",
                },
                "tooltip": [
                    {
                        "field": "seat_0_win_rate",
                        "type": "quantitative",
                        "label": "Seat 0 win rate",
                        "format": "percent",
                    },
                    {
                        "field": "seat_1_win_rate",
                        "type": "quantitative",
                        "label": "Seat 1 win rate",
                        "format": "percent",
                    },
                    {"field": "games", "type": "quantitative", "label": "Games"},
                ],
            },
            "unit": "pp",
            "layout": "full",
            "palette": {"kind": "diverging", "name": "blue-orange", "midpoint": 0},
            "referenceLines": [
                {
                    "axis": "y",
                    "value": 0,
                    "label": "No difference",
                    "color": "neutral",
                    "lineStyle": "solid",
                }
            ],
            "settings": {"orientation": "horizontal", "sort": "descending"},
        },
        {
            "id": "training_throughput_chart",
            "title": "Collected policy decisions per second",
            "subtitle": "Current vectorized-recipe runs; collected throughput uses rollout seconds, while total time remains available in tooltips.",
            "type": "horizontalBar",
            "intent": "comparison",
            "question": "Which measured training configuration collected decisions fastest?",
            "rationale": "Long configuration names and one comparable throughput measure favor horizontal bars.",
            "dataset": "training_throughput",
            "sourceId": NORMALIZED_SOURCE_ID,
            "encodings": {
                "x": {
                    "field": "configuration",
                    "type": "nominal",
                    "label": "Configuration",
                },
                "y": {
                    "field": "collected_policy_decisions_per_second",
                    "type": "quantitative",
                    "label": "Collected policy decisions/s",
                    "format": "compact",
                },
                "tooltip": [
                    {"field": "stage", "type": "nominal", "label": "Stage"},
                    {"field": "status", "type": "nominal", "label": "Run status"},
                    {
                        "field": "games_per_second",
                        "type": "quantitative",
                        "label": "Games/s",
                    },
                    {
                        "field": "total_seconds",
                        "type": "quantitative",
                        "label": "Total time",
                        "unit": "s",
                    },
                ],
            },
            "layout": "full",
            "palette": {"kind": "sequential", "name": "blue"},
            "settings": {"orientation": "horizontal", "sort": "descending"},
        },
        {
            "id": "seed_variance_chart",
            "title": "Final-recipe throughput across seeds",
            "subtitle": "Warm sequential and vectorized runs across three seeds; collected throughput is rollout-only.",
            "type": "bar",
            "intent": "distribution",
            "question": "How much does final-recipe throughput vary across recorded seeds?",
            "rationale": "Three discrete seed measurements are clearer as bars than as an underpowered distribution plot.",
            "dataset": "seed_variance",
            "sourceId": NORMALIZED_SOURCE_ID,
            "encodings": {
                "x": {"field": "seed", "type": "nominal", "label": "Seed"},
                "y": {
                    "field": "collected_policy_decisions_per_second",
                    "type": "quantitative",
                    "label": "Collected policy decisions/s",
                    "format": "compact",
                },
                "color": {
                    "field": "implementation",
                    "type": "nominal",
                    "label": "Implementation",
                },
                "tooltip": [
                    {
                        "field": "games_per_second",
                        "type": "quantitative",
                        "label": "Games/s",
                    },
                    {
                        "field": "total_seconds",
                        "type": "quantitative",
                        "label": "Total time",
                        "unit": "s",
                    },
                    {
                        "field": "policy_decisions",
                        "type": "quantitative",
                        "label": "Policy decisions",
                    },
                ],
            },
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "title": "Implementation"},
            "settings": {"groupMode": "grouped"},
        },
        {
            "id": "time_breakdown_chart",
            "title": "Final-recipe wall-time composition",
            "subtitle": "Rollout, preparation, and update seconds for current vectorized-recipe runs, plus legacy final rows.",
            "type": "stackedBar",
            "intent": "composition",
            "question": "Where is final-recipe training time spent?",
            "rationale": "Stacked bars compare total wall time and its mutually exclusive phases.",
            "dataset": "time_breakdown",
            "sourceId": NORMALIZED_SOURCE_ID,
            "encodings": {
                "x": {
                    "field": "configuration",
                    "type": "nominal",
                    "label": "Configuration",
                },
                "y": {
                    "field": "seconds",
                    "type": "quantitative",
                    "label": "Seconds",
                },
                "color": {"field": "phase", "type": "nominal", "label": "Phase"},
                "tooltip": [
                    {
                        "field": "total_seconds",
                        "type": "quantitative",
                        "label": "Total time",
                        "unit": "s",
                    }
                ],
            },
            "unit": "s",
            "layout": "full",
            "palette": {"kind": "categorical", "name": "blue-orange"},
            "legend": {"position": "bottom", "title": "Phase"},
            "settings": {"groupMode": "stacked"},
        },
    ]

    # The current native heatmap validator requires a numeric y encoding, so a
    # categorical-by-categorical cross-play matrix is rendered as an exact
    # lookup table rather than reimplemented with custom HTML.
    charts = [chart for chart in charts if chart["id"] != "crossplay_matrix"]

    tables = [
        {
            "id": "matchup_table",
            "title": "Tournament evidence and 95% confidence intervals",
            "subtitle": "Each row remains a separate source run; repeated pairs are not silently pooled.",
            "dataset": "matchups",
            "sourceId": NORMALIZED_SOURCE_ID,
            "defaultSort": {"field": "games", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "matchup", "label": "Matchup", "type": "text"},
                {"field": "player_count", "label": "Players", "type": "number"},
                {"field": "games", "label": "Games", "format": "number"},
                {"field": "win_rate", "label": "Win rate", "format": "percent"},
                {"field": "ci95_low", "label": "95% low", "format": "percent"},
                {"field": "ci95_high", "label": "95% high", "format": "percent"},
                {"field": "confidence_vs_even", "label": "Vs 50%", "type": "text"},
                {"field": "elapsed_seconds", "label": "Wall time (s)", "format": "number"},
                {"field": "environment_turns", "label": "Env turns", "format": "compact"},
                {"field": "seed", "label": "Seed", "format": "number"},
            ],
        },
        {
            "id": "crossplay_table",
            "title": "Observed cross-play win-rate matrix",
            "subtitle": "Blank cells were not evaluated; the largest-game run represents repeated model pairs.",
            "dataset": "crossplay_matrix",
            "sourceId": NORMALIZED_SOURCE_ID,
            "defaultSort": {"field": "model", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": crossplay_columns,
        },
        {
            "id": "training_table",
            "title": "Training and rollout performance detail",
            "subtitle": "Exact normalized metrics; blank cells were not reported or cannot be derived safely.",
            "dataset": "training_runs",
            "sourceId": NORMALIZED_SOURCE_ID,
            "defaultSort": {
                "field": "policy_decisions_per_second",
                "direction": "desc",
            },
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "configuration", "label": "Configuration", "type": "text"},
                {"field": "stage", "label": "Stage", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "player_count", "label": "Players", "type": "number"},
                {"field": "seed", "label": "Seed", "format": "number"},
                {"field": "games", "label": "Games", "format": "number"},
                {"field": "games_per_second", "label": "Games/s", "format": "number"},
                {
                    "field": "environment_turns_per_second",
                    "label": "End-to-end env turns/s",
                    "format": "number",
                },
                {
                    "field": "rollout_environment_turns_per_second",
                    "label": "Rollout env turns/s",
                    "format": "number",
                },
                {
                    "field": "policy_decisions_per_second",
                    "label": "End-to-end policy decisions/s",
                    "format": "number",
                },
                {
                    "field": "collected_policy_decisions_per_second",
                    "label": "Collected policy decisions/s",
                    "format": "number",
                },
                {"field": "update_seconds", "label": "Update (s)", "format": "number"},
                {"field": "total_seconds", "label": "Total (s)", "format": "number"},
                {
                    "field": "peak_cpu_memory_mib",
                    "label": "Peak RSS MiB",
                    "format": "number",
                },
                {
                    "field": "peak_gpu_memory_mib",
                    "label": "Peak GPU MiB",
                    "format": "number",
                },
            ],
        },
        {
            "id": "coverage_table",
            "title": "View and guardrail coverage",
            "subtitle": "No-data states are explicit so missing evidence is not mistaken for a zero.",
            "dataset": "coverage",
            "sourceId": NORMALIZED_SOURCE_ID,
            "defaultSort": {"field": "view", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "view", "label": "Requested view", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "evidence", "label": "Evidence / requirement", "type": "text"},
            ],
        },
        {
            "id": "guardrail_table",
            "title": "Correctness and information-safety guardrails",
            "subtitle": "Observed safeguards are not promoted to pass/fail without an explicit source verdict.",
            "dataset": "guardrails",
            "sourceId": NORMALIZED_SOURCE_ID,
            "defaultSort": {"field": "guardrail", "direction": "asc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "guardrail", "label": "Guardrail", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "run_id", "label": "Run", "type": "text"},
                {"field": "evidence", "label": "Evidence", "type": "text"},
            ],
        },
        {
            "id": "provenance_table",
            "title": "Run provenance and reproducibility fields",
            "subtitle": "Source hashes are recorded even where checkpoint, recipe, observation version, or benchmark git SHA is missing.",
            "dataset": "provenance",
            "sourceId": NORMALIZED_SOURCE_ID,
            "defaultSort": {"field": "recorded_at", "direction": "desc"},
            "density": "dense",
            "layout": "full",
            "columns": [
                {"field": "run_id", "label": "Run ID", "type": "text"},
                {"field": "kind", "label": "Kind", "type": "text"},
                {"field": "status", "label": "Status", "type": "text"},
                {"field": "recorded_at", "label": "Recorded", "type": "date"},
                {"field": "player_count", "label": "Players", "type": "number"},
                {"field": "seed", "label": "Seed", "format": "number"},
                {"field": "checkpoint_hash", "label": "Checkpoint SHA", "type": "text"},
                {"field": "observation_version", "label": "Observation", "type": "text"},
                {"field": "benchmark_git_sha", "label": "Benchmark git SHA", "type": "text"},
                {"field": "recipe", "label": "Recipe", "type": "text"},
                {"field": "hardware", "label": "Hardware", "type": "text"},
                {"field": "source_path", "label": "Source", "type": "text"},
                {"field": "source_sha256", "label": "Source SHA-256", "type": "text"},
            ],
        },
    ]

    manifest = {
        "version": 1,
        "surface": "dashboard",
        "title": "Pesten Benchmark Dashboard",
        "description": "Source-backed model, tournament, training-performance, guardrail, and reproducibility dashboard.",
        "generatedAt": generated_at,
        "filters": [
            {
                "id": "player_count",
                "label": "Players",
                "dataset": "matchups",
                "field": "player_count",
                "includeAll": True,
                "targets": [
                    {"dataset": "crossplay", "field": "player_count"},
                    {"dataset": "crossplay_matrix", "field": "player_count"},
                    {"dataset": "ratings", "field": "player_count"},
                    {"dataset": "training_runs", "field": "player_count"},
                    {"dataset": "training_throughput", "field": "player_count"},
                    {"dataset": "seed_variance", "field": "player_count"},
                    {"dataset": "time_breakdown", "field": "player_count"},
                ],
            }
        ],
        "cards": cards,
        "charts": charts,
        "tables": tables,
        "sources": sources,
        "blocks": [
            {
                "id": "overview_note",
                "type": "markdown",
                "body": (
                    "## Champion and time-to-beat\n\n"
                    "No current input explicitly designates a champion, so the dashboard does not "
                    "turn the descriptive Bradley–Terry leader into a champion. A time-to-beat is "
                    "reported only when future normalized data marks both champion and candidate "
                    "roles and the candidate's 95% confidence interval is entirely above 50%."
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": [
                    "champion",
                    "challenge",
                    "evaluation_time",
                    "evaluation_turns",
                    "training_throughput",
                    "training_time",
                ],
            },
            {
                "id": "tournament_note",
                "type": "markdown",
                "body": (
                    "## Tournament evidence\n\n"
                    "Point estimates, Wilson intervals, observed cross-play, schedule-aware ratings, "
                    "and seat effects are shown only for source-backed tournament rows."
                ),
            },
            {"id": "matchup_chart_block", "type": "chart", "chartId": "matchup_win_rates"},
            {"id": "matchup_table_block", "type": "table", "tableId": "matchup_table", "layout": "full"},
            {"id": "crossplay_block", "type": "table", "tableId": "crossplay_table", "layout": "full"},
            {"id": "ratings_block", "type": "chart", "chartId": "ratings"},
            {"id": "seat_bias_block", "type": "chart", "chartId": "seat_bias"},
            {
                "id": "learning_curve_no_data",
                "type": "markdown",
                "body": (
                    "## Learning curves\n\n"
                    "**No data.** Current inputs contain aggregate speed and tournament results, "
                    "not checkpoint metrics indexed by wall time or environment decisions. The "
                    "normalized schema includes `learning_curve` for future runs."
                ),
            },
            {
                "id": "training_note",
                "type": "markdown",
                "body": (
                    "## Training throughput and time\n\n"
                    "Current vectorized runs separate rollout collection throughput from end-to-end "
                    "throughput and report environment turns. The legacy file's samples map to "
                    "collected policy decisions, while its environment turns remain blank. "
                    "Baseline and interim caveats remain visible in detail."
                ),
            },
            {
                "id": "training_throughput_block",
                "type": "chart",
                "chartId": "training_throughput_chart",
            },
            {"id": "seed_variance_block", "type": "chart", "chartId": "seed_variance_chart"},
            {"id": "time_breakdown_block", "type": "chart", "chartId": "time_breakdown_chart"},
            {"id": "training_table_block", "type": "table", "tableId": "training_table", "layout": "full"},
            {
                "id": "resource_no_data",
                "type": "markdown",
                "body": (
                    "## Resources\n\n"
                    "**Partial data.** Vectorized runs report RSS peaks and CUDA allocation peaks; "
                    "the original CUDA baseline also reports a CUDA peak. CPU/GPU utilization and "
                    "utilization time series are absent, so no utilization chart is rendered."
                ),
            },
            {"id": "coverage_block", "type": "table", "tableId": "coverage_table", "layout": "full"},
            {
                "id": "guardrail_note",
                "type": "markdown",
                "body": (
                    "## Correctness and information safety\n\n"
                    "Source-described ratio checks and smoke-test configuration checks are preserved "
                    "with their original verdict strength. No illegal-action result or complete "
                    "benchmark-level information-leakage result is present; absence is not a pass."
                ),
            },
            {"id": "guardrail_block", "type": "table", "tableId": "guardrail_table", "layout": "full"},
            {
                "id": "provenance_note",
                "type": "markdown",
                "body": (
                    "## Reproducibility and provenance\n\n"
                    "Every normalized row records its source path and SHA-256. Checkpoint hashes, "
                    "recipes, seeds, hardware, observation versions, and benchmark git SHAs remain "
                    "explicitly missing when the source does not supply them."
                ),
            },
            {"id": "provenance_block", "type": "table", "tableId": "provenance_table", "layout": "full"},
        ],
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline": headline,
            "matchups": matchups,
            "crossplay": crossplay,
            "crossplay_matrix": crossplay_matrix,
            "ratings": ratings,
            "training_runs": training,
            "training_throughput": throughput_summary,
            "seed_variance": seed_variance,
            "time_breakdown": time_breakdown,
            "coverage": coverage,
            "guardrails": guardrails,
            "provenance": provenance,
        },
    }
    return {
        "surface": "dashboard",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }
