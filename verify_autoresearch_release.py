#!/usr/bin/env python3
"""Read-only verification of the frozen 2026-07-31 autoresearch release.

This module deliberately uses only recorded JSON, hashes, and Git metadata. It
does not import a game engine, player implementation, or benchmark evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Iterable


MAX_RAW_BYTES = 20 * 1024 * 1024
MAX_BOOTSTRAP_INDEX_BYTES = 16 * 1024 * 1024
SEALED_SEED_START = 4_294_000_000
SEALED_SEED_END = 4_294_900_000
SUMMARY_PATH = Path("benchmarks/autoresearch_summary.json")
REPORT_PATH = Path("AUTORESEARCH_20260731.md")


class VerificationError(RuntimeError):
    """A release artifact or cross-file binding is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, max_bytes: int) -> object:
    try:
        size = path.stat().st_size
    except FileNotFoundError as error:
        raise VerificationError(f"required artifact is missing: {path}") from error
    _require(
        size <= max_bytes,
        f"refusing oversized JSON artifact {path}: {size} > {max_bytes} bytes",
    )
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise VerificationError(f"invalid JSON artifact {path}: {error}") from error


def _resolve_inside(root: Path, relative_path: str, *, label: str) -> Path:
    candidate = Path(relative_path)
    _require(not candidate.is_absolute(), f"{label} path must be relative")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise VerificationError(f"{label} path escapes repository root") from error
    return resolved


def _close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    _require(bool(materialized), "cannot summarize an empty record set")
    return math.fsum(materialized) / len(materialized)


def _bootstrap_mean_interval(
    np,
    values: list[float],
    *,
    seed: int,
    samples: int,
    max_index_bytes: int = MAX_BOOTSTRAP_INDEX_BYTES,
) -> list[float]:
    """Replay NumPy's bootstrap RNG stream in bounded row chunks."""
    array = np.asarray(values, dtype=np.float64)
    _require(len(array) > 0, "cannot bootstrap an empty record set")
    _require(samples > 0, "bootstrap sample count must be positive")
    _require(max_index_bytes >= 8 * len(array), "bootstrap memory bound is too small")
    rows_per_chunk = max(1, max_index_bytes // (8 * len(array)))
    generator = np.random.default_rng(seed)
    means = []
    remaining = samples
    while remaining:
        rows = min(remaining, rows_per_chunk)
        indices = generator.integers(
            0,
            len(array),
            size=(rows, len(array)),
        )
        means.append(array[indices].mean(axis=1))
        remaining -= rows
    bootstrap_means = np.concatenate(means)
    tail = (1.0 - 0.95) / 2.0
    return [
        float(value)
        for value in np.quantile(
            bootstrap_means,
            (tail, 1.0 - tail),
        )
    ]


def _verify_pair(
    pair: dict,
    *,
    pair_index: int,
    deal_seed: int,
    max_turns: int,
    label: str,
) -> tuple[int, int]:
    _require(pair.get("pair_index") == pair_index, f"{label}: pair_index mismatch")
    _require(pair.get("deal_seed") == deal_seed, f"{label}: deal_seed mismatch")
    games = pair.get("games")
    _require(isinstance(games, list) and len(games) == 2, f"{label}: expected two games")
    scores = []
    turns = 0
    for evaluated_seat, game in enumerate(games):
        game_label = f"{label}.games[{evaluated_seat}]"
        _require(
            game.get("evaluated_seat") == evaluated_seat,
            f"{game_label}: rotated evaluated seat mismatch",
        )
        winner = game.get("winner")
        _require(winner in (0, 1, None), f"{game_label}: invalid winner")
        expected_result = (
            "draw"
            if winner is None
            else ("win" if winner == evaluated_seat else "loss")
        )
        expected_score = (
            0.5
            if winner is None
            else (1.0 if winner == evaluated_seat else 0.0)
        )
        _require(
            game.get("result") == expected_result,
            f"{game_label}: result is inconsistent with winner",
        )
        _require(
            game.get("score") == expected_score,
            f"{game_label}: score is inconsistent with winner",
        )
        game_turns = game.get("turns")
        _require(
            isinstance(game_turns, int)
            and not isinstance(game_turns, bool)
            and 1 <= game_turns <= max_turns,
            f"{game_label}: invalid turn count",
        )
        scores.append(expected_score)
        turns += game_turns
    _require(
        _close(pair.get("score"), _mean(scores)),
        f"{label}: pair score mismatch",
    )
    return len(games), turns


def _verify_git_release(root: Path, paths: list[Path]) -> None:
    git_dir = root / ".git"
    _require(git_dir.exists(), "Git tracking verification requested outside a clone")
    relative_paths = [str(path.relative_to(root)) for path in paths]
    tracked = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", *relative_paths],
        capture_output=True,
        text=True,
        check=False,
    )
    _require(
        tracked.returncode == 0,
        "release artifact is not tracked by Git: " + tracked.stderr.strip(),
    )
    clean = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", *relative_paths],
        check=False,
    )
    _require(
        clean.returncode == 0,
        "release artifacts differ from the checked-out commit",
    )


def verify_release(
    root: Path,
    *,
    require_git: bool = True,
    verify_bootstrap_cis: bool = False,
) -> dict:
    """Verify recorded release artifacts without executing games."""
    root = root.resolve()
    summary_path = root / SUMMARY_PATH
    report_path = root / REPORT_PATH
    summary = _load_json(summary_path, max_bytes=1024 * 1024)
    _require(isinstance(summary, dict), "summary root must be an object")
    sealed_summary = summary.get("sealed_evaluation")
    _require(isinstance(sealed_summary, dict), "summary lacks sealed_evaluation")

    raw_path = _resolve_inside(
        root, sealed_summary.get("raw_result", ""), label="raw result"
    )
    ledger_path = _resolve_inside(
        root, sealed_summary.get("claim_ledger", ""), label="claim ledger"
    )
    selected = summary.get("selected_player")
    _require(isinstance(selected, dict), "summary lacks selected_player")
    champion_path = _resolve_inside(
        root, selected.get("checkpoint", ""), label="champion"
    )
    entrypoint_path = _resolve_inside(
        root, selected.get("entrypoint", ""), label="player entrypoint"
    )

    raw = _load_json(raw_path, max_bytes=MAX_RAW_BYTES)
    ledger = _load_json(ledger_path, max_bytes=1024 * 1024)
    _require(isinstance(raw, dict), "raw result root must be an object")
    _require(isinstance(ledger, dict), "claim ledger root must be an object")
    try:
        report = report_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise VerificationError(f"required report is missing: {report_path}") from error

    raw_sha = _sha256(raw_path)
    ledger_sha = _sha256(ledger_path)
    champion_sha = _sha256(champion_path)
    summary_sha = _sha256(summary_path)
    report_sha = _sha256(report_path)
    _require(
        raw_sha == sealed_summary.get("raw_result_sha256"),
        "raw result SHA-256 does not match summary",
    )
    _require(
        ledger_sha == sealed_summary.get("claim_ledger_sha256"),
        "claim ledger SHA-256 does not match summary",
    )
    _require(
        champion_sha == selected.get("checkpoint_sha256"),
        "champion SHA-256 does not match summary",
    )
    for label, digest in (
        ("raw result", raw_sha),
        ("claim ledger", ledger_sha),
        ("champion", champion_sha),
    ):
        _require(digest in report, f"report does not bind {label} SHA-256")

    holdout = raw.get("sealed_holdout")
    _require(isinstance(holdout, dict), "raw result lacks sealed_holdout")
    _require(
        holdout.get("global_claim") == ledger,
        "embedded global claim differs from ledger",
    )
    _require(
        ledger.get("status") == "holdout_burned_before_evaluation",
        "ledger does not record pre-evaluation holdout burn",
    )
    _require(
        ledger.get("evaluator") == "suit_symmetry_projection",
        "unexpected sealed evaluator",
    )
    _require(
        ledger.get("candidate", {}).get("sha256") == champion_sha,
        "ledger candidate SHA-256 mismatch",
    )
    _require(
        Path(ledger.get("candidate", {}).get("path", "")).name
        == champion_path.name,
        "ledger candidate path does not name selected champion",
    )
    _require(
        Path(ledger.get("output_path", "")).name == raw_path.name,
        "ledger output path does not name tracked raw result",
    )
    _require(
        Path(ledger.get("ledger_path", "")).name == ledger_path.name,
        "ledger self-path does not name tracked claim ledger",
    )

    interval = ledger.get("seed_interval")
    _require(
        interval
        == {
            "start_inclusive": SEALED_SEED_START,
            "end_exclusive": SEALED_SEED_END,
        },
        "ledger seed interval is not the active sealed namespace",
    )
    _require(
        holdout.get("interval_start_inclusive") == SEALED_SEED_START
        and holdout.get("interval_end_exclusive") == SEALED_SEED_END,
        "raw holdout interval differs from ledger",
    )
    _require(
        bool(holdout.get("one_way"))
        and bool(holdout.get("confirmed"))
        and bool(sealed_summary.get("one_way")),
        "one-way sealed confirmation flags are incomplete",
    )
    base_seed = ledger.get("base_seed")
    pairs = ledger.get("pairs")
    _require(
        isinstance(base_seed, int)
        and SEALED_SEED_START <= base_seed < SEALED_SEED_END,
        "invalid sealed base seed",
    )
    _require(
        isinstance(pairs, int) and not isinstance(pairs, bool) and pairs > 0,
        "invalid sealed pair count",
    )
    _require(
        raw.get("base_seed") == base_seed
        == sealed_summary.get("base_seed"),
        "base seed mismatch across artifacts",
    )
    _require(
        raw.get("pairs_per_matchup") == pairs
        == sealed_summary.get("pairs_per_matchup"),
        "pair count mismatch across artifacts",
    )
    _require(
        holdout.get("bootstrap_seed_not_used_for_games")
        == base_seed - SEALED_SEED_START,
        "bootstrap seed metadata mismatch",
    )

    configuration = ledger.get("configuration")
    candidate = raw.get("candidate")
    projection = selected.get("projection")
    _require(isinstance(configuration, dict), "ledger configuration is invalid")
    _require(isinstance(candidate, dict), "raw candidate configuration is invalid")
    _require(isinstance(projection, dict), "selected projection is invalid")
    _require(
        candidate.get("wrapper") == "BatchedSuitAveragedActor"
        and candidate.get("ordinary_suit_permutations") == 24
        and candidate.get("ordinary_suit_permutations")
        == projection.get("ordinary_suit_permutations"),
        "suit projection metadata mismatch",
    )
    for key in ("alpha", "aggregation", "player_orbit_average"):
        _require(
            candidate.get(key) == configuration.get(key) == projection.get(key),
            f"projection setting mismatch: {key}",
        )
    _require(
        candidate.get("base_chunk_size") == configuration.get("base_chunk_size"),
        "recorded chunk size mismatch",
    )
    _require(
        raw.get("human_visible_information_only") is True,
        "raw result lacks human-visible-information-only assertion",
    )
    information_boundary = raw.get("information_boundary", {})
    _require(
        information_boundary.get("opponent_card_identities") is False
        and information_boundary.get("draw_pile_inspection") is False,
        "raw information boundary permits hidden information",
    )

    pool_names = configuration.get("pool")
    _require(
        isinstance(pool_names, list)
        and pool_names
        and all(isinstance(name, str) and name for name in pool_names),
        "invalid fixed opponent pool",
    )
    max_turns = configuration.get("max_turns")
    _require(
        isinstance(max_turns, int) and max_turns > 0,
        "invalid max_turns configuration",
    )
    namespace_names = ["direct", *(f"pool:{name}" for name in pool_names)]
    plan = holdout.get("collision_free_namespace_plan")
    _require(isinstance(plan, dict), "missing collision-free namespace plan")
    _require(set(plan) == set(namespace_names), "namespace plan names mismatch")

    all_seeds: set[int] = set()
    total_games = 0
    total_turns = 0
    direct = raw.get("direct")
    _require(isinstance(direct, dict), "raw result lacks direct section")
    direct_pairs = direct.get("per_pair")
    _require(
        isinstance(direct_pairs, list) and len(direct_pairs) == pairs,
        "direct record count mismatch",
    )
    direct_scores = []
    for pair_index, pair in enumerate(direct_pairs):
        deal_seed = base_seed + pair_index
        _require(deal_seed not in all_seeds, "duplicate direct deal seed")
        all_seeds.add(deal_seed)
        games, turns = _verify_pair(
            pair,
            pair_index=pair_index,
            deal_seed=deal_seed,
            max_turns=max_turns,
            label=f"direct[{pair_index}]",
        )
        total_games += games
        total_turns += turns
        direct_scores.append(pair["score"])

    direct_score = _mean(direct_scores)
    direct_advantage = _mean(2.0 * score - 1.0 for score in direct_scores)
    _require(
        _close(direct_score, direct.get("candidate_score", {}).get("mean")),
        "recomputed direct candidate score mismatch",
    )
    _require(
        _close(
            direct_advantage,
            direct.get("candidate_advantage", {}).get("mean"),
        ),
        "recomputed direct advantage mismatch",
    )

    pool = raw.get("pool")
    _require(
        isinstance(pool, list)
        and [matchup.get("opponent") for matchup in pool] == pool_names,
        "raw pool order or names mismatch",
    )
    opponent_deltas = {}
    combined_deltas = []
    for namespace_index, (opponent, matchup) in enumerate(
        zip(pool_names, pool), start=1
    ):
        records = matchup.get("per_pair")
        _require(
            isinstance(records, list) and len(records) == pairs,
            f"{opponent}: pool record count mismatch",
        )
        candidate_scores = []
        parent_scores = []
        deltas = []
        namespace_start = base_seed + namespace_index * pairs
        for pair_index, record in enumerate(records):
            label = f"pool:{opponent}[{pair_index}]"
            deal_seed = namespace_start + pair_index
            _require(record.get("pair_index") == pair_index, f"{label}: pair_index mismatch")
            _require(record.get("deal_seed") == deal_seed, f"{label}: deal_seed mismatch")
            _require(deal_seed not in all_seeds, f"{label}: duplicate deal seed")
            all_seeds.add(deal_seed)
            for evaluated_key in ("candidate", "parent"):
                games, turns = _verify_pair(
                    record.get(evaluated_key, {}),
                    pair_index=pair_index,
                    deal_seed=deal_seed,
                    max_turns=max_turns,
                    label=f"{label}.{evaluated_key}",
                )
                total_games += games
                total_turns += turns
            candidate_score = record["candidate"]["score"]
            parent_score = record["parent"]["score"]
            delta = candidate_score - parent_score
            _require(_close(delta, record.get("delta")), f"{label}: delta mismatch")
            candidate_scores.append(candidate_score)
            parent_scores.append(parent_score)
            deltas.append(delta)
            combined_deltas.append(delta)
        _require(
            _close(_mean(candidate_scores), matchup.get("candidate_score", {}).get("mean")),
            f"{opponent}: recomputed candidate score mismatch",
        )
        _require(
            _close(_mean(parent_scores), matchup.get("parent_score", {}).get("mean")),
            f"{opponent}: recomputed parent score mismatch",
        )
        delta_mean = _mean(deltas)
        _require(
            _close(delta_mean, matchup.get("paired_delta", {}).get("mean")),
            f"{opponent}: recomputed paired delta mismatch",
        )
        opponent_deltas[opponent] = delta_mean

    expected_seed_count = pairs * len(namespace_names)
    _require(
        len(all_seeds) == expected_seed_count,
        "deal seeds are not globally unique across namespaces",
    )
    _require(
        min(all_seeds) >= SEALED_SEED_START
        and max(all_seeds) < SEALED_SEED_END,
        "recorded game seed is outside active sealed interval",
    )
    for namespace_index, namespace in enumerate(namespace_names):
        start = base_seed + namespace_index * pairs
        _require(
            plan.get(namespace)
            == {
                "count": pairs,
                "start_inclusive": start,
                "end_exclusive": start + pairs,
            },
            f"namespace plan mismatch: {namespace}",
        )

    combined_delta = _mean(combined_deltas)
    _require(
        _close(combined_delta, raw.get("combined_pool_delta", {}).get("mean")),
        "recomputed combined pool delta mismatch",
    )
    _require(
        _close(
            direct_score,
            sealed_summary.get("direct_candidate_score", {}).get("mean"),
        )
        and _close(
            direct_advantage,
            sealed_summary.get("direct_advantage", {}).get("mean"),
        )
        and _close(
            combined_delta,
            sealed_summary.get("combined_pool_delta", {}).get("mean"),
        ),
        "summary headline means differ from recomputed records",
    )
    summary_opponents = sealed_summary.get("opponent_deltas", {})
    for opponent, delta_mean in opponent_deltas.items():
        _require(
            _close(delta_mean, summary_opponents.get(opponent, {}).get("mean")),
            f"summary opponent delta mismatch: {opponent}",
        )

    # Confidence intervals are copied byte-for-byte from the raw result into
    # the concise summary. Exact bootstrap replay is optional because it needs
    # NumPy's RNG; the default verifier remains stdlib-only.
    _require(
        sealed_summary.get("direct_candidate_score") == direct.get("candidate_score")
        and sealed_summary.get("direct_advantage") == direct.get("candidate_advantage")
        and sealed_summary.get("combined_pool_delta")
        == raw.get("combined_pool_delta")
        and summary_opponents
        == {
            matchup["opponent"]: matchup["paired_delta"]
            for matchup in pool
        },
        "summary statistics differ from raw result",
    )
    if verify_bootstrap_cis:
        _verify_bootstrap_cis(raw, configuration, base_seed)

    throughput = raw.get("throughput")
    _require(isinstance(throughput, dict), "raw result lacks throughput")
    _require(
        throughput.get("games") == total_games
        == sealed_summary.get("total_games"),
        "total game count mismatch",
    )
    _require(throughput.get("turns") == total_turns, "total turn count mismatch")
    elapsed = throughput.get("elapsed_seconds")
    _require(
        isinstance(elapsed, (int, float)) and elapsed > 0,
        "invalid elapsed time",
    )
    _require(
        _close(throughput.get("games_per_second"), total_games / elapsed)
        and _close(throughput.get("turns_per_second"), total_turns / elapsed),
        "throughput arithmetic mismatch",
    )
    _require(
        sealed_summary.get("throughput")
        == {
            "elapsed_seconds": elapsed,
            "games_per_second": throughput.get("games_per_second"),
        },
        "summary throughput differs from raw result",
    )

    actor_statistics = raw.get("actor_statistics")
    _require(isinstance(actor_statistics, dict), "missing actor statistics")
    _require(
        actor_statistics.get("augmented_observations")
        == 24 * actor_statistics.get("base_observations"),
        "24-way actor augmentation accounting mismatch",
    )
    _require(
        _close(
            actor_statistics.get("mean_base_batch_size"),
            actor_statistics.get("base_observations")
            / actor_statistics.get("forward_calls"),
        ),
        "actor mean batch arithmetic mismatch",
    )
    _require(
        _close(
            throughput.get("mean_actor_batch_size"),
            throughput.get("policy_decisions")
            / throughput.get("inference_calls"),
        ),
        "policy mean batch arithmetic mismatch",
    )

    tracked_paths = [
        champion_path,
        raw_path,
        ledger_path,
        summary_path,
        report_path,
        entrypoint_path,
    ]
    if require_git:
        _verify_git_release(root, tracked_paths)

    return {
        "status": "verified",
        "mode": "recorded_artifacts_only_no_game_execution",
        "pairs_per_matchup": pairs,
        "deal_seeds": len(all_seeds),
        "games": total_games,
        "turns": total_turns,
        "headline": {
            "direct_candidate_score": direct_score,
            "direct_advantage": direct_advantage,
            "combined_pool_delta": combined_delta,
            "opponent_deltas": opponent_deltas,
        },
        "sha256": {
            "champion": champion_sha,
            "raw_result": raw_sha,
            "claim_ledger": ledger_sha,
            "summary": summary_sha,
            "report": report_sha,
        },
        "git_tracking_verified": require_git,
        "bootstrap_cis_recomputed": verify_bootstrap_cis,
    }


def _verify_bootstrap_cis(raw: dict, configuration: dict, base_seed: int) -> None:
    try:
        import numpy as np
    except ImportError as error:
        raise VerificationError(
            "--verify-bootstrap-cis requires NumPy"
        ) from error

    bootstrap_samples = configuration.get("bootstrap_samples")
    _require(
        isinstance(bootstrap_samples, int) and bootstrap_samples > 0,
        "invalid bootstrap sample count",
    )
    bootstrap_base = base_seed - SEALED_SEED_START

    def derived_seed(namespace: str) -> int:
        material = f"{bootstrap_base}:{namespace}:0".encode("utf-8")
        return (
            int.from_bytes(hashlib.sha256(material).digest()[:4], "big")
            % 92_000_000
        )

    def interval(values: list[float], namespace: str) -> list[float]:
        return _bootstrap_mean_interval(
            np,
            values,
            seed=derived_seed(namespace),
            samples=bootstrap_samples,
        )

    checks = [
        (
            [pair["score"] for pair in raw["direct"]["per_pair"]],
            "direct_score",
            raw["direct"]["candidate_score"]["bootstrap_ci95"],
        ),
        (
            [
                2.0 * pair["score"] - 1.0
                for pair in raw["direct"]["per_pair"]
            ],
            "direct_advantage",
            raw["direct"]["candidate_advantage"]["bootstrap_ci95"],
        ),
    ]
    combined = []
    for matchup in raw["pool"]:
        opponent = matchup["opponent"]
        records = matchup["per_pair"]
        checks.extend(
            (
                (
                    [record["candidate"]["score"] for record in records],
                    f"candidate:{opponent}",
                    matchup["candidate_score"]["bootstrap_ci95"],
                ),
                (
                    [record["parent"]["score"] for record in records],
                    f"parent:{opponent}",
                    matchup["parent_score"]["bootstrap_ci95"],
                ),
                (
                    [record["delta"] for record in records],
                    f"delta:{opponent}",
                    matchup["paired_delta"]["bootstrap_ci95"],
                ),
            )
        )
        combined.extend(record["delta"] for record in records)
    checks.append(
        (
            combined,
            "combined_pool_delta",
            raw["combined_pool_delta"]["bootstrap_ci95"],
        )
    )
    for values, namespace, recorded in checks:
        recomputed = interval(values, namespace)
        _require(
            all(_close(left, right) for left, right in zip(recomputed, recorded)),
            f"bootstrap CI mismatch: {namespace}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="repository clone containing the frozen release",
    )
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="skip tracked-file and clean-HEAD checks (for archive verification)",
    )
    parser.add_argument(
        "--verify-bootstrap-cis",
        action="store_true",
        help="also replay every 20k-sample NumPy bootstrap interval",
    )
    args = parser.parse_args()
    try:
        result = verify_release(
            args.root,
            require_git=not args.skip_git,
            verify_bootstrap_cis=args.verify_bootstrap_cis,
        )
    except VerificationError as error:
        parser.exit(1, f"verification failed: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
