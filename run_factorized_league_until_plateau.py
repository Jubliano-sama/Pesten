#!/usr/bin/env python3
"""Train the factorized Puffer policy through diverse historical leagues.

Each round preserves 2-8 player coverage, rotates a decorrelated player-count
curriculum, adds the previous generation to a bounded frozen checkpoint pool,
and evaluates fresh progress against the fixed teacher, MLP, canonical random,
and retained historical generations. Training stops only after repeated blocks
fail a material-gain gate and a disjoint confirmation agrees.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from benchmark_rlvr_pool import OPPONENTS
from sota_distill_temporal import PLAYER_COUNT_SCHEDULE


LEAGUE_SCHEDULES = (
    (
        "current_selfplay",
        "current_selfplay",
        "historical_temporal",
        "historical_temporal",
        "historical_temporal",
        "ppo_champion",
        "canonical_random",
        "smart_rule",
        "smarter_rule",
        "current_selfplay",
        "historical_temporal",
        "ppo_champion",
        "canonical_random",
    ),
    (
        "historical_temporal",
        "current_selfplay",
        "ppo_champion",
        "historical_temporal",
        "smart_rule",
        "current_selfplay",
        "canonical_random",
        "historical_temporal",
        "smarter_rule",
        "ppo_champion",
        "current_selfplay",
        "historical_temporal",
        "canonical_random",
    ),
    (
        "current_selfplay",
        "historical_temporal",
        "canonical_random",
        "historical_temporal",
        "ppo_champion",
        "current_selfplay",
        "smarter_rule",
        "historical_temporal",
        "smart_rule",
        "current_selfplay",
        "ppo_champion",
        "historical_temporal",
        "canonical_random",
    ),
)


def _expanded_counts(counts: dict[int, int]) -> tuple[int, ...]:
    return tuple(
        players
        for players in range(2, 9)
        for _ in range(counts[players])
    )


PLAYER_COUNT_CURRICULA = (
    tuple(PLAYER_COUNT_SCHEDULE),
    _expanded_counts({2: 22, 3: 17, 4: 12, 5: 5, 6: 4, 7: 3, 8: 4}),
    _expanded_counts({2: 8, 3: 10, 4: 14, 5: 9, 6: 8, 7: 8, 8: 10}),
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(
            command,
            check=True,
            cwd=Path(__file__).resolve().parent,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )


def evaluate_pool(
    before: Path,
    after: Path,
    *,
    teacher: Path,
    mlp: Path,
    output: Path,
    blocks: int,
    workers: int,
    seed: int,
) -> dict:
    command = [
        sys.executable,
        "benchmark_rlvr_pool_batched.py",
        "--before",
        str(before),
        "--after",
        str(after),
        "--self-anchor",
        str(teacher),
        "--mlp",
        str(mlp),
        "--output",
        str(output),
        "--blocks-per-stratum",
        str(blocks),
        "--workers",
        str(workers),
        "--chunks-per-stratum",
        "6",
        "--seed",
        str(seed),
    ]
    run_logged(command, output.with_suffix(".log"))
    return json.loads(output.read_text(encoding="utf-8"))


def evaluate_temporal(
    candidate: Path,
    opponent: Path,
    *,
    mlp: Path,
    output: Path,
    blocks: int,
    workers: int,
    seed: int,
) -> dict:
    command = [
        sys.executable,
        "benchmark_temporal_1v1_batched.py",
        "--candidate",
        str(candidate),
        "--opponent",
        str(opponent),
        "--mlp",
        str(mlp),
        "--output",
        str(output),
        "--blocks",
        str(blocks),
        "--workers",
        str(workers),
        "--chunks",
        "24",
        "--seed",
        str(seed),
        "--noninferiority-margin",
        "0.02",
    ]
    run_logged(command, output.with_suffix(".log"))
    return json.loads(output.read_text(encoding="utf-8"))


def evaluate_rules(
    before: Path,
    after: Path,
    *,
    mlp: Path,
    output: Path,
    blocks: int,
    workers: int,
    seed: int,
) -> dict:
    command = [
        sys.executable,
        "benchmark_temporal_rule_pool_batched.py",
        "--before",
        str(before),
        "--after",
        str(after),
        "--mlp",
        str(mlp),
        "--output",
        str(output),
        "--blocks-per-stratum",
        str(blocks),
        "--workers",
        str(workers),
        "--chunks-per-stratum",
        "6",
        "--seed",
        str(seed),
    ]
    run_logged(command, output.with_suffix(".log"))
    return json.loads(output.read_text(encoding="utf-8"))


def _append_history(
    catalog: list[Path],
    checkpoint: Path,
) -> list[Path]:
    checkpoint = checkpoint.resolve()
    unique = [path for path in catalog if path.resolve() != checkpoint]
    unique.append(checkpoint)
    return unique


def _select_history(catalog: list[Path], *, limit: int) -> list[Path]:
    if limit < 1:
        raise ValueError("history limit must be positive")
    if len(catalog) <= limit:
        return list(catalog)
    if limit == 1:
        return [catalog[-1]]
    positions = [
        round(index * (len(catalog) - 1) / (limit - 1))
        for index in range(limit)
    ]
    return [catalog[position] for position in positions]


def _history_records(paths: list[Path]) -> list[dict]:
    return [
        {"path": str(path), "sha256": file_sha256(path)} for path in paths
    ]


def _plateau_screen(
    fixed_result: dict,
    rule_result: dict,
    *,
    materiality: float,
    history_noninferior: bool,
) -> tuple[bool, dict]:
    """Cheaply nominate a window for a powered, disjoint plateau test.

    The 300-block monitor is deliberately a screen, not the final confidence
    claim: at that sample size its 99.9% interval cannot resolve a one-point
    normalized-share margin.  Requiring both point estimates to stay below the
    materiality threshold avoids an impossible stopping condition while the
    later, larger confirmation retains the actual confidence gate.
    """
    fixed_estimate = float(fixed_result["delta_normalized_win_share"])
    rule_estimate = float(rule_result["delta_normalized_win_share"])
    passed = (
        fixed_estimate <= materiality
        and rule_estimate <= materiality
        and history_noninferior
    )
    return passed, {
        "kind": "low_cost_nomination_only",
        "fixed_pool_point_estimate": fixed_estimate,
        "rule_pool_point_estimate": rule_estimate,
        "maximum_material_gain_per_block": materiality,
        "historical_noninferiority_99_9": history_noninferior,
        "passed": passed,
        "confidence_claim_deferred_to_disjoint_confirmation": True,
    }


def _confirmation_passed(
    fixed_result: dict,
    rule_result: dict,
    *,
    maximum_gain: float,
) -> tuple[bool, dict]:
    fixed_upper = float(fixed_result["delta_normalized_ci99_9"][1])
    rule_upper = float(rule_result["delta_normalized_ci99_9"][1])
    passed = fixed_upper <= maximum_gain and rule_upper <= maximum_gain
    return passed, {
        "fixed_pool_upper_99_9": fixed_upper,
        "rule_pool_upper_99_9": rule_upper,
        "maximum_material_gain_over_window": maximum_gain,
        "passed": passed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--mlp", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-phases", type=int, default=0)
    parser.add_argument("--block-epochs", type=int, default=60)
    parser.add_argument("--games-per-epoch", type=int, default=4096)
    parser.add_argument(
        "--training-max-turns",
        type=int,
        default=500,
        help="training-only game cap; monitoring remains on its canonical limit",
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--history-limit", type=int, default=8)
    parser.add_argument(
        "--historical-checkpoint", action="append", type=Path, default=[]
    )
    parser.add_argument("--monitor-blocks-per-stratum", type=int, default=300)
    parser.add_argument("--history-check-blocks", type=int, default=10_000)
    parser.add_argument(
        "--confirmation-blocks-per-stratum", type=int, default=1200
    )
    parser.add_argument("--plateau-materiality", type=float, default=0.01)
    parser.add_argument("--plateau-consecutive-blocks", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--rollout-workers", type=int, default=10)
    parser.add_argument("--benchmark-workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20_261_300)
    return parser.parse_args()


def _same_path(left: str | Path, right: Path) -> bool:
    return Path(left).resolve() == right.resolve()


def _restore_campaign(
    manifest: dict,
    *,
    current: Path,
    teacher: Path,
    mlp: Path,
) -> tuple[Path, list[Path], int, int, Path | None]:
    """Restore the last accepted state while leaving partial rounds auditable."""
    if not _same_path(manifest["initial_checkpoint"], current):
        raise ValueError("campaign initial checkpoint does not match --current")
    if not _same_path(manifest["teacher"]["path"], teacher):
        raise ValueError("campaign teacher does not match --teacher")
    if not _same_path(manifest["mlp"]["path"], mlp):
        raise ValueError("campaign MLP does not match --mlp")
    restored = Path(manifest.get("current_checkpoint", current)).resolve()
    if not restored.is_file():
        raise FileNotFoundError(restored)

    catalog: list[Path] = []
    for record in manifest.get("initial_historical_pool", []):
        path = Path(record["path"]).resolve()
        if path.is_file():
            catalog = _append_history(catalog, path)
    for row in manifest.get("rounds", []):
        before = Path(row["before"]).resolve()
        if before.is_file():
            catalog = _append_history(catalog, before)
    catalog = _append_history(catalog, restored)

    complete_rows = [
        row for row in manifest.get("rounds", []) if row.get("status") == "complete"
    ]
    consecutive = 0
    plateau_window_start = None
    for row in reversed(complete_rows):
        if not row.get("plateau_signal", False):
            break
        consecutive += 1
        plateau_window_start = Path(row["before"]).resolve()

    for row in manifest.get("rounds", []):
        if row.get("status") not in {"complete", "abandoned_after_restart"}:
            row["status_before_restart"] = row.get("status")
            row["status"] = "abandoned_after_restart"
            row["restart_checkpoint"] = str(restored)
    phases = int(
        manifest.get(
            "completed_phases",
            manifest.get("protocol", {}).get("initial_phases", 0),
        )
    )
    return restored, catalog, phases, consecutive, plateau_window_start


def main() -> None:
    args = parse_args()
    positive = (
        args.block_epochs,
        args.games_per_epoch,
        args.training_max_turns,
        args.update_epochs,
        args.history_limit,
        args.monitor_blocks_per_stratum,
        args.history_check_blocks,
        args.confirmation_blocks_per_stratum,
        args.plateau_consecutive_blocks,
        args.rollout_workers,
        args.benchmark_workers,
    )
    if any(value < 1 for value in positive):
        raise ValueError("training, evaluation, and pool counts must be positive")
    if args.initial_phases < 0 or args.max_rounds < 0:
        raise ValueError("phase and round offsets must be nonnegative")
    if not 0 < args.plateau_materiality < 1:
        raise ValueError("plateau materiality must be between zero and one")
    current = args.current.resolve()
    teacher = args.teacher.resolve()
    mlp = args.mlp.resolve()
    initial_history = [path.resolve() for path in args.historical_checkpoint]
    for path in (current, teacher, mlp, *initial_history):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "league_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") == "confirmed_plateau":
            print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
            return
        (
            current,
            history_catalog,
            phases,
            consecutive,
            plateau_window_start,
        ) = _restore_campaign(
            manifest,
            current=current,
            teacher=teacher,
            mlp=mlp,
        )
        manifest.setdefault("resumed_at_unix", []).append(time.time())
        manifest["status"] = "running"
        manifest.pop("error", None)
        manifest.pop("finished_at_unix", None)
        manifest.pop("elapsed_seconds", None)
        protocol = manifest.setdefault("protocol", {})
        recorded_training_max_turns = int(
            protocol.get("training_max_turns", 1000)
        )
        if recorded_training_max_turns != args.training_max_turns:
            protocol["training_max_turns"] = args.training_max_turns
            manifest.setdefault("protocol_amendments", []).append(
                {
                    "revision": "training_turn_cap_v1",
                    "reason": (
                        "bound pathological rollout tails after an eight-player "
                        "cohort exceeded thirty CPU-minutes"
                    ),
                    "previous_training_max_turns": recorded_training_max_turns,
                    "training_max_turns": args.training_max_turns,
                    "evaluation_max_turns": 1000,
                    "effective_round": len(manifest.get("rounds", [])) + 1,
                    "recorded_at_unix": time.time(),
                }
            )
        if int(protocol.get("plateau_protocol_revision", 1)) < 2:
            protocol["plateau_protocol_revision"] = 2
            protocol["plateau_protocol_revision_effective_round"] = (
                len(manifest.get("rounds", [])) + 1
            )
            protocol["plateau_screen"] = (
                "both fixed-pool and rule-pool normalized point estimates "
                "at or below the per-block materiality threshold; nomination "
                "only, with no confidence claim"
            )
            protocol["plateau_confirmation"] = (
                "fresh disjoint fixed and rule pools must both have paired "
                "normal 99.9% upper bounds at or below materiality times the "
                "number of consecutive nominated blocks"
            )
            manifest.setdefault("protocol_amendments", []).append(
                {
                    "revision": 2,
                    "reason": (
                        "300-block screen cannot statistically resolve the "
                        "one-point materiality margin; retain it only as a "
                        "nomination gate and move confidence to powered "
                        "disjoint confirmation"
                    ),
                    "effective_round": len(manifest.get("rounds", [])) + 1,
                    "recorded_at_unix": time.time(),
                }
            )
            consecutive = 0
            plateau_window_start = None
        atomic_json(manifest, manifest_path)
    else:
        history_catalog = []
        for path in (teacher, *initial_history, current):
            history_catalog = _append_history(history_catalog, path)
        historical = _select_history(
            history_catalog, limit=args.history_limit
        )
        manifest = {
            "schema_version": 1,
            "kind": "factorized_historical_league_until_confirmed_plateau",
            "status": "running",
            "started_at_unix": time.time(),
            "initial_checkpoint": str(current),
            "teacher": {"path": str(teacher), "sha256": file_sha256(teacher)},
            "mlp": {"path": str(mlp), "sha256": file_sha256(mlp)},
            "protocol": {
                "initial_phases": args.initial_phases,
                "block_epochs": args.block_epochs,
                "games_per_epoch": args.games_per_epoch,
                "training_max_turns": args.training_max_turns,
                "evaluation_max_turns": 1000,
                "league_schedules": [list(schedule) for schedule in LEAGUE_SCHEDULES],
                "player_count_curricula": [
                    list(schedule) for schedule in PLAYER_COUNT_CURRICULA
                ],
                "all_curricula_cover_2_through_8": all(
                    set(schedule) == set(range(2, 9))
                    for schedule in PLAYER_COUNT_CURRICULA
                ),
                "history_limit": args.history_limit,
                "monitor_opponents": list(OPPONENTS),
                "rule_monitor_opponents": ["smart_rule", "smarter_rule"],
                "monitor_blocks_per_stratum": args.monitor_blocks_per_stratum,
                "history_check_blocks": args.history_check_blocks,
                "plateau_metric": "weighted normalized win-share delta",
                "plateau_confidence": "paired normal 99.9 percent",
                "plateau_protocol_revision": 2,
                "plateau_screen": (
                    "both fixed-pool and rule-pool normalized point estimates "
                    "at or below the per-block materiality threshold; nomination "
                    "only, with no confidence claim"
                ),
                "plateau_confirmation": (
                    "fresh disjoint fixed and rule pools must both have paired "
                    "normal 99.9% upper bounds at or below materiality times the "
                    "number of consecutive nominated blocks"
                ),
                "plateau_materiality_per_block": args.plateau_materiality,
                "consecutive_monitor_blocks": args.plateau_consecutive_blocks,
                "confirmation_blocks_per_stratum": (
                    args.confirmation_blocks_per_stratum
                ),
            },
            "initial_historical_pool": _history_records(historical),
            "rounds": [],
        }
        atomic_json(manifest, manifest_path)
        phases = args.initial_phases
        consecutive = 0
        plateau_window_start = None
    plateau = False

    try:
        round_index = len(manifest["rounds"])
        while not plateau:
            round_index += 1
            if args.max_rounds and round_index > args.max_rounds:
                manifest["status"] = "max_rounds_without_plateau"
                break
            round_dir = output_dir / f"round_{round_index:03d}"
            training_dir = round_dir / "training"
            previous = current
            schedule = LEAGUE_SCHEDULES[(round_index - 1) % len(LEAGUE_SCHEDULES)]
            player_counts = PLAYER_COUNT_CURRICULA[
                (round_index - 1) % len(PLAYER_COUNT_CURRICULA)
            ]
            history_catalog = _append_history(history_catalog, previous)
            historical = _select_history(
                history_catalog, limit=args.history_limit
            )
            command = [
                sys.executable,
                "train_puffer_fast.py",
                "--resume",
                str(previous),
                "--output-dir",
                str(training_dir),
                "--champion",
                str(mlp),
                "--epochs",
                str(args.block_epochs),
                "--games-per-epoch",
                str(args.games_per_epoch),
                "--max-turns",
                str(args.training_max_turns),
                "--rollout-workers",
                str(args.rollout_workers),
                "--rollout-engine",
                "vector",
                "--learning-rate",
                str(args.learning_rate),
                "--update-epochs",
                str(args.update_epochs),
                "--minibatch-trajectories",
                "128",
                "--diagnostic-minibatch-trajectories",
                "256",
                "--target-kl",
                "0.015",
                "--early-stop-diagnostic-trajectories",
                "512",
                "--checkpoint-every",
                str(args.block_epochs),
                "--evaluate-every",
                str(args.block_epochs + 1),
                "--evaluation-pairs",
                "1",
                "--schedule",
                ",".join(schedule),
                "--player-count-schedule",
                ",".join(str(value) for value in player_counts),
                "--seed",
                str(args.seed),
                "--campaign-offset",
                str(phases),
                "--persistent-replay",
                "--device",
                "cuda",
            ]
            for path in historical:
                command.extend(("--historical-checkpoint", str(path)))
            row = {
                "round": round_index,
                "phase_start": phases,
                "phase_end": phases + args.block_epochs - 1,
                "before": str(previous),
                "schedule": list(schedule),
                "player_count_schedule": list(player_counts),
                "player_count_histogram": {
                    str(players): player_counts.count(players)
                    for players in range(2, 9)
                },
                "historical_pool": _history_records(historical),
                "training_command": command,
                "status": "training",
            }
            manifest["rounds"].append(row)
            atomic_json(manifest, manifest_path)
            run_logged(command, round_dir / "training.log")
            current = training_dir / "latest.pt"
            if not current.is_file():
                raise RuntimeError("league block produced no checkpoint")
            phases += args.block_epochs
            row["after"] = str(current)
            row["after_sha256"] = file_sha256(current)
            row["status"] = "monitoring"
            atomic_json(manifest, manifest_path)

            pool_path = round_dir / "incremental_pool_evaluation.json"
            pool = evaluate_pool(
                previous,
                current,
                teacher=teacher,
                mlp=mlp,
                output=pool_path,
                blocks=args.monitor_blocks_per_stratum,
                workers=args.benchmark_workers,
                seed=args.seed + 200_000_000 + round_index * 10_000_000,
            )
            weighted = pool["weighted_result"]
            row["weighted_result"] = weighted
            row["opponent_results"] = pool["opponent_results"]

            rule_path = round_dir / "incremental_rule_evaluation.json"
            rules = evaluate_rules(
                previous,
                current,
                mlp=mlp,
                output=rule_path,
                blocks=args.monitor_blocks_per_stratum,
                workers=args.benchmark_workers,
                seed=args.seed + 300_000_000 + round_index * 10_000_000,
            )
            rule_weighted = rules["weighted_result"]
            row["rule_weighted_result"] = rule_weighted
            row["rule_opponent_results"] = rules["opponent_results"]

            history_results = []
            history_noninferior = True
            for index, opponent in enumerate(historical):
                history_path = round_dir / (
                    f"history_{index:02d}_{opponent.stem}_1v1.json"
                )
                result = evaluate_temporal(
                    current,
                    opponent,
                    mlp=mlp,
                    output=history_path,
                    blocks=args.history_check_blocks,
                    workers=args.benchmark_workers,
                    seed=(
                        args.seed
                        + 400_000_000
                        + round_index * 10_000_000
                        + index * 100_000
                    ),
                )
                history_results.append(
                    {
                        "opponent": str(opponent),
                        "opponent_sha256": file_sha256(opponent),
                        "evaluation": str(history_path),
                        "result": result["result"],
                    }
                )
                history_noninferior &= bool(
                    result["result"]["noninferiority_normal_99_9"]
                )
            row["historical_results"] = history_results
            row["historical_noninferiority_99_9"] = history_noninferior
            pool_regression = weighted["delta_normalized_ci99_9"][1] < 0
            rule_regression = (
                rule_weighted["delta_normalized_ci99_9"][1] < 0
            )
            candidate_accepted = (
                not pool_regression
                and not rule_regression
                and history_noninferior
            )
            row["regression_gates"] = {
                "fixed_pool_significant_regression_99_9": pool_regression,
                "rule_pool_significant_regression_99_9": rule_regression,
                "historical_noninferiority_99_9": history_noninferior,
                "candidate_accepted": candidate_accepted,
            }
            if not candidate_accepted:
                current = previous
                row["rollback_checkpoint"] = str(previous)
                row["rollback_checkpoint_sha256"] = file_sha256(previous)
            signal, screen = _plateau_screen(
                weighted,
                rule_weighted,
                materiality=args.plateau_materiality,
                history_noninferior=history_noninferior,
            )
            row["plateau_screen"] = screen
            row["plateau_signal"] = signal
            if signal:
                if consecutive == 0:
                    plateau_window_start = previous
                consecutive += 1
            else:
                consecutive = 0
                plateau_window_start = None
            row["consecutive_plateau_signals"] = consecutive
            row["status"] = "complete"
            manifest["current_checkpoint"] = str(current)
            manifest["current_checkpoint_sha256"] = file_sha256(current)
            manifest["completed_phases"] = phases
            atomic_json(manifest, manifest_path)

            if consecutive < args.plateau_consecutive_blocks:
                continue
            assert plateau_window_start is not None
            confirmation_path = round_dir / "plateau_confirmation.json"
            confirmation = evaluate_pool(
                plateau_window_start,
                current,
                teacher=teacher,
                mlp=mlp,
                output=confirmation_path,
                blocks=args.confirmation_blocks_per_stratum,
                workers=args.benchmark_workers,
                seed=args.seed + 900_000_000 + round_index * 10_000_000,
            )
            confirmation_weighted = confirmation["weighted_result"]
            rule_confirmation_path = (
                round_dir / "plateau_rule_confirmation.json"
            )
            rule_confirmation = evaluate_rules(
                plateau_window_start,
                current,
                mlp=mlp,
                output=rule_confirmation_path,
                blocks=args.confirmation_blocks_per_stratum,
                workers=args.benchmark_workers,
                seed=args.seed + 950_000_000 + round_index * 10_000_000,
            )
            rule_confirmation_weighted = rule_confirmation["weighted_result"]
            confirmation_limit = (
                args.plateau_materiality * args.plateau_consecutive_blocks
            )
            plateau, confirmation_gate = _confirmation_passed(
                confirmation_weighted,
                rule_confirmation_weighted,
                maximum_gain=confirmation_limit,
            )
            row["confirmation"] = {
                "fixed_pool": {
                    "path": str(confirmation_path),
                    "weighted_result": confirmation_weighted,
                },
                "rule_pool": {
                    "path": str(rule_confirmation_path),
                    "weighted_result": rule_confirmation_weighted,
                },
                "gate": confirmation_gate,
            }
            if not plateau:
                consecutive = 0
                plateau_window_start = None
            atomic_json(manifest, manifest_path)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        manifest["finished_at_unix"] = time.time()
        atomic_json(manifest, manifest_path)
        raise

    if plateau:
        manifest["status"] = "confirmed_plateau"
        manifest["plateau_checkpoint"] = str(current)
        manifest["plateau_checkpoint_sha256"] = file_sha256(current)
    manifest["finished_at_unix"] = time.time()
    manifest["elapsed_seconds"] = (
        manifest["finished_at_unix"] - manifest["started_at_unix"]
    )
    atomic_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
