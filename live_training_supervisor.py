#!/usr/bin/env python3
"""Keep warm MLP and native PufferLib training moving until today's deadline.

Each model is trained in a short child process.  The child is restarted from
its latest actor/critic/optimizer checkpoint after every segment, which makes
the loop both warm and able to pick up optimizer-side source changes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, time as daytime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parent
LIVE_ROOT = REPO / "live_training"
MLP_ROOT = LIVE_ROOT / "mlp"
PUFFER_ROOT = LIVE_ROOT / "puffer"
MEASURE_ROOT = LIVE_ROOT / "measurements"
SOURCE_FILES = (
    "ppo2.py",
    "temporal_training.py",
    "temporal_puffer_v4.py",
    "temporal_league.py",
    "live_mlp_segment.py",
    "live_puffer_segment.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fingerprint() -> dict[str, str]:
    return {name: _sha256(REPO / name) for name in SOURCE_FILES}


def _deadline(value: str) -> datetime:
    hour, minute = (int(part) for part in value.split(":", 1))
    now = datetime.now().astimezone()
    target = now.replace(
        hour=hour,
        minute=minute,
        second=0,
        microsecond=0,
    )
    if target <= now:
        # The process is normally started before the deadline.  If it is
        # restarted after 16:00, do not accidentally train through tomorrow.
        return target
    return target


def _run(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.monotonic() - started
    print(
        json.dumps(
            {
                "command": command[0],
                "log": str(log_path),
                "returncode": completed.returncode,
                "elapsed_seconds": elapsed,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return completed.returncode


def _write_state(state: dict) -> None:
    state["updated_at"] = datetime.now().astimezone().isoformat()
    state_path = LIVE_ROOT / "supervisor_state.json"
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_result(path: Path) -> dict:
    if not path.is_file():
        return {"missing": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return {"invalid_json": str(path), "error": str(error)}


def _measurement_summary(path: Path) -> dict:
    result = _load_result(path)
    if "results" in result:
        return {
            opponent: {
                "score": payload.get("score"),
                "ci95": payload.get("paired_bootstrap_ci95"),
                "games": payload.get("games"),
            }
            for opponent, payload in result["results"].items()
        }
    summary = {}
    for opponent in result.get("pool", []):
        summary[opponent["opponent"]] = {
            "score": opponent["candidate_score"].get("mean"),
            "ci95": opponent["candidate_score"].get("bootstrap_ci95"),
            "delta_vs_old": opponent["paired_delta"].get("mean"),
        }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline", default="16:00")
    parser.add_argument("--mlp-iterations", type=int, default=4)
    parser.add_argument("--mlp-episodes", type=int, default=512)
    parser.add_argument("--puffer-epochs", type=int, default=4)
    parser.add_argument("--puffer-games-per-epoch", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=250)
    parser.add_argument("--max-cycles", type=int, default=10_000)
    parser.add_argument("--mlp-seed", type=int, default=2026080101)
    parser.add_argument("--puffer-seed", type=int, default=54010001)
    parser.add_argument("--puffer-initial", type=Path, default=Path(
        "benchmarks/temporal/reward_only_checkpoints/seed5401/reward_seed5401_epoch8.pt"
    ))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    deadline = _deadline(args.deadline)
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    MLP_ROOT.mkdir(parents=True, exist_ok=True)
    PUFFER_ROOT.mkdir(parents=True, exist_ok=True)
    MEASURE_ROOT.mkdir(parents=True, exist_ok=True)

    mlp_python = REPO / "venv/bin/python"
    puffer_python = REPO / "puffer4-native/.venv/bin/python"
    if not mlp_python.is_file() or not puffer_python.is_file():
        raise FileNotFoundError("one of the required project Python runtimes is missing")

    mlp_resume = {
        "actor": (REPO / "ppo_actor_1.pth").resolve(),
        "critic": (REPO / "ppo_critic_1.pth").resolve(),
        "actor_optimizer": (REPO / "ppo_actor_opt_1.pth").resolve(),
        "critic_optimizer": (REPO / "ppo_critic_opt_1.pth").resolve(),
    }
    puffer_resume = args.puffer_initial.resolve()
    if not puffer_resume.is_file():
        raise FileNotFoundError(f"initial Puffer checkpoint is missing: {puffer_resume}")
    puffer_epoch = 8
    cycle = 1
    state = {
        "kind": "live_training_supervisor",
        "deadline": deadline.isoformat(),
        "started_at": datetime.now().astimezone().isoformat(),
        "source_fingerprint": source_fingerprint(),
        "warm_initialization": {
            "mlp": {name: str(path) for name, path in mlp_resume.items()},
            "puffer": str(puffer_resume),
        },
        "cycles": [],
    }
    _write_state(state)

    initial_puffer_measurement = MEASURE_ROOT / "initial_puffer_vs_old_random.json"
    if not initial_puffer_measurement.is_file() and datetime.now().astimezone() < deadline:
        _run(
            [
                str(puffer_python),
                str(REPO / "live_puffer_benchmark.py"),
                "--checkpoint", str(puffer_resume),
                "--pairs", str(args.pairs),
                "--seed", str(args.puffer_seed),
                "--output", str(initial_puffer_measurement),
            ],
            MEASURE_ROOT / "initial_puffer_benchmark.log",
        )

    for cycle in range(1, args.max_cycles + 1):
        now = datetime.now().astimezone()
        if now >= deadline:
            break
        cycle_record = {
            "cycle": cycle,
            "started_at": now.isoformat(),
            "source_fingerprint_before": source_fingerprint(),
        }

        mlp_dir = MLP_ROOT / f"segment_{cycle:04d}"
        mlp_command = [
            str(mlp_python), str(REPO / "live_mlp_segment.py"),
            "--resume-actor", str(mlp_resume["actor"]),
            "--resume-critic", str(mlp_resume["critic"]),
            "--resume-actor-optimizer", str(mlp_resume["actor_optimizer"]),
            "--resume-critic-optimizer", str(mlp_resume["critic_optimizer"]),
            "--output-dir", str(mlp_dir),
            "--iterations", str(args.mlp_iterations),
            "--episodes", str(args.mlp_episodes),
            "--seed", str(args.mlp_seed + cycle),
        ]
        mlp_log = mlp_dir / "training.log"
        cycle_record["mlp_returncode"] = _run(mlp_command, mlp_log)
        mlp_result = _load_result(mlp_dir / "segment_result.json")
        cycle_record["mlp_segment"] = mlp_result
        if cycle_record["mlp_returncode"] != 0:
            _write_state({**state, "cycles": state["cycles"] + [cycle_record]})
            break

        mlp_latest = {
            "actor": mlp_dir / f"ppo_actor_{args.mlp_iterations}.pth",
            "critic": mlp_dir / f"ppo_critic_{args.mlp_iterations}.pth",
            "actor_optimizer": mlp_dir / f"ppo_actor_opt_{args.mlp_iterations}.pth",
            "critic_optimizer": mlp_dir / f"ppo_critic_opt_{args.mlp_iterations}.pth",
        }
        mlp_measurement = MEASURE_ROOT / f"mlp_cycle_{cycle:04d}_vs_old_random.json"
        cycle_record["mlp_benchmark_returncode"] = _run(
            [
                str(mlp_python), str(REPO / "benchmark_checkpoint_batched.py"),
                "--candidate", str(mlp_latest["actor"]),
                "--parent", str(REPO / "ppo_actor_100run3.pth"),
                "--pairs", str(args.pairs),
                "--seed", str(82806001 + cycle * 10_000),
                "--pool", "random",
                "--device", "cuda",
                "--canonical-throughput-pairs", "0",
                "--output", str(mlp_measurement),
            ],
            MEASURE_ROOT / f"mlp_cycle_{cycle:04d}_benchmark.log",
        )
        cycle_record["mlp_measurement"] = {
            "path": str(mlp_measurement),
            "summary": _measurement_summary(mlp_measurement),
        }
        mlp_resume = {name: path.resolve() for name, path in mlp_latest.items()}

        now = datetime.now().astimezone()
        if now >= deadline:
            state["cycles"].append(cycle_record)
            state["last_measurement"] = cycle_record["mlp_measurement"]
            _write_state(state)
            break

        puffer_dir = PUFFER_ROOT / f"segment_{cycle:04d}"
        puffer_command = [
            str(puffer_python), str(REPO / "live_puffer_segment.py"),
            "--resume", str(puffer_resume),
            "--output-dir", str(puffer_dir),
            "--epochs", str(args.puffer_epochs),
            "--global-epoch", str(puffer_epoch + 1),
            "--seed", str(args.puffer_seed + cycle),
            "--games-per-epoch", str(args.puffer_games_per_epoch),
            "--development-pairs", str(max(10, min(50, args.pairs // 5))),
            "--device", "cuda",
        ]
        puffer_log = puffer_dir / "training.log"
        cycle_record["puffer_returncode"] = _run(puffer_command, puffer_log)
        cycle_record["puffer_segment"] = _load_result(puffer_dir / "segment_result.json")
        if cycle_record["puffer_returncode"] != 0:
            state["cycles"].append(cycle_record)
            _write_state(state)
            break

        puffer_epoch += args.puffer_epochs
        puffer_resume = (puffer_dir / "latest.pt").resolve()
        puffer_measurement = MEASURE_ROOT / f"puffer_cycle_{cycle:04d}_vs_old_random.json"
        cycle_record["puffer_benchmark_returncode"] = _run(
            [
                str(puffer_python), str(REPO / "live_puffer_benchmark.py"),
                "--checkpoint", str(puffer_resume),
                "--pairs", str(args.pairs),
                "--seed", str(args.puffer_seed + cycle * 10_000),
                "--output", str(puffer_measurement),
            ],
            MEASURE_ROOT / f"puffer_cycle_{cycle:04d}_benchmark.log",
        )
        cycle_record["puffer_measurement"] = {
            "path": str(puffer_measurement),
            "summary": _measurement_summary(puffer_measurement),
        }
        cycle_record["source_fingerprint_after"] = source_fingerprint()
        cycle_record["finished_at"] = datetime.now().astimezone().isoformat()
        state["cycles"].append(cycle_record)
        state["current_warm_resume"] = {
            "mlp": {name: str(path) for name, path in mlp_resume.items()},
            "puffer": str(puffer_resume),
            "puffer_epoch": puffer_epoch,
        }
        state["source_fingerprint"] = source_fingerprint()
        _write_state(state)
        print(json.dumps({
            "cycle": cycle,
            "mlp": cycle_record["mlp_measurement"]["summary"],
            "puffer": cycle_record["puffer_measurement"]["summary"],
            "source_changed": cycle_record["source_fingerprint_before"] != cycle_record["source_fingerprint_after"],
        }, sort_keys=True), flush=True)

    state["finished_at"] = datetime.now().astimezone().isoformat()
    state["finished_reason"] = "deadline_or_cycle_limit"
    _write_state(state)
    print(json.dumps({
        "finished_at": state["finished_at"],
        "deadline": state["deadline"],
        "cycles_completed": len(state["cycles"]),
        "state": str(LIVE_ROOT / "supervisor_state.json"),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
