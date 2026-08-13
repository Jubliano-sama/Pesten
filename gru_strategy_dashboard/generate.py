"""Generate measured GRU strategy data and its canonical dashboard artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .analyze import analyze_checkpoint, refresh_summary_language
from .artifact import build_artifact


DEFAULT_CHECKPOINT = Path(
    "live_training/rlvr_breakthrough_20260812/teacher/training/latest.pt"
)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build(
    *,
    root: Path,
    checkpoint: Path,
    summary_output: Path,
    artifact_output: Path,
    games_per_player_count: int,
    seed: int,
    device: str,
    max_turns: int,
    torch_threads: int,
    reuse_summary: bool,
) -> dict[str, object]:
    if reuse_summary:
        if not summary_output.is_file():
            raise FileNotFoundError(summary_output)
        summary = refresh_summary_language(
            json.loads(summary_output.read_text(encoding="utf-8"))
        )
        _write_json(summary_output, summary)
    else:
        summary = analyze_checkpoint(
            checkpoint,
            games_per_player_count=games_per_player_count,
            seed=seed,
            device=device,
            max_turns=max_turns,
            torch_threads=torch_threads,
        )
        summary["checkpoint"]["path"] = _relative(checkpoint, root)
        _write_json(summary_output, summary)
    artifact = build_artifact(
        summary, summary_path=_relative(summary_output, root)
    )
    _write_json(artifact_output, artifact)
    default_headline = next(
        row
        for row in summary["datasets"]["headline"]
        if row["action_style"] == "Varied choices"
        and row["view_players"] == "All table sizes"
    )
    return {
        "checkpoint": summary["checkpoint"],
        "summary_output": _relative(summary_output, root),
        "artifact_output": _relative(artifact_output, root),
        "games": default_headline["games"],
        "teacher_decisions": default_headline["teacher_decisions"],
        "snapshot_status": artifact["snapshot"]["status"],
        "reused_summary": reuse_summary,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("benchmarks/gru_strategy/strategy_summary.json"),
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=Path("benchmarks/gru_strategy/artifact.json"),
    )
    parser.add_argument("--games-per-player-count", type=int, default=512)
    parser.add_argument("--seed", type=int, default=202_608_120)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-turns", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument(
        "--reuse-summary",
        action="store_true",
        help="rebuild only artifact.json from an existing strategy summary",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    checkpoint = args.checkpoint
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    summary_output = args.summary_output
    if not summary_output.is_absolute():
        summary_output = root / summary_output
    artifact_output = args.artifact_output
    if not artifact_output.is_absolute():
        artifact_output = root / artifact_output
    try:
        receipt = build(
            root=root,
            checkpoint=checkpoint,
            summary_output=summary_output,
            artifact_output=artifact_output,
            games_per_player_count=args.games_per_player_count,
            seed=args.seed,
            device=args.device,
            max_turns=args.max_turns,
            torch_threads=args.torch_threads,
            reuse_summary=args.reuse_summary,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"GRU strategy dashboard generation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
