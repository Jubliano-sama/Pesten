"""Shared seed namespaces and recorded-artifact contamination checks.

Development generators are bounded below the active sealed namespace.  The
original sealed interval was retired after a legacy hash-derived development
benchmark landed inside it.  The replacement interval sits near the top of the
``uint32`` range and was selected only after scanning every recorded JSON/JSONL
benchmark artifact in this checkout.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Iterable, Iterator


RETIRED_SEALED_SEED_START = 92_000_000
RETIRED_SEALED_SEED_END = 93_000_000

# Mechanically verified unused on 2026-07-31 across all JSON/JSONL files under
# benchmarks/ and experiments/.  This provides 900,000 seeds, including ample
# room for the default four namespaces x 1,000 paired deals.
SEALED_SEED_START = 4_294_000_000
SEALED_SEED_END = 4_294_900_000
DEVELOPMENT_SEED_LIMIT = SEALED_SEED_START
DEFAULT_SEALED_CLAIM_LEDGER = (
    Path(__file__).resolve().parent
    / "benchmarks"
    / ".sealed_evaluation_claim.json"
)

_NON_GAME_SEED_MARKERS = (
    "bootstrap",
    "sealed",
    "interval",
    "limit",
    "start",
    "end",
)


@dataclass(frozen=True)
class RecordedSeedUse:
    path: str
    key_path: str
    seed: int


def claim_sealed_evaluation(
    *,
    evaluator: str,
    candidate_path: Path,
    candidate_sha256: str,
    base_seed: int,
    pairs: int,
    output_path: Path,
    configuration: dict,
    ledger_path: Path = DEFAULT_SEALED_CLAIM_LEDGER,
) -> dict:
    """Atomically burn the global holdout and bind its complete invocation.

    The exclusive create happens before model loading or game execution.  A
    crash after this function returns intentionally leaves the claim in place:
    an observed or partially observed holdout must never be reopened.
    """
    evaluator = str(evaluator).strip()
    if not evaluator:
        raise ValueError("sealed evaluator name cannot be empty")
    if not SEALED_SEED_START <= base_seed < SEALED_SEED_END:
        raise ValueError("sealed claim base seed is outside the active interval")
    if pairs < 1:
        raise ValueError("sealed claim pairs must be positive")
    normalized_sha = str(candidate_sha256).strip().lower()
    if (
        len(normalized_sha) != 64
        or any(character not in "0123456789abcdef" for character in normalized_sha)
    ):
        raise ValueError("sealed claim requires a complete SHA-256 digest")
    # JSON round-tripping both validates and freezes caller-owned configuration
    # before the ledger is created.
    normalized_configuration = json.loads(
        json.dumps(configuration, sort_keys=True, default=str)
    )
    resolved_ledger = Path(ledger_path).resolve()
    payload = {
        "schema_version": 1,
        "status": "holdout_burned_before_evaluation",
        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluator": evaluator,
        "candidate": {
            "path": str(Path(candidate_path).resolve()),
            "sha256": normalized_sha,
        },
        "seed_interval": {
            "start_inclusive": SEALED_SEED_START,
            "end_exclusive": SEALED_SEED_END,
        },
        "base_seed": int(base_seed),
        "pairs": int(pairs),
        "output_path": str(Path(output_path).resolve()),
        "configuration": normalized_configuration,
        "ledger_path": str(resolved_ledger),
    }
    rendered = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    resolved_ledger.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            resolved_ledger,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError as error:
        raise FileExistsError(
            "sealed holdout was already claimed; refusing every rerun: "
            f"{resolved_ledger}"
        ) from error
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        # Never remove a partially written claim: reaching exclusive creation
        # burns the holdout even if durable publication subsequently fails.
        raise
    return payload


def _iter_seed_values(value, key_path: tuple[str, ...] = ()) -> Iterator[tuple[str, int]]:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            child_path = key_path + (str(key),)
            if (
                isinstance(child, int)
                and not isinstance(child, bool)
                and "seed" in normalized
                and not any(
                    marker in normalized
                    for marker in _NON_GAME_SEED_MARKERS
                )
            ):
                yield ".".join(child_path), int(child)
            yield from _iter_seed_values(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_seed_values(
                child,
                key_path + (f"[{index}]",),
            )


def _iter_json_documents(path: Path) -> Iterator[object]:
    if path.suffix == ".jsonl":
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL: {error}"
                ) from error
        return
    try:
        yield json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error


def recorded_artifact_paths(roots: Iterable[Path]) -> Iterator[Path]:
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".json", ".jsonl"}:
                yield path


def scan_recorded_seed_interval(
    roots: Iterable[Path],
    *,
    start: int = SEALED_SEED_START,
    end: int = SEALED_SEED_END,
    allow_sealed_results: bool = True,
) -> list[RecordedSeedUse]:
    """Return every recorded game/development seed inside ``[start, end)``.

    Output artifacts with ``sealed`` in their filename are skipped by default:
    once a final evaluation is intentionally run, its seeds belong in the
    reserved interval.  Development artifacts must never appear in the result.
    """
    if not 0 <= start < end <= 2**32:
        raise ValueError("seed interval must be a non-empty uint32 range")
    uses = []
    for path in recorded_artifact_paths(roots):
        if allow_sealed_results and "sealed" in path.name.lower():
            continue
        for document in _iter_json_documents(path):
            for key_path, seed in _iter_seed_values(document):
                if start <= seed < end:
                    uses.append(
                        RecordedSeedUse(
                            path=str(path),
                            key_path=key_path,
                            seed=seed,
                        )
                    )
    return uses


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=(Path("benchmarks"), Path("experiments")),
    )
    parser.add_argument("--start", type=int, default=SEALED_SEED_START)
    parser.add_argument("--end", type=int, default=SEALED_SEED_END)
    parser.add_argument(
        "--include-sealed-results",
        action="store_true",
        help="also report intentional sealed-result artifacts",
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()
    uses = scan_recorded_seed_interval(
        args.roots,
        start=args.start,
        end=args.end,
        allow_sealed_results=not args.include_sealed_results,
    )
    print(
        json.dumps(
            {
                "interval_start_inclusive": args.start,
                "interval_end_exclusive": args.end,
                "contamination_count": len(uses),
                "uses": [asdict(use) for use in uses],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if uses:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
