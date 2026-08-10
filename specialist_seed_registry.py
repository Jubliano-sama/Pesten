"""Append-only seed accounting for the audited random-specialist campaign.

The campaign has to keep future development, confirmation, and final-audit
deals disjoint even when an artifact records only a ``base_seed`` plus a
count.  This module therefore treats intervals as first-class values instead
of checking a handful of enumerated seeds after the fact.

It deliberately does not reuse the already consumed global sealed holdout.
The new campaign's final audit is a separately named, separately ledgered
one-use evaluation.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence


UINT32_LIMIT = 2**32
DEFAULT_ALLOCATION_START = 4_291_000_000
REQUIRED_NAMESPACE_PREFIXES = (
    "calibration",
    "training",
    "development_screening",
    "confirmation",
    "bootstrap",
    "final_audit",
)


@dataclass(frozen=True, slots=True, order=True)
class SeedInterval:
    """A non-empty half-open uint32 seed interval."""

    start_inclusive: int
    end_exclusive: int
    namespace: str
    source: str

    def __post_init__(self) -> None:
        if not (
            0 <= self.start_inclusive < self.end_exclusive <= UINT32_LIMIT
        ):
            raise ValueError(
                "seed interval must be a non-empty uint32 half-open range"
            )
        if not self.namespace:
            raise ValueError("seed interval namespace cannot be empty")

    @property
    def count(self) -> int:
        return self.end_exclusive - self.start_inclusive

    def overlaps(self, other: "SeedInterval") -> bool:
        return (
            self.start_inclusive < other.end_exclusive
            and other.start_inclusive < self.end_exclusive
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "start_inclusive": self.start_inclusive,
            "end_exclusive": self.end_exclusive,
            "count": self.count,
        }


@dataclass(frozen=True, slots=True)
class AllocationRequest:
    """One requested campaign namespace allocation."""

    namespace: str
    count: int
    purpose: str
    start_inclusive: int | None = None

    def __post_init__(self) -> None:
        _validate_namespace(self.namespace)
        if self.count < 1:
            raise ValueError("seed allocation count must be positive")
        if self.start_inclusive is not None and not (
            0 <= self.start_inclusive < UINT32_LIMIT
        ):
            raise ValueError("seed allocation start must be uint32-compatible")


CONSUMED_LEGACY_SEALED_INTERVAL = SeedInterval(
    4_294_000_000,
    4_294_900_000,
    "legacy_consumed_sealed_holdout",
    "authoritative legacy sealed-holdout reservation",
)


def _validate_namespace(namespace: str) -> None:
    normalized = str(namespace).strip()
    if not normalized:
        raise ValueError("seed namespace cannot be empty")
    if not any(
        normalized == prefix or normalized.startswith(prefix + "_")
        for prefix in REQUIRED_NAMESPACE_PREFIXES
    ):
        raise ValueError(
            "namespace must be one of the required campaign families: "
            + ", ".join(REQUIRED_NAMESPACE_PREFIXES)
        )


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _path_label(parts: Sequence[str]) -> str:
    return ".".join(parts) if parts else "<root>"


def _explicit_interval(
    value: Mapping[str, object],
    *,
    namespace: str,
    source: str,
) -> SeedInterval | None:
    """Read explicit half-open or conservative closed interval encodings."""

    start = value.get("start_inclusive")
    end_exclusive = value.get("end_exclusive")
    if _is_int(start) and _is_int(end_exclusive):
        return SeedInterval(int(start), int(end_exclusive), namespace, source)

    # Existing artifacts sometimes spell an interval as ``seed_start`` /
    # ``seed_end`` without saying whether the end is inclusive.  Treat it as
    # closed to be conservative; that can reserve one extra seed but cannot
    # silently admit an overlapping campaign range.
    for start_key, end_key in (
        ("seed_start", "seed_end"),
        ("deal_seed_start", "deal_seed_end"),
        ("game_seed_start", "game_seed_end"),
    ):
        start = value.get(start_key)
        end = value.get(end_key)
        if _is_int(start) and _is_int(end):
            return SeedInterval(int(start), int(end) + 1, namespace, source)
    return None


def _declared_count(value: Mapping[str, object]) -> int | None:
    """Find a deal-count that derives a contiguous game-seed interval."""

    counts = []
    for key in (
        "pairs",
        "games",
        "game_count",
        "deal_count",
        "num_games",
        "seed_count",
        "count",
    ):
        candidate = value.get(key)
        if _is_int(candidate) and int(candidate) > 0:
            counts.append(int(candidate))
    # If an artifact exposes both games and pairs, using the larger value is
    # intentionally conservative.  The registry protects intervals, not a
    # guessed interpretation of a historical report.
    return max(counts, default=None)


def _derived_base_intervals(
    value: Mapping[str, object],
    *,
    namespace: str,
    source: str,
) -> Iterator[SeedInterval]:
    count = _declared_count(value)
    if count is None:
        return
    for key in ("base_seed", "deal_seed", "game_seed", "seed"):
        candidate = value.get(key)
        if not _is_int(candidate):
            continue
        start = int(candidate)
        if 0 <= start and start + count <= UINT32_LIMIT:
            yield SeedInterval(start, start + count, namespace, source)


def _iter_scalar_seed_intervals(
    value: Mapping[str, object],
    *,
    namespace: str,
    source: str,
) -> Iterator[SeedInterval]:
    """Conservatively reserve every scalar integer field named like a seed."""

    for key, candidate in value.items():
        lower = str(key).lower()
        if (
            "seed" not in lower
            or lower.endswith("_start")
            or lower.endswith("_end")
            or lower.endswith("_exclusive")
            or not _is_int(candidate)
        ):
            continue
        seed = int(candidate)
        if 0 <= seed < UINT32_LIMIT:
            yield SeedInterval(seed, seed + 1, namespace, source)


def _iter_json_intervals(
    value: object,
    *,
    artifact: Path,
    path: tuple[str, ...] = (),
) -> Iterator[SeedInterval]:
    """Yield explicit, derived, and individually recorded artifact ranges."""

    label = f"{artifact}:{_path_label(path)}"
    if isinstance(value, Mapping):
        namespace = f"artifact:{artifact.name}:{_path_label(path)}"
        explicit = _explicit_interval(
            value, namespace=namespace, source=label
        )
        if explicit is not None:
            yield explicit
        yield from _derived_base_intervals(
            value, namespace=namespace, source=label
        )
        yield from _iter_scalar_seed_intervals(
            value, namespace=namespace, source=label
        )

        # A two-item ``seed_range``/``seed_interval`` list is conventionally
        # half-open in this repository.  Read it before walking children so
        # a declared range is checked as a range, never only as endpoints.
        for key, child in value.items():
            lower = str(key).lower()
            if (
                lower in {"seed_range", "seed_interval", "deal_seed_range"}
                and isinstance(child, list)
                and len(child) == 2
                and all(_is_int(item) for item in child)
            ):
                start, end = int(child[0]), int(child[1])
                if 0 <= start < end <= UINT32_LIMIT:
                    yield SeedInterval(
                        start,
                        end,
                        namespace,
                        f"{label}.{key}",
                    )
            yield from _iter_json_intervals(
                child,
                artifact=artifact,
                path=path + (str(key),),
            )
        return

    if isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_json_intervals(
                child,
                artifact=artifact,
                path=path + (f"[{index}]",),
            )


def _iter_json_documents(path: Path) -> Iterator[object]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".jsonl":
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL seed artifact"
                ) from error
        return
    try:
        yield json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON seed artifact") from error


def scan_artifact_seed_intervals(
    roots: Iterable[Path],
    *,
    excluded_paths: Iterable[Path] = (),
) -> list[SeedInterval]:
    """Scan JSON/JSONL benchmark, experiment, and live-training evidence."""

    excluded = {Path(path).resolve() for path in excluded_paths}
    intervals: list[SeedInterval] = []
    for root in roots:
        resolved_root = Path(root).resolve()
        if not resolved_root.exists():
            continue
        for path in sorted(resolved_root.rglob("*")):
            if (
                not path.is_file()
                or path.suffix not in {".json", ".jsonl"}
                or path.resolve() in excluded
            ):
                continue
            for document in _iter_json_documents(path):
                intervals.extend(
                    _iter_json_intervals(document, artifact=path)
                )
    return intervals


def _record_intervals(record: Mapping[str, object], source: str) -> Iterator[SeedInterval]:
    namespace = str(record.get("namespace", "registry"))
    value = record.get("interval")
    if isinstance(value, Mapping):
        interval = _explicit_interval(
            value, namespace=namespace, source=source
        )
        if interval is not None:
            yield interval
    values = record.get("intervals")
    if isinstance(values, list):
        for item in values:
            if not isinstance(item, Mapping):
                continue
            item_namespace = str(item.get("namespace", namespace))
            interval_value = item.get("interval", item)
            if isinstance(interval_value, Mapping):
                interval = _explicit_interval(
                    interval_value,
                    namespace=item_namespace,
                    source=source,
                )
                if interval is not None:
                    yield interval


class SeedRegistry:
    """Interval-aware append-only JSONL registry with exclusive append locks."""

    def __init__(
        self,
        path: Path,
        *,
        artifact_roots: Iterable[Path] = (
            Path("benchmarks"),
            Path("experiments"),
            Path("live_training"),
        ),
    ) -> None:
        self.path = Path(path).resolve()
        self.artifact_roots = tuple(Path(root).resolve() for root in artifact_roots)

    def read_records(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{self.path}:{line_number}: invalid registry JSONL"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"{self.path}:{line_number}: registry record is not an object"
                )
            records.append(record)
        return records

    def registered_intervals(self) -> list[SeedInterval]:
        intervals = []
        for line_number, record in enumerate(self.read_records(), start=1):
            intervals.extend(
                _record_intervals(record, f"{self.path}:{line_number}")
            )
        return intervals

    def occupied_intervals(self) -> list[SeedInterval]:
        """Return registry, historical-artifact, and retired-holdout ranges."""

        return [
            CONSUMED_LEGACY_SEALED_INTERVAL,
            *self.registered_intervals(),
            *scan_artifact_seed_intervals(
                self.artifact_roots,
                excluded_paths=(self.path,),
            ),
        ]

    @contextmanager
    def _exclusive_append_handle(self):
        """Create once or append under an advisory exclusive filesystem lock."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
            )
        except FileExistsError:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield descriptor
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    @staticmethod
    def _find_free_start(
        count: int,
        occupied: Sequence[SeedInterval],
        *,
        preferred_start: int,
    ) -> int:
        if count < 1 or count > UINT32_LIMIT:
            raise ValueError("invalid requested seed interval size")
        ordered = sorted(occupied, key=lambda item: item.start_inclusive)
        for starting_point in (preferred_start, 0):
            candidate = starting_point
            while candidate + count <= UINT32_LIMIT:
                collisions = [
                    interval
                    for interval in ordered
                    if interval.start_inclusive < candidate + count
                    and candidate < interval.end_exclusive
                ]
                if not collisions:
                    return candidate
                candidate = max(
                    interval.end_exclusive for interval in collisions
                )
        raise ValueError("no collision-free uint32 seed interval remains")

    @staticmethod
    def _assert_no_overlap(
        candidate: SeedInterval,
        occupied: Iterable[SeedInterval],
    ) -> None:
        collisions = [item for item in occupied if candidate.overlaps(item)]
        if collisions:
            details = "; ".join(
                f"{item.namespace} [{item.start_inclusive}, "
                f"{item.end_exclusive}) from {item.source}"
                for item in collisions[:4]
            )
            raise ValueError(
                f"seed interval {candidate.namespace} "
                f"[{candidate.start_inclusive}, {candidate.end_exclusive}) "
                f"overlaps recorded allocation(s): {details}"
            )

    def allocate_many(
        self,
        requests: Sequence[AllocationRequest],
        *,
        preferred_start: int = DEFAULT_ALLOCATION_START,
    ) -> dict[str, SeedInterval]:
        """Atomically validate then append a sequence of non-overlapping rows.

        A process crash during append intentionally leaves any already-written
        allocations burned.  That is safer than silently reusing a partially
        published campaign range.
        """

        if not requests:
            raise ValueError("at least one seed allocation is required")
        names = [request.namespace for request in requests]
        if len(names) != len(set(names)):
            raise ValueError("allocation request namespaces must be unique")
        with self._exclusive_append_handle() as descriptor:
            records = self.read_records()
            prior_names = {
                str(record.get("namespace"))
                for record in records
                if record.get("record_type") == "seed_allocation"
            }
            duplicate_names = sorted(set(names) & prior_names)
            if duplicate_names:
                raise FileExistsError(
                    "append-only registry already allocates namespace(s): "
                    + ", ".join(duplicate_names)
                )
            occupied = self.occupied_intervals()
            result: dict[str, SeedInterval] = {}
            for request in requests:
                start = (
                    request.start_inclusive
                    if request.start_inclusive is not None
                    else self._find_free_start(
                        request.count,
                        occupied,
                        preferred_start=preferred_start,
                    )
                )
                candidate = SeedInterval(
                    start,
                    start + request.count,
                    request.namespace,
                    "pending registry allocation",
                )
                self._assert_no_overlap(candidate, occupied)
                occupied.append(candidate)
                result[request.namespace] = candidate

            now = datetime.now(timezone.utc).isoformat()
            for request in requests:
                interval = result[request.namespace]
                record = {
                    "schema_version": 1,
                    "record_type": "seed_allocation",
                    "created_at_utc": now,
                    "namespace": request.namespace,
                    "purpose": request.purpose,
                    "interval": interval.as_dict(),
                    "append_only": True,
                }
                payload = (
                    json.dumps(record, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                os.write(descriptor, payload)
            os.fsync(descriptor)
            return result

    def validate_registered_namespaces(
        self,
        namespaces: Iterable[str],
    ) -> dict[str, SeedInterval]:
        """Return required registered rows after checking pairwise intervals."""

        expected = tuple(namespaces)
        records = self.registered_intervals()
        by_name: dict[str, SeedInterval] = {}
        for interval in records:
            if interval.namespace in by_name:
                raise ValueError(
                    "append-only registry contains duplicate namespace: "
                    + interval.namespace
                )
            by_name[interval.namespace] = interval
        missing = [name for name in expected if name not in by_name]
        if missing:
            raise ValueError(
                "registry is missing required namespace(s): "
                + ", ".join(missing)
            )
        selected = [by_name[name] for name in expected]
        for index, interval in enumerate(selected):
            self._assert_no_overlap(interval, selected[:index])
            if interval.overlaps(CONSUMED_LEGACY_SEALED_INTERVAL):
                raise ValueError("campaign range overlaps consumed sealed holdout")
        return {name: by_name[name] for name in expected}
