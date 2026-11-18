"""A locked, append-only log of completed matched evaluation chunks."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def create_log(path, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent) as temporary:
        temporary.write(encode(header) + b"\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        os.link(temporary.name, path)


@contextmanager
def chunk_log(path, *, header, tasks, resume):
    if path is None:
        if resume:
            raise ValueError("resuming requires a checkpoint path")
        yield {}, None
        return
    definitions = {task["seed"]: task for task in tasks}
    header = {**header, "task_plan_sha256": hashlib.sha256(encode(tasks)).hexdigest()}
    if not resume:
        create_log(path, header)
    with path.open("r+b") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if handle.readline() != encode(header) + b"\n":
            raise ValueError("checkpoint belongs to a different freeze or task plan")
        saved = {}
        seen = set()

        def validate(row):
            seed = row["seed"]
            task = definitions.get(seed)
            if seed in seen or task is None or (row["players"], row["opponent"]) != (task["players"], task["opponent"]):
                raise ValueError("duplicate or unexpected completed chunk")
            for variant in task["variants"]:
                for suffix in ("scores", "win_shares", "loss_shares", "draw_shares"):
                    if len(row[f"{variant}_{suffix}"]) != task["blocks"]:
                        raise ValueError("completed chunk has wrong number of deals")
                totals = row["totals"][variant]
                n = task["players"]
                counts = {}
                for name, plural in (("win", "wins"), ("loss", "losses"), ("draw", "draws")):
                    values = row[f"{variant}_{name}_shares"]
                    if not all(0 <= v <= 1 and abs(n * v - round(n * v)) < 1e-9 for v in values):
                        raise ValueError("invalid completed outcome shares")
                    counts[name] = [round(n * v) for v in values]
                    if sum(counts[name]) != totals.get(plural, 0):
                        raise ValueError("completed outcome totals disagree with blocks")
                for w, loss, d, score in zip(counts["win"], counts["loss"], counts["draw"], row[f"{variant}_scores"]):
                    if w + loss + d != n or not math.isclose(score, (w + d / n) / n, abs_tol=1e-12):
                        raise ValueError("inconsistent completed deal")
            seen.add(seed)

        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                # A process can die partway through the last append. Preserve
                # its bytes before retrying the one chunk without a commit.
                backup = Path(f"{path}.incomplete-{offset}-{hashlib.sha256(line).hexdigest()}")
                if not backup.exists():
                    backup.write_bytes(line)
                elif backup.read_bytes() != line:
                    raise ValueError("incomplete checkpoint backup differs")
                handle.truncate(offset)
                handle.seek(offset)
                break
            record = json.loads(line)
            row = record["row"]
            if hashlib.sha256(encode(row)).hexdigest() != record["sha256"]:
                raise ValueError("completed chunk checksum mismatch")
            validate(row)
            saved[row["seed"]] = row

        def append(row):
            validate(row)
            digest = hashlib.sha256(encode(row)).hexdigest()
            handle.write(encode({"row": row, "sha256": digest}) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())

        yield saved, append
