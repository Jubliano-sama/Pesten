#!/usr/bin/env python3
"""Launch PPO training with the fastest verified process-level allocator."""

from __future__ import annotations

import os
from pathlib import Path
import sys


JEMALLOC_CANDIDATES = (
    Path("/usr/lib/x86_64-linux-gnu/libjemalloc.so.2"),
    Path("/lib/x86_64-linux-gnu/libjemalloc.so.2"),
)


def jemalloc_path() -> Path | None:
    """Return the first verified system jemalloc library, if available."""
    return next(
        (candidate for candidate in JEMALLOC_CANDIDATES if candidate.is_file()),
        None,
    )


def launch_environment(
    base: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build an exec environment that preloads jemalloc exactly once."""
    environment = dict(os.environ if base is None else base)
    allocator = jemalloc_path()
    if allocator is None:
        return environment
    existing = environment.get("LD_PRELOAD", "").split(":")
    allocator_text = str(allocator)
    if allocator_text not in existing:
        environment["LD_PRELOAD"] = ":".join(
            [allocator_text, *(entry for entry in existing if entry)]
        )
    return environment


def main() -> None:
    training_script = Path(__file__).resolve().with_name("ppo2.py")
    command = [
        sys.executable,
        str(training_script),
        *sys.argv[1:],
    ]
    os.execvpe(sys.executable, command, launch_environment())


if __name__ == "__main__":
    main()
