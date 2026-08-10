from pathlib import Path

import run_ppo_fast


def test_launch_environment_prepends_jemalloc_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    allocator = tmp_path / "libjemalloc.so.2"
    allocator.touch()
    monkeypatch.setattr(
        run_ppo_fast,
        "JEMALLOC_CANDIDATES",
        (allocator,),
    )

    environment = run_ppo_fast.launch_environment(
        {"LD_PRELOAD": "/example/existing.so", "KEEP": "yes"}
    )
    assert environment == {
        "LD_PRELOAD": f"{allocator}:/example/existing.so",
        "KEEP": "yes",
    }
    assert run_ppo_fast.launch_environment(environment) == environment


def test_launch_environment_falls_back_without_jemalloc(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        run_ppo_fast,
        "JEMALLOC_CANDIDATES",
        (tmp_path / "missing.so",),
    )
    assert run_ppo_fast.launch_environment({"KEEP": "yes"}) == {
        "KEEP": "yes"
    }
