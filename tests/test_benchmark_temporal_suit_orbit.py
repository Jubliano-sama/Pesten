from pathlib import Path

import pytest

import benchmark_temporal_suit_orbit as benchmark


def test_default_development_seed_namespace_is_below_sealed_floor():
    assert benchmark.DEFAULT_SEED < benchmark.SEALED_SEED_FLOOR
    assert (
        benchmark.DEFAULT_SEED
        + (len(benchmark.DEFAULT_OPPONENTS) - 1) * 100_000
        + 20
        < benchmark.SEALED_SEED_FLOOR
    )


def test_screen_rejects_sealed_deal_seed_before_gameplay(monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "load_residual_checkpoint",
        lambda *args, **kwargs: pytest.fail(
            "checkpoint loading should not be reached"
        ),
    )
    with pytest.raises(ValueError, match="sealed"):
        benchmark.run_screen(
            checkpoint=Path("fake.pt"),
            pairs=1,
            seed=benchmark.SEALED_SEED_FLOOR,
            opponent_names=("random",),
        )
