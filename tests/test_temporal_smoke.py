import math

from temporal_training import run_smoke_benchmark


def comparable(result):
    result = dict(result)
    result.pop("elapsed_seconds")
    return result


def test_deterministic_smoke_rollout_and_training():
    first = run_smoke_benchmark(
        seed=41, episode_count=1, max_turns=100
    )
    second = run_smoke_benchmark(
        seed=41, episode_count=1, max_turns=100
    )
    assert comparable(first) == comparable(second)
    assert first["rollout"]["episodes"] == 1
    assert first["rollout"]["turns"] > 0
    assert first["rollout"]["decisions"] > 0
    assert first["training"]["samples"] == first["rollout"]["decisions"]
    assert first["training"]["parameters_changed"]
    assert math.isfinite(first["training"]["loss"])
    assert not first["integration"]["native_pufferlib"]
    assert not first["model"]["critic_privileged"]
