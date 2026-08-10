import pytest

from ppo2 import (
    DEFAULT_COMPILE_BREAK_EVEN_GAMES,
    resolve_compile_updates,
)


@pytest.mark.parametrize("requested", [True, False])
def test_explicit_compile_choice_wins(requested):
    assert resolve_compile_updates(
        requested,
        iterations=1,
        episodes_per_iteration=1,
        cuda_available=False,
        training_actor_device="cpu",
    ) is requested


def test_auto_compile_requires_cuda_actor_and_break_even_work():
    threshold = DEFAULT_COMPILE_BREAK_EVEN_GAMES
    assert resolve_compile_updates(
        None,
        iterations=threshold // 1000,
        episodes_per_iteration=1000,
        cuda_available=True,
        training_actor_device="auto",
    )
    assert not resolve_compile_updates(
        None,
        iterations=threshold // 1000 - 1,
        episodes_per_iteration=1000,
        cuda_available=True,
        training_actor_device="auto",
    )
    assert not resolve_compile_updates(
        None,
        iterations=threshold,
        episodes_per_iteration=1,
        cuda_available=False,
        training_actor_device="auto",
    )
    assert not resolve_compile_updates(
        None,
        iterations=threshold,
        episodes_per_iteration=1,
        cuda_available=True,
        training_actor_device="cpu",
    )
