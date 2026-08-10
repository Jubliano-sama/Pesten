from benchmark_ppo_native_rollout_engine import select_native_engine


def test_native_engine_selection_requires_exact_material_speedup() -> None:
    summaries = {
        "python": {"median": {"total_seconds": 4.0}},
        "native": {"median": {"total_seconds": 3.9}},
    }
    exact = {
        "rollout_hash_exact": True,
        "actor_state_hash_exact": True,
        "critic_state_hash_exact": True,
    }
    assert select_native_engine(
        summaries,
        exact,
    )["selected_native_rollout_engine"]
    assert not select_native_engine(
        summaries,
        dict(exact, rollout_hash_exact=False),
    )["selected_native_rollout_engine"]
