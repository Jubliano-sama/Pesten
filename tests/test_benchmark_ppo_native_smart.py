from benchmark_ppo_native_smart import select_native


def test_native_selection_requires_exact_material_speedup() -> None:
    summaries = {
        "python": {"median": {"total_seconds": 4.0}},
        "native": {"median": {"total_seconds": 3.0}},
    }
    exact = {
        "rollout_hash_exact": True,
        "actor_state_hash_exact": True,
        "critic_state_hash_exact": True,
    }
    assert select_native(summaries, exact)["selected_native_smart_policy"]
    assert not select_native(
        summaries,
        dict(exact, critic_state_hash_exact=False),
    )["selected_native_smart_policy"]
