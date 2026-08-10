from benchmark_ppo_cpu_affinity import select_affinity


def test_affinity_selection_requires_exact_material_speedup() -> None:
    summaries = {
        "unpinned": {"median": {"total_seconds": 4.0}},
        "pinned": {"median": {"total_seconds": 3.8}},
    }
    exact = {
        "rollout_hash_exact": True,
        "actor_state_hash_exact": True,
        "critic_state_hash_exact": True,
    }
    assert select_affinity(
        summaries,
        exact,
    )["selected_physical_core_affinity"]
    assert not select_affinity(
        summaries,
        dict(exact, critic_state_hash_exact=False),
    )["selected_physical_core_affinity"]
