from benchmark_ppo_direct_sampling import select_direct_sampling


def test_direct_sampling_requires_exact_material_speedup() -> None:
    summaries = {
        "categorical": {"median": {"total_seconds": 4.0}},
        "direct": {"median": {"total_seconds": 3.9}},
    }
    exact = {
        "rollout_hash_exact": True,
        "actor_state_hash_exact": True,
        "critic_state_hash_exact": True,
    }
    assert select_direct_sampling(
        summaries,
        exact,
    )["selected_direct_rollout_sampling"]
    assert not select_direct_sampling(
        summaries,
        dict(exact, actor_state_hash_exact=False),
    )["selected_direct_rollout_sampling"]
