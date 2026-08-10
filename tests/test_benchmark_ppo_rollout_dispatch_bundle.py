from benchmark_ppo_rollout_dispatch_bundle import select_bundle


def test_bundle_selection_requires_exact_material_speedup() -> None:
    summaries = {
        "baseline": {"median": {"total_seconds": 4.0}},
        "bundle": {"median": {"total_seconds": 3.9}},
    }
    exact = {
        "rollout_hash_exact": True,
        "actor_state_hash_exact": True,
        "critic_state_hash_exact": True,
    }
    assert select_bundle(
        summaries,
        exact,
    )["selected_rollout_dispatch_bundle"]
    assert not select_bundle(
        summaries,
        dict(exact, rollout_hash_exact=False),
    )["selected_rollout_dispatch_bundle"]
