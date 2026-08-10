from benchmark_ppo_numpy_rollout_masks import select_numpy_masks


def test_numpy_mask_selection_requires_exact_material_speedup() -> None:
    summaries = {
        "torch": {"median": {"total_seconds": 4.0}},
        "numpy": {"median": {"total_seconds": 3.9}},
    }
    exact = {
        "rollout_hash_exact": True,
        "actor_state_hash_exact": True,
        "critic_state_hash_exact": True,
    }
    assert select_numpy_masks(
        summaries,
        exact,
    )["selected_numpy_rollout_masks"]
    assert not select_numpy_masks(
        summaries,
        dict(exact, actor_state_hash_exact=False),
    )["selected_numpy_rollout_masks"]
