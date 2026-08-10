from benchmark_ppo_past_policy_pool import select_past_policy_pool


def test_pool_selection_requires_paired_material_speedup() -> None:
    medians = {
        "full": {"seconds": 4.0},
        "pool8": {"seconds": 3.2},
    }
    assert select_past_policy_pool(
        medians,
        1.2,
    )["selected_past_policy_pool_size"] == 8
    assert select_past_policy_pool(
        medians,
        1.04,
    )["selected_past_policy_pool_size"] is None
