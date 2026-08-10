from benchmark_ppo_past_policy_pool_refinement import select_refined_pool


def test_refinement_requires_paired_material_speedup() -> None:
    medians = {
        "pool8": {"seconds": 4.0},
        "pool2": {"seconds": 3.0},
    }
    assert select_refined_pool(
        medians,
        1.2,
    )["selected_past_policy_pool_size"] == 2
    assert select_refined_pool(
        medians,
        1.09,
    )["selected_past_policy_pool_size"] == 8
