from benchmark_ppo_coordinated_engines import select_coordinated_engines


def test_coordination_selection_requires_paired_material_speedup() -> None:
    medians = {
        "sequential_engines": {"seconds": 4.0},
        "coordinated_engines": {"seconds": 3.0},
    }
    assert select_coordinated_engines(
        medians,
        1.2,
    )["selected_coordinate_rollout_engines"]
    assert not select_coordinated_engines(
        medians,
        1.04,
    )["selected_coordinate_rollout_engines"]
