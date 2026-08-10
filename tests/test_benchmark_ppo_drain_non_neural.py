from benchmark_ppo_drain_non_neural import select_non_neural_drain


def test_drain_selection_requires_paired_material_speedup() -> None:
    medians = {
        "one_step": {"seconds": 4.0},
        "drain_to_policy": {"seconds": 3.0},
    }
    assert select_non_neural_drain(
        medians,
        1.2,
    )["selected_drain_non_neural_rollout"]
    assert not select_non_neural_drain(
        medians,
        1.01,
    )["selected_drain_non_neural_rollout"]
