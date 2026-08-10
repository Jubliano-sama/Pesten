from benchmark_ppo_rollout_device import select_rollout_device


def test_cuda_selection_requires_speed_and_ratio_gates() -> None:
    medians = {
        "cpu": {"total_seconds": 2.0, "rollout_seconds": 1.5},
        "cuda_fp32": {"total_seconds": 1.0, "rollout_seconds": 0.75},
    }
    assert select_rollout_device(
        medians,
        1.8,
        0.001,
    )["selected_rollout_device"] == "cuda"
    assert select_rollout_device(
        medians,
        1.2,
        0.001,
    )["selected_rollout_device"] == "cpu"
    assert select_rollout_device(
        medians,
        1.8,
        0.02,
    )["selected_rollout_device"] == "cpu"
