from benchmark_ppo_cpu_critic_reuse import select_cpu_critic_reuse


def test_reuse_selection_requires_speed_and_exact_outputs() -> None:
    medians = {
        "separate_forward": {"total_seconds": 4.0},
        "reused_forward": {"total_seconds": 3.0},
    }
    assert select_cpu_critic_reuse(
        medians,
        1.2,
        True,
    )["selected_reuse_cpu_critic_forward"]
    assert not select_cpu_critic_reuse(
        medians,
        1.04,
        True,
    )["selected_reuse_cpu_critic_forward"]
    assert not select_cpu_critic_reuse(
        medians,
        1.2,
        False,
    )["selected_reuse_cpu_critic_forward"]
