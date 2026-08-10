from benchmark_ppo_public_events import select_public_event_recording


def test_public_event_selection_requires_paired_exact_speedup() -> None:
    medians = {"recorded": 4.0, "disabled": 3.8}
    assert select_public_event_recording(
        medians,
        1.04,
        True,
    )["selected_disable_rollout_public_events"]
    assert not select_public_event_recording(
        medians,
        1.0,
        True,
    )["selected_disable_rollout_public_events"]
    assert not select_public_event_recording(
        medians,
        1.04,
        False,
    )["selected_disable_rollout_public_events"]
