from benchmark_ppo_agent_set_coalescing import select_coalescing


def test_coalescing_selection_requires_paired_material_speedup() -> None:
    medians = {"baseline": 4.0, "coalesced": 3.2}
    assert select_coalescing(
        medians,
        1.2,
    )["selected_coalesce_rollout_agent_sets"]
    assert not select_coalescing(
        medians,
        1.04,
    )["selected_coalesce_rollout_agent_sets"]
