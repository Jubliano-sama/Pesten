from benchmark_ppo_rollout_workspaces import select_workspaces


def test_workspace_selection_requires_exact_material_speedup() -> None:
    summaries = {
        "baseline": {"median": {"total_seconds": 4.0}},
        "workspaces": {"median": {"total_seconds": 3.9}},
    }
    exact = {
        "rollout_hash_exact": True,
        "actor_state_hash_exact": True,
        "critic_state_hash_exact": True,
    }
    assert select_workspaces(
        summaries,
        exact,
    )["selected_reuse_rollout_workspaces"]
    assert not select_workspaces(
        summaries,
        dict(exact, rollout_hash_exact=False),
    )["selected_reuse_rollout_workspaces"]
