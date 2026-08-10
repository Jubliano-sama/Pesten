from types import SimpleNamespace

from benchmark_ppo_rollout_trace import select_trace, worker_command


def test_worker_command_changes_only_trace_mode() -> None:
    args = SimpleNamespace(
        seed=84_950_001,
        games=1024,
        batch_size=256,
        threads=8,
        native_smart_policy=True,
    )
    eager = worker_command(args, traced=False)
    traced = worker_command(args, traced=True)
    assert "--no-trace-cpu-rollout-actor" in eager
    assert "--trace-cpu-rollout-actor" in traced
    assert len(eager) == len(traced)


def test_selection_requires_exact_material_speedup() -> None:
    summaries = {
        "eager": {"median": {"total_seconds": 4.0}},
        "traced": {"median": {"total_seconds": 3.0}},
    }
    exact = {
        "rollout_hash_exact": True,
        "actor_state_hash_exact": True,
        "critic_state_hash_exact": True,
    }
    assert select_trace(summaries, exact)[
        "selected_trace_cpu_rollout_actor"
    ]
    assert not select_trace(
        summaries,
        dict(exact, rollout_hash_exact=False),
    )["selected_trace_cpu_rollout_actor"]
