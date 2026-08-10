from types import SimpleNamespace

from benchmark_ppo_rollout_batch_sweep import (
    exact_hash_contract,
    select_batch_size,
    summarize,
    worker_command,
)


def _run(total: float, suffix: str = "same") -> dict:
    return {
        "rollout_seconds": total * 0.7,
        "prep_seconds": total * 0.1,
        "update_seconds": total * 0.2,
        "total_seconds": total,
        "rss_peak_mib": 100.0,
        "rollout_buffer_peak_mib": 10.0,
        "rollout_sha256": f"rollout-{suffix}",
        "actor_state_sha256": f"actor-{suffix}",
        "critic_state_sha256": f"critic-{suffix}",
    }


def test_worker_command_binds_only_rollout_batch_size():
    args = SimpleNamespace(seed=84_920_001, games=1024, threads=8)
    command = worker_command(args, 512)
    assert command[command.index("--batch-size") + 1] == "512"
    assert command[command.index("--episodes") + 1] == "1024"
    assert "--trace-cpu-rollout-actor" in command
    assert command[-1] == "--no-compile-updates"


def test_summary_and_exact_hash_contract():
    raw = {
        "256": [_run(2.0), _run(2.2)],
        "512": [_run(1.5), _run(1.7)],
    }
    summary = summarize(raw["256"])
    assert summary["median"]["total_seconds"] == 2.1
    assert summary["repeat_rollout_hash_exact"]
    parity = exact_hash_contract(raw)
    assert parity["rollout_hash_exact"]
    assert parity["actor_state_hash_exact"]
    assert parity["critic_state_hash_exact"]


def test_selection_requires_material_speedup_and_exact_state():
    summaries = {
        "256": {"median": {"total_seconds": 2.0}},
        "512": {"median": {"total_seconds": 1.5}},
    }
    exact = {
        "rollout_hash_exact": True,
        "actor_state_hash_exact": True,
        "critic_state_hash_exact": True,
    }
    selected = select_batch_size(summaries, exact)
    assert selected["selected_batch_size"] == 512
    inexact = dict(exact, actor_state_hash_exact=False)
    rejected = select_batch_size(summaries, inexact)
    assert rejected["selected_batch_size"] == 256


def test_selection_rejects_noise_sized_improvement():
    summaries = {
        "256": {"median": {"total_seconds": 2.0}},
        "512": {"median": {"total_seconds": 1.99}},
    }
    exact = {
        "rollout_hash_exact": True,
        "actor_state_hash_exact": True,
        "critic_state_hash_exact": True,
    }
    decision = select_batch_size(summaries, exact)
    assert decision["selected_batch_size"] == 256
