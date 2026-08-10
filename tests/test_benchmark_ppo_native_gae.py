from types import SimpleNamespace

from benchmark_ppo_native_gae import exact_contract, worker_command


def test_native_gae_worker_command_switches_one_flag() -> None:
    args = SimpleNamespace(
        games=32,
        warmup_games=4,
        batch_size=64,
        threads=2,
        rollout_device="cpu",
        compile_updates=False,
    )
    tensor = worker_command(args, 85_120_030, native=False)
    native = worker_command(args, 85_120_030, native=True)

    assert "--no-native-gae" in tensor
    assert "--native-gae" in native
    assert tensor[: tensor.index("--no-native-gae")] == (
        native[: native.index("--native-gae")]
    )


def test_exact_contract_covers_rollout_and_both_models() -> None:
    baseline = {
        "rollout_sha256": "rollout",
        "actor_state_sha256": "actor",
        "critic_state_sha256": "critic",
        "collected_policy_decisions": 10,
        "environment_turns": 20,
    }
    assert exact_contract(baseline, dict(baseline))
    changed = dict(baseline, critic_state_sha256="changed")
    assert not exact_contract(baseline, changed)
