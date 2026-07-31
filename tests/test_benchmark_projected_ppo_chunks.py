from pathlib import Path

import torch

from benchmark_projected_ppo_chunks import (
    actor_state_difference,
    run_benchmark,
    structured_sha256,
)


CHAMPION = Path(__file__).resolve().parents[1] / "ppo_actor_100run3.pth"


def test_structured_hash_is_stable_and_tensor_sensitive():
    left = {
        "state": {0: {"step": torch.tensor(1), "x": torch.tensor([1.0])}},
        "groups": [{"lr": 3e-6, "params": [0]}],
    }
    right = {
        "groups": [{"params": [0], "lr": 3e-6}],
        "state": {0: {"x": torch.tensor([1.0]), "step": torch.tensor(1)}},
    }
    assert structured_sha256(left) == structured_sha256(right)
    right["state"][0]["x"][0] = 2
    assert structured_sha256(left) != structured_sha256(right)


def test_actor_state_difference_counts_and_measures_changes():
    reference = {
        "a": torch.tensor([1.0, 2.0]),
        "b": torch.tensor([3.0]),
    }
    candidate = {
        "a": torch.tensor([1.0, 2.5]),
        "b": torch.tensor([2.0]),
    }
    result = actor_state_difference(candidate, reference)
    assert result["differing_parameter_elements"] == 2
    assert result["total_parameter_elements"] == 3
    assert result["max_abs_parameter_difference"] == 1.0
    assert not result["bit_identical"]


def test_tiny_chunk_benchmark_resets_every_state_and_common_batch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "chunks"
    payload = run_benchmark(
        CHAMPION,
        output,
        chunk_sizes=(4, 8),
        seed=84_901_001,
        critic_games=2,
        critic_epochs=1,
        games=2,
        update_epochs=1,
        minibatch_size=16,
        diagnostic_chunk_size=8,
        rollout_batch_size=2,
        temperature=0.8,
        torch_threads=2,
    )

    assert payload["sealed_seeds_or_outcomes_used"] is False
    assert payload["reset_contract"]["identical_actor"]
    assert payload["reset_contract"]["identical_critic"]
    assert payload["reset_contract"]["identical_actor_optimizer"]
    assert payload["reset_contract"]["identical_critic_optimizer"]
    assert payload["reset_contract"]["identical_common_batch"]
    assert payload["reference_chunk_size"] == 8
    assert len(payload["trials"]) == 2
    assert len(
        {trial["common_batch_sha256"] for trial in payload["trials"]}
    ) == 1
    assert len(
        {
            trial["initial_projected_logits_sha256"]
            for trial in payload["trials"]
        }
    ) == 1
    initial_states = [
        trial["initial_state"]
        for trial in payload["trials"]
    ]
    assert initial_states[0] == initial_states[1]
    for trial in payload["trials"]:
        assert trial["exact_initial_logits_match_behavior"]
        assert trial["max_initial_ratio_error"] < 0.02
        assert trial["target_kl_satisfied"]
        assert trial["final_exact_behavior_kl_old_to_new"] <= trial[
            "target_kl"
        ]
        assert (output / trial["checkpoint"]["path"]).is_file()
        assert "checkpoint_difference_vs_reference" in trial
    assert payload["trials"][0][
        "peak_augmented_base_actor_call_observations"
    ] == 4 * 24
    assert payload["trials"][1][
        "peak_augmented_base_actor_call_observations"
    ] == 8 * 24
