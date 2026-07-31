import copy
from pathlib import Path

import numpy as np
import torch

from ai_observation import ACTION_COUNT, OBSERVATION_SIZE
from batched_suit_symmetry import (
    SUIT_PERMUTATIONS,
    BatchedSuitAveragedActor,
    _permutation_indices,
)
from ppo2 import build_action_masks
from projected_ppo import (
    collect_common_projected_batch,
    configure_projected_ppo,
    train_common_batch_candidate,
)


CHAMPION = Path(__file__).resolve().parents[1] / "ppo_actor_100run3.pth"


def test_projected_gradient_matches_explicit_24_view_tensor_program():
    torch.manual_seed(71)
    base = torch.nn.Linear(OBSERVATION_SIZE, ACTION_COUNT)
    manual_base = copy.deepcopy(base)
    observations = torch.randn(3, OBSERVATION_SIZE)
    weights = torch.randn(3, ACTION_COUNT)
    projected = BatchedSuitAveragedActor(
        base,
        alpha=1.0,
        aggregation="logits",
        base_chunk_size=2,
    )

    (projected(observations) * weights).sum().backward()

    observation_indices, action_indices = _permutation_indices()
    transformed = torch.gather(
        observations.unsqueeze(1).expand(
            -1,
            len(SUIT_PERMUTATIONS),
            -1,
        ),
        2,
        observation_indices.unsqueeze(0).expand(
            len(observations),
            -1,
            -1,
        ),
    )
    transformed_logits = manual_base(
        transformed.reshape(-1, OBSERVATION_SIZE)
    ).reshape(len(observations), len(SUIT_PERMUTATIONS), ACTION_COUNT)
    mapped = torch.gather(
        transformed_logits,
        2,
        action_indices.unsqueeze(0).expand(
            len(observations),
            -1,
            -1,
        ),
    )
    manual = mapped.mean(1)
    (manual * weights).sum().backward()

    for actual, expected in zip(base.parameters(), manual_base.parameters()):
        # Chunked backprop changes only the floating-point accumulation order
        # for the shared parameter gradient.
        torch.testing.assert_close(
            actual.grad,
            expected.grad,
            atol=1e-5,
            rtol=5e-5,
        )


def test_configuration_uses_one_projected_rollout_and_update_actor(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    ppo, base = configure_projected_ppo(
        CHAMPION,
        learning_rate=1e-6,
        rollout_batch_size=2,
        base_chunk_size=2,
        torch_threads=2,
    )

    assert ppo.actor is ppo.rollout_actor
    assert ppo.actor is ppo._actor_update_model
    assert ppo.actor.actor is base
    optimized = {
        id(parameter)
        for group in ppo.actor_optim.param_groups
        for parameter in group["params"]
    }
    assert optimized == {id(parameter) for parameter in base.parameters()}
    observations = torch.zeros(2, OBSERVATION_SIZE)
    assert torch.equal(
        ppo.get_action_masks(observations),
        build_action_masks(observations),
    )


def test_real_projected_common_batch_and_update_are_exact_and_audited(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    np.random.seed(84_601_777)
    torch.manual_seed(84_601_777)
    ppo, _ = configure_projected_ppo(
        CHAMPION,
        learning_rate=1e-6,
        rollout_batch_size=2,
        base_chunk_size=8,
        torch_threads=2,
    )
    common = collect_common_projected_batch(
        ppo,
        games=2,
        temperature=0.8,
        diagnostic_chunk_size=8,
    )

    assert common["max_initial_ratio_error"] < 0.02
    assert common["rollout"]["deal_seed_count"] == 2
    assert common["rollout"]["deal_seed_rng_replay_verified"]
    assert len(common["rollout"]["deal_seeds"]) == 2
    assert common["payoff_audit"]["terminal_payoffs_verified"]

    candidate, base = configure_projected_ppo(
        CHAMPION,
        learning_rate=1e-6,
        rollout_batch_size=2,
        base_chunk_size=8,
        torch_threads=2,
    )
    result = train_common_batch_candidate(
        candidate,
        base,
        common,
        {
            "name": "test",
            "learning_rate": 1e-6,
            "kl_coefficient": 0.2,
            "entropy_coefficient": 0.0,
            "target_kl": 1e-3,
        },
        update_epochs=1,
        minibatch_size=16,
        temperature=0.8,
        permutation_seed=84_602_777,
        diagnostic_chunk_size=8,
    )

    assert result["common_batch_sha256"] == common["sha256"]
    assert result["initial_exact_behavior_kl_old_to_new"] == 0.0
    assert result["max_initial_ratio_error"] < 0.02
    assert result["base_actor_sha256_before"] != result[
        "base_actor_sha256_after"
    ]
    assert result["peak_augmented_observations_per_minibatch"] <= 16 * 24


def test_target_kl_overshoot_rolls_back_entire_epoch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    np.random.seed(84_603_777)
    torch.manual_seed(84_603_777)
    behavior, _ = configure_projected_ppo(
        CHAMPION,
        learning_rate=1e-6,
        rollout_batch_size=2,
        base_chunk_size=8,
        torch_threads=2,
    )
    common = collect_common_projected_batch(
        behavior,
        games=2,
        temperature=0.8,
        diagnostic_chunk_size=8,
    )
    candidate, base = configure_projected_ppo(
        CHAMPION,
        learning_rate=1e-3,
        rollout_batch_size=2,
        base_chunk_size=8,
        torch_threads=2,
    )
    result = train_common_batch_candidate(
        candidate,
        base,
        common,
        {
            "name": "forced_rollback",
            "learning_rate": 1e-3,
            "kl_coefficient": 0.0,
            "entropy_coefficient": 0.0,
            "target_kl": 1e-12,
        },
        update_epochs=1,
        minibatch_size=16,
        temperature=0.8,
        permutation_seed=84_604_777,
        diagnostic_chunk_size=8,
    )

    assert result["update_epochs"][0]["rolled_back"]
    assert not result["update_epochs"][0]["accepted"]
    assert result["accepted_epochs"] == 0
    assert not result["policy_updated"]
    assert result["final_exact_behavior_kl_old_to_new"] == 0.0
    assert result["target_kl_satisfied"]
