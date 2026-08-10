import numpy as np
import torch

from ai_observation import ACTION_COUNT, OBSERVATION_SIZE
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    POLICY_RL,
    POLICY_SMART,
    collect_batched_rollout,
    collect_coordinated_batched_rollouts,
)


def _engines() -> list[BatchedPesten]:
    engines = []
    seed_start = 85_142_001
    for player_count in (2, 3, 5):
        game_count = 8
        engines.append(
            BatchedPesten(
                np.full(
                    (game_count, player_count),
                    POLICY_RL,
                    dtype=np.int8,
                ),
                np.full(
                    (game_count, player_count),
                    "current",
                    dtype=object,
                ),
                np.arange(
                    seed_start,
                    seed_start + game_count,
                    dtype=np.uint32,
                ),
                max_turns=500,
                native_rollout_engine=True,
                record_public_events=False,
            )
        )
        seed_start += game_count
    return engines


def _policy() -> dict[str, ActorPolicy]:
    torch.manual_seed(85_142_101)
    actor = torch.nn.Sequential(
        torch.nn.Linear(OBSERVATION_SIZE, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, ACTION_COUNT),
    ).eval()
    return {
        "current": ActorPolicy(
            actor=actor,
            device=torch.device("cpu"),
            temperature=0.8,
            collect_data=True,
            greedy=True,
        )
    }


def test_coordinated_engines_preserve_deterministic_rollouts() -> None:
    policies = _policy()
    sequential = [
        collect_batched_rollout(
            engine,
            policies,
            output_device=torch.device("cpu"),
            direct_action_sampling=True,
            numpy_action_masks=True,
            reuse_rollout_workspaces=True,
        )
        for engine in _engines()
    ]
    coordinated = collect_coordinated_batched_rollouts(
        _engines(),
        policies,
        output_device=torch.device("cpu"),
        direct_action_sampling=True,
        numpy_action_masks=True,
        reuse_rollout_workspaces=True,
    )

    for expected, actual in zip(sequential, coordinated):
        assert torch.equal(actual.observations, expected.observations)
        assert torch.equal(actual.actions, expected.actions)
        assert torch.allclose(
            actual.log_probabilities,
            expected.log_probabilities,
            rtol=0.0,
            atol=1e-6,
        )
        assert torch.equal(actual.rewards, expected.rewards)
        assert actual.episode_lengths == expected.episode_lengths
        for metric in (
            "environment_turns",
            "completed_games",
            "policy_decisions",
            "wins",
            "draws",
        ):
            assert getattr(actual, metric) == getattr(expected, metric)
    assert sum(result.inference_calls for result in coordinated) < sum(
        result.inference_calls for result in sequential
    )


def _smart_rl_engine() -> BatchedPesten:
    game_count = 32
    policy_kinds = np.tile(
        np.asarray([POLICY_SMART, POLICY_RL], dtype=np.int8),
        (game_count, 1),
    )
    actor_keys = np.full((game_count, 2), None, dtype=object)
    actor_keys[:, 1] = "current"
    return BatchedPesten(
        policy_kinds,
        actor_keys,
        np.arange(
            85_142_201,
            85_142_201 + game_count,
            dtype=np.uint32,
        ),
        max_turns=500,
        native_rollout_engine=True,
        record_public_events=False,
    )


def test_draining_to_policy_boundary_preserves_greedy_rollouts() -> None:
    policies = _policy()
    baseline = collect_batched_rollout(
        _smart_rl_engine(),
        policies,
        output_device=torch.device("cpu"),
        direct_action_sampling=True,
        numpy_action_masks=True,
        reuse_rollout_workspaces=True,
        drain_non_neural=False,
    )
    drained = collect_batched_rollout(
        _smart_rl_engine(),
        policies,
        output_device=torch.device("cpu"),
        direct_action_sampling=True,
        numpy_action_masks=True,
        reuse_rollout_workspaces=True,
        drain_non_neural=True,
    )

    assert torch.equal(drained.observations, baseline.observations)
    assert torch.equal(drained.actions, baseline.actions)
    assert torch.allclose(
        drained.log_probabilities,
        baseline.log_probabilities,
        rtol=0.0,
        atol=1e-6,
    )
    assert torch.equal(drained.rewards, baseline.rewards)
    assert drained.episode_lengths == baseline.episode_lengths
    assert drained.environment_turns == baseline.environment_turns
    assert drained.completed_games == baseline.completed_games
    assert drained.policy_decisions == baseline.policy_decisions
    assert drained.wins == baseline.wins
    assert drained.draws == baseline.draws
    assert drained.inference_calls < baseline.inference_calls
