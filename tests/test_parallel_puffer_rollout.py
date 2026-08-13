import pytest


pufferlib = pytest.importorskip("pufferlib")
if not str(getattr(pufferlib, "__version__", "")).startswith("4"):
    pytest.skip("requires pinned PufferLib 4 runtime", allow_module_level=True)

from benchmark_agents import load_ppo_agent
from benchmark_puffer_vs_puffer import run_parallel_benchmark
from parallel_puffer_rollout import collect_league_batch_parallel
from temporal_league import collect_league_batch, evaluate_paired
from temporal_puffer_agent import temporal_agent_factory
from temporal_puffer_v4 import (
    HawkRGLRUTemporalPolicy,
    NativePufferTemporalPolicy,
    policy_model_config,
)
from train_puffer_fast import _append_jsonl_idempotent, _effective_vector_workers
from vector_puffer_rollout import (
    VectorPufferRolloutPool,
    _build_tasks,
    collect_league_batch_vector_parallel,
)
import torch
import vector_puffer_rollout


def test_parallel_rollout_is_exactly_equal_to_sequential_schedule():
    policy = NativePufferTemporalPolicy(
        hidden_size=16,
        num_layers=1,
        mode_conditioned_heads=True,
    ).eval()
    champion = load_ppo_agent("ppo_actor_100run3.pth", "cpu")
    mixture = (
        ("current_selfplay", 2),
        ("historical_temporal", 1),
        ("ppo_champion", 1),
        ("bayesian_tree", 1),
        ("random_rule", 1),
    )
    arguments = {
        "games": 8,
        "seed": 20_260_812,
        "schedule_offset": 3,
        "mixture": mixture,
        "player_counts": (2, 3, 7),
        "max_turns": 5,
    }
    expected_episodes, expected_metadata = collect_league_batch(
        policy,
        champion,
        [policy],
        **arguments,
    )
    actual_episodes, actual_metadata = collect_league_batch_parallel(
        policy,
        workers=2,
        chunks_per_worker=2,
        **arguments,
    )

    assert actual_metadata == expected_metadata
    assert actual_episodes == expected_episodes


def test_vector_tasks_carry_the_training_turn_limit_to_every_backend():
    tasks, _, _ = _build_tasks(
        games=4,
        seed=7,
        schedule_offset=0,
        schedule=("current_selfplay", "bayesian_tree"),
        player_counts=(8, 2),
        workers=2,
        prepack_training=False,
        max_turns=17,
    )

    assert tasks
    assert {task["kind"] for task in tasks} == {"tree", "vector"}
    assert all(task["max_turns"] == 17 for task in tasks)


def test_parallel_rollout_validates_worker_count():
    policy = NativePufferTemporalPolicy(hidden_size=8)
    with pytest.raises(ValueError, match="workers"):
        collect_league_batch_parallel(
            policy,
            games=1,
            seed=1,
            mixture=(("random_rule", 1),),
            player_counts=(2,),
            workers=0,
        )


def test_vector_worker_count_does_not_replicate_idle_model_pools():
    assert _effective_vector_workers(
        10,
        ("current_selfplay", "historical_temporal"),
        (2, 3, 4, 5, 6, 7, 8),
    ) == 7
    assert _effective_vector_workers(
        5,
        ("current_selfplay",),
        (2, 3, 4, 5, 6, 7, 8),
    ) == 5


def test_vector_worker_count_preserves_tree_parallelism():
    assert _effective_vector_workers(
        10,
        ("current_selfplay", "bayesian_tree"),
        (2, 3, 4, 5, 6, 7, 8),
    ) == 10


def test_parallel_benchmark_matches_sequential_pairs(tmp_path):
    torch.manual_seed(13)
    candidate = NativePufferTemporalPolicy(
        hidden_size=8, mode_conditioned_heads=True
    ).eval()
    torch.manual_seed(17)
    opponent = NativePufferTemporalPolicy(
        hidden_size=8, mode_conditioned_heads=True
    ).eval()
    paths = []
    for name, policy in (("candidate", candidate), ("opponent", opponent)):
        path = tmp_path / f"{name}.pt"
        torch.save(
            {
                "model": {
                    "hidden_size": 8,
                    "num_layers": 1,
                    "mode_conditioned_heads": True,
                },
                "state_dict": policy.state_dict(),
            },
            path,
        )
        paths.append(path)

    expected = evaluate_paired(
        temporal_agent_factory(candidate, deterministic=True),
        temporal_agent_factory(opponent, deterministic=True),
        pairs=6,
        seed=2041,
        candidate_name="candidate",
        opponent_name="opponent",
    )
    actual = run_parallel_benchmark(
        paths[0],
        paths[1],
        pairs=6,
        seed=2041,
        workers=2,
        chunks_per_worker=2,
    )

    for key in ("pairs", "games", "wins", "losses", "draws", "score"):
        assert actual[key] == expected[key]
    assert actual["candidate_wins_as_seat_0"] == expected["wins_as_seat_0"]
    assert actual["candidate_wins_as_seat_1"] == expected["wins_as_seat_1"]


def test_training_curve_append_is_idempotent_by_epoch_and_state(tmp_path):
    path = tmp_path / "curve.jsonl"
    row = {"epoch": 7, "checkpoint": {"state_sha256": "abc"}}
    _append_jsonl_idempotent(path, row)
    _append_jsonl_idempotent(path, row)
    assert len(path.read_text().splitlines()) == 1

    conflicting = {"epoch": 7, "checkpoint": {"state_sha256": "def"}}
    with pytest.raises(RuntimeError, match="conflicting"):
        _append_jsonl_idempotent(path, conflicting)


def test_persistent_vector_pool_refreshes_shared_policy_exactly():
    torch.manual_seed(47)
    policy = NativePufferTemporalPolicy(
        hidden_size=8, mode_conditioned_heads=True
    ).eval()
    arguments = {
        "games": 12,
        "seed": 2047,
        "schedule_offset": 5,
        "schedule": ("current_selfplay", "random_rule"),
        "player_counts": (2, 3),
        "max_turns": 5,
    }
    with VectorPufferRolloutPool(policy, workers=2) as pool:
        with torch.no_grad():
            policy.puffer_policy.decoder.card_and_pass.bias.add_(0.5)
        actual_episodes, actual_metadata = pool.collect(policy, **arguments)
    expected_episodes, expected_metadata = collect_league_batch_vector_parallel(
        policy, workers=2, **arguments
    )
    assert actual_episodes == expected_episodes
    for name in (
        "games",
        "decisions",
        "total_turns",
        "trajectories",
        "player_counts",
        "deck_count_by_player_count",
    ):
        assert actual_metadata[name] == expected_metadata[name]


def test_vector_worker_loads_a_genuinely_frozen_historical_pool(tmp_path):
    torch.manual_seed(81)
    current = NativePufferTemporalPolicy(
        hidden_size=8, mode_conditioned_heads=True
    ).eval()
    torch.manual_seed(82)
    historical = NativePufferTemporalPolicy(
        hidden_size=8, mode_conditioned_heads=True
    ).eval()
    historical_path = tmp_path / "historical.pt"
    torch.save(
        {
            "model": policy_model_config(historical),
            "state_dict": historical.state_dict(),
        },
        historical_path,
    )

    vector_puffer_rollout._initialize_worker(
        current.state_dict(),
        policy_model_config(current),
        "ppo_actor_100run3.pth",
        (str(historical_path),),
    )

    loaded = vector_puffer_rollout._WORKER_HISTORICAL
    assert len(loaded) == 1
    assert loaded[0] is not vector_puffer_rollout._WORKER_POLICY
    for name, value in historical.state_dict().items():
        assert torch.equal(loaded[0].state_dict()[name], value)
    assert all(not parameter.requires_grad for parameter in loaded[0].parameters())


def test_vector_pool_supports_hawk_rglru_policy_registry():
    torch.manual_seed(71)
    policy = HawkRGLRUTemporalPolicy(
        hidden_size=16,
        num_layers=2,
        recurrent_size=20,
        mlp_expansion=2,
    ).eval()
    with VectorPufferRolloutPool(policy, workers=2) as pool:
        episodes, metadata = pool.collect(
            policy,
            games=6,
            seed=2071,
            schedule_offset=0,
            schedule=("current_selfplay", "ppo_champion", "random_rule"),
            player_counts=(2, 3, 4, 5, 6, 7, 8),
        )
    assert episodes
    assert metadata["trajectories"] == len(episodes)
    assert sum(metadata["games"].values()) == 6
    assert set(metadata["player_counts"]) <= {
        str(players) for players in range(2, 9)
    }
