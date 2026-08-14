import pytest
import sys


pufferlib = pytest.importorskip("pufferlib")
if not str(getattr(pufferlib, "__version__", "")).startswith("4"):
    pytest.skip("requires pinned PufferLib 4 runtime", allow_module_level=True)

from benchmark_agents import load_ppo_agent
from benchmark_puffer_vs_puffer import run_parallel_benchmark
from parallel_puffer_rollout import collect_league_batch_parallel
from temporal_league import collect_league_batch, evaluate_paired
from temporal_puffer_agent import temporal_agent_factory
from temporal_puffer_v4 import (
    HawkResidualTemporalPolicy,
    HawkRGLRUTemporalPolicy,
    MinGRUResidualTemporalPolicy,
    NativePufferTemporalPolicy,
    policy_model_config,
)
from train_puffer_fast import (
    AUTO_INT8_ROLLOUT_PARAMETER_THRESHOLD,
    _append_jsonl_idempotent,
    _configure_trainable_policy,
    _effective_vector_workers,
    _prune_frozen_optimizer_state,
    _resolve_compile_updates,
    _resolve_rollout_max_public_events,
    _resolve_rollout_precision,
    _resolve_update_precision,
)
from vector_puffer_rollout import (
    ROLLOUT_PRECISION_INT8_RECURRENT,
    ROLLOUT_PRECISION_INT8_RESIDUAL_BASE,
    VectorPufferRolloutPool,
    _build_rollout_policy,
    _build_tasks,
    _refresh_worker_policy,
    collect_league_batch_vector_parallel,
)
import torch
import vector_puffer_rollout


def test_fast_trainer_defaults_to_nonlagged_compiled_path(monkeypatch):
    from train_puffer_fast import parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_puffer_fast.py",
            "--resume",
            "source.pt",
            "--output-dir",
            "output",
        ],
    )
    args = parse_args()

    assert args.pipelined_rollouts is False
    assert args.compile_updates is None
    assert args.update_epochs == 1
    assert args.full_rollout_diagnostics is False
    assert args.rollout_precision == "auto"
    assert args.update_precision == "auto"


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


def test_vector_worker_count_uses_requested_sharded_pool():
    assert _effective_vector_workers(
        10,
        ("current_selfplay", "historical_temporal"),
        (2, 3, 4, 5, 6, 7, 8),
        2048,
    ) == 10
    assert _effective_vector_workers(
        10,
        ("current_selfplay",),
        (2, 3, 4, 5, 6, 7, 8),
        6,
    ) == 6
    assert _effective_vector_workers(
        16,
        ("current_selfplay", "historical_temporal"),
        (2, 3, 4, 5, 6, 7, 8),
        2048,
        ROLLOUT_PRECISION_INT8_RESIDUAL_BASE,
    ) == 8


def test_vector_worker_count_preserves_tree_parallelism():
    assert _effective_vector_workers(
        10,
        ("current_selfplay", "bayesian_tree"),
        (2, 3, 4, 5, 6, 7, 8),
        2048,
    ) == 10


def test_auto_rollout_precision_only_quantizes_large_models():
    assert _resolve_rollout_precision("auto", 20_000) == "float32"
    assert (
        _resolve_rollout_precision(
            "auto", AUTO_INT8_ROLLOUT_PARAMETER_THRESHOLD
        )
        == ROLLOUT_PRECISION_INT8_RECURRENT
    )
    assert _resolve_rollout_precision("float32", 5_000_000) == "float32"
    assert (
        _resolve_rollout_precision(
            "auto", 5_000_000, architecture="min_gru_residual"
        )
        == ROLLOUT_PRECISION_INT8_RESIDUAL_BASE
    )


def test_auto_rollout_event_limit_is_aggressive_only_for_residual_base():
    assert _resolve_rollout_max_public_events(
        None, ROLLOUT_PRECISION_INT8_RESIDUAL_BASE
    ) == 5
    assert _resolve_rollout_max_public_events(
        None, ROLLOUT_PRECISION_INT8_RECURRENT
    ) is None
    assert _resolve_rollout_max_public_events(
        0, ROLLOUT_PRECISION_INT8_RESIDUAL_BASE
    ) is None
    assert _resolve_rollout_max_public_events(3, "float32") == 3
    with pytest.raises(ValueError, match="cannot be negative"):
        _resolve_rollout_max_public_events(-1, "float32")


def test_residual_training_scope_updates_only_top_adapter_layers_and_heads():
    policy = MinGRUResidualTemporalPolicy(
        base_hidden_size=8,
        base_num_layers=1,
        adapter_hidden_size=12,
        adapter_num_layers=4,
    )
    input_width = policy.puffer_policy.adapter.encoder.encoder.in_features
    tokens = torch.randn(2, 3, input_width)
    with torch.no_grad():
        expected_logits, expected_values = policy.forward_native_sequence(tokens)
    scope = _configure_trainable_policy(
        policy, "min_gru_residual", None
    )
    with torch.no_grad():
        actual_logits, actual_values = policy.forward_native_sequence(tokens)

    assert scope["adapter_layers"] == 2
    assert scope["trainable_parameters"] < scope["total_parameters"]
    assert scope["trainable_fraction"] == pytest.approx(
        scope["trainable_parameters"] / scope["total_parameters"]
    )
    assert torch.equal(actual_logits, expected_logits)
    assert torch.equal(actual_values, expected_values)
    assert all(
        not parameter.requires_grad
        for parameter in policy.puffer_policy.base.parameters()
    )
    assert all(
        not parameter.requires_grad
        for parameter in policy.puffer_policy.adapter.encoder.parameters()
    )
    for index, layer in enumerate(policy.puffer_policy.adapter.network.layers):
        assert all(
            parameter.requires_grad == (index >= 2)
            for parameter in layer.parameters()
        )
    assert all(
        parameter.requires_grad
        for parameter in policy.puffer_policy.adapter.decoder.parameters()
    )
    policy.zero_grad(set_to_none=True)
    logits, values = policy.forward_native_sequence(tokens)
    (logits.square().mean() + values.square().mean()).backward()
    assert all(
        parameter.grad is None
        for parameter in policy.puffer_policy.base.parameters()
    )
    for index, layer in enumerate(policy.puffer_policy.adapter.network.layers):
        assert all(
            (parameter.grad is not None) == (index >= 2)
            for parameter in layer.parameters()
        )

    full_scope = _configure_trainable_policy(
        policy, "min_gru_residual", -1
    )
    assert full_scope["mode"] == "all"
    assert full_scope["trainable_fraction"] == 1.0
    assert all(parameter.requires_grad for parameter in policy.parameters())


def test_frozen_optimizer_state_is_pruned_and_remains_reloadable():
    policy = MinGRUResidualTemporalPolicy(
        base_hidden_size=8,
        base_num_layers=1,
        adapter_hidden_size=12,
        adapter_num_layers=3,
    )
    optimizer = torch.optim.Adam(policy.parameters())
    input_width = policy.puffer_policy.adapter.encoder.encoder.in_features
    logits, values = policy.forward_native_sequence(
        torch.randn(2, 3, input_width)
    )
    (logits.square().mean() + values.square().mean()).backward()
    optimizer.step()

    _configure_trainable_policy(policy, "min_gru_residual", 1)
    pruning = _prune_frozen_optimizer_state(optimizer)
    assert pruning["removed_parameters"] > 0
    assert pruning["removed_state_bytes"] > 0
    assert optimizer.state
    assert all(parameter.requires_grad for parameter in optimizer.state)

    resumed = torch.optim.Adam(policy.parameters())
    resumed.load_state_dict(optimizer.state_dict())
    assert _prune_frozen_optimizer_state(resumed)["removed_parameters"] == 0
    assert all(parameter.requires_grad for parameter in resumed.state)


def test_auto_update_precision_prefers_supported_cuda_bfloat16():
    assert (
        _resolve_update_precision(
            "auto", device_type="cuda", bf16_supported=True
        )
        == "bfloat16"
    )
    assert (
        _resolve_update_precision(
            "auto", device_type="cuda", bf16_supported=False
        )
        == "float32"
    )
    assert (
        _resolve_update_precision(
            "auto", device_type="cpu", bf16_supported=True
        )
        == "float32"
    )
    assert (
        _resolve_update_precision(
            "float32", device_type="cuda", bf16_supported=True
        )
        == "float32"
    )
    with pytest.raises(ValueError, match="supported CUDA"):
        _resolve_update_precision(
            "bfloat16", device_type="cpu", bf16_supported=True
        )
    with pytest.raises(ValueError, match="supported CUDA"):
        _resolve_update_precision(
            "bfloat16", device_type="cuda", bf16_supported=False
        )


def test_compile_auto_skips_hidden_large_pipeline_warmup():
    assert not _resolve_compile_updates(
        None,
        pipelined_rollouts=True,
        parameter_count=AUTO_INT8_ROLLOUT_PARAMETER_THRESHOLD,
    )
    assert _resolve_compile_updates(
        None,
        pipelined_rollouts=False,
        parameter_count=5_000_000,
    )
    assert _resolve_compile_updates(
        None,
        pipelined_rollouts=True,
        parameter_count=20_000,
    )
    assert _resolve_compile_updates(
        True,
        pipelined_rollouts=True,
        parameter_count=5_000_000,
    )
    assert not _resolve_compile_updates(
        False,
        pipelined_rollouts=False,
        parameter_count=5_000_000,
    )


def test_int8_rollout_policy_quantizes_only_recurrent_linears():
    torch.manual_seed(49)
    policy = NativePufferTemporalPolicy(
        hidden_size=16,
        num_layers=2,
        mode_conditioned_heads=True,
    ).eval()
    quantized = _build_rollout_policy(
        policy, ROLLOUT_PRECISION_INT8_RECURRENT
    )
    eager_quantized = _build_rollout_policy(
        policy,
        ROLLOUT_PRECISION_INT8_RECURRENT,
        trace_min_gru=False,
    )

    dynamic_linear = torch.ao.nn.quantized.dynamic.Linear
    assert quantized is not policy
    assert sum(
        isinstance(module, dynamic_linear)
        for module in eager_quantized.modules()
    ) == 2
    assert isinstance(
        eager_quantized.puffer_policy.encoder.encoder, torch.nn.Linear
    )
    assert isinstance(
        quantized.puffer_policy,
        vector_puffer_rollout._TracedMinGRURolloutCore,
    )
    with torch.inference_mode():
        for batch_size in (1, 3, 17):
            tokens = torch.randn(batch_size, 100)
            eager_state = eager_quantized.initial_state(batch_size)
            traced_state = quantized.initial_state(batch_size)
            expected = eager_quantized.puffer_policy.forward_eval(
                tokens, eager_state
            )
            actual = quantized.puffer_policy.forward_eval(
                tokens, traced_state
            )
            assert torch.equal(expected[0], actual[0])
            assert torch.equal(expected[1], actual[1])
            assert torch.equal(expected[2][0], actual[2][0])
            expected_events = (
                eager_quantized.puffer_policy.network.forward_eval(
                    eager_quantized.puffer_policy.encoder(tokens),
                    eager_state,
                )[1][0]
            )
            actual_events = quantized.puffer_policy.forward_events_eval(
                tokens, traced_state
            )[0]
            assert torch.equal(expected_events, actual_events)


def test_int8_residual_base_rollout_drops_expensive_adapter_only():
    torch.manual_seed(50)
    policy = HawkResidualTemporalPolicy(
        base_hidden_size=16,
        base_num_layers=2,
        adapter_hidden_size=24,
        adapter_num_layers=2,
        mode_conditioned_heads=True,
    ).eval()
    quantized = _build_rollout_policy(
        policy, ROLLOUT_PRECISION_INT8_RESIDUAL_BASE
    )
    eager_quantized = _build_rollout_policy(
        policy,
        ROLLOUT_PRECISION_INT8_RESIDUAL_BASE,
        trace_min_gru=False,
    )

    assert not hasattr(quantized.puffer_policy, "adapter")
    assert sum(
        isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
        for module in eager_quantized.modules()
    ) == 2
    tokens = torch.randn(3, 100)
    with torch.inference_mode():
        logits, values, state = quantized.puffer_policy.forward_eval(
            tokens, quantized.initial_state(3)
        )
    assert logits.shape == (3, 55)
    assert values.shape[0] == 3
    assert state[0].shape == (2, 3, 16)


def test_vector_tasks_shard_heavy_cohorts_without_changing_game_assignment():
    player_counts = (
        *(2,) * 14,
        *(3,) * 14,
        *(4,) * 14,
        *(5,) * 6,
        *(6,) * 5,
        *(7,) * 5,
        *(8,) * 5,
    )
    tasks, schedule, returned_counts = _build_tasks(
        games=2048,
        seed=2026081301,
        schedule_offset=456 * 2048,
        schedule=("current_selfplay", "historical_temporal"),
        player_counts=player_counts,
        workers=10,
        prepack_training=True,
        max_turns=500,
    )

    vector_tasks = [task for task in tasks if task["kind"] == "vector"]
    assert len(vector_tasks) == 10
    assert schedule == ("current_selfplay", "historical_temporal")
    assert returned_counts == player_counts
    assert {
        count: sum(task["player_count"] == count for task in vector_tasks)
        for count in range(2, 9)
    } == {2: 1, 3: 1, 4: 2, 5: 1, 6: 1, 7: 2, 8: 2}

    assignments = {
        game_index: task["player_count"]
        for task in vector_tasks
        for game_index in task["game_indices"]
    }
    assert len(assignments) == 2048
    assert set(assignments) == set(range(2048))
    for game_index, count in assignments.items():
        assert count == player_counts[
            (456 * 2048 + game_index) % len(player_counts)
        ]


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


def test_vector_cohort_sharding_preserves_seeded_schedule_and_coverage():
    torch.manual_seed(48)
    policy = NativePufferTemporalPolicy(
        hidden_size=8, mode_conditioned_heads=True
    ).eval()
    arguments = {
        "games": 12,
        "seed": 2048,
        "schedule_offset": 7,
        "schedule": (
            "current_selfplay",
            "historical_temporal",
            "ppo_champion",
            "random_rule",
            "smart_rule",
        ),
        "player_counts": (2, 3),
        "max_turns": 8,
        "max_public_events": 2,
    }
    expected_episodes, expected_metadata = collect_league_batch_vector_parallel(
        policy, workers=2, **arguments
    )
    actual_episodes, actual_metadata = collect_league_batch_vector_parallel(
        policy, workers=3, **arguments
    )

    assert len(actual_episodes) == len(expected_episodes)
    for (expected_trajectory, expected_won), (
        actual_trajectory,
        actual_won,
    ) in zip(expected_episodes, actual_episodes):
        assert actual_won == expected_won
        assert len(actual_trajectory) == len(expected_trajectory)
        for expected_step, actual_step in zip(
            expected_trajectory, actual_trajectory
        ):
            assert actual_step.decision == expected_step.decision
            assert actual_step.action == expected_step.action
            assert actual_step.old_log_prob == pytest.approx(
                expected_step.old_log_prob, abs=2e-6
            )
            assert actual_step.value == pytest.approx(
                expected_step.value, abs=2e-6
            )
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


def test_int8_worker_repacks_current_policy_after_shared_generation_refresh():
    torch.manual_seed(83)
    current = NativePufferTemporalPolicy(
        hidden_size=8, mode_conditioned_heads=True
    ).eval()
    vector_puffer_rollout._initialize_worker(
        current.state_dict(),
        policy_model_config(current),
        "ppo_actor_100run3.pth",
        (),
        ROLLOUT_PRECISION_INT8_RECURRENT,
    )

    _refresh_worker_policy(0)
    first = vector_puffer_rollout._WORKER_POLICY
    tokens = torch.randn(2, 100)
    with torch.inference_mode():
        first_logits, _, _ = first.puffer_policy.forward_eval(
            tokens, first.initial_state(2)
        )
    _refresh_worker_policy(0)
    assert vector_puffer_rollout._WORKER_POLICY is first
    with torch.no_grad():
        source = vector_puffer_rollout._WORKER_POLICY_SOURCE
        source.puffer_policy.network.layers[0].weight.add_(0.25)
    _refresh_worker_policy(1)
    second = vector_puffer_rollout._WORKER_POLICY
    with torch.inference_mode():
        second_logits, _, _ = second.puffer_policy.forward_eval(
            tokens, second.initial_state(2)
        )

    assert second is not first
    assert not torch.equal(second_logits, first_logits)
    assert vector_puffer_rollout._WORKER_HISTORICAL == [second]
    assert isinstance(
        second.puffer_policy,
        vector_puffer_rollout._TracedMinGRURolloutCore,
    )


def test_vector_pool_reports_int8_behavior_precision():
    torch.manual_seed(84)
    policy = NativePufferTemporalPolicy(
        hidden_size=8, mode_conditioned_heads=True
    ).eval()
    with VectorPufferRolloutPool(
        policy,
        workers=1,
        rollout_precision=ROLLOUT_PRECISION_INT8_RECURRENT,
        max_public_events=5,
    ) as pool:
        episodes, metadata = pool.collect(
            policy,
            games=4,
            seed=2084,
            schedule_offset=0,
            schedule=("current_selfplay", "random_rule"),
            player_counts=(2, 3),
            max_turns=5,
        )
    assert episodes
    assert metadata["rollout_precision"] == ROLLOUT_PRECISION_INT8_RECURRENT
    assert metadata["trace_min_gru"] is True
    assert metadata["max_public_events"] == 5


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
