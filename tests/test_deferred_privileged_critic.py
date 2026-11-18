"""Deferred baselines must preserve play and the data consumed by PPO."""

from dataclasses import fields
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pufferlib = pytest.importorskip("pufferlib")
if not str(getattr(pufferlib, "__version__", "")).startswith("4"):
    pytest.skip("requires pinned PufferLib 4 runtime", allow_module_level=True)

from packed_puffer_rollout import pack_puffer_episode
from privileged_critic import (
    AveragedPrivilegedCritic,
    CausalTemporalValueModel,
    flatten_recurrent_state,
)
from temporal_batched_speed import collect_league_batch_batched
from temporal_puffer_v4 import NativePufferTemporalPolicy
from temporal_training import _prepare_ppo_batch


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _models(backbone="mlp"):
    torch.manual_seed(940)
    policy = NativePufferTemporalPolicy(hidden_size=8, num_layers=1).eval()
    critic = CausalTemporalValueModel(
        actor_memory_size=flatten_recurrent_state(policy.initial_state()).size,
        backbone=backbone,
        hidden_size=32,
        num_layers=1,
        group_width=16,
        transformer_heads=4,
        dropout=0.2,
    )
    # Exercise the full snapshot-dependent value instead of its zero-initialized
    # residual head, including player-count calibration and public actor value.
    with torch.no_grad():
        critic.value_head[-1].weight.normal_(std=0.1)
        critic.value_head[-1].bias.fill_(0.35)
        critic.value_calibration_scale.copy_(torch.linspace(0.8, 1.2, 14))
        critic.value_calibration_bias.copy_(torch.linspace(-0.2, 0.2, 14))
    return policy, critic


def _collect(policy, critic=None, **kwargs):
    options = dict(
        games=8,
        seed=943,
        player_count=3,
        schedule=("current_selfplay", "historical_temporal", "canonical_random"),
        max_turns=80,
        privileged_critic=critic,
    )
    options.update(kwargs)
    return collect_league_batch_batched(policy, None, [policy], **options)


def _steps(episodes):
    return [step for trajectory, _ in episodes for step in trajectory]


def _assert_same_play(left, right):
    assert [outcome for _, outcome in left] == [outcome for _, outcome in right]
    assert [len(trajectory) for trajectory, _ in left] == [
        len(trajectory) for trajectory, _ in right
    ]
    for actual, expected in zip(_steps(left), _steps(right)):
        assert actual.action == expected.action
        assert actual.old_log_prob == expected.old_log_prob
        assert actual.decision == expected.decision


def _assert_same_snapshots(left, right):
    for actual, expected in zip(_steps(left), _steps(right)):
        for field in fields(actual.privileged_snapshot):
            value = getattr(actual.privileged_snapshot, field.name)
            reference = getattr(expected.privileged_snapshot, field.name)
            if isinstance(value, np.ndarray):
                assert value.dtype == reference.dtype
                assert value.shape == reference.shape
                assert value.tobytes() == reference.tobytes()
                assert not value.flags.writeable
            else:
                assert value == reference


@pytest.mark.parametrize(
    "ensemble,event_limit,batch_size", [(False, None, 32), (True, 2, 512)]
)
def test_deferred_values_preserve_play_snapshots_and_packed_ppo(
    monkeypatch, ensemble, event_limit, batch_size
):
    policy, member = _models()
    members = [member]
    critic = member
    if ensemble:
        _, second = _models()
        with torch.no_grad():
            second.value_head[-1].bias.add_(0.4)
        members.append(second)
        critic = AveragedPrivilegedCritic(members)
        with torch.no_grad():
            critic.value_calibration_scale.fill_(1.3)
            critic.value_calibration_bias.fill_(-0.15)
    batch_sizes = []
    original_step = member.step

    def count_step(batch, state):
        batch_sizes.append(len(batch["public_value"]))
        return original_step(batch, state)

    monkeypatch.setattr(member, "step", count_step)
    plain_trace, online_trace, deferred_trace = [], [], []
    plain, plain_metadata = _collect(
        policy, max_public_events=event_limit, trace_sink=plain_trace
    )
    online, online_metadata = _collect(
        policy, critic, privileged_critic_batch_size=None,
        max_public_events=event_limit, trace_sink=online_trace,
    )
    online_calls = len(batch_sizes)
    batch_sizes.clear()
    critic.train()  # Deferred setup must still disable dropout before play.
    rng_before = torch.get_rng_state().clone()
    deferred, deferred_metadata = _collect(
        policy, critic, privileged_critic_batch_size=batch_size,
        max_public_events=event_limit, trace_sink=deferred_trace,
    )
    assert torch.equal(rng_before, torch.get_rng_state())
    assert not critic.training
    assert all(not model.training for model in members)
    assert plain_metadata == online_metadata == deferred_metadata
    assert plain_trace == online_trace == deferred_trace
    _assert_same_play(plain, online)
    _assert_same_play(online, deferred)
    _assert_same_snapshots(online, deferred)
    steps = _steps(deferred)
    assert len(batch_sizes) == math.ceil(len(steps) / batch_size)
    assert len(batch_sizes) < online_calls
    assert max(batch_sizes) <= batch_size
    np.testing.assert_allclose(
        [step.value for step in steps],
        [step.value for step in _steps(online)],
        atol=2e-6, rtol=2e-6,
    )
    assert any(
        abs(step.value - step.privileged_snapshot.public_value) > 0.01
        for step in steps
    )

    packed_online = [pack_puffer_episode(policy, episode) for episode in online]
    packed_deferred = [pack_puffer_episode(policy, episode) for episode in deferred]
    for (actual, outcome), (expected, expected_outcome) in zip(
        packed_deferred, packed_online
    ):
        assert outcome == expected_outcome
        for field in fields(actual):
            if field.name == "behavior_values":
                np.testing.assert_allclose(
                    actual.behavior_values, expected.behavior_values,
                    atol=2e-6, rtol=2e-6,
                )
            else:
                np.testing.assert_array_equal(
                    getattr(actual, field.name), getattr(expected, field.name)
                )
    prepare = dict(
        gamma=0.995, gae_lambda=0.95, padded_batch_size=None,
        power_of_two_sequence_padding=False,
    )
    expected_ppo = _prepare_ppo_batch(policy, packed_online, **prepare)
    actual_ppo = _prepare_ppo_batch(policy, packed_deferred, **prepare)
    for name in ("returns", "advantages"):
        torch.testing.assert_close(
            getattr(actual_ppo, name), getattr(expected_ppo, name),
            atol=2e-6, rtol=2e-6,
        )


@pytest.mark.parametrize("backbone", ["gru", "lstm", "mixed_ensemble"])
def test_recurrent_critic_keeps_online_stream_state(monkeypatch, backbone):
    policy, critic = _models("gru" if backbone == "mixed_ensemble" else backbone)
    calls = []
    original_step = critic.step

    def count_step(batch, state):
        calls.append(state)
        return original_step(batch, state)

    monkeypatch.setattr(critic, "step", count_step)
    if backbone == "mixed_ensemble":
        _, mlp = _models()
        critic = AveragedPrivilegedCritic([mlp, critic])
    online, _ = _collect(
        policy, critic, games=2, max_turns=30, privileged_critic_batch_size=None
    )
    online_calls = len(calls)
    calls.clear()
    automatic, _ = _collect(policy, critic, games=2, max_turns=30)
    assert len(calls) == online_calls > 1
    assert all(state is not None for state in calls)
    _assert_same_play(online, automatic)
    np.testing.assert_array_equal(
        [step.value for step in _steps(online)],
        [step.value for step in _steps(automatic)],
    )


def test_custom_runtime_keeps_real_game_seat_ids_and_online_order():
    policy, _ = _models()

    class StatefulRuntime:
        model = SimpleNamespace(backbone_name="mlp")

        def __init__(self):
            self.visits = {}

        def evaluate(self, snapshots, stream_ids):
            values = []
            for stream_id in stream_ids:
                assert isinstance(stream_id, tuple) and len(stream_id) == 2
                self.visits[stream_id] = self.visits.get(stream_id, 0) + 1
                values.append(self.visits[stream_id])
            return {"value": torch.tensor(values)}

    runtime = StatefulRuntime()
    episodes, _ = _collect(policy, runtime, games=2, max_turns=30)
    assert runtime.visits
    for trajectory, _ in episodes:
        assert [step.value for step in trajectory] == list(
            range(1, len(trajectory) + 1)
        )


def test_empty_rollout_and_invalid_batch_size():
    policy, critic = _models()
    episodes, _ = _collect(policy, critic, games=0)
    assert episodes == []
    for invalid in (0, -1):
        with pytest.raises(ValueError, match="batch_size must be positive"):
            _collect(policy, critic, games=0, privileged_critic_batch_size=invalid)
