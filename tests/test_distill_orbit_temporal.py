"""Full public-history orbit targets and isolated student updates."""

from copy import deepcopy

import pytest
import torch

pytest.importorskip("pufferlib")

from distill_orbit_temporal import (
    OPPONENT_SCHEDULE, PLAYER_COUNT_SCHEDULE, build_orbit_teacher, parse_args,
)
from puffer_distillation import distill_puffer_policy, evaluate_puffer_distillation
from strongest_policy_search import NativeSuitOrbitCore
from temporal_observation import (
    DecisionInput, EventKind, PlayerPrivateObservation, PublicEventToken,
)
from temporal_puffer_agent import TrajectoryStep
from temporal_puffer_v4 import NativePufferTemporalPolicy
from temporal_suit_orbit import relabel_decision, relabel_public_event


def _decision(events=()):
    values = tuple(float(i % 7) / 7 for i in range(94))
    return DecisionInput(PlayerPrivateObservation(values), (True,) * 55, events)


@pytest.mark.parametrize("orbit_size,alpha", [(1, 1.0), (4, 0.4), (24, 1.0), (24, 0.0)])
def test_sequence_matches_independent_worlds_and_streamed_public_events(orbit_size, alpha):
    torch.manual_seed(351)
    policy = NativePufferTemporalPolicy(hidden_size=12, num_layers=2).eval()
    orbit = NativeSuitOrbitCore(policy.puffer_policy, orbit_size=orbit_size, alpha=alpha)
    events = (
        PublicEventToken(EventKind.EPISODE_START),
        PublicEventToken(EventKind.PLAY, 1, 14, 1, 1),
        PublicEventToken(EventKind.DRAW_COUNT, 1, count=2),
        PublicEventToken(EventKind.CHOOSE_SUIT, 1, suit_id=2),
        PublicEventToken(EventKind.PLAY, 1, 49, 4, 1),
    )
    decision = _decision()
    one_row = []
    for event in events:
        one_row.extend(policy._event_features_from_events([event]))
        one_row.append(policy._decision_features(decision))
    tokens = torch.tensor([one_row, list(reversed(one_row))])
    with torch.no_grad():
        logits, values = orbit(tokens)
        worlds, world_values = [], []
        # Use the independent safe-schema transforms, not orbit.relabel_tokens.
        for permutation in orbit.suit_permutations:
            rows = []
            for event in events:
                rows.extend(policy._event_features_from_events([
                    relabel_public_event(event, permutation)
                ]))
                rows.append(policy._decision_features(relabel_decision(decision, permutation)))
            expected_tokens = torch.tensor([rows, list(reversed(rows))])
            world_logits, world_value = policy.puffer_policy(expected_tokens)
            index = orbit.suit_permutations.index(permutation)
            worlds.append(world_logits[:, orbit.action_indices[index]])
            world_values.append(world_value)
        expected = torch.stack(worlds, 1)
        torch.testing.assert_close(logits, expected[:, 0] * (1 - alpha) + expected.mean(1) * alpha)
        torch.testing.assert_close(values, torch.stack(world_values, 1).mean(1).reshape(-1, 1))

        state = orbit.initial_state(2, "cpu")
        streamed_logits, streamed_values = [], []
        for step in range(tokens.shape[1]):
            step_logits, step_values, state = orbit.forward_eval(tokens[:, step], state)
            streamed_logits.append(step_logits)
            streamed_values.append(step_values)
        torch.testing.assert_close(logits, torch.stack(streamed_logits, 1).reshape(-1, 55))
        torch.testing.assert_close(values, torch.stack(streamed_values, 1).reshape(-1, 1))


def test_distillation_updates_only_student_using_full_event_history():
    torch.manual_seed(719)
    student = NativePufferTemporalPolicy(hidden_size=12, num_layers=2)
    teacher = build_orbit_teacher(student, orbit_size=4, alpha=1.0)
    before_teacher = deepcopy(teacher.state_dict())
    before_student = deepcopy(student.state_dict())
    decision = _decision((PublicEventToken(EventKind.PLAY, 1, 25, 2, 1),))
    episodes = [((TrajectoryStep(decision, 0, -0.1, 0.0),) * 2, True)] * 4
    curve = distill_puffer_policy(
        student, teacher, episodes, torch.optim.Adam(student.parameters(), lr=1e-3),
        epochs=1, minibatch_trajectories=2, seed=21,
    )
    assert curve[0]["samples"] == 8
    assert torch.isfinite(torch.tensor(curve[0]["loss"]))
    assert any(not torch.equal(value, before_student[name])
               for name, value in student.state_dict().items())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in student.parameters())
    assert all(parameter.grad is None and not parameter.requires_grad
               for parameter in teacher.parameters())
    for name, value in teacher.state_dict().items():
        torch.testing.assert_close(value, before_teacher[name], rtol=0, atol=0)
    result = evaluate_puffer_distillation(student, teacher, episodes, minibatch_trajectories=2)
    assert result["samples"] == 8
    assert torch.isfinite(torch.tensor(result["forward_kl"]))


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_teacher_maps_follow_student_device_for_public_play_tokens(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    student = NativePufferTemporalPolicy(hidden_size=12).to(device)
    teacher = build_orbit_teacher(student, orbit_size=4, alpha=1.0)
    expected_device = next(student.parameters()).device
    assert all(buffer.device == expected_device for buffer in teacher.buffers())
    event = PublicEventToken(EventKind.PLAY, 1, 14, 1, 1)
    tokens = torch.tensor([student._event_features_from_events([event])], device=device)
    with torch.no_grad():
        logits, values = teacher.forward_native_sequence(tokens)
    assert logits.device == values.device == expected_device
    assert torch.isfinite(logits).all()


def test_default_schedule_covers_all_opponents_at_every_player_count():
    pairs = {(OPPONENT_SCHEDULE[i % 5], PLAYER_COUNT_SCHEDULE[i % 7]) for i in range(35)}
    assert len(pairs) == 35
    assert "canonical_random" in OPPONENT_SCHEDULE
    assert "ppo_champion" in OPPONENT_SCHEDULE


def test_cli_rejects_zero_games_before_starting_workers():
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "unused", "--games", "0"])


def test_sequence_rejects_nonsequence_input():
    policy = NativePufferTemporalPolicy(hidden_size=12)
    with pytest.raises(ValueError, match="batch, time, 100"):
        NativeSuitOrbitCore(policy.puffer_policy)(torch.zeros(2, 100))
