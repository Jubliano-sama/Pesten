"""Native ensemble arithmetic and independent recurrent history contracts."""

import pytest
import torch
from torch import nn

from native_policy_ensemble import NativeWeightedEnsembleCore


class ToyCore(nn.Module):
    def __init__(self, scale, state_parts=1):
        super().__init__()
        self.scale = scale
        self.state_parts = state_parts

    def initial_state(self, batch_size, device):
        return tuple(torch.zeros(1, batch_size, 1, device=device) for _ in range(self.state_parts))

    def forward_events_eval(self, tokens, states):
        increment = tokens[:, :1].unsqueeze(0) * self.scale
        return tuple(state + increment * (index + 1) for index, state in enumerate(states))

    def forward_eval(self, tokens, states):
        next_states = self.forward_events_eval(tokens, states)
        value = sum(next_states).squeeze(0)
        logits = value + torch.arange(55, device=tokens.device).reshape(1, -1) * self.scale
        return logits, value, next_states


def test_single_member_matches_events_logits_values_and_states():
    member = ToyCore(2, state_parts=2)
    ensemble = NativeWeightedEnsembleCore([member], [7])
    separate = member.initial_state(2, "cpu")
    combined = ensemble.initial_state(2, "cpu")
    tokens = torch.tensor([[1.0], [3.0]])
    separate = member.forward_events_eval(tokens, separate)
    combined = ensemble.forward_events_eval(tokens, combined)
    expected_logits, expected_values, separate = member.forward_eval(tokens, separate)
    actual_logits, actual_values, combined = ensemble.forward_eval(tokens, combined)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, expected_values, rtol=0, atol=0)
    for actual, expected in zip(combined, separate):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_members_keep_independent_different_size_states_and_weighted_unmasked_logits():
    members = [ToyCore(2, state_parts=1), ToyCore(-3, state_parts=2)]
    ensemble = NativeWeightedEnsembleCore(members, [1, 3])
    combined = ensemble.initial_state(2, "cpu")
    separate = [member.initial_state(2, "cpu") for member in members]
    for turn in range(4):
        tokens = torch.tensor([[float(turn + 1)], [float(turn + 2)]])
        combined = ensemble.forward_events_eval(tokens, combined)
        separate = [member.forward_events_eval(tokens, state) for member, state in zip(members, separate)]
        actual_logits, actual_values, combined = ensemble.forward_eval(tokens, combined)
        outputs = [member.forward_eval(tokens, state) for member, state in zip(members, separate)]
        separate = [output[2] for output in outputs]
        torch.testing.assert_close(actual_logits, outputs[0][0] * 0.25 + outputs[1][0] * 0.75)
        torch.testing.assert_close(actual_values, outputs[0][1] * 0.25 + outputs[1][1] * 0.75)
        expected_states = tuple(part for state in separate for part in state)
        for actual, expected in zip(combined, expected_states):
            torch.testing.assert_close(actual, expected)
        assert actual_logits.shape == (2, 55)
        assert torch.isfinite(actual_logits).all()  # No legal-action masking here.
    assert len(combined) == 3
    assert combined[0].data_ptr() != combined[1].data_ptr()


@pytest.mark.parametrize("weights", [[], [0], [-1], [float("nan")], [float("inf")], [1, 2]])
def test_invalid_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        NativeWeightedEnsembleCore([ToyCore(1)], weights)


def test_large_finite_weights_normalize_without_overflow():
    ensemble = NativeWeightedEnsembleCore([ToyCore(1), ToyCore(2)], [1e308, 1e308])
    assert ensemble.weights == (0.5, 0.5)


def test_zero_weight_member_still_preserves_its_history():
    ensemble = NativeWeightedEnsembleCore([ToyCore(1), ToyCore(2)], [1, 0])
    states = ensemble.initial_state(1, "cpu")
    logits, values, states = ensemble.forward_eval(torch.ones(1, 1), states)
    assert values.item() == 1
    assert states[1].item() == 2
    assert logits[0, 0].item() == 1


def test_event_fallback_supports_native_cores_without_event_method():
    class DecisionOnlyCore(nn.Module):
        def initial_state(self, batch_size, device):
            return (torch.zeros(1, batch_size, 1, device=device),)

        def forward_eval(self, tokens, states):
            state = states[0] + 1
            return tokens, tokens, (state,)

    ensemble = NativeWeightedEnsembleCore([DecisionOnlyCore()], [1])
    states = ensemble.initial_state(1, "cpu")
    states = ensemble.forward_events_eval(torch.zeros(1, 55), states)
    assert states[0].item() == 1


def test_full_native_suit_orbit_members_preserve_separate_histories():
    pytest.importorskip("pufferlib")
    from strongest_policy_search import NativeSuitOrbitCore
    from temporal_observation import EventKind, PublicEventToken
    from temporal_puffer_v4 import NativePufferTemporalPolicy

    torch.manual_seed(517)
    policies = [NativePufferTemporalPolicy(hidden_size=12, num_layers=2) for _ in range(2)]
    members = [NativeSuitOrbitCore(policy.puffer_policy, orbit_size=24) for policy in policies]
    ensemble = NativeWeightedEnsembleCore(members, [0.5, 0.5])
    states = ensemble.initial_state(2, "cpu")
    independent = [member.initial_state(2, "cpu") for member in members]
    event = PublicEventToken(EventKind.PLAY, 1, 25, 2, 1)
    events = torch.tensor(policies[0]._event_features_from_events([event, event]))
    tokens = torch.zeros(2, 100)
    tokens[:, 99] = 1
    with torch.no_grad():
        for _ in range(3):
            states = ensemble.forward_events_eval(events, states)
            independent = [member.forward_events_eval(events, state) for member, state in zip(members, independent)]
            logits, values, states = ensemble.forward_eval(tokens, states)
            outputs = [member.forward_eval(tokens, state) for member, state in zip(members, independent)]
            independent = [output[2] for output in outputs]
            torch.testing.assert_close(logits, (outputs[0][0] + outputs[1][0]) / 2)
            torch.testing.assert_close(values, (outputs[0][1] + outputs[1][1]) / 2)
            for actual, expected in zip(states, (part for state in independent for part in state)):
                torch.testing.assert_close(actual, expected)


def test_native_parent_step_matches_batched_evaluator_with_variable_public_histories():
    pytest.importorskip("pufferlib")
    from copy import deepcopy
    from strongest_policy_search import NativeSuitOrbitCore
    from temporal_batched_speed import SafeEventCursor, TemporalSeatRuntime, _batch_temporal_actions
    from temporal_observation import DecisionInput, EventKind, PlayerPrivateObservation, PublicEventToken
    from temporal_puffer_v4 import NativePufferTemporalPolicy

    torch.manual_seed(839)
    members = [NativePufferTemporalPolicy(hidden_size=12, num_layers=2) for _ in range(2)]
    policy = deepcopy(members[0]).eval()
    policy.puffer_policy = NativeWeightedEnsembleCore(
        [NativeSuitOrbitCore(member.puffer_policy, orbit_size=24) for member in members],
        [0.5, 0.5],
    )
    legal = tuple(index in (4, 50) for index in range(55))
    observation = PlayerPrivateObservation((0.0,) * 94)
    decisions = [
        DecisionInput(observation, legal, (
            PublicEventToken(EventKind.PLAY, 1, 25, 2, 1),
            PublicEventToken(EventKind.DRAW_COUNT, 1, count=2),
        )),
        DecisionInput(observation, legal, ()),
    ]
    runtimes = [TemporalSeatRuntime(
        game_index=index, absolute_game_index=index, seat=0, policy=policy,
        generator=torch.Generator().manual_seed(index), state=policy.initial_state(),
        cursor=SafeEventCursor(),
    ) for index in range(2)]
    scalar_states = [policy.initial_state() for _ in range(2)]
    with torch.inference_mode():
        for _ in range(3):
            scalar_outputs = [policy.step(decision, state)
                              for decision, state in zip(decisions, scalar_states)]
            scalar_states = [output.state for output in scalar_outputs]
            actions = _batch_temporal_actions(
                policy, list(zip(runtimes, decisions)), None, deterministic=True
            )
            assert actions == [output.logits.argmax().item() for output in scalar_outputs]
            assert all(action in (4, 50) for action in actions)
            for runtime, expected in zip(runtimes, scalar_outputs):
                assert len(runtime.state) == 2
                torch.testing.assert_close(torch.tensor(runtime.trajectory[-1].value), expected.value)
                for actual_part, expected_part in zip(runtime.state, expected.state):
                    torch.testing.assert_close(actual_part, expected_part)
