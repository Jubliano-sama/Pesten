"""Causal relabeling and recurrent-state independence for native suit orbit."""

import pytest
import torch

pytest.importorskip("pufferlib")

from strongest_policy_search import NativeSuitOrbitCore
from temporal_observation import (
    DecisionInput, EventKind, PlayerPrivateObservation, PublicEventToken,
)
from temporal_puffer_v4 import NativePufferTemporalPolicy
from temporal_suit_orbit import relabel_decision, relabel_public_event


def _decision(events=()):
    values = tuple(float(i % 7) for i in range(94))
    return DecisionInput(PlayerPrivateObservation(values), (True,) * 55, events)


def test_native_tokens_match_existing_safe_schema_relabeling():
    policy = NativePufferTemporalPolicy(hidden_size=12)
    orbit = NativeSuitOrbitCore(policy.puffer_policy)
    decision = _decision()
    events = (
        PublicEventToken(EventKind.EPISODE_START),
        PublicEventToken(EventKind.PLAY, 1, 14, 1, 1),
        PublicEventToken(EventKind.PLAY, 1, 48, 4, 1),
        PublicEventToken(EventKind.PLAY, 1, 49, 4, 1),
        PublicEventToken(EventKind.DRAW_COUNT, 1, count=4),
        PublicEventToken(EventKind.CHOOSE_SUIT, 1, suit_id=2),
    )
    tokens = torch.tensor([
        policy._decision_features(decision),
        *policy._event_features_from_events(events),
    ])
    actual = orbit.relabel_tokens(tokens).reshape(len(tokens), 24, 100)
    for index, permutation in enumerate(orbit.suit_permutations):
        expected = torch.tensor([
            policy._decision_features(relabel_decision(decision, permutation)),
            *policy._event_features_from_events([
                relabel_public_event(event, permutation) for event in events
            ]),
        ])
        torch.testing.assert_close(actual[:, index], expected, rtol=0, atol=0)


def test_batched_orbit_matches_independent_recurrent_worlds():
    torch.manual_seed(731)
    policy = NativePufferTemporalPolicy(hidden_size=12, num_layers=2).eval()
    orbit = NativeSuitOrbitCore(policy.puffer_policy, alpha=0.4)
    states = orbit.initial_state(2, "cpu")
    independent = [policy.initial_state(2) for _ in range(24)]
    tokens = torch.tensor([policy._decision_features(_decision())] * 2)
    with torch.inference_mode():
        for step in range(3):
            tokens[:, step] += 1.0
            expected_logits = []
            expanded = orbit.relabel_tokens(tokens).reshape(2, 24, 100)
            for index in range(24):
                logits, _, independent[index] = policy.puffer_policy.forward_eval(
                    expanded[:, index], independent[index]
                )
                expected_logits.append(logits[:, orbit.action_indices[index]])
            actual, _, states = orbit.forward_eval(tokens, states)
            stacked = torch.stack(expected_logits, 1)
            torch.testing.assert_close(actual, stacked[:, 0] * 0.6 + stacked.mean(1) * 0.4)
            unpacked = orbit._unpack(states, 2)
            for part_index, part in enumerate(unpacked):
                by_world = part.reshape(part.shape[0], 2, 24, -1)
                for world in range(24):
                    torch.testing.assert_close(by_world[:, :, world], independent[world][part_index])


def test_zero_blend_is_identity_policy_even_after_events():
    torch.manual_seed(991)
    policy = NativePufferTemporalPolicy(hidden_size=12, num_layers=2).eval()
    orbit = NativeSuitOrbitCore(policy.puffer_policy, alpha=0)
    base_state = policy.initial_state(2)
    orbit_state = orbit.initial_state(2, "cpu")
    event = PublicEventToken(EventKind.PLAY, 1, 25, 2, 1)
    event_tokens = torch.tensor(policy._event_features_from_events([event, event]))
    with torch.inference_mode():
        orbit_state = orbit.forward_events_eval(event_tokens, orbit_state)
        _, _, base_state = policy.puffer_policy.forward_eval(event_tokens, base_state)
        tokens = torch.tensor([policy._decision_features(_decision())] * 2)
        expected, _, _ = policy.puffer_policy.forward_eval(tokens, base_state)
        actual, _, _ = orbit.forward_eval(tokens, orbit_state)
    torch.testing.assert_close(actual, expected)


def test_pool_reports_candidate_gain_over_reference_orbit():
    from strongest_policy_search import summarize, summarize_pool

    variants = {"base": {}, "reference_orbit24_a1": {}, "orbit24_a1": {}}
    rows = [{"opponent": "self_anchor", "players": 2,
             "base_scores": [0.5, 0.5],
             "reference_orbit24_a1_scores": [0.5, 1.0],
             "orbit24_a1_scores": [1.0, 1.0]}]
    summary = summarize(rows, variants)["self_anchor/2p"]["orbit24_a1"]
    assert summary["delta_vs_base"] == 0.5
    assert summary["delta_vs_reference_orbit"] == 0.25
    pool = summarize_pool(rows, variants)["orbit24_a1"]
    assert pool["weighted_delta_vs_base"] == 0.5
    assert pool["weighted_delta_vs_reference_orbit"] == 0.25
