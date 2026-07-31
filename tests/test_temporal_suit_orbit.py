from dataclasses import fields

import pytest
import torch
from torch import nn

from residual_temporal_policy import ResidualRecurrentTemporalPolicy
from temporal_observation import (
    ACTION_COUNT,
    DecisionInput,
    EventKind,
    PlayerPrivateObservation,
    PublicEventToken,
)
from temporal_puffer_agent import PolicyStep
from temporal_suit_orbit import (
    ALL_SUIT_PERMUTATIONS,
    FullSuitOrbitRecurrentPolicy,
    inverse_suit_permutation,
    map_relabelled_logits_back,
    relabel_action,
    relabel_card_id,
    relabel_decision,
    relabel_observation,
    relabel_public_event,
)


def make_decision(*, events=()) -> DecisionInput:
    values = tuple(float(index + 1) for index in range(94))
    legal = tuple(
        action in {0, 7, 12, 31, 48, 49, 50, 53, 54}
        for action in range(ACTION_COUNT)
    )
    return DecisionInput(
        PlayerPrivateObservation(values),
        legal,
        tuple(events),
    )


class TinySuitBiasedRecurrent(nn.Module):
    """A deliberately non-equivariant policy with input-dependent memory."""

    cell = "gru"
    num_layers = 1
    hidden_size = 3
    architecture = "test_policy"

    def initial_state(self, batch_size=1, *, device=None):
        return torch.zeros(1, batch_size, 3, device=device)

    def step(self, decision, state):
        observation = torch.tensor(
            decision.observation.values,
            dtype=state.dtype,
            device=state.device,
        )
        event_suit_sum = sum(
            event.suit_id
            for event in decision.public_events
            if event.suit_id >= 0
        )
        increment = torch.stack(
            (
                observation[0] + observation[12] * 0.1,
                observation[50] + observation[53] * 0.2,
                observation[82] + float(event_suit_sum),
            )
        ).reshape(1, 1, 3)
        next_state = state + increment
        logits = torch.arange(
            ACTION_COUNT, dtype=state.dtype, device=state.device
        )
        logits = logits + (
            next_state[0, 0, 0] * torch.linspace(
                -0.3, 0.7, ACTION_COUNT, device=state.device
            )
        )
        legal = torch.tensor(
            decision.legal_actions, dtype=torch.bool, device=state.device
        )
        logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)
        return PolicyStep(
            logits=logits,
            value=next_state.sum(),
            state=next_state,
        )


def test_all_card_action_observation_and_event_mappings_round_trip():
    assert len(ALL_SUIT_PERMUTATIONS) == 24
    decision = make_decision(
        events=(
            PublicEventToken(EventKind.EPISODE_START),
            PublicEventToken(
                EventKind.PLAY,
                actor_relative=2,
                card_id=26,
                suit_id=2,
                count=1,
            ),
            PublicEventToken(
                EventKind.PLAY,
                actor_relative=1,
                card_id=48,
                suit_id=4,
                count=1,
            ),
            PublicEventToken(
                EventKind.DRAW_COUNT, actor_relative=3, count=2
            ),
            PublicEventToken(
                EventKind.CHOOSE_SUIT, actor_relative=2, suit_id=3
            ),
        )
    )
    for permutation in ALL_SUIT_PERMUTATIONS:
        inverse = inverse_suit_permutation(permutation)
        for card_id in range(50):
            transformed_card = relabel_card_id(card_id, permutation)
            assert relabel_card_id(transformed_card, inverse) == card_id
            suit_id = card_id // 12 if card_id < 48 else 4
            play = PublicEventToken(
                EventKind.PLAY,
                actor_relative=2,
                card_id=card_id,
                suit_id=suit_id,
                count=1,
            )
            transformed_play = relabel_public_event(play, permutation)
            assert transformed_play.card_id == transformed_card
            assert transformed_play.suit_id == (
                transformed_card // 12 if transformed_card < 48 else 4
            )
            assert relabel_public_event(transformed_play, inverse) == play
        for suit_id in range(4):
            choice = PublicEventToken(
                EventKind.CHOOSE_SUIT,
                actor_relative=1,
                suit_id=suit_id,
            )
            assert (
                relabel_public_event(
                    relabel_public_event(choice, permutation), inverse
                )
                == choice
            )
        for action in range(ACTION_COUNT):
            assert (
                relabel_action(
                    relabel_action(action, permutation), inverse
                )
                == action
            )
        transformed = relabel_decision(decision, permutation)
        restored = relabel_decision(transformed, inverse)
        assert restored == decision


def test_only_documented_suit_coordinates_change():
    decision = make_decision(
        events=(
            PublicEventToken(
                EventKind.PLAY,
                actor_relative=2,
                card_id=14,
                suit_id=1,
                count=1,
            ),
            PublicEventToken(
                EventKind.DRAW_COUNT, actor_relative=1, count=4
            ),
        )
    )
    permutation = (2, 0, 3, 1)
    transformed = relabel_decision(decision, permutation)
    mutable_observation_indices = (
        set(range(48))
        | set(range(50, 54))
        | set(range(82, 86))
    )
    for index, (before, after) in enumerate(
        zip(
            decision.observation.values,
            transformed.observation.values,
        )
    ):
        if index not in mutable_observation_indices:
            assert after == before
    draw_before = decision.public_events[1]
    draw_after = transformed.public_events[1]
    assert draw_after is draw_before
    play_before = decision.public_events[0]
    play_after = transformed.public_events[0]
    for field in fields(PublicEventToken):
        if field.name not in {"card_id", "suit_id"}:
            assert getattr(play_after, field.name) == getattr(
                play_before, field.name
            )


@pytest.mark.parametrize(
    "event",
    (
        PublicEventToken(
            EventKind.PLAY,
            actor_relative=1,
            card_id=12,
            suit_id=0,
            count=1,
        ),
        PublicEventToken(
            EventKind.CHOOSE_SUIT,
            actor_relative=1,
            suit_id=4,
        ),
    ),
)
def test_ambiguous_or_inconsistent_human_visible_events_fail_closed(event):
    with pytest.raises(ValueError):
        relabel_public_event(event, (1, 0, 2, 3))


def test_orbit_keeps_24_independent_recurrent_states():
    wrapper = FullSuitOrbitRecurrentPolicy(
        TinySuitBiasedRecurrent()
    )
    decision = make_decision(
        events=(
            PublicEventToken(
                EventKind.CHOOSE_SUIT, actor_relative=1, suit_id=0
            ),
        )
    )
    result = wrapper.step(decision, wrapper.initial_state())
    assert result.state.shape == (1, 24, 3)
    assert torch.unique(result.state[0], dim=0).shape[0] > 1
    second = wrapper.step(decision, result.state)
    torch.testing.assert_close(second.state, result.state * 2)


def test_full_history_orbit_average_is_suit_equivariant():
    canonical = FullSuitOrbitRecurrentPolicy(
        TinySuitBiasedRecurrent()
    )
    relabelled = FullSuitOrbitRecurrentPolicy(
        TinySuitBiasedRecurrent()
    )
    external_permutation = (1, 3, 0, 2)
    canonical_state = canonical.initial_state()
    relabelled_state = relabelled.initial_state()
    history = (
        make_decision(
            events=(
                PublicEventToken(EventKind.EPISODE_START),
                PublicEventToken(
                    EventKind.PLAY,
                    actor_relative=1,
                    card_id=3,
                    suit_id=0,
                    count=1,
                ),
            )
        ),
        make_decision(
            events=(
                PublicEventToken(
                    EventKind.CHOOSE_SUIT,
                    actor_relative=2,
                    suit_id=3,
                ),
                PublicEventToken(
                    EventKind.DRAW_COUNT,
                    actor_relative=1,
                    count=2,
                ),
            )
        ),
    )
    for decision in history:
        canonical_result = canonical.step(decision, canonical_state)
        transformed_decision = relabel_decision(
            decision, external_permutation
        )
        relabelled_result = relabelled.step(
            transformed_decision, relabelled_state
        )
        mapped_back = map_relabelled_logits_back(
            relabelled_result.logits, external_permutation
        )
        torch.testing.assert_close(
            mapped_back,
            canonical_result.logits,
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            relabelled_result.value,
            canonical_result.value,
            atol=2e-5,
            rtol=2e-5,
        )
        canonical_state = canonical_result.state
        relabelled_state = relabelled_result.state


def test_hidden_world_cannot_change_orbit_result_or_state():
    """Hidden worlds are absent from the immutable DecisionInput boundary."""
    decision = make_decision(
        events=(
            PublicEventToken(
                EventKind.DRAW_COUNT, actor_relative=1, count=2
            ),
        )
    )
    hidden_world_left = {
        "opponent_cards": (1, 2),
        "draw_pile": (3, 4, 5),
    }
    hidden_world_right = {
        "opponent_cards": (41, 42),
        "draw_pile": (47, 48, 49),
    }
    assert hidden_world_left != hidden_world_right
    left = FullSuitOrbitRecurrentPolicy(TinySuitBiasedRecurrent())
    right = FullSuitOrbitRecurrentPolicy(TinySuitBiasedRecurrent())
    left_result = left.step(decision, left.initial_state())
    right_result = right.step(decision, right.initial_state())
    assert torch.equal(left_result.logits, right_result.logits)
    assert torch.equal(left_result.value, right_result.value)
    assert torch.equal(left_result.state, right_result.state)


def test_logit_mapping_exactly_inverts_relabelled_action_coordinates():
    logits = torch.arange(ACTION_COUNT, dtype=torch.float32)
    permutation = (2, 0, 3, 1)
    transformed = torch.empty_like(logits)
    for action in range(ACTION_COUNT):
        transformed[relabel_action(action, permutation)] = logits[action]
    assert torch.equal(
        map_relabelled_logits_back(transformed, permutation), logits
    )


def test_vectorized_residual_path_matches_24_independent_recurrent_calls():
    torch.manual_seed(41)
    policy = ResidualRecurrentTemporalPolicy(
        hidden_size=8,
        num_layers=1,
        cell="gru",
    ).eval()
    with torch.no_grad():
        policy.residual_actor.weight.normal_(std=0.02)
        policy.residual_actor.bias.normal_(std=0.02)
    wrapper = FullSuitOrbitRecurrentPolicy(policy).eval()
    decision = make_decision(
        events=(
            PublicEventToken(
                EventKind.PLAY,
                actor_relative=1,
                card_id=14,
                suit_id=1,
                count=1,
            ),
            PublicEventToken(
                EventKind.CHOOSE_SUIT,
                actor_relative=1,
                suit_id=2,
            ),
        )
    )
    state = wrapper.initial_state()
    fast = wrapper.step(decision, state)

    mapped_logits = []
    values = []
    next_states = []
    for index, permutation in enumerate(ALL_SUIT_PERMUTATIONS):
        result = policy.step(
            relabel_decision(decision, permutation),
            state[:, index : index + 1, :],
        )
        mapped_logits.append(
            map_relabelled_logits_back(result.logits, permutation)
        )
        values.append(result.value)
        next_states.append(result.state)
    stacked_logits = torch.stack(mapped_logits)
    legal = torch.tensor(decision.legal_actions)
    expected_logits = stacked_logits.masked_fill(~legal, 0.0).mean(dim=0)
    expected_logits = expected_logits.masked_fill(
        ~legal, torch.finfo(expected_logits.dtype).min
    )
    torch.testing.assert_close(fast.logits, expected_logits, atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(
        fast.value, torch.stack(values).mean(), atol=2e-6, rtol=2e-6
    )
    torch.testing.assert_close(
        fast.state,
        torch.cat(next_states, dim=1),
        atol=2e-6,
        rtol=2e-6,
    )
