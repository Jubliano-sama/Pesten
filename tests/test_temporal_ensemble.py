import torch

from benchmark_model_zoo import DEFAULT_TEACHER, safe_to_legacy_observation
from residual_temporal_policy import ResidualRecurrentTemporalPolicy
from temporal_ensemble import (
    COMBINATION_MODES,
    TemporalRecurrentEnsemble,
    combine_legal_logits,
)
from temporal_observation import (
    DecisionInput,
    EventKind,
    PublicEventToken,
    build_legal_actions,
    build_private_observation,
)


PUBLIC = {
    "current_suit": "harten",
    "current_rank": 5,
    "public_hand_sizes": (3, 4, 5),
    "observer_index": 0,
    "current_player_index": 0,
    "direction": 1,
    "penalty_amount": 0,
    "public_sorts_played": (2, 1, 0, 1, 0),
}


def make_decision(own_cards=(4, 16, 49), events=()):
    observation = build_private_observation(
        own_cards=own_cards, **PUBLIC
    )
    legal = build_legal_actions(
        own_cards=own_cards,
        current_suit=PUBLIC["current_suit"],
        current_rank=PUBLIC["current_rank"],
        penalty_amount=PUBLIC["penalty_amount"],
    )
    return DecisionInput(observation, legal, tuple(events))


def make_members():
    torch.manual_seed(31)
    gru = ResidualRecurrentTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        hidden_size=16,
        num_layers=1,
        cell="gru",
    )
    torch.manual_seed(32)
    lstm = ResidualRecurrentTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        hidden_size=16,
        num_layers=1,
        cell="lstm",
    )
    return gru, lstm


def champion_logits(member, decision):
    legacy = safe_to_legacy_observation(decision)
    with torch.no_grad():
        logits = member.teacher(legacy.unsqueeze(0)).squeeze(0)
    legal = torch.tensor(decision.legal_actions, dtype=torch.bool)
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)


def test_zero_residual_mean_logits_is_exact_champion_and_states_are_independent():
    decision = make_decision(
        events=(
            PublicEventToken(
                EventKind.DRAW_COUNT, actor_relative=1, count=2
            ),
        )
    )
    members = make_members()
    ensemble = TemporalRecurrentEnsemble(
        members, combination="mean_logits"
    )
    state = ensemble.initial_state()
    assert len(state) == 3  # GRU h plus LSTM h/c.
    assert len({part.data_ptr() for part in state}) == 3
    result = ensemble.step(decision, state)
    expected = champion_logits(members[0], decision)
    assert torch.equal(result.logits, expected)
    assert result.logits.argmax().item() == expected.argmax().item()
    assert len(result.state) == 3
    assert not hasattr(ensemble, "game")
    assert all(not member.training for member in ensemble.members)
    assert all(
        not parameter.requires_grad
        for parameter in ensemble.parameters()
    )


def test_hidden_worlds_cannot_change_ensemble_logits_or_member_states():
    hidden_world_a = ((7, 8, 9), 31)
    hidden_world_b = ((40, 41, 42), 47)
    assert hidden_world_a != hidden_world_b
    draw = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=2, count=1
    )
    decision_a = make_decision(own_cards=(4,), events=(draw,))
    decision_b = make_decision(own_cards=(4,), events=(draw,))
    assert decision_a == decision_b
    for mode in COMBINATION_MODES:
        ensemble = TemporalRecurrentEnsemble(
            make_members(), combination=mode
        )
        result_a = ensemble.step(
            decision_a, ensemble.initial_state()
        )
        result_b = ensemble.step(
            decision_b, ensemble.initial_state()
        )
        assert torch.equal(result_a.logits, result_b.logits)
        assert all(
            torch.equal(left, right)
            for left, right in zip(result_a.state, result_b.state)
        )
        legal = torch.tensor(
            decision_a.legal_actions, dtype=torch.bool
        )
        assert torch.all(
            result_a.logits[~legal]
            == torch.finfo(result_a.logits.dtype).min
        )


def test_probability_and_majority_combinations_are_deterministic_and_legal():
    legal = torch.zeros(55, dtype=torch.bool)
    legal[[4, 16, 54]] = True
    logits = torch.full((3, 55), torch.finfo(torch.float32).min)
    logits[:, legal] = torch.tensor(
        (
            (4.0, 1.0, 0.0),
            (0.0, 5.0, 1.0),
            (3.0, 2.0, 0.0),
        )
    )
    mean_probabilities = combine_legal_logits(
        logits, legal, mode="mean_probabilities"
    )
    majority = combine_legal_logits(
        logits, legal, mode="majority_vote"
    )
    assert mean_probabilities.argmax().item() in (4, 16)
    assert majority.argmax().item() == 4
    assert torch.all(
        majority[~legal] == torch.finfo(majority.dtype).min
    )
    assert torch.equal(
        majority,
        combine_legal_logits(logits, legal, mode="majority_vote"),
    )
