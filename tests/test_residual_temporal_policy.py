from dataclasses import dataclass
import random
import statistics
import time

import numpy as np
import pytest
import torch

import game
from benchmark_agents import load_ppo_agent
from benchmark_model_zoo import (
    DEFAULT_TEACHER,
    safe_to_legacy_observation,
)
from benchmark_residual_checkpoint import (
    deterministic_evaluation_agent_factory,
    deterministic_evaluation_runtime,
)
from randomAgent import AgentRandom
from residual_temporal_policy import ResidualCausalTemporalPolicy
from residual_temporal_policy import ResidualRecurrentTemporalPolicy
from temporal_model_zoo import SafeTemporalPolicy, pack_safe_episodes
from temporal_observation import (
    DecisionInput,
    EventKind,
    PublicEventToken,
    build_legal_actions,
    build_private_observation,
)
from temporal_puffer_agent import temporal_agent_factory


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


@dataclass(frozen=True)
class Sample:
    decision: DecisionInput
    action: int


def make_decision(own_cards=(4, 16, 49), events=()):
    observation = build_private_observation(
        own_cards=own_cards,
        **PUBLIC,
    )
    legal = build_legal_actions(
        own_cards=own_cards,
        current_suit=PUBLIC["current_suit"],
        current_rank=PUBLIC["current_rank"],
        penalty_amount=PUBLIC["penalty_amount"],
    )
    return DecisionInput(observation, legal, tuple(events))


def make_policy():
    torch.manual_seed(19)
    return ResidualCausalTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        d_model=32,
        num_layers=1,
        num_heads=4,
        feedforward_size=64,
        max_context=32,
    )


def make_recurrent_policy(cell="gru"):
    torch.manual_seed(29)
    return ResidualRecurrentTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        hidden_size=32,
        num_layers=1,
        cell=cell,
        residual_scale=0.7,
        logit_temperature=1.4,
    )


def champion_masked_logits(policy, decision):
    legacy = safe_to_legacy_observation(decision)
    with torch.no_grad():
        logits = policy.teacher(legacy.unsqueeze(0)).squeeze(0)
    legal = torch.tensor(decision.legal_actions, dtype=torch.bool)
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)


def test_zero_residual_has_exact_champion_logits_and_action():
    decision = make_decision(
        events=(
            PublicEventToken(
                EventKind.DRAW_COUNT, actor_relative=1, count=2
            ),
        )
    )
    policy = make_policy()
    assert isinstance(policy, SafeTemporalPolicy)
    policy.train()
    assert not policy.teacher.training
    assert all(not parameter.requires_grad for parameter in policy.teacher.parameters())
    result = policy.step(decision, policy.initial_state())
    expected = champion_masked_logits(policy, decision)
    assert torch.equal(result.logits, expected)
    assert result.logits.argmax().item() == expected.argmax().item()
    packed = pack_safe_episodes([[Sample(decision, int(expected.argmax()))]])
    packed_logits, _ = policy.forward_packed(packed)
    assert torch.equal(packed_logits[0], expected)
    assert not policy.critic_privileged


def test_packed_and_online_paths_match_after_nonzero_residual():
    first = make_decision(
        events=(
            PublicEventToken(
                EventKind.PLAY,
                actor_relative=1,
                card_id=23,
                suit_id=1,
                count=1,
            ),
        )
    )
    second = make_decision(
        own_cards=(4,),
        events=(
            PublicEventToken(
                EventKind.DRAW_COUNT, actor_relative=2, count=1
            ),
        ),
    )
    policy = make_policy().eval()
    with torch.no_grad():
        policy.residual_actor.weight.normal_(mean=0.0, std=0.02)
        policy.residual_actor.bias.normal_(mean=0.0, std=0.02)
    batch = pack_safe_episodes(
        [[Sample(first, 4), Sample(second, 54)]]
    )
    packed_logits, packed_values = policy.forward_packed(batch)

    state = policy.initial_state()
    online = []
    values = []
    for decision in (first, second):
        result = policy.step(decision, state)
        state = result.state
        online.append(result.logits)
        values.append(result.value)
    torch.testing.assert_close(torch.stack(online), packed_logits)
    torch.testing.assert_close(torch.stack(values), packed_values)


def test_legality_and_hidden_world_invariance():
    hidden_world_a = ((7, 8, 9), 31)
    hidden_world_b = ((40, 41, 42), 47)
    assert hidden_world_a != hidden_world_b
    public_draw = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=1, count=1
    )
    decision_a = make_decision(own_cards=(4,), events=(public_draw,))
    decision_b = make_decision(own_cards=(4,), events=(public_draw,))
    assert decision_a == decision_b
    policy = make_policy().eval()
    with torch.no_grad():
        result_a = policy.step(decision_a, policy.initial_state())
        result_b = policy.step(decision_b, policy.initial_state())
    assert torch.equal(result_a.logits, result_b.logits)
    assert torch.equal(result_a.value, result_b.value)
    legal = torch.tensor(decision_a.legal_actions, dtype=torch.bool)
    assert torch.all(
        result_a.logits[~legal]
        == torch.finfo(result_a.logits.dtype).min
    )


def test_residual_trains_while_champion_remains_immutable():
    decisions = (
        make_decision(own_cards=(4,)),
        make_decision(own_cards=(16,)),
        make_decision(own_cards=(49,)),
    )
    policy = make_policy()
    batch = pack_safe_episodes(
        [[Sample(decisions[0], 4), Sample(decisions[1], 54)],
         [Sample(decisions[2], 54)]]
    )
    teacher_before = {
        name: tensor.detach().clone()
        for name, tensor in policy.teacher.state_dict().items()
    }
    actor_before = policy.residual_actor.weight.detach().clone()
    projection_before = policy.token_projection[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    policy.train()
    for _ in range(2):
        logits, values = policy.forward_packed(batch)
        loss = (
            torch.nn.functional.cross_entropy(logits, batch.targets)
            + 0.01 * values.square().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert not torch.equal(actor_before, policy.residual_actor.weight)
    assert not torch.equal(projection_before, policy.token_projection[0].weight)
    assert not policy.teacher.training
    assert all(
        torch.equal(teacher_before[name], tensor)
        for name, tensor in policy.teacher.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in policy.teacher.parameters())


def test_checkpoint_round_trip_preserves_outputs_and_frozen_base(tmp_path):
    decision = make_decision(
        events=(
            PublicEventToken(
                EventKind.CHOOSE_SUIT,
                actor_relative=1,
                suit_id=2,
            ),
        )
    )
    policy = make_policy().eval()
    with torch.no_grad():
        policy.residual_actor.weight.normal_(std=0.01)
    before = policy.step(decision, policy.initial_state())
    checkpoint = tmp_path / "residual.pt"
    policy.save_checkpoint(checkpoint)
    payload = torch.load(checkpoint, weights_only=True)
    assert not any(
        key.startswith("teacher.")
        for key in payload["residual_state_dict"]
    )

    restored = ResidualCausalTemporalPolicy.load_checkpoint(
        checkpoint,
        champion_path=DEFAULT_TEACHER,
    ).eval()
    after = restored.step(decision, restored.initial_state())
    assert torch.equal(before.logits, after.logits)
    assert torch.equal(before.value, after.value)
    assert restored.champion_sha256 == policy.champion_sha256
    restored.train()
    assert not restored.teacher.training


def test_existing_temporal_game_adapter_advances_residual_history():
    policy = make_policy().eval()
    agent = temporal_agent_factory(policy, deterministic=True, seed=23)
    match = game.Game([agent, AgentRandom])
    match.auto_simulate(max_turns=4)
    temporal_player = match.players[0]
    assert temporal_player.safe_history
    assert temporal_player.state.ndim == 2
    assert temporal_player.state.shape[1] == 100


def test_recurrent_zero_residual_is_exact_champion_for_gru_and_lstm():
    decision = make_decision(
        events=(
            PublicEventToken(
                EventKind.DRAW_COUNT, actor_relative=1, count=2
            ),
        )
    )
    for cell in ("gru", "lstm"):
        policy = make_recurrent_policy(cell)
        assert isinstance(policy, SafeTemporalPolicy)
        policy.train()
        result = policy.step(decision, policy.initial_state())
        expected = champion_masked_logits(policy, decision)
        assert torch.equal(result.logits, expected)
        assert result.logits.argmax().item() == expected.argmax().item()
        assert not policy.teacher.training
        assert not policy.critic_privileged


def _completed_game_fingerprint(match):
    return (
        match.turn_count,
        match.winner,
        match.current_player_index,
        match.direction,
        match.current_sort,
        match.current_true_number,
        match.penalty_amount,
        tuple(
            tuple(held.number for held in player.mydeck.cards)
            for player in match.players
        ),
        tuple(held.number for held in match.grab_deck.cards),
        tuple(held.number for held in match.game_deck.cards),
        tuple(
            tuple(sorted(event.items()))
            for event in match.public_history
        ),
    )


@pytest.mark.parametrize("family", ["causal", "recurrent"])
def test_full_game_zero_residual_matches_production_ppo(family):
    policy = (
        make_policy().eval()
        if family == "causal"
        else make_recurrent_policy().eval()
    )
    temporal_factory = deterministic_evaluation_agent_factory(
        policy,
        seed=0,
    )
    production_factory = load_ppo_agent(DEFAULT_TEACHER, "cpu")

    with deterministic_evaluation_runtime("cpu", cpu_threads=1):
        for deal_seed in (70_039, 91_000_039):
            for temporal_seat in (0, 1):
                random.seed(deal_seed)
                np.random.seed(deal_seed)
                temporal_factories = (
                    [temporal_factory, production_factory]
                    if temporal_seat == 0
                    else [production_factory, temporal_factory]
                )
                temporal_match = game.Game(temporal_factories)
                temporal_match.auto_simulate()

                random.seed(deal_seed)
                np.random.seed(deal_seed)
                production_match = game.Game(
                    [production_factory, production_factory]
                )
                production_match.auto_simulate()

                assert _completed_game_fingerprint(
                    temporal_match
                ) == _completed_game_fingerprint(production_match)


def test_recurrent_packed_online_consistency_and_hidden_world_invariance():
    public_play = PublicEventToken(
        EventKind.PLAY,
        actor_relative=1,
        card_id=23,
        suit_id=1,
        count=1,
    )
    public_draw = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=2, count=1
    )
    first = make_decision(events=(public_play,))
    second_a = make_decision(own_cards=(4,), events=(public_draw,))
    hidden_world_a = ((7, 8, 9), 31)
    hidden_world_b = ((40, 41, 42), 47)
    assert hidden_world_a != hidden_world_b
    second_b = make_decision(own_cards=(4,), events=(public_draw,))
    assert second_a == second_b

    policy = make_recurrent_policy().eval()
    with torch.no_grad():
        policy.residual_actor.weight.normal_(std=0.02)
        policy.residual_actor.bias.normal_(std=0.02)
    batch = pack_safe_episodes(
        [[Sample(first, 4), Sample(second_a, 54)]]
    )
    packed_logits, packed_values = policy.forward_packed(batch)
    state = policy.initial_state()
    online_logits = []
    online_values = []
    for decision in (first, second_a):
        result = policy.step(decision, state)
        state = result.state
        online_logits.append(result.logits)
        online_values.append(result.value)
    torch.testing.assert_close(torch.stack(online_logits), packed_logits)
    torch.testing.assert_close(torch.stack(online_values), packed_values)

    state_a = policy.step(first, policy.initial_state()).state
    state_b = policy.step(first, policy.initial_state()).state
    result_a = policy.step(second_a, state_a)
    result_b = policy.step(second_b, state_b)
    assert torch.equal(result_a.logits, result_b.logits)
    legal = torch.tensor(second_a.legal_actions, dtype=torch.bool)
    assert torch.all(
        result_a.logits[~legal]
        == torch.finfo(result_a.logits.dtype).min
    )


def test_recurrent_residual_trains_and_checkpoint_round_trips(tmp_path):
    decisions = (
        make_decision(own_cards=(4,)),
        make_decision(own_cards=(16,)),
        make_decision(own_cards=(49,)),
    )
    batch = pack_safe_episodes(
        [[Sample(decisions[0], 4), Sample(decisions[1], 54)],
         [Sample(decisions[2], 54)]]
    )
    policy = make_recurrent_policy()
    teacher_before = {
        name: tensor.detach().clone()
        for name, tensor in policy.teacher.state_dict().items()
    }
    actor_before = policy.residual_actor.weight.detach().clone()
    projection_before = policy.token_projection[0].weight.detach().clone()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    for _ in range(2):
        logits, values = policy.forward_packed(batch)
        loss = (
            torch.nn.functional.cross_entropy(logits, batch.targets)
            + 0.01 * values.square().mean()
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    assert not torch.equal(actor_before, policy.residual_actor.weight)
    assert not torch.equal(projection_before, policy.token_projection[0].weight)
    assert all(
        torch.equal(teacher_before[name], tensor)
        for name, tensor in policy.teacher.state_dict().items()
    )
    assert all(parameter.grad is None for parameter in policy.teacher.parameters())

    checkpoint = tmp_path / "residual-gru.pt"
    policy.eval()
    before = policy.step(decisions[0], policy.initial_state())
    policy.save_checkpoint(checkpoint)
    restored = ResidualRecurrentTemporalPolicy.load_checkpoint(
        checkpoint, champion_path=DEFAULT_TEACHER
    ).eval()
    after = restored.step(decisions[0], restored.initial_state())
    assert torch.equal(before.logits, after.logits)
    assert torch.equal(before.value, after.value)
    assert not restored.teacher.training


def _online_decisions_per_second(policy, decisions, repeats=3):
    policy.eval()
    durations = []
    with torch.inference_mode():
        for _ in range(repeats):
            state = policy.initial_state()
            started = time.perf_counter()
            for decision in decisions:
                state = policy.step(decision, state).state
            durations.append(time.perf_counter() - started)
    return len(decisions) / statistics.median(durations)


def test_recurrent_online_path_is_faster_than_causal_history_recompute():
    event = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=1, count=1
    )
    decisions = tuple(
        make_decision(own_cards=(4, 16), events=(event,))
        for _ in range(96)
    )
    torch.manual_seed(101)
    recurrent = ResidualRecurrentTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        hidden_size=64,
        num_layers=2,
        cell="gru",
    )
    torch.manual_seed(101)
    causal = ResidualCausalTemporalPolicy(
        champion_path=DEFAULT_TEACHER,
        d_model=64,
        num_layers=2,
        num_heads=4,
        feedforward_size=192,
        max_context=128,
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        recurrent_rate = _online_decisions_per_second(
            recurrent, decisions
        )
        causal_rate = _online_decisions_per_second(causal, decisions)
    finally:
        torch.set_num_threads(previous_threads)
    assert recurrent_rate > causal_rate
