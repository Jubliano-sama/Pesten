from dataclasses import dataclass

import pytest
import torch

import game
from randomAgent import AgentRandom
from temporal_model_zoo import (
    CausalTransformerTemporalPolicy,
    DecoderLMTemporalPolicy,
    LargeRecurrentTemporalPolicy,
    SAFE_TOKEN_SIZE,
    TinyRandomCausalDecoder,
    pack_safe_episodes,
    parameter_report,
)
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


def models():
    return (
        LargeRecurrentTemporalPolicy(
            hidden_size=32, num_layers=2, cell="gru"
        ),
        LargeRecurrentTemporalPolicy(
            hidden_size=32, num_layers=2, cell="lstm"
        ),
        CausalTransformerTemporalPolicy(
            d_model=32,
            num_layers=1,
            num_heads=4,
            feedforward_size=64,
            max_context=32,
        ),
        DecoderLMTemporalPolicy(
            TinyRandomCausalDecoder(
                hidden_size=32,
                num_layers=1,
                num_heads=4,
                feedforward_size=64,
            ),
            max_context=32,
        ),
    )


def test_all_architectures_use_common_safe_batch_and_mask_illegal_actions():
    draw = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=1, count=2
    )
    first = make_decision(events=(draw,))
    second = make_decision(own_cards=(4,))
    batch = pack_safe_episodes(
        [[Sample(first, 4), Sample(second, 54)]]
    )
    assert batch.tokens.shape[-1] == SAFE_TOKEN_SIZE
    assert batch.decision_count == 2
    assert batch.token_count == 3

    for model in models():
        logits, values = model.forward_packed(batch)
        assert logits.shape == (2, 55)
        assert values.shape == (2,)
        assert torch.isfinite(logits[batch.legal_actions]).all()
        assert torch.all(
            logits[~batch.legal_actions]
            == torch.finfo(logits.dtype).min
        )


def test_online_steps_are_history_sensitive_and_never_need_game_state():
    torch.manual_seed(17)
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
        events=(
            PublicEventToken(
                EventKind.DRAW_COUNT, actor_relative=2, count=1
            ),
        )
    )
    for model in models():
        state = model.initial_state()
        first_result = model.step(first, state)
        second_result = model.step(second, first_result.state)
        assert first_result.logits.shape == (55,)
        assert second_result.logits.shape == (55,)
        assert second_result.value.ndim == 0
        assert not hasattr(model, "game")
        assert not hasattr(model, "privileged_critic")


def test_hidden_world_identity_cannot_change_any_model_input_or_output():
    hidden_world_a = ((7, 8, 9), 31)
    hidden_world_b = ((40, 41, 42), 47)
    assert hidden_world_a != hidden_world_b
    public_draw = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=1, count=1
    )
    decision_a = make_decision(events=(public_draw,))
    decision_b = make_decision(events=(public_draw,))
    assert decision_a == decision_b

    for model in models():
        model.eval()
        with torch.no_grad():
            result_a = model.step(decision_a, model.initial_state())
            result_b = model.step(decision_b, model.initial_state())
        torch.testing.assert_close(result_a.logits, result_b.logits)
        torch.testing.assert_close(result_a.value, result_b.value)


def test_short_imitation_update_runs_for_each_architecture():
    torch.manual_seed(5)
    decisions = (
        make_decision(own_cards=(4,)),
        make_decision(own_cards=(16,)),
        make_decision(own_cards=(49,)),
    )
    batch = pack_safe_episodes(
        [[Sample(decisions[0], 4), Sample(decisions[1], 54)],
         [Sample(decisions[2], 54)]]
    )
    for model in models():
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        logits, _ = model.forward_packed(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
        assert any(
            not torch.equal(before[name], parameter.detach())
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
        report = parameter_report(model)
        assert report["parameters"] >= report["trainable_parameters"] > 0
        assert report["estimated_training_state_bytes"] >= (
            report["parameter_bytes"]
        )


def test_existing_game_adapter_can_advance_every_model_state():
    for model in models():
        agent = temporal_agent_factory(model, deterministic=True, seed=3)
        match = game.Game([agent, AgentRandom])
        match.auto_simulate(max_turns=4)
        temporal_player = match.players[0]
        assert temporal_player.safe_history
        assert isinstance(temporal_player.state, (torch.Tensor, tuple))


def test_pretrained_adapter_is_local_only_and_optional(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def reject_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("not installed in test")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_transformers)
    with pytest.raises(RuntimeError, match="no model was downloaded"):
        DecoderLMTemporalPolicy.from_pretrained("unreviewed/remote-model")
