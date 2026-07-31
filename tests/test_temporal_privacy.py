import inspect
import json

import torch

import game
from randomAgent import AgentRandom
from temporal_observation import (
    DecisionInput,
    EventKind,
    PASS_ACTION,
    PublicEventToken,
    assert_policy_schema_is_hidden_state_free,
    build_legal_actions,
    build_private_observation,
    event_to_safe_dict,
    serialize_safe_history,
)
from temporal_puffer_agent import (
    PufferStyleTemporalNet,
    temporal_agent_factory,
)
from temporal_training import save_safe_checkpoint


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


def tensor_bytes(tensor):
    return tensor.detach().cpu().contiguous().numpy().tobytes()


def test_hidden_opponent_worlds_have_identical_tokens_state_logits_and_value():
    # These two environment-side worlds differ in every hidden opponent card
    # and in the private identity of the card drawn.  None of those fields can
    # be passed to make_decision or PublicEventToken.
    hidden_world_a = ((7, 8, 9, 10), (11, 12, 13, 14, 15), 31)
    hidden_world_b = ((31, 32, 33, 34), (40, 41, 42, 43, 44), 47)
    assert hidden_world_a != hidden_world_b

    public_draw = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=1, count=1
    )
    decision_a = make_decision(events=(public_draw,))
    decision_b = make_decision(events=(public_draw,))
    assert decision_a == decision_b
    assert (
        b"".join(
            int(value).to_bytes(2, "little", signed=True)
            for value in public_draw.as_tuple()
        )
        == b"".join(
            int(value).to_bytes(2, "little", signed=True)
            for value in decision_b.public_events[0].as_tuple()
        )
    )

    torch.manual_seed(9)
    policy = PufferStyleTemporalNet(hidden_size=24).eval()
    initial_a = policy.initial_state()
    initial_b = policy.initial_state()
    with torch.no_grad():
        result_a = policy.step(decision_a, initial_a)
        result_b = policy.step(decision_b, initial_b)
    assert tensor_bytes(result_a.state) == tensor_bytes(result_b.state)
    assert tensor_bytes(result_a.logits) == tensor_bytes(result_b.logits)
    assert tensor_bytes(result_a.value) == tensor_bytes(result_b.value)


def test_opponent_card_identity_appears_only_on_public_play():
    draw = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=1, count=1
    )
    assert draw.card_id == -1
    assert "card_id" not in event_to_safe_dict(draw)

    played = PublicEventToken(
        EventKind.PLAY,
        actor_relative=1,
        card_id=42,
        suit_id=3,
        count=1,
    )
    assert event_to_safe_dict(played)["card_id"] == 42


def test_own_draw_changes_only_owners_private_observation():
    player_zero_before = build_private_observation(
        own_cards=(4, 16, 49), **PUBLIC
    )
    player_zero_after = build_private_observation(
        own_cards=(4, 16, 31, 49),
        **{**PUBLIC, "public_hand_sizes": (4, 4, 5)},
    )
    other_public = {
        **PUBLIC,
        "observer_index": 1,
        "current_player_index": 0,
        "public_hand_sizes": (4, 4, 5),
    }
    player_one_before = build_private_observation(
        own_cards=(7, 8, 9, 10), **other_public
    )
    player_one_after = build_private_observation(
        own_cards=(7, 8, 9, 10), **other_public
    )
    assert player_zero_before.values[:50] != player_zero_after.values[:50]
    assert player_one_before.values[:50] == player_one_after.values[:50]
    # The public hand-size delta remains visible to both players.
    assert player_one_after.values[71] == 4.0


def test_safe_history_and_checkpoint_contain_no_opponent_draw_identity(tmp_path):
    draw = PublicEventToken(
        EventKind.DRAW_COUNT, actor_relative=2, count=3
    )
    serialized = serialize_safe_history([make_decision(events=(draw,))])
    decoded = json.loads(serialized)
    draw_json = decoded[0]["public_events"][0]
    assert draw_json == {
        "actor_relative": 2,
        "count": 3,
        "kind": "draw_count",
    }
    assert b"drawn_card" not in serialized
    assert b"draw_pile" not in serialized

    policy = PufferStyleTemporalNet(hidden_size=16)
    checkpoint = tmp_path / "safe.pt"
    save_safe_checkpoint(checkpoint, policy)
    payload = torch.load(checkpoint, weights_only=False)
    assert set(payload) == {
        "format",
        "integration",
        "observation_schema",
        "state_dict",
    }
    assert not any(
        "draw" in key.lower() or "opponent" in key.lower()
        for key in payload
    )


def test_policy_api_and_schema_cannot_accept_game_or_privileged_critic_state():
    assert_policy_schema_is_hidden_state_free()
    assert tuple(inspect.signature(PufferStyleTemporalNet.step).parameters) == (
        "self",
        "decision",
        "state",
    )
    policy = PufferStyleTemporalNet(hidden_size=16)
    assert not hasattr(policy, "game")
    assert not hasattr(policy, "privileged_critic")


def test_illegal_actions_are_masked_for_actor():
    decision = make_decision(own_cards=(4,))
    policy = PufferStyleTemporalNet(hidden_size=16)
    result = policy.step(decision, policy.initial_state())
    legal = decision.legal_actions
    assert legal[4]
    assert legal[PASS_ACTION]
    for index, is_legal in enumerate(legal):
        if not is_legal:
            assert result.logits[index] == torch.finfo(result.logits.dtype).min


def test_recurrent_memory_resets_during_episode_reset_deal():
    torch.manual_seed(3)
    policy = PufferStyleTemporalNet(hidden_size=16)
    agent_class = temporal_agent_factory(policy, deterministic=True)
    match = game.Game([agent_class, AgentRandom])
    match.reset()
    match.simulate_turn()
    agent = match.players[0]
    assert torch.count_nonzero(agent.state) > 0
    match.reset()
    assert torch.count_nonzero(agent.state) == 0
