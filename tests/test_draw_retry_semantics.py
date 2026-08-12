import numpy as np
import pytest
import torch
from torch import nn

import card
import deck
import game
import randomAgent
from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_START,
    PASS_ACTION,
)
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    PHASE_DRAW_RETRY,
    POLICY_FIRST_LEGAL,
    POLICY_RL,
    collect_batched_rollout,
)
from ppo2 import build_action_masks


DRAWN_CARD = 16
OTHER_COMPATIBLE_CARD = 4


class ScriptedRetryAgent:
    def __init__(self, retry_action):
        self.retry_action = retry_action
        self.calls = 0
        self.mydeck = deck.Deck([])

    def addCard(self, drawn_card):
        self.mydeck.cards.append(drawn_card)

    def remove(self, played_card):
        for held in self.mydeck.cards:
            if held.number == played_card.number:
                self.mydeck.cards.remove(held)
                return

    def playCard(self, current_sort, current_true_number):
        del current_sort, current_true_number
        self.calls += 1
        if self.calls == 1 or self.retry_action is None:
            return None
        return card.Card(self.retry_action)

    def changeSort(self):
        return "harten"


class PassiveAgent:
    def __init__(self):
        self.mydeck = deck.Deck([])

    def addCard(self, drawn_card):
        self.mydeck.cards.append(drawn_card)

    def remove(self, played_card):
        self.mydeck.cards.remove(played_card)

    def playCard(self, current_sort, current_true_number):
        del current_sort, current_true_number
        return None

    def changeSort(self):
        return "harten"


class RetryLogitActor(nn.Module):
    def __init__(self, *, prefer_pass):
        super().__init__()
        self.prefer_pass = prefer_pass

    def forward(self, observations):
        logits = torch.full(
            (len(observations), ACTION_COUNT),
            -100.0,
            device=observations.device,
            dtype=observations.dtype,
        )
        # This is the highest raw logit but must be masked during draw-retry.
        logits[:, OTHER_COMPATIBLE_CARD] = 100.0
        logits[:, DRAWN_CARD] = 10.0
        logits[:, PASS_ACTION] = 20.0 if self.prefer_pass else 0.0
        return logits


def install_hand(engine, player_index, numbers):
    numbers = np.asarray(numbers, dtype=np.int8)
    engine.hands[0, player_index].fill(-1)
    engine.hands[0, player_index, : len(numbers)] = numbers
    engine.hand_sizes[0, player_index] = len(numbers)
    engine.hand_counts[0, player_index].fill(0)
    for number in numbers:
        engine.hand_counts[0, player_index, number] += 1


def configured_retry_engine(*, max_turns=500):
    engine = BatchedPesten(
        np.asarray([[POLICY_RL, POLICY_FIRST_LEGAL]], dtype=np.int8),
        np.asarray([["current", None]], dtype=object),
        np.asarray([73_001], dtype=np.uint32),
        max_turns=max_turns,
    )
    install_hand(engine, 0, (OTHER_COMPATIBLE_CARD, DRAWN_CARD))
    install_hand(engine, 1, (28,))
    engine.current_player[0] = 0
    engine.turn_player[0] = 0
    engine.current_sort[0] = 0
    engine.current_rank[0] = 5
    engine.penalty_amount[0] = 0
    engine.phase[0] = PHASE_DRAW_RETRY
    engine.drawn_card[0] = DRAWN_CARD
    engine.cards_played[0] = 1
    engine.draw_counts[0] = 0
    engine.draw_piles[0].fill(-1)
    engine.discards[0].fill(-1)
    engine.discards[0, 0] = OTHER_COMPATIBLE_CARD
    engine.discard_counts[0] = 1
    return engine


def configured_canonical(retry_action):
    instance = game.Game([])
    actor = ScriptedRetryAgent(retry_action)
    opponent = PassiveAgent()
    actor.mydeck.cards = [card.Card(OTHER_COMPATIBLE_CARD)]
    opponent.mydeck.cards = [card.Card(28)]
    instance.players = [actor, opponent]
    instance.num_players = 2
    instance.current_player_index = 0
    instance.current_sort = "harten"
    instance.current_true_number = 5
    instance.grab_deck = deck.Deck([card.Card(DRAWN_CARD)])
    instance.game_deck = deck.Deck([card.Card(OTHER_COMPATIBLE_CARD)])
    instance.simulate_turn()
    return instance


def state_fingerprint(engine):
    return (
        int(engine.turn_counts[0]),
        int(engine.current_player[0]),
        int(engine.current_sort[0]),
        int(engine.current_rank[0]),
        int(engine.cards_played[0]),
        engine.hands[
            0, 0, : engine.hand_sizes[0, 0]
        ].astype(int).tolist(),
        engine.draw_piles[
            0, : engine.draw_counts[0]
        ].astype(int).tolist(),
        engine.discards[
            0, : engine.discard_counts[0]
        ].astype(int).tolist(),
    )


def canonical_fingerprint(instance):
    return (
        instance.turn_count,
        instance.current_player_index,
        card.sorts.index(instance.current_sort),
        instance.current_true_number,
        instance.cards_played,
        [held.number for held in instance.players[0].mydeck.cards],
        [held.number for held in instance.grab_deck.cards],
        [held.number for held in instance.game_deck.cards],
    )


def test_draw_retry_mask_exposes_only_drawn_card_and_pass():
    engine = configured_retry_engine()
    observation = torch.from_numpy(engine.observation(0))
    mask = build_action_masks(observation)

    assert observation[
        COMPATIBLE_START + DRAWN_CARD
    ] == 1
    assert observation[
        COMPATIBLE_START + OTHER_COMPATIBLE_CARD
    ] == 0
    assert torch.nonzero(mask, as_tuple=False).flatten().tolist() == [
        DRAWN_CARD,
        PASS_ACTION,
    ]


def test_canonical_random_plays_a_compatible_card_after_forced_draw():
    """AgentRandom cannot voluntarily pass, so its retry is unambiguous.

    Reaching the draw proves every card already held was incompatible.  The
    table state is unchanged by drawing, which makes a compatible drawn card
    the only compatible card in the hand on the retry call.
    """

    instance = game.Game([randomAgent.AgentRandom, PassiveAgent])
    actor = instance.players[0]
    actor.mydeck.cards = [card.Card(12)]  # Ruiten ace: not hearts or rank five.
    instance.players[1].mydeck.cards = [card.Card(28)]
    instance.current_player_index = 0
    instance.current_sort = "harten"
    instance.current_true_number = 5
    instance.grab_deck = deck.Deck([card.Card(DRAWN_CARD)])
    instance.game_deck = deck.Deck([card.Card(OTHER_COMPATIBLE_CARD)])

    instance.simulate_turn()

    assert [held.number for held in actor.mydeck.cards] == [12]
    assert [held.number for held in instance.game_deck.cards] == [
        OTHER_COMPATIBLE_CARD,
        DRAWN_CARD,
    ]
    assert instance.cards_played == 2  # Forced pass/draw, then drawn-card play.


@pytest.mark.parametrize(
    ("retry_action", "batched_action"),
    ((DRAWN_CARD, DRAWN_CARD), (None, PASS_ACTION)),
)
def test_draw_retry_play_and_pass_match_canonical_transition(
    retry_action, batched_action
):
    canonical = configured_canonical(retry_action)
    engine = configured_retry_engine()
    engine.apply_action(0, batched_action)
    assert state_fingerprint(engine) == canonical_fingerprint(canonical)


def test_rl_draw_retry_rejects_other_mask_incompatible_card_without_mutation():
    engine = configured_retry_engine()
    before = state_fingerprint(engine)
    with pytest.raises(ValueError, match="drawn card or pass"):
        engine.apply_action(0, OTHER_COMPATIBLE_CARD)
    assert state_fingerprint(engine) == before


@pytest.mark.parametrize(
    ("prefer_pass", "expected_action", "expected_reward"),
    ((False, DRAWN_CARD, 0.0), (True, PASS_ACTION, -0.1)),
)
def test_batched_rollout_masks_wrong_retry_logit_and_rewards_executed_action(
    prefer_pass, expected_action, expected_reward
):
    engine = configured_retry_engine(max_turns=1)
    result = collect_batched_rollout(
        engine,
        {
            "current": ActorPolicy(
                actor=RetryLogitActor(prefer_pass=prefer_pass),
                device=torch.device("cpu"),
                temperature=0.8,
                collect_data=True,
                greedy=True,
            )
        },
        output_device=torch.device("cpu"),
    )

    assert result.actions.tolist() == [expected_action]
    assert result.rewards.tolist() == pytest.approx([expected_reward])
    assert int(result.observations[0, COMPATIBLE_START + DRAWN_CARD]) == 1
    assert int(
        result.observations[
            0, COMPATIBLE_START + OTHER_COMPATIBLE_CARD
        ]
    ) == 0
    if expected_action == DRAWN_CARD:
        assert engine.hand_counts[0, 0, DRAWN_CARD] == 0
        assert engine.discards[0, engine.discard_counts[0] - 1] == DRAWN_CARD
    else:
        assert engine.hand_counts[0, 0, DRAWN_CARD] == 1
        assert engine.discards[0, engine.discard_counts[0] - 1] != DRAWN_CARD
