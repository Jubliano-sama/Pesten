import torch

from ai_observation import (
    ACTION_COUNT,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PASS_ACTION,
)
from benchmark_corrected_ppo import (
    UNSEALED_SEED_LIMIT,
    derive_unsealed_seed,
    run_benchmark,
)
import card
from corrected_ppo_agent import (
    CorrectedPPOStats,
    DrawRetryCorrectedPPOAgent,
    draw_retry_action_mask,
    make_corrected_ppo_agent_factory,
)
import deck
import game
from hybrid_agent import FrozenGreedyPPOAgent
from hybrid_agent import make_ppo_agent_factory


class PassiveAgent:
    def __init__(self):
        self.mydeck = deck.Deck([])

    def playCard(self, _current_sort, _current_true_number):
        return None

    def changeSort(self):
        return "harten"

    def addCard(self, drawn_card):
        self.mydeck.cards.append(drawn_card)

    def remove(self, played_card):
        self.mydeck.cards.remove(played_card)


class RetryTrapActor(torch.nn.Module):
    """Pass first, then prefer a compatible old card over the drawn card."""

    def forward(self, observations):
        logits = torch.full(
            (len(observations), ACTION_COUNT),
            -100.0,
            device=observations.device,
        )
        hand_sizes = observations[:, :50].sum(dim=-1)
        initial = hand_sizes <= 2
        logits[initial, PASS_ACTION] = 10.0
        logits[~initial, 3] = 10.0
        logits[~initial, 16] = 9.0
        logits[~initial, PASS_ACTION] = 8.0
        return logits


class ParityActor(torch.nn.Module):
    def forward(self, observations):
        logits = torch.full(
            (len(observations), ACTION_COUNT),
            -100.0,
            device=observations.device,
        )
        logits[:, 3] = 10.0
        logits[:, 52] = torch.where(
            observations[:, MODE_INDEX] != 0,
            torch.tensor(20.0, device=observations.device),
            torch.tensor(-100.0, device=observations.device),
        )
        return logits


def configured_game(agent_class, actor):
    def factory(instance):
        return agent_class(instance, actor, torch.device("cpu"))

    instance = game.Game([factory, PassiveAgent])
    agent = instance.players[0]
    agent.mydeck.cards[:] = [card.Card(3), card.Card(30)]
    instance.players[1].mydeck.cards[:] = [card.Card(25)]
    instance.current_player_index = 0
    instance.turn_count = 11
    instance.direction = 1
    instance.cards_played = 0
    instance.current_sort = "harten"
    instance.current_true_number = 5
    instance.penalty_amount = 0
    instance.sorts_played[:] = 0
    instance.game_over = False
    instance.winner = None
    instance.game_deck = deck.Deck([card.Card(4)])
    instance.grab_deck = deck.Deck([card.Card(16)])
    return instance, agent


def test_draw_retry_mask_is_exactly_drawn_card_plus_pass():
    observation = torch.zeros(OBSERVATION_SIZE)
    observation[3] = 1
    observation[16] = 1
    observation[50 + 3] = 1
    observation[50 + 16] = 1

    mask = draw_retry_action_mask(observation, 16)

    assert torch.nonzero(mask).flatten().tolist() == [16, PASS_ACTION]


def test_corrected_wrapper_plays_drawn_card_that_legacy_silently_ignores():
    actor = RetryTrapActor()
    stats = CorrectedPPOStats()

    def corrected_factory(instance):
        return DrawRetryCorrectedPPOAgent(
            instance,
            actor,
            torch.device("cpu"),
            stats=stats,
        )

    corrected, corrected_agent = configured_game(
        lambda instance, _actor, _device: corrected_factory(instance),
        actor,
    )
    corrected.simulate_turn()

    legacy, legacy_agent = configured_game(FrozenGreedyPPOAgent, actor)
    legacy.simulate_turn()

    assert corrected.current_sort == "ruiten"
    assert corrected.current_true_number == 5
    assert [held.number for held in corrected_agent.mydeck.cards] == [3, 30]
    assert legacy.current_sort == "harten"
    assert legacy.current_true_number == 5
    assert [held.number for held in legacy_agent.mydeck.cards] == [3, 30, 16]
    assert stats.normal_draws_tracked == 1
    assert stats.retry_decisions == 1
    assert stats.retry_overrides == 1
    assert stats.retry_legacy_wrong_card_actions == 1
    assert stats.retry_drawn_card_actions == 1
    assert stats.retry_pass_actions == 0
    assert corrected_agent.tracked_draw_retry is None


def test_actions_match_legacy_exactly_outside_retry_including_suit_choice():
    actor = ParityActor()
    stats = CorrectedPPOStats()
    corrected, corrected_agent = configured_game(
        lambda instance, actor, device: DrawRetryCorrectedPPOAgent(
            instance, actor, device, stats=stats
        ),
        actor,
    )
    legacy, legacy_agent = configured_game(FrozenGreedyPPOAgent, actor)

    assert corrected_agent.playCard("harten", 5).number == 3
    assert legacy_agent.playCard("harten", 5).number == 3
    assert corrected_agent.changeSort() == legacy_agent.changeSort()
    assert corrected_agent.changeSort() == card.sorts[2]
    assert stats.outside_retry_decisions == 3
    assert stats.outside_retry_action_mismatches == 0


def test_hidden_cards_and_draw_order_cannot_change_observation_or_action():
    actor = ParityActor()
    left, left_agent = configured_game(
        DrawRetryCorrectedPPOAgent, actor
    )
    right, right_agent = configured_game(
        DrawRetryCorrectedPPOAgent, actor
    )
    left.players[1].mydeck.cards[:] = [card.Card(1), card.Card(2)]
    right.players[1].mydeck.cards[:] = [card.Card(40), card.Card(41)]
    left.grab_deck.cards[:] = [card.Card(8), card.Card(9)]
    right.grab_deck.cards[:] = [card.Card(47), card.Card(48)]

    torch.testing.assert_close(left_agent.obs(), right_agent.obs())
    assert (
        left_agent.playCard("harten", 5).number
        == right_agent.playCard("harten", 5).number
        == 3
    )
    assert left_agent.tracked_draw_retry is None
    assert right_agent.tracked_draw_retry is None


def test_development_deal_seeds_are_deterministic_and_unsealed():
    first = [
        derive_unsealed_seed(82_607_031, "pool:smart", index)
        for index in range(100)
    ]
    second = [
        derive_unsealed_seed(82_607_031, "pool:smart", index)
        for index in range(100)
    ]
    assert first == second
    assert len(set(first)) == len(first)
    assert all(0 <= seed < UNSEALED_SEED_LIMIT for seed in first)


def test_benchmark_reports_seed_and_action_parity_audits():
    actor = ParityActor()
    stats = CorrectedPPOStats()
    corrected_factory = make_corrected_ppo_agent_factory(
        actor, torch.device("cpu"), stats=stats
    )
    legacy_factory = make_ppo_agent_factory(
        actor, torch.device("cpu"), shield=False
    )

    result = run_benchmark(
        corrected_factory,
        legacy_factory,
        pairs=1,
        base_seed=82_607_031,
        pool_names=("random",),
        max_turns=50,
    )

    assert result["action_audit"]["exact_action_parity_outside_retry"]
    assert result["action_audit"]["outside_retry_action_mismatches"] == 0
    assert result["deal_seed_audit"]["all_below_92m"]
    assert result["deal_seed_audit"]["count"] == 2
