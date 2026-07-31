import torch

from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_START,
    MODE_INDEX,
    OBSERVATION_SIZE,
    SORTS_PLAYED_START,
    SUIT_ACTION_START,
)
from benchmark_suit_symmetry import run_symmetry_screen
import card
import deck
import game
from hybrid_agent import make_ppo_agent_factory
from suit_symmetry_ppo_agent import (
    ALL_SUIT_PERMUTATIONS,
    SuitSymmetryPPOAgent,
    SuitSymmetryStats,
    make_suit_symmetry_ppo_factory,
    map_permuted_logits_back,
    permute_suit_observation,
    suit_averaged_logits,
)


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


class SuitBiasedActor(torch.nn.Module):
    def forward(self, observations):
        logits = torch.full(
            (len(observations), ACTION_COUNT),
            -100.0,
            device=observations.device,
        )
        for suit in range(4):
            logits[:, suit * 12 + 3] = float(suit)
            logits[:, SUIT_ACTION_START + suit] = float(suit)
        return logits


def public_game(agent_factory):
    instance = game.Game([agent_factory, PassiveAgent])
    agent = instance.players[0]
    agent.mydeck.cards[:] = [
        card.Card(3),
        card.Card(15),
        card.Card(27),
        card.Card(39),
    ]
    instance.players[1].mydeck.cards[:] = [card.Card(1), card.Card(2)]
    instance.current_sort = "harten"
    instance.current_true_number = 4
    instance.current_player_index = 0
    instance.turn_count = 9
    instance.direction = 1
    instance.penalty_amount = 0
    instance.sorts_played[:] = [1, 2, 3, 4, 0]
    instance.game_over = False
    instance.game_deck = deck.Deck([card.Card(3)])
    instance.grab_deck = deck.Deck([card.Card(20), card.Card(21)])
    return instance, agent


def test_observation_and_logit_permutations_have_exact_inverse_contract():
    observation = torch.arange(OBSERVATION_SIZE, dtype=torch.float32)
    permutation = (2, 0, 3, 1)
    inverse = tuple(permutation.index(suit) for suit in range(4))

    transformed = permute_suit_observation(observation, permutation)
    restored = permute_suit_observation(transformed, inverse)
    torch.testing.assert_close(restored, observation)

    logits = torch.arange(ACTION_COUNT, dtype=torch.float32)
    relabeled_logits = logits.clone()
    # Build transformed-coordinate logits by applying the inverse mapping.
    for original_suit, transformed_suit in enumerate(permutation):
        relabeled_logits[
            transformed_suit * 12 : (transformed_suit + 1) * 12
        ] = logits[original_suit * 12 : (original_suit + 1) * 12]
        relabeled_logits[SUIT_ACTION_START + transformed_suit] = logits[
            SUIT_ACTION_START + original_suit
        ]
    torch.testing.assert_close(
        map_permuted_logits_back(relabeled_logits, permutation),
        logits,
    )


def test_full_group_average_is_exactly_suit_equivariant():
    torch.manual_seed(7)
    actor = torch.nn.Linear(OBSERVATION_SIZE, ACTION_COUNT)
    observation = torch.randn(OBSERVATION_SIZE)
    permutation = (1, 3, 0, 2)

    averaged, _ = suit_averaged_logits(actor, observation)
    transformed_average, _ = suit_averaged_logits(
        actor,
        permute_suit_observation(observation, permutation),
    )

    torch.testing.assert_close(
        map_permuted_logits_back(transformed_average, permutation),
        averaged,
        atol=2e-6,
        rtol=2e-6,
    )
    assert len(ALL_SUIT_PERMUTATIONS) == 24


def test_agent_removes_accidental_suit_bias_for_cards_and_suit_choice():
    actor = SuitBiasedActor()
    stats = SuitSymmetryStats()

    def factory(instance):
        return SuitSymmetryPPOAgent(
            instance,
            actor,
            torch.device("cpu"),
            stats=stats,
        )

    _instance, agent = public_game(factory)

    assert agent.playCard("harten", 4).number == 3
    assert agent.changeSort() == "harten"
    assert stats.decisions == 2
    assert stats.changed_from_legacy == 2
    assert stats.ordinary_decisions == 1
    assert stats.suit_decisions == 1


def test_only_own_hand_and_public_features_affect_symmetry_decision():
    actor = SuitBiasedActor()

    def factory(instance):
        return SuitSymmetryPPOAgent(
            instance, actor, torch.device("cpu")
        )

    left, left_agent = public_game(factory)
    right, right_agent = public_game(factory)
    left.players[1].mydeck.cards[:] = [card.Card(4), card.Card(5)]
    right.players[1].mydeck.cards[:] = [card.Card(44), card.Card(45)]
    left.grab_deck.cards[:] = [card.Card(6), card.Card(7)]
    right.grab_deck.cards[:] = [card.Card(47), card.Card(48)]

    torch.testing.assert_close(left_agent.obs(), right_agent.obs())
    assert left_agent.playCard("harten", 4).number == 3
    assert right_agent.playCard("harten", 4).number == 3


def test_common_deal_screen_reports_decisions_and_unsealed_seeds():
    actor = SuitBiasedActor()
    stats = SuitSymmetryStats()
    symmetry = make_suit_symmetry_ppo_factory(
        actor, torch.device("cpu"), stats=stats
    )
    legacy = make_ppo_agent_factory(
        actor, torch.device("cpu"), shield=False
    )

    result = run_symmetry_screen(
        symmetry,
        legacy,
        pairs=1,
        base_seed=83_607_031,
        pool_names=("random",),
        max_turns=30,
    )

    assert result["decision_audit"]["decisions"] > 0
    assert result["deal_seed_audit"]["all_below_92m"]
    assert result["deal_seed_audit"]["count"] == 2
