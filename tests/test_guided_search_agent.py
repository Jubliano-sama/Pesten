import unittest

import numpy as np
import torch
import torch.nn as nn

from benchmark_guided_search import bootstrap_pair_ci
import card
import deck
import game
from guided_search_agent import (
    GuidedSearchConfig,
    PPOGuidedBayesianAgent,
    _minmax,
)


class FixedActor(nn.Module):
    def __init__(self, preferred_action=3, preferred_suit_action=50):
        super().__init__()
        self.preferred_action = preferred_action
        self.preferred_suit_action = preferred_suit_action

    def forward(self, observations):
        logits = torch.zeros(
            (len(observations), 55),
            dtype=observations.dtype,
            device=observations.device,
        )
        suit_mode = observations[:, 105] != 0
        logits[~suit_mode, self.preferred_action] = 10.0
        logits[suit_mode, self.preferred_suit_action] = 10.0
        return logits


class LifecycleSensitiveSuitActor(FixedActor):
    """Choose a different suit before versus after the jack leaves the hand."""

    def forward(self, observations):
        logits = super().forward(observations)
        suit_rows = observations[:, 105] != 0
        if torch.any(suit_rows):
            logits[suit_rows, 50:54] = 0.0
            jack_still_held = observations[suit_rows, 48] > 0
            row_indices = torch.nonzero(suit_rows, as_tuple=False).flatten()
            logits[row_indices[jack_still_held], 50] = 10.0
            logits[row_indices[~jack_still_held], 51] = 10.0
        return logits


class PassiveOpponent:
    def __init__(self, numbers=()):
        self.mydeck = deck.Deck([card.Card(number) for number in numbers])

    def addCard(self, held_card):
        self.mydeck.cards.append(held_card)

    def remove(self, held_card):
        self.mydeck.cards.remove(held_card)

    def playCard(self, current_sort, current_true_number):
        return None

    def changeSort(self):
        return card.sorts[0]


class CountOnlyHiddenHand:
    def __init__(self, count):
        self._count = count

    @property
    def cards(self):
        raise AssertionError("live opponent identities were inspected")

    def cardCount(self):
        return self._count


class ForbiddenDrawPile:
    def __getattribute__(self, name):
        raise AssertionError("live draw pile was inspected")


def configured_agent(
    *,
    own_numbers=(3, 6, 16),
    opponent_numbers=(0, 1),
    preferred_action=3,
        config=None,
    top_number=5,
):
    instance = game.Game([])
    agent = PPOGuidedBayesianAgent(
        instance,
        FixedActor(preferred_action),
        "cpu",
        config=config
        or GuidedSearchConfig(
            belief_samples=2,
            search_depth=2,
            beam_width=2,
        ),
    )
    opponent = PassiveOpponent(opponent_numbers)
    agent.mydeck = deck.Deck(
        [card.Card(number) for number in own_numbers]
    )
    instance.players = [agent, opponent]
    instance.num_players = 2
    instance.current_player_index = 0
    instance.turn_count = 19
    instance.direction = 1
    instance.current_sort = card.Card(top_number).sort
    instance.current_true_number = card.Card(top_number).truenumber
    instance.penalty_amount = 0
    instance.sorts_played = np.zeros(5, dtype=int)
    instance.game_deck = deck.Deck([card.Card(top_number)])
    instance.grab_deck = deck.Deck([])
    instance.public_history = []
    return instance, agent


class GuidedSearchAgentTests(unittest.TestCase):
    def test_non_tactical_state_is_exactly_the_ppo_default(self):
        _, agent = configured_agent(
            own_numbers=(3, 6, 16, 20, 30, 40),
            opponent_numbers=(0, 1, 2, 10, 11),
            preferred_action=3,
        )

        chosen = agent.playCard("harten", 6)

        self.assertEqual(chosen.number, 3)
        self.assertFalse(agent.last_search_diagnostics["invoked"])
        self.assertEqual(agent.search_count, 0)

    def test_public_tactical_state_invokes_posterior_search(self):
        _, agent = configured_agent(
            own_numbers=(3, 6, 16),
            opponent_numbers=(0, 1),
            preferred_action=3,
            config=GuidedSearchConfig(
                belief_samples=2,
                search_depth=2,
                beam_width=2,
                decision_mode="veto",
            ),
        )

        chosen = agent.playCard("harten", 6)

        self.assertIn(chosen.number, {3, 6})
        self.assertTrue(agent.last_search_diagnostics["invoked"])
        self.assertGreater(agent.last_search_diagnostics["nodes"], 0)
        self.assertEqual(agent.search_count, 1)

    def test_hidden_identities_and_live_draw_order_cannot_change_decision(self):
        def decision(opponent_numbers, draw_numbers):
            instance, agent = configured_agent(
                own_numbers=(3, 6, 16),
                opponent_numbers=opponent_numbers,
                preferred_action=3,
                config=GuidedSearchConfig(
                    belief_samples=2,
                    search_depth=2,
                    beam_width=2,
                    decision_mode="veto",
                ),
            )
            instance.grab_deck = deck.Deck(
                [card.Card(number) for number in draw_numbers]
            )
            chosen = agent.playCard(
                instance.current_sort,
                instance.current_true_number,
            )
            diagnostics = agent.last_search_diagnostics
            return (
                getattr(chosen, "number", None),
                diagnostics["posterior_action_values"],
                diagnostics["combined"],
            )

        first = decision((0, 1), (2, 4, 8, 9))
        second = decision((30, 40), (49, 48, 47, 46))

        self.assertEqual(first, second)

    def test_search_never_touches_live_hidden_cards_or_draw_pile(self):
        instance, agent = configured_agent(
            own_numbers=(3, 6, 16),
            opponent_numbers=(0, 1),
            preferred_action=3,
            config=GuidedSearchConfig(
                belief_samples=2,
                search_depth=2,
                beam_width=2,
                decision_mode="veto",
            ),
        )
        instance.players[1].mydeck = CountOnlyHiddenHand(2)
        instance.grab_deck = ForbiddenDrawPile()

        chosen = agent.playCard(
            instance.current_sort,
            instance.current_true_number,
        )

        self.assertIn(chosen.number, {3, 6})
        self.assertTrue(agent.last_search_diagnostics["invoked"])

    def test_every_returned_card_is_owned_and_engine_legal(self):
        for seed in range(12):
            np.random.seed(seed)

            def guided_factory(instance):
                return PPOGuidedBayesianAgent(
                    instance,
                    FixedActor(preferred_action=seed % 50),
                    "cpu",
                    config=GuidedSearchConfig(
                        belief_samples=1,
                        search_depth=2,
                        beam_width=2,
                    ),
                )

            instance = game.Game([guided_factory, PassiveOpponent])
            turns, winner = instance.auto_simulate(max_turns=50)
            self.assertGreater(turns, 0)
            self.assertIn(winner, {0, 1, None})

    def test_jack_search_choice_produces_a_valid_public_suit(self):
        _, agent = configured_agent(
            own_numbers=(3, 48),
            opponent_numbers=(0,),
            preferred_action=48,
            config=GuidedSearchConfig(
                belief_samples=1,
                search_depth=1,
                beam_width=2,
                trigger="always",
                decision_mode="veto",
            ),
        )

        chosen = agent.playCard("harten", 6)

        self.assertEqual(chosen.number, 48)
        self.assertIn(agent.changeSort(), card.sorts[:4])

    def test_no_override_defers_jack_suit_to_the_live_ppo_lifecycle(self):
        instance = game.Game([])
        agent = PPOGuidedBayesianAgent(
            instance,
            LifecycleSensitiveSuitActor(preferred_action=48),
            "cpu",
            config=GuidedSearchConfig(trigger="always"),
        )
        opponent = PassiveOpponent((0, 1))
        agent.mydeck = deck.Deck([card.Card(3), card.Card(48)])
        instance.players = [agent, opponent]
        instance.num_players = 2
        instance.current_player_index = 0
        instance.direction = 1
        instance.current_sort = "harten"
        instance.current_true_number = 6
        instance.penalty_amount = 0
        instance.sorts_played = np.zeros(5, dtype=int)
        instance.game_deck = deck.Deck([card.Card(5)])
        instance.public_history = []

        chosen = agent.playCard("harten", 6)
        agent.remove(chosen)

        self.assertEqual(chosen.number, 48)
        self.assertFalse(
            agent.last_search_diagnostics.get("overrode_ppo", False)
        )
        self.assertEqual(agent.changeSort(), "ruiten")

    def test_impossible_blend_override_skips_the_expensive_search(self):
        _, agent = configured_agent(
            own_numbers=(3, 6, 16),
            opponent_numbers=(0, 1),
            preferred_action=3,
        )

        chosen = agent.playCard("harten", 6)

        self.assertEqual(chosen.number, 3)
        self.assertEqual(agent.search_count, 0)
        self.assertEqual(
            agent.last_search_diagnostics["reason"],
            "search_cannot_clear_prior_margin",
        )

    def test_normalization_and_pair_bootstrap_are_pair_preserving(self):
        self.assertEqual(
            set(_minmax({"a": -5.0, "b": 5.0}).values()),
            {0.0, 1.0},
        )
        self.assertEqual(
            bootstrap_pair_ci(
                [0.25] * 8,
                samples=100,
                seed=17,
            ),
            (0.25, 0.25),
        )


if __name__ == "__main__":
    unittest.main()
