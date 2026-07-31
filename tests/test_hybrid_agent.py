import unittest

import numpy as np
import torch
import torch.nn as nn

from benchmark_hybrid import bootstrap_mean_ci
import card
import deck
import game
from hybrid_agent import (
    FrozenGreedyPPOAgent,
    HybridPPOAgent,
    make_ppo_agent_factory,
)


class FixedChoiceActor(nn.Module):
    def __init__(self, preferred_action):
        super().__init__()
        self.preferred_action = preferred_action

    def forward(self, observations):
        logits = torch.zeros(
            (len(observations), 55),
            dtype=observations.dtype,
            device=observations.device,
        )
        logits[:, self.preferred_action] = 10.0
        return logits


class PassiveOpponent:
    def __init__(self):
        self.mydeck = deck.Deck([])

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
        raise AssertionError("opponent card identities were inspected")

    def cardCount(self):
        return self._count


class ForbiddenDrawPile:
    def __getattribute__(self, name):
        raise AssertionError("draw pile was inspected")


def configured_agent(
    agent_class,
    *,
    preferred_action,
    hand_numbers,
    player_count=2,
):
    current_game = game.Game([])
    agent = agent_class(
        current_game,
        FixedChoiceActor(preferred_action),
        torch.device("cpu"),
    )
    opponents = [PassiveOpponent() for _ in range(player_count - 1)]
    current_game.players = [agent, *opponents]
    current_game.num_players = player_count
    current_game.current_player_index = 0
    current_game.current_sort = "harten"
    current_game.current_true_number = 9
    current_game.penalty_amount = 0
    current_game.direction = 1
    current_game.sorts_played = np.zeros(5, dtype=int)
    agent.mydeck.cards = [card.Card(number) for number in hand_numbers]
    return current_game, agent


class HybridAgentTests(unittest.TestCase):
    def test_lone_special_is_passed_instead_of_triggering_finish_penalty(self):
        _, agent = configured_agent(
            HybridPPOAgent,
            preferred_action=6,  # seven of hearts
            hand_numbers=[6],
        )

        chosen = agent.playCard("harten", 9)

        self.assertIsNone(chosen)
        self.assertEqual(agent.last_override_reason, "pass_lone_special")

    def test_lone_special_during_active_penalty_remains_a_ppo_decision(self):
        current_game, agent = configured_agent(
            HybridPPOAgent,
            preferred_action=49,  # joker
            hand_numbers=[49],
        )
        current_game.current_true_number = 0
        current_game.penalty_amount = 5

        chosen = agent.playCard("special", 0)

        self.assertEqual(chosen.number, 49)
        self.assertIsNone(agent.last_override_reason)

    def test_seven_is_played_before_guaranteed_ordinary_finisher(self):
        _, agent = configured_agent(
            HybridPPOAgent,
            preferred_action=3,  # PPO prefers ordinary four of hearts
            hand_numbers=[3, 6],  # seven retains turn; four then remains legal
            player_count=3,
        )

        chosen = agent.playCard("harten", 9)

        self.assertEqual(chosen.number, 6)
        self.assertEqual(
            agent.last_override_reason,
            "preserve_ordinary_finisher",
        )

    def test_two_player_eight_retains_turn_but_multiplayer_eight_does_not(self):
        _, two_player_agent = configured_agent(
            HybridPPOAgent,
            preferred_action=3,
            hand_numbers=[3, 7],  # eight of hearts
            player_count=2,
        )
        _, three_player_agent = configured_agent(
            HybridPPOAgent,
            preferred_action=3,
            hand_numbers=[3, 7],
            player_count=3,
        )

        self.assertEqual(
            two_player_agent.playCard("harten", 9).number,
            7,
        )
        self.assertEqual(
            three_player_agent.playCard("harten", 9).number,
            3,
        )
        self.assertIsNone(three_player_agent.last_override_reason)

    def test_plain_ppo_is_unchanged_for_shielded_tactical_case(self):
        _, plain = configured_agent(
            FrozenGreedyPPOAgent,
            preferred_action=3,
            hand_numbers=[3, 6],
        )
        _, hybrid = configured_agent(
            HybridPPOAgent,
            preferred_action=3,
            hand_numbers=[3, 6],
        )

        self.assertEqual(plain.playCard("harten", 9).number, 3)
        self.assertEqual(hybrid.playCard("harten", 9).number, 6)

    def test_decision_is_invariant_to_hidden_card_identities_and_draw_order(self):
        def decision(opponent_numbers, draw_numbers):
            current_game, agent = configured_agent(
                HybridPPOAgent,
                preferred_action=3,
                hand_numbers=[3, 6],
            )
            current_game.players[1].mydeck.cards = [
                card.Card(number) for number in opponent_numbers
            ]
            current_game.grab_deck = deck.Deck(
                [card.Card(number) for number in draw_numbers]
            )
            observation = agent.obs().clone()
            chosen = agent.playCard("harten", 9)
            return observation, chosen.number

        first_observation, first_action = decision([0, 1, 2], [4, 5, 8])
        second_observation, second_action = decision([20, 30, 40], [49, 48, 47])

        torch.testing.assert_close(first_observation, second_observation)
        self.assertEqual(first_action, second_action)

    def test_decision_never_touches_opponent_identities_or_draw_pile(self):
        current_game, agent = configured_agent(
            HybridPPOAgent,
            preferred_action=3,
            hand_numbers=[3, 6],
        )
        current_game.players[1].mydeck = CountOnlyHiddenHand(3)
        current_game.grab_deck = ForbiddenDrawPile()

        chosen = agent.playCard("harten", 9)

        self.assertEqual(chosen.number, 6)

    def test_factories_reuse_the_exact_frozen_actor(self):
        actor = FixedChoiceActor(3)
        plain_factory = make_ppo_agent_factory(actor, "cpu", shield=False)
        hybrid_factory = make_ppo_agent_factory(actor, "cpu", shield=True)
        current_game = game.Game([])

        plain = plain_factory(current_game)
        hybrid = hybrid_factory(current_game)

        self.assertIs(plain.actor, actor)
        self.assertIs(hybrid.actor, actor)
        self.assertFalse(plain.collect_data)
        self.assertFalse(hybrid.collect_data)

    def test_pair_bootstrap_preserves_constant_paired_effect(self):
        self.assertEqual(
            bootstrap_mean_ci(
                [0.25] * 8,
                samples=100,
                seed=17,
            ),
            (0.25, 0.25),
        )


if __name__ == "__main__":
    unittest.main()
