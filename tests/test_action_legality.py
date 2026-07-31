import unittest

import numpy as np

import card
import deck
import game
from ai_observation import PASS_ACTION
from batched_pesten import BatchedPesten, POLICY_FIRST_LEGAL


class ScriptedAgent:
    def __init__(self, action=None):
        self.action = action
        self.mydeck = deck.Deck([])

    def addCard(self, drawn_card):
        self.mydeck.cards.append(drawn_card)

    def remove(self, played_card):
        for held_card in self.mydeck.cards:
            if held_card.number == played_card.number:
                self.mydeck.cards.remove(held_card)
                return

    def playCard(self, current_sort, current_true_number):
        return self.action

    def changeSort(self):
        return "harten"


class ActionLegalityTests(unittest.TestCase):
    def configured_canonical(self):
        instance = game.Game([])
        actor = ScriptedAgent()
        opponent = ScriptedAgent()
        actor.mydeck = deck.Deck([card.Card(4), card.Card(17)])
        opponent.mydeck = deck.Deck([card.Card(28)])
        instance.players = [actor, opponent]
        instance.num_players = 2
        instance.current_player_index = 0
        instance.current_sort = "harten"
        instance.current_true_number = 5
        instance.game_deck = deck.Deck([card.Card(4)])
        instance.grab_deck = deck.Deck([card.Card(20)])
        return instance, actor

    def test_canonical_rejects_compatible_card_not_in_hand_without_mutation(self):
        instance, actor = self.configured_canonical()
        fabricated = card.Card(0)
        before = (
            tuple(held.number for held in actor.mydeck.cards),
            tuple(played.number for played in instance.game_deck.cards),
            tuple(instance.public_history),
            instance.cards_played,
        )

        with self.assertRaisesRegex(ValueError, "not in hand"):
            instance._process_played_card(
                fabricated,
                actor,
                is_last_card=False,
            )

        after = (
            tuple(held.number for held in actor.mydeck.cards),
            tuple(played.number for played in instance.game_deck.cards),
            tuple(instance.public_history),
            instance.cards_played,
        )
        self.assertEqual(after, before)

    def test_canonical_rejects_incompatible_owned_card(self):
        instance, actor = self.configured_canonical()
        incompatible = next(
            held for held in actor.mydeck.cards if held.number == 17
        )

        with self.assertRaisesRegex(ValueError, "not compatible"):
            instance._process_played_card(
                incompatible,
                actor,
                is_last_card=False,
            )

        self.assertEqual(
            [held.number for held in actor.mydeck.cards],
            [4, 17],
        )
        self.assertEqual([played.number for played in instance.game_deck.cards], [4])
        self.assertEqual(instance.public_history, [])

    def test_batched_rejects_card_not_in_hand_without_mutation(self):
        engine = BatchedPesten(
            np.full((1, 2), POLICY_FIRST_LEGAL, dtype=np.int8),
            np.full((1, 2), None, dtype=object),
            np.asarray([73], dtype=np.uint32),
        )
        player = int(engine.current_player[0])
        engine.current_sort[0] = 0
        missing_compatible = next(
            number
            for number in range(50)
            if engine.hand_counts[0, player, number] == 0
            and card.Card(number).compatible("harten", 5)
        )
        before = tuple(
            array.copy()
            for array in (
                engine.hands,
                engine.hand_counts,
                engine.hand_sizes,
                engine.discards,
                engine.discard_counts,
                engine.current_sort,
                engine.current_rank,
                engine.turn_counts,
            )
        )

        with self.assertRaisesRegex(ValueError, "not in hand"):
            engine.apply_action(0, missing_compatible)

        for previous, current in zip(
            before,
            (
                engine.hands,
                engine.hand_counts,
                engine.hand_sizes,
                engine.discards,
                engine.discard_counts,
                engine.current_sort,
                engine.current_rank,
                engine.turn_counts,
            ),
        ):
            np.testing.assert_array_equal(current, previous)

    def test_batched_rejects_out_of_range_card_action(self):
        engine = BatchedPesten(
            np.full((1, 2), POLICY_FIRST_LEGAL, dtype=np.int8),
            np.full((1, 2), None, dtype=object),
            np.asarray([91], dtype=np.uint32),
        )

        with self.assertRaisesRegex(ValueError, "invalid card action"):
            engine.apply_action(0, PASS_ACTION + 1)


if __name__ == "__main__":
    unittest.main()
