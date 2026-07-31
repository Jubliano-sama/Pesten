import unittest
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
from unittest import mock

import numpy as np

import card
import deck
import game
from audit_luna_benchmark import audit
from hardcodedAgent import (
    HardcodedStrategyAgent,
    _BayesianClues,
)
from luna_benchmark import LunaBenchmarkController
import randomAgent


def configured_game(own_cards, opponent_cards, *, top_card=5):
    instance = game.Game([])
    tree_agent = HardcodedStrategyAgent(
        instance,
        belief_samples=4,
        search_depth=3,
        beam_width=3,
    )
    opponent = randomAgent.AgentRandom()
    tree_agent.mydeck = deck.Deck([card.Card(number) for number in own_cards])
    opponent.mydeck = deck.Deck(
        [card.Card(number) for number in opponent_cards]
    )
    instance.players = [tree_agent, opponent]
    instance.num_players = 2
    instance.current_player_index = 0
    instance.direction = 1
    instance.current_sort = card.Card(top_card).sort
    instance.current_true_number = card.Card(top_card).truenumber
    instance.penalty_amount = 0
    instance.sorts_played = np.zeros(5, dtype=int)
    instance.game_deck = deck.Deck([card.Card(top_card)])
    instance.grab_deck = deck.Deck([])
    instance.public_history = []
    return instance, tree_agent


class BayesianTreeAgentTests(unittest.TestCase):
    def test_played_suit_evidence_increases_posterior_weight(self):
        clues = _BayesianClues(
            suit_counts={"harten": 5},
            rank_counts={},
            standard_plays=5,
            total_plays=5,
            estimated_draws=0,
        )

        heart_weight = HardcodedStrategyAgent._posterior_card_weight(5, clues)
        diamond_weight = HardcodedStrategyAgent._posterior_card_weight(
            17,
            clues,
        )

        self.assertGreater(heart_weight, diamond_weight)

    def test_hidden_opponent_cards_cannot_change_the_move(self):
        first_game, first_agent = configured_game(
            [4, 16, 28, 48],
            [0, 1, 2],
        )
        second_game, second_agent = configured_game(
            [4, 16, 28, 48],
            [36, 37, 49],
        )
        shared_history = [
            {
                "turn": 2,
                "player": 1,
                "action": "play",
                "card": 13,
                "sort": "ruiten",
                "rank": 2,
            }
        ]
        first_game.public_history = list(shared_history)
        second_game.public_history = list(shared_history)

        first_move = first_agent.playCard(
            first_game.current_sort,
            first_game.current_true_number,
        )
        second_move = second_agent.playCard(
            second_game.current_sort,
            second_game.current_true_number,
        )

        self.assertEqual(
            getattr(first_move, "number", None),
            getattr(second_move, "number", None),
        )
        self.assertEqual(
            first_agent.last_search_diagnostics["posterior_action_values"],
            second_agent.last_search_diagnostics["posterior_action_values"],
        )

    def test_last_special_card_is_passed_instead_of_played(self):
        instance, tree_agent = configured_game([49], [12, 13])

        move = tree_agent.playCard(
            instance.current_sort,
            instance.current_true_number,
        )

        self.assertIsNone(move)
        self.assertEqual(tree_agent.last_search_diagnostics["chosen"], "pass")

    def test_live_penalty_prefers_same_rank_when_available(self):
        instance, tree_agent = configured_game([1, 13, 25], [12, 24])
        instance.current_sort = "harten"
        instance.current_true_number = 2
        instance.penalty_amount = 2
        instance.game_deck = deck.Deck([card.Card(1)])

        root_state, _ = tree_agent._root_information_state()
        actions = tree_agent._legal_actions(root_state)

        self.assertEqual(
            {action.card_number for action in actions},
            {1, 13, 25},
        )

    def test_game_records_public_play_ownership(self):
        np.random.seed(11)
        instance = game.Game(
            [HardcodedStrategyAgent, randomAgent.AgentRandom]
        )
        instance.auto_simulate(max_turns=20)

        play_events = [
            event
            for event in instance.public_history
            if event["action"] == "play"
        ]
        self.assertTrue(play_events)
        self.assertTrue(
            all(event["player"] in {0, 1} for event in play_events)
        )

    def test_announced_suit_is_recorded_as_public_bayesian_evidence(self):
        instance, tree_agent = configured_game([48, 6], [12, 24])
        jack = next(
            held_card
            for held_card in tree_agent.mydeck.cards
            if held_card.number == 48
        )

        instance._update_state_with_card(jack, tree_agent)
        instance._apply_card_effects(jack, tree_agent)

        self.assertEqual(
            [event["action"] for event in instance.public_history],
            ["play", "choose_sort"],
        )
        self.assertIn(instance.public_history[-1]["sort"], card.sorts[:4])

    def test_tree_search_handles_every_player_in_three_player_game(self):
        np.random.seed(19)
        instance = game.Game(
            [
                HardcodedStrategyAgent,
                randomAgent.AgentRandom,
                randomAgent.AgentRandom,
            ]
        )

        turns, winner = instance.auto_simulate(max_turns=200)

        self.assertGreater(turns, 0)
        self.assertIn(winner, {0, 1, 2, None})
        diagnostics = instance.players[0].last_search_diagnostics
        self.assertEqual(diagnostics["belief_samples"], 8)
        self.assertGreater(diagnostics["nodes"], 0)

    def test_luna_protocol_keeps_one_stream_without_replaying_deals(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "result.json"
            controller = LunaBenchmarkController(
                games=2,
                seed=73,
                output=output_path,
            )
            scripted_stdin = io.StringIO("PASS\nHARTEN\n" * 2_000)
            captured_stdout = io.StringIO()

            with mock.patch.object(sys, "stdin", scripted_stdin):
                with redirect_stdout(captured_stdout):
                    controller.run()

            self.assertEqual(len(controller.results), 2)
            self.assertEqual(
                [result.luna_seat for result in controller.results],
                [0, 1],
            )
            self.assertEqual(
                len({result.seed for result in controller.results}),
                2,
            )
            self.assertTrue(output_path.is_file())
            self.assertTrue(
                output_path.with_suffix(".audit.jsonl").is_file()
            )
            protocol_output = captured_stdout.getvalue()
            self.assertIn("BENCHMARK_COMPLETE", protocol_output)
            for line in protocol_output.splitlines():
                if line.startswith(("GAME_START ", "GAME_RESULT ")):
                    self.assertNotIn('"seed"', line)
            for line in output_path.with_suffix(
                ".audit.jsonl"
            ).read_text(encoding="utf-8").splitlines():
                self.assertNotIn('"seed"', line)
            audit_report = audit(
                output_path.with_suffix(".audit.jsonl"),
                output_path,
            )
            self.assertEqual(audit_report["status"], "pass")
            self.assertTrue(audit_report["unique_hidden_deal_per_game"])


if __name__ == "__main__":
    unittest.main()
