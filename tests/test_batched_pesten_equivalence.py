import unittest

import numpy as np
import torch
import torch.nn as nn

import card
import deck
import game
import smartAgent
from ai_observation import (
    ACTION_COUNT,
    OBSERVATION_SIZE,
    PASS_ACTION,
)
from batched_pesten import (
    ActorPolicy,
    BatchedPesten,
    POLICY_FIRST_LEGAL,
    POLICY_RL,
    POLICY_SMART,
    collect_batched_rollout,
)


class FirstLegalAgent:
    """Deterministic policy shared by canonical and batched rule tests."""

    def __init__(self):
        self.mydeck = deck.Deck([])

    def addCard(self, drawn_card):
        self.mydeck.cards.append(drawn_card)

    def remove(self, played_card):
        for candidate in self.mydeck.cards:
            if candidate.number == played_card.number:
                self.mydeck.cards.remove(candidate)
                return

    def playCard(self, current_sort, current_true_number):
        for candidate in self.mydeck.cards:
            if candidate.compatible(current_sort, current_true_number):
                return candidate
        return None

    def changeSort(self):
        return "harten"


class LowestLegalActor(nn.Module):
    def forward(self, observations):
        scores = -torch.arange(
            ACTION_COUNT,
            dtype=observations.dtype,
            device=observations.device,
        )
        return scores.expand(len(observations), -1)


class BatchedPestenEquivalenceTests(unittest.TestCase):
    def test_matches_canonical_game_for_many_deterministic_seeds(self):
        seed_count = 128
        kinds = np.full(
            (seed_count, 2),
            POLICY_FIRST_LEGAL,
            dtype=np.int8,
        )
        keys = np.full((seed_count, 2), None, dtype=object)
        vector_games = BatchedPesten(
            kinds,
            keys,
            np.arange(seed_count, dtype=np.uint32),
            max_turns=500,
        )
        vector_games.run_non_neural()

        for seed in range(seed_count):
            np.random.seed(seed)
            canonical = game.Game([FirstLegalAgent, FirstLegalAgent])
            canonical.auto_simulate(max_turns=500)
            with self.subTest(seed=seed):
                self.assertEqual(
                    canonical.turn_count,
                    int(vector_games.turn_counts[seed]),
                )
                self.assertEqual(
                    -1 if canonical.winner is None else canonical.winner,
                    int(vector_games.winners[seed]),
                )
                self.assertEqual(
                    canonical.current_player_index,
                    int(vector_games.current_player[seed]),
                )
                self.assertEqual(
                    canonical.direction,
                    int(vector_games.direction[seed]),
                )
                self.assertEqual(
                    canonical.penalty_amount,
                    int(vector_games.penalty_amount[seed]),
                )
                self.assertEqual(
                    canonical.current_true_number,
                    int(vector_games.current_rank[seed]),
                )
                self.assertEqual(
                    card.sorts.index(canonical.current_sort),
                    int(vector_games.current_sort[seed]),
                )
                self.assertEqual(
                    canonical.cards_played,
                    int(vector_games.cards_played[seed]),
                )
                self.assertEqual(
                    canonical.game_over,
                    bool(vector_games.game_over[seed]),
                )
                canonical_hands = [
                    [held.number for held in player.mydeck.cards]
                    for player in canonical.players
                ]
                vector_hands = [
                    vector_games.hands[
                        seed,
                        player_index,
                        : vector_games.hand_sizes[seed, player_index],
                    ]
                    .astype(int)
                    .tolist()
                    for player_index in range(2)
                ]
                self.assertEqual(canonical_hands, vector_hands)
                self.assertEqual(
                    [held.number for held in canonical.grab_deck.cards],
                    vector_games.draw_piles[
                        seed, : vector_games.draw_counts[seed]
                    ]
                    .astype(int)
                    .tolist(),
                )
                self.assertEqual(
                    [held.number for held in canonical.game_deck.cards],
                    vector_games.discards[
                        seed, : vector_games.discard_counts[seed]
                    ]
                    .astype(int)
                    .tolist(),
                )

        # Prove that the differential corpus actually traversed every special
        # branch named in the rollout contract.
        for event in (
            "draw",
            "draw_retry",
            "penalty_draw",
            "reverse",
            "skip",
            "extra_turn",
            "suit_change",
            "duplicate_play",
        ):
            self.assertGreater(vector_games.event_counts[event], 0, event)

    def test_max_turn_truncation_matches_canonical_contract(self):
        np.random.seed(90210)
        canonical = game.Game([FirstLegalAgent, FirstLegalAgent])
        canonical.auto_simulate(max_turns=1)
        engine = BatchedPesten(
            np.full((1, 2), POLICY_FIRST_LEGAL, dtype=np.int8),
            np.full((1, 2), None, dtype=object),
            np.asarray([90210], dtype=np.uint32),
            max_turns=1,
        )
        engine.run_non_neural()

        self.assertEqual(canonical.turn_count, int(engine.turn_counts[0]))
        self.assertEqual(canonical.game_over, bool(engine.game_over[0]))
        self.assertEqual(
            -1 if canonical.winner is None else canonical.winner,
            int(engine.winners[0]),
        )

    def test_batched_smart_policy_matches_canonical_policy(self):
        rng = np.random.RandomState(7331)
        engine = BatchedPesten(
            np.full((1, 2), POLICY_SMART, dtype=np.int8),
            np.full((1, 2), None, dtype=object),
            np.asarray([1], dtype=np.uint32),
            smart_agents_have_game_reference=False,
        )
        for case_index in range(512):
            hand_size = int(rng.randint(1, 25))
            hand = rng.choice(50, hand_size, replace=True).astype(np.int8)
            engine.hands[0, 0].fill(-1)
            engine.hands[0, 0, :hand_size] = hand
            engine.hand_sizes[0, 0] = hand_size
            engine.hand_counts[0, 0].fill(0)
            for number in hand:
                engine.hand_counts[0, 0, number] += 1
            engine.current_player[0] = 0
            engine.current_sort[0] = rng.randint(0, 5)
            engine.current_rank[0] = rng.choice(
                (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
            )
            engine.penalty_amount[0] = rng.choice((0, 0, 0, 2, 5))

            canonical = smartAgent.Agent()
            canonical.mydeck.cards = [
                card.Card(int(number)) for number in hand
            ]
            selected = canonical.playCard(
                card.sorts[int(engine.current_sort[0])],
                int(engine.current_rank[0]),
            )
            expected_action = (
                PASS_ACTION if selected is None else selected.number
            )
            with self.subTest(case=case_index, decision="card"):
                self.assertEqual(engine.smart_action(0), expected_action)
            with self.subTest(case=case_index, decision="suit"):
                self.assertEqual(
                    engine.choose_suit(0, POLICY_SMART),
                    card.sorts.index(canonical.changeSort()),
                )

    def test_hidden_card_identities_never_enter_policy_observation(self):
        kinds = np.full((1, 2), POLICY_FIRST_LEGAL, dtype=np.int8)
        keys = np.full((1, 2), None, dtype=object)
        left = BatchedPesten(kinds, keys, np.asarray([44], dtype=np.uint32))
        right = BatchedPesten(kinds, keys, np.asarray([44], dtype=np.uint32))

        # Replace every opponent card identity while preserving only the public
        # hand size, and separately vary the private draw-pile order.
        opponent_size = int(left.hand_sizes[0, 1])
        right.hands[0, 1, :opponent_size] = np.arange(
            20, 20 + opponent_size, dtype=np.int8
        )
        right.hand_counts[0, 1].fill(0)
        for number in right.hands[0, 1, :opponent_size]:
            right.hand_counts[0, 1, number] += 1
        left.draw_piles[0, : left.draw_counts[0]] = left.draw_piles[
            0, : left.draw_counts[0]
        ][::-1]
        self.assertEqual(
            left.observation(0).tobytes(),
            right.observation(0).tobytes(),
        )

        # Opponents draw different private cards.  Only their new hand size is
        # public, so the observing player's policy bytes must still match.
        left.draw_piles[0, left.draw_counts[0] - 1] = 3
        right.draw_piles[0, right.draw_counts[0] - 1] = 31
        left._draw_card(0, 1)
        right._draw_card(0, 1)
        self.assertEqual(
            left.observation(0).tobytes(),
            right.observation(0).tobytes(),
        )

        # Once different cards are publicly played, rank/suit history is valid
        # public evidence and the observations should diverge.
        left.current_sort[0], left.current_rank[0] = 0, 4
        right.current_sort[0], right.current_rank[0] = 2, 8
        left.sorts_played[0, 0] += 1
        right.sorts_played[0, 2] += 1
        self.assertNotEqual(
            left.observation(0).tobytes(),
            right.observation(0).tobytes(),
        )

    def test_multiple_trainable_seats_share_batched_inference(self):
        games = 12
        players = 3
        actor = LowestLegalActor()
        engine = BatchedPesten(
            np.full((games, players), POLICY_RL, dtype=np.int8),
            np.full((games, players), "current", dtype=object),
            np.arange(1000, 1000 + games, dtype=np.uint32),
            max_turns=100,
        )
        result = collect_batched_rollout(
            engine,
            {
                "current": ActorPolicy(
                    actor=actor,
                    device=torch.device("cpu"),
                    temperature=0.8,
                    collect_data=True,
                    greedy=True,
                )
            },
            output_device=torch.device("cpu"),
        )

        self.assertEqual(result.observations.shape[1], OBSERVATION_SIZE)
        self.assertEqual(len(result.actions), sum(result.episode_lengths))
        self.assertEqual(len(result.episode_lengths), games * players)
        self.assertLess(result.inference_calls, result.policy_decisions)
        self.assertTrue(torch.all(result.actions >= 0))
        self.assertTrue(torch.all(result.actions <= PASS_ACTION))
        offset = 0
        for episode_id, length in enumerate(result.episode_lengths):
            game_index, player_index = divmod(episode_id, players)
            episode_actions = result.actions[offset : offset + length]
            episode_observations = result.observations[offset : offset + length]
            episode_rewards = result.rewards[offset : offset + length]
            for observation, action in zip(
                episode_observations,
                episode_actions,
            ):
                action_value = int(action)
                if observation[105] != 0:
                    self.assertIn(action_value, (50, 51, 52, 53))
                elif action_value != PASS_ACTION:
                    self.assertGreater(observation[50 + action_value], 0)
            base_rewards = torch.where(
                episode_actions == PASS_ACTION,
                -0.1,
                0.0,
            )
            torch.testing.assert_close(
                episode_rewards[:-1],
                base_rewards[:-1],
            )
            winner = int(engine.winners[game_index])
            terminal = (
                0.0
                if winner < 0
                else (1.0 if winner == player_index else -1.0)
            )
            self.assertAlmostEqual(
                float(episode_rewards[-1]),
                float(base_rewards[-1]) + terminal,
                places=6,
            )
            offset += length


if __name__ == "__main__":
    unittest.main()
