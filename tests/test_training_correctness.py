import math
import unittest

import torch
import torch.nn as nn

import card
import game
import randomAgent
from ai_observation import MODE_INDEX, OBSERVATION_SIZE, PASS_ACTION
from ppo2 import (
    RL_SupervisedAgent,
    build_action_masks,
    compute_gae,
)


class FixedActor(nn.Module):
    def forward(self, observations):
        logits = torch.zeros((len(observations), 55), device=observations.device)
        logits[:, 0] = 2.0
        logits[:, 1] = 1.0
        return logits


class TrainingCorrectnessTests(unittest.TestCase):
    def test_gae_resets_at_episode_boundaries(self):
        advantages, returns = compute_gae(
            rewards=torch.tensor([0.0, 1.0, 0.0, 10.0]),
            values=torch.zeros(4),
            episode_lengths=[2, 2],
            gamma_=1.0,
            lam=1.0,
        )

        torch.testing.assert_close(
            advantages,
            torch.tensor([1.0, 1.0, 10.0, 10.0]),
        )
        torch.testing.assert_close(returns, advantages)

    def test_action_masks_support_single_observations_and_batches(self):
        observations = torch.zeros((2, OBSERVATION_SIZE))
        observations[0, 50] = 1
        observations[0, 52] = 2
        observations[1, MODE_INDEX] = 1

        masks = build_action_masks(observations)

        self.assertEqual(masks.dtype, torch.bool)
        self.assertEqual(
            torch.nonzero(masks[0]).flatten().tolist(),
            [0, 2, PASS_ACTION],
        )
        self.assertEqual(
            torch.nonzero(masks[1]).flatten().tolist(),
            [50, 51, 52, 53],
        )
        torch.testing.assert_close(
            build_action_masks(observations[0]),
            masks[0],
        )

    def test_rollout_log_probability_uses_requested_temperature(self):
        observation = torch.zeros(OBSERVATION_SIZE)
        observation[50] = 1
        observation[51] = 1
        agent = RL_SupervisedAgent(
            game_instance=None,
            actor=FixedActor(),
            device=torch.device("cpu"),
            temperature=0.5,
        )

        torch.manual_seed(7)
        action, actual_log_probability = agent.act_rl(observation)
        action = PASS_ACTION if action is None else action

        legal_logits = torch.tensor([2.0, 1.0, 0.0]) / 0.5
        expected = torch.log_softmax(legal_logits, dim=0)
        legal_actions = [0, 1, PASS_ACTION]
        expected_log_probability = expected[legal_actions.index(action)].item()
        self.assertTrue(
            math.isclose(
                actual_log_probability,
                expected_log_probability,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        )

    def test_penalty_observation_only_marks_matching_rank_playable(self):
        current_game = game.Game([])
        agent = RL_SupervisedAgent(
            game_instance=current_game,
            actor=FixedActor(),
            device=torch.device("cpu"),
        )
        opponent = randomAgent.AgentRandom()
        current_game.players = [agent, opponent]
        current_game.num_players = 2
        current_game.current_sort = "harten"
        current_game.current_true_number = 2
        current_game.penalty_amount = 2
        current_game.sorts_played = [0, 0, 0, 0, 0]
        agent.mydeck.cards = [card.Card(0), card.Card(1)]

        observation = agent.obs()
        mask = agent.get_action_mask(observation)

        self.assertEqual(observation[50].item(), 0)
        self.assertEqual(observation[51].item(), 1)
        self.assertFalse(mask[0].item())
        self.assertTrue(mask[1].item())

    def test_game_reset_redeals_instead_of_accumulating_hands(self):
        current_game = game.Game(
            [randomAgent.AgentRandom, randomAgent.AgentRandom]
        )
        current_game.reset()
        self.assertEqual(
            [player.mydeck.cardCount() for player in current_game.players],
            [7, 7],
        )

        current_game.cards_played = 499
        current_game.reset()

        self.assertEqual(
            [player.mydeck.cardCount() for player in current_game.players],
            [7, 7],
        )
        self.assertEqual(current_game.cards_played, 0)


if __name__ == "__main__":
    unittest.main()
