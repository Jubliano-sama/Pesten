import unittest

import torch

from ai_observation import (
    DIRECTION_INDEX,
    MODE_INDEX,
    PLAYER_HAND_SIZES_START,
    PLAYER_INDEX,
    SORTS_PLAYED_START,
)
from web_server import AIPlayer, GameRoom, GameState, PlayerState


class FixedLogitModel:
    def __init__(self, scores):
        self.scores = scores
        self.last_observation = None

    def __call__(self, observation):
        self.last_observation = observation.detach().cpu().clone()
        logits = torch.zeros((1, 55), dtype=torch.float32)
        for action, score in self.scores.items():
            logits[0, action] = score
        return logits


def make_ai(scores=None):
    ai = AIPlayer.__new__(AIPlayer)
    ai.device = torch.device("cpu")
    ai.model = FixedLogitModel(scores or {})
    ai.model_loaded = True
    ai.fallback_reason = ""
    return ai


def make_state(players, *, direction=-1):
    return GameState(
        players=players,
        current_player_index=2,
        direction=direction,
        current_sort="harten",
        current_true_number=6,
        penalty_amount=0,
        game_deck=[5],
        grab_deck=[],
        game_over=False,
        winner=None,
        turn_count=0,
        sorts_played=[3, 5, 7, 11, 13],
    )


class WebAIContractTests(unittest.TestCase):
    def test_web_observation_matches_training_layout_and_direction(self):
        players = [
            PlayerState("p0", "zero", [0, 12]),
            PlayerState("p1", "one", [1, 13, 25, 37]),
            PlayerState("p2", "ai", [5, 17, 29, 41, 49, 50], is_ai=True),
            PlayerState("p3", "three", list(range(8))),
        ]
        state = make_state(players)
        ai = make_ai()

        observation = ai.get_observation(players[2], state)

        self.assertEqual(
            observation[SORTS_PLAYED_START:SORTS_PLAYED_START + 5].tolist(),
            [3, 5, 7, 11, 13],
        )
        self.assertEqual(
            observation[PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_START + 4].tolist(),
            [6, 4, 2, 8],
        )
        self.assertEqual(observation[PLAYER_INDEX], 2)
        self.assertEqual(observation[DIRECTION_INDEX], -1)
        self.assertEqual(observation[MODE_INDEX], 0)

    def test_suit_choice_sets_the_trained_mode_feature(self):
        players = [
            PlayerState("p0", "human", [0]),
            PlayerState("p1", "ai", [50], is_ai=True),
        ]
        state = make_state(players, direction=1)
        state.current_player_index = 1
        ai = make_ai({50: 5})

        self.assertEqual(ai.choose_action(players[1], state, is_change_sort=True), 50)
        captured = ai.model.last_observation.squeeze(0)
        self.assertEqual(captured[MODE_INDEX].item(), 1)
        self.assertEqual(captured[113].item(), 0)

    def test_drawn_card_decision_uses_the_training_action_mask(self):
        player = PlayerState("p0", "ai", [0, 1], is_ai=True)
        other = PlayerState("p1", "human", [12, 24])
        state = make_state([player, other], direction=1)
        state.current_player_index = 0
        ai = make_ai({0: 10, 1: 5, 54: 0})

        # Card 1 is the newly drawn card. Training considered every compatible
        # card after the draw, so the existing card 0 remains a competing action.
        self.assertEqual(ai.choose_action_with_drawn_card(player, state, 1), 0)
        captured = ai.model.last_observation.squeeze(0)
        self.assertEqual(captured[0].item(), 1)
        self.assertEqual(captured[1].item(), 1)

    def test_penalty_observation_only_marks_matching_rank_playable(self):
        player = PlayerState("p0", "ai", [0, 1], is_ai=True)
        other = PlayerState("p1", "human", [12, 24])
        state = make_state([player, other], direction=1)
        state.current_player_index = 0
        state.current_true_number = 2
        state.penalty_amount = 2
        ai = make_ai()

        observation = ai.get_observation(player, state)
        mask = ai.get_action_mask(observation)

        self.assertEqual(observation[50].item(), 0)
        self.assertEqual(observation[51].item(), 1)
        self.assertEqual(mask[0].item(), 0)
        self.assertEqual(mask[1].item(), 1)

    def test_played_jacks_increment_the_trained_special_sort_bucket(self):
        room = GameRoom("TEST")
        jack_player = PlayerState("p0", "jack", [50, 3])
        other = PlayerState("p1", "other", [12])
        room.game_state.players = [jack_player, other]
        room.game_state.current_player_index = 0
        room.game_state.current_sort = "harten"
        room.game_state.current_true_number = 6
        room.game_state.game_deck = [5]
        room.game_state.sorts_played = [0, 0, 0, 0, 0]

        result = room.play_card(jack_player.id, 50)

        self.assertTrue(result["success"])
        self.assertEqual(room.game_state.sorts_played, [0, 0, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
