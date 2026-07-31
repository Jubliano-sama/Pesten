import unittest

from ai_observation import (
    DIRECTION_INDEX,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PLAYER_HAND_SIZES_END,
    PLAYER_HAND_SIZES_START,
    PLAYER_INDEX,
    SORTS_PLAYED_START,
    populate_game_context,
    relative_player_indices,
)


class ObservationContractTests(unittest.TestCase):
    def test_reverse_direction_controls_relative_player_order(self):
        self.assertEqual(relative_player_indices(4, 2, -1), (2, 1, 0, 3))
        self.assertEqual(relative_player_indices(4, 2, 1), (2, 3, 0, 1))

    def test_context_uses_the_trained_feature_indices(self):
        observation = [99.0] * OBSERVATION_SIZE

        populate_game_context(
            observation,
            sorts_played=[3, 5, 7, 11, 13],
            hand_sizes=[2, 4, 6, 8],
            player_index=2,
            direction=-1,
        )

        self.assertEqual(
            observation[SORTS_PLAYED_START:SORTS_PLAYED_START + 5],
            [3, 5, 7, 11, 13],
        )
        self.assertEqual(observation[MODE_INDEX], 0)
        self.assertEqual(
            observation[PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_START + 4],
            [6, 4, 2, 8],
        )
        self.assertEqual(
            observation[PLAYER_HAND_SIZES_START + 4:PLAYER_HAND_SIZES_END],
            [0] * 9,
        )
        self.assertEqual(observation[PLAYER_INDEX], 2)
        self.assertEqual(observation[DIRECTION_INDEX], -1)

    def test_change_sort_mode_and_thirteen_player_capacity(self):
        observation = [0.0] * OBSERVATION_SIZE
        hand_sizes = list(range(1, 14))

        populate_game_context(
            observation,
            sorts_played=[0, 0, 0, 0, 0],
            hand_sizes=hand_sizes,
            player_index=12,
            direction=1,
            is_change_sort=True,
        )

        self.assertEqual(observation[MODE_INDEX], 1)
        self.assertEqual(
            observation[PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_END],
            [13] + list(range(1, 13)),
        )
        self.assertEqual(observation[PLAYER_INDEX], 12)
        self.assertEqual(observation[DIRECTION_INDEX], 1)

    def test_invalid_direction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "direction"):
            relative_player_indices(2, 0, 0)


if __name__ == "__main__":
    unittest.main()
