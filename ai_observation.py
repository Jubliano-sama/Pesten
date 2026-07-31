"""Shared feature and action layout for the trained Pesten policy.

The deployed policy consumes exactly the same 121-value observation produced
during PPO training.  Keep feature indices here so training and serving cannot
silently assign different meanings to the same tensor positions.
"""

OBSERVATION_SIZE = 121

HAND_START = 0
HAND_END = 50
COMPATIBLE_START = 50
COMPATIBLE_END = 100

SORTS_PLAYED_START = 100
SORT_COUNT = 5
MODE_INDEX = 105

PLAYER_HAND_SIZES_START = 106
PLAYER_HAND_SIZES_END = 119
PLAYER_HAND_SIZE_SLOTS = PLAYER_HAND_SIZES_END - PLAYER_HAND_SIZES_START
PLAYER_INDEX = 119
DIRECTION_INDEX = 120

ACTION_COUNT = 55
SUIT_ACTION_START = 50
SUIT_ACTION_END = 54
PASS_ACTION = 54


def relative_player_indices(player_count, player_index, direction):
    """Return player indices in the order encoded during training."""
    if player_count < 1:
        return ()
    if direction not in (-1, 1):
        raise ValueError(f"direction must be -1 or 1, got {direction}")

    # The training agents historically used player zero for ordering when they
    # were not present in the game, while recording -1 in PLAYER_INDEX.
    order_start = player_index if 0 <= player_index < player_count else 0
    return tuple(
        (order_start + offset * direction) % player_count
        for offset in range(player_count)
    )


def populate_game_context(
    observation,
    *,
    sorts_played,
    hand_sizes,
    player_index,
    direction,
    is_change_sort=False,
):
    """Populate observation indices 100..120 according to the trained layout."""
    if len(observation) != OBSERVATION_SIZE:
        raise ValueError(
            f"observation must contain {OBSERVATION_SIZE} values, "
            f"got {len(observation)}"
        )
    if len(sorts_played) < SORT_COUNT:
        raise ValueError(
            f"sorts_played must contain at least {SORT_COUNT} values, "
            f"got {len(sorts_played)}"
        )

    for offset in range(SORT_COUNT):
        observation[SORTS_PLAYED_START + offset] = sorts_played[offset]

    observation[MODE_INDEX] = 1 if is_change_sort else 0

    # Clear every reserved slot in case a caller reuses an observation.
    for feature_index in range(PLAYER_HAND_SIZES_START, PLAYER_HAND_SIZES_END):
        observation[feature_index] = 0

    order = relative_player_indices(len(hand_sizes), player_index, direction)
    for slot, ordered_player_index in enumerate(order[:PLAYER_HAND_SIZE_SLOTS]):
        observation[PLAYER_HAND_SIZES_START + slot] = hand_sizes[ordered_player_index]

    observation[PLAYER_INDEX] = player_index
    observation[DIRECTION_INDEX] = direction
    return observation
