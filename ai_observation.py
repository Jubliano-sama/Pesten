"""Shared feature and action layout for the trained Pesten policy.

The deployed policy consumes exactly the same 121-value observation produced
during PPO training.  Keep feature indices here so training and serving cannot
silently assign different meanings to the same tensor positions.
"""

import numpy as np
import torch


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

    if isinstance(sorts_played, (np.ndarray, torch.Tensor)):
        sorts = sorts_played[:SORT_COUNT]
    else:
        sorts = [
            sorts_played[offset]
            for offset in range(SORT_COUNT)
        ]
    if isinstance(observation, torch.Tensor):
        sort_target = observation[
            SORTS_PLAYED_START : SORTS_PLAYED_START + SORT_COUNT
        ]
        sort_values = torch.as_tensor(
            sorts,
            dtype=observation.dtype,
            device=observation.device,
        )
        if sort_values.ndim == 1 and sort_target.ndim > 1:
            sort_values = sort_values.reshape(
                (SORT_COUNT,) + (1,) * (sort_target.ndim - 1)
            )
        sort_target[:] = sort_values
        observation[
            PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_END
        ].zero_()
    elif isinstance(observation, np.ndarray):
        sort_target = observation[
            SORTS_PLAYED_START : SORTS_PLAYED_START + SORT_COUNT
        ]
        sort_values = np.asarray(sorts)
        if sort_values.ndim == 1 and sort_target.ndim > 1:
            sort_values = sort_values.reshape(
                (SORT_COUNT,) + (1,) * (sort_target.ndim - 1)
            )
        sort_target[:] = sort_values
        observation[
            PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_END
        ] = 0
    elif isinstance(observation, list):
        observation[
            SORTS_PLAYED_START : SORTS_PLAYED_START + SORT_COUNT
        ] = list(sorts)
        observation[
            PLAYER_HAND_SIZES_START:PLAYER_HAND_SIZES_END
        ] = [0] * PLAYER_HAND_SIZE_SLOTS
    else:
        for offset in range(SORT_COUNT):
            observation[SORTS_PLAYED_START + offset] = sorts[offset]
        for feature_index in range(
            PLAYER_HAND_SIZES_START,
            PLAYER_HAND_SIZES_END,
        ):
            observation[feature_index] = 0

    observation[MODE_INDEX] = 1 if is_change_sort else 0

    order = relative_player_indices(len(hand_sizes), player_index, direction)
    ordered_hand_sizes = [
        hand_sizes[ordered_player_index]
        for ordered_player_index in order[:PLAYER_HAND_SIZE_SLOTS]
    ]
    hand_size_end = PLAYER_HAND_SIZES_START + len(ordered_hand_sizes)
    if isinstance(observation, torch.Tensor):
        if ordered_hand_sizes:
            hand_size_target = observation[
                PLAYER_HAND_SIZES_START:hand_size_end
            ]
            hand_size_values = torch.as_tensor(
                ordered_hand_sizes,
                dtype=observation.dtype,
                device=observation.device,
            )
            if hand_size_values.ndim == 1 and hand_size_target.ndim > 1:
                hand_size_values = hand_size_values.reshape(
                    (len(ordered_hand_sizes),)
                    + (1,) * (hand_size_target.ndim - 1)
                )
            hand_size_target[:] = hand_size_values
    elif isinstance(observation, np.ndarray):
        if ordered_hand_sizes:
            hand_size_target = observation[
                PLAYER_HAND_SIZES_START:hand_size_end
            ]
            hand_size_values = np.asarray(ordered_hand_sizes)
            if hand_size_values.ndim == 1 and hand_size_target.ndim > 1:
                hand_size_values = hand_size_values.reshape(
                    (len(ordered_hand_sizes),)
                    + (1,) * (hand_size_target.ndim - 1)
                )
            hand_size_target[:] = hand_size_values
    elif isinstance(observation, list):
        observation[PLAYER_HAND_SIZES_START:hand_size_end] = (
            ordered_hand_sizes
        )
    else:
        for slot, size in enumerate(ordered_hand_sizes):
            observation[PLAYER_HAND_SIZES_START + slot] = size

    observation[PLAYER_INDEX] = player_index
    observation[DIRECTION_INDEX] = direction
    return observation
