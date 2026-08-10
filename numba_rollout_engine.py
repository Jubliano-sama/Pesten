"""Optional exact native kernels for common batched Pesten state transitions."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # pragma: no cover - exercised only without the extra.
    njit = None


CARD_SORT = np.asarray(
    [number // 12 if number < 48 else 4 for number in range(50)],
    dtype=np.int8,
)
CARD_RANK = np.asarray(
    [
        number % 12 + 1 if number < 48 else (13 if number == 48 else 0)
        for number in range(50)
    ],
    dtype=np.int8,
)


def _rl_has_card_decisions_impl(
    game_indices: np.ndarray,
    current_player: np.ndarray,
    hand_counts: np.ndarray,
    current_sort: np.ndarray,
    current_rank: np.ndarray,
    penalty_amount: np.ndarray,
    phase: np.ndarray,
    drawn_card: np.ndarray,
) -> np.ndarray:
    result = np.zeros(len(game_indices), dtype=np.bool_)
    for offset in range(len(game_indices)):
        game_index = int(game_indices[offset])
        if int(phase[game_index]) == 1:
            result[offset] = int(drawn_card[game_index]) >= 0
            continue
        player_index = int(current_player[game_index])
        suit = int(current_sort[game_index])
        rank = int(current_rank[game_index])
        under_penalty = int(penalty_amount[game_index]) > 0
        for number in range(50):
            # Preserve the existing uint8-and-bool vector expression exactly:
            # even duplicate counts have a zero low bit in this availability
            # precheck, while observation masks still expose the card.
            if (
                int(hand_counts[game_index, player_index, number]) & 1
            ) == 0:
                continue
            number_rank = int(CARD_RANK[number])
            if (
                number_rank == rank
                if under_penalty
                else (
                    number_rank == rank
                    or number_rank == 0
                    or number_rank == 13
                    or int(CARD_SORT[number]) == suit
                )
            ):
                result[offset] = True
                break
    return result


def _observations_into_impl(
    game_indices: np.ndarray,
    suit_modes: np.ndarray,
    current_player: np.ndarray,
    hand_counts: np.ndarray,
    current_sort: np.ndarray,
    current_rank: np.ndarray,
    penalty_amount: np.ndarray,
    phase: np.ndarray,
    drawn_card: np.ndarray,
    sorts_played: np.ndarray,
    direction: np.ndarray,
    hand_sizes: np.ndarray,
    num_players: int,
    observations: np.ndarray,
    action_masks: np.ndarray,
) -> None:
    """Populate the exact 121-feature view and 55-action mask in one kernel."""
    slots = min(num_players, 13)
    for offset in range(len(game_indices)):
        game_index = int(game_indices[offset])
        player = int(current_player[game_index])
        suit = int(current_sort[game_index])
        rank = int(current_rank[game_index])
        under_penalty = int(penalty_amount[game_index]) > 0
        retry = int(phase[game_index]) == 1
        drawn = int(drawn_card[game_index])
        for column in range(121):
            observations[offset, column] = 0.0
        for action in range(55):
            action_masks[offset, action] = False
        for number in range(50):
            count = int(hand_counts[game_index, player, number])
            observations[offset, number] = count
            number_rank = int(CARD_RANK[number])
            playable = (
                number == drawn
                if retry
                else (
                    number_rank == rank
                    if under_penalty
                    else (
                        number_rank == rank
                        or number_rank == 0
                        or number_rank == 13
                        or int(CARD_SORT[number]) == suit
                    )
                )
            )
            compatible = count if playable else 0
            observations[offset, 50 + number] = compatible
            action_masks[offset, number] = compatible >= 1
        for sort_index in range(5):
            observations[offset, 100 + sort_index] = sorts_played[
                game_index, sort_index
            ]
        suit_mode = bool(suit_modes[offset])
        observations[offset, 105] = 1.0 if suit_mode else 0.0
        direction_value = int(direction[game_index])
        for slot in range(slots):
            ordered_player = (
                player + direction_value * slot
            ) % num_players
            observations[offset, 106 + slot] = hand_sizes[
                game_index, ordered_player
            ]
        observations[offset, 119] = player
        observations[offset, 120] = direction_value
        if suit_mode:
            for action in range(55):
                action_masks[offset, action] = False
            for action in range(50, 54):
                action_masks[offset, action] = True
        else:
            action_masks[offset, 54] = True


def _apply_simple_card_actions_impl(
    game_indices: np.ndarray,
    actions: np.ndarray,
    num_players: int,
    card_sorts: np.ndarray,
    card_ranks: np.ndarray,
    card_is_penalty: np.ndarray,
    compatibility_masks: np.ndarray,
    hands: np.ndarray,
    hand_counts: np.ndarray,
    hand_sizes: np.ndarray,
    draw_piles: np.ndarray,
    draw_counts: np.ndarray,
    discards: np.ndarray,
    discard_counts: np.ndarray,
    current_sort: np.ndarray,
    current_rank: np.ndarray,
    current_player: np.ndarray,
    turn_player: np.ndarray,
    direction: np.ndarray,
    penalty_amount: np.ndarray,
    sorts_played: np.ndarray,
    turn_counts: np.ndarray,
    cards_played: np.ndarray,
    game_over: np.ndarray,
    winners: np.ndarray,
    phase: np.ndarray,
    turn_started_with_one_card: np.ndarray,
    drawn_card: np.ndarray,
    event_deltas: np.ndarray,
    handled: np.ndarray,
    acting_players: np.ndarray,
    public_event_kinds: np.ndarray,
) -> None:
    """Classify/apply common actions and mark entries needing Python fallback."""
    for offset in range(len(game_indices)):
        game_index = int(game_indices[offset])
        action = int(actions[offset])
        handled[offset] = False
        acting_players[offset] = -1
        public_event_kinds[offset] = 0
        if game_over[game_index]:
            continue
        player_index = int(current_player[game_index])
        phase_value = int(phase[game_index])
        retry_card = False
        penalty_stack = False
        if phase_value == 2:
            if action < 50 or action >= 54:
                continue
            handled[offset] = True
            acting_players[offset] = player_index
            public_event_kinds[offset] = 2
            current_sort[game_index] = action - 50
            event_deltas[6] += 1
            turn_counts[game_index] += 1
            acting_player = int(turn_player[game_index])
            if hand_sizes[game_index, acting_player] == 0:
                winners[game_index] = current_player[game_index]
                game_over[game_index] = True
            if not game_over[game_index]:
                current_player[game_index] = (
                    current_player[game_index] + direction[game_index]
                ) % num_players
            phase[game_index] = 0
            drawn_card[game_index] = -1
            if not game_over[game_index]:
                next_player = int(current_player[game_index])
                turn_player[game_index] = next_player
                turn_started_with_one_card[game_index] = (
                    hand_sizes[game_index, next_player] == 1
                )
            continue
        if phase_value == 1:
            if action != 54 and action != int(drawn_card[game_index]):
                continue
            if action == 54:
                handled[offset] = True
                acting_players[offset] = player_index
                turn_counts[game_index] += 1
                acting_player = int(turn_player[game_index])
                if hand_sizes[game_index, acting_player] == 0:
                    winners[game_index] = current_player[game_index]
                    game_over[game_index] = True
                if not game_over[game_index]:
                    current_player[game_index] = (
                        current_player[game_index] + direction[game_index]
                    ) % num_players
                phase[game_index] = 0
                drawn_card[game_index] = -1
                if not game_over[game_index]:
                    next_player = int(current_player[game_index])
                    turn_player[game_index] = next_player
                    turn_started_with_one_card[game_index] = (
                        hand_sizes[game_index, next_player] == 1
                    )
                continue
            if int(cards_played[game_index]) >= 500:
                continue
            retry_card = True
        elif phase_value != 0:
            continue
        else:
            penalty = int(penalty_amount[game_index])
            if penalty > 0:
                valid_stack = (
                    action >= 0
                    and action < len(card_ranks)
                    and int(card_ranks[action])
                    == int(current_rank[game_index])
                )
                if valid_stack:
                    if (
                        int(cards_played[game_index]) >= 500
                        or (
                            turn_started_with_one_card[game_index]
                            and card_is_penalty[action]
                        )
                        or hand_counts[
                            game_index,
                            player_index,
                            action,
                        ]
                        <= 0
                    ):
                        continue
                    penalty_stack = True
                else:
                    if (
                        action != 54
                        and (
                            action < 0
                            or action >= len(card_ranks)
                            or hand_counts[
                                game_index,
                                player_index,
                                action,
                            ]
                            <= 0
                        )
                    ):
                        continue
                    if int(draw_counts[game_index]) < penalty:
                        continue
                    handled[offset] = True
                    acting_players[offset] = player_index
                    for _ in range(penalty):
                        draw_offset = int(draw_counts[game_index]) - 1
                        drawn = int(draw_piles[game_index, draw_offset])
                        draw_piles[game_index, draw_offset] = -1
                        draw_counts[game_index] = draw_offset
                        hand_offset = int(
                            hand_sizes[game_index, player_index]
                        )
                        hands[
                            game_index,
                            player_index,
                            hand_offset,
                        ] = drawn
                        hand_sizes[
                            game_index,
                            player_index,
                        ] = hand_offset + 1
                        hand_counts[
                            game_index,
                            player_index,
                            drawn,
                        ] += 1
                        event_deltas[4] += 1
                        event_deltas[7] += 1
                    penalty_amount[game_index] = 0
                    if int(current_rank[game_index]) == 0:
                        phase[game_index] = 2
                        continue
                    turn_counts[game_index] += 1
                    acting_player = int(turn_player[game_index])
                    if hand_sizes[game_index, acting_player] == 0:
                        winners[game_index] = current_player[game_index]
                        game_over[game_index] = True
                    if not game_over[game_index]:
                        current_player[game_index] = (
                            current_player[game_index]
                            + direction[game_index]
                        ) % num_players
                    phase[game_index] = 0
                    drawn_card[game_index] = -1
                    if not game_over[game_index]:
                        next_player = int(current_player[game_index])
                        turn_player[game_index] = next_player
                        turn_started_with_one_card[game_index] = (
                            hand_sizes[game_index, next_player] == 1
                        )
                    continue
            elif int(cards_played[game_index]) >= 500:
                continue

        if action == 54:
            if int(draw_counts[game_index]) <= 0:
                continue
        else:
            if action < 0 or action >= len(card_ranks):
                continue
            if (
                not retry_card
                and not penalty_stack
                and turn_started_with_one_card[game_index]
                and card_is_penalty[action]
            ):
                continue
            if hand_counts[game_index, player_index, action] <= 0:
                continue
            if (
                not penalty_stack
                and not compatibility_masks[
                    int(current_sort[game_index]),
                    int(current_rank[game_index]),
                    action,
                ]
            ):
                continue
        handled[offset] = True
        acting_players[offset] = player_index
        if action != 54:
            public_event_kinds[offset] = 1

        cards_played[game_index] += 1

        if action == 54:
            draw_offset = int(draw_counts[game_index]) - 1
            drawn = int(draw_piles[game_index, draw_offset])
            draw_piles[game_index, draw_offset] = -1
            draw_counts[game_index] = draw_offset
            hand_offset = int(hand_sizes[game_index, player_index])
            hands[game_index, player_index, hand_offset] = drawn
            hand_sizes[game_index, player_index] = hand_offset + 1
            hand_counts[game_index, player_index, drawn] += 1
            event_deltas[4] += 1
            drawn_rank = int(card_ranks[drawn])
            compatible = (
                drawn_rank == int(current_rank[game_index])
                or drawn_rank == 0
                or drawn_rank == 13
                or int(card_sorts[drawn]) == int(current_sort[game_index])
            )
            if compatible:
                drawn_card[game_index] = drawn
                phase[game_index] = 1
                event_deltas[5] += 1
                continue

            turn_counts[game_index] += 1
            acting_player = int(turn_player[game_index])
            if hand_sizes[game_index, acting_player] == 0:
                winners[game_index] = current_player[game_index]
                game_over[game_index] = True
            if not game_over[game_index]:
                current_player[game_index] = (
                    current_player[game_index] + direction[game_index]
                ) % num_players
            phase[game_index] = 0
            drawn_card[game_index] = -1
            if not game_over[game_index]:
                next_player = int(current_player[game_index])
                turn_player[game_index] = next_player
                turn_started_with_one_card[game_index] = (
                    hand_sizes[game_index, next_player] == 1
                )
            continue

        if hand_counts[game_index, player_index, action] > 1:
            event_deltas[0] += 1
        discard_offset = int(discard_counts[game_index])
        discards[game_index, discard_offset] = action
        discard_counts[game_index] = discard_offset + 1
        card_sort = int(card_sorts[action])
        card_rank = int(card_ranks[action])
        current_sort[game_index] = card_sort
        current_rank[game_index] = card_rank
        sorts_played[game_index, card_sort] += 1

        size = int(hand_sizes[game_index, player_index])
        hand_offset = -1
        for index in range(size):
            if int(hands[game_index, player_index, index]) == action:
                hand_offset = index
                break
        if hand_offset >= 0:
            for index in range(hand_offset, size - 1):
                hands[game_index, player_index, index] = hands[
                    game_index,
                    player_index,
                    index + 1,
                ]
            hands[game_index, player_index, size - 1] = -1
            hand_sizes[game_index, player_index] = size - 1
            hand_counts[game_index, player_index, action] -= 1

        if card_rank == 7:
            current_player[game_index] = (
                current_player[game_index] - direction[game_index]
            ) % num_players
            event_deltas[1] += 1
        elif card_rank == 8:
            current_player[game_index] = (
                current_player[game_index] + direction[game_index]
            ) % num_players
            event_deltas[2] += 1
        elif card_rank == 1:
            direction[game_index] *= -1
            event_deltas[3] += 1
        elif card_rank == 0:
            penalty_amount[game_index] += 5
        elif card_rank == 13:
            phase[game_index] = 2
            continue
        elif card_rank == 2:
            penalty_amount[game_index] += 2

        turn_counts[game_index] += 1
        acting_player = int(turn_player[game_index])
        if hand_sizes[game_index, acting_player] == 0:
            winners[game_index] = current_player[game_index]
            game_over[game_index] = True
        if not game_over[game_index]:
            current_player[game_index] = (
                current_player[game_index] + direction[game_index]
            ) % num_players
        phase[game_index] = 0
        drawn_card[game_index] = -1
        if not game_over[game_index]:
            next_player = int(current_player[game_index])
            turn_player[game_index] = next_player
            turn_started_with_one_card[game_index] = (
                hand_sizes[game_index, next_player] == 1
            )


if njit is not None:
    rl_has_card_decisions_native = njit(
        cache=True,
        nogil=True,
    )(_rl_has_card_decisions_impl)
    observations_into_native = njit(
        cache=True,
        nogil=True,
    )(_observations_into_impl)
    apply_simple_card_actions_native = njit(
        cache=True,
        nogil=True,
    )(_apply_simple_card_actions_impl)
else:
    rl_has_card_decisions_native = None
    observations_into_native = None
    apply_simple_card_actions_native = None


def warm_native_rollout_engine() -> bool:
    """Compile/load the kernel before iteration timing begins."""
    if (
        apply_simple_card_actions_native is None
        or rl_has_card_decisions_native is None
        or observations_into_native is None
    ):
        return False
    empty = np.empty(0, dtype=np.int64)
    rl_has_card_decisions_native(
        empty,
        np.empty(1, dtype=np.int8),
        np.empty((1, 2, 50), dtype=np.uint8),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int16),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int8),
    )
    observations_into_native(
        empty,
        np.empty(0, dtype=np.bool_),
        np.empty(1, dtype=np.int8),
        np.empty((1, 2, 50), dtype=np.uint8),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int16),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int8),
        np.empty((1, 5), dtype=np.int32),
        np.empty(1, dtype=np.int8),
        np.empty((1, 2), dtype=np.uint8),
        2,
        np.empty((0, 121), dtype=np.float32),
        np.empty((0, 55), dtype=np.bool_),
    )
    apply_simple_card_actions_native(
        empty,
        empty,
        2,
        np.empty(50, dtype=np.int8),
        np.empty(50, dtype=np.int8),
        np.empty(50, dtype=np.bool_),
        np.empty((5, 14, 50), dtype=np.bool_),
        np.empty((1, 2, 54), dtype=np.int8),
        np.empty((1, 2, 50), dtype=np.uint8),
        np.empty((1, 2), dtype=np.uint8),
        np.empty((1, 54), dtype=np.int8),
        np.empty(1, dtype=np.uint8),
        np.empty((1, 54), dtype=np.int8),
        np.empty(1, dtype=np.uint8),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int16),
        np.empty((1, 5), dtype=np.int32),
        np.empty(1, dtype=np.int32),
        np.empty(1, dtype=np.int32),
        np.empty(1, dtype=np.bool_),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.int8),
        np.empty(1, dtype=np.bool_),
        np.empty(1, dtype=np.int8),
        np.zeros(8, dtype=np.int64),
        np.empty(0, dtype=np.bool_),
        np.empty(0, dtype=np.int8),
        np.empty(0, dtype=np.int8),
    )
    return True
