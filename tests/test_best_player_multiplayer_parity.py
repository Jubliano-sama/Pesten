"""Development-only frozen-player parity across supported player counts."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from ai_observation import (
    COMPATIBLE_END,
    COMPATIBLE_START,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PASS_ACTION,
)
from batched_pesten import (
    BatchedPesten,
    PHASE_CARD,
    PHASE_DRAW_RETRY,
    PHASE_SUIT,
    POLICY_FIRST_LEGAL,
    POLICY_RL,
)
from benchmark_checkpoint_batched import (
    BatchGameSpec,
    run_batched_specs,
)
from best_player import BestPlayerConfig, load_best_player
import card
import deck
import game
from ppo2 import build_action_masks


FULL_GAME_BASE_SEED = 88_607_031
FUZZ_BASE_SEED = 88_617_031
SUPPORTED_PLAYER_COUNTS = tuple(range(2, 8))
PHASES = ("ordinary", "suit_change", "penalty", "draw_retry")
DRAWN_CARD = 16
FIXED_CURRENT_HAND = (4, DRAWN_CARD, 28, 40)


class PassiveAgent:
    def __init__(self):
        self.mydeck = deck.Deck([])

    def playCard(self, _current_sort, _current_true_number):
        return None

    def changeSort(self):
        return "harten"

    def addCard(self, drawn_card):
        self.mydeck.cards.append(drawn_card)

    def remove(self, played_card):
        self.mydeck.cards.remove(played_card)


@pytest.fixture(scope="module")
def loaded_best():
    return load_best_player(BestPlayerConfig(device="cpu"))


def _install_engine_hand(engine, player_index, numbers):
    engine.hands[0, player_index].fill(-1)
    engine.hand_counts[0, player_index].fill(0)
    engine.hand_sizes[0, player_index] = len(numbers)
    engine.hands[0, player_index, : len(numbers)] = numbers
    for number in numbers:
        engine.hand_counts[0, player_index, number] += 1


def _state_hands(player_count, current_player, seed):
    rng = np.random.RandomState(seed)
    available = np.asarray(
        [
            number
            for number in range(50)
            if number not in FIXED_CURRENT_HAND
        ],
        dtype=np.int8,
    )
    rng.shuffle(available)
    hands = []
    cursor = 0
    for player_index in range(player_count):
        if player_index == current_player:
            hands.append(FIXED_CURRENT_HAND)
            continue
        hand_size = 2 + int(rng.randint(0, 3))
        hands.append(
            tuple(
                int(number)
                for number in available[cursor : cursor + hand_size]
            )
        )
        cursor += hand_size
    return hands


def _phase_state(loaded, player_count, phase, case_index):
    seed = FUZZ_BASE_SEED + 100 * player_count + case_index
    assert seed < 92_000_000
    rng = np.random.RandomState(seed)
    current_player = case_index
    direction = -1 if case_index % 2 else 1
    hands = _state_hands(player_count, current_player, seed)
    sorts_played = rng.randint(0, 20, size=5, dtype=np.int32)

    factories = [PassiveAgent] * player_count
    factories[current_player] = loaded.canonical_game_factory()
    canonical = game.Game(factories)
    for player, numbers in zip(canonical.players, hands):
        player.mydeck.cards[:] = [card.Card(number) for number in numbers]
    canonical.current_player_index = current_player
    canonical.current_sort = "harten"
    canonical.current_true_number = 5
    canonical.direction = direction
    canonical.penalty_amount = 2 if phase == "penalty" else 0
    canonical.sorts_played[:] = sorts_played
    canonical.turn_count = case_index + 10
    canonical.game_over = False
    agent = canonical.players[current_player]

    kinds = np.full(
        (1, player_count),
        POLICY_FIRST_LEGAL,
        dtype=np.int8,
    )
    kinds[0, current_player] = POLICY_RL
    keys = np.full((1, player_count), None, dtype=object)
    keys[0, current_player] = "best"
    engine = BatchedPesten(
        kinds,
        keys,
        np.asarray([seed], dtype=np.uint32),
        max_turns=100,
    )
    for player_index, numbers in enumerate(hands):
        _install_engine_hand(engine, player_index, numbers)
    engine.current_player[0] = current_player
    engine.turn_player[0] = current_player
    engine.current_sort[0] = 0
    engine.current_rank[0] = 5
    engine.direction[0] = direction
    engine.penalty_amount[0] = 2 if phase == "penalty" else 0
    engine.sorts_played[0] = sorts_played
    engine.turn_counts[0] = case_index + 10
    engine.game_over[0] = False
    if phase == "suit_change":
        engine.phase[0] = PHASE_SUIT
    elif phase == "draw_retry":
        engine.phase[0] = PHASE_DRAW_RETRY
        engine.drawn_card[0] = DRAWN_CARD
    else:
        engine.phase[0] = PHASE_CARD
        engine.drawn_card[0] = -1

    canonical_observation = agent.obs()
    if phase == "suit_change":
        canonical_observation[MODE_INDEX] = 1
    batched_observation = torch.from_numpy(
        engine.observation(
            0,
            suit_mode=phase == "suit_change",
        )
    )
    if phase == "draw_retry":
        non_compatibility = torch.ones(
            OBSERVATION_SIZE,
            dtype=torch.bool,
        )
        non_compatibility[COMPATIBLE_START:COMPATIBLE_END] = False
        assert torch.equal(
            canonical_observation[non_compatibility],
            batched_observation[non_compatibility],
        )
        # Canonical retains legacy compatibility features and corrects the
        # retry mask; BatchedPesten exposes only the executable drawn card.
        assert int(
            canonical_observation[COMPATIBLE_START:COMPATIBLE_END].sum()
        ) == len(FIXED_CURRENT_HAND)
        assert int(
            batched_observation[COMPATIBLE_START:COMPATIBLE_END].sum()
        ) == 1
        agent._draw_retry_card_number = DRAWN_CARD
        agent._draw_retry_turn = canonical.turn_count
        agent._in_draw_retry = True
    else:
        assert torch.equal(canonical_observation, batched_observation)

    action, _ = agent.act_rl(canonical_observation)
    canonical_action = PASS_ACTION if action is None else int(action)
    agent._clear_draw_tracking()
    return batched_observation, canonical_action, {
        "players": player_count,
        "phase": phase,
        "seed": seed,
        "current_player": current_player,
        "direction": direction,
    }


def test_fuzzed_phase_actions_match_grouped_batched_inference(loaded_best):
    observations = []
    expected_actions = []
    cases = []
    for player_count in SUPPORTED_PLAYER_COUNTS:
        for case_index in range(player_count):
            for phase in PHASES:
                observation, action, case = _phase_state(
                    loaded_best,
                    player_count,
                    phase,
                    case_index,
                )
                observations.append(observation)
                expected_actions.append(action)
                cases.append(case)

    batch = torch.stack(observations)
    with torch.inference_mode():
        logits = loaded_best.actor(batch)
        masks = build_action_masks(batch)
        actual_actions = logits.masked_fill(
            ~masks,
            torch.finfo(logits.dtype).min,
        ).argmax(-1)

    assert len(cases) == 108
    for case, expected, actual in zip(
        cases,
        expected_actions,
        actual_actions.tolist(),
    ):
        assert actual == expected, case
    assert {case["players"] for case in cases} == set(
        SUPPORTED_PLAYER_COUNTS
    )
    assert {case["phase"] for case in cases} == set(PHASES)
    assert {case["direction"] for case in cases} == {-1, 1}
    for player_count in SUPPORTED_PLAYER_COUNTS:
        assert {
            case["current_player"]
            for case in cases
            if case["players"] == player_count
        } == set(range(player_count))


@pytest.mark.parametrize("player_count", SUPPORTED_PLAYER_COUNTS)
def test_full_games_match_exact_winner_and_turns(loaded_best, player_count):
    seed = FULL_GAME_BASE_SEED + player_count
    assert seed < 92_000_000
    factory = loaded_best.canonical_game_factory()
    random.seed(seed)
    np.random.seed(seed)
    canonical = game.Game([factory] * player_count)
    expected_turns, expected_winner = canonical.auto_simulate(
        max_turns=1_000
    )

    spec = BatchGameSpec(
        section="development_multiplayer_parity",
        opponent=None,
        pair_index=player_count,
        evaluated_key="best",
        evaluated_seat=0,
        deal_seed=seed,
        policy_kinds=(POLICY_RL,) * player_count,
        actor_keys=("best",) * player_count,
    )
    outcomes, _statistics = run_batched_specs(
        [spec],
        {"best": loaded_best.batched_actor_policy()},
        max_turns=1_000,
    )

    assert outcomes[0].winner == expected_winner
    assert outcomes[0].turns == expected_turns
