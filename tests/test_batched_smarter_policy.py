import time

import numpy as np

import card
import deck
import game
from ai_observation import PASS_ACTION
from batched_pesten import (
    BatchedPesten,
    POLICY_SMARTER,
)
from ppo2 import PPO_Supervised, non_rl_policy_kind, parse_agent_sets
import smartAgent
import torch


class PassiveAgent:
    def __init__(self, numbers=()):
        self.mydeck = deck.Deck(
            [card.Card(int(number)) for number in numbers]
        )

    def addCard(self, held):
        self.mydeck.cards.append(held)

    def remove(self, held):
        self.mydeck.cards.remove(held)

    def playCard(self, current_sort, current_true_number):
        return None

    def changeSort(self):
        return "harten"


def install_hand(engine, game_index, player_index, numbers):
    numbers = np.asarray(numbers, dtype=np.int8)
    engine.hands[game_index, player_index].fill(-1)
    engine.hands[
        game_index, player_index, : len(numbers)
    ] = numbers
    engine.hand_sizes[game_index, player_index] = len(numbers)
    engine.hand_counts[game_index, player_index].fill(0)
    for number in numbers:
        engine.hand_counts[game_index, player_index, number] += 1


def canonical_action(
    hand,
    opponent_hands,
    *,
    current_sort,
    current_rank,
    current_player,
    direction,
    sorts_played,
    penalty_amount=0,
):
    instance = game.Game([])
    smarter = smartAgent.SmarterAgent(instance)
    players = [
        PassiveAgent(numbers) for numbers in opponent_hands
    ]
    players.insert(current_player, smarter)
    smarter.mydeck.cards = [card.Card(int(number)) for number in hand]
    instance.players = players
    instance.num_players = len(players)
    instance.current_player_index = current_player
    instance.direction = direction
    instance.current_sort = card.sorts[current_sort]
    instance.current_true_number = current_rank
    instance.penalty_amount = penalty_amount
    instance.sorts_played = np.asarray(sorts_played, dtype=int)
    selected = smarter.playCard(
        instance.current_sort, instance.current_true_number
    )
    return (
        PASS_ACTION if selected is None else selected.number,
        card.sorts.index(smarter.changeSort()),
    )


def make_live_smarter(hand, opponent_hand, *, penalty_amount=0):
    instance = game.Game([])
    smarter = smartAgent.SmarterAgent(instance)
    opponent = PassiveAgent(opponent_hand)
    smarter.mydeck.cards = [card.Card(int(number)) for number in hand]
    instance.players = [smarter, opponent]
    instance.num_players = 2
    instance.current_player_index = 0
    instance.direction = 1
    instance.current_sort = "harten"
    instance.current_true_number = 3
    instance.penalty_amount = penalty_amount
    instance.sorts_played = np.zeros(5, dtype=int)
    return instance, smarter, opponent


def test_smarter_removes_only_one_physical_duplicate_when_scoring():
    smarter = smartAgent.SmarterAgent()
    smarter.mydeck.cards = [card.Card(49), card.Card(49), card.Card(2)]

    remaining = smarter._remaining_after(smarter.mydeck.cards[0])

    assert [held.number for held in remaining] == [49, 2]


def test_smarter_stacks_only_the_active_penalty_rank():
    instance, smarter, _ = make_live_smarter(
        (49, 48, 15),
        (3, 4, 5, 6),
        penalty_amount=5,
    )
    instance.current_true_number = 0

    selected = smarter.playCard(instance.current_sort, instance.current_true_number)

    assert selected is not None
    assert selected.truenumber == 0

    engine = BatchedPesten(
        np.full((1, 2), POLICY_SMARTER, dtype=np.int8),
        np.full((1, 2), None, dtype=object),
        np.asarray([31_002], dtype=np.uint32),
    )
    install_hand(engine, 0, 0, (49, 48, 15))
    install_hand(engine, 0, 1, (3, 4, 5, 6))
    engine.current_player[0] = 0
    engine.current_sort[0] = 0
    engine.current_rank[0] = 0
    engine.penalty_amount[0] = 5
    assert engine.smarter_action(0) == 49


def test_smarter_uses_its_remaining_hand_for_jack_suit():
    instance, smarter, _ = make_live_smarter(
        (12, 13, 25),
        (3, 4, 5),
    )
    instance.sorts_played = np.asarray((100, 0, 0, 0, 0))

    assert smarter.changeSort() == "ruiten"


def test_smarter_endgame_rules_preserve_an_ordinary_finisher():
    instance, smarter, opponent = make_live_smarter(
        (48, 2),
        (3, 4),
    )

    selected = smarter.playCard(instance.current_sort, instance.current_true_number)

    assert selected is not None
    assert selected.truenumber == 13

    smarter.mydeck.cards = [card.Card(49)]
    assert smarter.playCard(instance.current_sort, instance.current_true_number) is None

    smarter.mydeck.cards = [card.Card(0), card.Card(1)]
    opponent.mydeck.cards = [card.Card(number) for number in (3, 4, 5)]
    assert smarter.playCard(instance.current_sort, instance.current_true_number) is None


def test_smarter_saves_wild_cards_until_a_rival_is_close():
    instance, smarter, opponent = make_live_smarter(
        (48, 49, 15, 16, 17),
        (3, 4, 5),
    )

    assert smarter.playCard(instance.current_sort, instance.current_true_number) is None

    opponent.mydeck.cards = [card.Card(3), card.Card(4)]
    selected = smarter.playCard(instance.current_sort, instance.current_true_number)
    assert selected is not None
    assert selected.truenumber == 0


def test_smarter_uses_distinct_two_player_last_card_pressure():
    instance, smarter, opponent = make_live_smarter(
        (49, 2, 14),
        (3, 4),
    )
    instance.current_true_number = 4

    selected = smarter.playCard(instance.current_sort, instance.current_true_number)
    assert selected is not None
    assert selected.truenumber == 3

    opponent.mydeck.cards = [card.Card(3)]
    selected = smarter.playCard(instance.current_sort, instance.current_true_number)
    assert selected is not None
    assert selected.truenumber == 0


def test_smarter_action_matches_live_canonical_agent_over_broad_public_states():
    rng = np.random.RandomState(23_731)
    for players in (2, 4):
        engine = BatchedPesten(
            np.full((1, players), POLICY_SMARTER, dtype=np.int8),
            np.full((1, players), None, dtype=object),
            np.asarray([31_001 + players], dtype=np.uint32),
        )
        for case_index in range(1_000):
            current_player = int(rng.randint(0, players))
            direction = int(rng.choice((-1, 1)))
            hand = rng.choice(
                50, int(rng.randint(1, 28)), replace=True
            ).astype(np.int8)
            opponent_hands = [
                rng.choice(
                    50, int(rng.randint(1, 18)), replace=True
                ).astype(np.int8)
                for _ in range(players - 1)
            ]
            install_hand(engine, 0, current_player, hand)
            opponent_index = 0
            for player_index in range(players):
                if player_index == current_player:
                    continue
                install_hand(
                    engine,
                    0,
                    player_index,
                    opponent_hands[opponent_index],
                )
                opponent_index += 1
            current_sort = int(rng.randint(0, 5))
            current_rank = int(rng.choice(range(14)))
            sorts_played = rng.randint(0, 40, size=5)
            penalty_amount = int(rng.choice((0, 0, 0, 2, 5)))
            engine.current_player[0] = current_player
            engine.direction[0] = direction
            engine.current_sort[0] = current_sort
            engine.current_rank[0] = current_rank
            engine.penalty_amount[0] = penalty_amount
            engine.sorts_played[0] = sorts_played

            expected_action, expected_suit = canonical_action(
                hand,
                opponent_hands,
                current_sort=current_sort,
                current_rank=current_rank,
                current_player=current_player,
                direction=direction,
                sorts_played=sorts_played,
                penalty_amount=penalty_amount,
            )
            assert engine.smarter_action(0) == expected_action, (
                players,
                case_index,
            )
            assert (
                engine.choose_suit(0, POLICY_SMARTER) == expected_suit
            ), (players, case_index)


def assert_smarter_full_game_transition_parity(players, seeds):
    engine = BatchedPesten(
        np.full((len(seeds), players), POLICY_SMARTER, dtype=np.int8),
        np.full((len(seeds), players), None, dtype=object),
        seeds,
        max_turns=500,
    )
    engine.run_non_neural()

    for offset, seed in enumerate(seeds):
        np.random.seed(int(seed))
        canonical = game.Game(
            [smartAgent.SmarterAgent] * players
        )
        canonical.auto_simulate(max_turns=500)
        assert int(engine.turn_counts[offset]) == canonical.turn_count
        assert int(engine.winners[offset]) == (
            -1 if canonical.winner is None else canonical.winner
        )
        assert int(engine.current_player[offset]) == (
            canonical.current_player_index
        )
        assert int(engine.direction[offset]) == canonical.direction
        assert int(engine.penalty_amount[offset]) == canonical.penalty_amount
        assert int(engine.current_rank[offset]) == canonical.current_true_number
        assert int(engine.current_sort[offset]) == card.sorts.index(
            canonical.current_sort
        )
        for player_index, player in enumerate(canonical.players):
            actual = engine.hands[
                offset,
                player_index,
                : engine.hand_sizes[offset, player_index],
            ].astype(int).tolist()
            assert actual == [
                held.number for held in player.mydeck.cards
            ]
        assert engine.draw_piles[
            offset, : engine.draw_counts[offset]
        ].astype(int).tolist() == [
            held.number for held in canonical.grab_deck.cards
        ]
        assert engine.discards[
            offset, : engine.discard_counts[offset]
        ].astype(int).tolist() == [
            held.number for held in canonical.game_deck.cards
        ]


def test_smarter_full_game_transition_parity_across_seeded_deals():
    seeds = np.arange(40_000, 40_192, dtype=np.uint32)
    assert_smarter_full_game_transition_parity(2, seeds)


def test_smarter_full_game_transition_parity_across_table_sizes():
    for players in range(3, 9):
        start = 41_000 + players * 100
        seeds = np.arange(start, start + 8, dtype=np.uint32)
        assert_smarter_full_game_transition_parity(players, seeds)


def test_smarter_uses_only_own_hand_and_public_state():
    kinds = np.full((1, 3), POLICY_SMARTER, dtype=np.int8)
    keys = np.full((1, 3), None, dtype=object)
    left = BatchedPesten(kinds, keys, np.asarray([51_001], dtype=np.uint32))
    right = BatchedPesten(kinds, keys, np.asarray([51_001], dtype=np.uint32))
    own = (3, 6, 7, 16, 48)
    install_hand(left, 0, 0, own)
    install_hand(right, 0, 0, own)
    for player_index in (1, 2):
        size = int(left.hand_sizes[0, player_index])
        replacement = (
            np.arange(size, dtype=np.int8) + 20 + 10 * player_index
        ) % 50
        install_hand(right, 0, player_index, replacement)
    right.draw_piles[0, : right.draw_counts[0]] = right.draw_piles[
        0, : right.draw_counts[0]
    ][::-1]
    for engine in (left, right):
        engine.current_player[0] = 0
        engine.direction[0] = -1
        engine.current_sort[0] = 0
        engine.current_rank[0] = 4
        engine.sorts_played[0] = (5, 9, 2, 9, 0)

    assert left.smarter_action(0) == right.smarter_action(0)
    assert left.choose_suit(0, POLICY_SMARTER) == right.choose_suit(
        0, POLICY_SMARTER
    )


def test_smarter_is_wired_into_agent_set_parsing_and_policy_generation():
    assert parse_agent_sets("smarter,rl;random,self") == [
        ["smarter", "rl"],
        ["random", "self"],
    ]
    assert non_rl_policy_kind("SMARTER") == POLICY_SMARTER


def test_smarter_rl_agent_set_generates_batched_ppo_data():
    np.random.seed(271_828)
    torch.manual_seed(271_828)
    trainer = PPO_Supervised(
        use_pretrained=False,
        old_save=(),
        rollout_device="cpu",
        training_actor_device="cpu",
        rollout_batch_size=4,
        compile_updates=False,
    )
    observations, actions, log_probs, rewards, lengths = (
        trainer.generate_data(
            num_episodes=4,
            agent_sets=[["smarter", "rl"]],
        )
    )
    assert len(lengths) == 4
    assert len(actions) > 0
    assert observations.shape == (len(actions), 121)
    assert log_probs.shape == rewards.shape == actions.shape
    assert trainer.last_rollout_metrics["completed_games"] == 4


def test_batched_smarter_throughput_smoke():
    games = 512
    engine = BatchedPesten(
        np.full((games, 2), POLICY_SMARTER, dtype=np.int8),
        np.full((games, 2), None, dtype=object),
        np.arange(60_000, 60_000 + games, dtype=np.uint32),
        max_turns=500,
    )
    started = time.perf_counter()
    engine.run_non_neural()
    elapsed = time.perf_counter() - started
    assert np.all(engine.game_over)
    assert games / elapsed > 100
