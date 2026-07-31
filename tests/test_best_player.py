from pathlib import Path

import numpy as np
import pytest
import torch

from ai_observation import PASS_ACTION
from best_player import (
    CHAMPION_SHA256,
    DEFAULT_CHAMPION_PATH,
    INFORMATION_BOUNDARY,
    OPT_IN_CPU_BASE_CHUNK_SIZE,
    BestPlayerConfig,
    load_batched_actor,
    load_best_player,
    load_canonical_game_factory,
    verify_champion,
)
from batched_pesten import (
    BatchedPesten,
    PHASE_CARD,
    PHASE_DRAW_RETRY,
    POLICY_RL,
)
from batched_suit_symmetry import BatchedSuitAveragedActor
import card
import deck
import game
from ppo2 import build_action_masks


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


def configured_game(factory):
    instance = game.Game([factory, PassiveAgent])
    agent = instance.players[0]
    agent.mydeck.cards[:] = [
        card.Card(3),
        card.Card(15),
        card.Card(27),
        card.Card(39),
    ]
    instance.players[1].mydeck.cards[:] = [card.Card(1), card.Card(2)]
    instance.current_sort = "harten"
    instance.current_true_number = 4
    instance.current_player_index = 0
    instance.turn_count = 7
    instance.direction = 1
    instance.penalty_amount = 0
    instance.sorts_played[:] = [1, 2, 3, 4, 0]
    instance.game_over = False
    instance.game_deck = deck.Deck([card.Card(3)])
    instance.grab_deck = deck.Deck([card.Card(20), card.Card(21)])
    return instance, agent


def test_defaults_pin_exact_current_champion_and_suit_only_projection():
    config = BestPlayerConfig()

    assert config.model_path == DEFAULT_CHAMPION_PATH
    assert config.alpha == 1.0
    assert config.aggregation == "logits"
    assert not config.player_orbit_average
    assert config.base_chunk_size == 512
    assert config.trace_cpu_actor
    assert OPT_IN_CPU_BASE_CHUNK_SIZE == 64
    assert verify_champion(config.model_path) == CHAMPION_SHA256


def test_champion_verification_fails_closed_on_wrong_artifact(tmp_path):
    wrong = tmp_path / "not-the-champion.pth"
    wrong.write_bytes(b"not a model")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verify_champion(wrong)
    with pytest.raises(FileNotFoundError, match="not found"):
        verify_champion(tmp_path / "missing.pth")


def test_loaded_player_exposes_canonical_and_batched_exact_same_actor():
    loaded = load_best_player(BestPlayerConfig(device="cpu"))
    factory = loaded.canonical_game_factory()
    policy = loaded.batched_actor_policy()
    _instance, agent = configured_game(factory)
    observation = agent.obs()
    with torch.inference_mode():
        logits = loaded.actor(observation.unsqueeze(0)).squeeze(0)
    expected_action = int(
        torch.argmax(
            logits.masked_fill(
                ~build_action_masks(observation),
                torch.finfo(logits.dtype).min,
            )
        )
    )
    selected = agent.playCard("harten", 4)
    actual_action = PASS_ACTION if selected is None else selected.number

    assert isinstance(loaded.actor, BatchedSuitAveragedActor)
    assert agent.actor is loaded.actor
    assert policy.actor is loaded.actor
    assert policy.greedy
    assert actual_action == expected_action
    assert factory.best_player_metadata is loaded.metadata


def test_loaded_player_canonical_and_batched_games_match_through_draw_retry():
    loaded = load_best_player(BestPlayerConfig(device="cpu"))
    factory = loaded.canonical_game_factory()
    policy = loaded.batched_actor_policy()
    canonical = game.Game([factory, PassiveAgent])
    canonical_agent = canonical.players[0]
    canonical_agent.mydeck.cards[:] = [card.Card(30), card.Card(31)]
    canonical.players[1].mydeck.cards[:] = [card.Card(25)]
    canonical.current_player_index = 0
    canonical.turn_count = 0
    canonical.cards_played = 0
    canonical.current_sort = "harten"
    canonical.current_true_number = 5
    canonical.penalty_amount = 0
    canonical.sorts_played[:] = 0
    canonical.game_over = False
    canonical.winner = None
    canonical.game_deck = deck.Deck([card.Card(4)])
    canonical.grab_deck = deck.Deck([card.Card(16)])

    batched = BatchedPesten(
        np.full((1, 2), POLICY_RL, dtype=np.int8),
        np.full((1, 2), "candidate", dtype=object),
        np.asarray([8_812_337], dtype=np.uint32),
    )
    batched.hands[0].fill(-1)
    batched.hand_counts[0].fill(0)
    batched.hand_sizes[0].fill(0)
    for player_index, numbers in enumerate(((30, 31), (25,))):
        batched.hands[0, player_index, : len(numbers)] = numbers
        batched.hand_sizes[0, player_index] = len(numbers)
        for number in numbers:
            batched.hand_counts[0, player_index, number] += 1
    batched.current_player[0] = 0
    batched.turn_player[0] = 0
    batched.turn_counts[0] = 0
    batched.cards_played[0] = 0
    batched.current_sort[0] = 0
    batched.current_rank[0] = 5
    batched.penalty_amount[0] = 0
    batched.sorts_played[0] = 0
    batched.game_over[0] = False
    batched.winners[0] = -1
    batched.phase[0] = PHASE_CARD
    batched.draw_piles[0].fill(-1)
    batched.draw_piles[0, 0] = 16
    batched.draw_counts[0] = 1
    batched.discards[0].fill(-1)
    batched.discards[0, 0] = 4
    batched.discard_counts[0] = 1

    canonical.simulate_turn()
    batched.apply_action(0, PASS_ACTION)
    assert batched.phase[0] == PHASE_DRAW_RETRY
    observation = torch.from_numpy(batched.observation(0))
    with torch.inference_mode():
        logits = policy.actor(observation.unsqueeze(0)).squeeze(0)
    action = int(
        torch.argmax(
            logits.masked_fill(
                ~build_action_masks(observation),
                torch.finfo(logits.dtype).min,
            )
        )
    )
    batched.apply_action(0, action)

    assert int(batched.current_sort[0]) == card.sorts.index(
        canonical.current_sort
    )
    assert int(batched.current_rank[0]) == canonical.current_true_number
    assert int(batched.current_player[0]) == canonical.current_player_index
    assert int(batched.turn_counts[0]) == canonical.turn_count
    assert int(batched.cards_played[0]) == canonical.cards_played
    assert batched.hands[
        0, 0, : batched.hand_sizes[0, 0]
    ].astype(int).tolist() == [
        held.number for held in canonical_agent.mydeck.cards
    ]
    assert factory.stats.retry_decisions > 0
    assert factory.stats.outside_retry_action_mismatches == 0


def test_projection_configuration_and_information_boundary_are_recorded():
    config = BestPlayerConfig(
        device="cpu",
        alpha=0.4,
        aggregation="probabilities",
        player_orbit_average=True,
        base_chunk_size=32,
    )
    loaded = load_best_player(config)
    projection = loaded.metadata["projection"]

    assert loaded.actor.alpha == 0.4
    assert loaded.actor.aggregation == "probabilities"
    assert loaded.actor.player_orbit_average
    assert projection["alpha"] == 0.4
    assert projection["aggregation"] == "probabilities"
    assert projection["player_orbit_average"]
    assert projection["base_chunk_size"] == 32
    assert projection["base_actor_execution"] == "torchscript_trace"
    assert loaded.metadata["information_boundary"] == INFORMATION_BOUNDARY
    assert not INFORMATION_BOUNDARY["opponent_card_identities"]
    assert not INFORMATION_BOUNDARY["draw_pile_inspection"]


def test_cpu_trace_is_bit_exact_for_projected_logits_and_actions():
    eager = load_best_player(
        BestPlayerConfig(device="cpu", trace_cpu_actor=False)
    )
    traced = load_best_player(
        BestPlayerConfig(device="cpu", trace_cpu_actor=True)
    )
    generator = torch.Generator().manual_seed(91_733_041)
    for batch_size in (1, 7, 64):
        observations = torch.randn(
            batch_size,
            121,
            generator=generator,
        )
        observations[:, 50:100] = (
            observations[:, 50:100] > 0
        )
        observations[::2, 120] = 0
        observations[1::2, 120] = 1
        with torch.inference_mode():
            expected = eager.actor(observations)
            actual = traced.actor(observations)
        masks = build_action_masks(observations)

        assert torch.equal(actual, expected)
        assert torch.equal(
            actual.masked_fill(~masks, -torch.inf).argmax(-1),
            expected.masked_fill(~masks, -torch.inf).argmax(-1),
        )
    assert (
        eager.metadata["projection"]["base_actor_execution"] == "eager"
    )
    assert (
        traced.metadata["projection"]["base_actor_execution"]
        == "torchscript_trace"
    )


def test_convenience_loaders_return_factory_and_batched_actor(monkeypatch):
    loaded = load_best_player(BestPlayerConfig(device="cpu"))
    monkeypatch.setattr(
        "best_player.load_best_player",
        lambda _config=None: loaded,
    )

    factory = load_canonical_game_factory()
    actor, device = load_batched_actor()

    assert callable(factory)
    assert actor is loaded.actor
    assert device == loaded.device


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"alpha": -0.1}, "alpha"),
        ({"alpha": 1.1}, "alpha"),
        ({"aggregation": "median"}, "aggregation"),
        ({"base_chunk_size": 0}, "base_chunk_size"),
    ],
)
def test_invalid_production_configuration_fails_closed(arguments, message):
    with pytest.raises(ValueError, match=message):
        BestPlayerConfig(**arguments)
