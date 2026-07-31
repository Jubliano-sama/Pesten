"""Active hidden-information tripwires for the canonical best player."""

from __future__ import annotations

import numpy as np
import pytest

from best_player import BestPlayerConfig, load_best_player
import card
import deck
import game


DEVELOPMENT_SEED = 89_707_031
SUPPORTED_PLAYER_COUNTS = tuple(range(2, 8))
PHASES = ("ordinary", "penalty", "suit_change", "draw_retry")
OWN_CARD_NUMBERS = (4, 16, 28, 40)
DRAWN_CARD = 16
PUBLIC_GAME_FIELDS = {
    "current_sort",
    "current_true_number",
    "direction",
    "game_over",
    "penalty_amount",
    "players",
    "sorts_played",
    "turn_count",
}


class PublicCountOnlyDeck:
    """Expose public hand size while trapping private identity access."""

    def __init__(self, hidden_card_numbers):
        self._hidden_card_numbers = tuple(hidden_card_numbers)
        self.count_reads = 0
        self.forbidden_reads = 0

    def cardCount(self):
        self.count_reads += 1
        return len(self._hidden_card_numbers)

    @property
    def cards(self):
        self.forbidden_reads += 1
        raise AssertionError("opponent card identities were accessed")


class HiddenOpponent:
    def __init__(self, hidden_card_numbers):
        self.mydeck = PublicCountOnlyDeck(hidden_card_numbers)


class ForbiddenHiddenPile:
    """Raise if any draw/discard identity or ordering operation is attempted."""

    def __init__(self, label, hidden_card_numbers):
        self._label = label
        self._hidden_card_numbers = tuple(hidden_card_numbers)
        self.access_count = 0

    def __getattribute__(self, name):
        allowed = {
            "_hidden_card_numbers",
            "_label",
            "access_count",
            "assert_untouched",
        }
        if name.startswith("__") or name in allowed:
            return object.__getattribute__(self, name)
        count = object.__getattribute__(self, "access_count")
        object.__setattr__(self, "access_count", count + 1)
        label = object.__getattribute__(self, "_label")
        raise AssertionError(f"{label} was accessed through attribute {name}")

    def assert_untouched(self):
        assert self.access_count == 0


class AuditedGame(game.Game):
    """Record every game-state field read during one policy decision."""

    def __init__(self):
        object.__setattr__(self, "_audit_reads", None)
        super().__init__([])
        object.__setattr__(self, "_audit_reads", [])

    def __getattribute__(self, name):
        if not name.startswith("_") and name not in {"audit_reads", "clear_audit"}:
            reads = object.__getattribute__(self, "_audit_reads")
            if reads is not None:
                reads.append(name)
        return super().__getattribute__(name)

    def clear_audit(self):
        self._audit_reads.clear()

    @property
    def audit_reads(self):
        return tuple(self._audit_reads)


@pytest.fixture(scope="module")
def loaded_best():
    return load_best_player(BestPlayerConfig(device="cpu"))


def _hidden_cards(rng, count, world_index, opponent_index):
    offset = 17 * world_index + 7 * opponent_index
    return tuple(
        int((offset + value) % 50)
        for value in rng.choice(50, size=count, replace=False)
    )


def _configured_world(
    loaded,
    *,
    player_count,
    best_seat,
    phase,
    world_index,
):
    seed = (
        DEVELOPMENT_SEED
        + 1_000 * player_count
        + 10 * best_seat
        + PHASES.index(phase)
    )
    assert seed < 92_000_000
    rng = np.random.RandomState(seed)
    match = AuditedGame()
    agent = loaded.canonical_game_factory()(match)
    agent.mydeck = deck.Deck(
        [card.Card(number) for number in OWN_CARD_NUMBERS]
    )
    public_sizes = [
        2 + int(rng.randint(0, 5))
        for _ in range(player_count)
    ]
    public_sizes[best_seat] = len(OWN_CARD_NUMBERS)
    opponents = []
    players = []
    for player_index, count in enumerate(public_sizes):
        if player_index == best_seat:
            players.append(agent)
            continue
        opponent = HiddenOpponent(
            _hidden_cards(
                rng,
                count,
                world_index,
                player_index,
            )
        )
        opponents.append(opponent)
        players.append(opponent)

    draw_pile = ForbiddenHiddenPile(
        "draw-pile identities",
        tuple(
            int((world_index * 23 + index) % 50)
            for index in range(30)
        ),
    )
    discard_pile = ForbiddenHiddenPile(
        "discard-pile identities",
        tuple(
            int((world_index * 31 + index) % 50)
            for index in range(8)
        ),
    )
    match.players = players
    match.num_players = player_count
    match.current_player_index = best_seat
    match.current_sort = "harten"
    match.current_true_number = 5
    match.direction = -1 if best_seat % 2 else 1
    match.penalty_amount = 2 if phase == "penalty" else 0
    match.sorts_played[:] = rng.randint(0, 20, size=5)
    match.turn_count = 100 + best_seat
    match.game_over = False
    match.grab_deck = draw_pile
    match.game_deck = discard_pile
    if phase == "draw_retry":
        agent._draw_retry_card_number = DRAWN_CARD
        agent._draw_retry_turn = match.turn_count

    match.clear_audit()
    for opponent in opponents:
        opponent.mydeck.count_reads = 0
        opponent.mydeck.forbidden_reads = 0
    return match, agent, opponents, draw_pile, discard_pile


def _decide(agent, phase):
    if phase == "suit_change":
        return ("suit", agent.changeSort())
    selected = agent.playCard("harten", 5)
    return (
        "card",
        None if selected is None else int(selected.number),
    )


def _assert_legal(decision, phase):
    kind, value = decision
    if phase == "suit_change":
        assert kind == "suit"
        assert value in card.sorts[:4]
        return
    assert kind == "card"
    if value is None:
        return
    assert value in OWN_CARD_NUMBERS
    held = card.Card(value)
    if phase == "penalty":
        assert held.truenumber == 5
    elif phase == "draw_retry":
        assert value == DRAWN_CARD
    else:
        assert held.compatible("harten", 5)


def test_canonical_best_never_reads_hidden_objects(loaded_best):
    cases = []
    for player_count in SUPPORTED_PLAYER_COUNTS:
        for best_seat in range(player_count):
            for phase in PHASES:
                paired_decisions = []
                for world_index in (0, 1):
                    (
                        match,
                        agent,
                        opponents,
                        draw_pile,
                        discard_pile,
                    ) = _configured_world(
                        loaded_best,
                        player_count=player_count,
                        best_seat=best_seat,
                        phase=phase,
                        world_index=world_index,
                    )
                    decision = _decide(agent, phase)
                    _assert_legal(decision, phase)
                    assert set(match.audit_reads) <= PUBLIC_GAME_FIELDS
                    assert "players" in match.audit_reads
                    assert "sorts_played" in match.audit_reads
                    for opponent in opponents:
                        assert opponent.mydeck.count_reads == 1
                        assert opponent.mydeck.forbidden_reads == 0
                    draw_pile.assert_untouched()
                    discard_pile.assert_untouched()
                    paired_decisions.append(decision)
                assert paired_decisions[0] == paired_decisions[1]
                cases.append((player_count, best_seat, phase))

    assert len(cases) == 108
    assert {case[0] for case in cases} == set(SUPPORTED_PLAYER_COUNTS)
    assert {case[2] for case in cases} == set(PHASES)
