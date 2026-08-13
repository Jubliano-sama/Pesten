"""Measure the Puffer GRU teacher's card choices in paired self-play deals.

The policy only receives the information-safe ``DecisionInput`` contract.  The
analysis observes those same decisions plus public table state, then combines
all ordinary suits into rank-level results as requested.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Iterable, Sequence


MODE_LABELS = {
    "natural": "Varied choices",
    "top": "Always picks its favorite",
}
PLAYER_LABELS = {0: "All table sizes", **{n: f"{n} players" for n in range(2, 9)}}
PLAYER_COUNTS = tuple(range(2, 9))
RANK_ORDER = tuple(range(1, 14)) + (0,)
EFFECT_RANKS = frozenset({0, 1, 2, 7, 8, 13})

INPUT_GROUPS = (
    ("Cards it holds", 0, 50),
    ("Suit on the table", 50, 55),
    ("Number on the table", 55, 69),
    ("Everyone's card counts", 69, 82),
    ("Suits played so far", 82, 87),
    ("Turn and rule details", 87, 94),
)

MEMORY_TESTS = (
    ("all", "Clear all stored history"),
    ("lower", "Clear the earlier memory stage"),
    ("upper", "Clear the later memory stage"),
)
STATE_SAMPLES_PER_STYLE_AND_TABLE = 80

MEMORY_STAGE_CHECK_DESCRIPTION = (
    "on the same turn, erase the earlier memory stage, later memory stage, "
    "or all stored history; replay the newest public updates and see whether "
    "the favorite legal move or overall choice mix changes"
)
CURRENT_TURN_CHECK_DESCRIPTION = (
    "on a fixed sample of turns, nudge each of the 94 current-turn numbers "
    "slightly and measure how much the favorite's lead over second place "
    "moves, adjusted for how much that number usually varies"
)
MEMORY_MAP_DESCRIPTION = (
    "place sampled 384-value memories on the two display directions that keep "
    "the biggest observed differences; the directions have no assigned meaning"
)
MODEL_SHAPE_ROWS = (
    {"stage": "Current-turn snapshot", "exact_shape": "players checked together x 94", "plain_meaning": "cards it holds plus visible table facts"},
    {"stage": "Public updates since its last turn", "exact_shape": "updates x players checked together x 5", "plain_meaning": "played cards, draw counts, and suit choices"},
    {"stage": "What the network reads each step", "exact_shape": "players checked together x 100", "plain_meaning": "94 snapshot places + 5 update places + 1 type marker"},
    {"stage": "Stored history", "exact_shape": "2 x players checked together x 192", "plain_meaning": "two stages with 192 remembered values each"},
    {"stage": "What comes out", "exact_shape": "players checked together x 55, plus x 1", "plain_meaning": "55 possible moves plus one estimate of the position"},
)

RANK_NAMES = {
    0: "Joker",
    1: "Ace",
    2: "2",
    3: "3",
    4: "4",
    5: "5",
    6: "6",
    7: "7",
    8: "8",
    9: "9",
    10: "10",
    11: "King",
    12: "Queen",
    13: "Jack",
}

RANK_EFFECTS = {
    0: "next player draws 5",
    1: "reverses direction",
    2: "next player draws 2",
    7: "take another turn",
    8: "skip the next player",
    13: "choose the next suit",
}


def rank_for_card(card_id: int) -> int:
    """Map one of the 50 card identities to its suit-free number."""

    if not 0 <= card_id < 50:
        raise ValueError(f"card id must be in [0, 49], got {card_id}")
    if card_id == 48:
        return 13
    if card_id == 49:
        return 0
    return card_id % 12 + 1


def rank_label(rank: int) -> str:
    name = RANK_NAMES[rank]
    effect = RANK_EFFECTS.get(rank)
    return f"{name} - {effect}" if effect else name


def hand_stage(hand_size: int) -> str:
    if hand_size <= 2:
        return "1-2 cards"
    if hand_size <= 5:
        return "3-5 cards"
    return "6+ cards"


def hand_size_band(hand_size: int) -> str:
    if hand_size == 1:
        return "1 card"
    if hand_size == 2:
        return "2 cards"
    if hand_size <= 5:
        return "3-5 cards"
    if hand_size <= 8:
        return "6-8 cards"
    return "9+ cards"


def choice_band(playable_types: int) -> str:
    return f"{playable_types}" if playable_types <= 3 else "4+"


def move_label(action: int) -> str:
    if 0 <= action < 50:
        return RANK_NAMES[rank_for_card(action)]
    if 50 <= action < 54:
        return "Choose a suit"
    if action == 54:
        return "Pass"
    return "Unknown move"


def progress_band(decision_number: int) -> str:
    if decision_number == 1:
        return "First decision"
    if decision_number <= 5:
        return "Decisions 2-5"
    if decision_number <= 10:
        return "Decisions 6-10"
    return "Decision 11+"


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Wilson 95% interval for a descriptive binary rate."""

    if total <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054
    rate = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (rate + z2 / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z2 / (4.0 * total * total))
        / denominator
    )
    return (max(0.0, center - margin), min(1.0, center + margin))


@dataclass(slots=True)
class OnlineCorrelation:
    """Small streaming Pearson correlation accumulator."""

    count: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def add(self, x: float, y: float) -> None:
        self.count += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x * x
        self.sum_yy += y * y
        self.sum_xy += x * y

    def value(self) -> float | None:
        if self.count < 2:
            return None
        numerator = self.count * self.sum_xy - self.sum_x * self.sum_y
        left = self.count * self.sum_xx - self.sum_x * self.sum_x
        right = self.count * self.sum_yy - self.sum_y * self.sum_y
        denominator = math.sqrt(max(0.0, left) * max(0.0, right))
        return None if denominator == 0.0 else numerator / denominator


@dataclass(slots=True)
class InputPullStats:
    """Streaming ingredients for a scale-aware local input check."""

    count: int = 0
    sums: list[float] = field(default_factory=lambda: [0.0] * 94)
    sums_squared: list[float] = field(default_factory=lambda: [0.0] * 94)
    absolute_slopes: list[float] = field(default_factory=lambda: [0.0] * 94)

    def add(self, observation: Sequence[float], slopes: Sequence[float]) -> None:
        if len(observation) != 94 or len(slopes) != 94:
            raise ValueError("input pull rows must contain exactly 94 values")
        self.count += 1
        for index, (value, slope) in enumerate(zip(observation, slopes)):
            numeric_value = float(value)
            self.sums[index] += numeric_value
            self.sums_squared[index] += numeric_value * numeric_value
            self.absolute_slopes[index] += abs(float(slope))

    def typical_pull(self, start: int, stop: int) -> float:
        """Average local slope times observed spread for one input section."""

        if self.count < 2:
            return 0.0
        pulls = []
        for index in range(start, stop):
            mean = self.sums[index] / self.count
            variance = max(0.0, self.sums_squared[index] / self.count - mean * mean)
            spread = math.sqrt(variance)
            mean_slope = self.absolute_slopes[index] / self.count
            pulls.append(spread * mean_slope)
        return sum(pulls) / len(pulls) if pulls else 0.0


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _counter_row(counter: Counter[str]) -> dict[str, int | float]:
    return {key: value for key, value in counter.items()}


class StrategyAccumulator:
    """Collect compact sufficient statistics rather than raw private hands."""

    def __init__(self) -> None:
        self.totals: defaultdict[tuple[str, int], Counter[str]] = defaultdict(Counter)
        self.ranks: defaultdict[tuple[str, int, int], Counter[str]] = defaultdict(Counter)
        self.rank_stages: defaultdict[
            tuple[str, int, int, str], Counter[str]
        ] = defaultdict(Counter)
        self.rank_pressure: defaultdict[
            tuple[str, int, int, str], Counter[str]
        ] = defaultdict(Counter)
        self.choice_pressure: defaultdict[
            tuple[str, int, str], Counter[str]
        ] = defaultdict(Counter)
        self.memory: defaultdict[
            tuple[str, int, str], Counter[str]
        ] = defaultdict(Counter)
        self.passes: defaultdict[
            tuple[str, int, str], Counter[str]
        ] = defaultdict(Counter)
        self.correlations: defaultdict[
            tuple[str, int, str], OnlineCorrelation
        ] = defaultdict(OnlineCorrelation)
        self.memory_tests: defaultdict[
            tuple[str, int, str], Counter[str]
        ] = defaultdict(Counter)
        self.input_pulls: defaultdict[
            tuple[str, int], InputPullStats
        ] = defaultdict(InputPullStats)
        self.state_samples: list[dict[str, Any]] = []
        self.state_sample_counts: Counter[tuple[str, int]] = Counter()
        self.skip_streaks: defaultdict[tuple[str, int, int, int, int], int] = defaultdict(int)

    @staticmethod
    def _views(mode: str, player_count: int) -> Iterable[tuple[str, int]]:
        yield (mode, player_count)
        yield (mode, 0)

    def add_games(
        self,
        mode: str,
        player_count: int,
        *,
        games: int,
        draws: int,
        turns: int,
        elapsed_seconds: float,
    ) -> None:
        for view in self._views(mode, player_count):
            target = self.totals[view]
            target["games"] += games
            target["draws"] += draws
            target["turns"] += turns
            target["elapsed_seconds"] += elapsed_seconds

    def add_decision(
        self,
        *,
        mode: str,
        player_count: int,
        game_index: int,
        seat: int,
        decision_number: int,
        observation: Sequence[float],
        legal_actions: Sequence[bool],
        action: int,
        top_probability: float,
        chosen_probability: float,
        uncertainty: float,
        value: float,
        memory_changed: bool,
        favorite_action: int | None = None,
        memory_tests: dict[str, dict[str, float | bool]] | None = None,
        input_slopes: Sequence[float] | None = None,
        memory_vector: Sequence[float] | None = None,
    ) -> None:
        suit_choice = bool(observation[93])
        own_hand = int(round(observation[91]))
        relative_sizes = [
            int(round(value)) for value in observation[69 : 69 + player_count]
        ]
        closest_rival = min(relative_sizes[1:])
        lead_over_closest = closest_rival - own_hand

        if (
            not suit_choice
            and memory_vector is not None
            and self.state_sample_counts[(mode, player_count)]
            < STATE_SAMPLES_PER_STYLE_AND_TABLE
        ):
            self.state_sample_counts[(mode, player_count)] += 1
            self.state_samples.append(
                {
                    "mode": mode,
                    "player_count": player_count,
                    "game_index": game_index,
                    "seat": seat,
                    "decision_number": decision_number,
                    "cards_in_hand": own_hand,
                    "closest_rival_cards": closest_rival,
                    "rival_pressure": (
                        "Rival has 1-2 cards"
                        if closest_rival <= 2
                        else "Rival has 3+ cards"
                    ),
                    "favorite_move": move_label(
                        action if favorite_action is None else favorite_action
                    ),
                    "move_taken": move_label(action),
                    "favorite_weight": top_probability,
                    "memory_changed": memory_changed,
                    "memory_vector": [float(item) for item in memory_vector],
                }
            )

        for view in self._views(mode, player_count):
            total = self.totals[view]
            total["decisions"] += 1
            if suit_choice:
                total["suit_choices"] += 1
                continue

            legal_cards = [index for index in range(50) if legal_actions[index]]
            playable_ranks = {rank_for_card(card_id) for card_id in legal_cards}
            chosen_rank = rank_for_card(action) if 0 <= action < 50 else None
            total["card_decisions"] += 1
            total["top_probability_sum"] += top_probability
            total["chosen_probability_sum"] += chosen_probability
            total["uncertainty_sum"] += uncertainty
            total["memory_changed"] += int(memory_changed)

            if memory_tests:
                for test_key, result in memory_tests.items():
                    tested = self.memory_tests[(view[0], view[1], test_key)]
                    tested["decisions"] += 1
                    tested["changed"] += int(bool(result["changed"]))
                    tested["weight_shift_sum"] += float(result["weight_shift"])
            if input_slopes is not None:
                self.input_pulls[(view[0], view[1])].add(
                    observation, input_slopes
                )
            if action == 54:
                total["passes"] += 1
            else:
                total["card_plays"] += 1
            if legal_cards:
                total["decisions_with_playable_card"] += 1
                if action == 54:
                    total["passes_with_playable_card"] += 1

            band = choice_band(len(legal_cards))
            choice = self.choice_pressure[(view[0], view[1], band)]
            choice["decisions"] += 1
            choice["top_probability_sum"] += top_probability
            choice["uncertainty_sum"] += uncertainty
            choice["passes"] += int(action == 54)
            choice["memory_changed"] += int(memory_changed)

            progress = self.memory[(view[0], view[1], progress_band(decision_number))]
            progress["decisions"] += 1
            progress["changed"] += int(memory_changed)

            if legal_cards:
                pass_counter = self.passes[
                    (view[0], view[1], hand_size_band(own_hand))
                ]
                pass_counter["opportunities"] += 1
                pass_counter["passes"] += int(action == 54)

            self.correlations[(view[0], view[1], "choices_and_certainty")].add(
                float(len(legal_cards)), top_probability
            )
            self.correlations[(view[0], view[1], "hand_lead_and_outlook")].add(
                float(lead_over_closest), value
            )
            self.correlations[(view[0], view[1], "time_and_memory")].add(
                float(decision_number), float(memory_changed)
            )

            if chosen_rank is not None:
                self.ranks[(view[0], view[1], chosen_rank)]["plays"] += 1

            pressure = "Rival has 1-2 cards" if closest_rival <= 2 else "Rival has 3+ cards"
            stage = hand_stage(own_hand)
            for rank in playable_ranks:
                kept = chosen_rank != rank
                rank_counter = self.ranks[(view[0], view[1], rank)]
                rank_counter["opportunities"] += 1
                rank_counter["held"] += int(kept)
                rank_counter["used"] += int(not kept)

                stage_counter = self.rank_stages[(view[0], view[1], rank, stage)]
                stage_counter["opportunities"] += 1
                stage_counter["used"] += int(not kept)

                pressure_counter = self.rank_pressure[
                    (view[0], view[1], rank, pressure)
                ]
                pressure_counter["opportunities"] += 1
                pressure_counter["used"] += int(not kept)

                streak_key = (view[0], view[1], game_index, seat, rank)
                if kept:
                    self.skip_streaks[streak_key] += 1
                else:
                    rank_counter["skipped_chances_before_use"] += self.skip_streaks[
                        streak_key
                    ]
                    rank_counter["uses_with_skip_history"] += 1
                    self.skip_streaks[streak_key] = 0

    def _totals_rows(self) -> list[dict[str, Any]]:
        rows = []
        for mode in MODE_LABELS:
            for players in (0, *PLAYER_COUNTS):
                counter = self.totals[(mode, players)]
                card_decisions = counter["card_decisions"]
                rank_opportunities = sum(
                    self.ranks[(mode, players, rank)]["opportunities"]
                    for rank in RANK_ORDER
                )
                held = sum(
                    self.ranks[(mode, players, rank)]["held"]
                    for rank in RANK_ORDER
                )
                rows.append(
                    {
                        "action_style": MODE_LABELS[mode],
                        "view_players": PLAYER_LABELS[players],
                        "games": int(counter["games"]),
                        "teacher_decisions": int(counter["decisions"]),
                        "card_decisions": int(card_decisions),
                        "rank_opportunities": int(rank_opportunities),
                        "rank_hold_rate": _ratio(held, rank_opportunities),
                        "strategic_pass_rate": _ratio(
                            counter["passes_with_playable_card"],
                            counter["decisions_with_playable_card"],
                        ),
                        "memory_change_rate": _ratio(
                            counter["memory_changed"], card_decisions
                        ),
                        "mean_top_probability": _ratio(
                            counter["top_probability_sum"], card_decisions
                        ),
                        "draw_rate": _ratio(counter["draws"], counter["games"]),
                        "average_turns": _ratio(counter["turns"], counter["games"]),
                        "elapsed_seconds": float(counter["elapsed_seconds"]),
                    }
                )
        return rows

    def _rank_rows(self) -> list[dict[str, Any]]:
        rows = []
        for mode in MODE_LABELS:
            for players in (0, *PLAYER_COUNTS):
                total_plays = self.totals[(mode, players)]["card_plays"]
                for rank in RANK_ORDER:
                    counter = self.ranks[(mode, players, rank)]
                    opportunities = int(counter["opportunities"])
                    held = int(counter["held"])
                    low, high = wilson_interval(held, opportunities)
                    close = self.rank_pressure[
                        (mode, players, rank, "Rival has 1-2 cards")
                    ]
                    safe = self.rank_pressure[
                        (mode, players, rank, "Rival has 3+ cards")
                    ]
                    close_rate = _ratio(close["used"], close["opportunities"])
                    safe_rate = _ratio(safe["used"], safe["opportunities"])
                    small_hand = self.rank_stages[
                        (mode, players, rank, "1-2 cards")
                    ]
                    large_hand = self.rank_stages[
                        (mode, players, rank, "6+ cards")
                    ]
                    small_hand_rate = _ratio(
                        small_hand["used"], small_hand["opportunities"]
                    )
                    large_hand_rate = _ratio(
                        large_hand["used"], large_hand["opportunities"]
                    )
                    rows.append(
                        {
                            "action_style": MODE_LABELS[mode],
                            "view_players": PLAYER_LABELS[players],
                            "rank": RANK_NAMES[rank],
                            "rank_with_effect": rank_label(rank),
                            "rank_order": RANK_ORDER.index(rank) + 1,
                            "kind": "Effect card" if rank in EFFECT_RANKS else "Ordinary card",
                            "effect": RANK_EFFECTS.get(rank, "no extra effect"),
                            "opportunities": opportunities,
                            "held": held,
                            "hold_rate": _ratio(held, opportunities),
                            "hold_rate_low": low,
                            "hold_rate_high": high,
                            "used": int(counter["used"]),
                            "plays": int(counter["plays"]),
                            "play_share": _ratio(counter["plays"], total_plays),
                            "average_skipped_chances_before_use": _ratio(
                                counter["skipped_chances_before_use"],
                                counter["uses_with_skip_history"],
                            ),
                            "use_rate_rival_close": close_rate,
                            "use_rate_rival_not_close": safe_rate,
                            "rival_pressure_change_pp": (close_rate - safe_rate) * 100.0,
                            "rival_close_opportunities": int(close["opportunities"]),
                            "rival_not_close_opportunities": int(safe["opportunities"]),
                            "play_rate_with_1_2_cards": small_hand_rate,
                            "play_rate_with_6_plus_cards": large_hand_rate,
                            "small_hand_change_pp": (
                                small_hand_rate - large_hand_rate
                            )
                            * 100.0,
                        }
                    )
        return rows

    def _stage_rows(self) -> list[dict[str, Any]]:
        rows = []
        stages = ("1-2 cards", "3-5 cards", "6+ cards")
        for mode in MODE_LABELS:
            for players in (0, *PLAYER_COUNTS):
                for rank in RANK_ORDER:
                    for stage in stages:
                        counter = self.rank_stages[(mode, players, rank, stage)]
                        rows.append(
                            {
                                "action_style": MODE_LABELS[mode],
                                "view_players": PLAYER_LABELS[players],
                                "rank": RANK_NAMES[rank],
                                "rank_order": RANK_ORDER.index(rank) + 1,
                                "cards_in_hand": stage,
                                "opportunities": int(counter["opportunities"]),
                                "play_rate": _ratio(
                                    counter["used"], counter["opportunities"]
                                ),
                            }
                        )
        return rows

    def _pressure_rows(self) -> list[dict[str, Any]]:
        rows = []
        for mode in MODE_LABELS:
            for players in (0, *PLAYER_COUNTS):
                for rank in RANK_ORDER:
                    if rank not in EFFECT_RANKS:
                        continue
                    for pressure in ("Rival has 1-2 cards", "Rival has 3+ cards"):
                        counter = self.rank_pressure[(mode, players, rank, pressure)]
                        rows.append(
                            {
                                "action_style": MODE_LABELS[mode],
                                "view_players": PLAYER_LABELS[players],
                                "rank": RANK_NAMES[rank],
                                "rank_order": RANK_ORDER.index(rank) + 1,
                                "rival_distance": pressure,
                                "opportunities": int(counter["opportunities"]),
                                "play_rate": _ratio(
                                    counter["used"], counter["opportunities"]
                                ),
                            }
                        )
        return rows

    def _choice_rows(self) -> list[dict[str, Any]]:
        rows = []
        bands = ("0", "1", "2", "3", "4+")
        for mode in MODE_LABELS:
            for players in (0, *PLAYER_COUNTS):
                for order, band in enumerate(bands):
                    counter = self.choice_pressure[(mode, players, band)]
                    decisions = counter["decisions"]
                    rows.append(
                        {
                            "action_style": MODE_LABELS[mode],
                            "view_players": PLAYER_LABELS[players],
                            "playable_card_types": band,
                            "choice_order": order,
                            "decisions": int(decisions),
                            "top_choice_probability": _ratio(
                                counter["top_probability_sum"], decisions
                            ),
                            "mean_uncertainty": _ratio(
                                counter["uncertainty_sum"], decisions
                            ),
                            "pass_rate": _ratio(counter["passes"], decisions),
                            "memory_change_rate": _ratio(
                                counter["memory_changed"], decisions
                            ),
                        }
                    )
        return rows

    def _memory_rows(self) -> list[dict[str, Any]]:
        rows = []
        bands = ("First decision", "Decisions 2-5", "Decisions 6-10", "Decision 11+")
        for mode in MODE_LABELS:
            for players in (0, *PLAYER_COUNTS):
                for order, band in enumerate(bands):
                    counter = self.memory[(mode, players, band)]
                    rows.append(
                        {
                            "action_style": MODE_LABELS[mode],
                            "view_players": PLAYER_LABELS[players],
                            "time_in_game": band,
                            "time_order": order,
                            "decisions": int(counter["decisions"]),
                            "memory_change_rate": _ratio(
                                counter["changed"], counter["decisions"]
                            ),
                        }
                    )
        return rows

    def _pass_rows(self) -> list[dict[str, Any]]:
        rows = []
        bands = ("1 card", "2 cards", "3-5 cards", "6-8 cards", "9+ cards")
        for mode in MODE_LABELS:
            for players in (0, *PLAYER_COUNTS):
                for order, band in enumerate(bands):
                    counter = self.passes[(mode, players, band)]
                    rows.append(
                        {
                            "action_style": MODE_LABELS[mode],
                            "view_players": PLAYER_LABELS[players],
                            "cards_in_hand": band,
                            "hand_order": order,
                            "opportunities": int(counter["opportunities"]),
                            "pass_rate": _ratio(
                                counter["passes"], counter["opportunities"]
                            ),
                        }
                    )
        return rows

    def _correlation_rows(self) -> list[dict[str, Any]]:
        definitions = (
            (
                "choices_and_certainty",
                "More playable card types and certainty",
                "Positive means the favorite move becomes clearer as more card types can be played.",
            ),
            (
                "hand_lead_and_outlook",
                "Card-count lead and model outlook",
                "Positive means the outlook rises when the model holds fewer cards than its closest rival.",
            ),
            (
                "time_and_memory",
                "Time in the game and memory changing the favorite move",
                "Positive means stored history matters more on later decisions.",
            ),
        )
        rows = []
        for mode in MODE_LABELS:
            for players in (0, *PLAYER_COUNTS):
                for key, label, explanation in definitions:
                    corr = self.correlations[(mode, players, key)]
                    value = corr.value()
                    rows.append(
                        {
                            "action_style": MODE_LABELS[mode],
                            "view_players": PLAYER_LABELS[players],
                            "relationship": label,
                            "relationship_score": value,
                            "decisions": corr.count,
                            "plain_meaning": explanation,
                        }
                    )
        return rows

    def _memory_test_rows(self) -> list[dict[str, Any]]:
        rows = []
        for mode in MODE_LABELS:
            for players in (0, *PLAYER_COUNTS):
                for order, (test_key, label) in enumerate(MEMORY_TESTS):
                    counter = self.memory_tests[(mode, players, test_key)]
                    decisions = int(counter["decisions"])
                    rows.append(
                        {
                            "action_style": MODE_LABELS[mode],
                            "view_players": PLAYER_LABELS[players],
                            "memory_test": label,
                            "test_order": order,
                            "decisions": decisions,
                            "favorite_change_rate": _ratio(
                                counter["changed"], decisions
                            ),
                            "move_weight_shift": _ratio(
                                counter["weight_shift_sum"], decisions
                            ),
                        }
                    )
        return rows

    def _input_pull_rows(self) -> list[dict[str, Any]]:
        rows = []
        for mode in MODE_LABELS:
            for players in (0, *PLAYER_COUNTS):
                stats = self.input_pulls[(mode, players)]
                pulls = [
                    stats.typical_pull(start, stop)
                    for _, start, stop in INPUT_GROUPS
                ]
                total_pull = sum(pulls)
                for order, ((label, start, stop), pull) in enumerate(
                    zip(INPUT_GROUPS, pulls)
                ):
                    rows.append(
                        {
                            "action_style": MODE_LABELS[mode],
                            "view_players": PLAYER_LABELS[players],
                            "input_part": label,
                            "part_order": order,
                            "values": stop - start,
                            "sampled_decisions": stats.count,
                            "local_pull": pull,
                            "share_of_local_pull": _ratio(pull, total_pull),
                        }
                    )
        return rows

    def _memory_map_rows(self) -> tuple[list[dict[str, Any]], dict[str, float]]:
        if len(self.state_samples) < 2:
            return ([], {"first_direction": 0.0, "second_direction": 0.0})

        import numpy as np

        matrix = np.asarray(
            [row["memory_vector"] for row in self.state_samples],
            dtype=np.float64,
        )
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        covariance = centered.T @ centered
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        chosen = order[:2]
        coordinates = centered @ eigenvectors[:, chosen]
        positive_total = float(np.maximum(eigenvalues, 0.0).sum())
        explained = [
            _ratio(float(max(0.0, eigenvalues[index])), positive_total)
            for index in chosen
        ]
        if coordinates.shape[1] == 1:
            coordinates = np.column_stack(
                (coordinates[:, 0], np.zeros(len(coordinates)))
            )
            explained.append(0.0)

        rows: list[dict[str, Any]] = []
        for sample, point in zip(self.state_samples, coordinates):
            base = {
                "action_style": MODE_LABELS[sample["mode"]],
                "memory_direction_1": float(point[0]),
                "memory_direction_2": float(point[1]),
                "players": sample["player_count"],
                "cards_in_hand": sample["cards_in_hand"],
                "closest_rival_cards": sample["closest_rival_cards"],
                "rival_pressure": sample["rival_pressure"],
                "favorite_move": sample["favorite_move"],
                "move_taken": sample["move_taken"],
                "favorite_weight": sample["favorite_weight"],
                "decision_number": sample["decision_number"],
                "memory_changed": (
                    "Earlier history changed the favorite"
                    if sample["memory_changed"]
                    else "Favorite stayed the same"
                ),
            }
            rows.append({**base, "view_players": PLAYER_LABELS[sample["player_count"]]})
            rows.append({**base, "view_players": PLAYER_LABELS[0]})
        return (
            rows,
            {
                "first_direction": explained[0] if explained else 0.0,
                "second_direction": explained[1] if len(explained) > 1 else 0.0,
            },
        )

    def _memory_readout_rows(self) -> list[dict[str, Any]]:
        """Test which table facts a small straight-line readout can recover."""

        if len(self.state_samples) < 20:
            return []

        import numpy as np

        rows: list[dict[str, Any]] = []
        questions = (
            ("Cards in its own hand", "number", 0.0),
            ("Cards held by the closest rival", "number", 0.0),
            ("How far the game has progressed", "number", 0.0),
            ("Whether a rival is down to 1-2 cards", "two groups", 0.5),
            ("Whether earlier history changes the favorite", "two groups", 0.5),
        )
        for mode in MODE_LABELS:
            samples = [row for row in self.state_samples if row["mode"] == mode]
            if len(samples) < 20:
                continue
            matrix = np.asarray(
                [row["memory_vector"] for row in samples], dtype=np.float64
            )
            targets = np.asarray(
                [
                    [
                        row["cards_in_hand"],
                        row["closest_rival_cards"],
                        row["decision_number"],
                        float(row["closest_rival_cards"] <= 2),
                        float(row["memory_changed"]),
                    ]
                    for row in samples
                ],
                dtype=np.float64,
            )
            test_mask = np.asarray(
                [
                    (
                        row["game_index"] * 31
                        + row["player_count"] * 7
                        + row["seat"] * 17
                    )
                    % 5
                    == 0
                    for row in samples
                ],
                dtype=bool,
            )
            if test_mask.sum() < 5 or (~test_mask).sum() < 10:
                continue
            train_x, test_x = matrix[~test_mask], matrix[test_mask]
            mean_x = train_x.mean(axis=0)
            spread_x = train_x.std(axis=0)
            spread_x[spread_x < 1e-8] = 1.0
            train_x = (train_x - mean_x) / spread_x
            test_x = (test_x - mean_x) / spread_x
            train_design = np.column_stack((train_x, np.ones(len(train_x))))
            test_design = np.column_stack((test_x, np.ones(len(test_x))))
            penalty = np.eye(train_design.shape[1]) * 10.0
            penalty[-1, -1] = 0.0
            coefficients = np.linalg.solve(
                train_design.T @ train_design + penalty,
                train_design.T @ targets[~test_mask],
            )
            predictions = test_design @ coefficients
            held_back = targets[test_mask]
            for index, (question, kind, baseline) in enumerate(questions):
                actual = held_back[:, index]
                predicted = predictions[:, index]
                if kind == "number":
                    denominator = float(((actual - actual.mean()) ** 2).sum())
                    score = (
                        1.0 - float(((actual - predicted) ** 2).sum()) / denominator
                        if denominator > 0.0
                        else 0.0
                    )
                    reading = (
                        "1 means exact recovery; 0 means no better than the held-back average."
                    )
                    score_name = "Share of variation recovered"
                else:
                    positive = actual >= 0.5
                    predicted_positive = predicted >= 0.5
                    positive_rate = _ratio(
                        float((predicted_positive & positive).sum()),
                        float(positive.sum()),
                    )
                    negative_rate = _ratio(
                        float((~predicted_positive & ~positive).sum()),
                        float((~positive).sum()),
                    )
                    score = (positive_rate + negative_rate) / 2.0
                    reading = "1 means perfect separation; 0.5 is chance-level balance."
                    score_name = "Balanced correctness"
                rows.append(
                    {
                        "action_style": MODE_LABELS[mode],
                        "question": question,
                        "score_kind": score_name,
                        "score": score,
                        "ordinary_baseline": baseline,
                        "held_back_examples": int(test_mask.sum()),
                        "plain_reading": reading,
                    }
                )
        return rows

    def final_rows(self) -> dict[str, list[dict[str, Any]]]:
        memory_map, memory_map_explained = self._memory_map_rows()
        return {
            "headline": self._totals_rows(),
            "rank_choices": self._rank_rows(),
            "effect_cards_by_rival_distance": self._pressure_rows(),
            "memory_layer_tests": self._memory_test_rows(),
            "input_pull": self._input_pull_rows(),
            "memory_map": memory_map,
            "memory_map_explained": [memory_map_explained],
        }


def _stack_state_parts(runtimes: Sequence[Any]) -> tuple[Any, ...]:
    import torch

    if not runtimes:
        raise ValueError("cannot stack empty model memory")
    return tuple(
        torch.cat([runtime.state[part] for runtime in runtimes], dim=1)
        for part in range(len(runtimes[0].state))
    )


def _scatter_state_parts(runtimes: Sequence[Any], states: tuple[Any, ...]) -> None:
    for batch_index, runtime in enumerate(runtimes):
        runtime.state = tuple(
            part[:, batch_index : batch_index + 1].detach()
            for part in states
        )


def _choose_actions(
    policy: Any,
    entries: Sequence[tuple[Any, Any]],
    *,
    deterministic: bool,
    sample_for_interpretability: Sequence[bool] | None = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Advance memory and run matched checks on the exact same decisions."""

    import torch

    if not entries:
        return [], []
    if sample_for_interpretability is None:
        sample_for_interpretability = [False] * len(entries)
    if len(sample_for_interpretability) != len(entries):
        raise ValueError("interpretability sample flags must match the decision batch")
    parameter = next(policy.parameters())
    device, dtype = parameter.device, parameter.dtype
    runtimes = [runtime for runtime, _ in entries]
    decisions = [decision for _, decision in entries]
    event_rows = [policy._event_features(decision) for decision in decisions]

    with torch.inference_mode():
        historical_states = tuple(
            part.clone() for part in _stack_state_parts(runtimes)
        )
        if not historical_states or historical_states[0].shape[0] < 2:
            raise RuntimeError("the memory-layer check requires a two-layer state")
        lower_states = [part.clone() for part in historical_states]
        upper_states = [part.clone() for part in historical_states]
        lower_states[0][0].zero_()
        upper_states[0][1].zero_()
        zero_states = tuple(
            part.clone() for part in policy.initial_state(len(entries), device=device)
        )
        layer_states = {
            "lower": tuple(lower_states),
            "upper": tuple(upper_states),
        }
        max_events = max((len(rows) for rows in event_rows), default=0)
        for event_index in range(max_events):
            selected = [
                index
                for index, rows in enumerate(event_rows)
                if event_index < len(rows)
            ]
            selected_runtimes = [runtimes[index] for index in selected]
            tokens = torch.tensor(
                [event_rows[index][event_index] for index in selected],
                device=device,
                dtype=dtype,
            )
            _, _, next_actual = policy.puffer_policy.forward_eval(
                tokens, _stack_state_parts(selected_runtimes)
            )
            _scatter_state_parts(selected_runtimes, next_actual)

            selected_index = torch.tensor(selected, device=device, dtype=torch.long)
            selected_zero = tuple(
                part.index_select(1, selected_index) for part in zero_states
            )
            _, _, next_zero = policy.puffer_policy.forward_eval(tokens, selected_zero)
            updated_zero = []
            for current, replacement in zip(zero_states, next_zero):
                updated = current.clone()
                updated.index_copy_(1, selected_index, replacement)
                updated_zero.append(updated)
            zero_states = tuple(updated_zero)

            for layer_key, current_states in tuple(layer_states.items()):
                selected_layer = tuple(
                    part.index_select(1, selected_index) for part in current_states
                )
                _, _, next_layer = policy.puffer_policy.forward_eval(
                    tokens, selected_layer
                )
                updated_layer = []
                for current, replacement in zip(current_states, next_layer):
                    updated = current.clone()
                    updated.index_copy_(1, selected_index, replacement)
                    updated_layer.append(updated)
                layer_states[layer_key] = tuple(updated_layer)

        predecision_states = tuple(
            part.clone() for part in _stack_state_parts(runtimes)
        )
        decision_tokens = torch.tensor(
            [policy._decision_features(decision) for decision in decisions],
            device=device,
            dtype=dtype,
        )
        logits, values, next_states = policy.puffer_policy.forward_eval(
            decision_tokens, predecision_states
        )
        zero_logits, _, _ = policy.puffer_policy.forward_eval(
            decision_tokens, zero_states
        )
        layer_logits = {
            layer_key: policy.puffer_policy.forward_eval(
                decision_tokens, states
            )[0]
            for layer_key, states in layer_states.items()
        }
        legal = torch.tensor(
            [decision.legal_actions for decision in decisions],
            device=device,
            dtype=torch.bool,
        )
        floor = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~legal, floor)
        zero_masked = zero_logits.masked_fill(~legal, floor)
        layer_masked = {
            layer_key: candidate.masked_fill(~legal, floor)
            for layer_key, candidate in layer_logits.items()
        }
        probabilities = torch.softmax(masked, dim=-1)
        zero_probabilities = torch.softmax(zero_masked, dim=-1)
        layer_probabilities = {
            layer_key: torch.softmax(candidate, dim=-1)
            for layer_key, candidate in layer_masked.items()
        }
        log_probabilities = torch.log_softmax(masked, dim=-1)
        top_actions = masked.argmax(dim=-1)
        zero_top_actions = zero_masked.argmax(dim=-1)
        layer_top_actions = {
            layer_key: candidate.argmax(dim=-1)
            for layer_key, candidate in layer_masked.items()
        }
        if deterministic:
            actions = top_actions.cpu().tolist()
        else:
            cpu_probabilities = probabilities.cpu()
            actions = [
                int(
                    torch.multinomial(
                        cpu_probabilities[index],
                        1,
                        generator=runtime.generator,
                    ).item()
                )
                for index, runtime in enumerate(runtimes)
            ]
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        top_probability = probabilities.max(dim=-1).values
        chosen_probability = probabilities[
            torch.arange(len(entries), device=device),
            torch.tensor(actions, device=device),
        ]
        zero_weight_shift = 0.5 * (
            probabilities - zero_probabilities
        ).abs().sum(dim=-1)
        layer_weight_shift = {
            layer_key: 0.5 * (probabilities - candidate).abs().sum(dim=-1)
            for layer_key, candidate in layer_probabilities.items()
        }
        _scatter_state_parts(runtimes, next_states)

    top_cpu = top_actions.cpu().tolist()
    zero_top_cpu = zero_top_actions.cpu().tolist()
    layer_top_cpu = {
        layer_key: candidate.cpu().tolist()
        for layer_key, candidate in layer_top_actions.items()
    }
    zero_shift_cpu = zero_weight_shift.cpu().tolist()
    layer_shift_cpu = {
        layer_key: candidate.cpu().tolist()
        for layer_key, candidate in layer_weight_shift.items()
    }
    value_cpu = values.reshape(-1).cpu().tolist()
    top_probability_cpu = top_probability.cpu().tolist()
    chosen_probability_cpu = chosen_probability.cpu().tolist()
    entropy_cpu = entropy.cpu().tolist()

    sampled_indices = [
        index
        for index, should_sample in enumerate(sample_for_interpretability)
        if should_sample
        and not bool(decisions[index].observation.values[93])
        and sum(bool(value) for value in decisions[index].legal_actions) >= 2
    ]
    input_slopes_by_index: dict[int, list[float]] = {}
    if sampled_indices:
        selected_index = torch.tensor(sampled_indices, device=device, dtype=torch.long)
        with torch.enable_grad():
            slope_tokens = torch.tensor(
                [
                    policy._decision_features(decisions[index])
                    for index in sampled_indices
                ],
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
            slope_states = tuple(
                part.index_select(1, selected_index).detach().clone()
                for part in predecision_states
            )
            slope_logits, _, _ = policy.puffer_policy.forward_eval(
                slope_tokens, slope_states
            )
            slope_legal = torch.tensor(
                [decisions[index].legal_actions for index in sampled_indices],
                device=device,
                dtype=torch.bool,
            )
            local_top = torch.tensor(
                [top_cpu[index] for index in sampled_indices],
                device=device,
                dtype=torch.long,
            )
            competitors = slope_legal.clone()
            competitors[
                torch.arange(len(sampled_indices), device=device), local_top
            ] = False
            runner_up = slope_logits.masked_fill(~competitors, floor).argmax(dim=-1)
            row_index = torch.arange(len(sampled_indices), device=device)
            favorite_gap = (
                slope_logits[row_index, local_top]
                - slope_logits[row_index, runner_up]
            )
            slopes = torch.autograd.grad(favorite_gap.sum(), slope_tokens)[0][
                :, :94
            ]
        for sample_index, values_row in zip(
            sampled_indices, slopes.detach().cpu().tolist()
        ):
            input_slopes_by_index[sample_index] = [
                float(value) for value in values_row
            ]

    memory_vectors_by_index: dict[int, list[float]] = {}
    for index, should_sample in enumerate(sample_for_interpretability):
        if not should_sample or bool(decisions[index].observation.values[93]):
            continue
        flattened = torch.cat(
            [part[:, index].reshape(-1) for part in next_states]
        )
        memory_vectors_by_index[index] = [
            float(value) for value in flattened.cpu().tolist()
        ]

    details = []
    for index in range(len(entries)):
        memory_tests = {
            "all": {
                "changed": top_cpu[index] != zero_top_cpu[index],
                "weight_shift": float(zero_shift_cpu[index]),
            },
            "lower": {
                "changed": top_cpu[index] != layer_top_cpu["lower"][index],
                "weight_shift": float(layer_shift_cpu["lower"][index]),
            },
            "upper": {
                "changed": top_cpu[index] != layer_top_cpu["upper"][index],
                "weight_shift": float(layer_shift_cpu["upper"][index]),
            },
        }
        details.append(
            {
                "value": float(value_cpu[index]),
                "top_probability": float(top_probability_cpu[index]),
                "chosen_probability": float(chosen_probability_cpu[index]),
                "uncertainty": float(entropy_cpu[index]),
                "favorite_action": int(top_cpu[index]),
                "favorite_without_earlier_history": int(zero_top_cpu[index]),
                "memory_changed": bool(memory_tests["all"]["changed"]),
                "memory_tests": memory_tests,
                "input_slopes": input_slopes_by_index.get(index),
                "memory_vector": memory_vectors_by_index.get(index),
            }
        )
    return [int(action) for action in actions], details


def _run_population(
    policy: Any,
    accumulator: StrategyAccumulator,
    *,
    mode: str,
    player_count: int,
    games: int,
    seed: int,
    max_turns: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    from batched_pesten import BatchedPesten
    from temporal_batched_speed import (
        SafeEventCursor,
        TemporalSeatRuntime,
        _make_vector_layout,
        build_safe_decision,
    )

    absolute_indices = tuple(range(games))
    vector_indices, policy_kinds, actor_keys, roles = _make_vector_layout(
        game_indices=absolute_indices,
        games=games,
        seed=seed,
        schedule_offset=0,
        schedule=("current_selfplay",),
        historical=(),
        random_opponent_seeds=None,
        player_count=player_count,
    )
    if vector_indices != list(absolute_indices):
        raise AssertionError("self-play analysis unexpectedly left the vector engine")
    engine = BatchedPesten(
        policy_kinds,
        actor_keys,
        np.asarray([seed + index for index in vector_indices], dtype=np.uint32),
        max_turns=max_turns,
        smart_agents_have_game_reference=True,
    )
    runtimes: dict[tuple[int, int], Any] = {}
    for game_index in range(games):
        for seat in range(player_count):
            role, _, agent_seed = roles[(game_index, seat)]
            if role != "current":
                raise AssertionError("all analysis seats must use the teacher")
            runtimes[(game_index, seat)] = TemporalSeatRuntime(
                game_index=game_index,
                absolute_game_index=game_index,
                seat=seat,
                policy=policy,
                generator=torch.Generator(device="cpu").manual_seed(agent_seed),
                state=policy.initial_state(),
                cursor=SafeEventCursor(),
            )

    started = time.perf_counter()
    while True:
        active = engine.active_games()
        if len(active) == 0:
            break
        entries = []
        contexts = []
        for game_value in active:
            game_index = int(game_value)
            seat = int(engine.current_player[game_index])
            runtime = runtimes[(game_index, seat)]
            decision = build_safe_decision(
                engine, game_index, seat, runtime.cursor
            )
            entries.append((runtime, decision))
            contexts.append((game_index, seat, runtime.cursor.decision_count))
        sample_flags = [
            accumulator.state_sample_counts[(mode, player_count)]
            < STATE_SAMPLES_PER_STYLE_AND_TABLE
            and (
                game_index * 131
                + seat * 17
                + decision_number * 29
                + player_count * 7
            )
            % 11
            == 0
            for game_index, seat, decision_number in contexts
        ]
        actions, details = _choose_actions(
            policy,
            entries,
            deterministic=(mode == "top"),
            sample_for_interpretability=sample_flags,
        )
        for (runtime, decision), context, action, detail in zip(
            entries, contexts, actions, details
        ):
            game_index, seat, decision_number = context
            accumulator.add_decision(
                mode=mode,
                player_count=player_count,
                game_index=game_index,
                seat=seat,
                decision_number=decision_number,
                observation=decision.observation.values,
                legal_actions=decision.legal_actions,
                action=action,
                top_probability=float(detail["top_probability"]),
                chosen_probability=float(detail["chosen_probability"]),
                uncertainty=float(detail["uncertainty"]),
                value=float(detail["value"]),
                memory_changed=bool(detail["memory_changed"]),
                favorite_action=int(detail["favorite_action"]),
                memory_tests=detail["memory_tests"],
                input_slopes=detail["input_slopes"],
                memory_vector=detail["memory_vector"],
            )
        engine.apply_actions(active, actions)

    elapsed = time.perf_counter() - started
    draws = int((engine.winners < 0).sum())
    turns = int(engine.turn_counts.sum())
    accumulator.add_games(
        mode,
        player_count,
        games=games,
        draws=draws,
        turns=turns,
        elapsed_seconds=elapsed,
    )
    return {
        "action_style": MODE_LABELS[mode],
        "players": player_count,
        "games": games,
        "draws": draws,
        "turns": turns,
        "elapsed_seconds": elapsed,
    }


def _default_findings(rows: dict[str, list[dict[str, Any]]]) -> list[str]:
    rank_rows = [
        row
        for row in rows["rank_choices"]
        if row["action_style"] == MODE_LABELS["natural"]
        and row["view_players"] == PLAYER_LABELS[0]
        and row["opportunities"] >= 100
    ]
    most_kept = max(rank_rows, key=lambda row: row["hold_rate"])
    pressure_candidates = [
        row
        for row in rank_rows
        if row["kind"] == "Effect card"
        and row["rival_close_opportunities"] >= 50
        and row["rival_not_close_opportunities"] >= 50
    ]
    pressure = max(pressure_candidates, key=lambda row: row["rival_pressure_change_pp"])
    findings = [
        (
            f"{most_kept['rank_with_effect']} was kept most often: "
            f"{most_kept['hold_rate']:.1%} of the times it could be played "
            f"({most_kept['opportunities']:,} chances)."
        ),
        (
            f"When a rival was down to one or two cards, {pressure['rank_with_effect']} "
            f"was used {pressure['rival_pressure_change_pp']:+.1f} percentage points "
            "more often than when every rival had at least three cards."
        ),
    ]
    layer_rows = [
        row
        for row in rows["memory_layer_tests"]
        if row["action_style"] == MODE_LABELS["natural"]
        and row["view_players"] == PLAYER_LABELS[0]
        and row["decisions"] > 0
    ]
    if layer_rows:
        strongest_layer = max(
            layer_rows, key=lambda row: row["favorite_change_rate"]
        )
        findings.append(
            f"{strongest_layer['memory_test']} changed the favorite move on "
            f"{strongest_layer['favorite_change_rate']:.1%} of otherwise identical "
            "card decisions."
        )
    input_rows = [
        row
        for row in rows["input_pull"]
        if row["action_style"] == MODE_LABELS["natural"]
        and row["view_players"] == PLAYER_LABELS[0]
        and row["sampled_decisions"] > 0
    ]
    if input_rows:
        strongest_input = max(
            input_rows, key=lambda row: row["share_of_local_pull"]
        )
        findings.append(
            f"Near the sampled decisions, {strongest_input['input_part'].lower()} "
            f"accounted for {strongest_input['share_of_local_pull']:.1%} of what "
            "could sway the favorite over the runner-up."
        )
    return findings


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refresh_summary_language(summary: dict[str, Any]) -> dict[str, Any]:
    """Refresh display wording without rerunning measured games."""

    summary["protocol"]["memory_layer_test"] = MEMORY_STAGE_CHECK_DESCRIPTION
    summary["protocol"]["input_pull_test"] = CURRENT_TURN_CHECK_DESCRIPTION
    summary["protocol"]["memory_map"] = MEMORY_MAP_DESCRIPTION
    summary["datasets"]["model_shape"] = [dict(row) for row in MODEL_SHAPE_ROWS]
    return summary


def analyze_checkpoint(
    checkpoint: Path,
    *,
    games_per_player_count: int = 512,
    seed: int = 202_608_120,
    modes: Sequence[str] = ("natural", "top"),
    device: str = "cpu",
    max_turns: int = 1000,
    torch_threads: int = 8,
) -> dict[str, Any]:
    """Run paired teacher self-play and return compact chart-ready results."""

    if games_per_player_count < 1:
        raise ValueError("games_per_player_count must be positive")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")
    if torch_threads < 1:
        raise ValueError("torch_threads must be positive")
    unknown_modes = set(modes) - set(MODE_LABELS)
    if unknown_modes:
        raise ValueError(f"unknown action styles: {sorted(unknown_modes)}")
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_hash_before = file_sha256(checkpoint)

    import torch

    from temporal_observation import ACTION_COUNT, EVENT_TOKEN_SIZE, OBSERVATION_SIZE
    from temporal_puffer_v4 import (
        NATIVE_INPUT_SIZE,
        load_temporal_policy_checkpoint,
        policy_model_config,
    )

    torch.set_num_threads(torch_threads)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    policy, payload = load_temporal_policy_checkpoint(checkpoint, device=device)
    policy.eval()
    model = policy_model_config(policy)
    state = policy.initial_state(3)
    state_shapes = [list(part.shape) for part in state]
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())

    accumulator = StrategyAccumulator()
    runs = []
    analysis_started = time.perf_counter()
    for mode in modes:
        for player_count in PLAYER_COUNTS:
            run_seed = seed + player_count * 1_000_000
            runs.append(
                _run_population(
                    policy,
                    accumulator,
                    mode=mode,
                    player_count=player_count,
                    games=games_per_player_count,
                    seed=run_seed,
                    max_turns=max_turns,
                )
            )
    rows = accumulator.final_rows()
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != checkpoint_hash_before:
        raise RuntimeError(
            "the teacher checkpoint changed during analysis; rerun against one stable file"
        )
    model_io = {
        "architecture": "two-layer compact GRU",
        "observation_values": OBSERVATION_SIZE,
        "public_event_values": EVENT_TOKEN_SIZE,
        "neural_step_values": NATIVE_INPUT_SIZE,
        "legal_action_flags": ACTION_COUNT,
        "memory_shape_for_batch_3": state_shapes,
        "memory_values_per_player": int(model["hidden_size"]) * int(model["num_layers"]),
        "move_scores": ACTION_COUNT,
        "outlook_values": 1,
        "parameters": parameter_count,
        "hidden_values_per_layer": int(model["hidden_size"]),
        "layers": int(model["num_layers"]),
        "mode_conditioned_heads": bool(model.get("mode_conditioned_heads", False)),
    }
    static_rows = {
        "input_groups": [
            {"part": "Cards it holds", "values": 50, "meaning": "one count for each exact card"},
            {"part": "Suit on the table", "values": 5, "meaning": "one of four suits or the special suit"},
            {"part": "Number on the table", "values": 14, "meaning": "Joker, Ace through Queen, or Jack"},
            {"part": "Everyone's card counts", "values": 13, "meaning": "its own and up to twelve other players"},
            {"part": "Suits played so far", "values": 5, "meaning": "how many cards of each suit were played"},
            {"part": "Turn and rule details", "values": 7, "meaning": "seat, turn, direction, penalty, own count, players, suit-choice mode"},
        ],
        "output_groups": [
            {"part": "Play a specific card", "values": 50, "meaning": "one score for each exact card"},
            {"part": "Choose a suit", "values": 4, "meaning": "used after a Jack"},
            {"part": "Pass", "values": 1, "meaning": "draw or accept a penalty"},
        ],
        "model_shape": [dict(row) for row in MODEL_SHAPE_ROWS],
    }
    return {
        "schema_version": 1,
        "kind": "puffer_gru_teacher_strategy_analysis",
        "generated_at": generated_at,
        "checkpoint": {
            "path": checkpoint.as_posix(),
            "sha256": checkpoint_hash,
            "epoch": int(payload.get("epoch", -1)),
            "model": model,
        },
        "protocol": {
            "population": "teacher self-play in every seat",
            "player_counts": list(PLAYER_COUNTS),
            "games_per_player_count_per_style": games_per_player_count,
            "action_styles": [MODE_LABELS[mode] for mode in modes],
            "paired_deals_across_action_styles": len(modes) > 1,
            "base_seed": seed,
            "max_turns": max_turns,
            "suit_handling": "all four ordinary suits combined by card number",
            "hold_definition": "a number was kept when at least one card of that number was legal but the model played another number or passed",
            "memory_test": "favorite legal move with normal stored history versus the same decision with earlier history cleared while keeping new public events",
            "memory_layer_test": MEMORY_STAGE_CHECK_DESCRIPTION,
            "input_pull_test": CURRENT_TURN_CHECK_DESCRIPTION,
            "memory_map": MEMORY_MAP_DESCRIPTION,
            "interpretability_samples": len(accumulator.state_samples),
            "device": device,
            "torch_threads": torch_threads,
            "elapsed_seconds": time.perf_counter() - analysis_started,
            "runs": runs,
        },
        "model_io": model_io,
        "findings": _default_findings(rows),
        "datasets": {**rows, **static_rows},
    }
