#!/usr/bin/env python3
"""Interactive benchmark protocol for one persistent Luna Max Codex agent.

The process owns all hidden state. The Codex player sees only the structured
public observations printed to stdout and responds through stdin, allowing one
long-lived model context to play and learn across every game in a run.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np

import card
import deck
import game
from hardcodedAgent import HardcodedStrategyAgent, SPECIAL_RANKS


@dataclass
class GameResult:
    game: int
    seed: int
    luna_seat: int
    winner: str
    turns: int
    luna_decisions: int
    invalid_responses: int


class LunaConsoleAgent:
    """Agent API that obtains each non-forced move from stdin."""

    def __init__(self, game_instance, controller):
        self.game = game_instance
        self.controller = controller
        self.mydeck = deck.Deck([])
        self.type = "luna_max_persistent"
        self.decision_count = 0
        self.invalid_responses = 0
        self.has_prompted = False
        self.last_drawn_number = None
        self.last_draw_turn = None

    def addCard(self, drawn_card):
        if drawn_card is not None:
            self.mydeck.cards.append(drawn_card)
            if self.has_prompted:
                self.last_drawn_number = drawn_card.number
                self.last_draw_turn = self.game.turn_count

    def remove(self, played_card):
        for held_card in self.mydeck.cards:
            if held_card.number == played_card.number:
                self.mydeck.cards.remove(held_card)
                return

    def playCard(self, current_sort, current_true_number):
        if self.game.penalty_amount > 0:
            playable = [
                held_card
                for held_card in self.mydeck.cards
                if held_card.truenumber == current_true_number
            ]
        else:
            playable = [
                held_card
                for held_card in self.mydeck.cards
                if held_card.compatible(current_sort, current_true_number)
            ]

        if not playable:
            observation = self._observation(
                kind="forced_pass",
                legal=["PASS"],
            )
            print(
                "FORCED_PASS "
                + json.dumps(observation, separators=(",", ":")),
                flush=True,
            )
            self.controller.record_audit(
                event="forced_pass",
                observation=observation,
                response="PASS",
            )
            self.has_prompted = True
            return None

        legal_numbers = sorted({held_card.number for held_card in playable})
        legal_tokens = [str(number) for number in legal_numbers] + ["PASS"]
        observation = self._observation(
            kind="play",
            legal=legal_tokens,
        )
        if (
            len(self.mydeck.cards) == 1
            and self.mydeck.cards[0].truenumber in SPECIAL_RANKS
        ):
            observation["rule_warning"] = (
                "A special card cannot finish: playing it draws two without "
                "removing it; PASS draws one."
            )

        token = self._ask("DECISION", observation, legal_tokens)
        self.has_prompted = True
        self.last_drawn_number = None
        self.last_draw_turn = None
        if token == "PASS":
            return None
        number = int(token)
        return next(
            held_card
            for held_card in self.mydeck.cards
            if held_card.number == number
        )

    def changeSort(self):
        legal = [suit.upper() for suit in card.sorts[:4]]
        observation = self._observation(kind="choose_suit", legal=legal)
        token = self._ask("SUIT_DECISION", observation, legal)
        return token.lower()

    def _ask(self, prefix, observation, legal):
        self.decision_count += 1
        print(
            f"{prefix} "
            + json.dumps(observation, separators=(",", ":")),
            flush=True,
        )
        print(f"RESPOND_WITH_ONE_OF {' '.join(legal)}", flush=True)
        while True:
            response = sys.stdin.readline()
            if response == "":
                raise EOFError("Luna benchmark stdin closed before completion")
            token = response.strip().upper()
            if token in legal:
                self.controller.record_audit(
                    event="decision",
                    observation=observation,
                    response=token,
                )
                return token
            self.invalid_responses += 1
            self.controller.record_audit(
                event="invalid_response",
                observation=observation,
                response=token,
            )
            print(
                f"INVALID_RESPONSE {json.dumps(token)}; "
                f"RESPOND_WITH_ONE_OF {' '.join(legal)}",
                flush=True,
            )

    def _observation(self, *, kind, legal):
        own_index = self.game.players.index(self)
        played_by_player = {}
        chosen_suits = {}
        for player_index in range(self.game.num_players):
            played_by_player[str(player_index)] = []
            chosen_suits[str(player_index)] = []
        for event in getattr(self.game, "public_history", []):
            player_key = str(event["player"])
            if event["action"] == "play":
                played_by_player[player_key].append(
                    self.controller.card_label(event["card"])
                )
            elif event["action"] == "choose_sort":
                chosen_suits[player_key].append(event["sort"])

        newly_drawn = None
        if self.last_draw_turn == self.game.turn_count:
            newly_drawn = self.controller.card_label(self.last_drawn_number)

        return {
            "game": self.controller.game_number,
            "games_total": self.controller.games,
            "kind": kind,
            "turn": self.game.turn_count,
            "your_seat": own_index,
            "current_player": self.game.current_player_index,
            "direction": self.game.direction,
            "top": {
                "sort": self.game.current_sort,
                "rank": self.game.current_true_number,
                "card": self.controller.card_label(
                    self.game.game_deck.topCard().number
                ),
            },
            "penalty": self.game.penalty_amount,
            "your_hand": [
                self.controller.card_label(held_card.number)
                for held_card in sorted(
                    self.mydeck.cards,
                    key=lambda held_card: held_card.number,
                )
            ],
            "newly_drawn": newly_drawn,
            "hand_sizes": [
                player.mydeck.cardCount() for player in self.game.players
            ],
            "public_play_history": played_by_player,
            "announced_suits": chosen_suits,
            "legal": legal,
        }


class LunaBenchmarkController:
    def __init__(
        self,
        *,
        games,
        seed,
        output,
        opponent_factory=HardcodedStrategyAgent,
        opponent_name="bayesian_tree_search",
        opponent_metadata=None,
        audit_output=None,
    ):
        self.games = games
        self.seed = seed
        self.output = output
        self.opponent_factory = opponent_factory
        self.opponent_name = opponent_name
        self.opponent_metadata = opponent_metadata or {}
        self.audit_output = audit_output or output.with_suffix(".audit.jsonl")
        self.game_number = 0
        self.results = []

    @staticmethod
    def card_label(number):
        if number is None:
            return None
        instance = card.Card(number)
        return {
            "id": number,
            "name": instance.toString(),
            "sort": instance.sort,
            "rank": instance.truenumber,
        }

    def record_audit(self, *, event, observation, response):
        record = {
            "event": event,
            "game": self.game_number,
            "observation": observation,
            "response": response,
        }
        with self.audit_output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def run(self):
        print(
            "PROTOCOL "
            + json.dumps(
                {
                    "player": "one persistent Luna Max Codex context",
                    "opponent": self.opponent_name,
                    "games": self.games,
                    "seat_rotation": (
                        "alternating seats with a fresh hidden deal every game"
                    ),
                    "deal_privacy": (
                        "deal seeds are never shown during the benchmark and "
                        "no deal is replayed"
                    ),
                    "learning": (
                        "Retain and revise your strategy after each result. "
                        "Do not reset the agent between games."
                    ),
                    "rules": {
                        "compatible": "same suit, same rank, Jack, or Joker",
                        "seven": "actor takes another turn",
                        "eight": "skip the next player",
                        "ace": "reverse direction",
                        "two": "add two; only another two stacks",
                        "joker": (
                            "add five; only another Joker stacks; the player "
                            "who takes the chain announces the next suit"
                        ),
                        "jack": "announce the next suit",
                        "special_finish": (
                            "7, 8, Ace, 2, Jack, and Joker cannot win as the "
                            "last card; the failed attempt draws two"
                        ),
                        "pass": (
                            "draw one; the exact drawn card may be played "
                            "immediately if compatible"
                        ),
                    },
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        self.audit_output.parent.mkdir(parents=True, exist_ok=True)
        self.audit_output.write_text("", encoding="utf-8")
        started = time.perf_counter()

        for game_offset in range(self.games):
            self.game_number = game_offset + 1
            game_seed = self.seed + game_offset
            luna_seat = game_offset % 2
            np.random.seed(game_seed)
            random.seed(game_seed)
            instance = game.Game([])

            def luna_factory(game_instance):
                return LunaConsoleAgent(game_instance, self)

            factories = (
                [luna_factory, self.opponent_factory]
                if luna_seat == 0
                else [self.opponent_factory, luna_factory]
            )
            for factory in factories:
                instance.players.append(factory(instance))
            instance.num_players = 2
            luna_agent = instance.players[luna_seat]
            instance.reset()
            print(
                "GAME_START "
                + json.dumps(
                    {
                        "game": self.game_number,
                        "your_seat": luna_seat,
                        "starting_player": 0,
                        "top": self.card_label(
                            instance.game_deck.topCard().number
                        ),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

            while not instance.game_over and instance.turn_count < 1_000:
                instance.simulate_turn()

            if instance.winner is None:
                winner = "draw"
            elif instance.winner == luna_seat:
                winner = "luna"
            else:
                winner = "opponent"
            result = GameResult(
                game=self.game_number,
                seed=game_seed,
                luna_seat=luna_seat,
                winner=winner,
                turns=instance.turn_count,
                luna_decisions=luna_agent.decision_count,
                invalid_responses=luna_agent.invalid_responses,
            )
            self.results.append(result)
            visible_result = {
                key: value
                for key, value in asdict(result).items()
                if key != "seed"
            }
            print(
                "GAME_RESULT "
                + json.dumps(
                    {
                        **visible_result,
                        "score_so_far": self._score(),
                        "instruction": (
                            "Use this outcome and remembered decisions to "
                            "update your strategy, then continue in the same "
                            "context."
                        ),
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

        payload = {
            "status": "complete",
            "agent": "gpt-5.6-luna",
            "reasoning_effort": "max",
            "persistent_context": True,
            "opponent": self.opponent_name,
            "opponent_metadata": self.opponent_metadata,
            "games": self.games,
            "seed": self.seed,
            "elapsed_seconds": time.perf_counter() - started,
            "score": self._score(),
            "learning_curve": self._learning_curve(),
            "audit_file": str(self.audit_output),
            "per_game": [asdict(result) for result in self.results],
        }
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.output.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "BENCHMARK_COMPLETE " + json.dumps(payload, separators=(",", ":")),
            flush=True,
        )
        print(f"RESULT_FILE {self.output}", flush=True)

    def _score(self):
        wins = sum(result.winner == "luna" for result in self.results)
        losses = sum(result.winner == "opponent" for result in self.results)
        draws = len(self.results) - wins - losses
        interval = wilson_interval(wins, len(self.results))
        return {
            "luna_wins": wins,
            "opponent": self.opponent_name,
            "opponent_wins": losses,
            "draws": draws,
            "luna_win_rate": wins / len(self.results) if self.results else 0.0,
            "luna_win_rate_ci95": interval,
            "luna_wins_as_seat_0": sum(
                result.winner == "luna" and result.luna_seat == 0
                for result in self.results
            ),
            "luna_wins_as_seat_1": sum(
                result.winner == "luna" and result.luna_seat == 1
                for result in self.results
            ),
        }

    def _learning_curve(self):
        split = len(self.results) // 2

        def rate(results):
            return (
                sum(result.winner == "luna" for result in results)
                / len(results)
                if results
                else 0.0
            )

        return {
            "first_half_win_rate": rate(self.results[:split]),
            "second_half_win_rate": rate(self.results[split:]),
        }


def wilson_interval(successes, trials):
    if not trials:
        return [0.0, 0.0]
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1 + z * z / trials
    centre = proportion + z * z / (2 * trials)
    margin = z * math.sqrt(
        proportion * (1 - proportion) / trials
        + z * z / (4 * trials * trials)
    )
    return [
        (centre - margin) / denominator,
        (centre + margin) / denominator,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--opponent",
        choices=("tree", "ppo"),
        default="tree",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("ppo_actor_100run3.pth"),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
    )
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        raise SystemExit("--games must be an even number of at least two")
    if args.opponent == "ppo":
        from benchmark_agents import file_sha256, load_ppo_agent

        if not args.model.is_file():
            raise SystemExit(f"PPO checkpoint not found: {args.model}")
        opponent_factory = load_ppo_agent(args.model, args.device)
        opponent_name = "live_ppo_greedy"
        opponent_metadata = {
            "checkpoint": str(args.model),
            "sha256": file_sha256(args.model),
        }
    else:
        opponent_factory = HardcodedStrategyAgent
        opponent_name = "bayesian_tree_search"
        opponent_metadata = {
            "belief_samples": 8,
            "search_depth": 3,
            "beam_width": 3,
        }
    output = args.output or Path(
        "benchmarks"
    ) / f"luna_max_vs_{args.opponent}.json"
    LunaBenchmarkController(
        games=args.games,
        seed=args.seed,
        output=output,
        opponent_factory=opponent_factory,
        opponent_name=opponent_name,
        opponent_metadata=opponent_metadata,
        audit_output=args.audit_output,
    ).run()


if __name__ == "__main__":
    main()
