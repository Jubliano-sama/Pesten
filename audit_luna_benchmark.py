#!/usr/bin/env python3
"""Validate and summarize a Luna benchmark's public game transcript."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


ALLOWED_OBSERVATION_FIELDS = {
    "game",
    "games_total",
    "kind",
    "turn",
    "your_seat",
    "current_player",
    "direction",
    "top",
    "penalty",
    "your_hand",
    "newly_drawn",
    "hand_sizes",
    "public_play_history",
    "announced_suits",
    "legal",
    "rule_warning",
}
FORBIDDEN_FIELD_NAMES = {
    "seed",
    "rng",
    "draw_pile",
    "grab_deck",
    "opponent_hand",
    "hidden_hand",
    "deck_order",
}


def nested_field_names(value):
    names = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            names.add(key)
            names.update(nested_field_names(nested))
    elif isinstance(value, list):
        for nested in value:
            names.update(nested_field_names(nested))
    return names


def audit(audit_path: Path, result_path: Path | None = None):
    violations = []
    event_counts = Counter()
    response_counts = Counter()
    voluntary_passes = 0
    special_finish_passes = 0
    special_finish_nonpass_plays = 0
    games_seen = set()

    records = []
    for line_number, line in enumerate(
        audit_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            violations.append(f"line {line_number}: invalid JSON: {error}")
            continue
        records.append(record)
        event = record.get("event")
        observation = record.get("observation")
        response = record.get("response")
        event_counts[event] += 1
        games_seen.add(record.get("game"))

        if not isinstance(observation, dict):
            violations.append(f"line {line_number}: observation is not an object")
            continue
        unknown_fields = set(observation) - ALLOWED_OBSERVATION_FIELDS
        if unknown_fields:
            violations.append(
                f"line {line_number}: unknown observation fields "
                f"{sorted(unknown_fields)}"
            )
        forbidden = nested_field_names(observation) & FORBIDDEN_FIELD_NAMES
        if forbidden:
            violations.append(
                f"line {line_number}: hidden-state fields {sorted(forbidden)}"
            )

        legal = observation.get("legal") or []
        if event in {"decision", "forced_pass"} and response not in legal:
            violations.append(
                f"line {line_number}: response {response!r} not in legal actions"
            )
        if event == "forced_pass" and response != "PASS":
            violations.append(
                f"line {line_number}: forced pass responded with {response!r}"
            )
        if event == "invalid_response" and response in legal:
            violations.append(
                f"line {line_number}: legal action logged as invalid"
            )

        if event == "decision":
            response_counts[response] += 1
            if observation.get("kind") == "play":
                if (
                    response == "PASS"
                    and len(legal) > 1
                    and not observation.get("rule_warning")
                ):
                    voluntary_passes += 1
                if response != "PASS":
                    hand_ids = {
                        str(card_record["id"])
                        for card_record in observation.get("your_hand", [])
                    }
                    if response not in hand_ids:
                        violations.append(
                            f"line {line_number}: played card is not in own hand"
                        )
            if observation.get("rule_warning"):
                if response == "PASS":
                    special_finish_passes += 1
                else:
                    # This is legal in the engine, but it draws two cards
                    # without shedding the last special and is usually a
                    # strategic mistake.
                    special_finish_nonpass_plays += 1

    unique_deals = None
    result_summary = None
    if result_path is not None:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        per_game = result.get("per_game") or []
        seeds = [game_result.get("seed") for game_result in per_game]
        unique_deals = len(seeds) == len(set(seeds))
        if not unique_deals:
            violations.append("result reuses at least one hidden deal seed")
        result_summary = result.get("score")
        expected_games = result.get("games")
        if expected_games != len(per_game):
            violations.append(
                f"result declares {expected_games} games but stores {len(per_game)}"
            )
        if set(range(1, expected_games + 1)) != games_seen:
            violations.append(
                "audit does not contain at least one event for every game"
            )

    return {
        "status": "pass" if not violations else "fail",
        "audit_file": str(audit_path),
        "result_file": str(result_path) if result_path else None,
        "records": len(records),
        "games_seen": sorted(game for game in games_seen if game is not None),
        "event_counts": dict(event_counts),
        "response_counts": dict(response_counts),
        "voluntary_passes": voluntary_passes,
        "special_finish_passes": special_finish_passes,
        "special_finish_nonpass_plays": special_finish_nonpass_plays,
        "unique_hidden_deal_per_game": unique_deals,
        "result_score": result_summary,
        "violations": violations,
        "scope_note": (
            "This proves the game protocol exposed only allowlisted public "
            "fields and that logged responses were legal. It cannot prove "
            "what an external model reasoned privately."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audit_file", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.audit_file, args.result)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    raise SystemExit(0 if report["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
