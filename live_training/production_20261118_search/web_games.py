"""Play real web-engine games and check the served move and table routing."""
from collections import Counter
import json
import logging
from pathlib import Path
import random
import sys

import torch

sys.path.insert(0, str(Path.cwd()))
from web_server import GameRoom, get_web_card_data, is_web_card_compatible
from web_temporal_player import production_model_status


def play(players, seed):
    random.seed(seed)
    room = GameRoom("RELEASE")
    room._continue_game_flow = room._start_turn_timer = room._stop_turn_timer = lambda: None
    for seat in range(players):
        ok, reason = room.add_player(f"Seat {seat}", None, is_ai=True)
        assert ok, reason
    assert room.start_game() and room.game_state.starting_player_count == players
    actor = room.ai_player.temporal_player
    prior_action = actor.choose_action
    search = actor.policy.public_search
    prior_score = search.score
    counts = Counter()

    def audit_score(decision, history, logits):
        result, next_history = prior_score(decision, history, logits)
        mask = torch.tensor(decision.legal_actions)
        counts["search_decisions"] += 1
        counts["changed_from_teacher"] += int(logits[0, mask].argmax() != result[0, mask].argmax())
        return result, next_history

    def audit_action(player, state, is_change_sort=False, drawn_card=None):
        action = prior_action(player, state, is_change_sort, drawn_card)
        counts["decisions"] += 1
        if is_change_sort:
            assert 50 <= action <= 53
            counts["suit_choices"] += 1
        elif drawn_card is not None:
            assert action in (drawn_card, 54)
            counts["draw_retries"] += 1
        elif action != 54:
            assert action in player.hand
            assert is_web_card_compatible(action, state.current_sort, state.current_true_number)
            if state.penalty_amount:
                assert get_web_card_data(action)["truenumber"] == state.current_true_number
        return action

    actor.choose_action = audit_action
    search.score = audit_score
    try:
        while not room.game_state.game_over:
            assert counts["turns"] < 3_000
            result = room.handle_ai_turn()
            assert result and result["success"], result
            counts["turns"] += 1
        if room.game_state.winner is None:
            assert not room.game_state.grab_deck and len(room.game_state.game_deck) <= 1
        else:
            assert sum(p.is_observer for p in room.game_state.players) == players - 1
        assert bool(counts["search_decisions"]) == (players <= 4)
        room.ai_player.reset_episode()
        assert not actor._memories
    finally:
        search.score = prior_score
    return {"players": players, "seed": seed, **counts}


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    torch.set_num_threads(1)
    games = [play(n, 3_450_000_000 + n * 10 + i)
             for n in (2, 3, 4, 5, 8) for i in range(2 if n <= 4 else 1)]
    for n in (2, 3, 4):
        for field in ("changed_from_teacher", "suit_choices", "draw_retries"):
            assert sum(g.get(field, 0) for g in games if g["players"] == n) > 0
    print(json.dumps({"status": "passed", "games": games, "model": production_model_status()}, indent=2))
