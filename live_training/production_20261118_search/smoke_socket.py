"""Exercise a private release-test room through the public Socket.IO interface."""
import argparse
import json
from queue import Empty, Queue
import time
from urllib.request import Request, urlopen

import socketio


def check(url):
    agent = {"User-Agent": "Mozilla/5.0", "Cache-Control": "no-cache"}
    with urlopen(Request(url + "/health", headers=agent), timeout=10) as response:
        health = json.load(response)
    assert health["profile_sha256"] == "03c8c111927fdfe6d3355fe47b6bc1d12e260ea31514622392e5de8aced08426"
    events = Queue()
    client = socketio.Client(reconnection=False)
    client.on("*", lambda name, data: events.put((name, data)))
    client.connect(url, headers=agent, wait_timeout=10)
    room = None
    human_moves = ai_replies = 0
    hand_received = waiting = False
    state = None
    try:
        client.emit("create_room", {"player_name": "Release check"})
        deadline = time.monotonic() + 55
        while time.monotonic() < deadline:
            try:
                name, data = events.get(timeout=1)
            except Empty:
                continue
            if name == "error":
                raise RuntimeError(data["message"])
            if name == "room_created":
                room = {"room_code": data["room_code"], "player_id": data["player_id"]}
                client.emit("add_ai_player", room)
            elif name == "ai_player_added":
                assert len(data["game_state"]["players"]) == 2
                client.emit("start_game", room)
            elif name == "private_state":
                hand_received |= bool(data["hand"])
            elif name in ("game_started", "game_update", "ai_move", "sort_changed"):
                state = data["game_state"]
                if name == "game_update" and data["acting_player"]["id"] == room["player_id"]:
                    assert data["details"]["success"]
                    human_moves += 1
                    waiting = False
                    if data["action"] == "penalty_and_choose_sort":
                        client.emit("change_sort", {**room, "new_sort": "harten"})
                        waiting = True
                if name == "sort_changed":
                    waiting = False
                if name == "ai_move":
                    assert data["acting_player"]["id"] != room["player_id"] and data["details"]["success"]
                    ai_replies += int(human_moves > 0)
            if state and hand_received:
                turn = state["players"][state["current_player_index"]]["id"]
                if ai_replies and (turn == room["player_id"] or state["game_over"]):
                    break
                if turn == room["player_id"] and not waiting:
                    client.emit("play_card", {**room, "card_num": None})
                    waiting = True
        assert room and hand_received and human_moves and ai_replies, "missing acknowledged human move or AI reply"
        return {"status": "passed", "profile_sha256": health["profile_sha256"], "room_code": room["room_code"],
                "transport": client.transport(), "private_hand_received": hand_received,
                "confirmed_human_moves": human_moves, "confirmed_ai_replies": ai_replies}
    finally:
        try:
            if room:
                client.call("leave_room", room, timeout=8)
        finally:
            client.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    print(json.dumps(check(parser.parse_args().url.rstrip("/")), indent=2))
