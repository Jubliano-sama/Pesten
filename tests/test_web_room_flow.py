from types import SimpleNamespace
from uuid import uuid4

import pytest

import web_server


@pytest.fixture
def table(monkeypatch):
    fake_ai = SimpleNamespace(reset_episode=lambda: None)
    monkeypatch.setattr(web_server, "AIPlayer", lambda: fake_ai)
    monkeypatch.setattr(web_server, "generate_room_code", lambda: uuid4().hex[:6].upper())
    host = web_server.socketio.test_client(web_server.app)
    guest = web_server.socketio.test_client(web_server.app)
    host.emit("create_room", {"player_name": "Host"})
    seat = event(host, "room_created")
    code = seat["room_code"]
    guest.emit("join_room", {"room_code": code, "player_name": "Guest"})
    guest_seat = event(guest, "joined_room")
    host.get_received()

    room = web_server.game_rooms[code]
    room.game_state.players.sort(key=lambda player: player.name != "Host")
    try:
        yield room, host, guest, seat, guest_seat
    finally:
        room._stop_turn_timer()
        for client in (host, guest):
            if client.is_connected():
                client.disconnect()
        with web_server.rooms_lock:
            web_server.game_rooms.pop(code, None)


def event(client, name):
    return next(packet["args"][0] for packet in client.get_received() if packet["name"] == name)


def configure_round(room, *, rank=6, top=5, host_hand=(3, 28), draw=(4, 12)):
    room.started = True
    state = room.game_state
    state.game_deck = [top]
    state.grab_deck = list(draw)
    state.current_sort = "harten"
    state.current_true_number = rank
    state.players[0].hand = list(host_hand)
    state.players[1].hand = [14, 15]
    room._update_turn_order()
    room._start_turn_timer()


def test_draw_stays_private_and_reload_resumes_the_draw_decision(table):
    room, host, guest, seat, _ = table
    configure_round(room)
    host.emit("play_card", {"room_code": room.room_code, "card_num": None})
    update = event(guest, "game_update")
    assert update["action"] == "draw_and_play_option"
    assert update["details"]["drawn_count"] == 1
    assert "card" not in update["details"]
    assert all("hand" not in player for player in update["game_state"]["players"])
    assert update["game_state"]["has_drawn_this_turn"]

    host.disconnect()
    resumed = web_server.socketio.test_client(web_server.app)
    try:
        resumed.emit("join_room", {"room_code": room.room_code, "resume_token": seat["resume_token"]})
        joined = event(resumed, "joined_room")
        assert joined["player_id"] == seat["player_id"]
        assert joined["game_state"]["has_drawn_this_turn"]
        assert joined["private_state"] == {"hand": [3, 28, 4], "playable_cards": [3, 4]}
        resumed.emit("play_card", {"room_code": room.room_code, "card_num": None})
        passed = event(resumed, "game_update")
        assert passed["action"] == "pass"
        assert not passed["details"].get("drawn_count")
        assert room.game_state.players[0].hand == [3, 28, 4]
        assert room.game_state.current_player_index == 1
    finally:
        resumed.disconnect()


def test_duplicate_name_and_other_seats_id_cannot_move(table):
    room, host, guest, seat, _ = table
    stranger = web_server.socketio.test_client(web_server.app)
    try:
        stranger.emit("join_room", {"room_code": room.room_code, "player_name": "Host"})
        assert "taken" in event(stranger, "error")["message"]
        stranger.emit("join_room", {"room_code": room.room_code, "resume_token": seat["player_id"]})
        assert event(stranger, "error")["code"] == "seat_expired"
        configure_round(room)
        guest.emit("play_card", {"room_code": room.room_code, "player_id": seat["player_id"], "card_num": 3})
        assert "Join this room" in event(guest, "error")["message"]
        assert room.game_state.players[0].socket_id is not None
        assert room.game_state.game_deck == [5]
    finally:
        stranger.disconnect()


def test_suit_timeout_clears_modal_and_rejected_actions_keep_timer(table):
    room, host, guest, _, _ = table
    configure_round(room, host_hand=(50, 28))
    args = {"room_code": room.room_code}
    host.emit("change_sort", {**args, "new_sort": "ruiten"})
    assert "no suit choice" in event(host, "error")["message"]
    host.emit("play_card", {**args, "card_num": 50})
    assert event(host, "game_update")["game_state"]["waiting_for_suit_choice"]
    timer = room.current_turn_timer

    host.emit("change_sort", {**args, "new_sort": "special"})
    assert event(host, "error")["message"] == "Invalid sort"
    host.emit("play_card", {**args, "card_num": None})
    assert "Choose the next suit" in event(host, "error")["message"]
    assert room.current_turn_timer is timer

    room._stop_turn_timer()
    with web_server.rooms_lock:
        room._auto_pass_player(room.game_state.players[0])
    timeout = event(guest, "player_auto_passed")
    assert timeout["drawn_count"] == 0
    update = timeout["game_state"]
    assert not update["waiting_for_suit_choice"]
    assert update["current_sort"] == "harten"
    assert update["current_player_index"] == 1
    assert update["turn_deadline_ms"]
    assert room.game_state.players[0].hand == [28]


def test_seven_starts_a_new_draw_opportunity(table):
    room, host, _, _, _ = table
    configure_round(room, draw=(6, 4))
    args = {"room_code": room.room_code}
    host.emit("play_card", {**args, "card_num": None})
    event(host, "game_update")
    host.emit("play_card", {**args, "card_num": 6})
    update = event(host, "game_update")["game_state"]
    assert update["current_player_index"] == 0
    assert not update["has_drawn_this_turn"]
    host.emit("play_card", {**args, "card_num": None})
    assert event(host, "game_update")["action"] == "draw_and_play_option"


def test_multiplayer_summary_remembers_who_went_out_first(table):
    room, host, guest, seat, _ = table
    configure_round(room, host_hand=(3,))
    room.add_player("Third", None)
    room.game_state.players[1].hand = [4]
    args = {"room_code": room.room_code}
    host.emit("play_card", {**args, "card_num": 3})
    first = event(host, "game_update")["game_state"]
    assert first["first_out"] == {"id": seat["player_id"], "name": "Host"}
    assert not first["game_over"]
    guest.get_received()

    guest.emit("play_card", {**args, "card_num": 4})
    last = event(guest, "game_update")["game_state"]
    assert last["game_over"]
    assert last["first_out"] == first["first_out"]


def test_penalty_and_timeout_report_actual_public_draw_count(table):
    room, host, guest, _, _ = table
    configure_round(room, rank=2, top=1, draw=(4, 12, 24))
    room.game_state.penalty_amount = 2
    host.emit("play_card", {"room_code": room.room_code, "card_num": None})
    penalty = event(guest, "game_update")
    assert penalty["details"] == {"success": True, "action": "penalty", "drawn_count": 2}
    assert penalty["game_state"]["players"][0]["hand_size"] == 4

    room._stop_turn_timer()
    player = room.game_state.players[1]
    with web_server.rooms_lock:
        room._auto_pass_player(player)
    timeout = event(guest, "player_auto_passed")
    assert timeout["acting_player"]["id"] == player.id
    assert timeout["drawn_count"] == 1


def test_ai_can_draw_and_play_without_a_net_hand_size_change(table):
    room, _, _, _, _ = table
    configure_round(room)
    player = room.game_state.players[0]
    player.is_ai = True
    room.ai_player.choose_action_with_drawn_card = lambda player, state, drawn: drawn
    result = room.play_card(player.id, None)
    assert result["action"] == "play"
    assert result["drawn_count"] == 1
    assert player.hand == [3, 28]
