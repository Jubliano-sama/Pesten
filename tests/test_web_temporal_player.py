from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

import production_temporal_policy
import web_server
import web_temporal_player
from temporal_observation import EventKind
from web_server import AIPlayer, GameRoom, GameState, PlayerState
from web_temporal_player import WebTemporalPlayer


class RecordingPolicy(torch.nn.Module):
    def __init__(self, scores=None):
        super().__init__()
        self.calls = []
        self.scores = scores or {54: 10}

    def initial_state(self, batch_size=1):
        return torch.zeros((2, batch_size, 192))

    def forward(self, tokens, state):
        self.calls.append((tokens.clone(), state.clone()))
        logits = torch.full((len(tokens), 55), -10.0)
        for action, score in self.scores.items():
            logits[:, action] = score
        return logits, torch.zeros((len(tokens), 1)), state + 1


def state_for(players, **overrides):
    data = dict(
        players=players, current_player_index=0, direction=1,
        current_sort="harten", current_true_number=6, penalty_amount=0,
        game_deck=[5], grab_deck=[], game_over=False, winner=None,
        turn_count=0, sorts_played=[0] * 5,
    )
    data.update(overrides)
    return GameState(**data)


def players():
    return [PlayerState("a", "AI", [0, 1], is_ai=True), PlayerState("b", "Other", [12, 24])]


def test_draw_retry_allows_only_the_drawn_card_or_pass():
    seats = players()
    policy = RecordingPolicy({0: 20, 1: 10, 54: 0})
    adapter = WebTemporalPlayer(policy)
    assert adapter.choose_action(seats[0], state_for(seats), drawn_card=1) == 1
    assert policy.calls[-1][0][0, 0] == 1
    assert policy.calls[-1][0][0, 1] == 1


def test_search_uses_per_seat_history_and_original_room_size():
    from types import SimpleNamespace

    seats = players()
    policy = RecordingPolicy({0: 20, 1: 10})
    seen = []

    def score(decision, history, logits):
        seen.append(history)
        ranked = logits.clone()
        ranked[0, 1] = 30
        return ranked, len(seen)

    policy.public_search = SimpleNamespace(score=score)
    adapter = WebTemporalPlayer(policy)
    room = state_for(seats, starting_player_count=2)
    assert adapter.choose_action(seats[0], room) == 1
    assert adapter.choose_action(seats[0], room) == 1
    assert WebTemporalPlayer(policy).choose_action(seats[0], room) == 1
    assert seen == [None, 1, None]

    # Earlier departures can leave precisely these same two seats in an eight-player game.
    larger = state_for(seats, starting_player_count=8)
    assert adapter.choose_action(seats[0], larger) == 0
    assert seen == [None, 1, None]


def test_orbit_draw_retry_masks_after_mean_and_keeps_each_permutation_memory():
    seats = players()
    core = RecordingPolicy({0: 100, 1: 50, 54: 0})
    policy = production_temporal_policy.ProductionSuitOrbitPolicy(core)
    adapter = WebTemporalPlayer(policy)
    state = state_for(seats)
    assert adapter.choose_action(seats[0], state, drawn_card=1) == 1
    assert adapter._memories["a"].state.shape == (2, 1, 24 * 192)
    assert core.calls[-1][0].shape == (24, 100)
    # Episode start and decision both update every independent orbit memory.
    assert torch.equal(adapter._memories["a"].state, torch.full((2, 1, 24 * 192), 2.0))


def test_web_jacks_map_to_generic_actor_card_and_back_to_exact_drawn_variant():
    seats = players()
    seats[0].hand = [50, 51, 3]
    policy = RecordingPolicy({48: 20, 52: 10, 54: 0})
    adapter = WebTemporalPlayer(policy)
    state = state_for(seats)
    assert adapter.choose_action(seats[0], state) == 50
    assert policy.calls[-1][0][0, 48] == 2
    assert adapter.choose_action(seats[0], state, drawn_card=51) == 51
    state.current_true_number = 13
    state.current_sort = "ruiten"
    assert adapter.choose_action(seats[0], state, is_change_sort=True) == 52
    decision = policy.calls[-1][0][0]
    assert decision[54] == 1  # Canonical current suit is special before Jack choice.
    assert decision[93] == 1


def test_public_history_reconstructs_draw_counts_without_draw_identities():
    seats = players()
    state = state_for(seats)
    adapter = WebTemporalPlayer(RecordingPolicy())
    first, _ = adapter._decision(seats[0], state, is_change_sort=False, drawn_card=None)
    assert [event.kind for event in first.public_events] == [EventKind.EPISODE_START]
    seats[1].hand = [24, 40]  # Public play removed 12, then a private draw restored size.
    state.public_history.append({"action": "play", "player_id": "b", "card": 12, "sort": "ruiten"})
    second, _ = adapter._decision(seats[0], state, is_change_sort=False, drawn_card=None)
    assert [event.kind for event in second.public_events] == [EventKind.PLAY, EventKind.DRAW_COUNT]
    assert second.public_events[0].card_id == 12
    assert second.public_events[1].count == 1
    assert second.public_events[1].card_id == -1
    assert all(event.card_id != 40 for event in second.public_events)


def test_memory_is_separate_for_seats_rooms_and_reset_rosters():
    seats = players()
    seats.append(PlayerState("c", "Third", [13, 25], is_ai=True))
    state = state_for(seats)
    shared = RecordingPolicy()
    adapter = WebTemporalPlayer(shared)
    other_room = WebTemporalPlayer(shared)
    adapter.choose_action(seats[0], state)
    state.current_player_index = 2
    adapter.choose_action(seats[2], state)
    assert set(adapter._memories) == {"a", "c"}
    assert adapter._memories["a"].state.data_ptr() != adapter._memories["c"].state.data_ptr()
    assert not other_room._memories
    seats[1].is_observer = True
    adapter.choose_action(seats[2], state)
    assert set(adapter._memories) == {"c"}
    assert adapter._memories["c"].last_hand_sizes == (2, 2)
    assert shared.calls[-1][0][0, 92] == 2  # Only active players enter observation.
    adapter.reset()
    assert not adapter._memories
    adapter.choose_action(seats[2], state)
    assert torch.equal(shared.calls[-2][1], torch.zeros((2, 1, 192)))


def test_full24_real_policy_memory_is_independent_for_rooms_seats_and_permutations():
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(4387)
        shared = production_temporal_policy.ProductionSuitOrbitPolicy(
            production_temporal_policy.ProductionTemporalPolicy(),
        ).eval().requires_grad_(False)
    seats = players()
    seats.append(PlayerState("c", "Third", [13, 25], is_ai=True))
    state = state_for(seats)
    adapter = WebTemporalPlayer(shared)
    other_room = WebTemporalPlayer(shared)
    adapter.choose_action(seats[0], state)
    a_state = adapter._memories["a"].state.clone()
    assert a_state.shape == (2, 1, 24 * 192)
    orbits = a_state.reshape(2, 24, 192)
    assert not torch.equal(orbits[:, 0], orbits[:, -1])
    state.current_player_index = 2
    adapter.choose_action(seats[2], state)
    other_room.choose_action(seats[2], state)
    assert torch.equal(adapter._memories["a"].state, a_state)
    assert torch.equal(adapter._memories["c"].state, other_room._memories["c"].state)
    assert adapter._memories["c"].state.data_ptr() != other_room._memories["c"].state.data_ptr()
    assert adapter._memories["a"].state.data_ptr() != adapter._memories["c"].state.data_ptr()
    adapter.reset()
    assert not adapter._memories
    assert set(other_room._memories) == {"c"}


def test_own_legal_last_special_card_finishes_under_existing_web_rules():
    seats = players()
    seats[0].hand = [49]
    policy = RecordingPolicy({54: 100})
    assert WebTemporalPlayer(policy).choose_action(seats[0], state_for(seats)) == 49
    assert not policy.calls
    seats[0].hand = [0]
    state = state_for(seats, current_true_number=2, penalty_amount=2)
    assert WebTemporalPlayer(policy).choose_action(seats[0], state) == 54


def test_opponent_cards_and_draw_pile_are_never_read():
    class PrivateHand:
        def __len__(self):
            return 3

        def __iter__(self):
            raise AssertionError("opponent cards were read")

        def __getitem__(self, key):
            raise AssertionError("opponent cards were read")

    seats = players()
    seats[1].hand = PrivateHand()
    state = state_for(seats)
    state.grab_deck = object()
    state.game_deck = object()
    assert WebTemporalPlayer(RecordingPolicy()).choose_action(seats[0], state) == 54


def test_global_policy_cache_rechecks_changed_artifacts_and_missing_files(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.write_bytes(b"valid")
    path.with_suffix(".json").write_text("{}")
    loaded = []

    def loader(model_path):
        if model_path.read_bytes() != b"valid":
            raise ValueError("weights mismatch")
        loaded.append(True)
        return RecordingPolicy()

    monkeypatch.setattr(production_temporal_policy, "load_production_policy", loader)
    monkeypatch.setattr(web_temporal_player, "_MODEL_CACHE", {})
    first, metadata = web_temporal_player.load_shared_model(path)
    second, _ = web_temporal_player.load_shared_model(path)
    assert first is second
    assert isinstance(first, production_temporal_policy.ProductionSuitOrbitPolicy)
    assert metadata["profile"] == "teacher4387_orbit24"
    assert metadata["orbit_size"] == 24
    assert metadata["aggregation"] == "mean_logits"
    assert metadata["qualification_gate_passed"] is False
    assert metadata["deployment_authorization"] == "user_override"
    assert len(loaded) == 1
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="mismatch"):
        web_temporal_player.load_shared_model(path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        web_temporal_player.load_shared_model(path)


def test_supported_tables_do_not_silently_fall_back_when_teacher_is_missing():
    ai = AIPlayer.__new__(AIPlayer)
    ai.temporal_player = None
    ai.temporal_load_error = "missing verified weights"
    seats = players()
    with pytest.raises(RuntimeError, match="missing verified weights"):
        ai.choose_action(seats[0], state_for(seats))


def test_oversized_tables_keep_explicit_legacy_mlp_path():
    ai = AIPlayer.__new__(AIPlayer)
    ai.temporal_player = WebTemporalPlayer(RecordingPolicy())
    ai.temporal_load_error = ""
    ai._reported_fallback_counts = set()
    ai.device = torch.device("cpu")
    ai.model = lambda observation: torch.zeros((1, 55))
    seats = [PlayerState(str(i), str(i), [0, 1], is_ai=True) for i in range(32)]
    assert ai.choose_action(seats[0], state_for(seats)) == 0
    assert "32 active players" in ai.fallback_reason
    assert not ai.temporal_player._memories


def test_health_reports_only_model_readiness_and_returns_503_on_failure(monkeypatch):
    client = web_server.app.test_client()
    metadata = {"status": "ok", "model_name": "teacher4387", "source_checkpoint_sha256": "source", "runtime_artifact_sha256": "runtime"}
    monkeypatch.setattr(web_server, "production_model_status", lambda: metadata)
    assert client.get("/health").get_json() == metadata

    def unavailable():
        raise ValueError("weights mismatch")

    monkeypatch.setattr(web_server, "production_model_status", unavailable)
    response = client.get("/health")
    assert response.status_code == 503
    assert response.get_json() == {"status": "error", "error": "ValueError: weights mismatch"}


def make_room(monkeypatch):
    fake_ai = SimpleNamespace(reset_episode=lambda: None, choose_action=lambda *args, **kwargs: 51)
    monkeypatch.setattr(web_server, "AIPlayer", lambda: fake_ai)
    room = GameRoom("TEST")
    room._start_turn_timer = lambda: None
    room._stop_turn_timer = lambda: None
    room._continue_game_flow = lambda: None
    return room


def test_room_records_jack_play_before_immediate_ai_suit_choice(monkeypatch):
    room = make_room(monkeypatch)
    seats = players()
    seats[0].hand = [50, 3]
    room.game_state = state_for(seats)
    captured = []

    def choose_suit(player, state, **kwargs):
        captured.extend(state.public_history)
        return 51

    room.ai_player.choose_action = choose_suit
    result = room.play_card("a", 50)
    assert result["success"]
    assert captured[0]["action"] == "play"
    assert captured[0]["card"] == 48
    assert captured[0]["sort"] == "special"
    assert [row["action"] for row in room.game_state.public_history] == ["play", "choose_sort"]


def test_room_records_human_and_timeout_suit_choices_and_resets_history(monkeypatch):
    room = make_room(monkeypatch)
    seats = players()
    seats[0].is_ai = False
    room.game_state = state_for(seats)
    assert room.change_sort("a", "ruiten")["success"]
    assert room.game_state.public_history[-1]["sort"] == "ruiten"
    room.game_state.current_player_index = 0
    room.game_state.current_true_number = 0
    room.game_state.penalty_amount = 1
    room.game_state.grab_deck = [4, 5]
    room._auto_pass_player(seats[0])
    assert room.game_state.public_history[-1]["sort"] == "harten"
    seats[0].is_observer = True
    resets = []
    room.ai_player.reset_episode = lambda: resets.append(True)
    room._reset_game()
    assert room.game_state.public_history == []
    assert not any(player.is_observer for player in room.game_state.players)
    assert resets == [True]


@pytest.mark.parametrize("player_count", [2, 4, 8])
def test_web_safe_decision_stream_matches_canonical_adapter_for_complete_games(player_count):
    import game
    from temporal_puffer_agent import PufferStyleTemporalNet, temporal_agent_factory

    torch.set_num_threads(1)
    torch.manual_seed(601)
    np.random.seed(2540917000 + player_count)
    random.seed(2540917000 + player_count)
    policy = PufferStyleTemporalNet(hidden_size=16).eval()
    factory = temporal_agent_factory(policy, deterministic=True)
    match = game.Game([factory] * player_count)
    seats = [PlayerState(str(i), str(i), [], is_ai=True) for i in range(player_count)]
    web_state = state_for(seats)
    adapter = WebTemporalPlayer(RecordingPolicy())
    checked = []

    def instrument(agent, seat):
        original = agent._decision

        def compare(*, change_suit_mode=False):
            expected = original(change_suit_mode=change_suit_mode)
            for index, current in enumerate(match.players):
                seats[index].hand = [held.number for held in current.mydeck.cards]
            for name in ("current_player_index", "direction", "current_sort", "current_true_number", "penalty_amount", "turn_count"):
                setattr(web_state, name, getattr(match, name))
            web_state.sorts_played = match.sorts_played.tolist()
            web_state.public_history = [
                {**item, "player_id": str(item["player"])} for item in match.public_history
            ]
            actual, _ = adapter._decision(
                seats[seat], web_state, is_change_sort=change_suit_mode,
                drawn_card=agent._retry_card_for_current_turn(),
            )
            assert actual == expected
            checked.append(True)
            return expected

        agent._decision = compare

    for seat, agent in enumerate(match.players):
        instrument(agent, seat)
    match.auto_simulate(max_turns=200)
    assert checked
