"""Public-information web adapter for the recurrent teacher and learned search."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from threading import Lock, RLock

import numpy as np
import torch

from temporal_observation import (
    DecisionInput, EventKind, MAX_PLAYERS, PASS_ACTION, PublicEventToken,
    build_draw_retry_legal_actions, build_legal_actions,
    build_private_observation, relative_index,
)


MODEL_NAME = "MinGRU 192x2 teacher epoch 4387, 24-suit mean logits"
SOURCE_CHECKPOINT_SHA256 = "f8f0aece2d759eadc60b3cbccca30c321c01da13322f7cae511a73c8da3a9f93"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models/best/puffer_min_gru_192x2_plateau_teacher_epoch4387_portable.pt"
_MODEL_LOCK = Lock()
_MODEL_CACHE = {}


def _digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_shared_model(model_path=None):
    """Cache immutable weights, checking current artifact integrity on each load."""
    from production_temporal_policy import ProductionSuitOrbitPolicy, load_production_policy

    path = Path(model_path or DEFAULT_MODEL_PATH).resolve()
    artifact_sha = _digest(path)
    manifest_sha = _digest(path.with_suffix(".json"))
    use_search = path == DEFAULT_MODEL_PATH.resolve()
    if use_search:
        from web_search import PROFILE_SHA256, WebSearch, verified_profile
        search_profile = verified_profile()
    key = (str(path), artifact_sha, manifest_sha, use_search)
    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            policy = ProductionSuitOrbitPolicy(load_production_policy(path))
            if use_search:
                policy.public_search = WebSearch(search_profile)
            policy.eval().requires_grad_(False)
            _MODEL_CACHE.clear()
            _MODEL_CACHE[key] = policy
        policy = _MODEL_CACHE[key]
    metadata = {
        "model_name": MODEL_NAME,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "runtime_artifact_sha256": artifact_sha,
        "runtime": "portable_pytorch", "device": "cpu", "precision": "float32",
        "profile": "teacher4387_orbit24", "orbit_size": 24,
        "aggregation": "mean_logits",
        "supported_active_players": [2, MAX_PLAYERS],
        "legacy_fallback": "raw MLP outside supported active player counts",
    }
    if use_search:
        metadata.update(
            model_name="MinGRU 192x2 with learned opponent hands and public rollouts",
            profile="teacher4387_orbit24_learned_search", profile_sha256=PROFILE_SHA256,
            deployment_authorization="user_requested", search_table_sizes=[2, 3, 4],
            search_samples=search_profile["samples"], search_horizon=search_profile["horizon"],
            search_trust=search_profile["trust"], evaluation_games=57_600,
            larger_table_policy="teacher4387_orbit24 for up to 13 active players",
        )
    else:
        metadata.update(qualification_gate_passed=False, deployment_authorization="user_override")
    return policy, metadata


def production_model_status():
    _, metadata = load_shared_model()
    return {"status": "ok", **metadata}


def normalize_card(number):
    number = int(number)
    if 50 <= number <= 53:
        return 48
    if not 0 <= number < 50:
        raise ValueError(f"invalid public/own card number: {number}")
    return number


@dataclass
class _SeatMemory:
    state: torch.Tensor
    cursor: int
    last_hand_sizes: tuple[int, ...]
    started: bool = False
    search_history: np.ndarray | None = None


class WebTemporalPlayer:
    """One room's independent recurrent memories around shared frozen weights."""

    def __init__(self, policy=None, *, model_path=None):
        if policy is None:
            self.policy, self.metadata = load_shared_model(model_path)
        else:
            self.policy = policy
            self.metadata = {"model_name": MODEL_NAME, "device": "cpu", "precision": "float32"}
        self._lock = RLock()
        self.reset()

    def reset(self):
        with self._lock:
            self._memories = {}
            self._roster = None
            self._game_identity = None
            self._history_start = 0

    @staticmethod
    def supports(game_state):
        active = sum(not player.is_observer for player in game_state.players)
        return 2 <= active <= MAX_PLAYERS

    def _decision(self, player, game_state, *, is_change_sort, drawn_card):
        active = [other for other in game_state.players if not other.is_observer]
        roster = tuple(other.id for other in active)
        if not 2 <= len(active) <= MAX_PLAYERS:
            raise ValueError("active player count is outside the temporal observation schema")
        if player.id not in roster:
            raise ValueError("an observer cannot request a temporal action")
        history = getattr(game_state, "public_history", ())
        if self._game_identity != id(game_state):
            self.reset()
            self._game_identity = id(game_state)
        if self._roster is not None and self._roster != roster:
            # Finishing/kicked players change relative seat identities. Start a
            # fresh public state instead of retaining differently indexed memory.
            self._memories.clear()
            self._history_start = len(history)
        self._roster = roster
        observer = roster.index(player.id)
        current_id = game_state.players[game_state.current_player_index].id
        if current_id not in roster:
            raise ValueError("current player is not active")
        current = roster.index(current_id)
        sizes = tuple(len(other.hand) for other in active)
        if player.id not in self._memories:
            self._memories[player.id] = _SeatMemory(
                self.policy.initial_state(batch_size=1), self._history_start, sizes,
            )
        memory = self._memories[player.id]
        predicted = list(memory.last_hand_sizes)
        events = []
        if not memory.started:
            events.append(PublicEventToken(EventKind.EPISODE_START))
        for item in history[memory.cursor:]:
            actor_id = item.get("player_id")
            if actor_id not in roster:
                continue
            actor = roster.index(actor_id)
            relative = relative_index(actor, observer, len(active), game_state.direction)
            if item.get("action") == "play":
                number = normalize_card(item["card"])
                events.append(PublicEventToken(
                    EventKind.PLAY, actor_relative=relative, card_id=number,
                    suit_id=number // 12 if number < 48 else 4, count=1,
                ))
                predicted[actor] -= 1
            elif item.get("action") == "choose_sort":
                events.append(PublicEventToken(
                    EventKind.CHOOSE_SUIT, actor_relative=relative,
                    suit_id=("harten", "ruiten", "schoppen", "klaveren").index(item["sort"]),
                ))
        for actor, (expected, actual) in enumerate(zip(predicted, sizes)):
            if actual > expected:
                events.append(PublicEventToken(
                    EventKind.DRAW_COUNT,
                    actor_relative=relative_index(actor, observer, len(active), game_state.direction),
                    count=actual - expected,
                ))
        memory.cursor = len(history)
        memory.last_hand_sizes = sizes
        memory.started = True
        own = tuple(normalize_card(number) for number in player.hand)
        # A suited web Jack is the generic special Jack for the trained actor.
        current_suit = ("special" if is_change_sort and game_state.current_true_number == 13
                        else game_state.current_sort)
        observation = build_private_observation(
            own_cards=own, current_suit=current_suit,
            current_rank=game_state.current_true_number, public_hand_sizes=sizes,
            observer_index=observer, current_player_index=current,
            direction=game_state.direction, penalty_amount=game_state.penalty_amount,
            public_sorts_played=game_state.sorts_played,
            change_suit_mode=is_change_sort,
        )
        legal = build_legal_actions(
            own_cards=own, current_suit=current_suit,
            current_rank=game_state.current_true_number,
            penalty_amount=game_state.penalty_amount, change_suit_mode=is_change_sort,
        )
        if drawn_card is not None and not is_change_sort:
            legal = build_draw_retry_legal_actions(normalize_card(drawn_card))
        return DecisionInput(observation, legal, tuple(events)), memory

    def choose_action(self, player, game_state, is_change_sort=False, drawn_card=None):
        from production_temporal_policy import encode_decision_token, encode_event_token

        with self._lock:
            decision, memory = self._decision(
                player, game_state, is_change_sort=is_change_sort, drawn_card=drawn_card,
            )
            # The live web rules permit finishing on a special card. This
            # public, legal immediate win differs from the training end rule.
            if not is_change_sort and len(player.hand) == 1:
                last_card = normalize_card(player.hand[0])
                if decision.legal_actions[last_card]:
                    return int(player.hand[0])
            with torch.inference_mode():
                for event in decision.public_events:
                    token = torch.tensor([encode_event_token(event)], dtype=torch.float32)
                    _, _, memory.state = self.policy(token, memory.state)
                token = torch.tensor([encode_decision_token(decision)], dtype=torch.float32)
                logits, _, memory.state = self.policy(token, memory.state)
                search = getattr(self.policy, "public_search", None)
                # Larger rooms can shrink to four active seats with a multi-deck shoe.
                original_count = getattr(game_state, "starting_player_count", 0) or len(game_state.players)
                if search is not None and original_count <= 4:
                    logits, memory.search_history = search.score(decision, memory.search_history, logits)
                legal = torch.tensor([decision.legal_actions], dtype=torch.bool)
                action = int(logits.masked_fill(~legal, torch.finfo(logits.dtype).min).argmax(-1).item())
                memory.state = memory.state.detach()
            if action == 48 and not is_change_sort:
                if drawn_card is not None and normalize_card(drawn_card) == 48:
                    return int(drawn_card)
                return next(int(number) for number in player.hand if normalize_card(number) == 48)
            return action
