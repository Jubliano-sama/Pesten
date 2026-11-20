#!/usr/bin/env python
"""
web_server.py

A Flask-based web server for multiplayer Pesten card game with AI support.
Supports room-based multiplayer, mixed human/AI games, and real-time updates via Socket.IO.
"""

import os
import random
import re
import secrets
import string
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from threading import Lock, Timer, current_thread
import logging

import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room

# Import game components
from supervisedAgent import StudentNet
from web_temporal_player import (
    WebTemporalPlayer, normalize_card, production_model_status,
)
from ai_observation import (
    ACTION_COUNT,
    COMPATIBLE_START,
    HAND_END,
    MODE_INDEX,
    OBSERVATION_SIZE,
    PASS_ACTION,
    SUIT_ACTION_END,
    SUIT_ACTION_START,
    populate_game_context,
)

# Custom card handling for web game
def get_web_card_data(card_num: int) -> Dict[str, Any]:
    """Get card data matching original card.py exactly"""
    # Validate card number range
    if not isinstance(card_num, int):
        logger.warning(f"Invalid card type: {type(card_num)}, expected int")
        return {"sort": "unknown", "truenumber": -1, "number": card_num}
    
    if card_num < 0 or card_num > 53:
        logger.warning(f"Invalid card number: {card_num}, expected 0-53")
        return {"sort": "unknown", "truenumber": -1, "number": card_num}
    
    sorts = ["harten", "ruiten", "schoppen", "klaveren", "special"]
    
    if card_num < 48:
        # Regular cards 0-47 (4 suits × 12 cards each)
        suit_index = card_num // 12
        value_index = card_num % 12
        true_number = value_index + 1
        
        return {
            "sort": sorts[suit_index],
            "truenumber": true_number,
            "number": card_num
        }
    elif card_num == 48:
        # Jack (special, truenumber 13) - overrides regular card logic
        return {
            "sort": sorts[4],  # special
            "truenumber": 13,
            "number": card_num
        }
    elif card_num == 49:
        # Joker (special, truenumber 0) - overrides regular card logic
        return {
            "sort": sorts[4],  # special
            "truenumber": 0,
            "number": card_num
        }
    elif card_num >= 50 and card_num <= 53:
        # Jack variants with specific suits (50=hearts, 51=diamonds, 52=spades, 53=clubs)
        suit_index = card_num - 50  # 0=hearts, 1=diamonds, 2=spades, 3=clubs
        return {
            "sort": sorts[suit_index],
            "truenumber": 13,
            "number": card_num
        }
    else:
        # Shouldn't reach here due to range check above, but safety fallback
        logger.error(f"Unexpected card number in valid range: {card_num}")
        return {
            "sort": "unknown",
            "truenumber": -1,
            "number": card_num
        }

def is_web_card_compatible(card_num: int, current_sort: str, current_true_number: int) -> bool:
    """Check if web card is compatible with current table state (basic compatibility, not penalty-aware)"""
    card_data = get_web_card_data(card_num)
    
    # Invalid cards are never compatible
    if card_data["truenumber"] == -1 or card_data["sort"] == "unknown":
        logger.warning(f"Invalid card {card_num} is not compatible with anything")
        return False
    
    # Validate current table state
    if not isinstance(current_true_number, int) or current_true_number < 0 or current_true_number > 13:
        logger.warning(f"Invalid current_true_number: {current_true_number}")
        return False
    
    valid_sorts = ["harten", "ruiten", "schoppen", "klaveren", "special"]
    if current_sort not in valid_sorts:
        logger.warning(f"Invalid current_sort: {current_sort}")
        return False
    
    # Jokers and Jacks are always compatible in normal play
    if card_data["truenumber"] == 0 or card_data["truenumber"] == 13:
        return True
    
    # Same number or same suit
    if card_data["truenumber"] == current_true_number or card_data["sort"] == current_sort:
        return True
    
    return False


def is_web_card_playable(card_num: int, game_state: "GameState") -> bool:
    """Apply the same penalty-aware legality used by the training engine."""
    if game_state.penalty_amount > 0:
        return (
            get_web_card_data(card_num)["truenumber"]
            == game_state.current_true_number
        )
    return is_web_card_compatible(
        card_num,
        game_state.current_sort,
        game_state.current_true_number,
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Generate or get secret key
def get_secret_key():
    """Get secret key from environment or generate a secure one"""
    secret_key = os.environ.get('FLASK_SECRET_KEY')
    if not secret_key:
        # Generate a secure random secret key
        secret_key = secrets.token_hex(32)
        logger.warning("No FLASK_SECRET_KEY environment variable found, generated new secret key")
        logger.info("Set FLASK_SECRET_KEY environment variable for production use")
    return secret_key

app = Flask(__name__)
app.config['SECRET_KEY'] = get_secret_key()
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state management
rooms_lock = Lock()
game_rooms: Dict[str, 'GameRoom'] = {}

@dataclass
class PlayerState:
    """Represents a player's state in the game"""
    id: str
    name: str
    hand: List[int]  # Card numbers
    is_ai: bool = False
    is_connected: bool = True
    socket_id: Optional[str] = None
    is_observer: bool = False  # Player finished game and is now observing
    consecutive_timeouts: int = 0  # Number of consecutive timeouts (reset when player acts)
    resume_token: str = field(default_factory=lambda: secrets.token_urlsafe(32), repr=False)

@dataclass
class GameState:
    """Represents the current game state"""
    players: List[PlayerState]
    current_player_index: int
    direction: int  # 1 = clockwise, -1 = counterclockwise
    current_sort: str
    current_true_number: int
    penalty_amount: int
    game_deck: List[int]  # Card numbers
    grab_deck: List[int]  # Card numbers
    game_over: bool
    winner: Optional[int]
    turn_count: int
    sorts_played: List[int]  # Count for each suit
    has_drawn_this_turn: bool = False  # Track if current player has drawn this turn
    waiting_for_suit_choice: bool = False  # True when waiting for human to choose suit
    turn_order: List[str] = None  # Player IDs in turn order from current player
    direction_just_changed: bool = False  # Flag for client to know direction just changed
    public_history: List[dict] = field(default_factory=list)
    starting_player_count: int = 0

class AIPlayer:
    """Promoted recurrent teacher, with the legacy MLP for oversized tables."""
    
    def __init__(self, model_path: str = "ppo_actor_100run3.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        self.fallback_reason = ""
        self.temporal_player = None
        self.temporal_load_error = ""
        self._reported_fallback_counts = set()
        try:
            self.temporal_player = WebTemporalPlayer()
        except Exception as error:
            self.temporal_load_error = f"{type(error).__name__}: {error}"
            logger.error("Promoted temporal AI is not ready: %s", self.temporal_load_error)
        
        # Try multiple model paths
        model_paths = [
            model_path,
            "ppo_actor_100run3.onnx",
            "ppo_actor_80run3.onnx",
            "models/" + model_path,
            os.path.join(os.path.dirname(__file__), model_path)
        ]
        
        # Load PyTorch model
        for path in model_paths:
            if not os.path.exists(path):
                continue
                
            try:
                self.model = StudentNet(
                    input_dim=OBSERVATION_SIZE,
                    output_dim=ACTION_COUNT,
                ).to(self.device)
                
                # Try loading with different map locations for compatibility
                try:
                    self.model.load_state_dict(torch.load(path, map_location=self.device))
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        # Fallback to CPU if CUDA issues
                        self.device = torch.device("cpu")
                        self.model = self.model.to(self.device)
                        self.model.load_state_dict(torch.load(path, map_location=self.device))
                    else:
                        raise e
                
                self.model.eval()
                self.model_loaded = True
                logger.info(f"Successfully loaded AI model from: {path}")
                logger.info(f"Using device: {self.device}")
                break
                
            except Exception as e:
                logger.warning(f"Failed to load model from {path}: {e}")
                continue
        
        if not self.model_loaded:
            self.fallback_reason = f"No valid model found in paths: {model_paths}"
            logger.error(f"Legacy MLP fallback is unavailable: {self.fallback_reason}")
            self.model = None

    def reset_episode(self):
        if self.temporal_player is not None:
            self.temporal_player.reset()

    def _temporal_action(self, player, game_state, *, is_change_sort=False, drawn_card=None):
        # Legacy test adapters and explicitly constructed old AI objects retain
        # their 121-feature API. Normal room instances always have this field.
        if not hasattr(self, "temporal_player"):
            return None
        if WebTemporalPlayer.supports(game_state):
            if self.temporal_player is None:
                raise RuntimeError(f"promoted temporal AI is unavailable: {self.temporal_load_error}")
            return self.temporal_player.choose_action(
                player, game_state, is_change_sort=is_change_sort, drawn_card=drawn_card,
            )
        count = sum(not other.is_observer for other in game_state.players)
        self.fallback_reason = f"{count} active players is outside the recurrent player's supported schema; using legacy MLP"
        if count not in self._reported_fallback_counts:
            logger.warning(self.fallback_reason)
            self._reported_fallback_counts.add(count)
        if self.model is None:
            raise RuntimeError(f"legacy MLP fallback is unavailable: {self.fallback_reason}")
        return None

    def get_observation(self, player: PlayerState, game_state: GameState) -> np.ndarray:
        """Create observation vector for AI player"""
        obs = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        
        # Player's hand (cards 0-49 at positions 0-49, compatible flags at 50-99)
        for card_num in player.hand:
            # Map Jack variants (50-53) to Jack (48) for AI observation
            obs_card_num = card_num
            if card_num >= 50 and card_num <= 53:
                obs_card_num = 48  # All Jack variants treated as generic Jack for AI
            
            if obs_card_num < HAND_END:  # Ensure valid range (0-49)
                obs[obs_card_num] += 1
                if is_web_card_playable(card_num, game_state):
                    obs[obs_card_num + COMPATIBLE_START] += 1

        player_idx = next(i for i, p in enumerate(game_state.players) if p.id == player.id)
        return populate_game_context(
            obs,
            sorts_played=game_state.sorts_played,
            hand_sizes=[len(other_player.hand) for other_player in game_state.players],
            player_index=player_idx,
            direction=game_state.direction,
        )

    def _is_compatible(self, card_num: int, current_sort: str, current_true_number: int) -> bool:
        """Check if card is compatible with current table state"""
        return is_web_card_compatible(card_num, current_sort, current_true_number)

    def get_action_mask(self, obs: np.ndarray, is_change_sort: bool = False) -> np.ndarray:
        """Get valid action mask"""
        mask = np.zeros(ACTION_COUNT, dtype=np.float32)
        
        if not is_change_sort:  # Play card mode
            # Check compatible cards (positions 50-99 in observation for cards 0-49)
            compatible_cards = []
            for card_num in range(HAND_END):  # Cards 0-49
                if obs[card_num + COMPATIBLE_START] >= 1:
                    mask[card_num] = 1
                    compatible_cards.append(card_num)
            mask[PASS_ACTION] = 1  # Can always pass
            logger.info(f"AI ACTION MASK: compatible cards {compatible_cards}, pass=True")
        else:  # Change sort mode
            for x in range(SUIT_ACTION_START, SUIT_ACTION_END):
                mask[x] = 1
            mask[PASS_ACTION] = 0  # Cannot pass when changing sort
            logger.info(f"AI SUIT MASK: suits 50-53 available, pass=False")
        
        return mask

    def choose_action(self, player: PlayerState, game_state: GameState, is_change_sort: bool = False) -> int:
        """Choose action using AI model"""
        temporal_action = self._temporal_action(player, game_state, is_change_sort=is_change_sort)
        if temporal_action is not None:
            return temporal_action
        if is_change_sort:
            logger.info(f"AI {player.name} received request to change sort after playing Jack or Joker penalty")
        else:
            logger.info(f"AI {player.name} received request to choose action on their turn")
            
        if self.model is None:
            return self._random_action(player, game_state, is_change_sort)
        
        obs = self.get_observation(player, game_state)
        if is_change_sort:
            obs[MODE_INDEX] = 1
        
        mask = self.get_action_mask(obs, is_change_sort)
        
        try:
            # PyTorch inference
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(obs_tensor).squeeze(0).cpu().numpy()
            
            # Apply mask and get action
            masked_logits = np.where(mask.astype(bool), logits, -1e9)
            action = np.argmax(masked_logits)
            
            # Map action back to actual card in hand if it's a Jack
            if action == 48 and not is_change_sort:
                # Find actual Jack variant in player's hand
                jack_variants = [card for card in player.hand if card >= 50 and card <= 53]
                if jack_variants:
                    action = jack_variants[0]  # Choose first available Jack variant
            
            if is_change_sort:
                logger.info(f"AI {player.name} chose to change sort to: {['hearts', 'diamonds', 'spades', 'clubs'][action - 50]}")
            else:
                logger.info(f"AI {player.name} chose action: {action} ({'pass' if action == 54 else f'play card {action}'})")
            return int(action)
            
        except Exception as e:
            logger.error(f"AI inference error: {e}")
            return self._random_action(player, game_state, is_change_sort)
    
    def choose_action_with_drawn_card(self, player: PlayerState, game_state: GameState, drawn_card: int) -> int:
        """Choose action considering a newly drawn card that could be played"""
        temporal_action = self._temporal_action(player, game_state, drawn_card=drawn_card)
        if temporal_action is not None:
            return temporal_action
        if self.model is None:
            return self._random_action_with_drawn_card(player, game_state, drawn_card)
        
        # The draw operation updates player.hand before this method is called,
        # so get_observation already contains the drawn card exactly once.
        obs = self.get_observation(player, game_state)

        # Training performed a normal policy decision after adding the drawn
        # card, then only played when the policy selected that exact card.
        mask = self.get_action_mask(obs)
        
        try:
            # PyTorch inference
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(obs_tensor).squeeze(0).cpu().numpy()
            
            # Apply mask and get action
            masked_logits = np.where(mask.astype(bool), logits, -1e9)
            action = np.argmax(masked_logits)
            
            # Map generic Jack back to the drawn card if that's what was chosen
            if action == 48 and drawn_card >= 50 and drawn_card <= 53:
                return drawn_card
            
            return int(action)
            
        except Exception as e:
            logger.error(f"AI inference error with drawn card: {e}")
            return self._random_action_with_drawn_card(player, game_state, drawn_card)

    def _random_action(self, player: PlayerState, game_state: GameState, is_change_sort: bool) -> int:
        """Fallback random action selection"""
        if is_change_sort:
            return random.randint(50, 53)  # Random suit
        
        # Find compatible cards
        compatible_cards = []
        for card_num in player.hand:
            if card_num <= 53 and is_web_card_playable(card_num, game_state):
                compatible_cards.append(card_num)
        
        if compatible_cards:
            return random.choice(compatible_cards)
        else:
            return 54  # Pass
    
    def _random_action_with_drawn_card(self, player: PlayerState, game_state: GameState, drawn_card: int) -> int:
        """Fallback random action selection including drawn card option"""
        # Find compatible cards in hand
        compatible_cards = []
        for card_num in player.hand:
            if card_num <= 53 and is_web_card_playable(card_num, game_state):
                compatible_cards.append(card_num)
        
        # 70% chance to play drawn card if it's compatible, otherwise random choice
        if drawn_card in compatible_cards and random.random() < 0.7:
            return drawn_card
        elif compatible_cards:
            return random.choice(compatible_cards)
        else:
            return 54  # Pass

class GameRoom:
    """Manages a single game room with multiple players"""
    
    def __init__(self, room_code: str):
        self.room_code = room_code
        self.game_state = GameState(
            players=[],
            current_player_index=0,
            direction=1,
            current_sort="harten",
            current_true_number=1,
            penalty_amount=0,
            game_deck=[],
            grab_deck=[],
            game_over=False,
            winner=None,
            turn_count=0,
            sorts_played=[0, 0, 0, 0, 0],
            has_drawn_this_turn=False,
            waiting_for_suit_choice=False,
            turn_order=[],
            direction_just_changed=False
        )
        self.ai_player = AIPlayer()
        self.started = False
        self.current_turn_timer: Optional[Timer] = None
        self.turn_timeout_seconds = 60
        self.timer_lock = Lock()
        self.turn_deadline_ms: Optional[int] = None
        self.ai_flow_running = False
        self.first_out: Optional[dict] = None
        
    def add_player(self, player_name: str, socket_id: str, is_ai: bool = False) -> tuple[bool, str]:
        """Add a player to the room. Returns (success, error_message)"""
        
        # Check for duplicate names (case-insensitive)
        for existing_player in self.game_state.players:
            if existing_player.name.lower() == player_name.lower():
                return False, f"Player name '{player_name}' is already taken in this room"
        
        # Check room capacity (max 32 players for reasonable gameplay)
        if len(self.game_state.players) >= 32:
            return False, "Room is full (maximum 32 players)"
        
        player = PlayerState(
            id=str(uuid.uuid4()),
            name=player_name,
            hand=[],
            is_ai=is_ai,
            socket_id=socket_id if not is_ai else None
        )
        
        self.game_state.players.append(player)
        return True, ""
    
    def remove_player(self, socket_id: str) -> bool:
        """Mark a player as disconnected (don't actually remove them)"""
        for player in self.game_state.players:
            if player.socket_id == socket_id:
                player.is_connected = False
                player.socket_id = None
                return True
        return False
    
    def start_game(self) -> bool:
        """Start the game"""
        if len(self.game_state.players) < 2 or self.started:
            return False
        
        self.started = True
        self._reset_game()
        return True
    
    def _reset_game(self):
        """Reset game state and deal cards"""
        self.game_state.public_history.clear()
        self.ai_player.reset_episode()
        # Randomize player order
        random.shuffle(self.game_state.players)
        logger.info(f"Player order randomized: {[p.name for p in self.game_state.players]}")
        
        # Each deck contains 54 cards. Reserve one card for the opening discard.
        num_players = len(self.game_state.players)
        self.game_state.starting_player_count = num_players
        cards_needed = num_players * 7
        num_decks = max(1, (cards_needed + 54) // 54)
        
        logger.info(f"Game setup: {num_players} players, {cards_needed} cards needed, using {num_decks} deck(s)")
        
        # Create multiple decks if needed
        all_cards = []
        for deck_num in range(num_decks):
            # Standard deck: 48 regular cards + 2 jokers + 4 suit-specific jacks
            deck_cards = list(range(48))  # 0-47 are regular cards (excluding card 48 which was generic Jack)
            deck_cards.append(49)  # Joker (card 49)
            deck_cards.append(49)  # Joker (card 49)
            deck_cards.append(50)  # Jack of hearts (card 50)
            deck_cards.append(51)  # Jack of diamonds (card 51) 
            deck_cards.append(52)  # Jack of spades (card 52)
            deck_cards.append(53)  # Jack of clubs (card 53)
            all_cards.extend(deck_cards)
        
        random.shuffle(all_cards)
        
        # Set aside a normal starting card before dealing.
        start_index = next(
            index for index, number in enumerate(all_cards)
            if get_web_card_data(number)["truenumber"] not in (13, 0, 1, 7, 8, 2)
        )
        start_card = all_cards.pop(start_index)
        start_data = get_web_card_data(start_card)
        self.game_state.game_deck = [start_card]
        self.game_state.current_sort = start_data["sort"]
        self.game_state.current_true_number = start_data["truenumber"]
        self.game_state.grab_deck = all_cards[cards_needed:]

        for index, player in enumerate(self.game_state.players):
            player.hand = all_cards[index * 7:(index + 1) * 7]

        # Reset other game state
        self.game_state.current_player_index = 0
        self.game_state.direction = 1
        self.game_state.penalty_amount = 0
        self.game_state.game_over = False
        self.game_state.winner = None
        self.first_out = None
        self.game_state.turn_count = 0
        self.game_state.sorts_played = [0, 0, 0, 0, 0]
        self.game_state.has_drawn_this_turn = False
        self.game_state.waiting_for_suit_choice = False
        
        # Reset all players' timeout counts
        for player in self.game_state.players:
            player.consecutive_timeouts = 0
            player.is_observer = False
        
        # Initialize turn order
        self._update_turn_order()
        
        # Start game flow (handles both AI and human first players)
        self._continue_game_flow()
    
    def play_card(self, player_id: str, card_num: Optional[int]) -> Dict[str, Any]:
        """Process a player's card play"""
        if self.game_state.game_over:
            return {"success": False, "error": "Game is over"}
        
        # Find player
        player_idx = None
        for i, player in enumerate(self.game_state.players):
            if player.id == player_id:
                player_idx = i
                break
        
        if player_idx is None:
            return {"success": False, "error": "Player not found"}
        
        if player_idx != self.game_state.current_player_index:
            player = self.game_state.players[player_idx] if player_idx is not None else None
            player_name = player.name if player else "Unknown"
            logger.warning(f"Player {player_name} tried to act on turn {self.game_state.current_player_index} (their index: {player_idx})")
            return {"success": False, "error": "Not your turn"}
        
        player = self.game_state.players[player_idx]
        logger.info(f"PLAY_CARD START: {player.name} (AI: {player.is_ai}) wants to play {card_num}, penalty_amount: {self.game_state.penalty_amount}")
        
        if player.is_observer or self.game_state.waiting_for_suit_choice:
            return {"success": False, "error": "Choose the next suit before playing"}

        if card_num is not None:
            if type(card_num) is not int or not 0 <= card_num <= 53:
                return {"success": False, "error": "Invalid card"}
            if card_num not in player.hand:
                return {"success": False, "error": "Card not in hand"}
            if not is_web_card_playable(card_num, self.game_state):
                return {"success": False, "error": "Card not compatible"}

        self._stop_turn_timer()
        player.consecutive_timeouts = 0
        if card_num is None:
            result = self._handle_pass(player)
        else:
            result = self._process_card_play(player, card_num, get_web_card_data(card_num))
        self._update_turn_order()
        return result

    def _handle_pass(self, player: PlayerState) -> Dict[str, Any]:
        """Handle when player passes"""
        if self.game_state.penalty_amount > 0:
            return self._apply_penalty(player)
        else:
            # Check if player has already drawn this turn
            if self.game_state.has_drawn_this_turn:
                # Player already drew this turn, just end turn
                self._next_turn()
                self._skip_observer_turns()
                return {"success": True, "action": "pass"}
            
            # Draw exactly one card (first and only draw this turn)
            drawn_card = self._draw_card(player)
            self.game_state.has_drawn_this_turn = True
            
            if drawn_card is not None:
                # Check if drawn card can be played immediately
                if is_web_card_compatible(drawn_card, self.game_state.current_sort, self.game_state.current_true_number):
                    if player.is_ai:
                        # AI makes a strategic choice about the drawn card
                        logger.info(f"AI {player.name} received request to decide whether to play drawn card {drawn_card}")
                        
                        try:
                            # Let AI decide whether to play the drawn card using special observation
                            ai_action = self.ai_player.choose_action_with_drawn_card(player, self.game_state, drawn_card)
                            
                            # If AI chooses to play the specific drawn card, play it
                            if ai_action == drawn_card:
                                card_data = get_web_card_data(drawn_card)
                                logger.info(f"AI {player.name} decided to play the drawn card {drawn_card}")
                                return self._process_card_play(player, drawn_card, card_data)
                            else:
                                logger.info(f"AI {player.name} decided not to play the drawn card {drawn_card}")
                                self._next_turn()
                                return {"success": True, "action": "pass"}
                        except Exception as e:
                            logger.error(f"AI decision error for {player.name}: {e}")
                            # Default to not playing the drawn card
                            self._next_turn()
                            return {"success": True, "action": "pass"}
                    else:
                        # Human gets option to play the drawn card
                        return {"success": True, "action": "draw_and_play_option", "card": drawn_card}
            
            # If no card was drawn or drawn card is not playable, end turn
            self._next_turn()
            # Auto-skip observer turns after pass
            self._skip_observer_turns()
            return {"success": True, "action": "pass"}
    
    def _apply_penalty(self, player: PlayerState) -> Dict[str, Any]:
        """Apply penalty to player"""
        for _ in range(self.game_state.penalty_amount):
            if self._draw_card(player) is None:
                break

        self.game_state.penalty_amount = 0
        if self.game_state.game_over:
            return {"success": True, "action": "penalty"}
        
        # Handle sort change for joker
        if self.game_state.current_true_number == 0:
            logger.info(f"JOKER PENALTY: player {player.name} (index {self.game_state.current_player_index}) received joker penalty, choosing suit")
            if player.is_ai:
                new_sort_action = self.ai_player.choose_action(player, self.game_state, is_change_sort=True)
                sorts = ["harten", "ruiten", "schoppen", "klaveren", "special"]
                new_sort = sorts[new_sort_action - 50]
                self.game_state.current_sort = new_sort
                self._record_suit_choice(player, new_sort)
                logger.info(f"JOKER AI CHOICE: {player.name} chose suit {new_sort}")
                self._next_turn()
                return {"success": True, "action": "penalty_and_sort_change", "new_sort": new_sort}
            else:
                # Don't advance turn - player needs to choose suit first
                logger.info(f"JOKER HUMAN CHOICE: {player.name} needs to choose suit")
                self.game_state.waiting_for_suit_choice = True
                return {"success": True, "action": "penalty_and_choose_sort"}
        
        self._next_turn()
        # Auto-skip observer turns after penalty
        self._skip_observer_turns()
        return {"success": True, "action": "penalty"}
    
    def _process_card_play(self, player: PlayerState, card_num: int, card_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a valid card play"""
        # Remove card from hand
        player.hand.remove(card_num)
        
        # Add to game deck
        self.game_state.game_deck.append(card_num)
        
        # Update game state
        self.game_state.current_sort = card_data["sort"]
        self.game_state.current_true_number = card_data["truenumber"]
        self.game_state.public_history.append({
            "turn": self.game_state.turn_count, "player_id": player.id,
            "action": "play", "card": normalize_card(card_num),
            "sort": "special" if card_data["truenumber"] == 13 else card_data["sort"],
        })
        
        # Update suits played
        sorts = ["harten", "ruiten", "schoppen", "klaveren", "special"]
        observed_sort = "special" if card_data["truenumber"] == 13 else card_data["sort"]
        if observed_sort in sorts:
            sort_index = sorts.index(observed_sort)
            self.game_state.sorts_played[sort_index] += 1
        
        # Check for win
        if len(player.hand) == 0:
            # Set player as observer instead of ending game
            player.is_observer = True
            if self.first_out is None:
                self.first_out = {"id": player.id, "name": player.name}
            logger.info(f"Player {player.name} wins and becomes observer")
            
            # Check if only one active player remains (game should end)
            active_players = [p for p in self.game_state.players if not p.is_observer]
            if len(active_players) <= 1:
                self.game_state.game_over = True
                self.game_state.winner = self.game_state.current_player_index
                logger.info(f"Game ends - only {len(active_players)} active player(s) remaining")
                return {"success": True, "action": "win"}
            
            # Don't return early - let the function continue to advance turns and skip observers
            result = {"success": True, "action": "win"}
        else:
            # Apply card effects
            result = self._apply_card_effects(card_data, player)
        
        if not result.get("skip_turn", False):
            old_player_index = self.game_state.current_player_index
            self._next_turn()
            logger.info(f"Turn advanced from player {old_player_index} to player {self.game_state.current_player_index}, penalty_amount: {self.game_state.penalty_amount}")
        else:
            logger.info(f"Turn NOT advanced due to skip_turn flag for card {card_data['truenumber']}, penalty_amount: {self.game_state.penalty_amount}")
        
        # Auto-skip observer turns after any turn advancement
        self._skip_observer_turns()
        
        # Reset timeout count for successful action
        player.consecutive_timeouts = 0
        
        return result
    
    def _apply_card_effects(self, card_data: Dict[str, Any], player: PlayerState) -> Dict[str, Any]:
        """Apply special card effects"""
        result = {"success": True, "action": "play", "card_effects": []}
        
        if card_data["truenumber"] == 7:  # Another turn
            result["card_effects"].append("another_turn")
            self.game_state.has_drawn_this_turn = False
            result["skip_turn"] = True
        elif card_data["truenumber"] == 8:  # Skip next player
            result["card_effects"].append("skip_player")
            # Skip one extra player (call _next_turn twice to skip active players)
            self._next_turn()  # Normal next turn
            self._next_turn()  # Skip one more active player
            result["skip_turn"] = True  # Don't call _next_turn() again
        elif card_data["truenumber"] == 1:  # Reverse direction
            result["card_effects"].append("reverse")
            self.game_state.direction *= -1
            # Set flag for client direction change animation (will be sent once and reset)
            self.game_state.direction_just_changed = True
            logger.info(f"🔄 JACK PLAYED: Set direction_just_changed = True, new direction = {self.game_state.direction}")
            # Update turn order after direction change
            self._update_turn_order()
        elif card_data["truenumber"] == 0:  # Joker - +5 penalty
            result["card_effects"].append("joker_penalty")
            self.game_state.penalty_amount += 5
            # The next player (who receives the penalty) should choose the suit
            # We'll handle this in the penalty application logic
        elif card_data["truenumber"] == 13:  # Jack - change sort
            result["card_effects"].append("change_sort")
            if player.is_ai:
                new_sort_action = self.ai_player.choose_action(player, self.game_state, is_change_sort=True)
                sorts = ["harten", "ruiten", "schoppen", "klaveren", "special"]
                new_sort = sorts[new_sort_action - 50]
                self.game_state.current_sort = new_sort
                self._record_suit_choice(player, new_sort)
                result["new_sort"] = new_sort
            else:
                result["choose_sort"] = True
                result["skip_turn"] = True  # Don't advance turn until suit is chosen
                self.game_state.waiting_for_suit_choice = True
        elif card_data["truenumber"] == 2:  # Two - +2 penalty
            result["card_effects"].append("two_penalty")
            self.game_state.penalty_amount += 2
        
        return result
    
    def _draw_card(self, player: PlayerState) -> Optional[int]:
        """Draw a card for the player"""
        if self.game_state.grab_deck:
            card_num = self.game_state.grab_deck.pop(0)
            player.hand.append(card_num)
            return card_num
        if len(self.game_state.game_deck) > 1:
            # Reshuffle game deck into grab deck (keep top card)
            self.game_state.grab_deck = self.game_state.game_deck[:-1]
            random.shuffle(self.game_state.grab_deck)
            self.game_state.game_deck = self.game_state.game_deck[-1:]
            
            if self.game_state.grab_deck:
                card_num = self.game_state.grab_deck.pop(0)
                player.hand.append(card_num)
                return card_num
            else:
                # Game ends in a tie - no more cards available
                logger.info("GAME TIE: No more cards available to draw, ending game in a tie")
                self._end_game_in_tie()
                return None
        
        # No cards available anywhere - game ends in tie
        logger.info("GAME TIE: No cards in grab deck and game deck too small, ending game in a tie")
        self._end_game_in_tie()
        return None

    def _record_suit_choice(self, player, new_sort):
        self.game_state.public_history.append({
            "turn": self.game_state.turn_count, "player_id": player.id,
            "action": "choose_sort", "sort": new_sort,
        })
    
    def _end_game_in_tie(self):
        """End the game in a tie and notify all players"""
        self.game_state.game_over = True
        self.game_state.winner = None  # None indicates a tie
        self._stop_turn_timer()  # Stop any active timers
        
        # Broadcast tie game to all players in the room
        socketio.emit('game_tied', {
            'message': 'Game ended in a tie - no more cards available!',
            'game_state': self.get_public_state()
        }, room=self.room_code)
        
        logger.info(f"Game in room {self.room_code} ended in a tie")
    
    def _next_turn(self):
        """Move to next player's turn, skipping observers"""
        if self.game_state.game_over:
            return
        original_index = self.game_state.current_player_index
        attempts = 0
        max_attempts = len(self.game_state.players)
        
        while attempts < max_attempts:
            self.game_state.current_player_index = (
                self.game_state.current_player_index + self.game_state.direction
            ) % len(self.game_state.players)
            
            current_player = self.game_state.players[self.game_state.current_player_index]
            if not current_player.is_observer:
                # Found an active player
                self.game_state.turn_count += 1
                # Reset draw flag for new turn
                self.game_state.has_drawn_this_turn = False
                # Don't reset direction_just_changed here - it should persist until after client gets the update
                # Update turn order for clients
                self._update_turn_order()
                return
            
            attempts += 1
        
        # This shouldn't happen if game logic is correct, but safety check
        logger.error("Could not find active player for next turn!")
        self.game_state.current_player_index = original_index
    
    def _update_turn_order(self):
        """Update turn_order with all active players starting from current player"""
        try:
            active_players = [p for p in self.game_state.players if not p.is_observer]
            if not active_players:
                self.game_state.turn_order = []
                return
            
            # Validate current player index
            if (self.game_state.current_player_index < 0 or 
                self.game_state.current_player_index >= len(self.game_state.players)):
                logger.error(f"Invalid current_player_index: {self.game_state.current_player_index}")
                self.game_state.turn_order = [p.id for p in active_players]
                return
            
            # Find current player in active players list
            current_player = self.game_state.players[self.game_state.current_player_index]
            current_index = -1
            for i, player in enumerate(active_players):
                if player.id == current_player.id:
                    current_index = i
                    break
            
            if current_index == -1:
                # Current player is observer, shouldn't happen but safety check
                logger.warning(f"Current player {current_player.name} not found in active players")
                self.game_state.turn_order = [p.id for p in active_players]
                return
            
            # Build turn order starting from current player
            turn_order = []
            player_count = len(active_players)
            for i in range(player_count):
                index = (current_index + (i * self.game_state.direction) + player_count) % player_count
                turn_order.append(active_players[index].id)
            
            self.game_state.turn_order = turn_order
        except Exception as e:
            logger.error(f"Error updating turn order: {e}")
            # Fallback: just use all active players in original order
            try:
                active_players = [p for p in self.game_state.players if not p.is_observer]
                self.game_state.turn_order = [p.id for p in active_players]
            except:
                self.game_state.turn_order = []
    
    def _skip_observer_turns(self):
        """Automatically skip observer turns until we reach an active player"""
        skipped_count = 0
        max_skips = len(self.game_state.players)
        
        while (skipped_count < max_skips and 
               self.game_state.current_player_index < len(self.game_state.players) and
               self.game_state.players[self.game_state.current_player_index].is_observer):
            
            logger.info(f"Skipping observer turn for player {self.game_state.current_player_index} ({self.game_state.players[self.game_state.current_player_index].name})")
            self._next_turn()
            skipped_count += 1
        
        if skipped_count > 0:
            logger.info(f"Skipped {skipped_count} observer turn(s), now at player {self.game_state.current_player_index}")
    
    def _start_turn_timer(self):
        """Keep one authoritative deadline for the current decision."""
        with self.timer_lock:
            if self.current_turn_timer or self.game_state.game_over:
                return
            if not 0 <= self.game_state.current_player_index < len(self.game_state.players):
                return
            current = self.game_state.players[self.game_state.current_player_index]
            if current.is_ai or current.is_observer:
                return
            self.turn_deadline_ms = int((time.time() + self.turn_timeout_seconds) * 1000)
            self.current_turn_timer = Timer(self.turn_timeout_seconds, self._handle_turn_timeout)
            self.current_turn_timer.daemon = True
            self.current_turn_timer.start()

    def _stop_turn_timer(self):
        with self.timer_lock:
            if self.current_turn_timer:
                self.current_turn_timer.cancel()
                self.current_turn_timer = None
            self.turn_deadline_ms = None

    def _handle_turn_timeout(self):
        with rooms_lock:
            with self.timer_lock:
                if self.current_turn_timer is not current_thread() or self.game_state.game_over:
                    return
                self.current_turn_timer = None
                self.turn_deadline_ms = None

            player = self.game_state.players[self.game_state.current_player_index]
            player.consecutive_timeouts += 1
            if player.consecutive_timeouts >= 3:
                self.remove_seat(player)
                self.publish("player_kicked", player_name=player.name)
                self.remove_empty_room()
            else:
                self._auto_pass_player(player)

    def remove_seat(self, player: PlayerState):
        """Remove a seat while preserving whose turn it is in either direction."""
        state = self.game_state
        index = state.players.index(player)
        was_current = index == state.current_player_index
        winner = state.players[state.winner] if state.winner is not None else None
        state.players.pop(index)
        state.grab_deck.extend(player.hand)
        state.winner = state.players.index(winner) if winner in state.players else None

        if was_current:
            self._stop_turn_timer()
            if state.waiting_for_suit_choice:
                self._record_suit_choice(player, "harten")
                state.current_sort = "harten"
                state.waiting_for_suit_choice = False
            state.has_drawn_this_turn = False
            if state.players:
                state.current_player_index = (
                    index - (state.direction == -1)
                ) % len(state.players)
        elif index < state.current_player_index:
            state.current_player_index -= 1

        if self.started and not state.game_over:
            active = [seat for seat in state.players if not seat.is_observer]
            if len(active) <= 1:
                state.game_over = True
                state.penalty_amount = 0
                state.winner = state.players.index(active[0]) if active else None
                self._stop_turn_timer()
            else:
                self._skip_observer_turns()
                self._continue_game_flow()
        self._update_turn_order()

    def remove_empty_room(self):
        if not any(not player.is_ai for player in self.game_state.players):
            self._stop_turn_timer()
            game_rooms.pop(self.room_code, None)

    def _auto_pass_player(self, player: PlayerState):
        """Resolve the same pending decision the player saw, then advance."""
        state = self.game_state
        if state.waiting_for_suit_choice or (state.penalty_amount and state.current_true_number == 0):
            state.current_sort = "harten"
            self._record_suit_choice(player, "harten")
        if state.penalty_amount:
            for _ in range(state.penalty_amount):
                if self._draw_card(player) is None:
                    break
        elif not state.waiting_for_suit_choice and not state.has_drawn_this_turn:
            self._draw_card(player)

        state.penalty_amount = 0
        state.waiting_for_suit_choice = False
        self._next_turn()
        self._continue_game_flow()
        self.publish("player_auto_passed", player_name=player.name,
                     consecutive_timeouts=player.consecutive_timeouts)

    def _continue_game_flow(self):
        state = self.game_state
        if state.game_over or not state.players:
            return
        current = state.players[state.current_player_index]
        if not current.is_ai:
            self._start_turn_timer()
        elif not self.ai_flow_running and not state.waiting_for_suit_choice:
            self.ai_flow_running = True
            socketio.start_background_task(self._run_ai_turns)

    def _run_ai_turns(self):
        while True:
            socketio.sleep(.7)
            with rooms_lock:
                state = self.game_state
                if (game_rooms.get(self.room_code) is not self or state.game_over or
                        state.waiting_for_suit_choice or not state.players):
                    self.ai_flow_running = False
                    return
                player = state.players[state.current_player_index]
                if not player.is_ai:
                    self.ai_flow_running = False
                    return
                try:
                    result = self.handle_ai_turn()
                    if result and not result.get("success"):
                        result = self.play_card(player.id, None)
                    if not result or not result.get("success"):
                        raise RuntimeError("No successful AI move")
                    self._continue_game_flow()
                    self.publish("ai_move", action=result["action"], details=result,
                                 acting_player={"name": player.name, "id": player.id})
                except Exception:
                    logger.exception("AI turn failed in room %s", self.room_code)
                    self.ai_flow_running = False
                    return

    def private_state(self, player: PlayerState):
        state = self.game_state
        playable = []
        if (self.started and not state.game_over and not state.waiting_for_suit_choice and
                not player.is_observer and state.players[state.current_player_index] is player):
            playable = [number for number in player.hand if is_web_card_playable(number, state)]
        return {"hand": list(player.hand), "playable_cards": playable}

    def publish(self, event, *, skip_sid=None, **details):
        socketio.emit(event, {"game_state": self.get_public_state(), **details},
                      to=self.room_code, skip_sid=skip_sid)
        for player in self.game_state.players:
            if player.socket_id and player.socket_id != skip_sid:
                socketio.emit("private_state", self.private_state(player), to=player.socket_id)

    def change_sort(self, player_id: str, new_sort: str) -> Dict[str, Any]:
        """Change the current sort"""
        player_idx = None
        for i, player in enumerate(self.game_state.players):
            if player.id == player_id:
                player_idx = i
                break
        
        if player_idx != self.game_state.current_player_index:
            return {"success": False, "error": "Not your turn"}
        
        if new_sort not in ("harten", "ruiten", "schoppen", "klaveren"):
            return {"success": False, "error": "Invalid sort"}

        self._stop_turn_timer()
        old_sort = self.game_state.current_sort
        self.game_state.current_sort = new_sort
        self._record_suit_choice(self.game_state.players[player_idx], new_sort)
        self.game_state.waiting_for_suit_choice = False
        old_player_index = self.game_state.current_player_index
        self._next_turn()
        
        # Auto-skip observer turns after suit change
        self._skip_observer_turns()
        
        # Reset timeout count for successful action and start timer for next player
        self.game_state.players[player_idx].consecutive_timeouts = 0
        
        logger.info(f"SUIT CHANGED: {old_sort} -> {new_sort} by player {player_idx}, turn advanced from {old_player_index} to {self.game_state.current_player_index}")
        
        return {"success": True, "new_sort": new_sort}
    
    def handle_ai_turn(self):
        """Handle AI player's turn"""
        if self.game_state.game_over:
            logger.info("AI TURN: Game over, returning None")
            return None
        
        current_player = self.game_state.players[self.game_state.current_player_index]
        if not current_player.is_ai:
            logger.info(f"AI TURN: Current player {self.game_state.current_player_index} ({current_player.name}) is not AI, returning None")
            return None
        
        logger.info(f"AI TURN START: Player {self.game_state.current_player_index} ({current_player.name})")
        logger.info(f"AI GAME STATE: table={self.game_state.current_true_number} of {self.game_state.current_sort}, penalty={self.game_state.penalty_amount}, waiting_for_suit={self.game_state.waiting_for_suit_choice}")
        logger.info(f"AI HAND: {current_player.hand} ({len(current_player.hand)} cards)")
        
        if self.game_state.penalty_amount > 0:
            # Check if AI can play matching penalty card
            action = self.ai_player.choose_action(current_player, self.game_state)
            logger.info(f"AI penalty action: {action} (card in hand: {action in current_player.hand if action <= 53 else 'pass/suit'})")
            if action <= 53:  # Playing a card (0-53)
                card_data = get_web_card_data(action)
                if card_data["truenumber"] == self.game_state.current_true_number:
                    return self.play_card(current_player.id, action)
            
            # Apply penalty if can't match
            return self.play_card(current_player.id, None)
        else:
            # Normal play
            action = self.ai_player.choose_action(current_player, self.game_state)
            logger.info(f"AI normal action: {action} (card in hand: {action in current_player.hand if action <= 53 else 'pass/suit'})")
            if action == 54:  # Pass
                return self.play_card(current_player.id, None)
            else:
                return self.play_card(current_player.id, action)
    
    def get_public_state(
        self,
        player_id: str = None,
        check_direction_changed: bool = False,
    ) -> Dict[str, Any]:
        """Get public game state for clients"""
        direction_changed_value = getattr(self.game_state, 'direction_just_changed', False)
        logger.info(
            "🌍 PUBLIC STATE: direction_changed = "
            f"{direction_changed_value}, checking = {check_direction_changed}"
        )
        
        # Only include direction_changed when the client explicitly checks, and
        # reset it after providing it.
        if check_direction_changed and direction_changed_value:
            logger.info(
                "🔄 CLIENT REQUESTED: Providing direction_changed = True "
                "and resetting flag"
            )
            self.game_state.direction_just_changed = False
        
        # Hide other players' hands
        public_players = []
        for i, player in enumerate(self.game_state.players):
            public_player = {
                "id": player.id,
                "name": player.name,
                "hand_size": len(player.hand),
                "is_ai": player.is_ai,
                "is_connected": player.is_connected,
                "is_observer": player.is_observer
            }
            
            # Show full hand only to the player themselves
            if player_id and player.id == player_id:
                public_player["hand"] = player.hand
            
            public_players.append(public_player)
        
        return {
            "room_code": self.room_code,
            "players": public_players,
            "current_player_index": self.game_state.current_player_index,
            "direction": self.game_state.direction,
            "current_sort": self.game_state.current_sort,
            "current_true_number": self.game_state.current_true_number,
            "penalty_amount": self.game_state.penalty_amount,
            "top_card": self.game_state.game_deck[-1] if self.game_state.game_deck else None,
            "game_over": self.game_state.game_over,
            "winner": self.game_state.winner,
            "first_out": self.first_out,
            "turn_count": self.game_state.turn_count,
            "turn_order": self.game_state.turn_order,
            "direction_changed": (
                direction_changed_value if check_direction_changed else False
            ),
            "started": self.started,
            "has_drawn_this_turn": self.game_state.has_drawn_this_turn,
            "waiting_for_suit_choice": self.game_state.waiting_for_suit_choice,
            "turn_deadline_ms": self.turn_deadline_ms,
            "grab_deck_size": len(self.game_state.grab_deck)
        }

def validate_player_name(name: str) -> tuple[bool, str]:
    """Validate player name and return (is_valid, error_message)"""
    if not isinstance(name, str) or not name.strip():
        return False, "Player name cannot be empty"
    
    name = name.strip()
    
    if len(name) < 2:
        return False, "Player name must be at least 2 characters long"
    
    if len(name) > 20:
        return False, "Player name cannot be longer than 20 characters"
    
    # Allow letters, numbers, spaces, and common symbols
    if not re.match(r'^[a-zA-Z0-9\s\-_.!@#$%^&*()]+$', name):
        return False, "Player name contains invalid characters"
    
    # Prevent names that are too similar to AI naming pattern
    if name.lower().startswith('ai '):
        return False, "Player name conflicts with AI naming pattern"
    if name.startswith('You!'):
        return False, "Player name conflicts with other players You Representaton"
    
    return True, ""

def generate_room_code() -> str:
    """Generate a unique 6-character room code"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

# Flask routes
@app.route('/health')
def health():
    try:
        return jsonify(production_model_status())
    except Exception as error:
        logger.error("Production model health check failed: %s", error)
        return jsonify({"status": "error", "error": f"{type(error).__name__}: {error}"}), 503

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/room/<room_code>')
def room(room_code):
    return render_template('game.html', room_code=room_code.upper())

# Socket.IO events
def room_identity(room, player):
    return {
        "room_code": room.room_code,
        "player_id": player.id,
        "resume_token": player.resume_token,
        "game_state": room.get_public_state(player.id),
        "private_state": room.private_state(player),
    }


def find_room(data):
    code = data.get("room_code")
    room = game_rooms.get(code.upper()) if isinstance(code, str) else None
    if room is None:
        emit("error", {"message": "Room not found"})
    return room


def find_seat(room, data=None):
    player = next((seat for seat in room.game_state.players if seat.socket_id == request.sid), None)
    if player and (not data or not data.get("player_id") or data["player_id"] == player.id):
        return player
    emit("error", {"message": "Join this room before taking an action"})
    return None


@socketio.on("create_room")
def handle_create_room(data):
    name = data.get("player_name", "")
    valid, error = validate_player_name(name)
    if not valid:
        emit("error", {"message": error})
        return

    with rooms_lock:
        code = generate_room_code()
        while code in game_rooms:
            code = generate_room_code()
        room = GameRoom(code)
        room.add_player(name.strip(), request.sid)
        game_rooms[code] = room
        join_room(code)
        emit("room_created", room_identity(room, room.game_state.players[-1]))


@socketio.on("join_room")
def handle_join_room(data):
    with rooms_lock:
        room = find_room(data)
        if room is None:
            return
        seats = room.game_state.players
        token = data.get("resume_token")
        player = next((seat for seat in seats if seat.socket_id == request.sid), None)
        if player is None and isinstance(token, str):
            player = next((seat for seat in seats if not seat.is_ai and seat.resume_token == token), None)
            if player is None:
                emit("error", {"message": "Your saved seat is no longer available", "code": "seat_expired"})
                return

        if player:
            old_socket = player.socket_id
            was_disconnected = not player.is_connected
            if old_socket and old_socket != request.sid:
                leave_room(room.room_code, sid=old_socket)
                socketio.emit("seat_moved", to=old_socket)
            player.socket_id = request.sid
            player.is_connected = True
            join_room(room.room_code)
            emit("joined_room", room_identity(room, player))
            if was_disconnected:
                room.publish("player_reconnected", player_name=player.name, skip_sid=request.sid)
            return

        if room.started:
            emit("error", {"message": "Game already started"})
            return

        name = data.get("player_name", "")
        valid, error = validate_player_name(name)
        if valid:
            valid, error = room.add_player(name.strip(), request.sid)
        if not valid:
            emit("error", {"message": error})
            return

        player = room.game_state.players[-1]
        join_room(room.room_code)
        emit("joined_room", room_identity(room, player))
        room.publish("player_joined", player_name=player.name, skip_sid=request.sid)


@socketio.on("add_ai_player")
def handle_add_ai_player(data):
    with rooms_lock:
        room = find_room(data)
        if room is None or not find_seat(room, data):
            return
        if room.started:
            emit("error", {"message": "Game already started"})
            return

        names = [
            "AI Jelle", "AI Luc", "AI Tadeema", "AI Ronald", "AI Ramon",
            "AI Remco", "AI Pim", "AI Lars", "AI Martin", "AI Levi",
            "AI Thomas", "AI Otger", "AI Dribbel", "AI Bartje", "AI Bowy",
            "AI Gertje", "AI Liesbeth",
        ]
        used = {player.name for player in room.game_state.players}
        available = [name for name in names if name not in used]
        name = random.choice(available) if available else f"AI {len(used) + 1}"
        success, error = room.add_player(name, None, is_ai=True)
        if not success:
            emit("error", {"message": error})
            return
        room.publish("ai_player_added", ai_name=name)


@socketio.on("start_game")
def handle_start_game(data):
    with rooms_lock:
        room = find_room(data)
        if room is None or not find_seat(room, data):
            return
        ready = sum(player.is_connected for player in room.game_state.players)
        if ready < 2 or not room.start_game():
            emit("error", {"message": "Cannot start game"})
            return
        room.publish("game_started")


@socketio.on("play_card")
def handle_play_card(data):
    with rooms_lock:
        room = find_room(data)
        if room is None:
            return
        player = find_seat(room, data)
        if player is None:
            return
        if not room.started:
            emit("error", {"message": "Game has not started"})
            return

        result = room.play_card(player.id, data.get("card_num"))
        if not result["success"]:
            emit("error", {"message": result["error"]})
            return

        # The drawn card is private until someone plays it.
        public_details = {key: value for key, value in result.items() if key != "card"}
        room._continue_game_flow()
        room.publish("game_update", action=result["action"], details=public_details,
                     acting_player={"name": player.name, "id": player.id})


@socketio.on("change_sort")
def handle_change_sort(data):
    with rooms_lock:
        room = find_room(data)
        if room is None:
            return
        player = find_seat(room, data)
        if player is None:
            return
        if room.game_state.game_over or not room.game_state.waiting_for_suit_choice:
            emit("error", {"message": "There is no suit choice pending"})
            return

        result = room.change_sort(player.id, data.get("new_sort"))
        if not result["success"]:
            emit("error", {"message": result["error"]})
            return
        room._continue_game_flow()
        room.publish("sort_changed", new_sort=result["new_sort"])


@socketio.on("leave_room")
def handle_leave_room(data):
    with rooms_lock:
        room = find_room(data)
        if room is None:
            return
        player = find_seat(room, data)
        if player is None:
            return
        room.remove_seat(player)
        leave_room(room.room_code)
        room.publish("player_left", player_name=player.name)
        room.remove_empty_room()
        return {"ok": True}


@socketio.on("disconnect")
def handle_disconnect():
    with rooms_lock:
        for room in game_rooms.values():
            player = next((seat for seat in room.game_state.players if seat.socket_id == request.sid), None)
            if player:
                player.is_connected = False
                player.socket_id = None
                room.publish("player_left", player_name=player.name, disconnected=True)


if __name__ == '__main__':
    debug_mode = os.environ.get("FLASK_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
    }
    socketio.run(
        app,
        debug=debug_mode,
        use_reloader=debug_mode,
        host="0.0.0.0",
        port=5000,
    )
