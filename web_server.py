#!/usr/bin/env python
"""
web_server.py

A Flask-based web server for multiplayer Pesten card game with AI support.
Supports room-based multiplayer, mixed human/AI games, and real-time updates via Socket.IO.
"""

import os
import json
import random
import re
import secrets
import string
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from threading import Lock, Timer
import logging

import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room, rooms

# Import game components
import card
import deck
from supervisedAgent import StudentNet

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

class AIPlayer:
    """AI Player using the trained PPO model"""
    
    def __init__(self, model_path: str = "ppo_actor_100run3.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        self.fallback_reason = ""
        
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
                self.model = StudentNet(input_dim=121, output_dim=55).to(self.device)
                
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
            logger.error(f"AI Model Loading Failed: {self.fallback_reason}")
            logger.info("AI will use random strategy as fallback")
            self.model = None

    def get_observation(self, player: PlayerState, game_state: GameState) -> np.ndarray:
        """Create observation vector for AI player"""
        obs = np.zeros(121, dtype=np.float32)
        
        # Player's hand (cards 0-49 at positions 0-49, compatible flags at 50-99)
        for card_num in player.hand:
            # Map Jack variants (50-53) to Jack (48) for AI observation
            obs_card_num = card_num
            if card_num >= 50 and card_num <= 53:
                obs_card_num = 48  # All Jack variants treated as generic Jack for AI
            
            if obs_card_num <= 49:  # Ensure valid range (0-49)
                obs[obs_card_num] += 1
                if self._is_compatible(card_num, game_state.current_sort, game_state.current_true_number):
                    obs[obs_card_num + 50] += 1  # Compatible flags start at position 50
        
        # Suits played (108-112)
        for i in range(5):
            obs[108 + i] = game_state.sorts_played[i]
        
        # Play mode flag (113)
        obs[113] = 0  # 0 = play card, 1 = change sort
        
        # Other players' hand sizes (114-117) - truncate to max 4 players in observation
        player_idx = next(i for i, p in enumerate(game_state.players) if p.id == player.id)
        for i in range(min(len(game_state.players), 4)):  # Truncate to max 4 players
            relative_idx = (player_idx + i) % len(game_state.players)
            obs[114 + i] = len(game_state.players[relative_idx].hand)
        
        # Player index and direction (118-119)
        obs[118] = player_idx
        obs[119] = game_state.direction
        
        return obs

    def _is_compatible(self, card_num: int, current_sort: str, current_true_number: int) -> bool:
        """Check if card is compatible with current table state"""
        return is_web_card_compatible(card_num, current_sort, current_true_number)

    def get_action_mask(self, obs: np.ndarray, is_change_sort: bool = False) -> np.ndarray:
        """Get valid action mask"""
        mask = np.zeros(55, dtype=np.float32)
        
        if not is_change_sort:  # Play card mode
            # Check compatible cards (positions 50-99 in observation for cards 0-49)
            compatible_cards = []
            for card_num in range(50):  # Cards 0-49
                if obs[card_num + 50] >= 1:  # Check compatible flag at position 50+card_num
                    mask[card_num] = 1
                    compatible_cards.append(card_num)
            mask[54] = 1  # Can always pass
            logger.info(f"AI ACTION MASK: compatible cards {compatible_cards}, pass=True")
        else:  # Change sort mode
            for x in range(50, 54):  # Suits 0-3
                mask[x] = 1
            mask[54] = 0  # Cannot pass when changing sort
            logger.info(f"AI SUIT MASK: suits 50-53 available, pass=False")
        
        return mask

    def choose_action(self, player: PlayerState, game_state: GameState, is_change_sort: bool = False) -> int:
        """Choose action using AI model"""
        if is_change_sort:
            logger.info(f"AI {player.name} received request to change sort after playing Jack or Joker penalty")
        else:
            logger.info(f"AI {player.name} received request to choose action on their turn")
            
        if self.model is None:
            return self._random_action(player, game_state, is_change_sort)
        
        obs = self.get_observation(player, game_state)
        if is_change_sort:
            obs[113] = 1
        
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
        if self.model is None:
            return self._random_action_with_drawn_card(player, game_state, drawn_card)
        
        # Create temporary observation with drawn card added to compatible cards
        obs = self.get_observation(player, game_state)
        
        # Temporarily add the drawn card to observation as if it's in hand
        obs_card_num = drawn_card
        if drawn_card >= 50 and drawn_card <= 53:
            obs_card_num = 48  # Map Jack variants to generic Jack for AI
        
        if obs_card_num <= 49:
            obs[obs_card_num] += 1  # Add to hand representation
            if self._is_compatible(drawn_card, game_state.current_sort, game_state.current_true_number):
                obs[obs_card_num + 50] += 1  # Add to compatible flags
        
        # Create action mask that includes the drawn card if compatible
        mask = np.zeros(55, dtype=np.float32)
        
        # Add drawn card to mask if compatible
        if self._is_compatible(drawn_card, game_state.current_sort, game_state.current_true_number):
            mask[drawn_card if drawn_card <= 49 else 48] = 1
            
        mask[54] = 1  # Can always pass
        
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
            if card_num <= 53 and self._is_compatible(card_num, game_state.current_sort, game_state.current_true_number):
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
            if card_num <= 53 and self._is_compatible(card_num, game_state.current_sort, game_state.current_true_number):
                compatible_cards.append(card_num)
        
        # Add drawn card if compatible
        if self._is_compatible(drawn_card, game_state.current_sort, game_state.current_true_number):
            compatible_cards.append(drawn_card)
        
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
        self.timer_lock = Lock()  # Synchronize timer operations
        
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
        # Randomize player order
        random.shuffle(self.game_state.players)
        logger.info(f"Player order randomized: {[p.name for p in self.game_state.players]}")
        
        # Calculate number of decks needed: ceil((players * 7) / 53)
        num_players = len(self.game_state.players)
        cards_needed = num_players * 7
        num_decks = max(1, (cards_needed + 52) // 53)  # Ceiling division, at least 1 deck (53 cards per deck)
        
        logger.info(f"Game setup: {num_players} players, {cards_needed} cards needed, using {num_decks} deck(s)")
        
        # Create multiple decks if needed
        all_cards = []
        for deck_num in range(num_decks):
            # Standard deck: 48 regular cards + 1 joker + 4 suit-specific jacks
            deck_cards = list(range(48))  # 0-47 are regular cards (excluding card 48 which was generic Jack)
            deck_cards.append(49)  # Joker (card 49)
            deck_cards.append(49)  # Joker (card 49)
            deck_cards.append(50)  # Jack of hearts (card 50)
            deck_cards.append(51)  # Jack of diamonds (card 51) 
            deck_cards.append(52)  # Jack of spades (card 52)
            deck_cards.append(53)  # Jack of clubs (card 53)
            all_cards.extend(deck_cards)
        
        random.shuffle(all_cards)
        
        # Deal 7 cards to each player
        card_index = 0
        for player in self.game_state.players:
            player.hand = all_cards[card_index:card_index + 7]
            card_index += 7
        
        # Set up game and grab decks
        self.game_state.grab_deck = all_cards[card_index:]
        
        # Find valid starting card
        while True:
            if not self.game_state.grab_deck:
                break
            start_card_num = self.game_state.grab_deck.pop(0)
            start_card_data = get_web_card_data(start_card_num)
            # Avoid special cards as starting card (no Jacks, Jokers, Aces, 7s, 8s, 2s)
            if start_card_data["truenumber"] not in (13, 0, 1, 7, 8, 2):
                self.game_state.game_deck = [start_card_num]
                self.game_state.current_sort = start_card_data["sort"]
                self.game_state.current_true_number = start_card_data["truenumber"]
                break
        
        # Reset other game state
        self.game_state.current_player_index = 0
        self.game_state.direction = 1
        self.game_state.penalty_amount = 0
        self.game_state.game_over = False
        self.game_state.winner = None
        self.game_state.turn_count = 0
        self.game_state.sorts_played = [0, 0, 0, 0, 0]
        
        # Reset all players' timeout counts
        for player in self.game_state.players:
            player.consecutive_timeouts = 0
        
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
        
        # Stop the turn timer since player is acting
        self._stop_turn_timer()
        
        # Handle pass
        if card_num is None:
            result = self._handle_pass(player)
            # Update turn order after pass
            self._update_turn_order()
            return result
        
        # Validate card is in hand
        if card_num not in player.hand:
            return {"success": False, "error": "Card not in hand"}
        
        # Validate card number
        if not isinstance(card_num, int) or card_num < 0 or card_num > 53:
            logger.error(f"Invalid card number {card_num} from player {player.name}")
            return {"success": False, "error": "Invalid card"}
        
        # Validate card compatibility
        card_data = get_web_card_data(card_num)
        
        # Check if card data is valid
        if card_data["truenumber"] == -1 or card_data["sort"] == "unknown":
            logger.error(f"Invalid card data for card {card_num} from player {player.name}")
            return {"success": False, "error": "Invalid card"}
        logger.info(f"Checking compatibility: card {card_num} ({card_data['truenumber']} of {card_data['sort']}) vs table {self.game_state.current_true_number} of {self.game_state.current_sort}, penalty: {self.game_state.penalty_amount}")
        
        if not is_web_card_compatible(card_num, self.game_state.current_sort, self.game_state.current_true_number):
            if self.game_state.penalty_amount == 0:
                logger.warning(f"Card {card_num} not compatible with {self.game_state.current_true_number} of {self.game_state.current_sort}")
                return {"success": False, "error": "Card not compatible"}
            else:
                logger.info(f"Card {card_num} doesn't match penalty card, applying penalty")
                result = self._apply_penalty(player)
                # Update turn order after penalty
                self._update_turn_order()
                return result
        
        # Additional penalty validation: during penalty, only matching numbers are allowed
        if self.game_state.penalty_amount > 0 and card_data['truenumber'] != self.game_state.current_true_number:
            logger.info(f"During penalty, card {card_num} (number {card_data['truenumber']}) doesn't match penalty card number {self.game_state.current_true_number}")
            result = self._apply_penalty(player)
            # Update turn order after penalty
            self._update_turn_order()
            return result
        
        # Play the card
        result = self._process_card_play(player, card_num, card_data)
        # Update turn order after any action
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
            self._draw_card(player)
        
        self.game_state.penalty_amount = 0
        
        # Handle sort change for joker
        if self.game_state.current_true_number == 0:
            logger.info(f"JOKER PENALTY: player {player.name} (index {self.game_state.current_player_index}) received joker penalty, choosing suit")
            if player.is_ai:
                new_sort_action = self.ai_player.choose_action(player, self.game_state, is_change_sort=True)
                sorts = ["harten", "ruiten", "schoppen", "klaveren", "special"]
                new_sort = sorts[new_sort_action - 50]
                self.game_state.current_sort = new_sort
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
        
        # Update suits played
        sorts = ["harten", "ruiten", "schoppen", "klaveren", "special"]
        if card_data["sort"] in sorts:
            sort_index = sorts.index(card_data["sort"])
            self.game_state.sorts_played[sort_index] += 1
        
        # Check for win
        if len(player.hand) == 0:
            # Set player as observer instead of ending game
            player.is_observer = True
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
        elif len(self.game_state.game_deck) > 1:
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
        """Start a timer for the current player's turn"""
        with self.timer_lock:
            self._stop_turn_timer_unsafe()  # Clear any existing timer
            
            if (self.game_state.current_player_index >= len(self.game_state.players) or 
                self.game_state.game_over):
                return
            
            current_player = self.game_state.players[self.game_state.current_player_index]
            
            # Only start timer for human players (not AI or observers)
            if not current_player.is_ai and not current_player.is_observer and current_player.is_connected:
                logger.info(f"Starting {self.turn_timeout_seconds}s timer for {current_player.name}")
                self.current_turn_timer = Timer(self.turn_timeout_seconds, self._handle_turn_timeout)
                self.current_turn_timer.start()
    
    def _stop_turn_timer(self):
        """Stop the current turn timer (thread-safe)"""
        with self.timer_lock:
            self._stop_turn_timer_unsafe()
    
    def _stop_turn_timer_unsafe(self):
        """Stop the current turn timer (NOT thread-safe - use only within timer_lock)"""
        if self.current_turn_timer:
            self.current_turn_timer.cancel()
            self.current_turn_timer = None
    
    def _handle_turn_timeout(self):
        """Handle when a player's turn times out"""
        with self.timer_lock:
            # Check if this timeout is still valid (game/player might have changed)
            if (self.game_state.game_over or 
                self.game_state.current_player_index >= len(self.game_state.players)):
                logger.info("Turn timeout triggered but game is over or invalid player index")
                return
            
            current_player = self.game_state.players[self.game_state.current_player_index]
            
            # Check if player is still the current player and not AI
            if current_player.is_ai or current_player.is_observer:
                logger.info(f"Turn timeout triggered for {current_player.name} but they are now AI/observer")
                return
            
            logger.info(f"Player {current_player.name} timed out")
            
            # Clear the timer since we're handling the timeout
            self.current_turn_timer = None
        
        # Increment consecutive timeouts
        current_player.consecutive_timeouts += 1
        
        # Check if player should be kicked (3 consecutive timeouts)
        if current_player.consecutive_timeouts >= 3:
            logger.info(f"Kicking player {current_player.name} for 3 consecutive timeouts")
            self._kick_player(current_player)
        else:
            # Auto-pass for the player
            logger.info(f"Auto-passing for {current_player.name} (timeout {current_player.consecutive_timeouts}/3)")
            self._auto_pass_player(current_player)
    
    def _kick_player(self, player: PlayerState):
        """Remove a player from the game"""
        logger.info(f"Removing player {player.name} from the game")
        
        # Send kick notification to the specific player first
        if player.id:  # Only send to human players
            socketio.emit('player_kicked', {
                'player_name': player.name,
                'reason': 'Three consecutive timeouts',
                'game_state': self.get_public_state()
            }, room=player.id)
        
        # Remove from players list
        self.game_state.players.remove(player)
        
        # Adjust current player index if needed
        if self.game_state.current_player_index >= len(self.game_state.players):
            self.game_state.current_player_index = 0
        
        # Broadcast player removal to remaining players
        socketio.emit('player_kicked', {
            'player_name': player.name,
            'reason': 'Three consecutive timeouts',
            'game_state': self.get_public_state()
        }, room=self.room_code)
        
        # Check if game should end
        active_players = [p for p in self.game_state.players if not p.is_observer]
        if len(active_players) <= 1:
            self.game_state.game_over = True
        
        # Continue with next player
        self._skip_observer_turns()
        self._start_turn_timer()
    
    def _auto_pass_player(self, player: PlayerState):
        """Automatically pass for a player who timed out"""
        old_player_index = self.game_state.current_player_index
        logger.info(f"AUTO-PASS: Player {player.name} timed out, current index: {old_player_index}")
        
        # Apply penalty if there is one, otherwise just pass
        if self.game_state.penalty_amount > 0:
            # Draw penalty cards
            for _ in range(self.game_state.penalty_amount):
                self._draw_card(player)
            
            # Handle joker suit choice if needed
            if self.game_state.current_true_number == 0:  # Joker penalty
                if player.is_ai:
                    # AI chooses suit
                    new_sort_action = self.ai_player.choose_action(player, self.game_state, is_change_sort=True)
                    sorts = ["harten", "ruiten", "schoppen", "klaveren", "special"]
                    new_sort = sorts[new_sort_action - 50]
                    self.game_state.current_sort = new_sort
                    result = {"success": True, "action": "penalty_and_sort_change", "new_sort": new_sort}
                    logger.info(f"AUTO-PASS: AI {player.name} chose suit {new_sort}")
                else:
                    # Human gets default suit (hearts)
                    self.game_state.current_sort = "harten"
                    result = {"success": True, "action": "penalty_and_sort_change", "new_sort": "harten"}
                    logger.info("AUTO-PASS: Human timed out, automatically chose hearts")
            else:
                result = {"success": True, "action": "penalty"}
            
            self.game_state.penalty_amount = 0
        else:
            # Normal pass - just draw one card
            self._draw_card(player)
            result = {"success": True, "action": "pass"}
        
        # Always advance turn after auto-pass (no option to play drawn card)
        self._next_turn()
        self._skip_observer_turns()
        
        new_player_index = self.game_state.current_player_index
        logger.info(f"AUTO-PASS: After pass/penalty, current index: {new_player_index} (was {old_player_index})")
        
        if new_player_index < len(self.game_state.players):
            current_player = self.game_state.players[new_player_index]
            logger.info(f"AUTO-PASS: Next player is {current_player.name} (AI: {current_player.is_ai}, Observer: {current_player.is_observer})")
        
        # Log the broadcast attempt
        logger.info(f"AUTO-PASS: Broadcasting player_auto_passed event to room {self.room_code}")
        
        # Send auto-pass event to the specific player first (before they potentially disconnect)
        if player.socket_id:
            logger.info(f"AUTO-PASS: Sending event to specific player {player.name} at socket {player.socket_id}")
            socketio.emit('player_auto_passed', {
                'player_name': player.name,
                'action': result.get('action', 'pass'),
                'consecutive_timeouts': player.consecutive_timeouts,
                'game_state': self.get_public_state(),
                'details': result,
                'acting_player': {'name': player.name, 'id': player.id}
            }, room=player.socket_id)
        
        # Broadcast the auto-pass to all players in the room
        logger.info(f"AUTO-PASS: Broadcasting to room {self.room_code} with {len(self.game_state.players)} players")
        socketio.emit('player_auto_passed', {
            'player_name': player.name,
            'action': result.get('action', 'pass'),
            'consecutive_timeouts': player.consecutive_timeouts,
            'game_state': self.get_public_state(),
            'details': result,
            'acting_player': {'name': player.name, 'id': player.id}
        }, room=self.room_code)
        
        logger.info(f"AUTO-PASS: Event broadcast completed")
        
        # Continue game flow: handle AI turns or start timer for human
        self._continue_game_flow()
    
    def _continue_game_flow_background(self):
        """Background task wrapper to continue game flow in proper thread context"""
        logger.info("BACKGROUND TASK: Starting _continue_game_flow_background")
        try:
            self._continue_game_flow()
            logger.info("BACKGROUND TASK: Completed _continue_game_flow_background")
        except Exception as e:
            logger.error(f"BACKGROUND TASK: Error in _continue_game_flow_background: {e}")
            import traceback
            traceback.print_exc()
    
    def _continue_game_flow(self):
        """Continue the game flow after a turn change - handle AI turns or start human timer"""
        if self.game_state.game_over:
            return
            
        # Handle AI turns
        ai_turn_count = 0
        max_ai_turns = 20  # Safety limit to prevent infinite loops
        
        logger.info(f"CONTINUE FLOW: game_over={self.game_state.game_over}, current_player={self.game_state.current_player_index}, is_ai={self.game_state.players[self.game_state.current_player_index].is_ai if self.game_state.current_player_index < len(self.game_state.players) else 'OUT_OF_BOUNDS'}, waiting_for_suit={self.game_state.waiting_for_suit_choice}")
        
        while (not self.game_state.game_over and 
               self.game_state.current_player_index < len(self.game_state.players) and
               self.game_state.players[self.game_state.current_player_index].is_ai and
               not self.game_state.players[self.game_state.current_player_index].is_observer and
               not self.game_state.waiting_for_suit_choice and
               ai_turn_count < max_ai_turns):
            
            ai_turn_count += 1
            logger.info(f"AI turn in continue flow {ai_turn_count}: Player {self.game_state.current_player_index}")
            
            # Get the AI player before the turn (in case turn advances)
            ai_player_before_turn = self.game_state.players[self.game_state.current_player_index]
            
            # No delay - process AI immediately
            
            ai_result = self.handle_ai_turn()
            logger.info(f"AI result in continue flow: {ai_result}")
            
            if ai_result and ai_result.get('success', False):
                public_state = self.get_public_state()
                socketio.emit('ai_move', {
                    'action': ai_result['action'],
                    'game_state': public_state,
                    'details': ai_result,
                    'acting_player': {'name': ai_player_before_turn.name, 'id': ai_player_before_turn.id}
                }, room=self.room_code)
                
                # Send updated private hands to human players
                for player in self.game_state.players:
                    if not player.is_ai and player.socket_id:
                        socketio.emit('private_state', {
                            'hand': player.hand
                        }, room=player.socket_id)
                
                # If AI won and became observer, the while loop condition will naturally fail
                # and we'll continue to the next iteration or exit the loop
            else:
                logger.error("AI turn returned None - breaking loop to prevent hang")
                break
        
        # Start timer for human player if it's their turn
        if (not self.game_state.game_over and 
            not self.game_state.waiting_for_suit_choice and
            self.game_state.current_player_index < len(self.game_state.players)):
            current_player = self.game_state.players[self.game_state.current_player_index]
            if not current_player.is_ai and not current_player.is_observer:
                logger.info(f"Starting timer for human player {current_player.name}")
                self._start_turn_timer()
    
    def change_sort(self, player_id: str, new_sort: str) -> Dict[str, Any]:
        """Change the current sort"""
        player_idx = None
        for i, player in enumerate(self.game_state.players):
            if player.id == player_id:
                player_idx = i
                break
        
        if player_idx != self.game_state.current_player_index:
            return {"success": False, "error": "Not your turn"}
        
        # Stop the turn timer since player is acting
        self._stop_turn_timer()
        
        sorts = ["harten", "ruiten", "schoppen", "klaveren", "special"]
        if new_sort not in sorts[:4]:
            return {"success": False, "error": "Invalid sort"}
        
        old_sort = self.game_state.current_sort
        self.game_state.current_sort = new_sort
        self.game_state.waiting_for_suit_choice = False
        old_player_index = self.game_state.current_player_index
        self._next_turn()
        
        # Auto-skip observer turns after suit change
        self._skip_observer_turns()
        
        # Reset timeout count for successful action and start timer for next player
        self.game_state.players[player_idx].consecutive_timeouts = 0
        self._start_turn_timer()
        
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
    
    def get_public_state(self, player_id: str = None) -> Dict[str, Any]:
        """Get public game state for clients"""
        direction_changed_value = getattr(self.game_state, 'direction_just_changed', False)
        logger.info(f"🌍 PUBLIC STATE: direction_changed = {direction_changed_value}")
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
            "turn_count": self.game_state.turn_count,
            "turn_order": self.game_state.turn_order,
            "direction_changed": getattr(self.game_state, 'direction_just_changed', False),
            "started": self.started,
            "grab_deck_size": len(self.game_state.grab_deck)
        }

def validate_player_name(name: str) -> tuple[bool, str]:
    """Validate player name and return (is_valid, error_message)"""
    if not name or not name.strip():
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
    if name.lower().startswith('ai ') or name.lower() == 'mechahhitler':
        return False, "Player name conflicts with AI naming pattern"
    if name.lower().startswith('You!'):
        return False, "Player name conflicts with other players You Representaton"
    
    return True, ""

def generate_room_code() -> str:
    """Generate a unique 6-character room code"""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/room/<room_code>')
def room(room_code):
    return render_template('game.html', room_code=room_code)

# Socket.IO events
@socketio.on('create_room')
def handle_create_room(data):
    """Create a new game room"""
    player_name = data.get('player_name', 'Anonymous')
    
    # Validate player name
    is_valid, error_msg = validate_player_name(player_name)
    if not is_valid:
        emit('error', {'message': f'Invalid player name: {error_msg}'})
        return
    
    player_name = player_name.strip()  # Clean up the name
    logger.info(f"Creating room for player '{player_name}', socket: {request.sid}")
    
    with rooms_lock:
        # Generate unique room code
        while True:
            room_code = generate_room_code()
            if room_code not in game_rooms:
                break
        
        logger.info(f"Generated room code: {room_code}")
        
        # Create room
        game_room = GameRoom(room_code)
        success, error_msg = game_room.add_player(player_name, request.sid)
        if not success:
            emit('error', {'message': error_msg})
            return
        game_rooms[room_code] = game_room
        
        logger.info(f"Room {room_code} created with {len(game_room.game_state.players)} players")
        logger.info(f"Current rooms: {list(game_rooms.keys())}")
        
        join_room(room_code)
        
        emit('room_created', {
            'room_code': room_code,
            'player_id': game_room.game_state.players[0].id,
            'game_state': game_room.get_public_state(game_room.game_state.players[0].id)
        })
        
        logger.info(f"Room creation complete for {room_code}, emitted to socket {request.sid}")

@socketio.on('join_room')
def handle_join_room(data):
    """Join an existing game room"""
    room_code = data.get('room_code', '').upper()
    player_name = data.get('player_name', 'Anonymous')
    
    # Validate player name
    is_valid, error_msg = validate_player_name(player_name)
    if not is_valid:
        emit('error', {'message': f'Invalid player name: {error_msg}'})
        return
    
    player_name = player_name.strip()  # Clean up the name
    logger.info(f"Player '{player_name}' trying to join room '{room_code}', socket: {request.sid}")
    logger.info(f"Available rooms: {list(game_rooms.keys())}")
    
    with rooms_lock:
        if room_code not in game_rooms:
            logger.error(f"Room '{room_code}' not found! Available: {list(game_rooms.keys())}")
            emit('error', {'message': 'Room not found'})
            return
        
        game_room = game_rooms[room_code]
        
        # Check if player is already in the room (reconnecting)
        existing_player = None
        for player in game_room.game_state.players:
            if player.socket_id == request.sid:
                existing_player = player
                break
        
        # Also check if a player with the same name exists (reconnecting after page navigation)
        if not existing_player:
            for player in game_room.game_state.players:
                if player.name == player_name and not player.is_ai:
                    logger.info(f"Reconnecting player {player_name} to room {room_code}")
                    # Update the socket_id for this player
                    player.socket_id = request.sid
                    player.is_connected = True
                    existing_player = player
                    break
        
        if existing_player:
            # Player is reconnecting
            join_room(room_code)
            logger.info(f"Player {player_name} successfully reconnected to room {room_code}")
            emit('joined_room', {
                'room_code': room_code,
                'player_id': existing_player.id,
                'game_state': game_room.get_public_state(existing_player.id)
            })
            return
        
        if game_room.started:
            emit('error', {'message': 'Game already started'})
            return
        
        success, error_msg = game_room.add_player(player_name, request.sid)
        if not success:
            emit('error', {'message': error_msg})
            return
        
        join_room(room_code)
        
        # Get the newly added player
        new_player = game_room.game_state.players[-1]
        
        emit('joined_room', {
            'room_code': room_code,
            'player_id': new_player.id,
            'game_state': game_room.get_public_state(new_player.id)
        })
        
        # Notify other players
        socketio.emit('player_joined', {
            'player_name': player_name,
            'game_state': game_room.get_public_state()
        }, room=room_code)

@socketio.on('add_ai_player')
def handle_add_ai_player(data):
    """Add an AI player to the room"""
    room_code = data.get('room_code', '').upper()
    
    logger.info(f"Adding AI player to room '{room_code}', socket: {request.sid}")
    logger.info(f"Available rooms: {list(game_rooms.keys())}")
    
    with rooms_lock:
        if room_code not in game_rooms:
            logger.error(f"Room '{room_code}' not found for AI add! Available: {list(game_rooms.keys())}")
            emit('error', {'message': 'Room not found'})
            return
        
        game_room = game_rooms[room_code]
        
        if game_room.started:
            emit('error', {'message': 'Game already started'})
            return
        
        ai_names = ["AI Jelle", "AI Luc", "AI Tadeema", "AI Ronald", "AI Ramon", "AI Remco", "AI Pim", "AI Lars", "AI Martin", "AI Levi", "AI Thomas", "AI Otger", "MechaHitler"]
        np.random.shuffle(ai_names)
        ai_name = ai_names[len([p for p in game_room.game_state.players if p.is_ai]) % len(ai_names)]
        
        success, error_msg = game_room.add_player(ai_name, None, is_ai=True)
        if not success:
            emit('error', {'message': error_msg})
            return
        
        socketio.emit('ai_player_added', {
            'ai_name': ai_name,
            'game_state': game_room.get_public_state()
        }, room=room_code)

@socketio.on('start_game')
def handle_start_game(data):
    """Start the game"""
    room_code = data.get('room_code', '').upper()
    
    with rooms_lock:
        if room_code not in game_rooms:
            emit('error', {'message': 'Room not found'})
            return
        
        game_room = game_rooms[room_code]
        
        if not game_room.start_game():
            emit('error', {'message': 'Cannot start game'})
            return
        
        socketio.emit('game_started', {
            'game_state': game_room.get_public_state()
        }, room=room_code)
        
        # Send private hand info to each player
        for player in game_room.game_state.players:
            if not player.is_ai and player.socket_id:
                logger.info(f"Sending private hand to {player.name}: {len(player.hand)} cards")
                socketio.emit('private_state', {
                    'hand': player.hand
                }, room=player.socket_id)

@socketio.on('play_card')
def handle_play_card(data):
    """Handle card play"""
    room_code = data.get('room_code', '').upper()
    player_id = data.get('player_id')
    card_num = data.get('card_num')  # None for pass
    
    logger.info(f"🎮 PLAY_CARD received: room={room_code}, player={player_id}, card={card_num}")
    
    # Process the human player's action first
    with rooms_lock:
        if room_code not in game_rooms:
            logger.error(f"❌ Room not found: {room_code}")
            emit('error', {'message': 'Room not found'})
            return
        
        game_room = game_rooms[room_code]
        logger.info(f"🎯 Calling play_card for player {player_id}")
        result = game_room.play_card(player_id, card_num)
        logger.info(f"📋 play_card result: {result}")
        
        if not result['success']:
            logger.info(f"❌ Emitting error: {result['error']}")
            emit('error', {'message': result['error']})
            return
        
        # Find the player who performed the action
        acting_player = None
        try:
            for player in game_room.game_state.players:
                if player.id == player_id:
                    acting_player = player
                    break
        except Exception as e:
            logger.error(f"Error finding acting player {player_id}: {e}")
            acting_player = None
        
        # Broadcast game update
        public_state = game_room.get_public_state()
        socketio.emit('game_update', {
            'action': result['action'],
            'game_state': public_state,
            'details': result,
            'acting_player': {'name': acting_player.name, 'id': acting_player.id} if acting_player else None
        }, room=room_code)
        
        # Reset direction change flag AFTER sending to clients
        if game_room.game_state.direction_just_changed:
            logger.info(f"🔄 AFTER UPDATE: Resetting direction_just_changed from True to False")
            game_room.game_state.direction_just_changed = False
        
        # Send updated private hands
        for player in game_room.game_state.players:
            if not player.is_ai and player.socket_id:
                logger.info(f"Sending updated private hand to {player.name}: {len(player.hand)} cards")
                socketio.emit('private_state', {
                    'hand': player.hand
                }, room=player.socket_id)
    
    # Handle AI turns outside the lock to prevent deadlock with other player requests
    ai_turn_count = 0
    max_ai_turns = 100  # Safety limit to prevent infinite loops
    
    logger.info(f"🤖 AI LOOP START: game_over={game_room.game_state.game_over}, current_player={game_room.game_state.current_player_index}, is_ai={game_room.game_state.players[game_room.game_state.current_player_index].is_ai}, waiting_for_suit={game_room.game_state.waiting_for_suit_choice}")
    
    while (not game_room.game_state.game_over and 
           game_room.game_state.current_player_index < len(game_room.game_state.players) and
           game_room.game_state.players[game_room.game_state.current_player_index].is_ai and
           not game_room.game_state.players[game_room.game_state.current_player_index].is_observer and
           not game_room.game_state.waiting_for_suit_choice and
           ai_turn_count < max_ai_turns):
            
            logger.info(f"🔄 AI LOOP iteration {ai_turn_count + 1}: processing AI turn")
            
            ai_turn_count += 1
            logger.info(f"AI turn {ai_turn_count}: Player {game_room.game_state.current_player_index}")
            
            # Get the AI player before the turn (in case turn advances)
            ai_player_before_turn = game_room.game_state.players[game_room.game_state.current_player_index]
            
            # Add 1-second delay for AI moves (non-blocking)
            socketio.sleep(2)
            
            ai_result = game_room.handle_ai_turn()
            logger.info(f"AI result: {ai_result}")
            
            if ai_result and ai_result.get('success', False):
                public_state = game_room.get_public_state()
                socketio.emit('ai_move', {
                    'action': ai_result['action'],
                    'game_state': public_state,
                    'details': ai_result,
                    'acting_player': {'name': ai_player_before_turn.name, 'id': ai_player_before_turn.id}
                }, room=room_code)
                
                # Reset direction change flag AFTER sending AI update to clients
                if game_room.game_state.direction_just_changed:
                    logger.info(f"🔄 AFTER AI UPDATE: Resetting direction_just_changed from True to False")
                    game_room.game_state.direction_just_changed = False
                
            elif ai_result and not ai_result.get('success', False):
                logger.error(f"AI turn failed: {ai_result.get('error', 'Unknown error')}")
                # Break the loop to prevent infinite retries
                break
            
            # Update private hands after AI move (only for successful moves)
            if ai_result and ai_result.get('success', False):
                for player in game_room.game_state.players:
                    if not player.is_ai and player.socket_id:
                        socketio.emit('private_state', {
                            'hand': player.hand
                        }, room=player.socket_id)
            else:
                logger.error("AI turn returned None - breaking loop to prevent hang")
                break
        
    logger.info(f"AI LOOP END: game_over={game_room.game_state.game_over}, current_player={game_room.game_state.current_player_index}, is_ai={game_room.game_state.players[game_room.game_state.current_player_index].is_ai if game_room.game_state.current_player_index < len(game_room.game_state.players) else 'INDEX_OUT_OF_BOUNDS'}, waiting_for_suit={game_room.game_state.waiting_for_suit_choice}, ai_turns={ai_turn_count}")
    
    if ai_turn_count >= max_ai_turns:
        logger.error(f"AI turn limit reached ({max_ai_turns}) - potential infinite loop prevented")
    
    # Start timer for human player if it's their turn and game is still active
    if (not game_room.game_state.game_over and 
        not game_room.game_state.waiting_for_suit_choice and
        game_room.game_state.current_player_index < len(game_room.game_state.players)):
        current_player = game_room.game_state.players[game_room.game_state.current_player_index]
        if not current_player.is_ai and not current_player.is_observer:
            logger.info(f"Starting timer for human player {current_player.name} after AI processing")
            game_room._start_turn_timer()

@socketio.on('change_sort')
def handle_change_sort(data):
    """Handle sort change"""
    room_code = data.get('room_code', '').upper()
    player_id = data.get('player_id')
    new_sort = data.get('new_sort')
    
    logger.info(f"SORT CHANGE REQUEST: player {player_id} wants to change suit to {new_sort} in room {room_code}")
    
    with rooms_lock:
        if room_code not in game_rooms:
            emit('error', {'message': 'Room not found'})
            return
        
        game_room = game_rooms[room_code]
        result = game_room.change_sort(player_id, new_sort)
        
        if not result['success']:
            logger.error(f"SORT CHANGE FAILED: {result['error']}")
            emit('error', {'message': result['error']})
            return
        
        logger.info(f"SORT CHANGE SUCCESS: broadcasted to room {room_code}")
        
        # Broadcast sort change
        socketio.emit('sort_changed', {
            'new_sort': new_sort,
            'game_state': game_room.get_public_state()
        }, room=room_code)
        
        # After suit change, check if AI should continue
        logger.info(f"CHECKING AI CONTINUATION: current_player={game_room.game_state.current_player_index}, is_ai={game_room.game_state.players[game_room.game_state.current_player_index].is_ai}, waiting_for_suit={game_room.game_state.waiting_for_suit_choice}")
        
        # Handle AI turns after suit change
        ai_turn_count = 0
        max_ai_turns = 20
        
        while (not game_room.game_state.game_over and 
               game_room.game_state.current_player_index < len(game_room.game_state.players) and
               game_room.game_state.players[game_room.game_state.current_player_index].is_ai and
               not game_room.game_state.players[game_room.game_state.current_player_index].is_observer and
               not game_room.game_state.waiting_for_suit_choice and
               ai_turn_count < max_ai_turns):
            
            ai_turn_count += 1
            logger.info(f"AI turn after suit change {ai_turn_count}: Player {game_room.game_state.current_player_index}")
            
            # Get the AI player before the turn (in case turn advances)
            ai_player_before_turn_2 = game_room.game_state.players[game_room.game_state.current_player_index]
            
            # Add 1-second delay for AI moves (non-blocking)
            socketio.sleep(2)
            
            ai_result = game_room.handle_ai_turn()
            logger.info(f"AI result after suit change: {ai_result}")
            
            if ai_result and ai_result.get('success', False):
                public_state = game_room.get_public_state()
                socketio.emit('ai_move', {
                    'action': ai_result['action'],
                    'game_state': public_state,
                    'details': ai_result,
                    'acting_player': {'name': ai_player_before_turn_2.name, 'id': ai_player_before_turn_2.id}
                }, room=room_code)
                
                # Update private hands after AI move
                for player in game_room.game_state.players:
                    if not player.is_ai and player.socket_id:
                        socketio.emit('private_state', {
                            'hand': player.hand
                        }, room=player.socket_id)
            elif ai_result and not ai_result.get('success', False):
                logger.error(f"AI turn failed after suit change: {ai_result.get('error', 'Unknown error')}")
                break
            else:
                logger.error("AI turn returned None after suit change - breaking loop")
                break

@socketio.on('leave_room')
def handle_leave_room(data):
    """Handle explicit room leaving"""
    room_code = data.get('room_code', '').upper()
    player_id = data.get('player_id')
    
    logger.info(f"Player {player_id} explicitly leaving room {room_code}")
    
    with rooms_lock:
        if room_code not in game_rooms:
            return
        
        game_room = game_rooms[room_code]
        
        # Remove player completely
        for i, player in enumerate(game_room.game_state.players):
            if player.id == player_id:
                logger.info(f"Removing player {player.name} from room {room_code}")
                # Stop timer if this was the current player
                if i == game_room.game_state.current_player_index:
                    game_room._stop_turn_timer()
                game_room.game_state.players.pop(i)
                # Adjust current player index if needed
                if game_room.game_state.current_player_index >= len(game_room.game_state.players):
                    game_room.game_state.current_player_index = 0
                # Start timer for next player if game is still running
                if not game_room.game_state.game_over and game_room.game_state.players:
                    game_room._skip_observer_turns()
                    game_room._start_turn_timer()
                break
        
        # Check if room should be deleted
        human_players = [p for p in game_room.game_state.players if not p.is_ai]
        if not human_players:
            logger.info(f"Deleting room {room_code} - player explicitly left and no humans remain")
            del game_rooms[room_code]
        else:
            socketio.emit('player_left', {
                'game_state': game_room.get_public_state()
            }, room=room_code)

@socketio.on('player_timeout')
def handle_player_timeout(data):
    """Handle client-side timeout notification"""
    room_code = data.get('room_code', '').upper()
    player_id = data.get('player_id')
    
    logger.info(f"CLIENT TIMEOUT: Player {player_id} in room {room_code} signaled timeout from client")
    
    with rooms_lock:
        if room_code not in game_rooms:
            emit('error', {'message': 'Room not found'})
            return
        
        game_room = game_rooms[room_code]
        
        # Find the player
        player = None
        for p in game_room.game_state.players:
            if p.id == player_id:
                player = p
                break
        
        if not player:
            logger.error(f"CLIENT TIMEOUT: Player {player_id} not found in room {room_code}")
            return
        
        # Check if it's actually this player's turn
        current_player = game_room.game_state.players[game_room.game_state.current_player_index]
        if current_player.id != player_id:
            logger.warning(f"CLIENT TIMEOUT: Player {player_id} sent timeout but it's not their turn (current: {current_player.name})")
            return
        
        logger.info(f"CLIENT TIMEOUT: Processing timeout for {player.name} on their turn")
        
        # Cancel any existing server-side timer to avoid race condition
        with game_room.timer_lock:
            if game_room.current_turn_timer:
                logger.info(f"CLIENT TIMEOUT: Cancelling server timer to avoid race condition")
                game_room.current_turn_timer.cancel()
                game_room.current_turn_timer = None
        
        # Process the auto-pass
        game_room._auto_pass_player(player)

@socketio.on('connect')
def handle_connect():
    """Handle new socket connection"""
    logger.info(f"New socket connection: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    """Handle player disconnect"""
    logger.info(f"Socket disconnected: {request.sid}")
    
    with rooms_lock:
        # Find player in any room and mark as disconnected
        for room_code, game_room in list(game_rooms.items()):
            for player in game_room.game_state.players:
                if player.socket_id == request.sid:
                    logger.info(f"Player {player.name} disconnected from room {room_code}, keeping room alive")
                    player.is_connected = False
                    player.socket_id = None
                    
                    # Don't delete rooms on disconnection - they persist for reconnection
                    # Rooms will be cleaned up when explicitly requested or on server restart
                    socketio.emit('player_left', {
                        'game_state': game_room.get_public_state()
                    }, room=room_code)
                    return

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)