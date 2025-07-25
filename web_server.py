#!/usr/bin/env python
"""
web_server.py

A Flask-based web server for multiplayer Pesten card game with AI support.
Supports room-based multiplayer, mixed human/AI games, and real-time updates via Socket.IO.
"""

import os
import json
import random
import string
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from threading import Lock
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
    else:
        return {
            "sort": "unknown",
            "truenumber": -1,
            "number": card_num
        }

def is_web_card_compatible(card_num: int, current_sort: str, current_true_number: int) -> bool:
    """Check if web card is compatible with current table state"""
    card_data = get_web_card_data(card_num)
    
    # Jokers and Jacks are always compatible
    if card_data["truenumber"] == 0 or card_data["truenumber"] == 13:
        return True
    
    # Same number or same suit
    if card_data["truenumber"] == current_true_number or card_data["sort"] == current_sort:
        return True
    
    return False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
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
    waiting_for_suit_choice: bool = False  # True when waiting for human to choose suit

class AIPlayer:
    """AI Player using the trained PPO model"""
    
    def __init__(self, model_path: str = "ppo_actor_100run3.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        
        # Load PyTorch model
        try:
            self.model = StudentNet(input_dim=121, output_dim=55).to(self.device)
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                logger.info(f"Loaded PyTorch model: {model_path}")
            else:
                logger.warning(f"No model file found at {model_path}, using random strategy")
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}, using random strategy")
            self.model = None

    def get_observation(self, player: PlayerState, game_state: GameState) -> np.ndarray:
        """Create observation vector for AI player"""
        obs = np.zeros(121, dtype=np.float32)
        
        # Player's hand (cards 0-49 at positions 0-49, compatible flags at 50-99)
        for card_num in player.hand:
            if card_num <= 49:  # Ensure valid range (0-49)
                obs[card_num] += 1
                if self._is_compatible(card_num, game_state.current_sort, game_state.current_true_number):
                    obs[card_num + 50] += 1  # Compatible flags start at position 50
        
        # Suits played (108-112)
        for i in range(5):
            obs[108 + i] = game_state.sorts_played[i]
        
        # Play mode flag (113)
        obs[113] = 0  # 0 = play card, 1 = change sort
        
        # Other players' hand sizes (114-117)
        player_idx = next(i for i, p in enumerate(game_state.players) if p.id == player.id)
        for i in range(len(game_state.players)):
            relative_idx = (player_idx + i) % len(game_state.players)
            if i < 4:  # Max 4 players
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
            
            logger.info(f"AI MODEL CHOICE: action={action}, is_change_sort={is_change_sort}")
            return int(action)
            
        except Exception as e:
            logger.error(f"AI inference error: {e}")
            return self._random_action(player, game_state, is_change_sort)

    def _random_action(self, player: PlayerState, game_state: GameState, is_change_sort: bool) -> int:
        """Fallback random action selection"""
        if is_change_sort:
            return random.randint(50, 53)  # Random suit
        
        # Find compatible cards
        compatible_cards = []
        for card_num in player.hand:
            if card_num <= 49 and self._is_compatible(card_num, game_state.current_sort, game_state.current_true_number):
                compatible_cards.append(card_num)
        
        if compatible_cards:
            return random.choice(compatible_cards)
        else:
            return 54  # Pass

class GameRoom:
    """Manages a single game room with multiple players"""
    
    def __init__(self, room_code: str, max_players: int = 4):
        self.room_code = room_code
        self.max_players = max_players
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
            waiting_for_suit_choice=False
        )
        self.ai_player = AIPlayer()
        self.started = False
        
    def add_player(self, player_name: str, socket_id: str, is_ai: bool = False) -> bool:
        """Add a player to the room"""
        if len(self.game_state.players) >= self.max_players:
            return False
        
        player = PlayerState(
            id=str(uuid.uuid4()),
            name=player_name,
            hand=[],
            is_ai=is_ai,
            socket_id=socket_id if not is_ai else None
        )
        
        self.game_state.players.append(player)
        return True
    
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
        # Create and shuffle deck to match deck.py standardDeck() exactly
        all_cards = list(range(50))  # 0-49 are regular cards (matching deck.py range(0,50))
        all_cards.append(49)  # Joker (card 49)
        all_cards.append(48)  # Jack 1 (card 48)
        all_cards.append(48)  # Jack 2 (card 48) 
        all_cards.append(48)  # Jack 3 (card 48)
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
            return {"success": False, "error": "Not your turn"}
        
        player = self.game_state.players[player_idx]
        logger.info(f"PLAY_CARD START: {player.name} (AI: {player.is_ai}) wants to play {card_num}, penalty_amount: {self.game_state.penalty_amount}")
        
        # Handle pass
        if card_num is None:
            return self._handle_pass(player)
        
        # Validate card is in hand
        if card_num not in player.hand:
            return {"success": False, "error": "Card not in hand"}
        
        # Validate card compatibility
        card_data = get_web_card_data(card_num)
        logger.info(f"Checking compatibility: card {card_num} ({card_data['truenumber']} of {card_data['sort']}) vs table {self.game_state.current_true_number} of {self.game_state.current_sort}, penalty: {self.game_state.penalty_amount}")
        
        if not is_web_card_compatible(card_num, self.game_state.current_sort, self.game_state.current_true_number):
            if self.game_state.penalty_amount == 0:
                logger.warning(f"Card {card_num} not compatible with {self.game_state.current_true_number} of {self.game_state.current_sort}")
                return {"success": False, "error": "Card not compatible"}
            elif card_data['truenumber'] != self.game_state.current_true_number:
                logger.info(f"Card {card_num} doesn't match penalty card, applying penalty")
                return self._apply_penalty(player)
        
        # Play the card
        return self._process_card_play(player, card_num, card_data)
    
    def _handle_pass(self, player: PlayerState) -> Dict[str, Any]:
        """Handle when player passes"""
        if self.game_state.penalty_amount > 0:
            return self._apply_penalty(player)
        else:
            # Draw a card
            drawn_card = self._draw_card(player)
            if drawn_card is not None:
                # Check if drawn card can be played immediately
                if is_web_card_compatible(drawn_card, self.game_state.current_sort, self.game_state.current_true_number):
                    if player.is_ai:
                        # AI makes a strategic choice about the drawn card
                        logger.info(f"AI drew compatible card {drawn_card}, making decision...")
                        
                        # Let AI decide whether to play the drawn card
                        # We'll temporarily add the card to their observation and let them choose

                        # TODO: only add the drawn card to the compatible cards in the observation and action mask temporarily

                        ai_action = self.ai_player.choose_action(player, self.game_state)
                        
                        # If AI chooses to play the specific drawn card, play it
                        if ai_action == drawn_card:
                            card_data = get_web_card_data(drawn_card)
                            logger.info(f"AI chose to play drawn card {drawn_card}")
                            return self._process_card_play(player, drawn_card, card_data)
                        else:
                            logger.info(f"AI chose not to play drawn card {drawn_card}")
                            self._next_turn()
                            return {"success": True, "action": "pass"}
                    else:
                        # Human gets option to play the drawn card
                        return {"success": True, "action": "draw_and_play_option", "card": drawn_card}
            
            self._next_turn()
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
            self.game_state.game_over = True
            self.game_state.winner = self.game_state.current_player_index
            return {"success": True, "action": "win"}
        
        # Apply card effects
        result = self._apply_card_effects(card_data, player)
        
        if not result.get("skip_turn", False):
            old_player_index = self.game_state.current_player_index
            self._next_turn()
            logger.info(f"Turn advanced from player {old_player_index} to player {self.game_state.current_player_index}, penalty_amount: {self.game_state.penalty_amount}")
        else:
            logger.info(f"Turn NOT advanced due to skip_turn flag for card {card_data['truenumber']}, penalty_amount: {self.game_state.penalty_amount}")
        
        return result
    
    def _apply_card_effects(self, card_data: Dict[str, Any], player: PlayerState) -> Dict[str, Any]:
        """Apply special card effects"""
        result = {"success": True, "action": "play", "card_effects": []}
        
        if card_data["truenumber"] == 7:  # Another turn
            result["card_effects"].append("another_turn")
            result["skip_turn"] = True
        elif card_data["truenumber"] == 8:  # Skip next player
            result["card_effects"].append("skip_player")
            # Skip one extra player (normal next turn + skip one more)
            self.game_state.current_player_index = (
                self.game_state.current_player_index + (2 * self.game_state.direction)
            ) % len(self.game_state.players)
            result["skip_turn"] = True  # Don't call _next_turn() again
        elif card_data["truenumber"] == 1:  # Reverse direction
            result["card_effects"].append("reverse")
            self.game_state.direction *= -1
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
        
        return None
    
    def _next_turn(self):
        """Move to next player's turn"""
        self.game_state.current_player_index = (
            self.game_state.current_player_index + self.game_state.direction
        ) % len(self.game_state.players)
        self.game_state.turn_count += 1
    
    def change_sort(self, player_id: str, new_sort: str) -> Dict[str, Any]:
        """Change the current sort"""
        player_idx = None
        for i, player in enumerate(self.game_state.players):
            if player.id == player_id:
                player_idx = i
                break
        
        if player_idx != self.game_state.current_player_index:
            return {"success": False, "error": "Not your turn"}
        
        sorts = ["harten", "ruiten", "schoppen", "klaveren", "special"]
        if new_sort not in sorts[:4]:
            return {"success": False, "error": "Invalid sort"}
        
        old_sort = self.game_state.current_sort
        self.game_state.current_sort = new_sort
        self.game_state.waiting_for_suit_choice = False
        old_player_index = self.game_state.current_player_index
        self._next_turn()
        
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
            logger.info(f"AI penalty action: {action} (card in hand: {action in current_player.hand if action < 50 else 'pass/suit'})")
            if action < 50:  # Playing a card (0-49)
                card_data = get_web_card_data(action)
                if card_data["truenumber"] == self.game_state.current_true_number:
                    return self.play_card(current_player.id, action)
            
            # Apply penalty if can't match
            return self.play_card(current_player.id, None)
        else:
            # Normal play
            action = self.ai_player.choose_action(current_player, self.game_state)
            logger.info(f"AI normal action: {action} (card in hand: {action in current_player.hand if action < 50 else 'pass/suit'})")
            if action == 54:  # Pass
                return self.play_card(current_player.id, None)
            else:
                return self.play_card(current_player.id, action)
    
    def get_public_state(self, player_id: str = None) -> Dict[str, Any]:
        """Get public game state for clients"""
        # Hide other players' hands
        public_players = []
        for i, player in enumerate(self.game_state.players):
            public_player = {
                "id": player.id,
                "name": player.name,
                "hand_size": len(player.hand),
                "is_ai": player.is_ai,
                "is_connected": player.is_connected
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
            "started": self.started,
            "grab_deck_size": len(self.game_state.grab_deck)
        }

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
    max_players = data.get('max_players', 4)
    
    logger.info(f"Creating room for player '{player_name}' with max {max_players} players, socket: {request.sid}")
    
    with rooms_lock:
        # Generate unique room code
        while True:
            room_code = generate_room_code()
            if room_code not in game_rooms:
                break
        
        logger.info(f"Generated room code: {room_code}")
        
        # Create room
        game_room = GameRoom(room_code, max_players)
        game_room.add_player(player_name, request.sid)
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
        
        if not game_room.add_player(player_name, request.sid):
            emit('error', {'message': 'Room is full'})
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
        
        ai_names = ["AI Jelle", "AI Luc", "AI Tadeema", "AI Ronald"]
        ai_name = ai_names[len([p for p in game_room.game_state.players if p.is_ai]) % len(ai_names)]
        
        if not game_room.add_player(ai_name, None, is_ai=True):
            emit('error', {'message': 'Room is full'})
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
    
    with rooms_lock:
        if room_code not in game_rooms:
            emit('error', {'message': 'Room not found'})
            return
        
        game_room = game_rooms[room_code]
        result = game_room.play_card(player_id, card_num)
        
        if not result['success']:
            emit('error', {'message': result['error']})
            return
        
        # Find the player who performed the action
        acting_player = None
        for player in game_room.game_state.players:
            if player.id == player_id:
                acting_player = player
                break
        
        # Broadcast game update
        public_state = game_room.get_public_state()
        socketio.emit('game_update', {
            'action': result['action'],
            'game_state': public_state,
            'details': result,
            'acting_player': {'name': acting_player.name, 'id': acting_player.id} if acting_player else None
        }, room=room_code)
        
        # Send updated private hands
        for player in game_room.game_state.players:
            if not player.is_ai and player.socket_id:
                logger.info(f"Sending updated private hand to {player.name}: {len(player.hand)} cards")
                socketio.emit('private_state', {
                    'hand': player.hand
                }, room=player.socket_id)
        
        # Handle AI turns
        ai_turn_count = 0
        max_ai_turns = 20  # Safety limit to prevent infinite loops
        
        logger.info(f"AI LOOP START: game_over={game_room.game_state.game_over}, current_player={game_room.game_state.current_player_index}, is_ai={game_room.game_state.players[game_room.game_state.current_player_index].is_ai}, waiting_for_suit={game_room.game_state.waiting_for_suit_choice}")
        
        while (not game_room.game_state.game_over and 
               game_room.game_state.players[game_room.game_state.current_player_index].is_ai and
               not game_room.game_state.waiting_for_suit_choice and
               ai_turn_count < max_ai_turns):
            
            ai_turn_count += 1
            logger.info(f"AI turn {ai_turn_count}: Player {game_room.game_state.current_player_index}")
            
            # Get the AI player before the turn (in case turn advances)
            ai_player_before_turn = game_room.game_state.players[game_room.game_state.current_player_index]
            
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
               game_room.game_state.players[game_room.game_state.current_player_index].is_ai and
               not game_room.game_state.waiting_for_suit_choice and
               ai_turn_count < max_ai_turns):
            
            ai_turn_count += 1
            logger.info(f"AI turn after suit change {ai_turn_count}: Player {game_room.game_state.current_player_index}")
            
            # Get the AI player before the turn (in case turn advances)
            ai_player_before_turn_2 = game_room.game_state.players[game_room.game_state.current_player_index]
            
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
                game_room.game_state.players.pop(i)
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