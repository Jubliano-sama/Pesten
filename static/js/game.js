// Game page JavaScript
const socket = io();

// Game state
let gameState = null;
let playerId = localStorage.getItem('playerId');
let roomCode = localStorage.getItem('roomCode') || document.getElementById('room-code').textContent;
let selectedCard = null;
let awaitingSortChoice = false;

// DOM elements
const lobbySection = document.getElementById('lobby-section');
const gameSection = document.getElementById('game-section');
const playersListElement = document.getElementById('players-list');
const addAiBtn = document.getElementById('add-ai-btn');
const startGameBtn = document.getElementById('start-game-btn');
const leaveGameBtn = document.getElementById('leave-game');
const playerHandElement = document.getElementById('player-hand');
const otherPlayersElement = document.getElementById('other-players');
const topCardElement = document.getElementById('top-card');
const currentTurnElement = document.getElementById('current-turn');
const gameDirectionElement = document.getElementById('game-direction');
const penaltyInfoElement = document.getElementById('penalty-info');
const penaltyAmountElement = document.getElementById('penalty-amount');
const passBtn = document.getElementById('pass-btn');
const sortModal = document.getElementById('sort-modal');
const logContent = document.getElementById('log-content');
const drawPileCount = document.getElementById('draw-pile-count');
const handCountElement = document.getElementById('hand-count');

// Initialize game - get room code from URL if not in localStorage
if (!roomCode) {
    const urlParts = window.location.pathname.split('/');
    roomCode = urlParts[urlParts.length - 1];
    if (roomCode) {
        localStorage.setItem('roomCode', roomCode);
    }
}

if (!roomCode) {
    window.location.href = '/';
}

// Event listeners
addAiBtn.addEventListener('click', () => {
    console.log('🤖 Adding AI player to room:', roomCode);
    socket.emit('add_ai_player', { room_code: roomCode });
});

startGameBtn.addEventListener('click', () => {
    socket.emit('start_game', { room_code: roomCode });
});

leaveGameBtn.addEventListener('click', () => {
    console.log('🚪 Leaving room explicitly');
    
    if (playerId && roomCode) {
        socket.emit('leave_room', {
            room_code: roomCode,
            player_id: playerId
        });
    }
    
    // Clear local storage and redirect
    localStorage.removeItem('playerId');
    localStorage.removeItem('roomCode');
    localStorage.removeItem('playerName');
    
    window.location.href = '/';
});

passBtn.addEventListener('click', () => {
    if (!awaitingSortChoice) {
        socket.emit('play_card', {
            room_code: roomCode,
            player_id: playerId,
            card_num: null // null means pass
        });
    }
});

// Sort modal handlers
document.querySelectorAll('.sort-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const newSort = btn.dataset.sort;
        socket.emit('change_sort', {
            room_code: roomCode,
            player_id: playerId,
            new_sort: newSort
        });
        sortModal.style.display = 'none';
        awaitingSortChoice = false;
    });
});

// Socket event handlers
socket.on('connect', () => {
    console.log('🔌 Connected to server');
    console.log('📍 Room code:', roomCode);
    console.log('👤 Player ID:', playerId);
    
    // Always try to join/rejoin the room
    const playerName = localStorage.getItem('playerName') || 'Player';
    console.log('🚪 Attempting to join room with name:', playerName);
    
    socket.emit('join_room', {
        room_code: roomCode,
        player_name: playerName
    });
});

socket.on('joined_room', (data) => {
    console.log('✅ Successfully joined room:', data);
    playerId = data.player_id;
    localStorage.setItem('playerId', playerId);
    updateGameState(data.game_state);
});

socket.on('player_joined', (data) => {
    console.log('👥 Player joined:', data);
    updateGameState(data.game_state);
    addLogEntry(`${data.player_name} joined the room`);
});

socket.on('ai_player_added', (data) => {
    console.log('🤖 AI player added:', data);
    updateGameState(data.game_state);
    addLogEntry(`${data.ai_name} joined the room`);
});

socket.on('game_started', (data) => {
    updateGameState(data.game_state);
    showGameSection();
    addLogEntry('Game started!', true);
});

socket.on('private_state', (data) => {
    console.log('🔒 Received private state update:', data);
    
    // Update player's private hand
    if (gameState) {
        const player = gameState.players.find(p => p.id === playerId);
        if (player) {
            console.log(`📤 Updating hand from ${player.hand?.length || 0} to ${data.hand?.length || 0} cards`);
            player.hand = data.hand;
            updatePlayerHand();
            updateHandCount();
        } else {
            console.warn('❌ Player not found in game state for private update');
        }
    } else {
        console.warn('❌ No game state available for private update');
    }
});

socket.on('game_update', (data) => {
    updateGameState(data.game_state);
    handleGameAction(data.action, data.details, data.acting_player);
});

socket.on('ai_move', (data) => {
    updateGameState(data.game_state);
    handleAIAction(data.action, data.details, data.acting_player);
});

socket.on('sort_changed', (data) => {
    updateGameState(data.game_state);
    addLogEntry(`Suit changed to ${getSuitSymbol(data.new_sort)}`);
});

socket.on('player_left', (data) => {
    updateGameState(data.game_state);
    addLogEntry('A player left the room');
});

socket.on('error', (data) => {
    console.error('❌ Server error:', data);
    showToast(data.message, 'error');
});

// Game logic functions
function updateGameState(newGameState) {
    console.log('🔄 Updating game state:', newGameState);
    
    // Preserve current player's hand if it exists
    let preservedHand = null;
    if (gameState && playerId) {
        const currentPlayer = gameState.players.find(p => p.id === playerId);
        if (currentPlayer && currentPlayer.hand) {
            preservedHand = [...currentPlayer.hand]; // Copy the array
            console.log('💾 Preserving hand with', preservedHand.length, 'cards');
        }
    }
    
    gameState = newGameState;
    
    // Restore preserved hand if new state doesn't have it or if it's empty
    if (preservedHand && playerId) {
        const newCurrentPlayer = gameState.players.find(p => p.id === playerId);
        if (newCurrentPlayer) {
            // If the new state has no hand data or empty hand, restore it
            if (!newCurrentPlayer.hand || newCurrentPlayer.hand.length === 0) {
                console.log('🔄 Restoring preserved hand due to missing/empty hand');
                newCurrentPlayer.hand = preservedHand;
            }
            // Also restore if the new state only has hand_size but no actual hand
            else if (Array.isArray(newCurrentPlayer.hand) && newCurrentPlayer.hand.length === 0 && newCurrentPlayer.hand_size > 0) {
                console.log('🔄 Restoring preserved hand due to hand_size mismatch');
                newCurrentPlayer.hand = preservedHand;
            }
        }
    }
    
    if (!gameState.started) {
        updateLobby();
    } else {
        updateGameInterface();
    }
}

function updateLobby() {
    playersListElement.innerHTML = '';
    
    gameState.players.forEach((player, index) => {
        const playerCard = document.createElement('div');
        playerCard.className = `player-card ${player.is_ai ? 'ai' : ''}`;
        
        playerCard.innerHTML = `
            <div class="player-name">${player.name}</div>
            <div class="player-status">${player.is_ai ? '🤖 AI Player' : '👤 Human'}</div>
        `;
        
        playersListElement.appendChild(playerCard);
    });
    
    // Update start button
    const minPlayers = 2;
    const canStart = gameState.players.length >= minPlayers;
    startGameBtn.disabled = !canStart;
    startGameBtn.textContent = canStart ? 'Start Game' : `Need ${minPlayers - gameState.players.length} more player(s)`;
}

function showGameSection() {
    lobbySection.style.display = 'none';
    gameSection.style.display = 'grid';
}

function updateGameInterface() {
    updateOtherPlayers();
    updatePlayerHand();
    updateTopCard();
    updateCurrentTurn();
    updateGameDirection();
    updatePenaltyInfo();
    updateDrawPile();
    updateHandCount();
}

function updateOtherPlayers() {
    otherPlayersElement.innerHTML = '';
    
    const currentPlayer = gameState.players.find(p => p.id === playerId);
    if (!currentPlayer) return;
    
    const otherPlayers = gameState.players.filter(p => p.id !== playerId);
    
    otherPlayers.forEach(player => {
        const playerIndex = gameState.players.indexOf(player);
        const isCurrentTurn = gameState.current_player_index === playerIndex;
        
        // Calculate turn order relative to current player
        const myIndex = gameState.players.indexOf(currentPlayer);
        const turnOrder = getTurnOrder(myIndex, playerIndex, gameState.direction, gameState.players.length);
        
        const playerElement = document.createElement('div');
        playerElement.className = `other-player ${isCurrentTurn ? 'current-player' : ''}`;
        
        const cardsHtml = Array(player.hand_size).fill(0).map(() => 
            '<div class="card-back-small">🂠</div>'
        ).join('');
        
        // Add turn order arrow
        const turnArrow = getTurnOrderArrow(turnOrder);
        
        playerElement.innerHTML = `
            <div class="other-player-info">
                <div class="other-player-name">
                    ${turnArrow} ${player.name} ${player.is_ai ? '🤖' : ''}
                    ${isCurrentTurn ? '⏰' : ''}
                </div>
                <div class="other-player-cards">${player.hand_size} cards</div>
            </div>
            <div class="other-player-hand">
                ${cardsHtml}
            </div>
        `;
        
        otherPlayersElement.appendChild(playerElement);
    });
}

function updatePlayerHand() {
    console.log(`🃏 Updating player hand for player ${playerId}`);
    
    playerHandElement.innerHTML = '';
    
    const currentPlayer = gameState.players.find(p => p.id === playerId);
    if (!currentPlayer) {
        console.warn(`❌ Current player not found: ${playerId}`);
        return;
    }
    
    if (!currentPlayer.hand) {
        console.warn(`❌ Player hand is undefined for ${currentPlayer.name}`);
        return;
    }
    
    console.log(`👤 Player ${currentPlayer.name} has ${currentPlayer.hand.length} cards:`, currentPlayer.hand);
    
    currentPlayer.hand.forEach(cardNum => {
        const cardElement = createCardElement(cardNum);
        
        // Check if card is playable
        if (isCardPlayable(cardNum)) {
            cardElement.classList.add('playable');
        }
        
        cardElement.addEventListener('click', () => {
            if (gameState.current_player_index !== gameState.players.indexOf(currentPlayer)) {
                showToast("It's not your turn!", 'error');
                return;
            }
            
            if (!isCardPlayable(cardNum)) {
                showToast("This card cannot be played right now", 'error');
                return;
            }
            
            if (awaitingSortChoice) {
                showToast("Please choose a suit first", 'error');
                return;
            }
            
            // Clear any previous selection
            document.querySelectorAll('.card.selected').forEach(card => {
                card.classList.remove('selected');
            });
            
            // Select this card
            cardElement.classList.add('selected');
            selectedCard = cardNum;
            
            // Play the card
            socket.emit('play_card', {
                room_code: roomCode,
                player_id: playerId,
                card_num: cardNum
            });
        });
        
        playerHandElement.appendChild(cardElement);
    });
}

function createCardElement(cardNum) {
    const cardElement = document.createElement('div');
    cardElement.className = 'card';
    
    const cardData = getCardData(cardNum);
    cardElement.classList.add(cardData.suit);
    
    cardElement.innerHTML = `
        <div class="card-suit">${cardData.symbol}</div>
        <div class="card-value">${cardData.value}</div>
        <span style="display:none;" data-card-num="${cardNum}"></span>
    `;
    
    return cardElement;
}

function updateTopCard() {
    if (!gameState.top_card) return;
    
    const cardData = getCardData(gameState.top_card);
    topCardElement.className = `card ${cardData.suit}`;
    topCardElement.innerHTML = `
        <div class="card-suit">${cardData.symbol}</div>
        <div class="card-value">${cardData.value}</div>
        <span style="display:none;" data-card-num="${gameState.top_card}"></span>
    `;
}

function updateCurrentTurn() {
    const currentPlayer = gameState.players[gameState.current_player_index];
    if (currentPlayer) {
        if (currentPlayer.id === playerId) {
            currentTurnElement.textContent = "Your turn";
            currentTurnElement.style.color = '#27ae60';
        } else {
            currentTurnElement.textContent = `${currentPlayer.name}'s turn`;
            currentTurnElement.style.color = '#f39c12';
        }
    }
}

function updateGameDirection() {
    gameDirectionElement.textContent = gameState.direction === 1 ? '↻' : '↺';
    gameDirectionElement.title = gameState.direction === 1 ? 'Clockwise' : 'Counter-clockwise';
}

function updatePenaltyInfo() {
    if (gameState.penalty_amount > 0) {
        penaltyInfoElement.style.display = 'block';
        penaltyAmountElement.textContent = gameState.penalty_amount;
    } else {
        penaltyInfoElement.style.display = 'none';
    }
}

function updateDrawPile() {
    drawPileCount.textContent = gameState.grab_deck_size;
}

function updateHandCount() {
    const currentPlayer = gameState.players.find(p => p.id === playerId);
    if (currentPlayer && currentPlayer.hand) {
        handCountElement.textContent = `${currentPlayer.hand.length} cards`;
    }
}

function isCardPlayable(cardNum) {
    const cardData = getCardData(cardNum);
    
    console.log(`🎴 Checking playability:`, {
        cardNum,
        cardData,
        currentSort: gameState.current_sort,
        currentNumber: gameState.current_true_number,
        penaltyAmount: gameState.penalty_amount
    });
    
    // During penalty, only matching penalty cards can be played (but Jacks and Jokers are always exceptions)
    if (gameState.penalty_amount > 0) {
        // Jacks and Jokers are always playable, even during penalties
        if (cardData.trueNumber === 0 || cardData.trueNumber === 13) {
            console.log(`✅ Special card always playable even in penalty: ${cardData.trueNumber} (${cardData.value})`);
            return true;
        }
        const canPlay = cardData.trueNumber === gameState.current_true_number;
        console.log(`⚠️ Penalty mode - can play: ${canPlay} (need ${gameState.current_true_number})`);
        return canPlay;
    }
    
    // Jokers and Jacks are always playable
    if (cardData.trueNumber === 0 || cardData.trueNumber === 13) {
        console.log(`✅ Special card always playable: ${cardData.trueNumber} (${cardData.value})`);
        return true;
    }
    
    // Check if it matches suit or value
    const suitMatch = cardData.suitName === gameState.current_sort;
    const numberMatch = cardData.trueNumber === gameState.current_true_number;
    const canPlay = suitMatch || numberMatch;
    
    console.log(`🔍 Normal play - suit match: ${suitMatch}, number match: ${numberMatch}, can play: ${canPlay}`);
    
    return canPlay;
}

function handleGameAction(action, details, actingPlayer = null) {
    // Use acting player if provided, otherwise fall back to current player
    const player = actingPlayer || gameState.players[gameState.current_player_index];
    
    console.log('🎮 Handling game action:', { action, details, player: player?.name, actingPlayer, isMyTurn: player?.id === playerId });
    
    switch (action) {
        case 'play':
            addLogEntry(`${player.name} played a card`);
            
            // Check if we need to show suit selection modal
            if (details.choose_sort) {
                console.log('🎭 Need to choose suit, acting player:', player?.name, 'is me:', player?.id === playerId);
                if (player.id === playerId) {
                    console.log('🎭 Showing sort modal for acting player');
                    showSortModal();
                } else {
                    console.log('🤖 AI will choose suit automatically');
                }
            }
            
            if (details.card_effects) {
                details.card_effects.forEach(effect => {
                    addLogEntry(getEffectMessage(effect));
                });
            }
            break;
            
        case 'pass':
            addLogEntry(`${player.name} passed`);
            break;
            
        case 'penalty':
            addLogEntry(`${player.name} drew ${gameState.penalty_amount || 'penalty'} cards`);
            break;
            
        case 'penalty_and_choose_sort':
            addLogEntry(`${player.name} drew penalty cards`);
            console.log('⚠️ Penalty and choose sort, acting player:', player?.name, 'is me:', player?.id === playerId);
            if (player.id === playerId) {
                console.log('🎭 Showing sort modal after penalty');
                showSortModal();
            }
            break;
            
        case 'penalty_and_sort_change':
            addLogEntry(`${player.name} drew penalty cards`);
            if (details.new_sort) {
                addLogEntry(`${player.name} changed suit to ${getSuitSymbol(details.new_sort)}`);
            }
            break;
            
        case 'win':
            addLogEntry(`🎉 ${player.name} wins!`, true);
            showToast(`🎉 ${player.name} wins the game!`, 'success');
            break;
            
        case 'draw_and_play_option':
            addLogEntry(`${player.name} drew a card`);
            if (player.id === playerId) {
                const drawnCard = details.card;
                showToast(`You drew a playable card! Click it to play or pass to keep it.`, 'info');
                // The drawn card will be in the updated hand, so just refresh the hand display
                updatePlayerHand();
            }
            break;
    }
}

function handleAIAction(action, details, actingPlayer = null) {
    // Use acting player if provided, otherwise fall back to current player  
    const player = actingPlayer || gameState.players[gameState.current_player_index];
    
    switch (action) {
        case 'play':
            addLogEntry(`🤖 ${player.name} played a card`);
            if (details.new_sort) {
                addLogEntry(`🤖 ${player.name} changed suit to ${getSuitSymbol(details.new_sort)}`);
            }
            break;
            
        case 'pass':
            addLogEntry(`🤖 ${player.name} passed`);
            break;
            
        case 'penalty':
            addLogEntry(`🤖 ${player.name} drew penalty cards`);
            break;
            
        case 'penalty_and_sort_change':
            addLogEntry(`🤖 ${player.name} drew penalty cards`);
            if (details.new_sort) {
                addLogEntry(`🤖 ${player.name} changed suit to ${getSuitSymbol(details.new_sort)}`);
            }
            break;
    }
}

function showSortModal() {
    console.log('🎭 Showing sort selection modal');
    awaitingSortChoice = true;
    sortModal.style.display = 'flex';
}

function getCardData(cardNum) {
    const suits = ['hearts', 'diamonds', 'spades', 'clubs'];
    const suitSymbols = ['♥️', '♦️', '♠️', '♣️'];
    // Map to actual game suit names from card.py
    const suitNames = ['harten', 'ruiten', 'schoppen', 'klaveren'];
    const values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'K', 'Q'];
    
    console.log(`🔍 Getting card data for: ${cardNum}`);
    
    if (cardNum < 48) {
        // Regular cards 0-47 (4 suits × 12 cards each)
        const suitIndex = Math.floor(cardNum / 12);
        const valueIndex = cardNum % 12;
        const trueNumber = valueIndex + 1;
        
        const cardData = {
            suit: suits[suitIndex],
            suitName: suitNames[suitIndex],
            symbol: suitSymbols[suitIndex],
            value: values[valueIndex],
            trueNumber: trueNumber
        };
        
        console.log(`📋 Regular card data:`, cardData);
        return cardData;
    } else if (cardNum === 48) {
        // Jack (card 48) - use generic Jack styling
        return {
            suit: 'special',
            suitName: 'special', 
            symbol: '🂿',  // Generic Jack symbol
            value: 'J',
            trueNumber: 13
        };
    } else if (cardNum === 49) {
        // Joker (card 49)
        return {
            suit: 'special',
            suitName: 'special', 
            symbol: '🃏',
            value: 'JKR',
            trueNumber: 0
        };
    }
    
    return {
        suit: 'unknown',
        suitName: 'unknown',
        symbol: '?',
        value: '?',
        trueNumber: -1
    };
}

function getSuitSymbol(suitName) {
    const suitMap = {
        'harten': '♥️',
        'ruiten': '♦️',
        'schoppen': '♠️',
        'klaveren': '♣️'
    };
    return suitMap[suitName] || suitName;
}

function getEffectMessage(effect) {
    const messages = {
        'another_turn': '🔄 Another turn!',
        'skip_player': '⏭️ Next player skipped',
        'reverse': '🔄 Direction reversed',
        'joker_penalty': '🃏 +5 penalty!',
        'change_sort': '🎭 Suit changed',
        'two_penalty': '2️⃣ +2 penalty!'
    };
    return messages[effect] || effect;
}

function getTurnOrder(myIndex, otherIndex, direction, totalPlayers) {
    // Calculate how many steps away in turn order
    let steps = 0;
    let currentIndex = myIndex;
    
    for (let i = 0; i < totalPlayers; i++) {
        currentIndex = (currentIndex + direction + totalPlayers) % totalPlayers;
        steps++;
        if (currentIndex === otherIndex) {
            break;
        }
    }
    
    return steps;
}

function getTurnOrderArrow(turnOrder) {
    // Return arrow based on turn order
    switch (turnOrder) {
        case 1: return '➡️'; // Next player
        case 2: return '⏭️'; // Two steps away
        case 3: return '⏭️⏭️'; // Three steps away
        default: return '🔄'; // Far away or same
    }
}

function addLogEntry(message, highlight = false) {
    const entry = document.createElement('div');
    entry.className = `log-entry ${highlight ? 'highlight' : ''}`;
    entry.textContent = message;
    
    logContent.appendChild(entry);
    logContent.scrollTop = logContent.scrollHeight;
    
    // Keep only last 50 entries
    while (logContent.children.length > 50) {
        logContent.removeChild(logContent.firstChild);
    }
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(toast);
    
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 3000);
}

// Handle page refresh - try to reconnect
window.addEventListener('beforeunload', () => {
    socket.disconnect();
});