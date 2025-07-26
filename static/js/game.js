// Game page JavaScript
const socket = io();

// Game state
let gameState = null;
let playerId = localStorage.getItem('playerId');
let roomCode = localStorage.getItem('roomCode') || document.getElementById('room-code').textContent;
let selectedCard = null;
let awaitingSortChoice = false;
let lastCardWarnings = new Set(); // Track which players have been warned about having 1 card
let turnTimer = null; // Current turn timer
let timeLeft = 0; // Seconds left in current turn
let lastDirectionChange = null; // Track when direction last changed to avoid duplicate animations

// DOM elements
const lobbySection = document.getElementById('lobby-section');
const gameSection = document.getElementById('game-section');
const playersListElement = document.getElementById('players-list');
const addAiBtn = document.getElementById('add-ai-btn');
const startGameBtn = document.getElementById('start-game-btn');
const leaveGameBtn = document.getElementById('leave-game');
const playerHandElement = document.getElementById('player-hand');
const otherPlayersElement = document.getElementById('other-players');
const playerQueueElement = document.getElementById('player-queue');
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
const winModal = document.getElementById('win-modal');
const winPlayerName = document.getElementById('win-player-name');
const winCloseBtn = document.getElementById('win-close-btn');
const popupContainer = document.getElementById('popup-container');

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
    const currentPlayer = gameState.players.find(p => p.id === playerId);
    if (currentPlayer && currentPlayer.is_observer) {
        showToast("You are observing - cannot pass", 'info');
        return;
    }
    
    if (!awaitingSortChoice) {
        stopTurnTimer(); // Stop timer since we're making a move
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
        stopTurnTimer(); // Stop timer since we're making a move
        socket.emit('change_sort', {
            room_code: roomCode,
            player_id: playerId,
            new_sort: newSort
        });
        sortModal.style.display = 'none';
        awaitingSortChoice = false;
    });
});

// Win modal handlers
winCloseBtn.addEventListener('click', () => {
    winModal.style.display = 'none';
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
    showGamePopup(`🎭 Suit changed to ${getSuitSymbol(data.new_sort)}!`, 'suit-change');
});

socket.on('player_left', (data) => {
    updateGameState(data.game_state);
    addLogEntry('A player left the room');
});

socket.on('player_auto_passed', (data) => {
    console.log('⏰ Received auto-pass event:', data);
    
    // Check if it was our timeout
    const currentPlayer = gameState?.players?.find(p => p.id === playerId);
    const wasMyTimeout = data.player_name === currentPlayer?.name;
    
    console.log('⏰ Auto-pass details:', {
        playerWhoTimedOut: data.player_name,
        wasMyTimeout: wasMyTimeout,
        newCurrentPlayerIndex: data.game_state.current_player_index,
        newCurrentPlayer: data.game_state.players[data.game_state.current_player_index]?.name
    });
    
    updateGameState(data.game_state);
    addLogEntry(`⏰ ${data.player_name} timed out and auto-passed (${data.consecutive_timeouts}/3)`);
    
    if (wasMyTimeout) {
        showToast('You timed out and auto-passed', 'warning');
        console.log('⏰ My timeout - game should continue to next player');
    } else {
        showToast(`${data.player_name} auto-passed due to timeout`, 'warning');
        console.log('⏰ Other player timed out - checking if it\'s now my turn');
    }
    
    // Explicitly ensure all UI elements are updated after auto-pass
    setTimeout(() => {
        console.log('⏰ Force updating all UI after auto-pass');
        updateGameInterface();
    }, 100);
});

socket.on('player_kicked', (data) => {
    updateGameState(data.game_state);
    addLogEntry(`❌ ${data.player_name} was kicked: ${data.reason}`);
    showToast(`${data.player_name} was kicked from the game`, 'error');
    showGamePopup(`❌ ${data.player_name} was kicked for ${data.reason}!`, 'warning');
    console.log('You got kicked');
});

socket.on('game_tied', (data) => {
    console.log('🤝 Game ended in a tie:', data);
    updateGameState(data.game_state);
    addLogEntry('🤝 Game ended in a tie - no more cards available!', true);
    showToast('🤝 Game ended in a tie!', 'info');
    showTieModal();
});

socket.on('error', (data) => {
    console.error('❌ Server error:', data);
    showToast(data.message, 'error');
    
    // Handle specific error types that might require client-side cleanup
    if (data.message.includes('Invalid player name') || 
        data.message.includes('Room is full') ||
        data.message.includes('already taken')) {
        // For player validation errors, we might want to redirect back to the main page
        console.log('📝 Player validation error, might need to re-enter name');
    }
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
    
    // Check for players with 1 card left (only during active game)
    if (gameState.started && !gameState.game_over) {
        gameState.players.forEach(player => {
            const handSize = player.hand ? player.hand.length : player.hand_size;
            if (handSize === 1 && !lastCardWarnings.has(player.id)) {
                // Player just got down to 1 card
                lastCardWarnings.add(player.id);
                if (player.id === playerId) {
                    showGamePopup('⚠️ Last card! You\'re almost out!', 'last-card');
                } else {
                    showGamePopup(`⚠️ ${player.name} has only 1 card left!`, 'warning');
                }
            } else if (handSize > 1 && lastCardWarnings.has(player.id)) {
                // Player got more cards, remove from warning set
                lastCardWarnings.delete(player.id);
            }
        });
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
        playerCard.className = `player-card ${player.is_ai ? 'ai' : ''} ${player.is_observer ? 'observer' : ''}`;
        
        let statusText = player.is_ai ? '🤖 AI Player' : '👤 Human';
        if (player.is_observer) {
            statusText += ' 👁️ Observer';
        }
        
        playerCard.innerHTML = `
            <div class="player-name">${player.name}</div>
            <div class="player-status">${statusText}</div>
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
    updatePlayerQueue();
    updatePlayerHand();
    updateTopCard();
    updateCurrentTurn();
    updateGameDirection();
    updatePenaltyInfo();
    updateDrawPile();
    updateHandCount();
}

function updatePlayerQueue() {
    console.log('🔄 Updating player queue');
    
    // Only trigger reversal animation once per direction change
    const currentTurnCount = gameState.turn_count;
    if (gameState.direction_changed && lastDirectionChange !== currentTurnCount) {
        console.log('↩️ DIRECTION CHANGED - triggering reversal animation once');
        lastDirectionChange = currentTurnCount;
        
        // Trigger arrow wobble animation
        const arrows = playerQueueElement.querySelectorAll('.queue-arrow');
        arrows.forEach(arrow => {
            arrow.classList.remove('wobble');
            void arrow.offsetWidth; // Force reflow
            arrow.classList.add('wobble');
        });
    }
    
    console.log('📊 GameState turn_order:', gameState.turn_order);
    
    if (!playerQueueElement || !gameState.turn_order) return;
    
    const currentPlayer = gameState.players.find(p => p.id === playerId);
    if (!currentPlayer) return;
    
    // Use server's turn_order directly - it's already calculated correctly
    const serverTurnOrder = gameState.turn_order;
    
    // Find my position in the server's turn order
    const myIndexInTurnOrder = serverTurnOrder.findIndex(id => id === playerId);
    
    // Build queue with "You" at the start, others following server order
    const queueOrder = [];
    
    if (myIndexInTurnOrder !== -1) {
        // Start with me, then add others in server turn order
        for (let i = 0; i < serverTurnOrder.length; i++) {
            const adjustedIndex = (myIndexInTurnOrder + i) % serverTurnOrder.length;
            const playerId_at_index = serverTurnOrder[adjustedIndex];
            const player = gameState.players.find(p => p.id === playerId_at_index);
            if (player && !player.is_observer) {
                queueOrder.push(player);
            }
        }
    }
    
    // Clear existing queue
    playerQueueElement.innerHTML = '';
    
    queueOrder.forEach((player, queueIndex) => {
        const playerIndex = gameState.players.indexOf(player);
        const isCurrentTurn = gameState.current_player_index === playerIndex;
        const isMe = player.id === playerId;
        
        // Create queue player element
        const queuePlayer = document.createElement('div');
        let classes = ['queue-player'];
        
        if (isCurrentTurn) {
            classes.push('spotlight');
        }
        if (isMe) {
            classes.push('you');
        }
        
        queuePlayer.className = classes.join(' ');
        queuePlayer.setAttribute('data-player-id', player.id);
        
        // Player avatar (first letter of name or AI icon)
        const avatar = document.createElement('div');
        avatar.className = 'queue-player-avatar';
        avatar.textContent = player.is_ai ? '🤖' : player.name.charAt(0).toUpperCase();
        
        // Player name
        const name = document.createElement('div');
        name.className = 'queue-player-name';
        name.textContent = isMe ? 'You!' : player.name;
        
        // Card count
        const cards = document.createElement('div');
        cards.className = 'queue-player-cards';
        cards.textContent = `${player.hand_size} cards`;
        
        queuePlayer.appendChild(avatar);
        queuePlayer.appendChild(name);
        queuePlayer.appendChild(cards);
        
        playerQueueElement.appendChild(queuePlayer);
        
        // Add arrow between players (except after last player)
        if (queueIndex < queueOrder.length - 1) {
            const arrow = document.createElement('div');
            arrow.className = 'queue-arrow';
            arrow.textContent = '→'; // Always point right
            playerQueueElement.appendChild(arrow);
        }
    });
    
    // Scroll spotlight player into view
    setTimeout(() => {
        const spotlightPlayer = playerQueueElement.querySelector('.queue-player.spotlight');
        if (spotlightPlayer) {
            spotlightPlayer.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'nearest', 
                inline: 'center' 
            });
        }
    }, 100);
    
    // Trigger reversal animation if server says direction changed
    if (gameState.direction_changed) {
        console.log('🔄 Server reports direction changed, triggering animation');
        animateDirectionReversal();
    }
}


function animateSkippedPlayer(playerId) {
    console.log(`⏭️ Animating skip for player ${playerId}`);
    
    const queuePlayer = playerQueueElement.querySelector(`[data-player-id="${playerId}"]`);
    if (queuePlayer) {
        queuePlayer.classList.add('skipped');
        
        // Remove animation class after animation completes
        setTimeout(() => {
            queuePlayer.classList.remove('skipped');
        }, 600);
    }
}

function animateDirectionReversal() {
    console.log('🔄 Animating direction reversal');
    
    // Add reversal animation to all queue players (except "You")
    const queuePlayers = playerQueueElement.querySelectorAll('.queue-player:not(.you)');
    queuePlayers.forEach(player => {
        player.classList.add('reversing');
    });
    
    // Add wobble animation to all arrows
    const arrows = playerQueueElement.querySelectorAll('.queue-arrow');
    arrows.forEach(arrow => {
        arrow.classList.add('wobble');
    });
    
    // Remove animation classes and rebuild queue after animation completes
    setTimeout(() => {
        queuePlayers.forEach(player => {
            player.classList.remove('reversing');
        });
        arrows.forEach(arrow => {
            arrow.classList.remove('wobble');
        });
        // Rebuild queue with new order
        updatePlayerQueue();
    }, 800);
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
            if (currentPlayer.is_observer) {
                showToast("You are observing - cannot play cards", 'info');
                return;
            }
            
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
            stopTurnTimer(); // Stop timer since we're making a move
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
        console.log('🔄 Turn update:', { 
            currentPlayerName: currentPlayer.name, 
            isMyTurn: currentPlayer.id === playerId,
            isObserver: currentPlayer.is_observer,
            isAI: currentPlayer.is_ai,
            playerIndex: gameState.current_player_index,
            totalPlayers: gameState.players.length
        });
        
        if (currentPlayer.id === playerId && !currentPlayer.is_observer) {
            // Start timer for my turn
            console.log('🔄 Starting timer - it\'s my turn');
            startTurnTimer();
        } else {
            // Stop timer if it's not my turn
            console.log('🔄 Stopping timer - not my turn or I\'m observing');
            stopTurnTimer();
        }
        
        // Update the display
        updateTurnDisplay();
    } else {
        console.warn('🔄 updateCurrentTurn: No current player found');
    }
}

function updateTurnDisplay() {
    const currentPlayer = gameState.players[gameState.current_player_index];
    if (!currentPlayer) return;
    
    let turnText = '';
    let color = '#f39c12';
    
    if (currentPlayer.id === playerId) {
        turnText = "Your turn";
        color = '#27ae60';
    } else {
        turnText = `${currentPlayer.name}'s turn`;
        color = '#f39c12';
    }
    
    // Add timer display if it's a human player and timer is running
    if (!currentPlayer.is_ai && !currentPlayer.is_observer && turnTimer && timeLeft > 0) {
        turnText += ` (${timeLeft}s)`;
        
        // Change color as time runs out
        if (timeLeft <= 10) {
            color = '#e74c3c'; // Red for urgency
        } else if (timeLeft <= 20) {
            color = '#f39c12'; // Orange for warning
        }
    }
    
    currentTurnElement.textContent = turnText;
    currentTurnElement.style.color = color;
}

function startTurnTimer() {
    console.log('⏱️ Starting turn timer');
    stopTurnTimer(); // Clear any existing timer
    timeLeft = 60;
    updateTurnDisplay(); // Initial display
    
    turnTimer = setInterval(() => {
        timeLeft--;
        updateTurnDisplay(); // Update the display every second
        
        if (timeLeft <= 0) {
            console.log('⏱️ Client timer expired, notifying server');
            stopTurnTimer();
            // Show that we're waiting for server response
            currentTurnElement.textContent = "Time's up! Waiting for server...";
            currentTurnElement.style.color = '#e74c3c';
            
            // Proactively notify server of timeout to avoid race condition
            console.log('⏱️ Sending timeout signal to server');
            socket.emit('player_timeout', {
                room_code: roomCode,
                player_id: playerId
            });
        }
    }, 1000);
}

function stopTurnTimer() {
    if (turnTimer) {
        console.log('⏱️ Stopping turn timer');
        clearInterval(turnTimer);
        turnTimer = null;
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
    
    // Invalid cards are never playable
    if (cardData.trueNumber === -1 || cardData.suitName === 'invalid' || cardData.suitName === 'unknown') {
        console.warn(`❌ Invalid card ${cardNum} is not playable`);
        return false;
    }
    
    // During penalty, ONLY matching penalty cards can be played (NO exceptions)
    if (gameState.penalty_amount > 0) {
        const canPlay = cardData.trueNumber === gameState.current_true_number;
        console.log(`⚠️ Penalty mode - can play: ${canPlay} (need exact match ${gameState.current_true_number}, have ${cardData.trueNumber})`);
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
                    
                    // Handle special animations
                    if (effect === 'skip_player') {
                        // Find the next player who would be skipped
                        const currentIndex = gameState.players.findIndex(p => p.id === player.id);
                        const nextIndex = (currentIndex + gameState.direction + gameState.players.length) % gameState.players.length;
                        const skippedPlayer = gameState.players[nextIndex];
                        if (skippedPlayer && !skippedPlayer.is_observer) {
                            animateSkippedPlayer(skippedPlayer.id);
                        }
                    }
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
            showWinModal(player.name);
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
                showGamePopup(`🎭 ${player.name} changed suit to ${getSuitSymbol(details.new_sort)}!`, 'suit-change');
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
                showGamePopup(`🎭 ${player.name} changed suit to ${getSuitSymbol(details.new_sort)}!`, 'suit-change');
            }
            break;
            
        case 'win':
            addLogEntry(`🤖🎉 ${player.name} wins!`, true);
            showToast(`🤖🎉 ${player.name} wins the game!`, 'success');
            showWinModal(player.name);
            break;
    }
}

function showSortModal() {
    console.log('🎭 Showing sort selection modal');
    awaitingSortChoice = true;
    sortModal.style.display = 'flex';
}

function showWinModal(playerName) {
    console.log('🎉 Showing win modal for:', playerName);
    
    const currentPlayer = gameState.players.find(p => p.id === playerId);
    const isWinner = currentPlayer && currentPlayer.name === playerName;
    
    if (isWinner) {
        // You won - congratulations and become observer
        winPlayerName.textContent = `You Win! 🎉`;
        document.getElementById('win-message').textContent = 'Congratulations!';
        document.getElementById('win-close-btn').textContent = 'Continue as Observer';
    } else {
        // Someone else won - just acknowledgment
        winPlayerName.textContent = `${playerName} Wins! 🎉`;
        document.getElementById('win-message').textContent = `${playerName} finished the game!`;
        document.getElementById('win-close-btn').textContent = 'Continue Playing';
    }
    
    winModal.style.display = 'flex';
}

function showTieModal() {
    console.log('🤝 Showing tie modal');
    
    // Reuse the win modal but with tie content
    winPlayerName.textContent = `Game Tied! 🤝`;
    document.getElementById('win-message').textContent = 'No more cards available - it\'s a tie!';
    document.getElementById('win-close-btn').textContent = 'Back to Lobby';
    
    // Add special styling for tie
    winModal.classList.add('tie-modal');
    
    winModal.style.display = 'flex';
    
    // Redirect to lobby after closing
    winCloseBtn.onclick = () => {
        winModal.style.display = 'none';
        winModal.classList.remove('tie-modal');
        window.location.href = '/';
    };
}

function showGamePopup(message, type = 'default') {
    console.log('📢 Showing game popup:', message, 'type:', type);
    
    const popup = document.createElement('div');
    popup.className = `game-popup ${type}`;
    popup.textContent = message;
    
    popupContainer.appendChild(popup);
    
    // Remove popup after animation completes (3 seconds total)
    setTimeout(() => {
        if (popup.parentNode) {
            popup.parentNode.removeChild(popup);
        }
    }, 3000);
}

function getCardData(cardNum) {
    const suits = ['hearts', 'diamonds', 'spades', 'clubs'];
    const suitSymbols = ['♥️', '♦️', '♠️', '♣️'];
    // Map to actual game suit names from card.py
    const suitNames = ['harten', 'ruiten', 'schoppen', 'klaveren'];
    const values = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'K', 'Q'];
    
    console.log(`🔍 Getting card data for: ${cardNum}`);
    
    // Validate card number
    if (typeof cardNum !== 'number' || cardNum < 0 || cardNum > 53) {
        console.warn(`❌ Invalid card number: ${cardNum}`);
        return {
            suit: 'invalid',
            suitName: 'invalid',
            symbol: '❌',
            value: 'INV',
            trueNumber: -1
        };
    }
    
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
    } else if (cardNum >= 50 && cardNum <= 53) {
        // Jack variants with specific suits (50=hearts, 51=diamonds, 52=spades, 53=clubs)
        const suitIndex = cardNum - 50; // 0=hearts, 1=diamonds, 2=spades, 3=clubs
        return {
            suit: suits[suitIndex],
            suitName: suitNames[suitIndex],
            symbol: suitSymbols[suitIndex],
            value: 'J',
            trueNumber: 13
        };
    }
    
    // Should not reach here due to range check above, but safety fallback
    console.error(`❌ Unexpected card number in valid range: ${cardNum}`);
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