// Index page JavaScript
const socket = io();

// DOM elements
const createRoomForm = document.getElementById('create-room-form');
const joinRoomForm = document.getElementById('join-room-form');
const loadingOverlay = document.getElementById('loading');

// Form handlers
createRoomForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const playerName = document.getElementById('player-name').value.trim();
    const maxPlayers = parseInt(document.getElementById('max-players').value);
    
    if (!playerName) {
        showToast('Please enter your name', 'error');
        return;
    }
    
    showLoading();
    
    // Store player name for later use
    localStorage.setItem('playerName', playerName);
    
    socket.emit('create_room', {
        player_name: playerName,
        max_players: maxPlayers
    });
});

joinRoomForm.addEventListener('submit', (e) => {
    e.preventDefault();
    
    const playerName = document.getElementById('join-player-name').value.trim();
    const roomCode = document.getElementById('room-code').value.trim().toUpperCase();
    
    if (!playerName || !roomCode) {
        showToast('Please enter both name and room code', 'error');
        return;
    }
    
    if (roomCode.length !== 6) {
        showToast('Room code must be 6 characters', 'error');
        return;
    }
    
    showLoading();
    
    // Store player name for later use
    localStorage.setItem('playerName', playerName);
    
    socket.emit('join_room', {
        player_name: playerName,
        room_code: roomCode
    });
});

// Auto-uppercase room code input
document.getElementById('room-code').addEventListener('input', (e) => {
    e.target.value = e.target.value.toUpperCase();
});

// Socket event handlers
socket.on('room_created', (data) => {
    hideLoading();
    
    // Store player info in localStorage
    localStorage.setItem('playerId', data.player_id);
    localStorage.setItem('roomCode', data.room_code);
    
    showToast('Room created successfully!', 'success');
    
    // Redirect to game room
    setTimeout(() => {
        window.location.href = `/room/${data.room_code}`;
    }, 1000);
});

socket.on('joined_room', (data) => {
    hideLoading();
    
    // Store player info in localStorage
    localStorage.setItem('playerId', data.player_id);
    localStorage.setItem('roomCode', data.room_code);
    
    showToast('Joined room successfully!', 'success');
    
    // Redirect to game room
    setTimeout(() => {
        window.location.href = `/room/${data.room_code}`;
    }, 1000);
});

socket.on('error', (data) => {
    hideLoading();
    showToast(data.message, 'error');
});

// Utility functions
function showLoading() {
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    // Create container if it doesn't exist
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        container.className = 'toast-container';
        document.body.appendChild(container);
    }
    
    container.appendChild(toast);
    
    // Remove toast after 3 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 3000);
}

// Clear any existing player data when page loads
localStorage.removeItem('playerId');
localStorage.removeItem('roomCode');