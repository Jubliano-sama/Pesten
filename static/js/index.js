import { forgetSeat, rememberSeat, resumeToken } from './room-session.js';

const form = document.getElementById('room-form');
const nameField = document.getElementById('player-name');
const codeField = document.getElementById('room-code');
const codeGroup = document.getElementById('code-field');
const submit = document.getElementById('enter-room');
const errorMessage = document.getElementById('entry-error');
const tabs = [...document.querySelectorAll('[data-mode]')];
let mode = 'create';
let pending = false;
let queuedRequest = null;

nameField.value = localStorage.getItem('playerName') ?? '';

function setError(message = '') {
    errorMessage.textContent = message;
    errorMessage.hidden = !message;
}

function updateForm() {
    const joining = mode === 'join';
    codeGroup.hidden = !joining;
    codeField.required = joining;
    submit.firstChild.nodeValue = pending ? 'Connecting… ' : joining ? 'Join room ' : 'Create room ';
    submit.disabled = pending;
    for (const tab of tabs) {
        tab.setAttribute('aria-pressed', String(tab.dataset.mode === mode));
        tab.disabled = pending;
    }
}

for (const tab of tabs) {
    tab.addEventListener('click', () => {
        mode = tab.dataset.mode;
        setError();
        updateForm();
    });
}

const socket = window.io();

function requestFailed(message) {
    pending = false;
    queuedRequest = null;
    updateForm();
    setError(message);
}

form.addEventListener('submit', event => {
    event.preventDefault();
    const playerName = nameField.value.trim();
    if (playerName.length < 2) {
        setError('Enter a name with at least two characters.');
        nameField.focus();
        return;
    }

    const roomCode = codeField.value.toUpperCase();
    pending = true;
    setError();
    updateForm();
    localStorage.setItem('playerName', playerName);
    queuedRequest = mode === 'create' ?
        ['create_room', { player_name: playerName }] :
        ['join_room', {
            room_code: roomCode,
            player_name: playerName,
            resume_token: resumeToken(roomCode),
        }];
    sendRequest();
});

function sendRequest() {
    if (!socket.connected || !queuedRequest) return;
    socket.emit(...queuedRequest);
    queuedRequest = null;
}

function enterRoom(data) {
    rememberSeat(data);
    location.assign(`/room/${encodeURIComponent(data.room_code)}`);
}

socket.on('room_created', enterRoom);
socket.on('joined_room', enterRoom);
socket.on('error', ({ message, code }) => {
    if (code === 'seat_expired') forgetSeat(codeField.value.toUpperCase());
    requestFailed(message);
});
socket.on('connect', () => {
    sendRequest();
    if (!pending) setError();
});
socket.on('connect_error', () => requestFailed('Could not connect to Pesten. Check your connection and try again.'));
socket.on('disconnect', () => {
    if (pending) requestFailed('Connection lost. Please try again.');
});

import('./table3d.js?v=2')
    .then(({ mountHero }) => mountHero(document.getElementById('hero-scene')))
    .catch(() => {});
