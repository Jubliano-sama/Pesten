import { cardData, paintCard, suits } from './cards.js';
import { playSoundEffect } from './audio.js';
import { forgetSeat, rememberSeat, resumeToken } from './room-session.js';

const $ = id => document.getElementById(id);
const code = $('room-code').textContent.trim().toUpperCase();
const socket = window.io();
const passButton = $('pass-btn');
const drawPile = $('draw-pile');
const handElement = $('player-hand');
const suitDialog = $('sort-modal');
const winDialog = $('win-modal');
const table = $('round-table');
const viewControls = $('view-controls');
const viewButtons = [...viewControls.querySelectorAll('button')];
const suitButtons = [...suitDialog.querySelectorAll('button')];
const tableSeats = new Map();
const drawBatches = new WeakMap();
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');

let identity = null;
let seatReady = false;
let handReady = false;
let state = null;
let hand = [];
let playable = new Set();
let pending = false;
let finishingSuitDraw = false;
let handVersion = '';
let discardVersion;
let lastTurn;
let lastDirection;
let lastPenalty = 0;
let lastClock = null;
let outcomeShown = false;
let table3d;
let loading3d = false;
let preferredView = localStorage.getItem('pesten.tableView') === '2d' ? '2d' : '3d';

function self() {
    return state?.players.find(player => player.id === identity);
}

function current() {
    return state?.players[state.current_player_index];
}

function canAct() {
    return seatReady && handReady && state?.started && !state.game_over && identity &&
        current()?.id === identity && !self()?.is_observer;
}

function showEntry(message, askName = false) {
    $('entry-status').textContent = message;
    $('room-entry').hidden = false;
    $('name-form').hidden = !askName;
}

function toast(message, type = '') {
    const item = document.createElement('div');
    item.className = `toast ${type}`;
    item.textContent = message;
    $('toast-container').append(item);
    setTimeout(() => item.remove(), 3600);
}

function log(message) {
    const entry = document.createElement('li');
    entry.textContent = message;
    const list = $('log-content');
    list.prepend(entry);
    if (list.childElementCount > 40) list.lastElementChild.remove();
}

function roomAction(event, payload = {}) {
    if (!seatReady || pending) return;
    pending = true;
    renderActions();
    socket.emit(event, { room_code: code, player_id: identity, ...payload });
}

function play(number) {
    if (!canAct() || state.waiting_for_suit_choice || (number !== null && !playable.has(number))) return;
    roomAction('play_card', { card_num: number });
}

function setView(view) {
    table.dataset.view = view;
    for (const button of viewButtons) {
        button.setAttribute('aria-pressed', String(button.dataset.view === view));
    }
    table3d?.show(view === '3d');
}

function renderScene() {
    if (!table3d || !state?.started) return;
    table3d.update({
        topCard: state.top_card,
        deckSize: state.grab_deck_size,
    });
}

async function selectView(view) {
    if (view === '2d') {
        setView('2d');
        return;
    }
    if (table3d) {
        setView('3d');
        return;
    }
    if (loading3d) return;

    const button = viewButtons.find(item => item.dataset.view === '3d');
    loading3d = true;
    button.disabled = true;
    try {
        const { mountTable } = await import('./table3d.js?v=3');
        if (preferredView !== '3d') return;
        table3d = mountTable($('table-scene'), {
            onLost() {
                table3d = null;
                setView('2d');
                toast('3D stopped. The table switched to 2D.');
            },
            onPiles(draw, discard, width, height) {
                table.style.setProperty('--draw-3d-x', `${draw.x}px`);
                table.style.setProperty('--draw-3d-y', `${draw.y}px`);
                table.style.setProperty('--discard-3d-x', `${discard.x}px`);
                table.style.setProperty('--discard-3d-y', `${discard.y}px`);
                table.style.setProperty('--card-3d-width', `${width}px`);
                table.style.setProperty('--card-3d-height', `${height}px`);
            },
        });
        renderScene();
        setView('3d');
    } catch {
        setView('2d');
        toast('3D is unavailable here. You can keep playing in 2D.');
    } finally {
        loading3d = false;
        button.disabled = false;
    }
}

for (const button of viewButtons) {
    button.addEventListener('click', () => {
        preferredView = button.dataset.view;
        localStorage.setItem('pesten.tableView', preferredView);
        selectView(preferredView);
    });
}

function renderSeats(container) {
    const isPlaying = state.started;
    const active = isPlaying && !state.game_over ? current()?.id : null;
    const next = active ? state.turn_order?.[1] : null;
    const rows = state.players.map(player => {
        const row = document.createElement('li');
        const isActive = player.id === active;
        row.className = isActive ? 'seat is-turn' : 'seat';
        if (isActive) row.setAttribute('aria-current', 'true');

        const avatar = document.createElement('span');
        avatar.className = 'seat-avatar';
        avatar.textContent = player.is_ai ? 'AI' : player.name.split(/\s+/).slice(0, 2).map(word => word[0]).join('').toUpperCase();
        avatar.setAttribute('aria-hidden', 'true');

        const label = document.createElement('div');
        label.className = 'seat-copy';
        const name = document.createElement('strong');
        name.className = 'seat-name';
        name.textContent = player.name + (player.id === identity ? ' (you)' : '');
        const status = document.createElement('span');
        status.className = 'seat-subtitle';
        status.textContent = [
            player.is_ai ? 'Computer' : player.is_connected ? 'Player' : 'Disconnected',
            player.is_observer ? 'Out' : isActive ? 'Playing' : player.id === next ? 'Up next' : '',
        ].filter(Boolean).join(' · ');
        label.append(name, status);
        row.append(avatar, label);

        if (isPlaying && !player.is_observer) {
            const tally = document.createElement('span');
            tally.className = player.hand_size === 1 ? 'seat-tally last-card' : 'seat-tally';
            tally.textContent = pluralCards(player.hand_size);
            row.append(tally);
        }
        return row;
    });
    container.replaceChildren(...rows);
}

function pluralCards(count) {
    return `${count} ${count === 1 ? 'card' : 'cards'}`;
}

function makeTableSeat(player, ownSeat) {
    const element = document.createElement(ownSeat ? 'div' : 'li');
    element.className = 'table-seat';
    element.dataset.playerId = player.id;
    const header = document.createElement('div');
    header.className = 'table-seat-header';
    const name = document.createElement('span');
    name.className = 'table-seat-name';
    name.textContent = ownSeat ? 'You' : player.name;
    name.title = player.name;
    const count = document.createElement('span');
    count.className = 'table-seat-count';
    count.setAttribute('role', 'img');
    header.append(name, count);
    const backs = document.createElement('div');
    backs.className = 'table-seat-cards';
    backs.setAttribute('aria-hidden', 'true');
    element.append(header, backs);
    return { element, count, backs };
}

function renderTableSeats() {
    const opponents = $('opponent-seats');
    const own = $('your-seat');
    const visible = new Set();
    const opponentElements = [];
    const active = state.game_over ? null : current()?.id;
    for (const player of state.players) {
        const ownSeat = player.id === identity;
        let seat = tableSeats.get(player.id);
        if (!seat) {
            seat = makeTableSeat(player, ownSeat);
            tableSeats.set(player.id, seat);
        }
        const { element, count, backs } = seat;
        const size = player.hand_size;
        visible.add(player.id);
        element.dataset.active = String(player.id === active);
        if (element.dataset.cardCount !== String(size)) {
            element.dataset.cardCount = size;
            count.textContent = size;
            count.setAttribute('aria-label', pluralCards(size));
            const missing = size - backs.childElementCount;
            if (missing < 0) {
                backs.replaceChildren(...Array.from(backs.children).slice(0, size));
            }
            for (let i = 0; i < missing; i++) {
                const back = document.createElement('span');
                back.className = 'seat-card-back';
                backs.append(back);
            }
        }
        if (ownSeat) {
            if (element.parentNode !== own) own.replaceChildren(element);
        } else {
            opponentElements.push(element);
        }
    }
    for (const [id, { element }] of tableSeats) {
        if (visible.has(id)) continue;
        element.remove();
        tableSeats.delete(id);
    }
    if (opponentElements.some((element, i) => opponents.children[i] !== element)) {
        opponents.replaceChildren(...opponentElements);
    }
    opponents.style.setProperty('--seat-columns', Math.max(1, Math.min(4, opponentElements.length)));
}

function animateDraw(playerId, amount) {
    const seat = tableSeats.get(playerId);
    if (!seat || !Number.isSafeInteger(amount) || amount <= 0) return;
    playSoundEffect('card');
    if (reducedMotion.matches || document.hidden) return;

    const layer = $('draw-animations');
    const bounds = layer.getBoundingClientRect();
    const source = drawPile.getBoundingClientRect();
    if (!bounds.width || !source.width) return;

    const batch = Symbol();
    drawBatches.set(seat.element, batch);
    seat.element.dataset.drawing = '';
    const width = Math.min(source.width, 64);
    const height = Math.min(source.height, 86);
    const startX = source.left + source.width / 2 - bounds.left;
    const startY = source.top + source.height / 2 - bounds.top;
    const playersBacks = Array.from(seat.backs.children).slice(-amount);
    const stagger = Math.min(75, 450 / Math.max(1, amount - 1));
    const animations = [];

    for (let i = 0; i < amount; i++) {
        const target = (playersBacks[i] ?? seat.count).getBoundingClientRect();
        const endX = target.left + target.width / 2 - bounds.left;
        const endY = target.top + target.height / 2 - bounds.top;
        const arc = (endX > startX ? -1 : 1) * Math.min(44, bounds.width / 10);
        const transform = (x, y, angle, scale) =>
            `translate3d(${x - width / 2}px, ${y - height / 2}px, 0) rotate(${angle}deg) scale(${scale})`;
        const card = document.createElement('span');
        card.className = 'flying-card';
        card.style.width = `${width}px`;
        card.style.height = `${height}px`;
        layer.append(card);
        const animation = card.animate([
            { transform: transform(startX, startY, -8, 1), opacity: 0 },
            { transform: transform(startX, startY - 6, -11, 1), opacity: 1, offset: .12 },
            { transform: transform((startX + endX) / 2 + arc, (startY + endY) / 2 - 12, arc / 2, .78), opacity: 1, offset: .56 },
            { transform: transform(endX, endY, 0, Math.min(target.width / width, .3)), opacity: .9 },
        ], { duration: 480, delay: i * stagger, fill: 'both', easing: 'cubic-bezier(.23,.72,.30,1)' });
        animations.push(animation.finished.catch(() => {}).then(() => card.remove()));
    }
    return Promise.all(animations).then(() => {
        if (drawBatches.get(seat.element) === batch) delete seat.element.dataset.drawing;
    });
}

function renderHand() {
    const version = hand.join(',');
    if (version !== handVersion) {
        const focused = document.activeElement;
        const hadCardFocus = handElement.contains(focused);
        const buttons = hand.map(number => {
            const button = document.createElement('button');
            button.type = 'button';
            button.dataset.card = number;
            button.className = 'playing-card';
            paintCard(button, number);
            button.addEventListener('click', () => play(number));
            return button;
        });
        handElement.replaceChildren(...buttons);
        handVersion = version;
        if (hadCardFocus) {
            const replacement = buttons.find(button => button.dataset.card === focused.dataset.card &&
                playable.has(Number(button.dataset.card)));
            if (replacement) replacement.focus({ preventScroll: true });
        }
    }
}

function renderLobbyActions() {
    if (!state || state.started) return;
    const connected = seatReady && self();
    const ready = state.players.filter(player => player.is_connected || player.is_ai).length;
    $('add-ai-btn').disabled = !connected || pending || state.players.length >= 32;
    $('start-game-btn').disabled = !connected || pending || ready < 2;
    $('lobby-hint').textContent = !seatReady ? 'Reconnecting…' :
        ready < 2 ? 'You need at least two players to start.' : '';
}

function renderActions() {
    if (!state?.started) {
        renderLobbyActions();
        return;
    }
    const myTurn = seatReady && !state.game_over && current()?.id === identity && !self()?.is_observer;
    const choosing = myTurn && state.waiting_for_suit_choice;
    const drawLabel = state.penalty_amount ? `Draw ${state.penalty_amount} cards` :
        state.has_drawn_this_turn ? 'Pass turn' : 'Draw card';
    const actionDisabled = !myTurn || !handReady || choosing || pending;
    passButton.textContent = drawLabel;
    passButton.disabled = actionDisabled;
    drawPile.setAttribute('aria-label', drawLabel);
    drawPile.disabled = actionDisabled;
    const offer = !actionDisabled && playable.size > 0;
    for (const button of handElement.children) {
        const legal = playable.has(Number(button.dataset.card));
        button.classList.toggle('playable', legal && myTurn && !choosing);
        button.disabled = !offer || !legal;
    }
    $('hand-hint').textContent = !seatReady ? 'Reconnecting…' :
        state.game_over ? 'This round is over.' :
        self()?.is_observer ? 'You’re out. Watch the remaining players finish.' :
        !self() ? 'Your seat is no longer at the table.' :
        choosing ? 'Choose the suit to continue.' :
        !myTurn ? 'Waiting for the current player.' :
        state.has_drawn_this_turn ? 'Play a highlighted card or pass.' :
        playable.size ? 'Choose a highlighted card to play.' : `No playable cards. ${drawLabel}.`;
    for (const button of suitButtons) button.disabled = pending || !handReady;
    if (choosing && !finishingSuitDraw && !suitDialog.open) suitDialog.showModal();
    if (!choosing && suitDialog.open) suitDialog.close();
}

function updateClock() {
    const clock = $('turn-timer');
    const remaining = state?.turn_deadline_ms && current()?.id === identity && socket.connected ?
        Math.max(0, Math.ceil((state.turn_deadline_ms - Date.now()) / 1000)) : null;
    if (lastClock === remaining) return;
    lastClock = remaining;
    clock.hidden = remaining === null;
    clock.textContent = remaining === null ? '' : `${remaining}s`;
    clock.classList.toggle('urgent', remaining !== null && remaining <= 10);
}

function displayState(next, privateState = null) {
    const justStarted = next.started && !state?.started;
    state = next;
    handReady = Boolean(privateState);
    if (privateState) {
        hand = privateState.hand;
        playable = new Set(privateState.playable_cards);
    } else {
        // Public state arrives first. Use the matching private packet before allowing a move.
        playable = new Set();
        if (!self() || self().hand_size === 0) hand = [];
    }
    $('room-entry').hidden = true;
    $('lobby-section').hidden = state.started;
    $('game-section').hidden = !state.started;
    $('view-controls').hidden = !state.started;

    if (!state.started) {
        renderSeats($('players-list'));
        $('lobby-count').textContent = `${state.players.length} seated`;
        renderLobbyActions();
        return;
    }

    const acting = state.game_over ? null : current();
    const title = state.game_over ? 'Round over' : acting?.id === identity ? 'Your turn' :
        acting ? `${acting.name}’s turn` : 'Waiting for players';
    $('current-turn').textContent = title;
    $('seat-count').textContent = state.players.length;
    renderTableSeats();
    renderSeats($('player-queue'));
    $('hand-count').textContent = pluralCards(self()?.hand_size ?? 0);
    $('direction-label').textContent = state.direction === -1 ? 'Counterclockwise' : 'Clockwise';
    $('game-direction').textContent = state.direction === -1 ? '↺' : '↻';
    $('penalty-info').hidden = !state.penalty_amount;
    $('penalty-info').textContent = `Penalty +${state.penalty_amount}`;
    $('draw-pile-count').textContent = `Draw · ${state.grab_deck_size}`;
    const suit = suits[state.current_sort];
    const symbol = document.createElement('span');
    symbol.className = suit?.red ? 'suit-glyph red' : 'suit-glyph';
    symbol.textContent = suit?.symbol ?? '–';
    symbol.setAttribute('aria-hidden', 'true');
    $('active-suit').replaceChildren(symbol, suit?.name || (state.waiting_for_suit_choice ? 'Suit choice' : 'Wild'));

    if (discardVersion !== state.top_card) {
        paintCard($('top-card'), state.top_card);
        const discard = cardData(state.top_card);
        $('discard-label').textContent = discard ? `Discard · ${discard.value}${discard.joker ? '' : discard.symbol}` : 'Discard';
        discardVersion = state.top_card;
    }
    if (state.penalty_amount > lastPenalty) playSoundEffect('penalty');
    if (lastDirection != null && lastDirection !== state.direction) playSoundEffect('reverse');
    if (acting?.id === identity && lastTurn !== identity) playSoundEffect('turn');
    lastDirection = state.direction;
    lastTurn = acting?.id;
    lastPenalty = state.penalty_amount;

    renderHand();
    renderActions();
    if (justStarted) selectView(preferredView);
    renderScene();
    updateClock();

    if (state.game_over && !outcomeShown) {
        outcomeShown = true;
        const winner = state.first_out ?? state.players[state.winner];
        showOutcome(winner ? `${winner.name} wins` : 'Round ended',
            winner?.id === identity ? 'You were first to clear your hand.' : 'The table is closed for this round.');
    }
}

function showOutcome(title, message) {
    $('win-player-name').textContent = title;
    $('win-message').textContent = message;
    if (!winDialog.open) winDialog.showModal();
    playSoundEffect('win');
}

function gameUpdate({ action, details = {}, acting_player: player, game_state: next }) {
    pending = false;
    const oldTop = state?.top_card;
    const alreadyOut = self()?.is_observer;
    const drawsBeforeSuit = details.drawn_count > 0 && next.waiting_for_suit_choice && player?.id === identity;
    if (drawsBeforeSuit) finishingSuitDraw = true;
    displayState(next);
    const name = player?.id === identity ? 'You' : player?.name ?? 'A player';
    if (details.drawn_count) {
        const flight = animateDraw(player?.id, details.drawn_count);
        if (drawsBeforeSuit) {
            Promise.resolve(flight).then(() => {
                finishingSuitDraw = false;
                renderActions();
            });
        }
        log(`${name} drew ${pluralCards(details.drawn_count)}.`);
    }
    if (action === 'play' || action === 'win') {
        const card = cardData(next.top_card)?.label ?? 'a card';
        log(`${name} played ${card}.`);
        if (!details.drawn_count && next.top_card !== oldTop) playSoundEffect('card');
        for (const effect of details.card_effects ?? []) {
            if (effect === 'skip_player') {
                log('The next turn was skipped.');
                playSoundEffect('skip');
            }
            if (effect === 'another_turn') log(`${name} ${name === 'You' ? 'play' : 'plays'} again.`);
            if (effect === 'reverse') log('Play reversed.');
        }
        if (action === 'win') {
            log(`${name} went out.`);
            if (!alreadyOut && player?.id === identity && !next.game_over) {
                showOutcome('You’re out', 'You cleared your hand. You can watch the remaining players finish.');
            }
        }
    } else if (action === 'pass' && !details.drawn_count) {
        log(`${name} passed.`);
    }
    if (details.new_sort) log(`Suit changed to ${suits[details.new_sort]?.name ?? 'wild'}.`);
}

passButton.addEventListener('click', () => play(null));
drawPile.addEventListener('click', () => play(null));
$('add-ai-btn').addEventListener('click', () => roomAction('add_ai_player'));
$('start-game-btn').addEventListener('click', () => roomAction('start_game'));
$('leave-game').addEventListener('click', () => {
    $('leave-game').disabled = true;
    if (identity) forgetSeat(code);
    const exit = () => location.assign('/');
    if (socket.connected && identity) {
        socket.timeout(1000).emit('leave_room', { room_code: code, player_id: identity }, exit);
    } else exit();
});
$('copy-invite').addEventListener('click', async () => {
    const link = `${location.origin}/room/${code}`;
    try {
        await navigator.clipboard.writeText(link);
        toast('Invite link copied.');
    } catch {
        window.prompt('Copy this invite link', link);
    }
});

suitDialog.addEventListener('cancel', event => event.preventDefault());
for (const button of suitButtons) {
    button.addEventListener('click', () => {
        if (canAct() && state.waiting_for_suit_choice) roomAction('change_sort', { new_sort: button.dataset.sort });
    });
}

$('guest-name').value = localStorage.getItem('playerName') ?? '';
$('name-form').addEventListener('submit', event => {
    event.preventDefault();
    if (!socket.connected) return;
    const playerName = $('guest-name').value.trim();
    localStorage.setItem('playerName', playerName);
    showEntry('Joining the room…');
    socket.emit('join_room', { room_code: code, player_name: playerName });
});

socket.on('connect', () => {
    const token = resumeToken(code);
    if (token) {
        socket.emit('join_room', { room_code: code, resume_token: token });
    } else if (!identity) {
        showEntry('Enter your name to join this room.', true);
    }
});

socket.on('joined_room', data => {
    pending = false;
    seatReady = true;
    identity = data.player_id;
    rememberSeat(data);
    displayState(data.game_state, data.private_state);
});

socket.on('private_state', data => {
    if (!state?.started) return;
    handReady = true;
    hand = data.hand;
    playable = new Set(data.playable_cards);
    renderHand();
    renderActions();
});

socket.on('player_joined', data => {
    displayState(data.game_state);
    log(`${data.player_name} joined the room.`);
});
socket.on('ai_player_added', data => {
    pending = false;
    displayState(data.game_state);
    log(`${data.ai_name} joined the room.`);
});
socket.on('game_started', data => {
    pending = false;
    displayState(data.game_state);
    log('The round started.');
});
socket.on('game_update', gameUpdate);
socket.on('ai_move', gameUpdate);
socket.on('sort_changed', data => {
    pending = false;
    displayState(data.game_state);
    log(`Suit changed to ${suits[data.new_sort].name}.`);
});
socket.on('player_left', data => {
    displayState(data.game_state);
    if (data.player_name) log(`${data.player_name} ${data.disconnected ? 'disconnected' : 'left the room'}.`);
});
socket.on('player_reconnected', data => {
    displayState(data.game_state);
    log(`${data.player_name} reconnected.`);
});
socket.on('player_auto_passed', data => {
    pending = false;
    displayState(data.game_state);
    animateDraw(data.acting_player?.id, data.drawn_count);
    log(`${data.player_name} timed out and passed.`);
});
socket.on('player_kicked', data => {
    pending = false;
    displayState(data.game_state);
    log(`${data.player_name} was removed after three timeouts.`);
    if (!self()) {
        forgetSeat(code);
        toast('Your seat was removed after three timeouts.', 'error');
    }
});
socket.on('game_tied', data => {
    displayState(data.game_state);
    log('The round ended in a tie. No cards remain to draw.');
});
socket.on('error', ({ message, code: errorCode }) => {
    pending = false;
    if (errorCode === 'seat_expired') forgetSeat(code);
    if ($('room-entry').hidden) {
        toast(message, 'error');
        renderActions();
    } else {
        showEntry(message, message !== 'Room not found' && message !== 'Game already started');
    }
});
socket.on('disconnect', () => {
    seatReady = false;
    pending = false;
    renderActions();
    updateClock();
    if (!state) showEntry('Connection lost. Reconnecting…');
});
socket.on('connect_error', () => {
    if (!state) showEntry('Could not connect to Pesten. Reconnecting…');
});
socket.on('seat_moved', () => {
    socket.disconnect();
    identity = null;
    hand = [];
    playable.clear();
    state = null;
    renderHand();
    $('lobby-section').hidden = true;
    $('game-section').hidden = true;
    $('view-controls').hidden = true;
    setView('2d');
    showEntry('This seat is now open in another tab.');
});

setInterval(updateClock, 250);
