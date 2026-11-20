const soundToggle = document.getElementById('sound-toggle');
const musicToggle = document.getElementById('music-toggle');
let musicTimer = null;
let audioContext = null;
let masterGain = null;
let soundEffectsEnabled = localStorage.getItem('pestenSoundEffects') !== 'off';
let musicEnabled = localStorage.getItem('pestenMusic') === 'on';

function updateAudioControls() {
    if (soundToggle) {
        soundToggle.setAttribute('aria-pressed', String(soundEffectsEnabled));
        soundToggle.title = soundEffectsEnabled ? 'Mute sound effects' : 'Enable sound effects';
        const label = soundToggle.querySelector('.audio-control-label');
        if (label) label.textContent = soundEffectsEnabled ? 'Sound on' : 'Sound off';
        const icon = soundToggle.querySelector('.audio-control-icon');
        if (icon) icon.textContent = soundEffectsEnabled ? '◖' : '◦';
    }
    if (musicToggle) {
        musicToggle.setAttribute('aria-pressed', String(musicEnabled));
        musicToggle.title = musicEnabled ? 'Turn background music off' : 'Turn background music on';
        const label = musicToggle.querySelector('.audio-control-label');
        if (label) label.textContent = musicEnabled ? 'Music on' : 'Music off';
        const icon = musicToggle.querySelector('.audio-control-icon');
        if (icon) icon.textContent = musicEnabled ? '♫' : '♪';
    }
}

function ensureAudioContext() {
    if (audioContext) {
        if (audioContext.state === 'suspended') audioContext.resume().catch(() => {});
        return audioContext;
    }

    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) return null;

    try {
        audioContext = new AudioContextClass();
        masterGain = audioContext.createGain();
        masterGain.gain.value = 0.055;
        masterGain.connect(audioContext.destination);
        audioContext.resume().catch(() => {});
    } catch (error) {
        console.warn('Audio unavailable:', error);
        audioContext = null;
        masterGain = null;
    }
    return audioContext;
}

function scheduleTone(frequency, duration, startTime, volume = 0.16, wave = 'sine') {
    if (!audioContext || !masterGain) return;
    const oscillator = audioContext.createOscillator();
    const gain = audioContext.createGain();
    oscillator.type = wave;
    oscillator.frequency.setValueAtTime(frequency, startTime);
    gain.gain.setValueAtTime(0.0001, startTime);
    gain.gain.exponentialRampToValueAtTime(Math.max(volume, 0.0002), startTime + 0.015);
    gain.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);
    oscillator.connect(gain);
    gain.connect(masterGain);
    oscillator.start(startTime);
    oscillator.stop(startTime + duration + 0.025);
}

export function playSoundEffect(effect) {
    if (!soundEffectsEnabled || !audioContext || !masterGain) return;
    const now = audioContext.currentTime + 0.005;
    switch (effect) {
        case 'card':
            scheduleTone(420, 0.09, now, 0.11, 'triangle');
            scheduleTone(620, 0.12, now + 0.045, 0.08, 'sine');
            break;
        case 'turn':
            scheduleTone(330, 0.11, now, 0.08, 'sine');
            scheduleTone(494, 0.18, now + 0.08, 0.07, 'sine');
            break;
        case 'reverse':
            scheduleTone(520, 0.12, now, 0.08, 'triangle');
            scheduleTone(390, 0.12, now + 0.08, 0.08, 'triangle');
            scheduleTone(520, 0.17, now + 0.16, 0.07, 'triangle');
            break;
        case 'penalty':
            scheduleTone(180, 0.18, now, 0.1, 'sawtooth');
            scheduleTone(130, 0.23, now + 0.12, 0.08, 'sawtooth');
            break;
        case 'skip':
            scheduleTone(260, 0.08, now, 0.06, 'square');
            scheduleTone(190, 0.13, now + 0.06, 0.05, 'square');
            break;
        case 'win':
            [523, 659, 784, 1047].forEach((frequency, index) => {
                scheduleTone(frequency, 0.26, now + index * 0.13, 0.09, 'sine');
            });
            break;
        default:
            scheduleTone(440, 0.08, now, 0.06, 'sine');
    }
}

function scheduleMusicPhrase() {
    if (!musicEnabled || !audioContext || document.hidden) return;
    const now = audioContext.currentTime + 0.02;
    [196, 247, 294].forEach((frequency, index) => {
        scheduleTone(frequency, 1.8, now + index * 0.22, 0.018, 'sine');
    });
}

function startMusicLoop() {
    if (!musicEnabled || musicTimer) return;
    if (!ensureAudioContext()) return;
    scheduleMusicPhrase();
    musicTimer = window.setInterval(scheduleMusicPhrase, 2400);
}

function stopMusicLoop() {
    if (musicTimer) {
        window.clearInterval(musicTimer);
        musicTimer = null;
    }
}

function initializeAudioFromGesture() {
    if (soundEffectsEnabled || musicEnabled) ensureAudioContext();
    if (musicEnabled) startMusicLoop();
}

updateAudioControls();
document.addEventListener('pointerdown', initializeAudioFromGesture, { passive: true });
document.addEventListener('keydown', initializeAudioFromGesture, { passive: true });

soundToggle?.addEventListener('click', () => {
    soundEffectsEnabled = !soundEffectsEnabled;
    localStorage.setItem('pestenSoundEffects', soundEffectsEnabled ? 'on' : 'off');
    updateAudioControls();
    if (soundEffectsEnabled) {
        ensureAudioContext();
        playSoundEffect('turn');
    }
});

musicToggle?.addEventListener('click', () => {
    musicEnabled = !musicEnabled;
    localStorage.setItem('pestenMusic', musicEnabled ? 'on' : 'off');
    updateAudioControls();
    if (musicEnabled) {
        ensureAudioContext();
        startMusicLoop();
    } else {
        stopMusicLoop();
    }
});

document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        stopMusicLoop();
    } else if (musicEnabled && audioContext) {
        startMusicLoop();
    }
});
