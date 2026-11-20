export function resumeToken(code) {
    return localStorage.getItem(`pesten.room.${code}`);
}

export function rememberSeat({ room_code, resume_token }) {
    if (resume_token) localStorage.setItem(`pesten.room.${room_code}`, resume_token);
}

export function forgetSeat(code) {
    localStorage.removeItem(`pesten.room.${code}`);
}
