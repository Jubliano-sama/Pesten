export const suits = {
    harten: { name: 'Hearts', symbol: '♥', red: true },
    ruiten: { name: 'Diamonds', symbol: '♦', red: true },
    schoppen: { name: 'Spades', symbol: '♠' },
    klaveren: { name: 'Clubs', symbol: '♣' },
    special: { name: '', symbol: '★' },
};

const ordinarySuits = Object.keys(suits).slice(0, 4);
const ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'K', 'Q'];
const rankNames = { A: 'Ace', K: 'King', Q: 'Queen', J: 'Jack' };

export function cardData(number) {
    if (!Number.isInteger(number) || number < 0 || number > 53) return null;
    const key = number < 48 ? ordinarySuits[Math.floor(number / 12)] :
        number >= 50 ? ordinarySuits[number - 50] : 'special';
    const value = number < 48 ? ranks[number % 12] : number === 49 ? '★' : 'J';
    const suit = suits[key];
    const rank = rankNames[value] ?? value;
    const label = number === 49 ? 'Joker' : suit.name ? `${rank} of ${suit.name}` : rank;
    return { ...suit, value, label, joker: number === 49 };
}

export function paintCard(element, number) {
    const card = cardData(number);
    element.replaceChildren();
    element.classList.toggle('red', Boolean(card?.red));
    element.classList.toggle('joker', Boolean(card?.joker));
    element.setAttribute('aria-label', card?.label ?? 'No discard yet');
    if (!card) return;

    const span = (className, content) => {
        const node = document.createElement('span');
        node.className = className;
        node.textContent = content;
        node.setAttribute('aria-hidden', 'true');
        return node;
    };
    const corner = span('card-corner', card.value);
    if (!card.joker) corner.append(span('', card.symbol));
    const bottom = corner.cloneNode(true);
    bottom.classList.add('card-corner-bottom');
    element.append(corner, span('card-pip', card.symbol), bottom);
}
