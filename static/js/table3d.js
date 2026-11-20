import {
    AmbientLight, BoxGeometry, CanvasTexture, CylinderGeometry, DirectionalLight, Group,
    Mesh, MeshBasicMaterial, MeshStandardMaterial, OrthographicCamera,
    PlaneGeometry, Scene, SRGBColorSpace, Vector3, WebGLRenderer,
} from '../vendor/three-r187.js';
import { cardData } from './cards.js';

const drawSpot = -1.1;
const discardSpot = 1.1;
const cardWidth = 1.18;
const cardLength = 1.67;

function canvasTexture(width, height, paint) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    paint(canvas.getContext('2d'));
    const texture = new CanvasTexture(canvas);
    texture.colorSpace = SRGBColorSpace;
    return texture;
}

function feltTexture() {
    return canvasTexture(512, 512, ctx => {
        const pixels = ctx.createImageData(512, 512);
        let seed = 5483;
        for (let i = 0; i < pixels.data.length; i += 4) {
            seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
            const grain = (seed >>> 27) - 16;
            pixels.data.set([31 + grain, 100 + grain, 73 + grain, 255], i);
        }
        ctx.putImageData(pixels, 0, 0);
        for (const [width, height] of [[235, 231], [232, 228]]) {
            ctx.strokeStyle = '#c7ae745f';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.ellipse(256, 256, width, height, 0, 0, Math.PI * 2);
            ctx.stroke();
        }
    });
}

function cardTexture(card) {
    return canvasTexture(192, 272, ctx => {
        ctx.fillStyle = '#eee2c9';
        ctx.beginPath();
        ctx.roundRect(0, 0, 192, 272, 11);
        ctx.fill();
        if (!card) {
            ctx.fillStyle = '#244a3c';
            ctx.fillRect(9, 9, 174, 254);
            ctx.save();
            ctx.beginPath();
            ctx.rect(12, 12, 168, 248);
            ctx.clip();
            ctx.strokeStyle = '#c0b78938';
            for (let x = -272; x < 460; x += 10) {
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x + 272, 272);
                ctx.moveTo(x, 0);
                ctx.lineTo(x - 272, 272);
                ctx.stroke();
            }
            ctx.restore();
            ctx.fillStyle = '#17372b';
            ctx.strokeStyle = '#cfb479';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(96, 136, 35, 0, Math.PI * 2);
            ctx.fill();
            ctx.stroke();
            ctx.fillStyle = '#e3c789';
            ctx.textAlign = 'center';
            ctx.font = '39px Georgia';
            ctx.fillText('♠', 96, 150);
            return;
        }

        ctx.strokeStyle = '#a48c614e';
        ctx.lineWidth = 1.5;
        ctx.strokeRect(7, 7, 178, 258);
        ctx.fillStyle = card.red ? '#a6302b' : card.joker ? '#7c582c' : '#1a3025';
        for (const rotated of [false, true]) {
            ctx.save();
            if (rotated) {
                ctx.translate(192, 272);
                ctx.rotate(Math.PI);
            }
            ctx.textAlign = 'center';
            ctx.font = 'bold 30px Georgia';
            ctx.fillText(card.value, 30, 42);
            ctx.font = '23px Georgia';
            ctx.fillText(card.symbol, 30, 66);
            ctx.restore();
        }
        ctx.textAlign = 'center';
        ctx.font = '110px Georgia';
        ctx.fillText(card.symbol, 96, 173);
    });
}

function createScene(host, { hero = false, onLost = () => {}, onPiles = () => {} } = {}) {
    const renderer = new WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    renderer.domElement.tabIndex = -1;
    host.append(renderer.domElement);

    const scene = new Scene();
    const camera = new OrthographicCamera(-6, 6, 4, -4, 0.1, 50);
    camera.position.set(0, hero ? 10 : 12.4, hero ? 10.5 : 10.9);
    camera.lookAt(0, -.22, 0);
    scene.add(new AmbientLight(0xffeedb, 1.7));
    const light = new DirectionalLight(0xffebc7, 2.3);
    light.position.set(-3, 8, 3);
    scene.add(light);

    const shadowMap = canvasTexture(256, 256, ctx => {
        const blur = ctx.createRadialGradient(128, 128, 45, 128, 128, 128);
        blur.addColorStop(0, '#000b');
        blur.addColorStop(.72, '#0007');
        blur.addColorStop(1, '#0000');
        ctx.fillStyle = blur;
        ctx.fillRect(0, 0, 256, 256);
    });
    const shadow = new Mesh(new PlaneGeometry(13.7, 10.8), new MeshBasicMaterial({ map: shadowMap, transparent: true, depthWrite: false }));
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(.28, -.92, .32);
    scene.add(shadow);

    function oval(radiusX, radiusZ, depth, y, material) {
        const mesh = new Mesh(new CylinderGeometry(1, 1, depth, 112), material);
        mesh.scale.set(radiusX, 1, radiusZ);
        mesh.position.y = y;
        scene.add(mesh);
    }
    const material = color => new MeshStandardMaterial({ color, roughness: .78 });
    const felt = feltTexture();
    oval(4.98, 3.46, .31, -.54, material(0x31231a));
    oval(5.24, 3.62, .17, -.34, material(0x201710));
    oval(5.32, 3.69, .035, -.25, material(0x95744a));
    oval(5.36, 3.72, .22, -.12, material(0x574031));
    oval(5.02, 3.44, .025, .003, material(0xb99b66));
    oval(4.96, 3.39, .026, .014, new MeshStandardMaterial({ map: felt, roughness: 1 }));

    const cardBody = new BoxGeometry(cardWidth * .98, .041, cardLength * .985);
    const cardSurface = new PlaneGeometry(cardWidth, cardLength);
    const cardEdge = material(0xe6d7b8);
    const activeEdge = material(0xe0a745);
    const backs = new MeshBasicMaterial({ map: cardTexture(null), alphaTest: .5 });
    const faceMaterials = new Map();

    function makeCard(x, y, z, rotation, face = backs, scale = 1) {
        const group = new Group();
        const edge = new Mesh(cardBody, cardEdge);
        group.add(edge);
        const faceMesh = new Mesh(cardSurface, face);
        faceMesh.rotation.x = -Math.PI / 2;
        faceMesh.position.y = .022;
        group.add(faceMesh);
        group.position.set(x, y, z);
        group.rotation.y = rotation;
        group.scale.setScalar(scale);
        scene.add(group);
        return { group, faceMesh, edge };
    }

    function faceFor(number) {
        if (!faceMaterials.has(number)) {
            faceMaterials.set(number, new MeshBasicMaterial({ map: cardTexture(cardData(number)), alphaTest: .5 }));
        }
        return faceMaterials.get(number);
    }

    const drawCards = [];
    let discard;
    const fans = [];
    if (hero) {
        for (let i = 0; i < 5; i++) makeCard(-2.9 + i * .018, .06 + i * .047, -.94, -.16, backs, 1.02);
        makeCard(-1.38, .10, .43, .38, faceFor(42), 1.65);
        makeCard(.18, .18, .28, .03, faceFor(11), 1.65);
        makeCard(1.63, .26, .43, -.3, faceFor(24), 1.65);
    } else {
        for (let i = 0; i < 5; i++) drawCards.push(makeCard(drawSpot + i * .031, .056 + i * .045, .17, -.08));
        discard = makeCard(discardSpot, .06, .17, .12);
        discard.group.visible = false;
        for (let i = 0; i < 7; i++) {
            const fan = new Group();
            for (let j = 0; j < 3; j++) {
                const card = makeCard((j - 1) * .23, .048 + j * .023, Math.abs(j - 1) * .06, (1 - j) * .14, backs, .46);
                fan.add(card.group);
                (fan.userData.cards ??= []).push(card);
            }
            scene.add(fan);
            fans.push(fan);
        }
    }

    let visible = true;
    let frame = 0;
    let disposed = false;
    let animationStart = 0;
    let reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
    const render = now => {
        frame = 0;
        if (!visible || disposed || !host.clientWidth || !host.clientHeight) return;
        if (animationStart && discard) {
            const progress = Math.min((now - animationStart) / 230, 1);
            const eased = 1 - (1 - progress) ** 3;
            discard.group.position.set(discardSpot + .75 * (1 - eased), .06 + .8 * (1 - eased), .17 + .28 * (1 - eased));
            discard.group.rotation.y = .12 + .2 * (1 - eased);
            if (progress < 1) requestRender();
            else animationStart = 0;
        }
        renderer.render(scene, camera);
    };
    const requestRender = () => {
        if (!frame && visible && !disposed) frame = requestAnimationFrame(render);
    };

    function resize() {
        const { clientWidth: width, clientHeight: height } = host;
        if (!width || !height || disposed) return;
        const scale = Math.min(width / (hero ? 11.4 : 11.65), height / (hero ? 8.3 : 6.65));
        camera.left = -width / scale / 2;
        camera.right = -camera.left;
        camera.bottom = -height / scale / 2;
        camera.top = -camera.bottom;
        camera.updateProjectionMatrix();
        camera.updateMatrixWorld();
        renderer.setSize(width, height);
        const screen = (x, y, z) => {
            const point = new Vector3(x, y, z).project(camera);
            return { x: (1 + point.x) * width / 2, y: (1 - point.y) * height / 2 };
        };
        if (!hero) {
            const draw = screen(drawSpot + .06, .17, .17);
            const top = screen(discardSpot, .06, .17);
            const height = Math.max(48, Math.abs(screen(0, .06, -cardLength / 2).y -
                screen(0, .06, cardLength / 2).y) + 4);
            onPiles(draw, top, cardWidth * scale + 8, height);
        }
        requestRender();
    }

    const sizeObserver = new ResizeObserver(resize);
    sizeObserver.observe(host);
    let lastCard;
    const lost = event => {
        event.preventDefault();
        dispose();
        onLost();
    };
    renderer.domElement.addEventListener('webglcontextlost', lost);
    resize();

    function dispose() {
        disposed = true;
        sizeObserver.disconnect();
        cancelAnimationFrame(frame);
        renderer.domElement.removeEventListener('webglcontextlost', lost);
        const resources = new Set();
        scene.traverse(item => {
            if (!item.isMesh) return;
            resources.add(item.geometry);
            resources.add(item.material);
            if (item.material.map) resources.add(item.material.map);
        });
        for (const face of faceMaterials.values()) {
            resources.add(face);
            resources.add(face.map);
        }
        resources.forEach(resource => resource.dispose());
        renderer.dispose();
        renderer.forceContextLoss();
        renderer.domElement.remove();
    }

    return {
        show(active) {
            visible = active;
            if (active) resize();
        },
        update({ topCard, deckSize, players, currentPlayerId }) {
            drawCards.forEach((card, i) => {
                card.group.visible = i < Math.max(1, Math.min(5, deckSize));
            });
            const hasCard = cardData(topCard) !== null;
            discard.group.visible = hasCard;
            if (hasCard && lastCard !== topCard) {
                discard.faceMesh.material = faceFor(topCard);
                if (lastCard != null && visible && !reducedMotion.matches) animationStart = performance.now();
                lastCard = topCard;
            }
            const active = players.filter(player => !player.is_observer).slice(0, 7);
            fans.forEach((fan, i) => {
                const player = active[i];
                fan.visible = Boolean(player);
                if (!player) return;
                const angle = Math.PI + (i - (active.length - 1) / 2) * (active.length > 4 ? .44 : .6);
                fan.position.set(Math.sin(angle) * 3.95, 0, Math.cos(angle) * 2.57);
                fan.rotation.y = Math.sin(angle) * .66;
                fan.userData.cards.forEach((card, j) => {
                    card.group.visible = j < player.hand_size;
                    card.edge.material = player.id === currentPlayerId ? activeEdge : cardEdge;
                });
            });
            requestRender();
        },
    };
}

export function mountHero(host) {
    const wrapper = host.parentElement;
    try {
        createScene(host, {
            hero: true,
            onLost: () => delete wrapper.dataset.rendered,
        });
        wrapper.dataset.rendered = '';
    } catch {
        // The CSS table remains visible when WebGL is unavailable.
    }
}

export function mountTable(host, options) {
    return createScene(host, options);
}
