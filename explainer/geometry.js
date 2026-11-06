// The geometric post-training explainer (#gx in post_template.html), played live in the page.
//
// One story, told on one stage: the 50-step sampler (a card that travels along a track of 50 cells, one per pass of
// f_theta), the reward (VGGT-Omega and the headset calibration, compared on a top-down compass of the head), the memory
// a full backward pass would need (a GPU memory bar that overflows), and the two shortcuts (the copied gradient, then
// one step drawn with p_n). Under the stage, one line of math follows the story: the Euler update, the loss, then a
// single gradient expression that turns from the exact gradient into the copied one, then the one-step estimate, then
// the expectation that brings the copied gradient back.
//
// Every visual is a pure function of the story time t (render(t)), so the controls only move t. The story's lines are
// the .gx-lede elements: each has an id (data-id) and a length in seconds (data-dur), and the scene's choreography
// refers to them by id. Above the stage, the current line's headline and text; under it, a seek bar with one segment per
// line, grouped into the story's parts (the .gx-part buttons, each naming its first line in data-from), play/pause and
// replay. The arrows (and the keyboard's, while the explainer is on screen) step a line back or forward; a step plays
// that line and holds. It plays only while on screen, loops after a pause at the end, and starts paused for readers who
// ask for reduced motion (a step then shows its line's last frame). ?gx=<seconds> opens it paused at that time.
(() => {
    'use strict';
    const root = document.getElementById('gx');
    if (!root || !window.katex) return;
    const NS = 'http://www.w3.org/2000/svg';
    const W = 960, H = 540;
    const COL = { video: '#1d4ed8', motion: '#047857', model: '#6d28d9', grad: '#dc2626', ink: '#1f2328',
        muted: '#6b7280', rule: '#cbd5e1', faint: '#e5e7eb', ok: '#15803d' };
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ---------- helpers ----------
    const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
    const lerp = (a, b, p) => a + (b - a) * p;
    const ease = p => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);
    const smooth = p => p * p * (3 - 2 * p);
    const rad = d => d * Math.PI / 180;
    const mulberry = seed => () => { // a small seeded PRNG, so the story is the same on every load
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        let r = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
        return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
    };
    const gauss = rnd => Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());

    // ---------- the story's lines and their times ----------
    // Each line's choreography is written in its scripted seconds (0 .. data-dur). TRIM then plays stretches of that
    // script faster: [from, to, seconds] plays scripted seconds from..to in that many real seconds. Most are still
    // stretches (nothing scripted moves, so only the shortening shows); in the first part they also speed up its
    // appearance, its first step (1 s), its rollout (4 s) and its video. The continuous motions (the head turning, the wobble, the blinking and the shaking, the
    // walking figure) run on real time, so they keep their speed. Every line still stays up long enough to read.
    const TRIM = {
        noise: [[0, 3.5, 1.6], [3.5, 7, 1.9]],
        step: [[2.1, 3.5, 0.4], [3.5, 5.5, 1], [5.5, 8, 0.9]],
        steps: [[0, 0.8, 0.4], [0.8, 8.3, 4], [8.3, 10, 0.9]],
        sample: [[0, 0.5, 0.3], [0.5, 5.6, 4.3], [5.6, 7, 0.7]],
        worn: [[5.4, 8, 0.5]],
        drift: [[2.5, 7, 1.3]],
        vggt: [[3.2, 8, 0.9]],
        calib: [[3.2, 7, 0.8]],
        score: [[8, 10, 0.8]],
        g: [[3.1, 8, 2.2]],
        chain: [[1.2, 1.5, 0.1], [2, 2.5, 0.2], [3, 3.5, 0.2], [4, 5, 0.3], [9, 11, 0.9]],
        acts: [[1.1, 1.3, 0.1], [1.5, 2, 0.2], [4, 8, 1.8]],
        oom: [[0, 0.8, 0.4], [7.6, 10, 1.2]],
        copy: [[2, 2.5, 0.2], [6.8, 10, 0.9]],
        many: [[0.2, 1, 0.4], [7.3, 9, 0.9]],
        draw: [[2.3, 3, 0.3], [7.1, 9, 0.9]],
        nograd: [[7.3, 9, 0.7]],
        rerun: [[4.8, 5, 0.1], [7.5, 10, 1]],
        cancel: [[4.5, 5, 0.2], [6.5, 9, 1]],
        converge: [[10.2, 12, 0.6]],
    };
    const LINES = [...root.querySelectorAll('.gx-lede')].map(el => ({ id: el.dataset.id, script: +el.dataset.dur, el, beat: el.closest('.gx-beat') }));
    for (const l of LINES) { // the knots of the map from scripted to real seconds
        l.ka = [0]; l.kr = [0];
        for (const [a, b, keep] of (TRIM[l.id] || []).slice().sort((x, y) => x[0] - y[0])) {
            const r = l.kr.at(-1) + (a - l.ka.at(-1));     // real time at a: at the script's own pace since the last knot
            l.ka.push(a, b); l.kr.push(r, r + keep);
        }
        l.kr.push(l.kr.at(-1) + (l.script - l.ka.at(-1))); l.ka.push(l.script);
        l.dur = l.kr.at(-1);
    }
    const interp = (xs, ys, x) => { // piecewise linear, slope 1 past either end
        if (x <= xs[0]) return ys[0] + (x - xs[0]);
        for (let i = 1; i < xs.length; i++) if (x <= xs[i]) return xs[i] > xs[i - 1] ? lerp(ys[i - 1], ys[i], (x - xs[i - 1]) / (xs[i] - xs[i - 1])) : ys[i];
        return ys.at(-1) + (x - xs.at(-1));
    };
    const S = {}, D = {}, IDX = {}, LN = {};
    let total = 0;
    LINES.forEach((l, i) => { l.t0 = total; S[l.id] = total; D[l.id] = l.dur; IDX[l.id] = i; LN[l.id] = l; total += l.dur; l.t1 = total; });
    const TOTAL = total, LOOP_HOLD = 4;
    const at = (id, a = 0) => S[id] + interp(LN[id].ka, LN[id].kr, a);         // real time of line id's scripted second a
    const local = (t, id) => interp(LN[id].kr, LN[id].ka, t - S[id]);           // line id's scripted second at real time t
    const P = (t, id, a, b) => clamp((local(t, id) - a) / (b - a));            // progress through scripted seconds a..b
    // Opacity over [t0, t1]: fades in over fi after t0, out over fo before t1.
    const win = (t, t0, t1 = Infinity, fi = 0.5, fo = fi) => clamp((t - t0) / fi) * (t1 === Infinity ? 1 : clamp((t1 - t) / fo));
    const lineAt = t => { let i = 0; while (i < LINES.length - 1 && t >= LINES[i].t1) i++; return i; };

    // ---------- the sampler's schedule and the draw p_n ----------
    // The motion's cosine schedule (LVP algorithms/hunyuan15/modules/motion_noise_schedule.py) over S = 50 steps, and
    // p_n proportional to |d sigma_n| / sigma_n: the last step is drawn 25.3% of the time, step 48 10.5%.
    const NSTEP = 50;
    const phase = x => (x + 0.008) / 1.008 * Math.PI / 2;
    const sigmaOf = u => {
        const a = clamp(Math.cos(phase(u)) ** 2 / Math.cos(phase(0)) ** 2);
        return Math.sqrt(1 - a) / (Math.sqrt(a) + Math.sqrt(1 - a));
    };
    const SIGMA = Array.from({ length: NSTEP + 1 }, (_, k) => sigmaOf(1 - k / NSTEP));
    const WEIGHT = Array.from({ length: NSTEP }, (_, n) => (SIGMA[n] - SIGMA[n + 1]) / SIGMA[n]);
    const WSUM = WEIGHT.reduce((a, b) => a + b, 0);
    const PN = WEIGHT.map(w => w / WSUM);
    const PMAX = Math.max(...PN);
    const DRAWN = 48;
    const sigmaAt = s => { const k = clamp(Math.floor(s), 0, NSTEP - 1); return lerp(SIGMA[k], SIGMA[k + 1], clamp(s - k)); };

    // ---------- layout (viewBox units) ----------
    const X0 = 96, DX = 9.6, CELL_Y = 206, CELL_H = 20, CELL_W = 7;
    const cellX = n => X0 + n * DX;
    const CARD_W = 66, VID_Y = 66, MOT_Y = 140, MOT_H = 52, CARD_SPAN = 520;     // the card's centre runs X0 .. X0 + 520
    const cardX = u => X0 + u * CARD_SPAN;
    const END_X = cardX(1);
    const VID_C = VID_Y + CARD_W / 2, MOT_C = MOT_Y + MOT_H / 2;
    const BOX_X = 668, BOX_W = 100, BOX_H = 36;
    const CX = 862, CY = 128, CR = 74;                                           // the compass
    const edgeX = y => CX - Math.sqrt(CR * CR - (y - CY) ** 2) - 3;
    const BASE = 314, BAR_H = 50;                                                // the p_n bars
    const MEM_X = 96, MEM_Y = 350, BLOCK = 20, CAP = 12;                          // the memory bar holds CAP blocks
    const ORIGIN = [640, 314], GV = [190, -175];                                  // the 2D picture of the estimates

    // ---------- the stage ----------
    const stage = root.querySelector('.gx-stage');
    const svg = document.createElementNS(NS, 'svg');
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    svg.setAttribute('aria-hidden', 'true');
    stage.appendChild(svg);
    const mathLayer = document.createElement('div');
    mathLayer.className = 'gx-math';
    mathLayer.setAttribute('aria-hidden', 'true');
    stage.appendChild(mathLayer);
    const fit = () => { mathLayer.style.transform = `scale(${stage.clientWidth / W})`; };
    new ResizeObserver(fit).observe(stage);
    fit();

    const el = (tag, attrs = {}, parent = scene) => {
        const e = document.createElementNS(NS, tag);
        for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
        parent.appendChild(e);
        return e;
    };
    const set = (e, attrs) => { for (const [k, v] of Object.entries(attrs)) { const s = typeof v === 'number' ? v.toFixed(2) : v; if (e._a?.[k] !== s) { e.setAttribute(k, s); (e._a ||= {})[k] = s; } } };
    const op = (e, o) => set(e, { opacity: clamp(o), display: o <= 0.001 ? 'none' : 'inline' });

    let scene = svg;
    const defs = el('defs');
    for (const [name, c] of Object.entries(COL)) {
        const m = el('marker', { id: `gx-ah-${name}`, viewBox: '0 0 10 10', refX: 8, refY: 5, markerWidth: 3.6, markerHeight: 3.6, orient: 'auto-start-reverse' }, defs);
        el('path', { d: 'M1,1 L9,5 L1,9 Z', fill: c, stroke: c, 'stroke-width': 1, 'stroke-linejoin': 'round' }, m);
    }
    const tileClip = el('clipPath', { id: 'gx-tile' }, defs);
    el('rect', { x: 0, y: VID_Y, width: CARD_W, height: CARD_W, rx: 5 }, tileClip);
    scene = el('g', {}, svg);
    const sceneLabels = document.createElement('div');
    sceneLabels.className = 'gx-scene';
    mathLayer.appendChild(sceneLabels);
    // The shots: each a box of the scene to fill the space above the line of math (y < TOP).
    const TOP = 400;
    const SHOTS = { sampler: [14, 14, 650, 262], reward: [520, 18, 958, 252], all: [0, 0, 960, TOP] };
    const camera = box => {
        const [x0, y0, x1, y1] = box, s = Math.min(W / (x1 - x0), TOP / (y1 - y0));
        return { s, tx: (W - s * (x1 - x0)) / 2 - s * x0, ty: (TOP - s * (y1 - y0)) / 2 - s * y0 };
    };

    // KaTeX labels on the stage (in the layer over the SVG, in the same units); tex is re-typeset only when it changes.
    const texCache = new Map();
    const tex = s => { if (!texCache.has(s)) texCache.set(s, katex.renderToString(s, { throwOnError: false })); return texCache.get(s); };
    const label = (cls = '', size = 18, color = COL.ink) => {
        const d = document.createElement('div');
        d.className = `gx-m ${cls}`;
        d.style.fontSize = `${size}px`;
        d.style.color = color;
        sceneLabels.appendChild(d);
        return d;
    };
    const place = (d, x, y, o, html, anchor = 'c') => {
        const show = o > 0.001;
        if (d._show !== show) { d.style.display = show ? '' : 'none'; d._show = show; }
        if (!show) return;
        if (html !== undefined && d._html !== html) { d.innerHTML = html; d._html = html; }
        const key = `${x.toFixed(1)},${y.toFixed(1)},${anchor}`;
        if (d._key !== key) {
            const tx = anchor === 'l' ? '0' : anchor === 'r' ? '-100%' : '-50%';
            d.style.left = `${x}px`; d.style.top = `${y}px`; d.style.transform = `translate(${tx}, -50%)`; d._key = key;
        }
        const o2 = clamp(o).toFixed(3);
        if (d._o !== o2) { d.style.opacity = o2; d._o = o2; }
    };
    // Text with inline \( \) math, for the small notes.
    const mixed = s => s.split(/(\\\(.*?\\\))/).map(p => (p.startsWith('\\(') ? tex(p.slice(2, -2)) : p)).join('');

    // ---------- the track: one cell per sampling step ----------
    const gTrack = el('g');
    const cells = Array.from({ length: NSTEP }, (_, n) => el('rect', { x: cellX(n) - CELL_W / 2, y: CELL_Y, width: CELL_W, height: CELL_H, rx: 2 }, gTrack));
    const pulse = el('line', { stroke: COL.grad, 'stroke-width': 2.6, 'marker-end': 'url(#gx-ah-grad)', 'stroke-linecap': 'round' });
    const trackNote = el('text', { x: X0 - CELL_W, y: CELL_Y + CELL_H + 17, 'font-size': 12, fill: COL.muted });
    trackNote.textContent = 'each cell: one pass of the network';

    // ---------- the p_n bars and the draw ----------
    const gBars = el('g');
    const bars = Array.from({ length: NSTEP }, (_, n) => el('rect', { x: cellX(n) - CELL_W / 2, width: CELL_W, rx: 1.5 }, gBars));
    const barBase = el('line', { x1: X0 - 8, x2: cellX(NSTEP - 1) + 8, y1: BASE, y2: BASE, stroke: COL.rule, 'stroke-width': 1 }, gBars);
    const marker = el('path', { d: 'M0,0 L-6,9 L6,9 Z', fill: COL.grad });

    // ---------- the GPU memory bar ----------
    const gMem = el('g');
    const memFrame = el('rect', { x: MEM_X, y: MEM_Y, width: CAP * BLOCK + 4, height: 24, rx: 5, fill: '#fff', 'stroke-width': 1.6 }, gMem);
    const memLabel = el('text', { x: MEM_X, y: MEM_Y - 7, 'font-size': 12.5, fill: COL.muted }, gMem);
    memLabel.textContent = 'GPU memory';
    const oomLabel = el('text', { x: MEM_X + CAP * BLOCK + 4, y: MEM_Y - 7, 'font-size': 13, 'font-weight': 700, fill: COL.grad, 'text-anchor': 'end' }, gMem);
    oomLabel.textContent = 'out of memory!';
    const blocks = Array.from({ length: NSTEP }, () => el('rect', { width: BLOCK - 3, height: 18, rx: 2.5 }, gMem));
    const memNote = el('text', { 'font-size': 12, fill: COL.model }, gMem);

    // ---------- the copied gradient's arcs ----------
    const gArcs = el('g');
    const ARC0 = [END_X - 52, 52];
    const arcPath = n => { const x = cellX(n), y = CELL_Y - 2; return `M${ARC0[0]},${ARC0[1]} Q${lerp(ARC0[0], x, 0.5)},${ARC0[1] - 6} ${x},${y}`; };
    const arcs = Array.from({ length: NSTEP }, (_, n) => el('path', { d: arcPath(n), fill: 'none', stroke: COL.grad, 'stroke-width': 1.1, 'stroke-opacity': 0.55 }, gArcs));
    const arcLen = arcs.map(a => a.getTotalLength());
    const oneArc = el('path', { d: arcPath(DRAWN), fill: 'none', stroke: COL.grad, 'stroke-width': 2.4, 'marker-end': 'url(#gx-ah-grad)' });
    const oneLen = oneArc.getTotalLength();
    const counter = el('text', { x: X0 - CELL_W, y: CELL_Y - 10, 'font-size': 13, fill: COL.grad, 'font-weight': 600 });

    // ---------- the sample: the video tile over the motion tile ----------
    const gCard = el('g');
    const gTile = el('g', { 'clip-path': 'url(#gx-tile)' }, gCard);
    const sprite = el('image', { href: 'explainer/frames.jpg', y: VID_Y, height: CARD_W, width: CARD_W * 12, preserveAspectRatio: 'none' }, gTile);
    const noiseImgs = [0, 1].map(k => el('image', { href: `explainer/noise${k}.png`, x: 0, y: VID_Y, width: CARD_W, height: CARD_W, preserveAspectRatio: 'none', style: 'image-rendering: pixelated' }, gTile));
    el('rect', { x: 0, y: VID_Y, width: CARD_W, height: CARD_W, rx: 5, fill: 'none', stroke: COL.video, 'stroke-width': 2.2 }, gCard);
    el('rect', { x: 0, y: MOT_Y, width: CARD_W, height: MOT_H, rx: 5, fill: '#ecfdf5', stroke: COL.motion, 'stroke-width': 2.2 }, gCard);
    const BONES = [['neck', 'hip'], ['neck', 'le'], ['le', 'lh'], ['neck', 're'], ['re', 'rh'], ['hip', 'lk'], ['lk', 'lf'], ['hip', 'rk'], ['rk', 'rf']];
    const bones = BONES.map(() => el('line', { stroke: COL.motion, 'stroke-width': 2.4, 'stroke-linecap': 'round' }, gCard));
    const headDot = el('circle', { r: 4, fill: COL.motion }, gCard);
    const JOINTS = ['head', 'neck', 'hip', 'lk', 'lf', 'rk', 'rf', 'le', 'lh', 're', 'rh'];
    const JITTER = [1, 2].map(seed => { const r = mulberry(seed * 977); return JOINTS.map(() => [gauss(r), gauss(r)]); });
    const vidLabel = el('text', { 'font-size': 12, fill: COL.video, 'text-anchor': 'end' });
    vidLabel.textContent = 'video';
    const motLabel = el('text', { 'font-size': 12, fill: COL.motion, 'text-anchor': 'end' });
    motLabel.textContent = 'motion';
    const gModel = el('g');
    const modelBox = el('rect', { x: -27, y: 22, width: 54, height: 30, rx: 7, 'stroke-width': 2, stroke: COL.model }, gModel);
    el('line', { x1: 0, y1: 53, x2: 0, y2: VID_Y - 3, stroke: COL.model, 'stroke-width': 1.8, 'marker-end': 'url(#gx-ah-model)' }, gModel);

    function pose(phi) { // a side-on stick figure climbing, in the motion tile's coordinates
        const p = {}, hip = [32, MOT_Y + 28], neck = [35, MOT_Y + 14];
        p.hip = hip; p.neck = neck; p.head = [37, MOT_Y + 8];
        for (const [side, off] of [['l', 0], ['r', Math.PI]]) {
            const a = 0.5 * Math.sin(phi + off), k = 0.35 + 0.55 * Math.max(0, Math.sin(phi + off + 0.6));
            const knee = [hip[0] + 10 * Math.sin(a), hip[1] + 10 * Math.cos(a)];
            p[side + 'k'] = knee; p[side + 'f'] = [knee[0] + 10 * Math.sin(a - k), knee[1] + 10 * Math.cos(a - k)];
            const c = -0.45 * Math.sin(phi + off);
            const elbow = [neck[0] + 7 * Math.sin(c), neck[1] + 7 * Math.cos(c)];
            p[side + 'e'] = elbow; p[side + 'h'] = [elbow[0] + 7 * Math.sin(c + 0.7), elbow[1] + 7 * Math.cos(c + 0.7)];
        }
        return p;
    }

    // ---------- the reward: VGGT-Omega, the calibration and the compass ----------
    const gReward = el('g');
    const box = (y, title, sub, color, fill) => {
        const g = el('g', {}, gReward);
        el('rect', { x: BOX_X, y: y - BOX_H / 2, width: BOX_W, height: BOX_H, rx: 7, fill, stroke: color, 'stroke-width': 1.8 }, g);
        const a = el('text', { x: BOX_X + BOX_W / 2, y: y - 2, 'font-size': 14, 'font-weight': 600, fill: color, 'text-anchor': 'middle' }, g);
        a.textContent = title;
        const b = el('text', { x: BOX_X + BOX_W / 2, y: y + 11, 'font-size': 10.5, fill: COL.muted, 'text-anchor': 'middle' }, g);
        b.textContent = sub;
        return g;
    };
    const vggtBox = box(VID_C, 'VGGT-Ω', 'frozen 3D model', COL.video, '#eff6ff');
    const calBox = box(MOT_C, 'calibration', 'of the headset', COL.motion, '#ecfdf5');
    const arrow = (color, width = 1.8) => el('line', { stroke: color, 'stroke-width': width, 'marker-end': `url(#gx-ah-${color === COL.grad ? 'grad' : color === COL.video ? 'video' : 'motion'})` }, gReward);
    const fwd = { v1: arrow(COL.video), v2: arrow(COL.video), m1: arrow(COL.motion), m2: arrow(COL.motion) };
    const back = { v1: arrow(COL.grad, 2), v2: arrow(COL.grad, 2), m1: arrow(COL.grad, 2), m2: arrow(COL.grad, 2) };
    const grow = (e, x1, y1, x2, y2, p, o = 1) => { op(e, p > 0.01 ? o : 0); set(e, { x1, y1, x2: lerp(x1, x2, p), y2: lerp(y1, y2, p) }); };

    const gCompass = el('g');
    el('circle', { cx: CX, cy: CY, r: CR, fill: '#fafafa', stroke: COL.faint, 'stroke-width': 1.5 }, gCompass);
    const cone = el('path', { fill: COL.video, 'fill-opacity': 0.14, stroke: COL.video, 'stroke-width': 1.6 }, gCompass);
    const arc = el('path', { fill: 'none', stroke: COL.grad, 'stroke-width': 2.4 }, gCompass);
    const angleText = el('text', { 'font-size': 13, 'font-weight': 700, fill: COL.grad, 'text-anchor': 'middle', 'dominant-baseline': 'central' }, gCompass);
    const headArrow = el('line', { x1: CX, y1: CY, stroke: COL.motion, 'stroke-width': 2.6, 'marker-end': 'url(#gx-ah-motion)' }, gCompass);
    const gHead = el('g', {}, gCompass);
    el('path', { d: 'M-4.5,-14 L0,-20 L4.5,-14 Z', fill: '#e2e8f0', stroke: '#64748b', 'stroke-width': 1.2 }, gHead);
    el('circle', { r: 15, fill: '#f1f5f9', stroke: '#64748b', 'stroke-width': 1.4 }, gHead);
    el('rect', { x: -11, y: -14.5, width: 22, height: 4.2, rx: 2, fill: '#334155' }, gHead);
    const above = el('text', { x: CX, y: CY + CR + 16, 'font-size': 11.5, fill: COL.muted, 'text-anchor': 'middle' });
    above.textContent = 'the head, seen from above';
    const dir = (deg, r) => [CX + r * Math.sin(rad(deg)), CY - r * Math.cos(rad(deg))];

    // Yaw over a sweep of the video's 49 frames (the head's C^k and the video's camera R^k, top-down), illustrative.
    const NK = 49;
    const psiK = k => 30 * Math.sin(2 * Math.PI * k / 32), delK = k => 14 + 9 * Math.sin(2 * Math.PI * k / 19 + 0.8);
    const DEL_K = Array.from({ length: NK }, (_, k) => delK(k));
    const MEAN_K = DEL_K.map((_, k) => DEL_K.slice(0, k + 1).reduce((a, b) => a + b, 0) / (k + 1));
    const meanK = kf => { const i = Math.min(NK - 2, Math.floor(kf)); return lerp(MEAN_K[i], MEAN_K[i + 1], clamp(kf - i)); };

    // ---------- the 2D picture of the estimates ----------
    const gPlot = el('g');
    el('circle', { cx: ORIGIN[0], cy: ORIGIN[1], r: 3, fill: COL.ink }, gPlot);
    const NUPD = 14;
    const hats = Array.from({ length: NUPD }, () => el('line', { x1: ORIGIN[0], y1: ORIGIN[1], stroke: COL.grad, 'stroke-width': 1.6, 'marker-end': 'url(#gx-ah-grad)' }, gPlot));
    const gArrow = el('line', { x1: ORIGIN[0], y1: ORIGIN[1], x2: ORIGIN[0] + GV[0], y2: ORIGIN[1] + GV[1], stroke: COL.ink, 'stroke-width': 3.2, 'marker-end': 'url(#gx-ah-ink)' }, gPlot);
    const meanArrow = el('line', { x1: ORIGIN[0], y1: ORIGIN[1], stroke: COL.model, 'stroke-width': 3.6, 'marker-end': 'url(#gx-ah-model)' }, gPlot);
    const updText = el('text', { x: ORIGIN[0] - 6, y: ORIGIN[1] + 22, 'font-size': 12.5, fill: COL.muted }, gPlot);
    // Each step's term of G_copy, T_n = p_n (G + e_n), with the e_n weighted to mean zero, so sum_n T_n = G exactly and
    // the estimate T_n / p_n = G + e_n; the draws are the one the story makes (step 48) and then a seeded sample of p.
    const E = (() => {
        const r = mulberry(31), e = PN.map(() => { const a = r() * 2 * Math.PI, m = 0.15 + 0.3 * r(); return [m * Math.cos(a), m * Math.sin(a)]; });
        const mx = e.reduce((s, v, n) => s + PN[n] * v[0], 0), my = e.reduce((s, v, n) => s + PN[n] * v[1], 0);
        return e.map(([x, y]) => [x - mx, y - my]);
    })();
    const sampleP = r => { let u = r(), n = 0; while (n < NSTEP - 1 && u > PN[n]) { u -= PN[n]; n++; } return n; };
    const DRAWS = (() => { // the seed whose running mean ends closest to G_copy, of the first 300
        let best = null;
        for (let seed = 1; seed <= 300; seed++) {
            const r = mulberry(seed), d = [DRAWN, ...Array.from({ length: NUPD - 1 }, () => sampleP(r))];
            const m = [0, 1].map(c => d.reduce((s, n) => s + E[n][c], 0) / d.length);
            const err = Math.hypot(m[0], m[1]);
            if (!best || err < best.err) best = { err, d };
        }
        return best.d;
    })();
    const GLEN = Math.hypot(GV[0], GV[1]);
    const hatVec = n => [GV[0] + GLEN * E[n][0], GV[1] + GLEN * E[n][1]];

    // ---------- stage labels (KaTeX) ----------
    const L = {
        model: label('', 19, COL.model), state: label('', 17, COL.ink), chip: label('gx-chip', 17, COL.grad),
        rhat: label('', 17, COL.video), ck: label('', 17, COL.motion), head: label('gx-word', 12.5, COL.motion),
        cam: label('gx-word', 12.5, COL.video), score: label('', 18, COL.ink), pn: label('', 15, COL.model),
        drawn: label('', 15, COL.grad), weight: label('', 15, COL.grad), gcopy: label('', 17, COL.ink),
        ghat: label('', 16, COL.grad), mean: label('gx-word', 12.5, COL.model), frame: label('', 13, COL.video),
    };

    // ---------- the line of math under the stage ----------
    const band = document.createElement('div');
    band.className = 'gx-band';
    mathLayer.appendChild(band);
    const row = (pieces, note) => {
        const box = document.createElement('div');
        box.className = 'gx-row';
        band.appendChild(box);
        const r = document.createElement('div');
        r.className = 'gx-eq';
        box.appendChild(r);
        const out = {};
        for (const p of pieces) {
            const w = document.createElement('span'); w.className = 'gx-piece';
            const inner = document.createElement('span'); inner.className = 'gx-inner';
            inner.innerHTML = tex(String.raw`\displaystyle ${p.tex}`);
            if (p.color) inner.style.color = p.color;
            const strike = document.createElement('span'); strike.className = 'gx-strike';
            w.append(inner, strike);
            const labels = (p.labels || []).map(l => {
                const d = document.createElement('span');
                d.className = `gx-brace ${l.side || 'below'}`;
                d.style.color = l.color || COL.muted;
                d.innerHTML = `<i></i><b>${mixed(l.text)}</b>`;
                w.appendChild(d);
                return { d, show: l.show, row: typeof l.row === 'function' ? l.row : () => l.row || 0 };
            });
            r.appendChild(w);
            out[p.id] = { w, inner, strike, labels, color: p.hl || COL.grad, nat: 0 };
        }
        let noteEl = null;
        if (note) { noteEl = document.createElement('div'); noteEl.className = 'gx-note'; noteEl.innerHTML = mixed(note); box.appendChild(noteEl); }
        return { box, pieces: out };
    };
    const show = (a, b = Infinity) => t => win(t, a, b, 0.5); // label visibility windows, in story seconds
    const ROWS = {
        euler: row([{ id: 'e', tex: String.raw`y_{n+1} = y_n + \Delta_n\, f_\theta(y_n)` }],
            String.raw`\(y_n = (z_n, x_n)\): the video latent and the motion at step \(n\)`),
        score: row([{ id: 's', tex: String.raw`\mathcal{L}_{\text{geom}} = \frac{1}{N+1}\sum_{k=0}^{N}\angle\big(\hat R^k,\, C^k\big)` }],
            String.raw`the angle between the video's camera \(\hat R^k\) and the head's camera \(C^k\), averaged over the \(N + 1 = 49\) frames`),
        grad: row([
            { id: 'lhsE', tex: String.raw`\mathbb{E}_{n\sim p}\big[\hat G\big]` },
            { id: 'lhsX', tex: String.raw`\frac{d\mathcal{L}_{\text{geom}}}{d\theta}` },
            { id: 'lhsC', tex: String.raw`G_{\text{copy}}` },
            { id: 'lhsH', tex: String.raw`\hat G` },
            { id: 'eq', tex: '=' },
            { id: 'sum', tex: String.raw`\sum_{n=0}^{S-1}`, hl: COL.model, labels: [
                { text: 'one term per step', side: 'above', show: show(at('chain', 1.5), at('copy', 0.5)) },
                { text: '50 terms: 50 backward passes', side: 'above', color: COL.grad, show: show(at('many', 1), at('draw', 0.5)) }] },
            { id: 'pn', tex: String.raw`p_n \,\cdot`, color: COL.model, labels: [
                { text: 'chance of drawing \\(n\\)', color: COL.model, show: show(at('cancel', 2), at('cancel', 3)) }] },
            { id: 'inv', tex: String.raw`\frac{1}{p_n}`, color: COL.grad, labels: [
                { text: 'weight', row: 1, color: COL.grad, show: show(at('rerun', 7), at('cancel', 3)) }] },
            { id: 'g', tex: 'g', color: COL.grad, labels: [
                { text: 'endpoint gradient', row: 1, color: COL.grad, show: show(at('chain', 2.5), at('rerun', 5)) }] },
            { id: 'chain', tex: String.raw`\frac{\partial y_S}{\partial y_{S-1}}\cdots\frac{\partial y_{n+2}}{\partial y_{n+1}}`, labels: [
                { text: 'back through every later step', color: COL.grad, show: show(at('chain', 5), at('copy', 1)) }] },
            { id: 'jac', tex: String.raw`\frac{\partial y_{n+1}}{\partial\theta}`, color: COL.model, hl: COL.model, labels: [
                { text: 'step \\(n\\)’s Jacobian', row: t => (t < at('copy', 3) ? 1 : 0), color: COL.model,
                  show: t => Math.max(win(t, at('chain', 3.5), at('copy', 2), 0.5), win(t, at('copy', 4.5), at('rerun', 5), 0.5)) }] },
            { id: 'tail', tex: String.raw`=\; G_{\text{copy}}` },
        ]),
    };
    const measure = () => { for (const r of Object.values(ROWS)) for (const p of Object.values(r.pieces)) { p.inner.style.maxWidth = 'none'; p._w = null; p.nat = p.inner.scrollWidth; } };
    measure();
    document.fonts?.ready.then(() => { measure(); dirty = true; });
    // A piece's state: present (0..1), struck through (0..1), highlighted (0..1). A piece that leaves first fades, then
    // gives up its width; one that arrives first takes its width, then fades in; so the formula reflows between them.
    const piece = (p, present, strike = 0, hl = 0, t = 0) => {
        const w = (p.nat * smooth(clamp(present / 0.6))).toFixed(1);
        if (p._w !== w) { p.inner.style.maxWidth = `${w}px`; p._w = w; }
        const o = clamp((present - 0.4) / 0.6).toFixed(3);
        if (p._o !== o) { p.inner.style.opacity = o; p._o = o; }
        const s = `${(clamp(strike) * 100).toFixed(1)}%`, so = present > 0.6 && strike > 0 ? '1' : '0';
        if (p._s !== s + so) { p.strike.style.width = s; p.strike.style.opacity = so; p._s = s + so; }
        const h = clamp(hl).toFixed(3);
        if (p._h !== h) { p.w.style.backgroundColor = hl > 0 ? hexA(p.color, 0.13 * hl) : 'transparent'; p._h = h; }
        for (const l of p.labels) {
            const v = (l.show(t) * clamp((present - 0.4) / 0.6)).toFixed(3); if (l._v !== v) { l.d.style.opacity = v; l._v = v; }
            const r = l.row(t); if (l._r !== r) { l.d.classList.toggle('row1', r === 1); l._r = r; }
        }
    };
    const hexA = (hex, a) => { const n = parseInt(hex.slice(1), 16); return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`; };
    const rowOp = (r, o) => { const v = clamp(o).toFixed(3); if (r._o !== v) { r.box.style.opacity = v; r.box.style.visibility = o > 0.001 ? 'visible' : 'hidden'; r._o = v; } };

    // ---------- the scene at time t ----------
    function cardU(t) {
        if (t < at('nograd', 0.6)) {
            if (t < at('steps')) return P(t, 'step', 3.5, 5.5) / NSTEP;
            return lerp(1 / NSTEP, 1, P(t, 'steps', 0.8, 8.3));
        }
        return P(t, 'nograd', 1.2, 7.2);
    }
    // The yaw of the head and of the video's camera on the compass, and the frame k shown (or -1).
    function yaw(t) {
        const free = t0 => 30 * Math.sin(2 * Math.PI * Math.max(0, t - t0) / 4.2);
        if (t < at('score')) {
            const psi = free(at('worn', 1.2));
            const d = 26 * smooth(P(t, 'drift', 0.5, 2.5)) + 3 * Math.sin(t * 1.7) * smooth(P(t, 'drift', 0.5, 2.5));
            return { psi, chi: psi + d, k: -1 };
        }
        const kf = (NK - 1) * P(t, 'score', 2, 8), k = Math.round(kf);   // the compass turns smoothly; frames step
        const w = smooth(P(t, 'score', 0.3, 1.8));
        const freePsi = free(at('worn', 1.2)), freeD = 26 + 3 * Math.sin(t * 1.7);
        const psi = lerp(freePsi, psiK(kf), w), d = lerp(freeD, delK(kf), w);
        return { psi, chi: psi + d, k: t < at('score', 2) ? -1 : k, kf };
    }

    function render(t) {
        const li = lineAt(t), cur = LINES[li].id, tau = local(t, cur);
        const on = id => cur === id, past = id => li > IDX[id];

        // ---- the camera: the sampler, then the reward, then everything
        const toReward = ease(P(t, 'worn', 0, 1.4)), toAll = ease(P(t, 'g', 0, 1.4));
        const shot = SHOTS.sampler.map((v, i) => lerp(lerp(v, SHOTS.reward[i], toReward), SHOTS.all[i], toAll));
        const cam = camera(shot);
        set(scene, { transform: `translate(${cam.tx},${cam.ty}) scale(${cam.s.toFixed(4)})` });
        const camKey = `${cam.tx.toFixed(2)},${cam.ty.toFixed(2)},${cam.s.toFixed(4)}`;
        if (sceneLabels._k !== camKey) { sceneLabels.style.transform = `translate(${cam.tx}px, ${cam.ty}px) scale(${cam.s})`; sceneLabels._k = camKey; }

        // ---- the sample
        const u = cardU(t), x = cardX(u), sig = sigmaAt(u * NSTEP);
        const resetDip = on('nograd') ? (tau < 0.6 ? 1 - tau / 0.6 : clamp((tau - 0.6) / 0.6)) : 1;
        op(gCard, P(t, 'noise', 0.3, 1.2) * resetDip);
        set(gCard, { transform: `translate(${x - CARD_W / 2},0)` });
        const second = t >= at('nograd', 0.6);
        op(noiseImgs[0], second ? 0 : Math.pow(sig, 0.85));
        op(noiseImgs[1], second ? Math.pow(sig, 0.85) : 0);
        const { psi, chi, k, kf } = yaw(t);
        let frame = 11;
        if (t < at('sample', 0.5) || second) frame = 0;
        else if (on('sample')) frame = Math.min(11, Math.floor((tau - 0.5) / 0.42));
        else if (k >= 0 && t < at('g')) frame = Math.round(k / (NK - 1) * 11);
        set(sprite, { x: -frame * CARD_W });
        const walking = on('sample') ? (t - at('sample')) * 5.2 : k >= 0 && t < at('g') ? k * 0.5 : 0.6;
        const jit = JITTER[second ? 1 : 0];
        const p = pose(walking);
        const J = Object.fromEntries(JOINTS.map((j, i) => [j, [p[j][0] + sig * 11 * (jit[i][0] + 0.25 * Math.sin(2.1 * t + i)), p[j][1] + sig * 9 * (jit[i][1] + 0.25 * Math.cos(1.7 * t + 2 * i))]]));
        BONES.forEach(([a, b], i) => set(bones[i], { x1: J[a][0], y1: J[a][1], x2: J[b][0], y2: J[b][1] }));
        set(headDot, { cx: J.head[0], cy: J.head[1] });
        const tileWords = win(t, at('noise', 1.5), at('g', 0.5), 0.6);
        op(vidLabel, tileWords * resetDip); set(vidLabel, { x: x - CARD_W / 2 - 6, y: VID_C + 4 });
        op(motLabel, tileWords * resetDip); set(motLabel, { x: x - CARD_W / 2 - 6, y: MOT_C + 4 });

        // ---- the network f_theta, riding on the sample while it is sampled
        const rolling = (on('step') && tau >= 3.5 && tau < 5.6) || (on('steps') && tau >= 0.8 && tau < 8.4) || (on('nograd') && tau >= 1.2 && tau < 7.3);
        const modelO = Math.max(win(t, at('step', 0.3), at('worn', 0.6), 0.8), on('nograd') ? win(t, at('nograd', 0.6), at('nograd', 8.2), 0.5) : 0);
        op(gModel, modelO);
        set(gModel, { transform: `translate(${x},0)` });
        const beat = rolling ? 1 - ((u * NSTEP) % 1) : 0;
        set(modelBox, { fill: hexA(COL.model, 0.08 + (second ? 0.08 : 0.3) * beat) });
        place(L.model, x, 37, modelO, tex(String.raw`f_\theta`));
        const n = Math.min(NSTEP, Math.floor(u * NSTEP + 1e-6));
        const stateTex = u >= 0.999 ? 'y_S' : `y_{${n}}`;
        const stateO = Math.max(win(t, at('noise', 1.5), at('worn', 0.6), 0.6), on('nograd') ? win(t, at('nograd', 0.6), at('nograd', 8.2), 0.5) : 0);
        place(L.state, x, CELL_Y + CELL_H + 18, stateO, tex(stateTex));

        // ---- the track's cells
        op(gTrack, win(t, at('noise', 2.5), Infinity, 1));
        op(trackNote, win(t, at('steps', 1.5), at('worn'), 0.6));
        const pulseX = lerp(cellX(NSTEP - 1) + 6, cellX(0) - 8, ease(P(t, 'chain', 5, 9)));
        const flashN = nn => clamp(1 - Math.abs(nn) / 0.25);
        for (let c = 0; c < NSTEP; c++) {
            let fill = '#ffffff', stroke = COL.rule, dash = 'none';
            const ran = c < u * NSTEP;
            if (li <= IDX.score) { if (ran) { fill = '#ede9fe'; stroke = COL.model; } }
            else if (on('g')) { fill = '#ede9fe'; stroke = COL.model; }
            else if (on('chain')) { fill = '#ede9fe'; stroke = COL.model; if (tau >= 5 && cellX(c) >= pulseX) { fill = '#fee2e2'; stroke = COL.grad; } }
            else if (on('acts')) { if (c === NSTEP - 1 && tau > 1.3) { fill = '#ddd6fe'; stroke = COL.model; } }
            else if (on('oom')) { if (tau >= blockTime(c)) { fill = '#ede9fe'; stroke = COL.model; } }
            else if (on('copy')) { if (tau >= arcTime(c) + 0.5) { stroke = COL.grad; fill = '#fff1f2'; } }
            else if (on('many')) {
                stroke = COL.grad; fill = '#fff1f2';
                const ft = 1.5 + (NSTEP - 1 - c) * 0.11;
                if (tau >= ft) fill = '#fecaca';
                if (flashN(tau - ft - 0.1) > 0) fill = COL.grad;
            }
            else if (on('draw')) { /* neutral */ }
            else if (on('nograd')) { if (ran) { fill = '#f3f4f6'; stroke = '#9ca3af'; dash = '2 1.5'; } if (c === DRAWN && ran) { stroke = COL.model; dash = 'none'; } }
            else { // rerun, cancel, converge
                fill = '#f3f4f6'; stroke = '#9ca3af'; dash = '2 1.5';
                if (c === DRAWN) { stroke = COL.model; dash = 'none'; if (!on('rerun') || tau >= 2.8) { fill = '#fecaca'; stroke = COL.grad; } }
                if (on('converge')) {
                    for (let j = 0; j < NUPD; j++) if (DRAWS[j] === c && flashN(tau - updTime(j) - 0.1) > 0) { fill = COL.grad; stroke = COL.grad; }
                }
            }
            set(cells[c], { fill, stroke, 'stroke-dasharray': dash, 'stroke-width': 1.3 });
        }
        op(pulse, on('chain') ? (tau >= 5 ? 1 : 0) : on('acts') ? 1 - clamp(tau / 0.6) : 0);
        set(pulse, { x1: cellX(NSTEP - 1) + 6, y1: CELL_Y + CELL_H + 6, x2: pulseX, y2: CELL_Y + CELL_H + 6 });

        // ---- the reward
        op(gReward, win(t, at('vggt', 0.3), at('converge', 0.8), 0.6));
        op(vggtBox, P(t, 'vggt', 0.3, 1));
        op(calBox, P(t, 'calib', 0.3, 1));
        const atEnd = smooth(clamp((u - 0.97) / 0.03)) * (on('nograd') ? 0 : 1);
        const vy = VID_C - 5, my = MOT_C - 5;
        grow(fwd.v1, END_X + CARD_W / 2 + 3, vy, BOX_X - 3, vy, P(t, 'vggt', 1, 1.8), atEnd);
        grow(fwd.v2, BOX_X + BOX_W + 3, vy, edgeX(vy), vy, P(t, 'vggt', 1.8, 2.6), atEnd);
        grow(fwd.m1, END_X + CARD_W / 2 + 3, my, BOX_X - 3, my, P(t, 'calib', 1, 1.8), atEnd);
        grow(fwd.m2, BOX_X + BOX_W + 3, my, edgeX(my), my, P(t, 'calib', 1.8, 2.6), atEnd);
        const by = VID_C + 7, bm = MOT_C + 7;
        const backO = (on('nograd') ? 0 : 1) * (on('rerun') ? P(t, 'rerun', 0.2, 0.6) : 1);
        const bp = on('rerun') ? [P(t, 'rerun', 0.3, 0.8), P(t, 'rerun', 0.8, 1.3)] : [P(t, 'g', 0.5, 1.5), P(t, 'g', 1.5, 2.5)];
        grow(back.v1, edgeX(by), by, BOX_X + BOX_W + 3, by, bp[0], backO);
        grow(back.v2, BOX_X - 3, by, END_X + CARD_W / 2 + 3, by, bp[1], backO);
        grow(back.m1, edgeX(bm), bm, BOX_X + BOX_W + 3, bm, bp[0], backO);
        grow(back.m2, BOX_X - 3, bm, END_X + CARD_W / 2 + 3, bm, bp[1], backO);

        // ---- the compass
        const compassO = win(t, at('worn', 0.3), at('converge', 0.8), 0.8);
        op(gCompass, compassO);
        set(gHead, { transform: `translate(${CX},${CY}) rotate(${psi})` });
        const [hx, hy] = dir(psi, CR - 10);
        set(headArrow, { x2: hx, y2: hy });
        const half = 17, rc = CR - 4;
        const [c1x, c1y] = dir(chi - half, rc), [c2x, c2y] = dir(chi + half, rc);
        set(cone, { d: `M${CX},${CY} L${c1x},${c1y} A${rc},${rc} 0 0 1 ${c2x},${c2y} Z` });
        const dAng = chi - psi, arcO = on('worn') ? 0 : clamp((Math.abs(dAng) - 2) / 4) * (t >= at('drift', 0.5) ? 1 : 0);
        const ra = 44, [a1x, a1y] = dir(psi, ra), [a2x, a2y] = dir(chi, ra);
        set(arc, { d: `M${a1x},${a1y} A${ra},${ra} 0 0 ${dAng > 0 ? 1 : 0} ${a2x},${a2y}` });
        op(arc, arcO);
        set(angleText, { x: CX, y: CY + 40 });   // in the compass's empty lower half, clear of the names round the rim
        angleText.textContent = `${Math.round(Math.abs(dAng))}° error`;
        op(angleText, arcO);
        op(above, win(t, at('worn', 1.3), at('score', 1), 0.6));
        const [hlx, hly] = dir(psi - 24, CR + 17), [clx, cly] = dir(chi + 24, CR + 17);
        place(L.head, hlx, hly, compassO * (1 - P(t, 'calib', 2.4, 3)), 'head');
        place(L.cam, clx, cly, compassO * (1 - P(t, 'vggt', 2.4, 3)), 'camera');
        place(L.rhat, clx, cly, compassO * P(t, 'vggt', 2.6, 3.2), tex(String.raw`\hat R^k`));
        place(L.ck, hlx, hly, compassO * P(t, 'calib', 2.6, 3.2), tex('C^k'));
        const meanNow = k >= 0 ? meanK(kf) : MEAN_K[NK - 1];
        const rescore = on('rerun') ? 1 + 0.6 * Math.sin(Math.PI * P(t, 'rerun', 0, 0.8)) : 1;
        place(L.score, CX, CY + CR + 20, compassO * P(t, 'score', 2, 2.6),
            `<span style="display:inline-block;transform:scale(${rescore.toFixed(3)})">${tex(String.raw`\mathcal{L}_{\text{geom}} = ${meanNow.toFixed(1)}^\circ`)}</span>`);
        place(L.frame, END_X, VID_Y - 10, k >= 0 && t < at('g') ? 1 : 0, tex(`\\text{frame } k = ${k}`));

        // ---- the endpoint gradient g, on the finished sample
        const chipO = Math.max(win(t, at('g', 2.5), at('nograd', 0.5), 0.6), win(t, at('rerun', 1), Infinity, 0.5));
        place(L.chip, END_X, 36, chipO, tex(String.raw`g = \partial\mathcal{L}_{\text{geom}} / \partial y_S`));

        // ---- the GPU memory
        op(gMem, Math.max(win(t, at('acts', 0.3), at('copy', 0.9), 0.8, 0.6) , win(t, at('nograd', 0.8), Infinity, 0.6)));
        const overflow = on('oom') ? clamp((tau - blockTime(CAP)) / 0.3) : 0;
        const fits = on('nograd') || on('rerun') || on('cancel') || on('converge');
        set(memFrame, { stroke: overflow > 0 ? COL.grad : fits ? COL.ok : '#94a3b8', 'stroke-width': overflow > 0 ? 2.4 : 1.6 });
        op(oomLabel, overflow * (0.65 + 0.35 * Math.sin(t * 9)));
        const shake = overflow * 2.2 * Math.sin(t * 47);
        set(gMem, { transform: `translate(${shake},0)` });
        for (let b = 0; b < NSTEP; b++) {
            let o = 0, bx = MEM_X + 3 + b * BLOCK, byy = MEM_Y + 3, w = BLOCK - 3, color = '#8b5cf6';
            if (on('acts') && b === 0) {
                const q = P(t, 'acts', 2, 3.4);
                o = q > 0 ? 1 : 0;
                bx = lerp(cellX(NSTEP - 1) - 8, bx, ease(q)); byy = lerp(CELL_Y, byy, ease(q));
            } else if (on('oom')) {
                const q = clamp((tau - blockTime(b)) / 0.35);
                o = q > 0 ? 1 : 0;
                bx = lerp(cellX(b) - 8, bx, ease(q)); byy = lerp(CELL_Y, byy, ease(q));
                if (b >= CAP) color = COL.grad;
            } else if (on('copy')) {
                o = b < NSTEP ? 1 - clamp(tau / 0.8) : 0; if (b >= CAP) color = COL.grad;
            } else if ((on('nograd') && u * NSTEP > DRAWN) || (on('rerun') && tau < 3)) {
                if (b === 0) { o = 1; w = 4; }
            } else if (on('rerun') || on('cancel') || on('converge')) {
                if (b === 0) {
                    const q = on('rerun') ? P(t, 'rerun', 3, 3.6) : 1;
                    o = 1; bx = lerp(cellX(DRAWN) - 8, bx, ease(q)); byy = lerp(CELL_Y, byy, ease(q));
                }
            }
            op(blocks[b], o);
            if (o > 0) set(blocks[b], { x: bx, y: byy, width: w, fill: color, 'fill-opacity': 0.8 });
        }
        let note = '', noteO = 0, noteColor = COL.model;
        if (on('acts')) { note = 'one step’s activations'; noteO = P(t, 'acts', 3.4, 4); }
        else if ((on('nograd') && u * NSTEP > DRAWN) || (on('rerun') && tau < 3)) { note = 'only step n’s input'; noteO = 1; }
        else if (on('rerun') || on('cancel') || on('converge')) { note = '✓ one step’s activations: it fits'; noteO = on('rerun') ? P(t, 'rerun', 3.6, 4.2) : 1; noteColor = COL.ok; }
        memNote.textContent = note;
        set(memNote, { x: MEM_X + (on('acts') || (on('rerun') && tau >= 3) || on('cancel') || on('converge') ? BLOCK + 8 : 12), y: MEM_Y + 16.5, fill: noteColor });
        op(memNote, noteO);

        // ---- copying g to every step, and counting the backward passes
        op(gArcs, on('copy') ? 1 : on('many') ? 1 : on('draw') ? 1 - clamp(tau / 0.6) : 0);
        for (let c = 0; c < NSTEP; c++) {
            const q = on('copy') ? clamp((tau - arcTime(c)) / 0.6) : on('many') || on('draw') ? 1 : 0;
            set(arcs[c], { 'stroke-dasharray': `${arcLen[c]} ${arcLen[c]}`, 'stroke-dashoffset': arcLen[c] * (1 - q) });
        }
        const passes = on('many') ? clamp(Math.floor((tau - 1.5) / 0.11) + 1, 0, NSTEP) : NSTEP;
        counter.textContent = `backward passes: ${passes}`;
        op(counter, on('many') ? P(t, 'many', 1.2, 1.6) : on('draw') ? 1 - clamp(tau / 0.6) : 0);
        const oq = on('rerun') ? P(t, 'rerun', 1.6, 2.6) : past('rerun') ? 1 : 0;
        op(oneArc, oq > 0.01 && !on('converge') ? 1 : on('converge') ? 1 - clamp(tau / 0.8) : 0);
        set(oneArc, { 'stroke-dasharray': `${oneLen} ${oneLen}`, 'stroke-dashoffset': oneLen * (1 - oq) });

        // ---- the draw p_n
        const barsUp = ease(P(t, 'draw', 0.8, 2.3));
        op(gBars, barsUp > 0 ? 1 : 0);
        const glow = on('cancel') ? 0.35 * Math.sin(Math.PI * P(t, 'cancel', 0.5, 3)) : 0;
        let markN = null;
        if (on('draw')) {
            const hops = [30, 44, 12, 47, 38, 49, 45, DRAWN], q = P(t, 'draw', 3, 6.5) * (hops.length - 1);
            const i = Math.floor(q), f = ease(clamp((q - i) / 0.6));
            markN = tau < 3 ? null : lerp(hops[i], hops[Math.min(i + 1, hops.length - 1)], f);
        } else if (past('draw') && !on('converge')) markN = DRAWN;
        else if (on('converge')) {
            let j = -1; for (let i = 0; i < NUPD; i++) if (tau >= updTime(i)) j = i;
            markN = j < 0 ? DRAWN : DRAWS[j];
        }
        for (let c = 0; c < NSTEP; c++) {
            const h = BAR_H * PN[c] / PMAX * barsUp;
            const picked = markN !== null && Math.abs(markN - c) < 0.5 && (past('draw') || P(t, 'draw', 6.5, 6.6) >= 1);
            set(bars[c], { y: BASE - h, height: h, fill: picked ? COL.grad : '#a78bfa', 'fill-opacity': picked ? 0.9 : 0.55 + glow });
        }
        op(marker, markN === null ? 0 : 1);
        if (markN !== null) set(marker, { transform: `translate(${cellX(markN)},${BASE + 3})` });
        place(L.pn, cellX(NSTEP - 1) + 16, BASE - BAR_H + 6, barsUp, tex('p_n'), 'l');
        const landed = on('draw') ? P(t, 'draw', 6.6, 7.1) : past('draw') && !on('converge') ? 1 : on('converge') ? 1 - clamp(tau / 0.6) : 0;
        place(L.drawn, cellX(DRAWN) - 14, BASE + 16, landed, tex(String.raw`n = ${DRAWN},\ p_n = ${PN[DRAWN].toFixed(3)}`), 'r');
        const wq = on('rerun') ? P(t, 'rerun', 4.2, 4.8) : on('cancel') ? 1 : on('converge') ? 1 - clamp(tau / 0.6) : 0;
        place(L.weight, cellX(DRAWN) - 14, BASE - 36, wq, tex(String.raw`\times\,1/p_n \approx ${(1 / PN[DRAWN]).toFixed(1)}`), 'r');

        // ---- many updates average out to G_copy
        const plotO = on('converge') ? P(t, 'converge', 0.8, 1.6) : 0;
        op(gPlot, plotO);
        let jNow = -1;
        if (on('converge')) for (let i = 0; i < NUPD; i++) if (tau >= updTime(i)) jNow = i;
        let mean = [0, 0];
        for (let i = 0; i < NUPD; i++) {
            const v = hatVec(DRAWS[i]), grown = on('converge') ? clamp((tau - updTime(i)) / 0.3) : 0;
            op(hats[i], grown > 0 ? (i === jNow ? 1 : 0.22) : 0);
            set(hats[i], { x2: ORIGIN[0] + v[0] * grown, y2: ORIGIN[1] + v[1] * grown, 'stroke-width': i === jNow ? 2.2 : 1.4 });
        }
        if (jNow >= 0) {
            const prevMean = jNow > 0 ? DRAWS.slice(0, jNow).map(hatVec).reduce((a, v) => [a[0] + v[0], a[1] + v[1]], [0, 0]).map(c => c / jNow) : [0, 0];
            const curMean = DRAWS.slice(0, jNow + 1).map(hatVec).reduce((a, v) => [a[0] + v[0], a[1] + v[1]], [0, 0]).map(c => c / (jNow + 1));
            const q = ease(clamp((tau - updTime(jNow)) / 0.4));
            mean = [lerp(prevMean[0], curMean[0], q), lerp(prevMean[1], curMean[1], q)];
        }
        op(meanArrow, jNow >= 0 ? 1 : 0);
        set(meanArrow, { x2: ORIGIN[0] + mean[0], y2: ORIGIN[1] + mean[1] });
        updText.textContent = `updates: ${jNow + 1}`;
        place(L.gcopy, ORIGIN[0] + GV[0] + 16, ORIGIN[1] + GV[1] - 12, plotO, tex(String.raw`G_{\text{copy}}`), 'l');
        const latest = jNow >= 0 ? hatVec(DRAWS[jNow]) : [0, 0];
        place(L.ghat, ORIGIN[0] + latest[0] + 10, ORIGIN[1] + latest[1] + 2, jNow >= 0 ? plotO : 0, tex(String.raw`\hat G`), 'l');
        place(L.mean, ORIGIN[0] + mean[0] * 0.55 + 12, ORIGIN[1] + mean[1] * 0.55 + 14, jNow >= 1 ? plotO : 0, 'average so far', 'l');

        // ---- the line of math
        rowOp(ROWS.euler, win(t, at('step', 1.5), at('worn', 0.5), 0.6));
        rowOp(ROWS.score, win(t, at('score', 0.5), at('chain', 0.5), 0.6));
        rowOp(ROWS.grad, win(t, at('chain', 0.6), Infinity, 0.6));
        const G = ROWS.grad.pieces;
        const copyC = P(t, 'copy', 2.5, 4), rerunC = P(t, 'rerun', 5, 7), canA = P(t, 'cancel', 0.5, 2), canB = P(t, 'cancel', 5, 6.5);
        piece(G.lhsX, 1 - copyC, 0, 0, t);
        piece(G.lhsC, copyC - rerunC, 0, 0, t);
        piece(G.lhsH, rerunC - canA, 0, 0, t);
        piece(G.lhsE, canA, 0, 0, t);
        piece(G.eq, 1, 0, 0, t);
        piece(G.sum, 1 - rerunC + canA, 0, on('many') ? win(t, at('many', 0.5), at('many', 8.5), 0.5) : on('draw') ? 0.6 * win(t, at('draw', 0.8), at('draw', 7), 0.5) : 0, t);
        piece(G.pn, canA - canB, P(t, 'cancel', 3, 4.2), 0, t);
        piece(G.inv, rerunC - canB, P(t, 'cancel', 3.3, 4.5), on('rerun') ? win(t, at('rerun', 7), Infinity, 0.5) : 0, t);
        piece(G.g, 1, 0, on('copy') ? win(t, at('copy', 4), at('copy', 7.5), 0.5) : 0, t);
        piece(G.chain, 1 - copyC, P(t, 'copy', 1, 2), on('chain') ? win(t, at('chain', 5), Infinity, 0.5) : on('acts') || on('oom') ? 1 : 0, t);
        piece(G.jac, 1, 0, on('many') ? win(t, at('many', 0.5), at('many', 8.5), 0.5) : 0, t);
        piece(G.tail, canB, 0, 0, t);
    }
    // When each cell's activations land in memory (the full backward pass), its copied gradient arrives, and update j.
    function blockTime(c) { return 0.8 + 6.4 * Math.sqrt(c / (NSTEP - 1)); }
    function arcTime(c) { return 4 + (NSTEP - 1 - c) / (NSTEP - 1) * 2.2; }
    function updTime(j) { return 2 + j * 0.6; }

    // ---------- the controls ----------
    const beats = [...root.querySelectorAll('.gx-beat')];
    const partBtns = [...root.querySelectorAll('.gx-part')];
    const parts = partBtns.map(b => IDX[b.dataset.from]);
    const partOf = li => { let p = 0; parts.forEach((first, i) => { if (li >= first) p = i; }); return p; };
    const track = root.querySelector('.gx-track');
    const knob = document.createElement('div'); knob.className = 'gx-knob';
    const segs = [];
    parts.forEach((first, pi) => {
        const last = pi + 1 < parts.length ? parts[pi + 1] : LINES.length;
        const chap = document.createElement('div'); chap.className = 'gx-chap';
        let dur = 0;
        for (let i = first; i < last; i++) {
            const s = document.createElement('div'); s.className = 'gx-seg'; s.style.flexGrow = LINES[i].dur;
            const fill = document.createElement('i'); s.appendChild(fill);
            chap.appendChild(s); segs[i] = { s, fill }; dur += LINES[i].dur;
        }
        chap.style.flexGrow = dur;
        partBtns[pi].style.flexGrow = dur;
        track.appendChild(chap);
    });
    track.appendChild(knob);
    const playBtn = root.querySelector('.gx-play');

    let t = 0, playing = !reduceMotion, stopAt = null, visible = false, last = null, dirty = true, dragging = false;
    const param = new URLSearchParams(location.search).get('gx');
    if (param !== null) { t = clamp(+param, 0, TOTAL - 1e-3); playing = false; }
    else if (reduceMotion) t = LINES[0].t1 - 1e-3;
    const endOf = i => LINES[i].t1 - 1e-3;
    const go = (i, step) => { // to line i: a step plays it and holds (or shows its last frame, with reduced motion)
        i = clamp(i, 0, LINES.length - 1);
        if (reduceMotion) { t = endOf(i); playing = false; stopAt = null; }
        else { t = LINES[i].t0; playing = true; stopAt = step ? endOf(i) : null; }
        dirty = true;
    };
    root.querySelector('.gx-next').addEventListener('click', () => go(lineAt(t) + 1, true));
    root.querySelector('.gx-prev').addEventListener('click', () => { const i = lineAt(t); go(t - LINES[i].t0 > 1.5 && !reduceMotion ? i : i - 1, true); });
    partBtns.forEach((b, pi) => b.addEventListener('click', () => go(parts[pi], false)));
    root.querySelector('.gx-replay').addEventListener('click', () => go(parts[partOf(lineAt(t))], false));
    playBtn.addEventListener('click', () => {
        if (playing) { playing = false; } else { if (t >= TOTAL - 0.01) t = 0; playing = true; stopAt = null; }
        dirty = true;
    });
    const seekTo = clientX => {
        let best = null;
        segs.forEach(({ s }, i) => {
            const r = s.getBoundingClientRect(), f = clamp((clientX - r.left) / r.width), d = clientX < r.left ? r.left - clientX : clientX > r.right ? clientX - r.right : 0;
            if (!best || d < best.d) best = { d, t: LINES[i].t0 + f * LINES[i].dur };
        });
        t = clamp(best.t, 0, TOTAL - 1e-3); playing = false; stopAt = null; dirty = true;
    };
    track.addEventListener('pointerdown', e => { dragging = true; track.setPointerCapture(e.pointerId); track.classList.add('dragging'); seekTo(e.clientX); });
    track.addEventListener('pointermove', e => { if (dragging) seekTo(e.clientX); });
    const endDrag = () => { dragging = false; track.classList.remove('dragging'); };
    track.addEventListener('pointerup', endDrag);
    track.addEventListener('pointercancel', endDrag);
    track.addEventListener('keydown', e => {
        if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { t = clamp(t + (e.key === 'ArrowLeft' ? -2 : 2), 0, TOTAL - 1e-3); playing = false; dirty = true; e.preventDefault(); e.stopPropagation(); }
    });
    document.addEventListener('keydown', e => {
        if (!visible || e.metaKey || e.ctrlKey || e.altKey || /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '')) return;
        if (e.key === 'ArrowRight') { go(lineAt(t) + 1, true); e.preventDefault(); }
        else if (e.key === 'ArrowLeft') { const i = lineAt(t); go(t - LINES[i].t0 > 1.5 && !reduceMotion ? i : i - 1, true); e.preventDefault(); }
        else if (e.key === ' ' && root.contains(document.activeElement)) { playBtn.click(); e.preventDefault(); }
    });
    new IntersectionObserver(es => { for (const e of es) { visible = e.intersectionRatio >= 0.35; last = null; } }, { threshold: [0, 0.35, 0.6] }).observe(stage);

    let shownLine = -1, shownPlaying = null;
    function ui() {
        const li = lineAt(Math.min(t, TOTAL - 1e-3));
        if (li !== shownLine) {
            LINES.forEach((l, i) => l.el.classList.toggle('on', i === li));
            beats.forEach(b => b.classList.toggle('on', b === LINES[li].beat));
            const pi = partOf(li);
            partBtns.forEach((b, i) => b.classList.toggle('on', i === pi));
            shownLine = li;
        }
        const tt = Math.min(t, TOTAL);
        LINES.forEach((l, i) => { const f = clamp((tt - l.t0) / l.dur); if (segs[i]._f !== f) { segs[i].fill.style.width = `${f * 100}%`; segs[i]._f = f; } });
        const s = segs[li].s, f = clamp((tt - LINES[li].t0) / LINES[li].dur);
        knob.style.left = `${s.offsetLeft + s.parentNode.offsetLeft + f * s.offsetWidth}px`;
        track.setAttribute('aria-valuenow', tt.toFixed(1));
        if (shownPlaying !== playing) {
            playBtn.classList.toggle('paused', !playing);
            playBtn.setAttribute('aria-label', playing ? 'Pause' : 'Play');
            shownPlaying = playing;
        }
    }
    track.setAttribute('aria-valuemin', 0); track.setAttribute('aria-valuemax', TOTAL.toFixed(1));
    function frameLoop(now) {
        if (visible && playing && last !== null) {
            t += Math.min(0.1, (now - last) / 1000);
            if (stopAt !== null && t >= stopAt) { t = stopAt; playing = false; stopAt = null; }
            if (t >= TOTAL + LOOP_HOLD) t = 0;
            dirty = true;
        }
        last = now;
        if (dirty && (visible || param !== null)) { render(Math.min(t, TOTAL - 1e-3)); ui(); dirty = false; }
        requestAnimationFrame(frameLoop);
    }
    window.gxTimes = LINES.map(l => [l.id, +l.t0.toFixed(3), +l.t1.toFixed(3)]);
    window.gxSeek = s => { t = clamp(s, 0, TOTAL - 1e-3); playing = false; render(t); ui(); return { line: LINES[lineAt(t)].id, total: TOTAL }; };
    render(Math.min(t, TOTAL - 1e-3)); ui();
    requestAnimationFrame(frameLoop);
})();
