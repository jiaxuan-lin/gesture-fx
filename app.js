import { FilesetResolver, GestureRecognizer, FaceLandmarker } from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';

const WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const GESTURE_MODEL = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task';
const FACE_MODEL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task';

const HAND_LINKS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];
const TIPS = [4, 8, 12, 16, 20];
const HAND_HUE = { Left: 190, Right: 300 };
// Shader mode for each filter name; the lists below only set the order they cycle in.
const FILTER_MODE = { 'Cyberpunk': 0, 'Thermal': 1, 'Night Vision': 2, 'X-Ray': 3, 'Pixel': 4, 'Comic': 5, 'Glitch': 6, 'Rio de Janeiro': 7, 'Rainbow': 8 };
const BOX_FILTERS = ['Rainbow', 'Pixel', 'Rio de Janeiro', 'Glitch', 'Cyberpunk', 'Comic', 'Night Vision', 'X-Ray', 'Thermal'];
const FOG_SCALE = 0.25;
const BLOW_ROUGH = 4;
const FOG_HOLD = 3000; // ms the fog lingers after the last breath or stroke before it clears itself

const $ = (s) => document.querySelector(s);
const videoEl = $('#video');
const stage = $('#stage');
const vcan = $('#videoCanvas');
const vctx = vcan.getContext('2d');
// Mirrored video at screen layout; the single source for display and both filters.
const scene = document.createElement('canvas'); const sctx = scene.getContext('2d');
const fx = $('#fxCanvas');
const ctx = fx.getContext('2d');

const opts = { handSkeleton: true };
let W = 0, H = 0, DPR = 1;
const view = { ox: 0, oy: 0, dw: 1, dh: 1 };
let boxIdx = 0;
const CAM_FILTERS = ['Natural', 'Cyberpunk', 'Glitch', 'Rainbow', 'Thermal', 'Rio de Janeiro', 'X-Ray', 'Pixel', 'Night Vision', 'Comic'];
let camIdx = 0, lastCamSwitch = -1e9;
let gesture = null, faceLm = null;
let running = false, paused = false;
let helpTouched = false;

const hands = {};
const face = { present: false, lastSeen: -1e9, mouthOpen: false, blowShape: false, pucker: 0, jaw: 0, mouth: null, top: null, angle: 0, w: 0 };
const frame = { quad: null, alpha: 0, live: false, active: false, collapsedAt: -1e9, rearm: false };
const fog = { breath: 0, any: false, evapAcc: 0, checkAcc: 0, drips: [], lastActive: -1e9 };
const snd = { on: false, ctx: null, an: null, buf: null, bg: 0.003, lastT: -1e9, peak: 0, onsets: 0 };
const blow = { an: null, db: null, floor: null, on: false, run: 0, lastT: -1e9, stats: null };
const wipe = { start: -1e9, fade: 1, lastT: -1e9 };
const WIPE_MS = 700;
const particles = [];
const MAX_PARTICLES = 1500;
const debug = { errors: [], detections: 0, events: [] };
const freeze = { hands: false, face: false };

const rand = (a, b) => a + Math.random() * (b - a);
const lerp = (a, b, t) => a + (b - a) * t;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const d3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
const approach = (v, target, step) => (v < target ? Math.min(target, v + step) : Math.max(target, v - step));
const easeOutBack = (t) => { const c1 = 1.70158, c3 = c1 + 1; return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2); };
const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
const logEvent = (name) => { debug.events.push(name); if (debug.events.length > 60) debug.events.shift(); };

// Debounces a boolean: needs `on` consecutive true samples to switch on and `off` false samples to switch off.
function stab(o, key, raw, on = 3, off = 4) {
  const c = o._c || (o._c = {});
  if (raw) { c[key] = Math.max(0, c[key] || 0) + 1; if (c[key] >= on) o[key] = true; }
  else { c[key] = Math.min(0, c[key] || 0) - 1; if (-c[key] >= off) o[key] = false; }
  return !!o[key];
}

// ---------- layout ----------
const fogMask = document.createElement('canvas'); const fmc = fogMask.getContext('2d');
const fogVid = document.createElement('canvas'); const fvc = fogVid.getContext('2d');
const fogComp = document.createElement('canvas'); const fcc = fogComp.getContext('2d');
const droplets = document.createElement('canvas');

function updateView() {
  const vw = videoEl.videoWidth || 1280, vh = videoEl.videoHeight || 720;
  const s = Math.max(W / vw, H / vh);
  view.dw = vw * s; view.dh = vh * s;
  view.ox = (W - view.dw) / 2; view.oy = (H - view.dh) / 2;
}

function resizeFog() {
  const fw = Math.max(32, Math.round(W * FOG_SCALE)), fh = Math.max(18, Math.round(H * FOG_SCALE));
  const old = document.createElement('canvas'); old.width = fogMask.width; old.height = fogMask.height;
  if (old.width && old.height) old.getContext('2d').drawImage(fogMask, 0, 0);
  fogMask.width = fogComp.width = droplets.width = fw;
  fogMask.height = fogComp.height = droplets.height = fh;
  if (old.width && old.height) fmc.drawImage(old, 0, 0, fw, fh);
  fogVid.width = Math.max(16, Math.round(fw / 4)); fogVid.height = Math.max(9, Math.round(fh / 4));
  const dc = droplets.getContext('2d');
  dc.clearRect(0, 0, fw, fh);
  for (let i = 0; i < fw * fh / 90; i++) {
    const x = rand(0, fw), y = rand(0, fh), r = rand(0.25, 0.7);
    dc.fillStyle = `rgba(255,255,255,${rand(0.08, 0.25)})`;
    dc.beginPath(); dc.arc(x, y, r, 0, Math.PI * 2); dc.fill();
  }
}

// Tileable fractal noise used for drifting fog wisps.
function makeClouds(N = 128) {
  const c = document.createElement('canvas'); c.width = c.height = N;
  const x = c.getContext('2d'); const img = x.createImageData(N, N);
  const sm = (t) => t * t * (3 - 2 * t);
  const octave = (period) => {
    const g = Array.from({ length: period * period }, () => Math.random());
    const at = (i, j) => g[((j % period) + period) % period * period + ((i % period) + period) % period];
    return (px, py) => {
      const fx = px / N * period, fy = py / N * period, x0 = Math.floor(fx), y0 = Math.floor(fy);
      const tx = sm(fx - x0), ty = sm(fy - y0);
      return lerp(lerp(at(x0, y0), at(x0 + 1, y0), tx), lerp(at(x0, y0 + 1), at(x0 + 1, y0 + 1), tx), ty);
    };
  };
  const os = [[4, 0.5], [8, 0.25], [16, 0.15], [32, 0.1]].map(([p, w]) => [octave(p), w]);
  for (let py = 0; py < N; py++) for (let px = 0; px < N; px++) {
    let v = 0; for (const [f, w] of os) v += f(px, py) * w;
    const o = (py * N + px) * 4;
    img.data[o] = img.data[o + 1] = img.data[o + 2] = 255;
    img.data[o + 3] = clamp((v - 0.35) * 2.2, 0, 1) * 255;
  }
  x.putImageData(img, 0, 0);
  return c;
}
const clouds = makeClouds();
let cloudPattern = null;
// A repeating pattern samples across tile edges; drawing tiles one by one leaves visible seams.
function tile(c, img, ox, oy, size, w, h) {
  if (!cloudPattern) cloudPattern = c.createPattern(img, 'repeat');
  cloudPattern.setTransform(new DOMMatrix().translate(ox, oy).scale(size / img.width));
  c.fillStyle = cloudPattern; c.fillRect(0, 0, w, h);
}
const breathSprite = (() => {
  const s = document.createElement('canvas'); s.width = s.height = 64;
  const g = s.getContext('2d'), grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, 'rgba(255,255,255,1)'); grd.addColorStop(0.4, 'rgba(250,252,255,0.6)'); grd.addColorStop(1, 'rgba(250,252,255,0)');
  g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
  return s;
})();

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = innerWidth; H = innerHeight;
  for (const c of [vcan, fx]) { c.width = W * DPR; c.height = H * DPR; c.style.width = W + 'px'; c.style.height = H + 'px'; }
  for (const c of [vctx, ctx]) c.setTransform(DPR, 0, 0, DPR, 0, 0);
  scene.width = Math.round(W); scene.height = Math.round(H);
  updateView();
  resizeFog();
}

// The video is drawn mirrored (selfie view), so landmark x is flipped.
const toScreen = (lm) => ({ x: view.ox + (1 - lm.x) * view.dw, y: view.oy + lm.y * view.dh });

// ---------- sprites & particles ----------
const spriteCache = new Map();
function glowSprite(hue, light = 60) {
  // Quantize both inputs: every distinct key allocates a canvas, and unbounded keys exhaust GPU memory.
  const h = ((Math.round(hue / 10) * 10) % 360 + 360) % 360;
  light = Math.round(light / 5) * 5;
  const key = h + '_' + light;
  let s = spriteCache.get(key);
  if (!s) {
    s = document.createElement('canvas'); s.width = s.height = 64;
    const g = s.getContext('2d');
    const grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grd.addColorStop(0, `hsla(${h},100%,${Math.min(100, light + 35)}%,1)`);
    grd.addColorStop(0.25, `hsla(${h},100%,${light}%,0.8)`);
    grd.addColorStop(1, `hsla(${h},100%,${light}%,0)`);
    g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
    spriteCache.set(key, s);
  }
  return s;
}
function drawGlow(x, y, r, hue, alpha = 1, light = 60) {
  ctx.globalAlpha = alpha;
  ctx.drawImage(glowSprite(hue, light), x - r, y - r, r * 2, r * 2);
  ctx.globalAlpha = 1;
}
function spawn(o) {
  if (particles.length >= MAX_PARTICLES) return;
  particles.push(Object.assign({ x: 0, y: 0, vx: 0, vy: 0, life: 1, decay: 1, size: 6, hue: 0, type: 'glow', g: 0, drag: 0.98 }, o));
}
function emit(rate, dt, fn) {
  let n = rate * dt;
  while (n >= 1) { fn(); n--; }
  if (Math.random() < n) fn();
}
function updateParticles(dt) {
  let j = 0;
  for (let i = 0; i < particles.length; i++) {
    const p = particles[i];
    const d = Math.pow(p.drag, dt * 60);
    p.vx *= d; p.vy *= d; p.vy += p.g * dt;
    p.x += p.vx * dt; p.y += p.vy * dt;
    p.life -= p.decay * dt;
    if (p.life > 0 && p.y > -300 && p.y < H + 300) particles[j++] = p;
  }
  particles.length = j;
}

function drawParticles() {
  ctx.globalCompositeOperation = 'lighter';
  for (const p of particles) {
    const l = clamp(p.life, 0, 1);
    if (p.type === 'glow') drawGlow(p.x, p.y, p.size * (0.4 + l), p.hue, l);
    else if (p.type === 'ring') {
      const r = p.rMax * (1 - Math.pow(l, 3));
      ctx.strokeStyle = `hsla(${p.hue},100%,65%,${l})`;
      ctx.lineWidth = Math.max(1, p.size * l);
      ctx.beginPath(); ctx.arc(p.x, p.y, Math.max(1, r), 0, Math.PI * 2); ctx.stroke();
    }
  }
  ctx.globalCompositeOperation = 'source-over';
  for (const p of particles) {
    const l = clamp(p.life, 0, 1);
    if (p.type === 'smoke') {
      ctx.globalAlpha = l * 0.16;
      const r = p.size * (3 - 2 * l);
      ctx.drawImage(breathSprite, p.x - r, p.y - r, r * 2, r * 2);
      ctx.globalAlpha = 1;
    }
  }
}

// ---------- WebGL box filters ----------
const FRAG = `
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
varying vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uRes;
uniform float uTime;
uniform int uMode;
vec3 vid(vec2 p) { return texture2D(uTex, clamp(p / uRes, 0.001, 0.999)).rgb; }
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }
float edge(vec2 p, float k) {
  float tl = lum(vid(p + vec2(-k, -k))), t = lum(vid(p + vec2(0.0, -k))), tr = lum(vid(p + vec2(k, -k)));
  float l = lum(vid(p + vec2(-k, 0.0))), r = lum(vid(p + vec2(k, 0.0)));
  float bl = lum(vid(p + vec2(-k, k))), b = lum(vid(p + vec2(0.0, k))), br = lum(vid(p + vec2(k, k)));
  float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
  float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
  return length(vec2(gx, gy));
}
vec3 heat(float t) {
  vec3 c = mix(vec3(0.0, 0.0, 0.2), vec3(0.4, 0.0, 0.8), smoothstep(0.0, 0.25, t));
  c = mix(c, vec3(1.0, 0.0, 0.3), smoothstep(0.25, 0.5, t));
  c = mix(c, vec3(1.0, 0.6, 0.0), smoothstep(0.5, 0.75, t));
  return mix(c, vec3(1.0, 1.0, 0.8), smoothstep(0.75, 1.0, t));
}
void main() {
  vec2 p = vec2(vUv.x, 1.0 - vUv.y) * uRes;
  vec3 c;
  if (uMode == 0) {
    vec3 base = vec3(vid(p + vec2(4.0, 0.0)).r, vid(p).g, vid(p - vec2(4.0, 0.0)).b);
    float l = lum(base);
    vec3 duo = mix(vec3(0.08, 0.0, 0.2), vec3(0.0, 0.85, 1.0), smoothstep(0.15, 0.8, l));
    duo = mix(duo, vec3(1.0, 0.25, 0.85), smoothstep(0.6, 1.0, l) * 0.7);
    float e = smoothstep(0.15, 0.6, edge(p, 1.5));
    c = duo * 0.85 + vec3(1.0, 0.15, 0.9) * e;
    c *= 0.82 + 0.18 * sin(p.y * 1.4 + uTime * 8.0);
  } else if (uMode == 1) {
    c = heat(lum(vid(p)));
  } else if (uMode == 2) {
    float l = lum(vid(p)) * 1.8 + (hash(p + fract(uTime)) - 0.5) * 0.25;
    c = vec3(0.1, 1.0, 0.25) * l;
    c *= 0.8 + 0.2 * sin(p.y * 2.0);
  } else if (uMode == 3) {
    float l = 1.0 - lum(vid(p));
    c = vec3(0.55, 0.8, 1.0) * pow(l, 1.5) + vec3(0.8, 0.95, 1.0) * edge(p, 1.5) * 0.8;
  } else if (uMode == 4) {
    float s = 14.0;
    vec2 q = (floor(p / s) + 0.5) * s;
    c = floor(vid(q) * 6.0 + 0.5) / 6.0;
    vec2 f = fract(p / s);
    c *= 0.85 + 0.15 * step(0.08, min(f.x, f.y));
  } else if (uMode == 5) {
    vec3 col = floor(vid(p) * 4.0 + 0.5) / 4.0;
    float l = lum(col);
    vec2 q = mat2(0.707, -0.707, 0.707, 0.707) * p;
    vec2 cell = fract(q / 7.0) - 0.5;
    float dotv = step(length(cell), (1.0 - l) * 0.6);
    c = mix(col * 1.15, col * 0.25, dotv * 0.6);
    c = mix(c, vec3(0.0), smoothstep(0.35, 0.7, edge(p, 1.5)));
  } else if (uMode == 6) {
    float row = floor(p.y / 18.0);
    float n = hash(vec2(row, floor(uTime * 8.0)));
    float shift = n > 0.8 ? (n - 0.8) * 300.0 * sign(n - 0.9) : 0.0;
    vec2 q = p + vec2(shift, 0.0);
    c = vec3(vid(q + vec2(7.0, 0.0)).r, vid(q).g, vid(q - vec2(7.0, 0.0)).b);
    c += (hash(p * 0.5 + uTime) - 0.5) * 0.12;
    if (hash(vec2(floor(uTime * 12.0), row)) > 0.95) c = c.gbr;
  } else if (uMode == 7) {
    // Rio de Janeiro: vivid and bright, rosy-magenta cast, lavender-lifted shadows, golden highlights.
    vec3 v = vid(p);
    float l = lum(v);
    v = clamp(mix(vec3(l), v, 1.5), 0.0, 1.0);
    v = pow(v, vec3(0.82));
    v *= vec3(1.12, 0.93, 1.06);
    v += vec3(0.16, 0.04, 0.2) * (1.0 - l) * 0.7;
    v += vec3(0.1, 0.06, -0.03) * smoothstep(0.45, 1.0, l);
    vec2 q = p / uRes - 0.5;
    v *= 1.0 - dot(q, q) * 0.35;
    c = clamp(v, 0.0, 1.0);
  } else {
    // Rainbow: diagonal bands of hue drift across the picture, with brightness and edges from the video.
    vec3 v = vid(p);
    float l = lum(v);
    float h = fract((p.x + p.y * 0.6) / uRes.x * 0.9 + l * 0.35 - uTime * 0.12);
    vec3 rgb = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    c = rgb * (0.25 + 0.95 * l);
    c = mix(c, vec3(1.0), smoothstep(0.85, 1.0, l) * 0.5);
    c += rgb * smoothstep(0.2, 0.7, edge(p, 1.5)) * 0.5;
    c = clamp(c, 0.0, 1.0);
  }
  gl_FragColor = vec4(c, 1.0);
}`;
const glc = document.createElement('canvas');
let gl = null, glU = null, glTex = null;
function initGL() {
  gl = glc.getContext('webgl', { premultipliedAlpha: false, preserveDrawingBuffer: true, alpha: false });
  if (!gl) return false;
  const sh = (type, src) => {
    const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error('Shader: ' + gl.getShaderInfoLog(s));
    return s;
  };
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, 'attribute vec2 a; varying vec2 vUv; void main(){ vUv = a * 0.5 + 0.5; gl_Position = vec4(a, 0.0, 1.0); }'));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('Link: ' + gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'a');
  gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  glTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, glTex);
  for (const [k, v] of [[gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE], [gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE], [gl.TEXTURE_MIN_FILTER, gl.LINEAR], [gl.TEXTURE_MAG_FILTER, gl.LINEAR]]) gl.texParameteri(gl.TEXTURE_2D, k, v);
  glU = Object.fromEntries(['uTex', 'uRes', 'uTime', 'uMode'].map((n) => [n, gl.getUniformLocation(prog, n)]));
  gl.uniform1i(glU.uTex, 0);
  return true;
}
function renderGL(now, mode, upload) {
  const gw = Math.round(W), gh = Math.round(H);
  if (glc.width !== gw || glc.height !== gh) { glc.width = gw; glc.height = gh; }
  gl.viewport(0, 0, gw, gh);
  gl.bindTexture(gl.TEXTURE_2D, glTex);
  if (upload) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, scene);
  gl.uniform2f(glU.uRes, W, H);
  gl.uniform1f(glU.uTime, (now / 1000) % 1000);
  gl.uniform1i(glU.uMode, mode);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
}

// ---------- hands ----------
function newHand(label) {
  return { label, present: false, pts: null, Q: null, palm: { x: 0, y: 0 }, size: 80, lastSeen: 0,
    rawGesture: 'None', point: false, victory: false, pinch: false, t48: 1, euro: null, euroT: 0,
    frameL: false, lastWrite: null, loadN: 0, afterLoad: 99, loadT: -1e9, loadReach: 0, snapMaybe: 0, _c: {} };
}

// One-Euro filter (Casiez et al. 2012): strong smoothing while a point is nearly still (kills jitter),
// almost none while it moves fast (kills lag).
const EURO = { minCutoff: 1.4, beta: 0.03, dCutoff: 1.2 };
const euroAlpha = (cutoff, dt) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));
function euroPts(h, pts, now) {
  const dt = clamp((now - h.euroT) / 1000, 1 / 120, 0.2);
  h.euroT = now;
  if (!h.euro) { h.euro = pts.map((p) => ({ x: p.x, y: p.y, dx: 0, dy: 0 })); return pts.map((p) => ({ x: p.x, y: p.y })); }
  const ad = euroAlpha(EURO.dCutoff, dt);
  return pts.map((p, j) => {
    const f = h.euro[j];
    f.dx = lerp(f.dx, (p.x - f.x) / dt, ad); f.dy = lerp(f.dy, (p.y - f.y) / dt, ad);
    const a = euroAlpha(EURO.minCutoff + EURO.beta * Math.hypot(f.dx, f.dy), dt);
    f.x = lerp(f.x, p.x, a); f.y = lerp(f.y, p.y, a);
    return { x: f.x, y: f.y };
  });
}

const palmCenter = (pts) => [0, 5, 9, 13, 17].reduce((a, j) => ({ x: a.x + pts[j].x / 5, y: a.y + pts[j].y / 5 }), { x: 0, y: 0 });

// Labels follow where each hand is rather than MediaPipe's handedness guess, which flips when a hand turns
// over and would otherwise swap the box corners mid-gesture.
function assignLabels(dets, now) {
  const tracked = ['Left', 'Right'].filter((l) => hands[l]?.present && hands[l].pts && now - hands[l].lastSeen < 250);
  const cost = (d, l) => dist(d.c, hands[l].palm) / Math.max(hands[l].size, d.size);
  if (dets.length === 2 && tracked.length === 2) {
    const keep = cost(dets[0], 'Left') + cost(dets[1], 'Right'), swap = cost(dets[0], 'Right') + cost(dets[1], 'Left');
    [dets[0].label, dets[1].label] = keep <= swap ? ['Left', 'Right'] : ['Right', 'Left'];
    return;
  }
  const free = new Set(['Left', 'Right']), open = new Set(dets), pairs = [];
  for (const d of dets) for (const l of tracked) { const c = cost(d, l); if (c < 3) pairs.push([c, d, l]); }
  pairs.sort((a, b) => a[0] - b[0]);
  for (const [, d, l] of pairs) if (open.has(d) && free.has(l)) { d.label = l; open.delete(d); free.delete(l); }
  const rest = [...open];
  // Two new hands with the same handedness guess: the mirrored view puts the left hand on the left.
  if (rest.length === 2 && rest[0].label === rest[1].label) rest.sort((a, b) => a.c.x - b.c.x).forEach((d, k) => { d.label = k ? 'Right' : 'Left'; });
  for (const d of rest) { if (!free.has(d.label)) d.label = [...free][0]; free.delete(d.label); }
}

function setCamFilter(i) {
  camIdx = (i + CAM_FILTERS.length) % CAM_FILTERS.length;
  $('#camName').textContent = CAM_FILTERS[camIdx].toLowerCase();
  toast(`Camera filter: ${CAM_FILTERS[camIdx]}`);
  logEvent('cam:' + CAM_FILTERS[camIdx]);
}

function setBoxFilter(i, quiet) {
  boxIdx = (i + BOX_FILTERS.length) % BOX_FILTERS.length;
  $('#filterName').textContent = BOX_FILTERS[boxIdx].toLowerCase();
  if (!quiet) toast(`Box filter: ${BOX_FILTERS[boxIdx]}`);
  logEvent('filter:' + BOX_FILTERS[boxIdx]);
}

function classifyHand(h, now) {
  const Q = h.Q, P = h.pts;
  const s3 = Math.max(1, d3(Q[0], Q[9]));
  const curled = (m, t) => d3(Q[t], Q[0]) < d3(Q[m], Q[0]) * 1.35;
  const extended = (m, p, t) => {
    const ax = Q[p].x - Q[m].x, ay = Q[p].y - Q[m].y, az = Q[p].z - Q[m].z;
    const bx = Q[t].x - Q[p].x, by = Q[t].y - Q[p].y, bz = Q[t].z - Q[p].z;
    const cos = (ax * bx + ay * by + az * bz) / ((Math.hypot(ax, ay, az) * Math.hypot(bx, by, bz)) || 1);
    return cos > 0.7 && d3(Q[m], Q[t]) > 0.55 * s3;
  };
  const othersCurled = curled(9, 12) && curled(13, 16) && curled(17, 20);
  const indexExt = extended(5, 6, 8);
  const t48 = d3(Q[4], Q[8]) / s3, t412 = d3(Q[4], Q[12]) / s3, reach = d3(Q[12], Q[0]) / s3;
  h.dbg = { t48: +t48.toFixed(2), t412: +t412.toFixed(2), reach: +reach.toFixed(2), othersCurled, indexExt };

  h.t48 = t48;
  h.pinch = h.pinch ? t48 < 0.32 : t48 < 0.2;
  detectSnap(h, now, t412, reach);

  const victoryRaw = h.rawGesture === 'Victory' || (indexExt && extended(9, 10, 12) && curled(13, 16) && curled(17, 20) && !curled(9, 12) && t48 > 0.35);
  stab(h, 'victory', victoryRaw, 3, 5);
  stab(h, 'point', othersCurled && indexExt && !h.pinch && !h.victory, 3, 4);
  h.frameL = indexExt && t48 > 0.5 && d3(Q[4], Q[5]) / s3 > 0.45 && !h.victory;
  const fingersOut = [indexExt, extended(9, 10, 12), extended(13, 14, 16), extended(17, 18, 20)].filter(Boolean).length;
  h.open = h.rawGesture === 'Open_Palm' || fingersOut >= 3;
}

// Finger snap. Loaded: thumb pad pressed on the middle fingertip. Released: thumb and middle fly apart and the
// middle finger slaps down toward the palm. The release takes a few milliseconds, so on a 30 fps camera it
// shows up as one jump between frames, often blurred. A clean jump fires on its own; a blurry or partial
// one (or the hand vanishing mid-snap) only fires if the microphone heard a snap at the same moment.
function detectSnap(h, now, t412, reach) {
  const wasLoaded = h.loadN >= 2;
  if (t412 < 0.3 && reach > 0.85) {
    if (++h.loadN >= 2) { h.loadT = now; h.loadReach = reach; }
    h.afterLoad = 0;
    return;
  }
  h.loadN = 0;
  // Frames since the loaded pose: a real snap completes within one frame, a deliberate curl takes several.
  const n = wasLoaded ? (h.afterLoad = 1) : ++h.afterLoad;
  if (now - h.loadT > 350 || frame.live || now - frame.collapsedAt < 500) return;
  const slapped = reach < h.loadReach - 0.3 || reach < 1.15;
  const notOpening = reach < h.loadReach + 0.1;
  if (t412 > 0.5 && slapped && n === 1 && now - h.loadT <= 150) fireSnap(h, now, 'motion');
  else if (!h.snapMaybe && notOpening && t412 > 0.42 && n <= 3) h.snapMaybe = now;
}

function fireSnap(h, now, how) {
  h.loadT = -1e9; h.snapMaybe = 0;
  if (now - lastCamSwitch < 600) return;
  lastCamSwitch = now;
  // Left hand steps back through the camera filters, right hand steps forward.
  const step = h.label === 'Left' ? -1 : 1;
  logEvent(`snap:${how}:${h.label}`);
  setCamFilter(camIdx + step);
}

// Snap sound: a short, sharp burst of high-frequency energy well above the room's background level.
async function startMic() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const ac = new AudioContext();
    const hp = ac.createBiquadFilter(); hp.type = 'highpass'; hp.frequency.value = 1800;
    const an = ac.createAnalyser(); an.fftSize = 1024;
    const src = ac.createMediaStreamSource(stream);
    src.connect(hp); hp.connect(an);
    const ba = blowAnalyser(ac), ta = waveAnalyser(ac);
    src.connect(ba); src.connect(ta);
    Object.assign(blow, { an: ba, tan: ta, db: new Float32Array(ba.frequencyBinCount), wave: new Float32Array(ta.fftSize), floor: null });
    Object.assign(snd, { on: true, ctx: ac, an, buf: new Float32Array(an.fftSize) });
    // Autoplay rules can leave the context suspended until the page gets a click.
    if (ac.state !== 'running') window.addEventListener('pointerdown', () => ac.resume(), { once: true });
  } catch (e) {
    debug.errors.push('mic: ' + e);
    toast('No microphone: snaps use finger tracking only');
  }
}

function pollSound(now) {
  if (!snd.on || snd.ctx.state !== 'running') return;
  snd.an.getFloatTimeDomainData(snd.buf);
  let peak = 0, sum = 0;
  for (const v of snd.buf) { const a = Math.abs(v); if (a > peak) peak = a; sum += v * v; }
  const rms = Math.sqrt(sum / snd.buf.length);
  snd.peak = peak;
  if (peak > 0.06 && peak > snd.bg * 10 && peak > rms * 3.5 && now - snd.lastT > 150) { snd.lastT = now; snd.onsets++; logEvent('sound'); }
  // Background follows quiet levels quickly and loud ones slowly, so one snap barely raises it.
  snd.bg = lerp(snd.bg, Math.max(rms, 0.0005), rms > snd.bg ? 0.01 : 0.1);
}

// Blowing: breath noise that stays well above the room's level for a moment. It is noisy rather than tonal
// (voices and whistles have sharp harmonic peaks), covers most of 100 Hz-4 kHz, and keeps a good share of its
// energy below 1.5 kHz (an "sss" hiss does not). Clicks and snaps are too short to count.
// About 23 Hz per bin at any sample rate, so the noisy-versus-tonal measure behaves the same on every mic.
function blowAnalyser(ac) {
  const an = ac.createAnalyser();
  an.fftSize = 2 ** Math.round(Math.log2(ac.sampleRate / 23.4)); an.smoothingTimeConstant = 0.6;
  return an;
}
// Raw waveform for the pitch check.
function waveAnalyser(ac) {
  const an = ac.createAnalyser();
  an.fftSize = 2 ** Math.ceil(Math.log2(ac.sampleRate * 0.036));
  return an;
}
const micLive = () => snd.on && snd.ctx.state === 'running' && !!blow.an;
// Strength of a repeating pitch (70-450 Hz) in the last 36 ms of sound (short enough that a gliding voice still
// lines up with itself): high for voices, low for breath noise. Only
// a real peak of the autocorrelation counts; the low rumble of air hitting the mic is merely slow, so its
// correlation just slides downhill from zero lag.
function voicing(wave, sr) {
  const w = wave.subarray(Math.max(0, wave.length - Math.round(sr * 0.036)));
  const k = Math.max(1, Math.round(sr / 12000)), fs = sr / k, n = Math.floor(w.length / k);
  const d = blow.diff && blow.diff.length === n - 1 ? blow.diff : (blow.diff = new Float32Array(n - 1));
  let prev = 0;
  for (let i = 0; i < n; i++) {
    let s = 0; for (let j = 0; j < k; j++) s += w[i * k + j];
    s /= k; if (i) d[i - 1] = s - 0.95 * prev; prev = s;
  }
  const lagMin = Math.round(fs / 450), lagMax = Math.round(fs / 70), N = d.length - lagMax - 2;
  if (N < lagMax) return 0;
  let e0 = 0; for (let i = 0; i < N; i++) e0 += d[i] * d[i];
  let el = 0; for (let i = lagMin; i < lagMin + N; i++) el += d[i] * d[i];
  let best = 0, r1 = -1, r2 = -1;
  for (let lag = lagMin; lag <= lagMax + 1; lag++) {
    let c = 0; for (let i = 0; i < N; i++) c += d[i] * d[i + lag];
    const r = c / Math.sqrt(e0 * el + 1e-20);
    if (lag > lagMin + 1 && r1 > r2 && r1 >= r && r1 > best) best = r1;
    r2 = r1; r1 = r;
    el += d[lag + N] * d[lag + N] - d[lag] * d[lag];
  }
  return best;
}
function pollBlow(now) {
  if (!micLive()) { blow.on = false; return; }
  const dt = clamp((now - (blow.polledAt ?? now)) / 1000, 0, 0.25); blow.polledAt = now;
  const db = blow.db; blow.an.getFloatFrequencyData(db);
  blow.tan.getFloatTimeDomainData(blow.wave);
  let pk = 0, ss = 0;
  for (const s of blow.wave) { const a = Math.abs(s); if (a > pk) pk = a; ss += s * s; }
  const crest = pk / Math.max(1e-9, Math.sqrt(ss / blow.wave.length));
  const voiced = voicing(blow.wave, snd.ctx.sampleRate) > 0.6;
  // Speech flips between voiced and breathy sounds; blowing never has a voice in it.
  if (voiced) blow.voicedAt = now;
  const hz = snd.ctx.sampleRate / blow.an.fftSize;
  const b0 = Math.ceil(100 / hz), b1 = Math.floor(4000 / hz), bLow = Math.floor(1500 / hz), bRough = Math.ceil(300 / hz), half = 5;
  const v = (i) => Math.max(db[i], -140);
  if (!blow.floor) { blow.floor = Float32Array.from(db, (x) => Math.max(x, -140)); blow.since = now; blow.level = undefined; }
  // Learn the room for a moment before listening (the first frames can be pure digital silence).
  const warm = now - blow.since < 1500;
  const fl = blow.floor;
  let p = 0, low = 0, act = 0, dev = 0;
  for (let i = b0; i <= b1; i++) {
    const x = v(i), e = 10 ** (x / 10);
    p += e; if (i <= bLow) low += e;
    if (i >= bRough && x > fl[i] + 6) { let m = 0, c = 0; for (let j = Math.max(0, i - half); j <= Math.min(db.length - 1, i + half); j++) { m += v(j); c++; } dev += (x - m / c) ** 2; act++; }
  }
  const level = 10 * Math.log10(p + 1e-14);
  blow.level ??= level;
  const lift = level - blow.level, lowShare = low / p, cover = act / (b1 - bRough + 1), rough = act ? Math.sqrt(dev / act) : 99;
  const blowy = !warm && lift > 6 && cover > 0.15 && rough < BLOW_ROUGH && lowShare > 0.3 && crest < 5 && now - (blow.voicedAt ?? -1e9) > 250;
  blow.stats = { lift: +lift.toFixed(1), cover: +cover.toFixed(2), rough: +rough.toFixed(1), low: +lowShare.toFixed(2), crest: +crest.toFixed(1), voiced, blowy };
  // It takes a short unbroken stretch of breath noise to start, so the brief hiss of a consonant does not count.
  if (blowy) { blow.run = Math.min(0.6, blow.run + dt); blow.lastT = now; } else blow.run = Math.max(0, blow.run - dt * 4);
  const was = blow.on;
  blow.on = was ? now - blow.lastT < 300 : blowy && blow.run > 0.2;
  if (blow.on !== was) logEvent(blow.on ? 'blow' : 'blow:end');
  // The room level drops to quiet sounds fast and rises over a few seconds, so a long breath stays well above it
  // while a fan that switches on stops counting.
  const up = 1 - Math.exp(-dt / (warm ? 0.2 : 4)), down = 1 - Math.exp(-dt / 0.15);
  for (let i = 0; i < fl.length; i++) { const x = v(i); fl[i] += (x - fl[i]) * (x > fl[i] ? up : down); }
  blow.level += (level - blow.level) * (level > blow.level ? up : down);
}
function processHands(r, now) {
  const seen = new Set();
  const vw = videoEl.videoWidth || 1280, vh = videoEl.videoHeight || 720;
  const dets = (r.landmarks || []).slice(0, 2).map((lms, i) => {
    const pts = lms.map(toScreen);
    // MediaPipe assumes a mirrored input image; ours is raw, so handedness is swapped.
    return { lms, i, pts, c: palmCenter(pts), size: Math.max(20, dist(pts[0], pts[9])),
      label: r.handedness?.[i]?.[0]?.categoryName === 'Left' ? 'Right' : 'Left' };
  });
  assignLabels(dets, now);
  for (const { lms, i, pts, label } of dets) {
    seen.add(label);
    const h = hands[label] || (hands[label] = newHand(label));
    const g = r.gestures?.[i]?.[0];
    h.rawGesture = g && g.score > 0.55 ? g.categoryName : 'None';
    if (!h.present || !h.pts) { h.euro = null; h.lastWrite = null; }
    h.pts = euroPts(h, pts, now);
    h.Q = lms.map((p) => ({ x: p.x * vw, y: p.y * vh, z: p.z * vw }));
    h.palm = palmCenter(h.pts);
    h.size = Math.max(20, dist(h.pts[0], h.pts[9]));
    h.present = true; h.lastSeen = now;
    classifyHand(h, now);
    detectSwipe(h, now);
  }
  for (const h of Object.values(hands)) {
    // Motion blur during a snap often makes the tracker drop the hand for a frame or two.
    if (!seen.has(h.label) && h.present && now - h.loadT < 200 && !h.snapMaybe) h.snapMaybe = now;
    if (!seen.has(h.label) && h.present && now - h.lastSeen > 200) {
      Object.assign(h, { present: false, point: false, victory: false, pinch: false, frameL: false, open: false, lastWrite: null, lastWipe: null, euro: null, loadN: 0, _c: {} });
    }
  }
}

// ---------- face ----------
function processFace(r, now) {
  const lm = r.faceLandmarks?.[0];
  if (!lm) { if (now - face.lastSeen > 300) { face.present = false; face.mouthOpen = false; face.blowShape = false; } return; }
  const P = (i) => toScreen(lm[i]);
  const top = P(10), chin = P(152), a = P(234), b = P(454), up13 = P(13), lo14 = P(14);
  const [l, rr] = a.x < b.x ? [a, b] : [b, a];
  const faceH = Math.max(1, dist(top, chin));
  const angle = Math.atan2(rr.y - l.y, rr.x - l.x);
  const w = dist(l, rr);
  const upv = { x: Math.sin(angle), y: -Math.cos(angle) };
  const headTop = { x: top.x + upv.x * faceH * 0.05, y: top.y + upv.y * faceH * 0.05 };
  const k = face.present ? 0.5 : 1;
  face.top = face.top ? { x: lerp(face.top.x, headTop.x, k), y: lerp(face.top.y, headTop.y, k) } : headTop;
  face.angle = lerp(face.angle, angle, k);
  face.w = lerp(face.w || w, w, k);
  face.mouth = mid(up13, lo14);
  const bs = r.faceBlendshapes?.[0]?.categories;
  const score = (n) => bs?.find((c) => c.categoryName === n)?.score ?? 0;
  const jaw = score('jawOpen');
  const gap = dist(up13, lo14) / faceH;
  face.jaw = jaw; face.gap = gap; face.pucker = Math.max(score('mouthFunnel'), score('mouthPucker'));
  stab(face, 'mouthOpen', face.mouthOpen ? jaw > 0.25 || gap > 0.05 : jaw > 0.4 || gap > 0.09, 2, 3);
  // Blowing is usually done through rounded, slightly parted lips rather than a wide-open mouth.
  stab(face, 'blowShape', face.mouthOpen || (face.pucker > 0.35 && gap > 0.012), 2, 3);
  face.present = true; face.lastSeen = now;
}


// ---------- per-frame logic ----------
function orderQuad(pts) {
  const c = pts.reduce((a, p) => ({ x: a.x + p.x / pts.length, y: a.y + p.y / pts.length }), { x: 0, y: 0 });
  return pts.slice().sort((a, b) => Math.atan2(a.y - c.y, a.x - c.x) - Math.atan2(b.y - c.y, b.x - c.x));
}
function polyArea(q) {
  let s = 0;
  for (let i = 0; i < q.length; i++) { const a = q[i], b = q[(i + 1) % q.length]; s += a.x * b.y - b.x * a.y; }
  return Math.abs(s) / 2;
}
function perimeter(q) { let s = 0; for (let i = 0; i < q.length; i++) s += dist(q[i], q[(i + 1) % q.length]); return s; }

function updateFrame(now, dt) {
  const L = hands.Left, R = hands.Right;
  if (!(L?.present && R?.present)) {
    if (frame.live) logEvent('frame:lost');
    frame.live = false; frame.rearm = false;
  } else {
    // Fixed connectivity (index-index, thumb-thumb, index-thumb per hand): flipping one hand makes the
    // edges cross, so the shape twists into two triangles. The hull is only used to judge its size.
    const quad = [L.pts[8], R.pts[8], R.pts[4], L.pts[4]];
    const s = (L.size + R.size) / 2;
    const hull = orderQuad(quad);
    const area = polyArea(hull);
    // Twice area over perimeter is about the width of a thin shape, so a long flat sliver still counts as closed.
    const thick = (2 * area) / Math.max(1, perimeter(hull));
    frame.dbg = { area: +(area / (s * s)).toFixed(2), thick: +(thick / s).toFixed(2), pinchL: L.pinch, pinchR: R.pinch };
    const fresh = L.frameL && R.frameL && area > s * s * 0.6 && thick > s * 0.35 && now - frame.collapsedAt > 300;
    // Right after a pinch, the box comes back as soon as the fingers part again (no full L-shape needed),
    // so pinch, open, pinch, open flips through filters in one fluid motion.
    const reopen = frame.rearm && now - frame.collapsedAt > 60 && now - frame.collapsedAt < 1500 && L.t48 > 0.36 && R.t48 > 0.36 && area > s * s * 0.4 && thick > s * 0.28;
    if (!frame.live && (fresh || reopen)) {
      frame.live = true;
      if (reopen) frame.alpha = 1;
      logEvent(reopen ? 'frame:reopen' : 'frame');
    }
    if (frame.live) {
      frame.quad = quad;
      // Closing the shape (both hands pinched, or squeezed flat) switches the filter once; the box returns
      // with the new filter when the fingers open into a frame again.
      const closed = (L.t48 < 0.32 && R.t48 < 0.32) || area < s * s * 0.3 || thick < s * 0.24;
      if (closed) {
        frame.live = false; frame.collapsedAt = now; frame.rearm = true;
        logEvent('frame:collapse');
        setBoxFilter(boxIdx + 1, true);
      }
    }
  }
  // A closed shape is hidden at once rather than lingering as a thin sliver.
  if (now - frame.collapsedAt < 50) frame.alpha = 0;
  else frame.alpha = approach(frame.alpha, frame.live ? 1 : 0, dt * 12);
  frame.active = frame.alpha > 0.05;
}

function updateFog(now, dt) {
  const fw = fogMask.width, fh = fogMask.height;
  // Fog needs an open (or rounded) mouth and the sound of blowing; without a working mic the mouth alone does it.
  if (face.present && face.mouth && (micLive() ? face.blowShape && blow.on : face.mouthOpen)) {
    if (fog.breath === 0) logEvent('fog');
    fog.breath += dt;
    fog.lastActive = now;
    const m = face.mouth;
    const R = clamp(face.w * (0.8 + fog.breath * 2.2), 10, Math.max(W, H) * 1.1) * FOG_SCALE;
    const a = clamp(dt * 3, 0, 0.3);
    const g = fmc.createRadialGradient(m.x * FOG_SCALE, m.y * FOG_SCALE, 0, m.x * FOG_SCALE, m.y * FOG_SCALE, R);
    g.addColorStop(0, `rgba(255,255,255,${a})`);
    g.addColorStop(0.6, `rgba(255,255,255,${a * 0.6})`);
    g.addColorStop(1, 'rgba(255,255,255,0)');
    fmc.globalCompositeOperation = 'source-over';
    fmc.fillStyle = g; fmc.fillRect(0, 0, fw, fh);
    // A long breath fogs the whole screen, not just the area around the mouth.
    if (fog.breath > 0.8) { fmc.fillStyle = `rgba(255,255,255,${clamp(dt * 0.5, 0, 0.05)})`; fmc.fillRect(0, 0, fw, fh); }
    fog.any = true;
    emit(25, dt, () => spawn({ type: 'smoke', x: m.x + rand(-8, 8), y: m.y, vx: rand(-60, 60), vy: rand(-40, 30), size: face.w * rand(0.12, 0.25), decay: rand(0.8, 1.3), drag: 0.96 }));
  } else fog.breath = 0;

  for (const h of Object.values(hands)) {
    if (!h.present || !h.point || frame.active) { h.lastWrite = null; continue; }
    const tip = h.pts[8];
    const p = h.lastWrite ? { x: lerp(h.lastWrite.x, tip.x, 0.6), y: lerp(h.lastWrite.y, tip.y, 0.6) } : tip;
    if (h.lastWrite && dist(p, h.lastWrite) < h.size * 2) {
      const lw = clamp(h.size * 0.22, 12, 40);
      fog.lastActive = now;
      eraseFog(h.lastWrite, p, lw * 1.7, 0.3);
      eraseFog(h.lastWrite, p, lw, 1);
      if (Math.random() < dt * 2.5) fog.drips.push({ x: p.x + rand(-lw, lw) * 0.25, y: p.y + lw * 0.4, v: rand(50, 110), life: rand(0.8, 2.4), w: lw * rand(0.2, 0.32) });
    }
    h.lastWrite = p;
  }

  // An open palm wipes the fog like a hand on a misted window: a wide, soft-edged swath along its path.
  for (const h of Object.values(hands)) {
    if (!h.present || !h.open || !fog.any || frame.active) { h.lastWipe = null; continue; }
    const p = h.lastWipe ? { x: lerp(h.lastWipe.x, h.palm.x, 0.6), y: lerp(h.lastWipe.y, h.palm.y, 0.6) } : h.palm;
    if (h.lastWipe && dist(p, h.lastWipe) < h.size * 3) {
      const lw = h.size * 1.3;
      fog.lastActive = now;
      // A tiny offset keeps a still palm erasing (a zero-length stroke may draw nothing).
      const q = { x: p.x + 0.01, y: p.y };
      eraseFog(h.lastWipe, q, lw * 1.35, 0.25);
      eraseFog(h.lastWipe, q, lw, 0.85);
      if (Math.random() < dt * 1.5) fog.drips.push({ x: p.x + rand(-lw, lw) * 0.3, y: p.y + lw * 0.45, v: rand(40, 90), life: rand(0.8, 2), w: lw * rand(0.06, 0.1) });
    }
    h.lastWipe = p;
  }

  for (const d of fog.drips) {
    const ny = d.y + d.v * dt, nx = d.x + rand(-0.6, 0.6);
    eraseFog({ x: d.x, y: d.y }, { x: nx, y: ny }, d.w, 0.9);
    d.x = nx; d.y = ny; d.v *= Math.pow(0.7, dt); d.w *= Math.pow(0.75, dt); d.life -= dt;
  }
  fog.drips = fog.drips.filter((d) => d.life > 0 && d.w > 2 && d.y < H);

  // After the hold, clear in 0.1 s steps (8-bit alpha can't represent tiny per-frame fades), edges first like
  // a drying window, ramping up so the fade starts gently and the fog is gone about 3 s later.
  const idle = now - fog.lastActive;
  fog.evapAcc += dt;
  const clearing = fog.any && idle > FOG_HOLD;
  if (clearing && fog.evapAcc > 0.1) {
    fog.evapAcc = 0;
    const k = clamp((idle - FOG_HOLD) / 1500, 0.25, 1);
    const g = fmc.createRadialGradient(fw / 2, fh / 2, 0, fw / 2, fh / 2, Math.hypot(fw, fh) / 2);
    g.addColorStop(0, `rgba(0,0,0,${0.07 * k})`);
    g.addColorStop(1, `rgba(0,0,0,${0.22 * k})`);
    fmc.globalCompositeOperation = 'destination-out';
    fmc.fillStyle = g; fmc.fillRect(0, 0, fw, fh);
    fmc.globalCompositeOperation = 'source-over';
  }
  fog.checkAcc += dt;
  if (fog.any && fog.checkAcc > (clearing ? 0.5 : 3)) {
    fog.checkAcc = 0;
    const d = fmc.getImageData(0, 0, fw, fh).data;
    let sum = 0, n = 0; for (let i = 3; i < d.length; i += 16) { sum += d[i]; n++; }
    // Rounding in 8-bit destination-out leaves a faint residue that never fades, so wipe it once it is nearly gone.
    if (sum / n < (clearing ? 8 : 0.5) || idle > FOG_HOLD + 6000) { fmc.clearRect(0, 0, fw, fh); fog.any = false; fog.drips = []; }
  }
}

function eraseFog(a, b, width, alpha) {
  fmc.save();
  fmc.globalCompositeOperation = 'destination-out';
  fmc.lineCap = 'round'; fmc.lineJoin = 'round';
  fmc.strokeStyle = `rgba(0,0,0,${alpha})`;
  fmc.lineWidth = width * FOG_SCALE;
  fmc.beginPath(); fmc.moveTo(a.x * FOG_SCALE, a.y * FOG_SCALE); fmc.lineTo(b.x * FOG_SCALE, b.y * FOG_SCALE); fmc.stroke();
  fmc.restore();
}

// Swipe: an open hand sweeping left to right across a good part of the screen, quickly and mostly level.
function detectSwipe(h, now) {
  const tr = h.trail || (h.trail = []);
  tr.push({ t: now, x: h.palm.x, y: h.palm.y, open: h.open });
  while (tr.length && now - tr[0].t > 700) tr.shift();
  if (frame.live || now - wipe.lastT < 1200 || !h.open) return;
  const cur = tr[tr.length - 1];
  let i0 = 0;
  for (let i = 1; i < tr.length; i++) if (tr[i].x < tr[i0].x) i0 = i;
  const s = tr[i0], dx = cur.x - s.x, dy = Math.abs(cur.y - s.y), dtm = now - s.t;
  if (dx < W * 0.3 || dtm > 650 || dtm < 60 || dy > dx * 0.6) return;
  const seg = tr.slice(i0);
  if (seg.filter((p) => p.open).length < seg.length * 0.6) return;
  // Must keep moving right: a hand that went right and came back is not a swipe.
  if (Math.max(...seg.map((p) => p.x)) - cur.x > W * 0.08) return;
  h.trail = [];
  startWipe(now);
}

function startWipe(now) {
  wipe.start = now; wipe.lastT = now;
  logEvent('swipe');
}

// After a swipe the fog and the camera filter fade out together, then everything resets underneath.
function updateWipe(now) {
  if (wipe.start < 0) return;
  const p = (now - wipe.start) / WIPE_MS;
  if (p >= 1) {
    wipe.start = -1e9; wipe.fade = 1;
    fmc.clearRect(0, 0, fogMask.width, fogMask.height);
    fog.any = false; fog.drips = []; fog.breath = 0; fog.lastActive = -1e9;
    particles.length = 0;
    for (const h of Object.values(hands)) h.lastWrite = null;
    if (camIdx !== 0) { camIdx = 0; $('#camName').textContent = CAM_FILTERS[0].toLowerCase(); logEvent('cam:' + CAM_FILTERS[0]); }
    return;
  }
  const k = clamp(p, 0, 1);
  wipe.fade = 1 - k * k * (3 - 2 * k);
}

function clearFog() {
  fmc.clearRect(0, 0, fogMask.width, fogMask.height);
  fog.any = false; fog.drips = [];
  toast('Fog cleared');
}

function update(dt, now) {
  updateWipe(now);
  updateFrame(now, dt);
  updateFog(now, dt);
  pollSound(now);
  pollBlow(now);
  // A finger-tracked snap that was too blurry to be sure about only counts if a snap sound backs it up.
  for (const h of Object.values(hands)) {
    if (!h.snapMaybe) continue;
    if (snd.lastT > h.snapMaybe - 400 && snd.lastT < h.snapMaybe + 250) fireSnap(h, now, 'motion+sound');
    else if (now - h.snapMaybe > 250) h.snapMaybe = 0;
  }

  updateParticles(dt);
}

// ---------- drawing ----------
function drawVideo(now) {
  sctx.setTransform(scene.width / W, 0, 0, scene.height / H, 0, 0);
  sctx.save(); sctx.translate(W, 0); sctx.scale(-1, 1);
  sctx.drawImage(videoEl, view.ox, view.oy, view.dw, view.dh);
  sctx.restore();
  vctx.clearRect(0, 0, W, H);
  vctx.drawImage(scene, 0, 0, W, H);
  // Both filters read the unfiltered scene, so the box shows its own filter rather than one stacked on the camera's.
  let upload = true;
  if (gl && camIdx > 0) {
    renderGL(now, FILTER_MODE[CAM_FILTERS[camIdx]], upload); upload = false;
    vctx.globalAlpha = wipe.fade;
    vctx.drawImage(glc, 0, 0, W, H);
    vctx.globalAlpha = 1;
  }
  if (gl && frame.alpha > 0.01 && frame.quad) {
    renderGL(now, FILTER_MODE[BOX_FILTERS[boxIdx]], upload);
    vctx.save();
    vctx.beginPath(); frame.quad.forEach((p, i) => (i ? vctx.lineTo(p.x, p.y) : vctx.moveTo(p.x, p.y))); vctx.closePath();
    vctx.clip();
    vctx.globalAlpha = frame.alpha;
    vctx.drawImage(glc, 0, 0, W, H);
    vctx.restore();
  }
}

function drawFog(now) {
  if (!fog.any) return;
  const fw = fogComp.width, fh = fogComp.height;
  fvc.drawImage(scene, 0, 0, fogVid.width, fogVid.height);
  fcc.globalCompositeOperation = 'source-over';
  fcc.clearRect(0, 0, fw, fh);
  fcc.drawImage(fogVid, 0, 0, fw, fh);
  const haze = fcc.createLinearGradient(0, 0, 0, fh);
  haze.addColorStop(0, 'rgba(238,242,248,0.74)'); haze.addColorStop(1, 'rgba(212,222,236,0.82)');
  fcc.fillStyle = haze; fcc.fillRect(0, 0, fw, fh);
  const t = now / 1000;
  fcc.globalAlpha = 0.55; tile(fcc, clouds, t * 5, t * 1.5, fh * 0.6, fw, fh);
  fcc.globalAlpha = 0.35; tile(fcc, clouds, -t * 3, t * 0.8, fh * 1.1, fw, fh);
  fcc.globalAlpha = 1;
  fcc.drawImage(droplets, 0, 0);
  fcc.globalCompositeOperation = 'destination-in';
  fcc.drawImage(fogMask, 0, 0);
  fcc.globalCompositeOperation = 'source-over';
  ctx.globalAlpha = wipe.fade;
  ctx.drawImage(fogComp, 0, 0, W, H);
  ctx.globalAlpha = 1;
}

function drawHand(h, now) {
  const hue = HAND_HUE[h.label];
  const lw = clamp(h.size * 0.04, 1.5, 5);
  if (opts.handSkeleton) {
    ctx.save();
    ctx.globalCompositeOperation = 'lighter'; ctx.lineCap = 'round';
    ctx.strokeStyle = `hsla(${hue},100%,60%,0.6)`; ctx.lineWidth = lw;
    ctx.shadowColor = `hsl(${hue},100%,60%)`; ctx.shadowBlur = 10;
    ctx.beginPath();
    for (const [i, j] of HAND_LINKS) { ctx.moveTo(h.pts[i].x, h.pts[i].y); ctx.lineTo(h.pts[j].x, h.pts[j].y); }
    ctx.stroke();
    ctx.restore();
    ctx.globalCompositeOperation = 'lighter';
    for (const t of TIPS) drawGlow(h.pts[t].x, h.pts[t].y, lw * 3, hue, 0.8);
    ctx.globalCompositeOperation = 'source-over';
  }
  if (h.open && fog.any && !frame.active) {
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.45)'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(h.palm.x, h.palm.y, h.size * 0.65, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
  }
  if (h.point && !frame.active) {
    const tip = h.pts[8], r = clamp(h.size * 0.11, 6, 20);
    ctx.save();
    ctx.strokeStyle = 'rgba(255,255,255,0.9)'; ctx.lineWidth = 2; ctx.shadowColor = '#9ff'; ctx.shadowBlur = 10;
    ctx.beginPath(); ctx.arc(tip.x, tip.y, r, 0, Math.PI * 2); ctx.stroke();
    ctx.restore();
  }
}

function drawFrame(now) {
  if (!frame.quad || frame.alpha <= 0.01) return;
  const q = frame.quad;
  ctx.save();
  ctx.globalAlpha = frame.alpha;
  const g = ctx.createLinearGradient(q[0].x, q[0].y, q[2].x, q[2].y);
  g.addColorStop(0, '#0ff'); g.addColorStop(1, '#f0f');
  ctx.strokeStyle = g; ctx.lineWidth = 3; ctx.lineJoin = 'round';
  ctx.shadowColor = '#0ff'; ctx.shadowBlur = 18;
  ctx.setLineDash([14, 8]); ctx.lineDashOffset = -now / 30;
  ctx.beginPath(); q.forEach((p, i) => (i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y))); ctx.closePath(); ctx.stroke();
  ctx.setLineDash([]);
  ctx.globalCompositeOperation = 'lighter';
  for (const p of q) drawGlow(p.x, p.y, 16, 190, frame.alpha);
  ctx.globalCompositeOperation = 'source-over';
  const topP = q.reduce((a, p) => (p.y < a.y ? p : a));
  const cx = q.reduce((a, p) => a + p.x, 0) / 4;
  ctx.font = '700 15px system-ui, sans-serif'; ctx.textAlign = 'center';
  ctx.shadowColor = '#000'; ctx.shadowBlur = 6; ctx.fillStyle = '#fff';
  ctx.fillText(`${BOX_FILTERS[boxIdx]}  ·  pinch closed for next`, cx, Math.max(20, topP.y - 16));
  ctx.restore();
}


function render(now) {
  ctx.clearRect(0, 0, W, H);
  drawFog(now);
  for (const h of Object.values(hands)) if (h.present) drawHand(h, now);
  drawFrame(now);
  drawParticles();
}

let toastTimer = 0;
function toast(msg) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.classList.remove('show'), 1400);
}

let hudT = 0;
function updateHUD(now, fps) {
  if (now - hudT < 150) return;
  hudT = now;
  $('#fps').textContent = `${fps.toFixed(0)}`;
  const chips = [];
  if (camIdx > 0) chips.push([`Camera filter: ${CAM_FILTERS[camIdx]}`, true]);
  if (frame.active) chips.push([`Box filter: ${BOX_FILTERS[boxIdx]}`, true]);
  for (const h of Object.values(hands)) {
    if (!h.present) continue;
    const s = now - h.loadT < 350 ? 'snap ready' : h.pinch ? 'pinch' : h.point && !frame.active ? 'pointing: writing in fog' : h.open && fog.any && !frame.active ? 'open palm: wiping fog' : h.frameL ? 'framing' : 'tracked';
    chips.push([`${h.label} hand: ${s}`, s !== 'tracked']);
  }
  if (snd.on) chips.push([snd.ctx.state !== 'running' ? 'Mic: click the page to enable' : blow.on ? 'Mic: blowing' : 'Mic: listening', now - snd.lastT < 400 || blow.on]);
  if (face.present) {
    const fogging = fog.breath > 0;
    const t = fogging ? 'Blowing: fogging' : micLive() && face.blowShape ? 'Mouth open: blow to fog' : 'Face tracked';
    chips.push([micLive() ? t : face.mouthOpen ? 'Mouth open: fogging' : 'Face tracked', fogging]);
  }
  if (!chips.length) chips.push(['Step into view', false]);
  $('#chips').innerHTML = chips.map(([t, hot]) => `<span class="chip${hot ? ' hot' : ''}">${t}</span>`).join('');
}

// ---------- main loop ----------
let lastVideoTime = -1, lastT = performance.now(), fps = 60;
function loop() {
  requestAnimationFrame(loop);
  const now = performance.now();
  const dt = Math.min(0.05, (now - lastT) / 1000); lastT = now;
  fps = lerp(fps, 1 / Math.max(dt, 1e-3), 0.05);
  if (!paused && videoEl.readyState >= 2 && videoEl.currentTime !== lastVideoTime) {
    lastVideoTime = videoEl.currentTime;
    try {
      if (gesture && !freeze.hands) { processHands(gesture.recognizeForVideo(videoEl, now), now); debug.detections++; }
      if (faceLm && !freeze.face) processFace(faceLm.detectForVideo(videoEl, now), now);
    } catch (e) { if (debug.errors.length < 20) debug.errors.push(String(e)); console.error(e); }
  }
  drawVideo(now);
  update(dt, now);
  render(now);
  updateHUD(now, fps);
}

async function loadModels(status) {
  const fileset = await FilesetResolver.forVisionTasks(WASM);
  const make = (delegate) => Promise.all([
    GestureRecognizer.createFromOptions(fileset, { baseOptions: { modelAssetPath: GESTURE_MODEL, delegate }, runningMode: 'VIDEO', numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5 }),
    FaceLandmarker.createFromOptions(fileset, { baseOptions: { modelAssetPath: FACE_MODEL, delegate }, runningMode: 'VIDEO', numFaces: 1, outputFaceBlendshapes: true }),
  ]);
  try { [gesture, faceLm] = await make('GPU'); debug.delegate = 'GPU'; }
  catch (e) { console.warn('GPU delegate failed, using CPU', e); status('GPU unavailable, using CPU…'); [gesture, faceLm] = await make('CPU'); debug.delegate = 'CPU'; }
}

async function start() {
  if (running) return;
  const status = (s) => { $('#status').textContent = s; };
  if (!navigator.mediaDevices?.getUserMedia) { status(isSecureContext ? 'This browser cannot access a camera. Try a recent Chrome, Edge, Safari, or Firefox.' : 'Camera access needs a secure page. Open this site with https:// (or via http://localhost).'); return; }
  $('#btnStart').disabled = true;
  status('Requesting camera and loading tracking models…');
  try {
    const cam = navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }, audio: false });
    const models = loadModels(status);
    videoEl.srcObject = await cam;
    await videoEl.play();
    if (videoEl.readyState < 2) await new Promise((r) => videoEl.addEventListener('loadeddata', r, { once: true }));
    resize();
    status('Camera ready, loading models…');
    await models;
    try { if (!initGL()) toast('WebGL unavailable: box filters disabled'); } catch (e) { gl = null; debug.errors.push(String(e)); console.error(e); }
  } catch (e) {
    console.error(e);
    debug.errors.push(String(e));
    status(e?.name === 'NotAllowedError' ? 'Camera permission was denied. Allow camera access in the address bar and reload.' : `Could not start: ${e?.message || e}`);
    $('#btnStart').disabled = false;
    return;
  }
  running = true;
  debug.ready = true;
  startMic();
  $('#overlay').classList.add('hidden');
  toast('Press H for the gesture list');
  setTimeout(() => { if (!helpTouched) $('#help').classList.add('hidden'); }, 12000);
  requestAnimationFrame(loop);
}

// ---------- UI ----------
function snapshot() {
  const c = document.createElement('canvas');
  c.width = fx.width; c.height = fx.height;
  const x = c.getContext('2d');
  x.drawImage(vcan, 0, 0); x.drawImage(fx, 0, 0);
  c.toBlob((b) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(b); a.download = `gesture-fx-${Date.now()}.png`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  });
  toast('Snapshot saved');
}
// Pausing freezes every tracked shape where it is (the box keeps its filter in place) while the video stays live.
function setPaused(p) {
  if (!running || p === paused) return;
  paused = p;
  if (p) {
    face.mouthOpen = false; face.blowShape = false;
    for (const h of Object.values(hands)) { h.point = false; h.lastWrite = null; h.snapMaybe = 0; h.loadT = -1e9; h.trail = []; }
    fog.lastActive = Math.min(fog.lastActive, performance.now());
  }
  $('#pauseTag').classList.toggle('show', p);
  logEvent(p ? 'pause' : 'resume');
}
const toggleHelp = () => { helpTouched = true; $('#help').classList.toggle('hidden'); };
$('#btnStart').addEventListener('click', start);
$('#stage').addEventListener('click', () => setPaused(!paused));
$('#btnFilter').addEventListener('click', () => setBoxFilter(boxIdx + 1));
$('#btnCam').addEventListener('click', () => setCamFilter(camIdx + 1));
$('#btnClear').addEventListener('click', clearFog);
$('#btnSnap').addEventListener('click', snapshot);
$('#btnHelp').addEventListener('click', toggleHelp);
window.addEventListener('resize', resize);
window.addEventListener('keydown', (e) => {
  const k = e.key.toLowerCase();
  if (k >= '1' && k <= String(BOX_FILTERS.length)) setBoxFilter(Number(k) - 1);
  else if (k === 'f' || k === 'arrowright') setBoxFilter(boxIdx + 1);
  else if (k === 'arrowleft') setBoxFilter(boxIdx - 1);
  else if (k === 'g') setCamFilter(camIdx + 1);
  else if (k === ' ') { e.preventDefault(); setPaused(!paused); }
  else if (k === 'c') clearFog();
  else if (k === 'p') snapshot();
  else if (k === 'h' || k === '?') toggleHelp();
  else if (k === 's') { opts.handSkeleton = !opts.handSkeleton; toast(`Hand skeleton ${opts.handSkeleton ? 'on' : 'off'}`); }
});

window.__fx = { hands, face, frame, fog, view, snd, blow, particles, debug, freeze, opts,
  get boxFilter() { return BOX_FILTERS[boxIdx]; }, get camFilter() { return CAM_FILTERS[camIdx]; }, get paused() { return paused; }, sim: { startWipe, setPaused, setBoxFilter, setCamFilter, classifyHand, processHands, soundAt: (t) => { snd.lastT = t; }, spriteCount: () => spriteCache.size, fogMask } };
resize();
if (new URLSearchParams(location.search).has('autostart')) start();
