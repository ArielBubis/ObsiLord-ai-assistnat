// Face loop: reads the state machine, idle liveness, and the current mouth
// value, builds a frame description, and hands it to the skin. The idle loop
// runs under every state. The skin is only called when the frame changes, so
// a static face costs one cheap comparison per animation frame.

import { FACE_CONFIG } from "./config.js";
import { createStateMachine, isValidState } from "./state.js";
import { createIdle, buildFrame, frameKey, clampMouth, STILL_IDLE } from "./idle.js";
import * as skin from "./skin_code.js";

const machine = createStateMachine("idle", performance.now());
const idle = createIdle(Math.random);
let mouth = 0;
let idleEnabled = true;
let ctx = null;
let lastKey = "";
let started = false;

// Returns true if the name was valid. Unknown names are logged and ignored.
// enteredAgoMs backdates the state entry; only the debug panel uses it, to
// skip past entry animations (shutter flash) for deterministic screenshots.
export function setState(name, enteredAgoMs = 0) {
  if (!isValidState(name)) {
    console.warn("face: ignoring unknown state", name);
    return false;
  }
  machine.set(name, performance.now() - Math.max(0, Number(enteredAgoMs) || 0));
  return true;
}

export function getState() {
  return machine.state;
}

export function onStateChange(fn) {
  return machine.subscribe(fn);
}

// Mouth frame 0..3. Phase 3 will feed this from the playback AnalyserNode RMS
// using FACE_CONFIG.mouthThresholds.
export function setMouth(value) {
  mouth = clampMouth(value);
}

export function getMouth() {
  return mouth;
}

// Debug only: freeze blink, glance, and bob for deterministic screenshots.
export function setIdleEnabled(enabled) {
  idleEnabled = Boolean(enabled);
}

// Pick an integer number of device pixels per logical pixel (k), then size
// the canvas in CSS px as 64*k/dpr, so every logical pixel is exactly k
// device pixels wide even at 125% or 150% display scaling.
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const availDevice = Math.min(
    (window.innerWidth * dpr) / FACE_CONFIG.logicalWidth,
    (window.innerHeight * 0.8 * dpr) / FACE_CONFIG.logicalHeight,
  );
  const k = Math.max(1, Math.floor(availDevice));
  canvas.style.width = `${(FACE_CONFIG.logicalWidth * k) / dpr}px`;
  canvas.style.height = `${(FACE_CONFIG.logicalHeight * k) / dpr}px`;
}

// devicePixelRatio changes (zoom, moving to another monitor) do not always
// fire resize, so watch the current resolution and re-arm on each change.
function watchDpr(canvas) {
  const mq = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
  mq.addEventListener(
    "change",
    () => {
      fitCanvas(canvas);
      watchDpr(canvas);
    },
    { once: true },
  );
}

function tick(now) {
  const idleValues = idleEnabled ? idle.update(now) : STILL_IDLE;
  const frame = buildFrame({
    state: machine.state,
    idle: idleValues,
    mouth,
    sinceStateMs: now - machine.enteredAt,
  });
  const key = frameKey(frame);
  if (key !== lastKey) {
    lastKey = key;
    skin.draw(ctx, frame);
  }
  requestAnimationFrame(tick);
}

export function start(canvas) {
  if (started) return;
  started = true;
  canvas.width = FACE_CONFIG.logicalWidth;
  canvas.height = FACE_CONFIG.logicalHeight;
  ctx = canvas.getContext("2d", { alpha: false });
  ctx.imageSmoothingEnabled = false;
  fitCanvas(canvas);
  window.addEventListener("resize", () => fitCanvas(canvas));
  watchDpr(canvas);
  requestAnimationFrame(tick);
}
