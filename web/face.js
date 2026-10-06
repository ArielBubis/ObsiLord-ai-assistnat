// Face loop: reads the state machine, idle liveness, and the current mouth
// value, builds a frame description, and hands it to the skin. The idle loop
// runs under every state. The skin is only called when the frame changes, so
// a static face costs one cheap comparison per animation frame.

import { FACE_CONFIG } from "./config.js";
import { createStateMachine, isValidState } from "./state.js";
import { createIdle, buildFrame, frameKey, clampMouth, STILL_IDLE } from "./idle.js";
import { createMouthEnvelope } from "./mouth.js";
import * as skin from "./skin_code.js";

const machine = createStateMachine("idle", performance.now());
const idle = createIdle(Math.random);
const envelope = createMouthEnvelope();
// Mouth precedence: a manual value (debug slider or ?mouth=) wins over the
// text envelope until clearMouthOverride() is called. Without an override the
// envelope drives the mouth; it rests at 0 when no text is streaming.
let manualMouth = null;
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

// Manual mouth frame 0..3 (debug). Overrides the envelope. Phase 3 will feed
// the mouth from the playback AnalyserNode RMS using
// FACE_CONFIG.mouthThresholds.
export function setMouth(value) {
  manualMouth = clampMouth(value);
  mouth = manualMouth;
}

export function clearMouthOverride() {
  manualMouth = null;
}

// Streamed reply text: each new chunk energizes the fake mouth envelope.
export function mouthText(chunk) {
  envelope.energize(chunk, performance.now());
}

// Reply finished: the envelope decays and the mouth closes on its own.
export function mouthEnd() {
  envelope.end(performance.now());
}

export function mouthReset() {
  envelope.reset();
}

export function getMouth() {
  return mouth;
}

// Debug only: freeze blink, glance, and bob for deterministic screenshots.
export function setIdleEnabled(enabled) {
  idleEnabled = Boolean(enabled);
}

// Vertical CSS px kept free for the subtitle, input, error, and status rows
// under the face (see style.css; #subtitles has a fixed height).
const UI_RESERVE_PX = 250;

// Pick an integer number of device pixels per logical pixel (k), then size
// the canvas in CSS px as 64*k/dpr, so every logical pixel is exactly k
// device pixels wide even at 125% or 150% display scaling.
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const availDevice = Math.min(
    (window.innerWidth * dpr) / FACE_CONFIG.logicalWidth,
    (Math.max(FACE_CONFIG.logicalHeight, window.innerHeight - UI_RESERVE_PX) * dpr) / FACE_CONFIG.logicalHeight,
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
  mouth = manualMouth !== null ? manualMouth : envelope.frame(now);
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
