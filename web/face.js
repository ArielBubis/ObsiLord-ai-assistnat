// Face loop: reads the state machine, idle liveness, and the current mouth
// value, builds a frame description, and hands it to the skin. The idle loop
// runs under every state. The skin is only called when the frame changes, so
// a static face costs one cheap comparison per animation frame.
//
// The skin module ([face] skin: "dither" or "code") is imported once the face
// config has loaded. The loop runs from start(); frames are drawn once the
// skin is in.

import { FACE_CONFIG, loadFaceConfig, getFaceSkin, skinModulePath } from "./config.js";
import { createStateMachine, isValidState } from "./state.js";
import { createIdle, buildFrame, frameKey, clampMouth, STILL_IDLE } from "./idle.js";
import { createMouthEnvelope, createLevelSmoother, rmsToFrame, mouthSource } from "./mouth.js";

const machine = createStateMachine("idle", performance.now());
const idle = createIdle(Math.random);
const envelope = createMouthEnvelope();
const smoother = createLevelSmoother();
// Mouth precedence: a manual value (debug slider or ?mouth=) wins until
// clearMouthOverride() is called. Otherwise, in audio mode (the turn has TTS
// audio) the playback RMS drives the mouth; without audio the fake text
// envelope does. Both rest at 0 when nothing is playing or streaming.
let manualMouth = null;
let mouth = 0;
let audioMode = false;
let audioLevel = () => 0;
let lastTickT = null;
let idleEnabled = true;
let ctx = null;
let lastKey = "";
let started = false;
let skin = null;
let resolveSkin;
let drawErrorLogged = false;

// Resolves with the name of the skin in use ("dither" or "code") once its
// module has loaded, before its first draw. Never rejects (stays pending if
// no skin loads).
export const skinReady = new Promise((resolve) => {
  resolveSkin = resolve;
});

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

// Manual mouth frame 0..3 (debug). Overrides audio and envelope.
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
  smoother.reset();
}

// fn() returns the current playback RMS (0 when silent).
export function setAudioLevelSource(fn) {
  audioLevel = typeof fn === "function" ? fn : () => 0;
}

// On: the mouth follows playback audio for this turn. Off: back to the text
// envelope (set at the start of each turn).
export function setAudioMouth(on) {
  audioMode = Boolean(on);
  smoother.reset();
}

export function getMouth() {
  return mouth;
}

// Debug only: freeze blink, glance, and bob for deterministic screenshots.
export function setIdleEnabled(enabled) {
  idleEnabled = Boolean(enabled);
}

// CSS px kept free around the face for the chrome in style.css (the sum is
// spelled out in its comment on main). Vertical: the top bar, the panel
// padding and caption, the transcript, the control row, the hint and error
// rows (all fixed heights), the gaps between them, and the page foot.
// Horizontal: the page and panel side padding.
const UI_RESERVE_PX = 408;
const UI_RESERVE_X_PX = 72;

// Pick an integer number of device pixels per logical pixel (k), then size
// the canvas in CSS px as 64*k/dpr, so every logical pixel is exactly k
// device pixels wide even at 125% or 150% display scaling.
function fitCanvas(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const availDevice = Math.min(
    (Math.max(FACE_CONFIG.logicalWidth, window.innerWidth - UI_RESERVE_X_PX) * dpr) / FACE_CONFIG.logicalWidth,
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
  const dt = lastTickT === null ? 0 : now - lastTickT;
  lastTickT = now;
  const source = mouthSource(manualMouth !== null, audioMode);
  if (source === "manual") {
    mouth = manualMouth;
  } else if (source === "audio") {
    mouth = rmsToFrame(smoother.update(audioLevel(), dt));
  } else {
    mouth = envelope.frame(now);
  }
  const frame = buildFrame({
    state: machine.state,
    idle: idleValues,
    mouth,
    sinceStateMs: now - machine.enteredAt,
  });
  const key = frameKey(frame);
  // lastKey stays "" until the skin is in, so its first frame is drawn.
  // A draw that throws must not stop the loop. lastKey is still updated, so
  // the same frame is not retried every animation frame, and the next
  // different frame is drawn as usual. Only the first error is logged.
  if (skin && key !== lastKey) {
    lastKey = key;
    try {
      skin.draw(ctx, frame);
    } catch (err) {
      if (!drawErrorLogged) {
        drawErrorLogged = true;
        console.error("face: the skin failed to draw a frame (later errors are not logged)", err);
      }
    }
  }
  requestAnimationFrame(tick);
}

// Imports one skin module. Returns null (and logs why) if it does not load.
async function importSkin(name) {
  try {
    return await import(skinModulePath(name));
  } catch (err) {
    console.error(`face: the ${name} skin did not load`, err);
    return null;
  }
}

// Mouth thresholds and skin from the server (the fallbacks apply if that
// fails), then the skin module. A dither skin module that does not load (the
// adapter handles WebGL failures itself) falls back to the code skin. Never
// throws.
async function loadSkin() {
  await loadFaceConfig();
  let name = getFaceSkin();
  let mod = await importSkin(name);
  if (!mod && name !== "code") {
    console.warn("face: using the code skin instead");
    name = "code";
    mod = await importSkin(name);
  }
  if (!mod) return;
  skin = mod;
  resolveSkin(name);
}

export function start(canvas) {
  if (started) return;
  started = true;
  loadSkin();
  canvas.width = FACE_CONFIG.logicalWidth;
  canvas.height = FACE_CONFIG.logicalHeight;
  ctx = canvas.getContext("2d", { alpha: false });
  ctx.imageSmoothingEnabled = false;
  fitCanvas(canvas);
  window.addEventListener("resize", () => fitCanvas(canvas));
  watchDpr(canvas);
  requestAnimationFrame(tick);
}
