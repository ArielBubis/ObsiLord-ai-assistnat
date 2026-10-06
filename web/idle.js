// Idle liveness (blink, glance, bob) and the frame builder. Pure: no DOM.
// Everything is a function of absolute time in ms plus an injected random
// function, so tests can use a seeded rand and fake clocks. Events are
// scheduled by timestamp, never by counting frames.

import { isValidState } from "./state.js";

export const BLINK = Object.freeze({
  minIntervalMs: 2000,
  maxIntervalMs: 6000,
  // open -> half -> closed -> half -> open, about 120 ms in total.
  halfMs: 40,
  closedMs: 40,
  get durationMs() {
    return this.halfMs * 2 + this.closedMs;
  },
});

export const GLANCE = Object.freeze({
  minIntervalMs: 4000,
  maxIntervalMs: 10000,
  minDurationMs: 500,
  maxDurationMs: 1500,
  minPx: 1,
  maxPx: 2,
});

export const BOB = Object.freeze({
  periodMs: 4000,
});

export const SHUTTER = Object.freeze({
  durationMs: 300,
  steps: 4,
});

export const THINKING_DOT_MS = 400;

function between(rand, lo, hi) {
  return lo + rand() * (hi - lo);
}

// Eight compass directions. Straight moves are 1 or 2 px; diagonals are
// always 1 px per axis (about 1.4 px), so the total offset stays in 1..2 px.
const DIRECTIONS = [
  [1, 0], [-1, 0], [0, 1], [0, -1],
  [1, 1], [1, -1], [-1, 1], [-1, -1],
];

export function blinkPhase(msIntoBlink) {
  if (msIntoBlink < 0 || msIntoBlink >= BLINK.durationMs) return "open";
  if (msIntoBlink < BLINK.halfMs) return "half";
  if (msIntoBlink < BLINK.halfMs + BLINK.closedMs) return "closed";
  return "half";
}

export function bobAt(nowMs) {
  // Integer in {-1, 0, 1}. "+ 0" turns -0 into 0.
  return Math.round(Math.sin((2 * Math.PI * nowMs) / BOB.periodMs)) + 0;
}

export function createIdle(rand = Math.random) {
  let started = false;
  let blinkStart = 0;
  let glanceStart = 0;
  let glanceEnd = 0;
  let glanceDx = 0;
  let glanceDy = 0;

  function scheduleBlink(fromMs) {
    blinkStart = fromMs + between(rand, BLINK.minIntervalMs, BLINK.maxIntervalMs);
  }

  function scheduleGlance(fromMs) {
    glanceStart = fromMs + between(rand, GLANCE.minIntervalMs, GLANCE.maxIntervalMs);
    glanceEnd = glanceStart + between(rand, GLANCE.minDurationMs, GLANCE.maxDurationMs);
    const pickMax = rand() >= 0.5;
    const [ux, uy] = DIRECTIONS[Math.min(DIRECTIONS.length - 1, Math.floor(rand() * DIRECTIONS.length))];
    const diagonal = ux !== 0 && uy !== 0;
    const mag = pickMax && !diagonal ? GLANCE.maxPx : GLANCE.minPx;
    glanceDx = ux * mag;
    glanceDy = uy * mag;
  }

  return {
    // Exposed for tests and debugging.
    get nextBlinkAt() {
      return blinkStart;
    },
    get nextGlance() {
      return { start: glanceStart, end: glanceEnd, dx: glanceDx, dy: glanceDy };
    },
    update(nowMs) {
      if (!started) {
        started = true;
        scheduleBlink(nowMs);
        scheduleGlance(nowMs);
      }
      // A finished blink schedules the next one from its end. If the tab was
      // hidden and we are far past it, schedule from now instead of
      // replaying missed blinks.
      const blinkEnd = blinkStart + BLINK.durationMs;
      if (nowMs >= blinkEnd) {
        scheduleBlink(nowMs - blinkEnd > BLINK.maxIntervalMs ? nowMs : blinkEnd);
      }
      if (nowMs >= glanceEnd) {
        scheduleGlance(nowMs - glanceEnd > GLANCE.maxIntervalMs ? nowMs : glanceEnd);
      }
      const glancing = nowMs >= glanceStart && nowMs < glanceEnd;
      return {
        blink: blinkPhase(nowMs - blinkStart),
        glance: glancing ? { dx: glanceDx, dy: glanceDy } : { dx: 0, dy: 0 },
        bobY: bobAt(nowMs),
      };
    },
  };
}

// Idle values with no motion, used by the debug "idle=0" switch.
export const STILL_IDLE = Object.freeze({ blink: "open", glance: Object.freeze({ dx: 0, dy: 0 }), bobY: 0 });

export function clampMouth(value) {
  const n = Math.round(Number(value));
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(3, n));
}

export function shutterAt(state, sinceStateMs) {
  if (state !== "looking" || sinceStateMs < 0 || sinceStateMs >= SHUTTER.durationMs) return 0;
  // Quantized so the frame description only changes a few times.
  return Math.ceil((1 - sinceStateMs / SHUTTER.durationMs) * SHUTTER.steps) / SHUTTER.steps;
}

export function thinkingDotsAt(state, sinceStateMs) {
  if (state !== "thinking") return 0;
  return 1 + (Math.floor(Math.max(0, sinceStateMs) / THINKING_DOT_MS) % 3);
}

// Builds the frame description the skin draws. Pure.
// shutter: 0..1 white flash strength (looking only, right after entry).
// dots: 0 or 1..3 thinking dots shown (thinking only).
export function buildFrame({ state, idle, mouth = 0, sinceStateMs = 0 }) {
  if (!isValidState(state)) {
    throw new Error(`unknown face state: ${state}`);
  }
  const i = idle || STILL_IDLE;
  return {
    state,
    blink: i.blink,
    glance: { dx: i.glance.dx, dy: i.glance.dy },
    bobY: i.bobY,
    mouth: clampMouth(mouth),
    shutter: shutterAt(state, sinceStateMs),
    dots: thinkingDotsAt(state, sinceStateMs),
  };
}

export function frameKey(f) {
  return `${f.state}|${f.blink}|${f.glance.dx},${f.glance.dy}|${f.bobY}|${f.mouth}|${f.shutter}|${f.dots}`;
}
