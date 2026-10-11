// Fake mouth envelope for text-only replies (Phase 2). Pure: no DOM, no
// timers. Time is passed in by the caller in ms.
//
// Model:
//   level: 0..1 loudness stand-in. Each chunk of new text raises it and
//     extends a "hold" window proportional to the chunk length (about speech
//     pace), so a big chunk keeps the mouth moving for a while. After the hold
//     ends, level falls linearly to 0 over decayMs.
//   flap: a syllable oscillator. Time is cut into syllables of syllableMs;
//     inside each syllable openness follows sin(pi * phase), so the mouth
//     closes between syllables. Each syllable gets a pseudo-random amplitude
//     from a hash of its index, scaled by how "open" the vowels in the last
//     chunk were (a, o open wide; i, u less).
//   frame = quantize(level * flap) into 0..3.
//
// end() (stream finished) cuts the hold short, so the mouth closes within
// decayMs no matter how much text was still "owed".
//
// When the turn has audio (Phase 3), the mouth follows the playback
// AnalyserNode RMS instead: see smoothLevel and rmsToFrame below.

import { getMouthThresholds } from "./config.js";

export const ENVELOPE = Object.freeze({
  msPerChar: 30,          // hold time added per new character
  maxHoldAheadMs: 2000,   // hold never extends further than this past now
  decayMs: 450,           // full level to 0 after the hold ends
  syllableMs: 150,
  baseGain: 0.5,          // level added by any chunk
  vowelGain: 0.08,        // extra level per vowel in the chunk
  // Cutoffs on level * flap for frames 1, 2, 3.
  thresholds: Object.freeze([0.12, 0.4, 0.68]),
});

const OPEN_VOWELS = "aoAO";
const VOWELS = "aeiouyAEIOUY";

export function countVowels(text) {
  let vowels = 0;
  let open = 0;
  for (const ch of String(text)) {
    if (VOWELS.includes(ch)) vowels++;
    if (OPEN_VOWELS.includes(ch)) open++;
  }
  return { vowels, open };
}

// Integer hash to 0..1, deterministic per syllable index.
export function hash01(n) {
  let x = (n | 0) ^ 0x9e3779b9;
  x = Math.imul(x ^ (x >>> 16), 0x85ebca6b);
  x = Math.imul(x ^ (x >>> 13), 0xc2b2ae35);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

// The server sends cumulative assistant text. Returns the part that is new
// since prev; if the text was rewritten (no common prefix), all of it is new.
export function appendedText(prev, next) {
  const a = String(prev ?? "");
  const b = String(next ?? "");
  return b.startsWith(a) ? b.slice(a.length) : b;
}

export function levelToFrame(v, thresholds = ENVELOPE.thresholds) {
  const n = Number(v);
  if (!Number.isFinite(n) || n < thresholds[0]) return 0;
  if (n < thresholds[1]) return 1;
  if (n < thresholds[2]) return 2;
  return 3;
}

// Openness 0..1 inside the syllable at nowMs, with a per-syllable amplitude.
export function flapAt(nowMs, openBias = 0.5, cfg = ENVELOPE) {
  const t = Math.max(0, nowMs);
  const index = Math.floor(t / cfg.syllableMs);
  const phase = (t - index * cfg.syllableMs) / cfg.syllableMs;
  const amp = 0.45 + 0.35 * hash01(index) + 0.2 * openBias;
  return Math.min(1, amp) * Math.sin(Math.PI * phase);
}

export function createMouthEnvelope(cfg = ENVELOPE) {
  let level = 0;
  let holdUntil = -Infinity;
  let lastT = null;
  let openBias = 0.5;

  // Advance level to nowMs. Level is constant during hold, then falls
  // linearly. Time never runs backwards.
  function advance(nowMs) {
    if (lastT === null) {
      lastT = nowMs;
      return;
    }
    if (nowMs <= lastT) return;
    const decayFrom = Math.max(lastT, holdUntil);
    if (nowMs > decayFrom) {
      level = Math.max(0, level - (nowMs - decayFrom) / cfg.decayMs);
    }
    lastT = nowMs;
  }

  return {
    // New text arrived. Empty chunks are ignored.
    energize(text, nowMs) {
      const s = String(text ?? "");
      if (s.length === 0) return;
      advance(nowMs);
      const { vowels, open } = countVowels(s);
      level = Math.min(1, level + cfg.baseGain + cfg.vowelGain * vowels);
      if (vowels > 0) openBias = open / vowels;
      const from = Math.max(holdUntil, nowMs);
      holdUntil = Math.min(nowMs + cfg.maxHoldAheadMs, from + s.length * cfg.msPerChar);
    },
    // Stream finished: drop the remaining hold so the mouth closes within
    // decayMs from now.
    end(nowMs) {
      advance(nowMs);
      holdUntil = Math.min(holdUntil, nowMs);
    },
    // Immediate close, used on disconnect.
    reset() {
      level = 0;
      holdUntil = -Infinity;
      lastT = null;
    },
    level(nowMs) {
      advance(nowMs);
      return level;
    },
    frame(nowMs) {
      advance(nowMs);
      if (level <= 0) return 0;
      return levelToFrame(level * flapAt(nowMs, openBias, cfg), cfg.thresholds);
    },
  };
}

// Audio-driven mouth. Time constants for the RMS follower: opens fast,
// closes a little slower so the mouth does not chatter between frames.
export const AUDIO_MOUTH = Object.freeze({
  attackMs: 15,
  releaseMs: 60,
});

// One step of an exponential follower toward target over dtMs. Uses the
// attack time constant when rising and the release one when falling, so the
// result does not depend on the frame rate.
export function smoothLevel(prev, target, dtMs, cfg = AUDIO_MOUTH) {
  const t = Number.isFinite(target) ? Math.max(0, target) : 0;
  const dt = Number.isFinite(dtMs) ? Math.max(0, dtMs) : 0;
  const tau = t > prev ? cfg.attackMs : cfg.releaseMs;
  return prev + (t - prev) * (1 - Math.exp(-dt / tau));
}

export function createLevelSmoother(cfg = AUDIO_MOUTH) {
  let value = 0;
  return {
    update(target, dtMs) {
      value = smoothLevel(value, target, dtMs, cfg);
      return value;
    },
    reset() {
      value = 0;
    },
    get value() {
      return value;
    },
  };
}

// Playback RMS to a mouth frame 0..3 using the configured cutoffs (from the
// server's /config/face when loaded, else the fallback in config.js).
export function rmsToFrame(level, thresholds = getMouthThresholds()) {
  return levelToFrame(level, thresholds);
}

// Which input drives the mouth: a touched debug slider always wins; a turn
// that has audio uses the analyser; otherwise the fake text envelope.
export function mouthSource(manualSet, audioMode) {
  if (manualSet) return "manual";
  return audioMode ? "audio" : "envelope";
}
