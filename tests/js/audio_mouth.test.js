import { test } from "node:test";
import assert from "node:assert/strict";

import { FACE_CONFIG } from "../../web/config.js";
import {
  AUDIO_MOUTH, smoothLevel, createLevelSmoother, rmsToFrame, mouthSource,
} from "../../web/mouth.js";
import { rms } from "../../web/audio.js";

test("rmsToFrame uses the configured thresholds", () => {
  assert.deepEqual([...FACE_CONFIG.mouthThresholds], [0.05, 0.15, 0.3]);
  assert.equal(rmsToFrame(0), 0);
  assert.equal(rmsToFrame(0.049), 0);
  assert.equal(rmsToFrame(0.05), 1);
  assert.equal(rmsToFrame(0.149), 1);
  assert.equal(rmsToFrame(0.15), 2);
  assert.equal(rmsToFrame(0.299), 2);
  assert.equal(rmsToFrame(0.3), 3);
  assert.equal(rmsToFrame(1), 3);
  assert.equal(rmsToFrame(NaN), 0);
});

test("smoothLevel: attack is faster than release", () => {
  assert.ok(AUDIO_MOUTH.attackMs < AUDIO_MOUTH.releaseMs);
  const up = smoothLevel(0, 1, 16);
  const down = 1 - smoothLevel(1, 0, 16);
  assert.ok(up > down, `up ${up} down ${down}`);
  assert.ok(up > 0.5, "opens within one frame");
});

test("smoothLevel does not depend on the frame rate", () => {
  // 100 ms of attack at 60 fps vs 144 fps vs one big step.
  const run = (fps, target, from, ms) => {
    let v = from;
    const steps = Math.round((ms * fps) / 1000);
    for (let i = 0; i < steps; i++) v = smoothLevel(v, target, ms / steps);
    return v;
  };
  const a = run(60, 0.4, 0, 100);
  const b = run(144, 0.4, 0, 100);
  const c = smoothLevel(0, 0.4, 100);
  assert.ok(Math.abs(a - c) < 1e-3 && Math.abs(b - c) < 1e-3, `${a} ${b} ${c}`);
  const r60 = run(60, 0, 0.4, 150);
  const r144 = run(144, 0, 0.4, 150);
  assert.ok(Math.abs(r60 - r144) < 1e-3);
});

test("smoothLevel handles zero, negative, and bad dt and target", () => {
  assert.equal(smoothLevel(0.3, 1, 0), 0.3);
  assert.equal(smoothLevel(0.3, 1, -5), 0.3);
  assert.equal(smoothLevel(0.3, 1, NaN), 0.3);
  assert.ok(smoothLevel(0.3, NaN, 1000) < 0.01);
  assert.ok(smoothLevel(0.3, -1, 1000) >= 0);
});

test("mouth closes within about 250 ms after audio stops", () => {
  const s = createLevelSmoother();
  for (let i = 0; i < 30; i++) s.update(0.5, 16);
  assert.equal(rmsToFrame(s.value), 3);
  let t = 0;
  while (rmsToFrame(s.value) !== 0) {
    s.update(0, 16);
    t += 16;
    assert.ok(t < 400, "still open");
  }
  assert.ok(t <= 250, `closed after ${t} ms`);
  s.reset();
  assert.equal(s.value, 0);
});

test("a tone with a syllable envelope makes the mouth move through several frames", () => {
  // 220 Hz tone at 48 kHz, amplitude 0.9 * |sin| at 4 syllables per second,
  // analysed in 1024-sample windows at about 60 fps like the face loop.
  const rate = 48000;
  const total = rate * 2;
  const tone = new Float32Array(total);
  for (let i = 0; i < total; i++) {
    const t = i / rate;
    tone[i] = 0.9 * Math.abs(Math.sin(2 * Math.PI * 2 * t)) * Math.sin(2 * Math.PI * 220 * t);
  }
  const s = createLevelSmoother();
  const seen = new Map();
  for (let ms = 0; ms < 2000; ms += 16) {
    const end = Math.floor((ms / 1000) * rate);
    const win = tone.subarray(Math.max(0, end - 1024), end);
    const f = rmsToFrame(s.update(rms(win), 16));
    seen.set(f, (seen.get(f) || 0) + 1);
  }
  for (const f of [0, 1, 2, 3]) assert.ok(seen.has(f), `frame ${f} never shown: ${[...seen]}`);
});

test("mouthSource precedence: manual, then audio, then envelope", () => {
  assert.equal(mouthSource(true, true), "manual");
  assert.equal(mouthSource(true, false), "manual");
  assert.equal(mouthSource(false, true), "audio");
  assert.equal(mouthSource(false, false), "envelope");
});
