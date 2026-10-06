import { test } from "node:test";
import assert from "node:assert/strict";

import {
  BLINK, GLANCE, SHUTTER,
  createIdle, blinkPhase, bobAt, buildFrame, clampMouth, frameKey, shutterAt, thinkingDotsAt,
} from "../../web/idle.js";
import { STATES } from "../../web/state.js";

// Deterministic PRNG (mulberry32) so interval tests are reproducible.
function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test("blink sequence is open, half, closed, half, open within about 120 ms", () => {
  assert.ok(BLINK.durationMs >= 100 && BLINK.durationMs <= 140, `duration ${BLINK.durationMs}`);
  const seq = [];
  for (let t = -10; t <= BLINK.durationMs + 10; t++) {
    const p = blinkPhase(t);
    if (seq[seq.length - 1] !== p) seq.push(p);
  }
  assert.deepEqual(seq, ["open", "half", "closed", "half", "open"]);
});

test("blink sequence through createIdle", () => {
  const idle = createIdle(seeded(1));
  idle.update(0);
  const start = idle.nextBlinkAt;
  const seq = [];
  for (let t = start - 20; t < start + 200; t += 5) {
    const p = idle.update(t).blink;
    if (seq[seq.length - 1] !== p) seq.push(p);
  }
  assert.deepEqual(seq, ["open", "half", "closed", "half", "open"]);
});

test("blink intervals stay within 2 to 6 s with jitter", () => {
  const idle = createIdle(seeded(42));
  const starts = [];
  let last = "open";
  for (let t = 0; t <= 600000; t += 10) {
    const p = idle.update(t).blink;
    if (p !== "open" && last === "open") starts.push(t);
    last = p;
  }
  assert.ok(starts.length > 90, `only ${starts.length} blinks in 10 min`);
  const gaps = starts.slice(1).map((s, i) => s - starts[i]);
  for (const g of gaps) {
    // Gap is measured start to start: interval (2 to 6 s) plus one blink,
    // with up to one 10 ms sampling step of slack.
    assert.ok(g >= BLINK.minIntervalMs - 10 && g <= BLINK.maxIntervalMs + BLINK.durationMs + 10, `gap ${g}`);
  }
  assert.ok(new Set(gaps).size > 10, "intervals should vary");
});

test("glance magnitude is 1 to 2 px and duration 0.5 to 1.5 s, every 4 to 10 s", () => {
  const idle = createIdle(seeded(7));
  const glances = [];
  let cur = null;
  for (let t = 0; t <= 600000; t += 10) {
    const g = idle.update(t).glance;
    const active = g.dx !== 0 || g.dy !== 0;
    if (active && !cur) cur = { start: t, dx: g.dx, dy: g.dy };
    if (!active && cur) {
      cur.end = t;
      glances.push(cur);
      cur = null;
    }
  }
  assert.ok(glances.length > 40, `only ${glances.length} glances`);
  for (const g of glances) {
    const mag = Math.hypot(g.dx, g.dy);
    assert.ok(mag >= GLANCE.minPx && mag <= GLANCE.maxPx, `mag ${mag} at (${g.dx},${g.dy})`);
    assert.ok(!(Math.abs(g.dx) === 2 && Math.abs(g.dy) === 2), "no 2 px diagonal");
    assert.ok(Number.isInteger(g.dx) && Number.isInteger(g.dy));
    const dur = g.end - g.start;
    assert.ok(dur >= GLANCE.minDurationMs - 10 && dur <= GLANCE.maxDurationMs + 10, `duration ${dur}`);
  }
  for (let i = 1; i < glances.length; i++) {
    const gap = glances[i].start - glances[i - 1].end;
    assert.ok(gap >= GLANCE.minIntervalMs - 10 && gap <= GLANCE.maxIntervalMs + 10, `gap ${gap}`);
  }
});

test("glance covers straight 1 px, straight 2 px, and 1 px diagonals", () => {
  const idle = createIdle(seeded(11));
  const kinds = new Set();
  for (let t = 0; t <= 1200000; t += 10) {
    const { dx, dy } = idle.update(t).glance;
    if (dx === 0 && dy === 0) continue;
    const diagonal = dx !== 0 && dy !== 0;
    kinds.add(diagonal ? `diag${Math.abs(dx)}${Math.abs(dy)}` : `straight${Math.abs(dx) + Math.abs(dy)}`);
  }
  assert.deepEqual([...kinds].sort(), ["diag11", "straight1", "straight2"]);
});

test("long pause (hidden tab) does not replay missed blinks", () => {
  const idle = createIdle(seeded(3));
  idle.update(0);
  idle.update(3600000);
  const next = idle.nextBlinkAt;
  assert.ok(next >= 3600000 + BLINK.minIntervalMs && next <= 3600000 + BLINK.maxIntervalMs);
});

test("bob is an integer within +-1 px and uses both directions", () => {
  const seen = new Set();
  for (let t = 0; t < 20000; t += 7) {
    const y = bobAt(t);
    assert.ok(Number.isInteger(y) && y >= -1 && y <= 1, `bob ${y}`);
    assert.ok(!Object.is(y, -0));
    seen.add(y);
  }
  assert.deepEqual([...seen].sort(), [-1, 0, 1]);
});

test("buildFrame returns the frame description shape", () => {
  const idle = createIdle(seeded(9));
  for (const state of STATES) {
    const f = buildFrame({ state, idle: idle.update(1234), mouth: 2, sinceStateMs: 50 });
    assert.deepEqual(Object.keys(f).sort(), ["blink", "bobY", "dots", "glance", "mouth", "shutter", "state"]);
    assert.equal(f.state, state);
    assert.ok(["open", "half", "closed"].includes(f.blink));
    assert.deepEqual(Object.keys(f.glance).sort(), ["dx", "dy"]);
    assert.ok(Number.isInteger(f.bobY));
    assert.equal(f.mouth, 2);
    assert.ok(f.shutter >= 0 && f.shutter <= 1);
    assert.equal(typeof frameKey(f), "string");
  }
  assert.throws(() => buildFrame({ state: "dancing" }), /unknown face state/);
});

test("mouth is clamped to 0..3 integers", () => {
  const cases = [[-5, 0], [0, 0], [1, 1], [2.4, 2], [2.6, 3], [3, 3], [9, 3], ["2", 2], [NaN, 0], [undefined, 0], [Infinity, 0]];
  for (const [input, want] of cases) {
    assert.equal(clampMouth(input), want, `input ${input}`);
    assert.equal(buildFrame({ state: "speaking", mouth: input }).mouth, want);
  }
});

test("shutter flashes only on entering looking and fades out", () => {
  assert.equal(shutterAt("looking", 0), 1);
  assert.ok(shutterAt("looking", SHUTTER.durationMs / 2) > 0);
  assert.equal(shutterAt("looking", SHUTTER.durationMs), 0);
  assert.equal(shutterAt("looking", 5000), 0);
  assert.equal(shutterAt("idle", 0), 0);
  let prev = 1;
  for (let t = 0; t < SHUTTER.durationMs; t += 5) {
    const s = shutterAt("looking", t);
    assert.ok(s <= prev && s > 0);
    prev = s;
  }
});

test("thinking dots cycle 1..3 only in thinking", () => {
  const seen = new Set();
  for (let t = 0; t < 5000; t += 50) seen.add(thinkingDotsAt("thinking", t));
  assert.deepEqual([...seen].sort(), [1, 2, 3]);
  assert.equal(thinkingDotsAt("idle", 1000), 0);
});

test("frames with equal content have equal keys", () => {
  const a = buildFrame({ state: "idle", mouth: 0 });
  const b = buildFrame({ state: "idle", mouth: 0 });
  const c = buildFrame({ state: "idle", mouth: 1 });
  assert.equal(frameKey(a), frameKey(b));
  assert.notEqual(frameKey(a), frameKey(c));
});
