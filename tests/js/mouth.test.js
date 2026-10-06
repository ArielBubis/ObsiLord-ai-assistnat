import { test } from "node:test";
import assert from "node:assert/strict";

import {
  ENVELOPE, createMouthEnvelope, levelToFrame, flapAt, appendedText, countVowels, hash01,
} from "../../web/mouth.js";

test("level starts at 0 and the frame is closed", () => {
  const env = createMouthEnvelope();
  assert.equal(env.level(0), 0);
  assert.equal(env.frame(0), 0);
  assert.equal(env.frame(5000), 0);
});

test("energize raises the level, empty chunks do nothing", () => {
  const env = createMouthEnvelope();
  env.energize("", 0);
  assert.equal(env.level(0), 0);
  env.energize("hello", 0);
  const l = env.level(0);
  assert.ok(l > 0 && l <= 1, `level ${l}`);
});

test("level never exceeds 1", () => {
  const env = createMouthEnvelope();
  for (let i = 0; i < 50; i++) env.energize("aaaaaaaaaa", i);
  assert.ok(env.level(50) <= 1);
});

test("level holds for about msPerChar per char, then decays linearly to 0", () => {
  const env = createMouthEnvelope();
  env.energize("abcdefghij", 0); // 10 chars -> hold 300 ms
  const start = env.level(0);
  const holdMs = 10 * ENVELOPE.msPerChar;
  assert.equal(env.level(holdMs - 1), start);
  const mid = env.level(holdMs + ENVELOPE.decayMs / 4);
  assert.ok(mid < start && mid > 0, `mid ${mid}`);
  assert.equal(env.level(holdMs + ENVELOPE.decayMs + 1), 0);
});

test("hold is capped at maxHoldAheadMs past now", () => {
  const env = createMouthEnvelope();
  env.energize("x".repeat(5000), 0);
  assert.equal(env.level(ENVELOPE.maxHoldAheadMs + ENVELOPE.decayMs + 1), 0);
});

test("after end() the level reaches 0 within decayMs, and the frame closes", () => {
  const env = createMouthEnvelope();
  env.energize("A long sentence with plenty of vowels in it, really.", 0);
  env.end(50);
  assert.ok(ENVELOPE.decayMs >= 300 && ENVELOPE.decayMs <= 600);
  assert.ok(env.level(50 + ENVELOPE.decayMs / 2) > 0, "still decaying halfway");
  assert.equal(env.level(50 + ENVELOPE.decayMs), 0);
  for (let t = 50 + ENVELOPE.decayMs; t < 3000; t += 7) {
    assert.equal(env.frame(t), 0, `frame at ${t}`);
  }
});

test("time going backwards does not change the level", () => {
  const env = createMouthEnvelope();
  env.energize("hello", 1000);
  env.end(1000);
  const a = env.level(1200);
  assert.equal(env.level(1100), a);
});

test("reset closes immediately", () => {
  const env = createMouthEnvelope();
  env.energize("hello there", 0);
  env.reset();
  assert.equal(env.level(1), 0);
  assert.equal(env.frame(1), 0);
});

test("levelToFrame maps to 0..3 and clamps", () => {
  const [t1, t2, t3] = ENVELOPE.thresholds;
  assert.equal(levelToFrame(-5), 0);
  assert.equal(levelToFrame(0), 0);
  assert.equal(levelToFrame(t1), 1);
  assert.equal(levelToFrame(t2), 2);
  assert.equal(levelToFrame(t3), 3);
  assert.equal(levelToFrame(99), 3);
  assert.equal(levelToFrame(NaN), 0);
  assert.equal(levelToFrame("x"), 0);
});

test("flap stays in 0..1 and closes at syllable boundaries", () => {
  for (let t = 0; t < 5000; t += 3) {
    const f = flapAt(t, Math.random());
    assert.ok(f >= 0 && f <= 1, `flap ${f} at ${t}`);
  }
  assert.ok(flapAt(ENVELOPE.syllableMs * 7, 1) < 1e-9);
});

test("while streaming the mouth flaps: frames vary and include closed and open", () => {
  const env = createMouthEnvelope();
  const seen = new Set();
  // Text arrives every 200 ms for 3 s; sample at about 60 fps.
  let nextChunk = 0;
  for (let t = 0; t < 3000; t += 16) {
    if (t >= nextChunk) {
      env.energize("Hello there, how are you today? ", t);
      nextChunk += 200;
    }
    const f = env.frame(t);
    assert.ok(Number.isInteger(f) && f >= 0 && f <= 3, `frame ${f}`);
    seen.add(f);
  }
  assert.ok(seen.has(0), "closes between syllables");
  assert.ok(seen.size >= 3, `only frames ${[...seen]}`);
});

test("appendedText returns the new suffix of cumulative text", () => {
  assert.equal(appendedText("", "Hi"), "Hi");
  assert.equal(appendedText("Hi", "Hi there"), " there");
  assert.equal(appendedText("Hi there", "Hi there"), "");
  assert.equal(appendedText("Hello", "Bye"), "Bye");
  assert.equal(appendedText(undefined, "x"), "x");
});

test("countVowels and hash01 helpers", () => {
  assert.deepEqual(countVowels("banana split"), { vowels: 4, open: 3 });
  for (let i = -5; i < 200; i++) {
    const h = hash01(i);
    assert.ok(h >= 0 && h < 1);
    assert.equal(h, hash01(i));
  }
});
