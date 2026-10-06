import { test } from "node:test";
import assert from "node:assert/strict";

import { busyAfterMessage, WATCHDOG_MS, WATCHDOG_MESSAGE } from "../../web/turn.js";

const state = (s) => ({ type: "state", state: s });
const error = { type: "error", message: "x" };
const delta = { type: "transcript", role: "assistant", text: "Hi", final: false };
const final = { type: "transcript", role: "assistant", text: "Hi", final: true };

test("not busy stays not busy whatever arrives", () => {
  for (const msg of [state("thinking"), state("idle"), error, delta, null]) {
    assert.equal(busyAfterMessage(false, msg, "idle"), false);
  }
});

test("state idle ends the turn, other states keep it", () => {
  assert.equal(busyAfterMessage(true, state("idle"), "speaking"), false);
  assert.equal(busyAfterMessage(true, state("thinking"), "idle"), true);
  assert.equal(busyAfterMessage(true, state("speaking"), "thinking"), true);
  assert.equal(busyAfterMessage(true, state("looking"), "thinking"), true);
});

test("transcripts never end the turn, even final", () => {
  assert.equal(busyAfterMessage(true, delta, "speaking"), true);
  assert.equal(busyAfterMessage(true, final, "speaking"), true);
});

test("error mid-turn keeps the turn until state idle follows", () => {
  assert.equal(busyAfterMessage(true, error, "thinking"), true);
  assert.equal(busyAfterMessage(true, error, "speaking"), true);
  assert.equal(busyAfterMessage(true, state("idle"), "thinking"), false);
});

test("error while the face is idle (no key, busy, cap) ends the turn", () => {
  assert.equal(busyAfterMessage(true, error, "idle"), false);
});

test("unknown or malformed messages leave busy unchanged", () => {
  assert.equal(busyAfterMessage(true, { type: "tts_start" }, "speaking"), true);
  assert.equal(busyAfterMessage(true, null, "speaking"), true);
  assert.equal(busyAfterMessage(true, "state", "speaking"), true);
});

test("watchdog is longer than the server's 60 s turn timeout", () => {
  assert.equal(WATCHDOG_MS, 75000);
  assert.equal(WATCHDOG_MESSAGE, "No reply from server.");
});
