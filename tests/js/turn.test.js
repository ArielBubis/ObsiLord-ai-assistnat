import { test } from "node:test";
import assert from "node:assert/strict";

import {
  busyAfterMessage, stateAction, drainTimeoutMs, textDrivesMouth, closesOpenSentence, DRAIN_GRACE_MS,
  WATCHDOG_MS, WATCHDOG_MESSAGE,
} from "../../web/turn.js";

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

test("state idle is deferred while audio is queued or playing", () => {
  assert.equal(stateAction(state("idle"), true), "defer");
  assert.equal(stateAction(state("idle"), false), "apply");
});

test("other messages are never deferred", () => {
  for (const msg of [state("thinking"), state("speaking"), state("looking"), error, delta, final,
    { type: "tts_start", sentence_id: 1, sample_rate: 22050 }, null]) {
    assert.equal(stateAction(msg, true), "apply");
    assert.equal(stateAction(msg, false), "apply");
  }
});

test("drain safety timeout is the queued audio plus a grace period", () => {
  assert.equal(drainTimeoutMs(0), DRAIN_GRACE_MS);
  assert.equal(drainTimeoutMs(1500), 1500 + DRAIN_GRACE_MS);
  assert.equal(drainTimeoutMs(-5), DRAIN_GRACE_MS);
  assert.equal(drainTimeoutMs(NaN), DRAIN_GRACE_MS);
  assert.ok(DRAIN_GRACE_MS >= 1000 && DRAIN_GRACE_MS < WATCHDOG_MS);
});

test("the text envelope moves the mouth only in speaking turns without audio", () => {
  assert.equal(textDrivesMouth("speaking", false), true);
  assert.equal(textDrivesMouth("speaking", true), false);
  assert.equal(textDrivesMouth("thinking", false), false);
  assert.equal(textDrivesMouth("idle", false), false);
});

test("state idle and errors close an open sentence, other messages do not", () => {
  assert.equal(closesOpenSentence(state("idle")), true);
  assert.equal(closesOpenSentence(error), true);
  assert.equal(closesOpenSentence({ type: "error", message: "Voice output failed." }), true);
  for (const msg of [state("thinking"), state("speaking"), delta, final,
    { type: "tts_start", sentence_id: 2, sample_rate: 22050 }, { type: "tts_end", sentence_id: 1 }, null, "x"]) {
    assert.equal(closesOpenSentence(msg), false);
  }
});
