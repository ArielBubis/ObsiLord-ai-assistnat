import { test } from "node:test";
import assert from "node:assert/strict";

import {
  busyAfterMessage, stateAction, drainTimeoutMs, textDrivesMouth, closesOpenSentence, DRAIN_GRACE_MS,
  WATCHDOG_MS, WATCHDOG_MESSAGE, pressAction, recordingAction, controlsDisabled, orphanAction,
  ACK_TIMEOUT_MS, BUSY_FLASH_MS, ORPHAN_HOLD_MS,
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

// Push-to-talk (Phase 4).

const err = (message) => ({ type: "error", message });
const userFinal = { type: "transcript", role: "user", text: "What is the capital of France?", final: true };

test("press: start when idle, barge in while only playback drains, busy while the server works", () => {
  const base = { connected: true, busy: false, pendingIdle: false, recording: false };
  assert.equal(pressAction(base), "start");
  assert.equal(pressAction({ ...base, busy: true, pendingIdle: true }), "barge-in");
  assert.equal(pressAction({ ...base, busy: true }), "busy");
  assert.equal(pressAction({ ...base, connected: false }), "offline");
  assert.equal(pressAction({ ...base, recording: true }), "ignore");
  // An open recording wins over everything (still flushing after release).
  assert.equal(pressAction({ ...base, recording: true, busy: true, connected: false }), "ignore");
});

test("before the server answers audio_start: listening accepts, an error or other state refuses", () => {
  assert.equal(recordingAction("pending", state("listening")), "accept");
  assert.equal(recordingAction("pending", err("Voice input is not available.")), "refuse");
  assert.equal(recordingAction("pending", err("Busy, wait for the current reply.")), "refuse");
  assert.equal(recordingAction("pending", err("No API key configured.")), "refuse");
  // Any error before listening is a refusal, whatever its text.
  assert.equal(recordingAction("pending", err("Something new.")), "refuse");
  assert.equal(recordingAction("pending", { type: "error" }), "refuse");
  assert.equal(recordingAction("pending", state("idle")), "refuse");
  assert.equal(recordingAction("pending", state("thinking")), "refuse");
  assert.equal(recordingAction("pending", delta), "none");
  assert.equal(recordingAction("pending", { type: "tts_start", sentence_id: 1, sample_rate: 22050 }), "none");
});

test("after listening: an error ends the recording locally (audio_end follows)", () => {
  for (const message of ["Recording too long, using the first 30 s.", "Unexpected audio frame.",
    "No recording in progress."]) {
    assert.equal(recordingAction("accepted", err(message)), "end", message);
  }
});

test("after listening: listening again is nothing, another state means the server ended it", () => {
  assert.equal(recordingAction("accepted", state("listening")), "none");
  assert.equal(recordingAction("accepted", state("idle")), "abort");
  assert.equal(recordingAction("accepted", state("thinking")), "abort");
  assert.equal(recordingAction("accepted", userFinal), "none");
});

test("mic still opening (audio_start not sent): server messages do not touch the recording", () => {
  for (const msg of [state("listening"), state("idle"), err("x"), delta]) {
    assert.equal(recordingAction("none", msg), "none");
  }
});

test("malformed messages do not touch the recording", () => {
  for (const ack of ["none", "pending", "accepted"]) {
    for (const msg of [null, undefined, "state", 5]) assert.equal(recordingAction(ack, msg), "none");
  }
});

test("text input and Send are off while busy, recording, or an orphan waits", () => {
  assert.equal(controlsDisabled(false, false), false);
  assert.equal(controlsDisabled(false, false, false), false);
  assert.equal(controlsDisabled(true, false), true);
  assert.equal(controlsDisabled(false, true), true);
  assert.equal(controlsDisabled(true, true), true);
  assert.equal(controlsDisabled(false, false, true), true);
});

// Busy tracking over whole voice turns. busy turns true when audio_end is
// sent; the face shows listening (set locally on the press) until the server
// moves it.
function runTurn(messages, face = "listening") {
  let busy = true;
  const trace = [];
  for (const msg of messages) {
    busy = busyAfterMessage(busy, msg, face);
    if (msg.type === "state") face = msg.state;
    trace.push(busy);
  }
  return trace;
}

test("voice turn, too short: Didn't catch that, then idle ends it", () => {
  assert.deepEqual(runTurn([err("Didn't catch that."), state("idle")]), [true, false]);
});

test("voice turn, STT failure after thinking: error keeps it, idle ends it", () => {
  assert.deepEqual(runTurn([state("thinking"), err("Speech recognition failed."), state("idle")]), [true, true, false]);
  assert.deepEqual(runTurn([state("thinking"), err("Didn't catch that."), state("idle")]), [true, true, false]);
});

test("voice turn, valid utterance: user transcript then the Phase 3 sequence", () => {
  const msgs = [state("thinking"), userFinal, delta, { type: "tts_start", sentence_id: 1, sample_rate: 22050 },
    state("speaking"), final, { type: "tts_end", sentence_id: 1 }, state("idle")];
  assert.deepEqual(runTurn(msgs), [true, true, true, true, true, true, true, false]);
});

test("voice turn, cap reached: the cap error then idle", () => {
  assert.deepEqual(runTurn([state("thinking"), userFinal, err("Daily token cap reached."), state("idle")]),
    [true, true, true, false]);
});

test("recording timers", () => {
  assert.ok(ACK_TIMEOUT_MS >= 1000 && ACK_TIMEOUT_MS < WATCHDOG_MS);
  assert.ok(ORPHAN_HOLD_MS >= 1000 && ORPHAN_HOLD_MS < WATCHDOG_MS);
  assert.ok(BUSY_FLASH_MS >= 500 && BUSY_FLASH_MS <= 3000);
});

test("press while an abandoned recording waits for the server: Busy", () => {
  const base = { connected: true, busy: false, pendingIdle: false, recording: false };
  assert.equal(pressAction({ ...base, orphan: true }), "busy");
  assert.equal(pressAction({ ...base, orphan: false }), "start");
  assert.equal(pressAction({ ...base, orphan: true, connected: false }), "offline");
  assert.equal(pressAction({ ...base, orphan: true, recording: true }), "ignore");
});

test("orphan: a late listening sends audio_end only if none went out yet", () => {
  assert.equal(orphanAction(false, state("listening")), "send-end");
  assert.equal(orphanAction(true, state("listening")), "swallow");
});

test("orphan: state idle clears it; errors and other messages pass through", () => {
  assert.equal(orphanAction(true, state("idle")), "clear");
  assert.equal(orphanAction(false, state("idle")), "clear");
  for (const msg of [err("Didn't catch that."), err("No recording in progress."), state("thinking"),
    delta, userFinal, null, "x"]) {
    assert.equal(orphanAction(true, msg), "none", JSON.stringify(msg));
  }
});

test("ack timeout then the slow server's answers: listening swallowed, error shown, idle clears", () => {
  // The client sent audio_start, then audio_end at the timeout. The server
  // reads in order: listening for audio_start, then its answer to audio_end.
  let endSent = true;
  let open = true;
  const seen = [];
  for (const msg of [state("listening"), err("Didn't catch that."), state("idle")]) {
    if (!open) break;
    const a = orphanAction(endSent, msg);
    if (a === "send-end") endSent = true;
    if (a === "clear") open = false;
    seen.push(a);
  }
  assert.deepEqual(seen, ["swallow", "none", "clear"]);
});
