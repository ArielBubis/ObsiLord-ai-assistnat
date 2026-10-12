import { test } from "node:test";
import assert from "node:assert/strict";

import {
  isSpace, isTextEntry, keyDownAction, keyUpAction, pointerEndAction, focusLossAction, blurCheckAction,
  hintFor, earlyReleaseHint, HINTS, BLUR_GRACE_MS, SLOW_MIC_MS,
} from "../../web/ptt.js";

const body = { tagName: "BODY" };
const textInput = { tagName: "INPUT", type: "text" };
const button = { tagName: "BUTTON" };
const range = { tagName: "INPUT", type: "range" };

function key(target = body, extra = {}) {
  return { code: "Space", key: " ", repeat: false, target, ...extra };
}

test("Space is recognized by code or key", () => {
  assert.equal(isSpace({ code: "Space" }), true);
  assert.equal(isSpace({ key: " " }), true);
  assert.equal(isSpace({ key: "Spacebar" }), true);
  assert.equal(isSpace({ code: "Enter", key: "Enter" }), false);
  assert.equal(isSpace(null), false);
});

test("text fields, textareas, selects, and contenteditable count as text entry", () => {
  for (const t of [textInput, { tagName: "input" }, { tagName: "INPUT", type: "search" },
    { tagName: "INPUT", type: "EMAIL" }, { tagName: "TEXTAREA" }, { tagName: "SELECT" },
    { tagName: "DIV", isContentEditable: true }]) {
    assert.equal(isTextEntry(t), true, JSON.stringify(t));
  }
  for (const t of [body, button, range, { tagName: "INPUT", type: "checkbox" }, null, undefined, "INPUT"]) {
    assert.equal(isTextEntry(t), false, JSON.stringify(t));
  }
});

test("Space outside a text field starts talking", () => {
  assert.equal(keyDownAction(key()), "press");
  assert.equal(keyDownAction(key(button)), "press");
  assert.equal(keyDownAction(key(range)), "press");
});

test("key repeat is ignored but swallowed (no scrolling)", () => {
  assert.equal(keyDownAction(key(body, { repeat: true })), "swallow");
  assert.equal(keyDownAction(key(button, { repeat: true })), "swallow");
});

test("Space in the text input types normally", () => {
  assert.equal(keyDownAction(key(textInput)), "none");
  assert.equal(keyDownAction(key(textInput, { repeat: true })), "none");
  assert.equal(keyUpAction(key(textInput), null), "none");
});

test("modified Space and other keys are left alone", () => {
  for (const mod of ["ctrlKey", "altKey", "metaKey"]) {
    assert.equal(keyDownAction(key(body, { [mod]: true })), "none");
  }
  assert.equal(keyDownAction({ code: "KeyA", key: "a", target: body }), "none");
  assert.equal(keyUpAction({ code: "KeyA", key: "a", target: body }, "key"), "none");
});

test("only genuine key repeats are swallowed; a fresh keydown is always a press", () => {
  // The caller ignores a press while a hold is on (see replay below), so a
  // stale hold cannot swallow the next real press.
  assert.equal(keyDownAction(key(body, { repeat: false })), "press");
  assert.equal(keyDownAction(key(body, { repeat: true })), "swallow");
});

test("keyup ends a key hold wherever focus is", () => {
  assert.equal(keyUpAction(key(), "key"), "release");
  assert.equal(keyUpAction(key(textInput), "key"), "release");
  assert.equal(keyUpAction(key(body, { ctrlKey: true }), "key"), "release");
});

test("keyup without a key hold is swallowed outside text fields (no button click)", () => {
  assert.equal(keyUpAction(key(button), null), "swallow");
  assert.equal(keyUpAction(key(), "pointer"), "swallow");
});

test("pointer end releases only a pointer hold", () => {
  assert.equal(pointerEndAction("pointer"), "release");
  assert.equal(pointerEndAction("key"), "none");
  assert.equal(pointerEndAction(null), "none");
});

test("a hidden tab (or a blur that lasts) ends any hold or open recording", () => {
  assert.equal(focusLossAction("key", true), "release");
  assert.equal(focusLossAction("pointer", true), "release");
  assert.equal(focusLossAction("key", false), "release");
  // 30 s auto-stop already cleared the hold, recording still flushing.
  assert.equal(focusLossAction(null, true), "release");
  assert.equal(focusLossAction(null, false), "none");
});

test("hint line: listening, flash, help, or nothing", () => {
  const base = { connected: true, recording: false, released: false, serverBusy: false, inputFocused: false };
  assert.deepEqual(hintFor(base), { mode: "help", text: HINTS.help });
  assert.deepEqual(hintFor({ ...base, inputFocused: true }), { mode: "help", text: HINTS.helpInInput });
  // Held: "Opening mic..." until audio flows and listening came, then talk.
  assert.deepEqual(hintFor({ ...base, recording: true }), { mode: "opening", text: "Opening mic..." });
  assert.deepEqual(hintFor({ ...base, recording: true, live: true }), { mode: "recording", text: "Listening..." });
  // Released, waiting for the last frame or the server's answer.
  assert.deepEqual(hintFor({ ...base, recording: true, released: true }), { mode: "off", text: "" });
  assert.deepEqual(hintFor({ ...base, serverBusy: true }), { mode: "off", text: "" });
  assert.deepEqual(hintFor({ ...base, connected: false }), { mode: "off", text: "" });
  assert.deepEqual(hintFor({ ...base, serverBusy: true, flash: "Busy" }), { mode: "flash", text: "Busy" });
  // Holding wins over a flash.
  assert.equal(hintFor({ ...base, recording: true, live: true, flash: "Busy" }).mode, "recording");
  assert.equal(hintFor({ ...base, recording: true, flash: "Busy" }).mode, "opening");
  assert.equal(HINTS.help, "Hold Space to talk");
  assert.equal(HINTS.busy, "Busy");
});

test("blur: the hold ends only if the window is still unfocused after the grace delay", () => {
  assert.equal(BLUR_GRACE_MS, 250);
  // Focus came back (a mic indicator took it for a moment): keep recording.
  assert.equal(blurCheckAction(true, "key", true), "none");
  assert.equal(blurCheckAction(true, "pointer", true), "none");
  // Still unfocused: release.
  assert.equal(blurCheckAction(false, "key", true), "release");
  assert.equal(blurCheckAction(false, "pointer", true), "release");
  assert.equal(blurCheckAction(false, null, true), "release");
  // Nothing held by then (released meanwhile): nothing to do.
  assert.equal(blurCheckAction(false, null, false), "none");
});

// Replays events through the rules the way ws.js does: a press while a
// hold is on is ignored, a press otherwise starts a key hold, release
// clears it, and "ended" (refusal, abort, ack timeout, mic error, release
// before the mic opened) clears it too. Returns the actions taken.
function replay(events, hold = null) {
  const out = [];
  for (const e of events) {
    let a;
    if (e === "down") a = keyDownAction(key());
    else if (e === "repeat") a = keyDownAction(key(body, { repeat: true }));
    else if (e === "up") a = keyUpAction(key(), hold);
    else if (e === "pointer-up") a = pointerEndAction(hold);
    else if (e === "blur-lasting") a = blurCheckAction(false, hold, hold !== null);
    else if (e === "blur-brief") a = blurCheckAction(true, hold, hold !== null);
    else if (e === "ended") a = "ended";
    if (a === "press" && hold === null) hold = "key";
    if (a === "release" || a === "ended") hold = null;
    out.push(a);
  }
  return { out, hold };
}

test("brief blur, keyup lost to another window: the next press and release end the hold", () => {
  // Hold Space, focus leaves for a moment and Space is let go there (the
  // keyup never arrives). Focus is back, the hold is still on.
  const { out, hold } = replay(["down", "repeat", "blur-brief"]);
  assert.deepEqual(out, ["press", "swallow", "none"]);
  assert.equal(hold, "key");
  // The user presses and lets go again: that press is ignored (hold on),
  // the keyup ends the hold, and the pair after that works normally.
  const again = replay(["down"], hold);
  assert.equal(again.hold, "key", "press ignored while holding");
  const next = replay(["up", "down", "up"], again.hold);
  assert.deepEqual(next.out, ["release", "press", "release"]);
  assert.equal(next.hold, null);
});

test("lasting blur ends the hold; Space still down on return is not a new press", () => {
  const { out, hold } = replay(["down", "blur-lasting"]);
  assert.deepEqual(out, ["press", "release"]);
  assert.equal(hold, null);
  // Back in the window with Space still held: repeats are swallowed, the
  // keyup is swallowed, then a fresh press works.
  const next = replay(["repeat", "repeat", "up", "down", "up"], hold);
  assert.deepEqual(next.out, ["swallow", "swallow", "swallow", "press", "release"]);
});

test("a recording that ends while the key is held (refusal, timeout, mic error) frees the next press", () => {
  // Keyup lost during a short blur, then the server refuses: the hold must
  // not stay "key", or the next press would be ignored.
  const { hold } = replay(["down", "blur-brief", "ended"]);
  assert.equal(hold, null);
  const next = replay(["down"], hold);
  assert.deepEqual(next.out, ["press"]);
  assert.equal(next.hold, "key");
  // Space still physically held after a refusal: repeats and the keyup are
  // swallowed, then a fresh press works.
  const held = replay(["down", "repeat", "ended", "repeat", "up", "down"]);
  assert.deepEqual(held.out, ["press", "swallow", "ended", "swallow", "swallow", "press"]);
  assert.equal(held.hold, "key");
});

test("after a recording ends, the Hold to talk button is not blocked by a stale key hold", () => {
  const { hold } = replay(["down", "ended"]);
  assert.equal(hold, null);
  assert.equal(pointerEndAction(hold), "none");
});

test("let go before the mic opened: quick tap vs slow mic or open prompt", () => {
  assert.equal(SLOW_MIC_MS, 1000);
  assert.equal(earlyReleaseHint(0), HINTS.tooShort);
  assert.equal(earlyReleaseHint(999), "Keep holding while you talk");
  assert.equal(earlyReleaseHint(1000), HINTS.micNotReady);
  assert.equal(earlyReleaseHint(30000), "Microphone not ready. Check the browser prompt, then hold again.");
});
