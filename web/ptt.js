// Push-to-talk input rules. Pure: no DOM access beyond reading the event
// fields passed in, no timers.
//
// Hold Space anywhere except in a text field (Space types normally there),
// or hold the "Hold to talk" button. holdSource is what started the current
// hold: "key", "pointer", or null. Releasing the other source does not end
// it. A hidden tab ends any hold at once; a window blur ends it only if the
// window is still unfocused BLUR_GRACE_MS later (a mic-sharing indicator
// may take focus for a moment when the mic opens).

const TEXT_INPUT_TYPES = new Set([
  "text", "search", "url", "tel", "email", "password", "number",
  "date", "datetime-local", "month", "time", "week",
]);

export function isSpace(event) {
  return Boolean(event) && (event.code === "Space" || event.key === " " || event.key === "Spacebar");
}

// True for elements where Space should type or act on its own.
export function isTextEntry(target) {
  if (!target || typeof target !== "object") return false;
  if (target.isContentEditable === true) return true;
  const tag = String(target.tagName || "").toUpperCase();
  if (tag === "TEXTAREA" || tag === "SELECT") return true;
  if (tag !== "INPUT") return false;
  return TEXT_INPUT_TYPES.has(String(target.type || "text").toLowerCase());
}

function hasModifier(event) {
  return Boolean(event.ctrlKey || event.altKey || event.metaKey);
}

// keydown. Returns:
//   "press"    a fresh Space press (caller also calls preventDefault). The
//              caller ignores it while a hold is on: then the last keyup
//              went to another window during a short blur, the hold goes
//              on, and the coming keyup ends it.
//   "swallow"  a key repeat; call preventDefault so the page does not
//              scroll or click a button
//   "none"     not ours, leave the event alone
// The current hold does not matter here, so a stale hold can never turn a
// fresh press into a swallowed one.
export function keyDownAction(event) {
  if (!isSpace(event) || hasModifier(event) || isTextEntry(event.target)) return "none";
  if (event.repeat) return "swallow";
  return "press";
}

// keyup. "release" ends a key hold wherever focus is now; "swallow" stops
// Space from clicking a focused button; "none" leaves it alone.
export function keyUpAction(event, holdSource) {
  if (!isSpace(event)) return "none";
  if (holdSource === "key") return "release";
  if (hasModifier(event) || isTextEntry(event.target)) return "none";
  return "swallow";
}

// pointerup, pointercancel, pointerleave on the talk button.
export function pointerEndAction(holdSource) {
  return holdSource === "pointer" ? "release" : "none";
}

// The tab became hidden, or the window stayed unfocused after a blur:
// keyup and pointerup may never arrive, so any hold ends.
export function focusLossAction(holdSource, recording) {
  return holdSource !== null || recording ? "release" : "none";
}

export const BLUR_GRACE_MS = 250;

// BLUR_GRACE_MS after a window blur. hasFocus is document.hasFocus() then:
// focus came back, so the hold goes on ("none"); otherwise as focusLossAction.
export function blurCheckAction(hasFocus, holdSource, recording) {
  if (hasFocus) return "none";
  return focusLossAction(holdSource, recording);
}

export const HINTS = Object.freeze({
  help: "Hold Space to talk",
  helpInInput: "Esc, then hold Space to talk",
  opening: "Opening mic...",
  listening: "Listening...",
  busy: "Busy",
  tooShort: "Keep holding while you talk",
  micNotReady: "Microphone not ready. Check the browser prompt, then hold again.",
});

// Let go before the mic opened. A quick tap gets "keep holding"; a long
// wait means the mic is slow or a permission prompt is still open.
export const SLOW_MIC_MS = 1000;

export function earlyReleaseHint(heldMs) {
  return heldMs >= SLOW_MIC_MS ? HINTS.micNotReady : HINTS.tooShort;
}

// The line under the input. mode: "opening" (held, not capturing yet),
// "recording" (lit bar: talk now), "help", "flash" (brief notice), or "off"
// (empty). live: the worklet has audio and the server sent listening; only
// then does "Listening..." show, so the first word is not cut off.
// flash is a short message to show instead of the help, or null.
export function hintFor({ connected, recording, released, live = false, serverBusy, inputFocused, flash = null }) {
  if (recording && !released) {
    return live ? { mode: "recording", text: HINTS.listening } : { mode: "opening", text: HINTS.opening };
  }
  if (flash) return { mode: "flash", text: flash };
  if (!connected || recording || serverBusy) return { mode: "off", text: "" };
  return { mode: "help", text: inputFocused ? HINTS.helpInInput : HINTS.help };
}
