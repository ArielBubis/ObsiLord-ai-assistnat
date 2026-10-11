// Turn bookkeeping for the chat input. Pure: no DOM, no timers.
// A turn is in progress (input disabled) from send until the server is done.

// If a turn is in progress and the server sends nothing for this long, the
// client gives up and re-enables input. The server has its own 60 s timeout.
export const WATCHDOG_MS = 75000;
export const WATCHDOG_MESSAGE = "No reply from server.";

// Whether a turn is still in progress after a server message.
// faceState is the face state before the message is applied.
// - state idle ends the turn; other states keep it going.
// - error ends the turn only when the face is already idle (for example no
//   API key, sent before any state change). Errors raised mid-turn are always
//   followed by state idle, which ends the turn then.
export function busyAfterMessage(busy, msg, faceState) {
  if (!busy || !msg || typeof msg !== "object") return busy;
  if (msg.type === "state") return msg.state !== "idle";
  if (msg.type === "error") return faceState !== "idle";
  return busy;
}

// The server sends state idle as soon as it has sent the last audio; it does
// not wait for playback. What to do with a state message given whether audio
// is still queued or playing:
//   "defer": keep the face and the input as they are until the queue drains
//   "apply": apply the state now
export function stateAction(msg, audioPending) {
  if (!msg || typeof msg !== "object" || msg.type !== "state") return "apply";
  return msg.state === "idle" && audioPending ? "defer" : "apply";
}

// Safety net for a deferred idle: if the drain event never comes (for
// example the audio context stayed suspended), give up this long after the
// queued audio should have finished.
export const DRAIN_GRACE_MS = 3000;

export function drainTimeoutMs(remainingMs) {
  const r = Number.isFinite(remainingMs) ? Math.max(0, remainingMs) : 0;
  return r + DRAIN_GRACE_MS;
}

// The fake text envelope is only for turns without audio (voice disabled on
// the server). Text that arrives before state speaking (still thinking) does
// not move the mouth.
export function textDrivesMouth(faceState, turnHasAudio) {
  return faceState === "speaking" && !turnHasAudio;
}

// The server cancels TTS on error paths without sending tts_end. State idle
// or an error closes any sentence still open, so its tail fades out at its
// scheduled end instead of stopping with a click.
export function closesOpenSentence(msg) {
  if (!msg || typeof msg !== "object") return false;
  return msg.type === "error" || (msg.type === "state" && msg.state === "idle");
}
