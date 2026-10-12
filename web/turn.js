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

// Push-to-talk (Phase 4). A recording runs from the press until audio_end is
// sent. It is not a turn yet: busy stays false and the watchdog stays off
// (the server sends nothing but listening meanwhile), but the text input is
// disabled. Sending audio_end starts the turn like a text send.

// The client stops recording on its own after max_utterance_s from
// GET /config/audio (config.js, default 30 s).

// The server answers audio_start with state listening or an error at once.
// Without an answer by then the recording is dropped locally and audio_end
// is sent, so a server that was only slow closes its recording too (it
// answers "Didn't catch that." and idle).
export const ACK_TIMEOUT_MS = 5000;
export const BUSY_FLASH_MS = 1200;

// What a push-to-talk press does:
//   "start"     start recording
//   "barge-in"  the server is done and only local playback is draining: stop
//               it, apply idle, then start recording
//   "busy"      the server turn is still running, or the server has not
//               answered for an abandoned recording yet: ignore, flash "Busy"
//   "offline"   no connection
//   "ignore"    a recording is already open (held, or released and waiting
//               for the server or the last frame)
export function pressAction({ connected, busy, pendingIdle, recording, orphan = false }) {
  if (recording) return "ignore";
  if (!connected) return "offline";
  if (orphan) return "busy";
  if (busy) return pendingIdle ? "barge-in" : "busy";
  return "start";
}

// A server message while a recording is open (audio_start sent, audio_end
// not yet). ack is "none" (mic still opening, audio_start not sent),
// "pending" (no answer to audio_start yet), or "accepted" (listening came).
//   "accept"  listening: frames may flow
//   "refuse"  any error, or any other state, before listening (voice
//             disabled, busy, no API key): stop capturing and release the
//             mic, drop the frames, no audio_end, back to idle locally
//   "end"     error after listening (too long, protocol slip): stop
//             capturing and send audio_end so the server can finish
//   "abort"   a state other than listening after listening: the server
//             ended the recording itself; stop capturing, no audio_end
//   "none"    nothing to do
export function recordingAction(ack, msg) {
  if (!msg || typeof msg !== "object") return "none";
  if (ack === "pending") {
    if (msg.type === "state") return msg.state === "listening" ? "accept" : "refuse";
    if (msg.type === "error") return "refuse";
    return "none";
  }
  if (ack === "accepted") {
    if (msg.type === "error") return "end";
    if (msg.type === "state" && msg.state !== "listening") return "abort";
  }
  return "none";
}

// Text input and Send are off during a turn, while recording, and while
// an orphan recording waits for the server.
export function controlsDisabled(busy, recording, orphan = false) {
  return Boolean(busy || recording || orphan);
}

// A recording dropped locally while its audio_start was unanswered (ack
// timeout) is an orphan until the server answers for it. The server reads
// messages in order, so its next messages belong to the orphan: a late
// listening, then "Didn't catch that." and idle for the audio_end. While an
// orphan is open, presses and typed sends wait (input off, "Busy"), so
// those messages are not taken for a new recording or turn. It is
// forgotten on state idle, on disconnect, or ORPHAN_HOLD_MS after the
// timeout if the server never answers.
export const ORPHAN_HOLD_MS = 5000;

// endSent: audio_end already went out for the orphan. What a server message
// means for it:
//   "send-end"  late listening, no audio_end yet: send it now; do not show
//               listening
//   "swallow"   late listening, audio_end already sent: the server closes
//               the recording with that one (a second audio_end would only
//               draw "No recording in progress."); do not show listening
//   "clear"     state idle: the server is done; forget the orphan (the
//               message is then handled as usual)
//   "none"      anything else, handled as usual (errors are shown)
export function orphanAction(endSent, msg) {
  if (!msg || typeof msg !== "object" || msg.type !== "state") return "none";
  if (msg.state === "listening") return endSent ? "swallow" : "send-end";
  if (msg.state === "idle") return "clear";
  return "none";
}
