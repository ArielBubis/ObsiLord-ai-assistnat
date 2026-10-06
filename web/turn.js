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
