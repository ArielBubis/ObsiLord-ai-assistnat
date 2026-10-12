// WebSocket client, text chat UI, and push-to-talk. Shows connection status,
// sends typed messages, shows the streamed assistant reply as a subtitle,
// applies server state events to the face, and plays TTS audio (binary
// frames) through audio.js. Turns without audio feed new reply text to the
// fake mouth envelope instead. Reconnects with a short backoff (capped at
// 2 s) after a close.
//
// The server sends state idle once it has sent the last audio, without
// waiting for playback. If audio is still queued then, the face stays as it
// is and the input stays disabled until the queue drains.
//
// Push-to-talk: hold Space (outside the text box) or the "Hold to talk"
// button. The mic opens on the press (audio_in.js), then audio_start goes
// out. Frames are held back until the server answers with state listening,
// so a refused start never sends audio. The hint says "Opening mic..." until
// audio flows and listening came, then "Listening..." (talk now). Release,
// a window blur that lasts (BLUR_GRACE_MS), a hidden tab, or
// max_utterance_s ends the recording: the last partial frame is flushed,
// then audio_end goes out (only after listening came) and the turn runs
// like a text turn. The mic tracks are stopped after every recording.

import {
  setState, getState, mouthText, mouthEnd, mouthReset, setAudioLevelSource, setAudioMouth,
} from "./face.js";
import { appendedText } from "./mouth.js";
import { createPlayer } from "./audio.js";
import { createMic, micErrorMessage } from "./audio_in.js";
import { loadAudioSettings, MAX_UTTERANCE_DEFAULT_S } from "./config.js";
import {
  busyAfterMessage, stateAction, drainTimeoutMs, textDrivesMouth, closesOpenSentence,
  pressAction, recordingAction, controlsDisabled, orphanAction,
  WATCHDOG_MS, WATCHDOG_MESSAGE, ACK_TIMEOUT_MS, BUSY_FLASH_MS, ORPHAN_HOLD_MS,
} from "./turn.js";
import {
  keyDownAction, keyUpAction, pointerEndAction, focusLossAction, blurCheckAction, hintFor,
  earlyReleaseHint, HINTS, BLUR_GRACE_MS,
} from "./ptt.js";

const MAX_TEXT = 2000;
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 2000;
const NOT_CONNECTED = "Not connected";

const statusEl = document.getElementById("status");
const form = document.getElementById("chat");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("chat-send");
const talkBtn = document.getElementById("chat-talk");
const hintEl = document.getElementById("ptt-hint");
const hintText = document.getElementById("ptt-text");
const userLine = document.getElementById("user-line");
const subtitle = document.getElementById("subtitle");
const errorLine = document.getElementById("error-line");

let ws = null;
let connected = false;
let reconnectDelay = RECONNECT_MIN_MS;
// A turn is in progress from send (or audio_end) until state idle or an error.
let busy = false;
// Cumulative assistant text seen so far in the current reply.
let replyText = "";
// One-shot timer, re-armed on every server message while a turn is active.
let watchdog = null;
// The server said idle but audio is still playing: idle is applied on drain.
let pendingIdle = false;
let drainTimer = null;
// The current turn received tts_start, so the mouth follows the audio.
let turnHasAudio = false;
// Focus the text box when a turn ends only if the turn was typed, so Space
// keeps working for push-to-talk after a voice turn.
let refocusInput = true;

// The open recording, from the press until audio_end is sent or the
// recording is dropped:
//   handle    from mic.begin(), null while the mic is opening
//   ack       "none" (audio_start not sent yet), "pending", "accepted"
//   released  the hold ended (release, blur, hidden tab, 30 s, error)
//   flushed   the worklet has posted the last frame and the mic is closed
//   audioFlowing  the worklet has seen the first audio
//   queue     frames waiting for listening (sent then, dropped on refusal)
let rec = null;
// A recording dropped on the ack timeout whose audio_start the server has
// not answered yet: { endSent, timer } (see orphanAction in turn.js).
let orphan = null;
// Pending check after a window blur.
let blurTimer = null;
// Client-side recording limit, from max_utterance_s in GET /config/audio.
let maxUtteranceMs = MAX_UTTERANCE_DEFAULT_S * 1000;
// What holds push-to-talk: "key", "pointer", or null.
let holdSource = null;
// Short notice in the hint line ("Busy"), cleared by a timer.
let flash = null;
let flashTimer = null;

// Exported for the debug panel's EQ sliders.
export const player = createPlayer({ onDrained: () => finishPendingIdle() });
setAudioLevelSource(() => player.level());
// EQ and the recording limit from GET /config/audio (defaults if that
// fails). The player applies the EQ now or when its AudioContext is created
// on the first Send or press. Resolves to the EQ (debug panel).
export const audioConfigReady = loadAudioSettings().then(({ eq, maxUtteranceS }) => {
  player.setEq(eq);
  maxUtteranceMs = maxUtteranceS * 1000;
  return eq;
});
// Mic capture in the same AudioContext as playback. Exported for debugging
// (mic.stats has the open and close latency of the last recording).
export const mic = createMic({ getContext: () => player.context() });

function isOpen() {
  return connected && ws !== null && ws.readyState === WebSocket.OPEN;
}

function clearDrainTimer() {
  if (drainTimer !== null) clearTimeout(drainTimer);
  drainTimer = null;
}

function cancelPendingIdle() {
  pendingIdle = false;
  clearDrainTimer();
  updateHint();
}

function deferIdle() {
  pendingIdle = true;
  clearWatchdog();
  clearDrainTimer();
  drainTimer = setTimeout(() => {
    drainTimer = null;
    console.warn("audio: queue did not drain in time, stopping playback");
    player.stopAll();
    finishPendingIdle();
  }, drainTimeoutMs(player.remainingMs()));
  updateHint();
}

function finishPendingIdle() {
  if (!pendingIdle) return;
  cancelPendingIdle();
  setState("idle");
  mouthEnd();
  setBusy(false);
}

function clearWatchdog() {
  if (watchdog !== null) clearTimeout(watchdog);
  watchdog = null;
}

function armWatchdog() {
  clearWatchdog();
  watchdog = setTimeout(() => {
    watchdog = null;
    if (!busy) return;
    showError(WATCHDOG_MESSAGE);
    player.stopAll();
    setState("idle");
    mouthEnd();
    setBusy(false);
  }, WATCHDOG_MS);
}

// Held and not yet released.
function recordingHeld() {
  return rec !== null && !rec.released;
}

// Speech is being captured and the server takes it: time to talk.
function recordingLive() {
  return recordingHeld() && rec.audioFlowing && rec.ack === "accepted";
}

function updateHint() {
  const hint = hintFor({
    connected,
    recording: rec !== null,
    released: rec !== null && rec.released,
    live: recordingLive(),
    serverBusy: busy && !pendingIdle,
    inputFocused: document.activeElement === input,
    flash,
  });
  hintEl.dataset.mode = hint.mode;
  hintText.textContent = hint.text;
}

function updateControls() {
  const off = controlsDisabled(busy, rec !== null, orphan !== null);
  input.disabled = off;
  sendBtn.disabled = off;
  talkBtn.classList.toggle("recording", recordingLive());
  talkBtn.classList.toggle("opening", recordingHeld() && !recordingLive());
  updateHint();
}

function flashHint(text) {
  flash = text;
  if (flashTimer !== null) clearTimeout(flashTimer);
  flashTimer = setTimeout(() => {
    flash = null;
    flashTimer = null;
    updateHint();
  }, BUSY_FLASH_MS);
  updateHint();
}

function clearFlash() {
  flash = null;
  if (flashTimer !== null) clearTimeout(flashTimer);
  flashTimer = null;
}

function setBusy(value) {
  busy = value;
  updateControls();
  if (value) {
    armWatchdog();
  } else {
    clearWatchdog();
    if (refocusInput) input.focus();
  }
}

function showError(message) {
  errorLine.textContent = String(message || "Error");
}

function showSubtitle(text) {
  subtitle.textContent = text;
  // Keep the tail visible when the reply is longer than the visible lines.
  subtitle.scrollTop = subtitle.scrollHeight;
}

// Clears the previous exchange at the start of a new one.
function beginExchange(userText) {
  turnHasAudio = false;
  setAudioMouth(false);
  replyText = "";
  showSubtitle("");
  userLine.textContent = userText;
}

// Push-to-talk ---------------------------------------------------------------

function sendFrame(buf) {
  if (isOpen()) ws.send(buf);
}

function onMicFrame(r, buf) {
  if (rec !== r) return;
  if (r.ack === "accepted") sendFrame(buf);
  else r.queue.push(buf);
}

// Forgets the recording (no audio_end). The mic must already be released or
// be released by the caller.
function endRecording(r) {
  if (r.holdTimer !== null) clearTimeout(r.holdTimer);
  if (r.ackTimer !== null) clearTimeout(r.ackTimer);
  r.holdTimer = null;
  r.ackTimer = null;
  r.queue = [];
  if (rec === r) {
    rec = null;
    // Whatever ended the recording, the hold is over, so a keyup that went
    // to another window cannot block the next press.
    holdSource = null;
  }
  updateControls();
}

function dropRecording(r) {
  if (r.handle) r.handle.abort();
  endRecording(r);
}

// audio_end goes out once the server accepted the recording and the last
// frame is out. From then on this is a turn like a text send.
function maybeSendEnd(r) {
  if (rec !== r || r.ack !== "accepted" || !r.flushed) return;
  for (const buf of r.queue) sendFrame(buf);
  endRecording(r);
  if (!isOpen()) return;
  ws.send(JSON.stringify({ type: "audio_end" }));
  refocusInput = false;
  setBusy(true);
}

async function startRecording() {
  const r = {
    handle: null, ack: "none", released: false, flushed: false, audioFlowing: false, queue: [],
    holdTimer: null, ackTimer: null, t0: performance.now(),
  };
  rec = r;
  errorLine.textContent = "";
  clearFlash();
  beginExchange("");
  // Listening face right away; the hint says "Opening mic..." until audio
  // flows and the server's listening state came.
  setState("listening");
  updateControls();
  r.holdTimer = setTimeout(() => {
    r.holdTimer = null;
    if (rec === r) release();
  }, maxUtteranceMs);

  let handle;
  try {
    handle = await mic.begin({
      onFrame: (buf) => onMicFrame(r, buf),
      onFirstAudio: () => {
        if (rec !== r) return;
        r.audioFlowing = true;
        updateControls();
      },
    });
  } catch (err) {
    // A newer press took over, or that press is still going: stay quiet.
    if (err && err.superseded) return;
    if (rec !== r && rec !== null) return;
    // Also shown if the hold already ended (the permission prompt was
    // answered with Block after the key was let go).
    console.warn("mic: could not start", err && err.cause ? err.cause : err);
    showError(micErrorMessage(err));
    if (rec === r) {
      endRecording(r);
      setState("idle");
    }
    return;
  }
  if (rec !== r) {
    // Ended while the mic was opening (let go, 30 s, connection closed):
    // nothing was sent, close the mic again.
    handle.abort();
    return;
  }
  r.handle = handle;
  if (!isOpen()) {
    handle.abort();
    endRecording(r);
    setState("idle");
    showError(NOT_CONNECTED);
    return;
  }
  ws.send(JSON.stringify({ type: "audio_start" }));
  r.ack = "pending";
  r.ackTimer = setTimeout(() => {
    r.ackTimer = null;
    if (rec !== r || r.ack !== "pending") return;
    showError(WATCHDOG_MESSAGE);
    dropRecording(r);
    setState("idle");
    // Close the recording on the server too, in case it was only slow.
    openOrphan();
  }, ACK_TIMEOUT_MS);
}

function openOrphan() {
  clearOrphan();
  const o = { endSent: false, timer: null };
  if (isOpen()) {
    ws.send(JSON.stringify({ type: "audio_end" }));
    o.endSent = true;
  }
  o.timer = setTimeout(() => {
    if (orphan === o) clearOrphan();
  }, ORPHAN_HOLD_MS);
  orphan = o;
  updateControls();
}

function clearOrphan() {
  if (orphan === null) return;
  clearTimeout(orphan.timer);
  orphan = null;
  updateControls();
}

function press(source) {
  // Already holding. For a fresh Space keydown this means the last keyup
  // went to another window during a short blur: the hold goes on and the
  // coming keyup ends it.
  if (holdSource !== null) return;
  const action = pressAction({
    connected: isOpen(), busy, pendingIdle, recording: rec !== null, orphan: orphan !== null,
  });
  if (action === "offline") {
    showError(NOT_CONNECTED);
    return;
  }
  if (action === "busy") {
    flashHint(HINTS.busy);
    return;
  }
  if (action === "ignore") return;
  holdSource = source;
  refocusInput = false;
  // Inside the press gesture, so autoplay rules let the context run.
  player.unlock();
  if (action === "barge-in") {
    // The server is done; only local playback was left.
    player.stopAll();
    finishPendingIdle();
  }
  startRecording();
}

// Ends the hold: release, blur, hidden tab, the length limit, or a server
// error.
function release() {
  holdSource = null;
  const r = rec;
  if (!r || r.released) return;
  r.released = true;
  if (r.holdTimer !== null) clearTimeout(r.holdTimer);
  r.holdTimer = null;
  if (!r.handle) {
    // The mic is not open yet (very short tap, or a permission prompt
    // nobody answered): nothing was sent. End now so the page does not wait
    // on the prompt; startRecording closes a mic that opens later.
    endRecording(r);
    setState("idle");
    flashHint(earlyReleaseHint(performance.now() - r.t0));
    return;
  }
  updateControls();
  r.handle.end().then(() => {
    if (rec !== r) return;
    r.flushed = true;
    maybeSendEnd(r);
  });
}

// Server messages that concern the open recording. Runs before the general
// handling, which still applies the state and shows the error.
function handleRecordingMessage(msg) {
  const r = rec;
  switch (recordingAction(r.ack, msg)) {
    case "accept":
      r.ack = "accepted";
      if (r.ackTimer !== null) clearTimeout(r.ackTimer);
      r.ackTimer = null;
      updateControls();
      for (const buf of r.queue) sendFrame(buf);
      r.queue = [];
      maybeSendEnd(r);
      break;
    case "refuse":
      // Voice disabled or busy: no audio_end, back to idle.
      dropRecording(r);
      setState("idle");
      break;
    case "end":
      release();
      break;
    case "abort":
      dropRecording(r);
      if (msg.state !== "idle") {
        refocusInput = false;
        setBusy(true);
      }
      break;
    default:
      break;
  }
}

// ---------------------------------------------------------------------------

function handleTranscript(msg) {
  const text = typeof msg.text === "string" ? msg.text : "";
  if (msg.role === "user") {
    userLine.textContent = text;
    return;
  }
  if (msg.role !== "assistant") return;
  const added = appendedText(replyText, text);
  replyText = text;
  if (added && textDrivesMouth(getState(), turnHasAudio)) mouthText(added);
  showSubtitle(text);
  if (msg.final === true) mouthEnd();
}

function handleBinary(data) {
  player.pushChunk(data);
  if (busy && !pendingIdle) armWatchdog();
}

function handleMessage(event) {
  if (event.data instanceof ArrayBuffer) {
    handleBinary(event.data);
    return;
  }
  if (typeof event.data !== "string") return;
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    console.warn("ws: invalid JSON from server");
    return;
  }
  if (!msg || typeof msg !== "object") return;
  if (orphan !== null) {
    const action = orphanAction(orphan.endSent, msg);
    if (action === "send-end") {
      // Late listening for a recording dropped before audio_end went out.
      if (isOpen()) ws.send(JSON.stringify({ type: "audio_end" }));
      orphan.endSent = true;
      return;
    }
    if (action === "swallow") return;
    if (action === "clear") clearOrphan();
  }
  if (rec !== null) handleRecordingMessage(msg);
  if (closesOpenSentence(msg)) player.closeOpenSentence();
  if (busy && stateAction(msg, player.isBusy()) === "defer") {
    // Keep busy and the face as is; finishPendingIdle runs on drain.
    deferIdle();
    return;
  }
  const nextBusy = busyAfterMessage(busy, msg, getState());
  switch (msg.type) {
    case "state":
      setState(msg.state);
      if (msg.state === "idle") mouthEnd();
      break;
    case "tts_start":
      if (!turnHasAudio) setAudioMouth(true);
      turnHasAudio = true;
      player.startSentence(msg.sentence_id, msg.sample_rate);
      break;
    case "tts_end":
      player.endSentence(msg.sentence_id);
      break;
    case "transcript":
      handleTranscript(msg);
      break;
    case "error":
      showError(msg.message);
      break;
    default:
      break;
  }
  if (busy && !nextBusy) setBusy(false);
  else if (busy && !pendingIdle) armWatchdog();
}

function connect() {
  statusEl.textContent = "connecting";
  const sock = new WebSocket(`ws://${location.host}/ws`);
  sock.binaryType = "arraybuffer";
  ws = sock;

  sock.addEventListener("open", () => {
    connected = true;
    reconnectDelay = RECONNECT_MIN_MS;
    statusEl.textContent = "connected";
    if (errorLine.textContent === NOT_CONNECTED) errorLine.textContent = "";
    sock.send(JSON.stringify({ type: "ping" }));
    updateHint();
  });

  sock.addEventListener("message", handleMessage);

  sock.addEventListener("close", () => {
    if (ws !== sock) return;
    connected = false;
    statusEl.textContent = "disconnected";
    // Drop any recording; the server forgets it with the connection.
    if (rec !== null) dropRecording(rec);
    clearOrphan();
    clearBlurTimer();
    holdSource = null;
    // Do not leave the face stuck in looking or speaking after a disconnect.
    player.stopAll();
    cancelPendingIdle();
    setState("idle");
    mouthReset();
    setBusy(false);
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(RECONNECT_MAX_MS, reconnectDelay * 2);
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy || rec !== null || orphan !== null) return;
  const text = input.value.trim();
  if (!text) return;
  errorLine.textContent = "";
  if (!isOpen()) {
    showError(NOT_CONNECTED);
    return;
  }
  if (text.length > MAX_TEXT) {
    showError(`Message too long (max ${MAX_TEXT} characters)`);
    return;
  }
  // Inside the submit gesture, so autoplay rules let the context run.
  player.unlock();
  // Leftover audio from the previous turn (input is normally disabled until
  // it drains, so this should not happen).
  player.stopAll();
  cancelPendingIdle();
  beginExchange(text);
  input.value = "";
  ws.send(JSON.stringify({ type: "text", text }));
  refocusInput = true;
  setBusy(true);
});

// Space anywhere but in a text field. Repeats are swallowed.
window.addEventListener("keydown", (event) => {
  const action = keyDownAction(event);
  if (action === "none") return;
  event.preventDefault();
  if (action === "press") press("key");
});

window.addEventListener("keyup", (event) => {
  const action = keyUpAction(event, holdSource);
  if (action === "none") return;
  event.preventDefault();
  if (action === "release") release();
});

// keyup and pointerup never arrive once the window or tab loses focus. A
// hidden tab ends the hold at once. A window blur ends it only if the window
// is still unfocused BLUR_GRACE_MS later: a mic-sharing indicator may take
// focus for a moment when the mic opens. If Space was let go meanwhile
// (keyup sent elsewhere), the hold goes on and the next Space press and
// release ends it (see keyDownAction).
function clearBlurTimer() {
  if (blurTimer !== null) clearTimeout(blurTimer);
  blurTimer = null;
}

window.addEventListener("blur", () => {
  if (focusLossAction(holdSource, recordingHeld()) !== "release") return;
  clearBlurTimer();
  blurTimer = setTimeout(() => {
    blurTimer = null;
    if (blurCheckAction(document.hasFocus(), holdSource, recordingHeld()) === "release") release();
  }, BLUR_GRACE_MS);
});

window.addEventListener("focus", clearBlurTimer);

document.addEventListener("visibilitychange", () => {
  if (!document.hidden) return;
  clearBlurTimer();
  if (focusLossAction(holdSource, recordingHeld()) === "release") release();
});

talkBtn.addEventListener("pointerdown", (event) => {
  if (event.button !== 0) return;
  // No focus change, no text selection.
  event.preventDefault();
  press("pointer");
});

for (const type of ["pointerup", "pointercancel", "pointerleave"]) {
  talkBtn.addEventListener(type, () => {
    if (pointerEndAction(holdSource) === "release") release();
  });
}

talkBtn.addEventListener("contextmenu", (event) => event.preventDefault());

// Esc leaves the text box so Space talks again.
input.addEventListener("keydown", (event) => {
  if (event.key === "Escape") input.blur();
});
input.addEventListener("focus", updateHint);
input.addEventListener("blur", updateHint);

updateControls();
connect();
