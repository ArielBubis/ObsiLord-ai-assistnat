// WebSocket client and text chat UI. Shows connection status, sends typed
// messages, shows the streamed assistant reply as a subtitle, applies server
// state events to the face, and feeds new reply text to the fake mouth
// envelope. Reconnects with a short backoff (capped at 2 s) after a close.

import { setState, getState, mouthText, mouthEnd, mouthReset } from "./face.js";
import { appendedText } from "./mouth.js";
import { busyAfterMessage, WATCHDOG_MS, WATCHDOG_MESSAGE } from "./turn.js";

const MAX_TEXT = 2000;
const RECONNECT_MIN_MS = 500;
const RECONNECT_MAX_MS = 2000;
const NOT_CONNECTED = "Not connected";

const statusEl = document.getElementById("status");
const form = document.getElementById("chat");
const input = document.getElementById("chat-input");
const sendBtn = document.getElementById("chat-send");
const userLine = document.getElementById("user-line");
const subtitle = document.getElementById("subtitle");
const errorLine = document.getElementById("error-line");

let ws = null;
let connected = false;
let reconnectDelay = RECONNECT_MIN_MS;
// A turn is in progress from send until state idle or an error.
let busy = false;
// Cumulative assistant text seen so far in the current reply.
let replyText = "";
// One-shot timer, re-armed on every server message while a turn is active.
let watchdog = null;

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
    setState("idle");
    mouthEnd();
    setBusy(false);
  }, WATCHDOG_MS);
}

function setBusy(value) {
  busy = value;
  input.disabled = value;
  sendBtn.disabled = value;
  if (value) {
    armWatchdog();
  } else {
    clearWatchdog();
    input.focus();
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

function handleTranscript(msg) {
  const text = typeof msg.text === "string" ? msg.text : "";
  if (msg.role === "user") {
    userLine.textContent = text;
    return;
  }
  if (msg.role !== "assistant") return;
  const added = appendedText(replyText, text);
  replyText = text;
  if (added) mouthText(added);
  showSubtitle(text);
  if (msg.final === true) mouthEnd();
}

function handleMessage(event) {
  if (typeof event.data !== "string") return;
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    console.warn("ws: invalid JSON from server");
    return;
  }
  if (!msg || typeof msg !== "object") return;
  const nextBusy = busyAfterMessage(busy, msg, getState());
  switch (msg.type) {
    case "state":
      setState(msg.state);
      if (msg.state === "idle") mouthEnd();
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
  else if (busy) armWatchdog();
}

function connect() {
  statusEl.textContent = "connecting";
  const sock = new WebSocket(`ws://${location.host}/ws`);
  ws = sock;

  sock.addEventListener("open", () => {
    connected = true;
    reconnectDelay = RECONNECT_MIN_MS;
    statusEl.textContent = "connected";
    if (errorLine.textContent === NOT_CONNECTED) errorLine.textContent = "";
    sock.send(JSON.stringify({ type: "ping" }));
  });

  sock.addEventListener("message", handleMessage);

  sock.addEventListener("close", () => {
    if (ws !== sock) return;
    connected = false;
    statusEl.textContent = "disconnected";
    // Do not leave the face stuck in looking or speaking after a disconnect.
    setState("idle");
    mouthReset();
    setBusy(false);
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(RECONNECT_MAX_MS, reconnectDelay * 2);
  });
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  if (busy) return;
  const text = input.value.trim();
  if (!text) return;
  errorLine.textContent = "";
  if (!connected || !ws || ws.readyState !== WebSocket.OPEN) {
    showError(NOT_CONNECTED);
    return;
  }
  if (text.length > MAX_TEXT) {
    showError(`Message too long (max ${MAX_TEXT} characters)`);
    return;
  }
  replyText = "";
  showSubtitle("");
  userLine.textContent = text;
  input.value = "";
  ws.send(JSON.stringify({ type: "text", text }));
  setBusy(true);
});

connect();
