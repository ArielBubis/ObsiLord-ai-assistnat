// WebSocket client. Shows connection status and applies server state events
// to the face. Other message types are ignored until later phases.

import { setState } from "./face.js";

const statusEl = document.getElementById("status");
const ws = new WebSocket(`ws://${location.host}/ws`);

ws.addEventListener("open", () => {
  statusEl.textContent = "connected";
  ws.send(JSON.stringify({ type: "ping" }));
});

ws.addEventListener("close", () => {
  statusEl.textContent = "disconnected";
  // Do not leave the face stuck in looking or speaking after a disconnect.
  setState("idle");
});

ws.addEventListener("message", (event) => {
  if (typeof event.data !== "string") return;
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    console.warn("ws: invalid JSON from server");
    return;
  }
  if (msg && msg.type === "state") {
    setState(msg.state);
  }
});
