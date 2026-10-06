// Minimal WebSocket client for Phase 0. face-dev owns web/ from Phase 1.
(function () {
  const statusEl = document.getElementById("status");
  const ws = new WebSocket(`ws://${location.host}/ws`);

  ws.addEventListener("open", () => {
    statusEl.textContent = "connected";
    ws.send(JSON.stringify({ type: "ping" }));
  });

  ws.addEventListener("close", () => {
    statusEl.textContent = "disconnected";
  });

  ws.addEventListener("message", (event) => {
    console.log("ws message", event.data);
  });
})();
