// Debug panel, shown only with ?debug=1.
// Extra debug-only URL params for deterministic screenshots:
//   state=<name>  force a face state on load
//   since=<ms>    pretend the forced state was entered this long ago
//   mouth=0..3    force a mouth frame
// Mouth precedence: touching the slider (or mouth=) sets a manual mouth that
// overrides the text envelope; the "mouth auto" button hands control back.
//   idle=0        freeze blink, glance, and bob

import { STATES } from "./state.js";
import {
  setState, setMouth, getMouth, getState, onStateChange, setIdleEnabled, clearMouthOverride,
} from "./face.js";

export function initDebug(panel) {
  const params = new URLSearchParams(location.search);
  if (params.get("debug") !== "1") return;

  panel.hidden = false;

  const buttons = panel.querySelector("#debug-states");
  for (const name of STATES) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = name;
    btn.dataset.state = name;
    btn.addEventListener("click", () => setState(name));
    buttons.appendChild(btn);
  }

  const slider = panel.querySelector("#debug-mouth");
  const sliderValue = panel.querySelector("#debug-mouth-value");
  slider.addEventListener("input", () => {
    setMouth(slider.value);
    sliderValue.textContent = String(getMouth());
  });
  panel.querySelector("#debug-mouth-auto").addEventListener("click", () => {
    clearMouthOverride();
    sliderValue.textContent = "auto";
  });

  function markActive(current) {
    for (const btn of buttons.children) {
      btn.classList.toggle("active", btn.dataset.state === current);
    }
  }
  onStateChange((current) => markActive(current));

  if (params.has("state")) setState(params.get("state"), params.get("since"));
  if (params.has("mouth")) setMouth(params.get("mouth"));
  if (params.get("idle") === "0") setIdleEnabled(false);

  slider.value = String(getMouth());
  sliderValue.textContent = String(getMouth());
  markActive(getState());
}
