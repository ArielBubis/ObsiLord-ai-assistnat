// Debug panel, shown only with ?debug=1.
// Extra debug-only URL params for deterministic screenshots:
//   state=<name>  force a face state on load
//   since=<ms>    pretend the forced state was entered this long ago
//   mouth=0..3    force a mouth frame
// Mouth precedence: touching the slider (or mouth=) sets a manual mouth that
// overrides the text envelope; the "mouth auto" button hands control back.
//   idle=0        freeze blink, glance, and bob
// EQ sliders (bass = low shelf dB, treble = high shelf dB) change the
// playback filters live, for tuning by ear. They start at the values from
// /config/audio; copy the line shown into the [tts] section of config.toml
// to keep them.

import { STATES } from "./state.js";
import {
  setState, setMouth, getMouth, getState, onStateChange, setIdleEnabled, clearMouthOverride,
} from "./face.js";
import { player, audioConfigReady } from "./ws.js";

function formatDb(v) {
  return `${v > 0 ? "+" : ""}${v} dB`;
}

function initEqSliders(panel) {
  const sliders = [
    ["#debug-bass", "#debug-bass-value", "low_shelf_db"],
    ["#debug-treble", "#debug-treble-value", "high_shelf_db"],
  ];
  const eqLine = panel.querySelector("#debug-eq-config");
  function show() {
    const eq = player.getEq();
    for (const [, valueSel, key] of sliders) panel.querySelector(valueSel).textContent = formatDb(eq[key]);
    eqLine.textContent = `[tts] low_shelf_hz = ${eq.low_shelf_hz}, low_shelf_db = ${eq.low_shelf_db}, `
      + `high_shelf_hz = ${eq.high_shelf_hz}, high_shelf_db = ${eq.high_shelf_db}`;
  }
  for (const [sliderSel, , key] of sliders) {
    const slider = panel.querySelector(sliderSel);
    slider.disabled = true;
    slider.addEventListener("input", () => {
      player.setEq({ ...player.getEq(), [key]: Number(slider.value) });
      show();
    });
  }
  // Enable once the server values (or defaults) are in, so a late config
  // response cannot overwrite a value the user is tuning.
  audioConfigReady.then((eq) => {
    for (const [sliderSel, , key] of sliders) {
      const slider = panel.querySelector(sliderSel);
      slider.value = String(eq[key]);
      slider.disabled = false;
    }
    show();
  });
}

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
  initEqSliders(panel);
}
