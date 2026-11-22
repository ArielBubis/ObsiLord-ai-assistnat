// Photos setting (Settings, Photos): what happens to a look's photo after the
// reply. "off": used for that reply only. "followup": the newest photos stay
// with the conversation (in memory) for a local model that sees photos.
// "disk": as followup, and every look's photo is saved on this PC.
//
// The server owns the choice. It sends {"type": "photos", "mode"} on connect
// and to every open page after a change; the page shows that and sends
// {"type": "photos", "mode"} only when the user picks one. Nothing is kept
// in this browser.
//
// The pure helpers at the top are tested with node --test.

export const MODES = ["off", "followup", "disk"];

export function photosMessage(mode) {
  return MODES.includes(mode) ? JSON.stringify({ type: "photos", mode }) : null;
}

// The mode in a server "photos" message, or null.
export function modeFrom(msg) {
  if (!msg || typeof msg !== "object" || msg.type !== "photos") return null;
  return MODES.includes(msg.mode) ? msg.mode : null;
}

// el: { radios } (one radio input per mode, value = the mode). send(text)
// sends a JSON text frame if the socket is open. Returns { show }.
export function initPhotos(el, send) {
  function check(mode) {
    for (const r of el.radios) r.checked = r.value === mode;
  }

  for (const r of el.radios) {
    r.addEventListener("change", () => {
      if (!r.checked) return;
      const text = photosMessage(r.value);
      if (text) send(text);
    });
  }

  check("off");

  return {
    show(msg) {
      const mode = modeFrom(msg);
      if (mode) check(mode);
    },
  };
}
