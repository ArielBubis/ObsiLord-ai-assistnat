// Photos setting (Settings, Photos): what happens to a look's photo after the
// reply. "off": used for that reply only. "followup": the newest photos stay
// with the conversation (in memory) for a local model that sees photos.
// "disk": as followup, and every look's photo is saved on this PC.
//
// The choice is remembered in this browser (localStorage) and sent as
// {"type": "photos", "mode": ...} on every connect and change; the server
// answers with the same message. Without a stored choice the page shows the
// server's default (/config/photos) and sends nothing, so the server keeps it.
//
// The pure helpers at the top are tested with node --test.

export const MODES = ["off", "followup", "disk"];
export const STORAGE_KEY = "obsilord.photos";

export function photosMessage(mode) {
  return MODES.includes(mode) ? JSON.stringify({ type: "photos", mode }) : null;
}

// The mode in a server "photos" message, or null.
export function modeFrom(msg) {
  if (!msg || typeof msg !== "object" || msg.type !== "photos") return null;
  return MODES.includes(msg.mode) ? msg.mode : null;
}

export function readPreference(storage) {
  try {
    const value = storage ? storage.getItem(STORAGE_KEY) : null;
    return MODES.includes(value) ? value : null;
  } catch {
    return null;
  }
}

export function writePreference(storage, mode) {
  try {
    if (storage) storage.setItem(STORAGE_KEY, mode);
  } catch {
    // Not remembered.
  }
}

// el: { radios } (one radio input per mode, value = the mode). send(text)
// sends a JSON text frame if the socket is open.
// Returns { announce, setDefault, show }.
export function initPhotos(el, storage, send) {
  let chosen = readPreference(storage);

  function check(mode) {
    for (const r of el.radios) r.checked = r.value === mode;
  }

  for (const r of el.radios) {
    r.addEventListener("change", () => {
      if (!r.checked) return;
      chosen = r.value;
      writePreference(storage, chosen);
      const text = photosMessage(chosen);
      if (text) send(text);
    });
  }

  check(chosen || "off");

  return {
    // Tell a newly opened socket the stored choice (none: the server default stays).
    announce() {
      const text = chosen ? photosMessage(chosen) : null;
      if (text) send(text);
    },
    // The server's default, shown while the page has no choice of its own.
    setDefault(mode) {
      if (!chosen && MODES.includes(mode)) check(mode);
    },
    show(msg) {
      const mode = modeFrom(msg);
      if (mode) check(mode);
    },
  };
}
