// Snapshot panel: shows the frame the camera took for a look and the crop
// that was sent to the web image search, when the Snapshots switch is on.
//
// The switch is off by default and remembered in this browser (localStorage).
// The server only sends snapshot messages while the page has asked for them
// ({"type": "snapshots", "on": true}, sent on every connect and on every
// change of the switch). The pictures arrive as base64 JPEG inside a JSON
// message and are shown through data: URLs, so they stay in memory: nothing
// is fetched over HTTP and nothing reaches the browser's disk cache. Turning
// the switch off clears both pictures.
//
// The pure helpers at the top are tested with node --test.

export const STORAGE_KEY = "obsilord.snapshots";
export const NOTE_IDLE = "Nothing sent to the search.";
export const NOTE_SENT = "This crop was uploaded.";

const BASE64 = /^[A-Za-z0-9+/]+={0,2}$/;

// The data: URL for a valid snapshot message, or null.
export function snapshotSrc(msg) {
  if (!msg || typeof msg !== "object" || msg.type !== "snapshot") return null;
  if (msg.kind !== "camera" && msg.kind !== "search") return null;
  if (typeof msg.jpeg_b64 !== "string" || !BASE64.test(msg.jpeg_b64)) return null;
  return `data:image/jpeg;base64,${msg.jpeg_b64}`;
}

export function switchMessage(on) {
  return JSON.stringify({ type: "snapshots", on: Boolean(on) });
}

// Storage may be unavailable (private mode, blocked): then the switch just
// starts off and is not remembered.
export function readPreference(storage) {
  try {
    return storage.getItem(STORAGE_KEY) === "1";
  } catch {
    return false;
  }
}

export function writePreference(storage, on) {
  try {
    storage.setItem(STORAGE_KEY, on ? "1" : "0");
  } catch {
    // Not remembered.
  }
}

// What a snapshot message changes: which picture gets the src, and whether
// the search picture is cleared (a new camera frame starts a new look).
export function snapshotUpdate(msg) {
  const src = snapshotSrc(msg);
  if (src === null) return null;
  return msg.kind === "camera"
    ? { camera: src, search: null, note: NOTE_IDLE }
    : { search: src, note: NOTE_SENT };
}

// el: { toggle, panel, camera, search, note }. send(text) sends a JSON text
// frame if the socket is open. Returns { isOn, show, announce }.
export function initSnapshots(el, storage, send) {
  function setImage(img, src) {
    if (src) {
      img.src = src;
      img.hidden = false;
    } else {
      img.removeAttribute("src");
      img.hidden = true;
    }
  }

  function clear() {
    setImage(el.camera, null);
    setImage(el.search, null);
    el.note.textContent = NOTE_IDLE;
  }

  function apply(on) {
    el.toggle.checked = on;
    el.panel.hidden = !on;
    if (!on) clear();
  }

  el.toggle.addEventListener("change", () => {
    const on = el.toggle.checked;
    writePreference(storage, on);
    apply(on);
    send(switchMessage(on));
  });

  clear();
  apply(readPreference(storage));

  return {
    isOn: () => el.toggle.checked,
    // Tell a newly opened socket what the switch says.
    announce: () => send(switchMessage(el.toggle.checked)),
    show(msg) {
      if (!el.toggle.checked) return;
      const update = snapshotUpdate(msg);
      if (update === null) return;
      if ("camera" in update) setImage(el.camera, update.camera);
      setImage(el.search, update.search);
      el.note.textContent = update.note;
    },
  };
}
