// Front-end constants. The server serves the [face] section of config.toml at
// GET /config/face; loadFaceConfig() fetches it at startup. The values below
// are the fallback if the fetch fails or the response is invalid, and they
// mirror config.example.toml.
//
// The 64x48 grid is fixed in JS (the skin is drawn for it); a different size
// from the server only produces a warning. Only the mouth thresholds are
// taken from the server.

export const FACE_CONFIG = Object.freeze({
  logicalWidth: 64,
  logicalHeight: 48,
  // Playback RMS cutoffs for mouth frames 1, 2, 3 (fallback).
  mouthThresholds: Object.freeze([0.05, 0.15, 0.3]),
});

export const FACE_CONFIG_URL = "/config/face";

let activeThresholds = FACE_CONFIG.mouthThresholds;

// Thresholds in use: from the server if loaded and valid, else the fallback.
export function getMouthThresholds() {
  return activeThresholds;
}

// Three strictly increasing finite numbers in (0, 1].
export function validThresholds(t) {
  if (!Array.isArray(t) || t.length !== 3) return false;
  for (let i = 0; i < 3; i++) {
    const v = t[i];
    if (typeof v !== "number" || !Number.isFinite(v) || v <= 0 || v > 1) return false;
    if (i > 0 && !(v > t[i - 1])) return false;
  }
  return true;
}

// Checks a /config/face response. Pure.
// Returns { thresholds, warnings }: thresholds is a frozen copy of valid
// server values or null (use the fallback); warnings lists problems to log.
export function validateFaceConfig(raw, fallback = FACE_CONFIG) {
  const warnings = [];
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { thresholds: null, warnings: ["face config: response is not an object"] };
  }
  let thresholds = null;
  if (validThresholds(raw.mouth_thresholds)) {
    thresholds = Object.freeze([...raw.mouth_thresholds]);
  } else {
    warnings.push(
      `face config: invalid mouth_thresholds ${JSON.stringify(raw.mouth_thresholds)}, using fallback`,
    );
  }
  if (raw.logical_width !== fallback.logicalWidth || raw.logical_height !== fallback.logicalHeight) {
    warnings.push(
      `face config: server grid ${raw.logical_width}x${raw.logical_height} differs from the fixed `
        + `${fallback.logicalWidth}x${fallback.logicalHeight}, keeping ${fallback.logicalWidth}x${fallback.logicalHeight}`,
    );
  }
  return { thresholds, warnings };
}

// Shared by the config loaders: GET url and parse JSON. Throws on network
// errors, non-OK status, or bad JSON.
export async function fetchConfigJson(url, fetchFn = globalThis.fetch) {
  const resp = await fetchFn(url, { cache: "no-store" });
  if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
  return resp.json();
}

function errText(err) {
  return err && err.message ? err.message : String(err);
}

// Fetches the face config once and applies valid thresholds. Never throws.
// Returns the thresholds in use afterwards.
export async function loadFaceConfig(fetchFn = globalThis.fetch, warn = console.warn) {
  let raw;
  try {
    raw = await fetchConfigJson(FACE_CONFIG_URL, fetchFn);
  } catch (err) {
    warn(`face config: fetch failed (${errText(err)}), using fallback`);
    return activeThresholds;
  }
  const { thresholds, warnings } = validateFaceConfig(raw);
  if (warnings.length > 0) warn(warnings.join("; "));
  if (thresholds) activeThresholds = thresholds;
  return activeThresholds;
}

// Test helper: back to the fallback thresholds.
export function resetFaceConfig() {
  activeThresholds = FACE_CONFIG.mouthThresholds;
}

// Playback EQ, served from the [tts] section of config.toml (flat keys
// low_shelf_hz, low_shelf_db, high_shelf_hz, high_shelf_db) at
// GET /config/audio. Defaults equal the backend defaults and are used if the
// fetch fails or the response is invalid. Keys match the server response.
export const AUDIO_CONFIG_URL = "/config/audio";

export const AUDIO_EQ_DEFAULTS = Object.freeze({
  low_shelf_hz: 250.0,
  low_shelf_db: -6.0,
  high_shelf_hz: 3500.0,
  high_shelf_db: 0.0,
});

export const EQ_LIMITS = Object.freeze({ minHz: 20, maxHz: 10000, minDb: -24, maxDb: 24 });

function inRange(v, lo, hi) {
  return typeof v === "number" && Number.isFinite(v) && v >= lo && v <= hi;
}

// Checks a /config/audio response. Pure. All four values must be valid,
// otherwise the whole response is rejected (eq null) with one warning.
// Extra keys are ignored.
export function validateAudioConfig(raw) {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return { eq: null, warnings: ["audio config: response is not an object"] };
  }
  const bad = [];
  for (const key of ["low_shelf_hz", "high_shelf_hz"]) {
    if (!inRange(raw[key], EQ_LIMITS.minHz, EQ_LIMITS.maxHz)) bad.push(`${key}=${JSON.stringify(raw[key])}`);
  }
  for (const key of ["low_shelf_db", "high_shelf_db"]) {
    if (!inRange(raw[key], EQ_LIMITS.minDb, EQ_LIMITS.maxDb)) bad.push(`${key}=${JSON.stringify(raw[key])}`);
  }
  if (bad.length > 0) {
    return { eq: null, warnings: [`audio config: invalid ${bad.join(", ")}, using defaults`] };
  }
  const eq = Object.freeze({
    low_shelf_hz: raw.low_shelf_hz,
    low_shelf_db: raw.low_shelf_db,
    high_shelf_hz: raw.high_shelf_hz,
    high_shelf_db: raw.high_shelf_db,
  });
  return { eq, warnings: [] };
}

// Fetches the EQ settings once. Never throws; returns the settings to use
// (server values if valid, else the defaults) and logs one warning otherwise.
export async function loadAudioConfig(fetchFn = globalThis.fetch, warn = console.warn) {
  let raw;
  try {
    raw = await fetchConfigJson(AUDIO_CONFIG_URL, fetchFn);
  } catch (err) {
    warn(`audio config: fetch failed (${errText(err)}), using defaults`);
    return AUDIO_EQ_DEFAULTS;
  }
  const { eq, warnings } = validateAudioConfig(raw);
  if (warnings.length > 0) warn(warnings.join("; "));
  return eq || AUDIO_EQ_DEFAULTS;
}
