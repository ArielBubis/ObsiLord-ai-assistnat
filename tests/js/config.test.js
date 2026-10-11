import { test } from "node:test";
import assert from "node:assert/strict";

import {
  FACE_CONFIG, FACE_CONFIG_URL, validThresholds, validateFaceConfig, loadFaceConfig,
  getMouthThresholds, resetFaceConfig, AUDIO_CONFIG_URL, AUDIO_EQ_DEFAULTS, EQ_LIMITS,
  validateAudioConfig, loadAudioConfig, fetchConfigJson,
} from "../../web/config.js";
import { rmsToFrame } from "../../web/mouth.js";

const good = { logical_width: 64, logical_height: 48, mouth_thresholds: [0.04, 0.12, 0.25] };

test("fallback thresholds mirror config.example.toml", () => {
  assert.deepEqual([...FACE_CONFIG.mouthThresholds], [0.05, 0.15, 0.3]);
  assert.ok(validThresholds(FACE_CONFIG.mouthThresholds));
});

test("validThresholds: three strictly increasing numbers in (0, 1]", () => {
  assert.ok(validThresholds([0.05, 0.15, 0.3]));
  assert.ok(validThresholds([0.1, 0.5, 1]));
  for (const bad of [
    [0, 0.1, 0.2], [0.1, 0.1, 0.2], [0.3, 0.2, 0.1], [0.1, 0.2, 1.01], [-0.1, 0.2, 0.3],
    [0.1, 0.2], [0.1, 0.2, 0.3, 0.4], [0.1, "0.2", 0.3], [0.1, NaN, 0.3], [0.1, 0.2, Infinity],
    null, undefined, "0.1,0.2,0.3", {},
  ]) {
    assert.equal(validThresholds(bad), false, JSON.stringify(bad));
  }
});

test("validateFaceConfig accepts a good response without warnings", () => {
  const { thresholds, warnings } = validateFaceConfig(good);
  assert.deepEqual([...thresholds], good.mouth_thresholds);
  assert.ok(Object.isFrozen(thresholds));
  assert.deepEqual(warnings, []);
});

test("validateFaceConfig rejects bad thresholds with a warning", () => {
  const { thresholds, warnings } = validateFaceConfig({ ...good, mouth_thresholds: [0.3, 0.2, 0.1] });
  assert.equal(thresholds, null);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /mouth_thresholds/);
});

test("validateFaceConfig warns on a different grid but keeps valid thresholds", () => {
  const { thresholds, warnings } = validateFaceConfig({ ...good, logical_width: 128 });
  assert.deepEqual([...thresholds], good.mouth_thresholds);
  assert.equal(warnings.length, 1);
  assert.match(warnings[0], /128x48/);
});

test("validateFaceConfig rejects non-objects", () => {
  for (const raw of [null, undefined, 5, "x", [1, 2, 3]]) {
    const { thresholds, warnings } = validateFaceConfig(raw);
    assert.equal(thresholds, null);
    assert.equal(warnings.length, 1);
  }
});

function fakeFetch(body, { ok = true, status = 200, throws = false } = {}) {
  const calls = [];
  const fn = async (url) => {
    calls.push(url);
    if (throws) throw new Error("network down");
    return { ok, status, json: async () => (typeof body === "function" ? body() : body) };
  };
  fn.calls = calls;
  return fn;
}

test("loadFaceConfig applies valid server thresholds and rmsToFrame uses them", async () => {
  resetFaceConfig();
  const warnings = [];
  const fetchFn = fakeFetch(good);
  const t = await loadFaceConfig(fetchFn, (w) => warnings.push(w));
  assert.deepEqual(fetchFn.calls, [FACE_CONFIG_URL]);
  assert.deepEqual([...t], good.mouth_thresholds);
  assert.equal(getMouthThresholds(), t);
  assert.deepEqual(warnings, []);
  // 0.045 is frame 1 with the server cutoffs, frame 0 with the fallback.
  assert.equal(rmsToFrame(0.045), 1);
  resetFaceConfig();
  assert.equal(rmsToFrame(0.045), 0);
});

test("loadFaceConfig keeps the fallback and warns once on failures", async () => {
  const cases = [
    fakeFetch(null, { throws: true }),
    fakeFetch(good, { ok: false, status: 404 }),
    fakeFetch(() => { throw new SyntaxError("bad json"); }),
    fakeFetch({ ...good, mouth_thresholds: [0.5, 0.4, 2] }),
  ];
  for (const fetchFn of cases) {
    resetFaceConfig();
    const warnings = [];
    const t = await loadFaceConfig(fetchFn, (w) => warnings.push(w));
    assert.equal(t, FACE_CONFIG.mouthThresholds);
    assert.equal(getMouthThresholds(), FACE_CONFIG.mouthThresholds);
    assert.equal(warnings.length, 1, JSON.stringify(warnings));
  }
  resetFaceConfig();
});

const eqGood = { low_shelf_hz: 250.0, low_shelf_db: -6.0, high_shelf_hz: 3500.0, high_shelf_db: 0.0 };

test("audio EQ defaults equal the backend defaults", () => {
  assert.deepEqual({ ...AUDIO_EQ_DEFAULTS }, eqGood);
  assert.deepEqual(validateAudioConfig(AUDIO_EQ_DEFAULTS).warnings, []);
});

test("validateAudioConfig accepts in-range values including the limits", () => {
  const { eq, warnings } = validateAudioConfig({ ...eqGood, extra: 1 });
  assert.deepEqual({ ...eq }, eqGood);
  assert.deepEqual(warnings, []);
  const edge = { low_shelf_hz: EQ_LIMITS.minHz, low_shelf_db: EQ_LIMITS.minDb,
    high_shelf_hz: EQ_LIMITS.maxHz, high_shelf_db: EQ_LIMITS.maxDb };
  assert.deepEqual({ ...validateAudioConfig(edge).eq }, edge);
});

test("validateAudioConfig rejects bad values with one warning", () => {
  const bads = [
    { low_shelf_hz: 19.9 }, { high_shelf_hz: 10001 }, { low_shelf_db: -24.1 }, { high_shelf_db: 25 },
    { low_shelf_hz: "250" }, { low_shelf_db: NaN }, { high_shelf_db: Infinity }, { high_shelf_hz: null },
    { low_shelf_db: true }, { high_shelf_db: undefined },
  ];
  for (const change of bads) {
    const { eq, warnings } = validateAudioConfig({ ...eqGood, ...change });
    assert.equal(eq, null, JSON.stringify(change));
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], new RegExp(Object.keys(change)[0]));
  }
  for (const raw of [null, 3, "x", [1, 2, 3, 4]]) {
    assert.equal(validateAudioConfig(raw).eq, null);
  }
});

test("loadAudioConfig returns server values or the defaults with one warning", async () => {
  const server = { low_shelf_hz: 180, low_shelf_db: -9, high_shelf_hz: 5000, high_shelf_db: 2 };
  const ok = fakeFetch(server);
  const warnings = [];
  assert.deepEqual({ ...(await loadAudioConfig(ok, (w) => warnings.push(w))) }, server);
  assert.deepEqual(ok.calls, [AUDIO_CONFIG_URL]);
  assert.deepEqual(warnings, []);
  for (const fetchFn of [
    fakeFetch(null, { throws: true }), fakeFetch(server, { ok: false, status: 500 }),
    fakeFetch(() => { throw new SyntaxError("bad"); }), fakeFetch({ ...server, high_shelf_db: 30 }),
  ]) {
    const w = [];
    assert.equal(await loadAudioConfig(fetchFn, (x) => w.push(x)), AUDIO_EQ_DEFAULTS);
    assert.equal(w.length, 1, JSON.stringify(w));
  }
});

test("fetchConfigJson throws on HTTP errors", async () => {
  await assert.rejects(fetchConfigJson("/x", fakeFetch({}, { ok: false, status: 404 })), /HTTP 404/);
  assert.deepEqual(await fetchConfigJson("/x", fakeFetch({ a: 1 })), { a: 1 });
});
