import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";

import {
  FACE_CONFIG, FACE_CONFIG_URL, validThresholds, validateFaceConfig, loadFaceConfig,
  getMouthThresholds, resetFaceConfig, FACE_SKINS, getFaceSkin, skinModulePath,
  AUDIO_CONFIG_URL, AUDIO_EQ_DEFAULTS, EQ_LIMITS,
  validateAudioConfig, loadAudioSettings, fetchConfigJson, validateMaxUtterance,
  MAX_UTTERANCE_DEFAULT_S, MAX_UTTERANCE_LIMITS,
} from "../../web/config.js";
import { rmsToFrame } from "../../web/mouth.js";

// skin "code" differs from the fallback, so applying it is observable.
const good = { logical_width: 64, logical_height: 48, mouth_thresholds: [0.04, 0.12, 0.25], skin: "code" };

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
    const { thresholds, skin, warnings } = validateFaceConfig(raw);
    assert.equal(thresholds, null);
    assert.equal(skin, "dither");
    assert.equal(warnings.length, 1);
  }
});

test("fallback skin is dither (mirrors config.example.toml), allowed skins are dither and code", () => {
  assert.equal(FACE_CONFIG.skin, "dither");
  assert.deepEqual([...FACE_SKINS], ["dither", "code"]);
  assert.ok(Object.isFrozen(FACE_SKINS));
});

test("validateFaceConfig accepts each allowed skin without warnings", () => {
  for (const name of FACE_SKINS) {
    const { thresholds, skin, warnings } = validateFaceConfig({ ...good, skin: name });
    assert.equal(skin, name);
    assert.deepEqual([...thresholds], good.mouth_thresholds);
    assert.deepEqual(warnings, []);
  }
});

test("validateFaceConfig: a missing skin gives the fallback and one warning", () => {
  const noSkin = { ...good };
  delete noSkin.skin;
  const { thresholds, skin, warnings } = validateFaceConfig(noSkin);
  assert.equal(skin, "dither");
  assert.deepEqual([...thresholds], good.mouth_thresholds);
  assert.deepEqual(warnings, ['face config: invalid skin undefined, using fallback "dither"']);
});

test("validateFaceConfig: an unknown skin name gives the fallback and one warning", () => {
  for (const name of ["pixel", "Dither", "CODE", " code", "3d", ""]) {
    const { thresholds, skin, warnings } = validateFaceConfig({ ...good, skin: name });
    assert.equal(skin, "dither", JSON.stringify(name));
    assert.deepEqual([...thresholds], good.mouth_thresholds);
    assert.deepEqual(warnings, [`face config: invalid skin ${JSON.stringify(name)}, using fallback "dither"`]);
  }
});

test("validateFaceConfig: a skin of the wrong type gives the fallback and one warning", () => {
  for (const value of [null, 0, 1, true, ["code"], { name: "code" }]) {
    const { skin, warnings } = validateFaceConfig({ ...good, skin: value });
    assert.equal(skin, "dither", JSON.stringify(value));
    assert.deepEqual(warnings, [`face config: invalid skin ${JSON.stringify(value)}, using fallback "dither"`]);
  }
});

test("validateFaceConfig checks skin and thresholds independently", () => {
  const both = validateFaceConfig({ ...good, mouth_thresholds: [0.3, 0.2, 0.1], skin: "nope" });
  assert.equal(both.thresholds, null);
  assert.equal(both.skin, "dither");
  assert.equal(both.warnings.length, 2);
  assert.match(both.warnings[0], /mouth_thresholds/);
  assert.match(both.warnings[1], /skin "nope"/);
  const skinOnly = validateFaceConfig({ ...good, mouth_thresholds: [0.3, 0.2, 0.1] });
  assert.equal(skinOnly.thresholds, null);
  assert.equal(skinOnly.skin, "code");
  assert.equal(skinOnly.warnings.length, 1);
});

test("validateFaceConfig uses the given fallback skin", () => {
  const fallback = { ...FACE_CONFIG, skin: "code" };
  assert.equal(validateFaceConfig({ ...good, skin: "x" }, fallback).skin, "code");
  assert.equal(validateFaceConfig(null, fallback).skin, "code");
});

test("skinModulePath maps each skin to its module next to web/config.js", () => {
  assert.equal(skinModulePath("dither"), "./ai-face/obsi-lord-skin.js");
  assert.equal(skinModulePath("code"), "./skin_code.js");
  const webDir = new URL("../../web/", import.meta.url);
  for (const name of FACE_SKINS) {
    assert.ok(existsSync(new URL(skinModulePath(name), webDir)), name);
  }
});

test("skinModulePath gives the fallback skin's module for unknown names", () => {
  for (const name of ["pixel", "", undefined, null, 3, "toString", "__proto__"]) {
    assert.equal(skinModulePath(name), "./ai-face/obsi-lord-skin.js", String(name));
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
  assert.equal(getFaceSkin(), "code");
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

test("getFaceSkin is the fallback until a valid skin loads, and resetFaceConfig restores it", async () => {
  resetFaceConfig();
  assert.equal(getFaceSkin(), "dither");
  await loadFaceConfig(fakeFetch(good), () => {});
  assert.equal(getFaceSkin(), "code");
  resetFaceConfig();
  assert.equal(getFaceSkin(), "dither");
  assert.equal(getMouthThresholds(), FACE_CONFIG.mouthThresholds);
});

test("loadFaceConfig keeps the fallback skin when the fetch fails", async () => {
  for (const fetchFn of [
    fakeFetch(null, { throws: true }), fakeFetch(good, { ok: false, status: 500 }),
    fakeFetch(() => { throw new SyntaxError("bad json"); }),
  ]) {
    resetFaceConfig();
    const warnings = [];
    await loadFaceConfig(fetchFn, (w) => warnings.push(w));
    assert.equal(getFaceSkin(), "dither");
    assert.equal(warnings.length, 1, JSON.stringify(warnings));
  }
  resetFaceConfig();
});

test("loadFaceConfig: an invalid skin gives the fallback, keeps valid thresholds, warns once", async () => {
  for (const skin of [undefined, "pixel", 7]) {
    resetFaceConfig();
    const warnings = [];
    const t = await loadFaceConfig(fakeFetch({ ...good, skin }), (w) => warnings.push(w));
    assert.equal(getFaceSkin(), "dither");
    assert.deepEqual([...t], good.mouth_thresholds);
    assert.equal(warnings.length, 1, JSON.stringify(warnings));
    assert.match(warnings[0], /invalid skin/);
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

test("max_utterance_s: 1 to 120 s accepted, anything else falls back to 30 s with one warning", () => {
  assert.equal(MAX_UTTERANCE_DEFAULT_S, 30);
  assert.deepEqual({ ...MAX_UTTERANCE_LIMITS }, { minS: 1, maxS: 120 });
  for (const v of [1, 30, 45.5, 120]) {
    assert.deepEqual(validateMaxUtterance({ max_utterance_s: v }), { seconds: v, warnings: [] });
  }
  for (const v of [0.99, 120.01, 0, -5, NaN, Infinity, "30", null, true, undefined]) {
    const { seconds, warnings } = validateMaxUtterance({ max_utterance_s: v });
    assert.equal(seconds, 30, String(v));
    assert.equal(warnings.length, 1);
    assert.match(warnings[0], /max_utterance_s/);
  }
  for (const raw of [null, undefined, 3, "x"]) assert.equal(validateMaxUtterance(raw).seconds, 30);
});

test("loadAudioSettings: one fetch gives the EQ and the recording limit", async () => {
  const server = { low_shelf_hz: 180, low_shelf_db: -9, high_shelf_hz: 5000, high_shelf_db: 2, max_utterance_s: 45 };
  const ok = fakeFetch(server);
  const warnings = [];
  const got = await loadAudioSettings(ok, (w) => warnings.push(w));
  assert.deepEqual({ ...got.eq }, { low_shelf_hz: 180, low_shelf_db: -9, high_shelf_hz: 5000, high_shelf_db: 2 });
  assert.equal(got.maxUtteranceS, 45);
  assert.deepEqual(ok.calls, [AUDIO_CONFIG_URL]);
  assert.deepEqual(warnings, []);
});

test("loadAudioSettings: EQ and limit fall back independently, one warning each", async () => {
  const eqServer = { low_shelf_hz: 180, low_shelf_db: -9, high_shelf_hz: 5000, high_shelf_db: 2 };
  // Limit missing or bad: EQ kept, 30 s.
  for (const extra of [{}, { max_utterance_s: 0 }, { max_utterance_s: 500 }]) {
    const w = [];
    const got = await loadAudioSettings(fakeFetch({ ...eqServer, ...extra }), (x) => w.push(x));
    assert.deepEqual({ ...got.eq }, eqServer);
    assert.equal(got.maxUtteranceS, 30);
    assert.equal(w.length, 1, JSON.stringify(w));
    assert.match(w[0], /max_utterance_s/);
  }
  // EQ bad: EQ defaults, limit kept.
  const w = [];
  const got = await loadAudioSettings(fakeFetch({ ...eqServer, high_shelf_db: 30, max_utterance_s: 12 }),
    (x) => w.push(x));
  assert.equal(got.eq, AUDIO_EQ_DEFAULTS);
  assert.equal(got.maxUtteranceS, 12);
  assert.equal(w.length, 1, JSON.stringify(w));
  assert.match(w[0], /high_shelf_db/);
});

test("loadAudioSettings: fetch failures give both defaults with one warning", async () => {
  const server = { low_shelf_hz: 180, low_shelf_db: -9, high_shelf_hz: 5000, high_shelf_db: 2, max_utterance_s: 45 };
  for (const fetchFn of [
    fakeFetch(null, { throws: true }), fakeFetch(server, { ok: false, status: 500 }),
    fakeFetch(() => { throw new SyntaxError("bad"); }),
  ]) {
    const w = [];
    const got = await loadAudioSettings(fetchFn, (x) => w.push(x));
    assert.equal(got.eq, AUDIO_EQ_DEFAULTS);
    assert.equal(got.maxUtteranceS, 30);
    assert.equal(w.length, 1, JSON.stringify(w));
  }
});

test("fetchConfigJson throws on HTTP errors", async () => {
  await assert.rejects(fetchConfigJson("/x", fakeFetch({}, { ok: false, status: 404 })), /HTTP 404/);
  assert.deepEqual(await fetchConfigJson("/x", fakeFetch({ a: 1 })), { a: 1 });
});
