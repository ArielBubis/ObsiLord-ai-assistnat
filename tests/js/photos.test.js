import { test } from "node:test";
import assert from "node:assert/strict";

import {
  MODES, STORAGE_KEY, photosMessage, readPreference, writePreference, modeFrom, initPhotos,
} from "../../web/photos.js";

function memoryStorage(initial = {}) {
  const data = { ...initial };
  return {
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => { data[k] = String(v); },
    data,
  };
}

function radios() {
  return MODES.map((value) => {
    const listeners = [];
    return {
      value,
      checked: false,
      addEventListener: (type, fn) => listeners.push(fn),
      fire() { for (const fn of listeners) fn(); },
    };
  });
}

const checked = (rs) => rs.filter((r) => r.checked).map((r) => r.value);

test("the three modes, off first", () => {
  assert.deepEqual(MODES, ["off", "followup", "disk"]);
});

test("photosMessage sends a known mode only", () => {
  assert.equal(photosMessage("disk"), JSON.stringify({ type: "photos", mode: "disk" }));
  assert.equal(photosMessage("always"), null);
});

test("modeFrom reads a photos message", () => {
  assert.equal(modeFrom({ type: "photos", mode: "followup" }), "followup");
  for (const bad of [null, { type: "photos" }, { type: "photos", mode: "x" }, { type: "state", mode: "off" }]) {
    assert.equal(modeFrom(bad), null);
  }
});

test("the preference is remembered and a broken storage is ignored", () => {
  const storage = memoryStorage();
  assert.equal(readPreference(storage), null);
  writePreference(storage, "disk");
  assert.equal(storage.data[STORAGE_KEY], "disk");
  assert.equal(readPreference(storage), "disk");
  assert.equal(readPreference(memoryStorage({ [STORAGE_KEY]: "bogus" })), null);
  const broken = { getItem() { throw new Error("blocked"); }, setItem() { throw new Error("blocked"); } };
  assert.equal(readPreference(broken), null);
  writePreference(broken, "off"); // does not throw
  assert.equal(readPreference(null), null);
});

test("without a stored choice the server default shows and nothing is announced", () => {
  const rs = radios();
  const sent = [];
  const photos = initPhotos({ radios: rs }, memoryStorage(), (t) => sent.push(t));
  assert.deepEqual(checked(rs), ["off"]);
  photos.setDefault("followup");
  assert.deepEqual(checked(rs), ["followup"]);
  photos.announce();
  assert.deepEqual(sent, []);
});

test("a stored choice wins over the default and is announced on connect", () => {
  const rs = radios();
  const sent = [];
  const photos = initPhotos({ radios: rs }, memoryStorage({ [STORAGE_KEY]: "disk" }), (t) => sent.push(t));
  photos.setDefault("off");
  assert.deepEqual(checked(rs), ["disk"]);
  photos.announce();
  assert.deepEqual(sent, [photosMessage("disk")]);
});

test("picking a mode stores and sends it, and the server's answer shows", () => {
  const rs = radios();
  const sent = [];
  const storage = memoryStorage();
  const photos = initPhotos({ radios: rs }, storage, (t) => sent.push(t));
  rs[1].checked = true;
  rs[1].fire();
  assert.equal(storage.data[STORAGE_KEY], "followup");
  assert.deepEqual(sent, [photosMessage("followup")]);
  photos.show({ type: "photos", mode: "disk" });
  assert.deepEqual(checked(rs), ["disk"]);
});
