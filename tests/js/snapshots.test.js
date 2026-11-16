import { test } from "node:test";
import assert from "node:assert/strict";

import {
  snapshotSrc, switchMessage, readPreference, writePreference, snapshotUpdate, initSnapshots,
  STORAGE_KEY, NOTE_IDLE, NOTE_SENT,
} from "../../web/snapshots.js";

const camera = { type: "snapshot", kind: "camera", jpeg_b64: "ZnJhbWU=" };
const search = { type: "snapshot", kind: "search", jpeg_b64: "Y3JvcA==" };

test("snapshotSrc builds a data URL for valid messages only", () => {
  assert.equal(snapshotSrc(camera), "data:image/jpeg;base64,ZnJhbWU=");
  assert.equal(snapshotSrc(search), "data:image/jpeg;base64,Y3JvcA==");
  for (const bad of [
    null,
    "x",
    { type: "state", state: "idle" },
    { type: "snapshot", kind: "other", jpeg_b64: "ZnJhbWU=" },
    { type: "snapshot", kind: "camera" },
    { type: "snapshot", kind: "camera", jpeg_b64: "" },
    { type: "snapshot", kind: "camera", jpeg_b64: 'x" onerror="alert(1)' },
    { type: "snapshot", kind: "camera", jpeg_b64: "abc,def" },
  ]) {
    assert.equal(snapshotSrc(bad), null);
  }
});

test("switchMessage is the JSON the server expects", () => {
  assert.equal(switchMessage(true), '{"type":"snapshots","on":true}');
  assert.equal(switchMessage(0), '{"type":"snapshots","on":false}');
});

function fakeStorage(initial = {}) {
  const data = { ...initial };
  return {
    data,
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => { data[k] = String(v); },
  };
}

const brokenStorage = {
  getItem() { throw new Error("blocked"); },
  setItem() { throw new Error("blocked"); },
};

test("the preference is off by default and survives broken storage", () => {
  assert.equal(readPreference(fakeStorage()), false);
  const storage = fakeStorage();
  writePreference(storage, true);
  assert.equal(storage.data[STORAGE_KEY], "1");
  assert.equal(readPreference(storage), true);
  writePreference(storage, false);
  assert.equal(readPreference(storage), false);
  assert.equal(readPreference(brokenStorage), false);
  writePreference(brokenStorage, true);
});

test("a camera frame clears the last crop, a crop keeps the frame", () => {
  assert.deepEqual(snapshotUpdate(camera), {
    camera: "data:image/jpeg;base64,ZnJhbWU=", search: null, note: NOTE_IDLE,
  });
  assert.deepEqual(snapshotUpdate(search), {
    search: "data:image/jpeg;base64,Y3JvcA==", note: NOTE_SENT,
  });
  assert.equal(snapshotUpdate({ type: "snapshot", kind: "x", jpeg_b64: "ZnJhbWU=" }), null);
});

function fakeImage() {
  return {
    hidden: false,
    src: undefined,
    removeAttribute(name) { if (name === "src") this.src = undefined; },
  };
}

function fakePage(storage) {
  const listeners = {};
  const el = {
    toggle: { checked: false, addEventListener: (name, fn) => { listeners[name] = fn; } },
    panel: { hidden: false },
    camera: fakeImage(),
    search: fakeImage(),
    note: { textContent: "" },
  };
  const sent = [];
  const snapshots = initSnapshots(el, storage, (text) => sent.push(text));
  const flip = (on) => { el.toggle.checked = on; listeners.change(); };
  return { el, sent, snapshots, flip };
}

test("the panel starts hidden and ignores snapshots while off", () => {
  const { el, sent, snapshots } = fakePage(fakeStorage());
  assert.equal(el.panel.hidden, true);
  assert.equal(snapshots.isOn(), false);
  snapshots.show(camera);
  assert.equal(el.camera.src, undefined);
  snapshots.announce();
  assert.deepEqual(sent, ['{"type":"snapshots","on":false}']);
});

test("switching on shows pictures, switching off clears them", () => {
  const storage = fakeStorage();
  const { el, sent, snapshots, flip } = fakePage(storage);
  flip(true);
  assert.equal(el.panel.hidden, false);
  assert.equal(storage.data[STORAGE_KEY], "1");
  assert.deepEqual(sent, ['{"type":"snapshots","on":true}']);

  snapshots.show(camera);
  assert.equal(el.camera.src, "data:image/jpeg;base64,ZnJhbWU=");
  assert.equal(el.camera.hidden, false);
  assert.equal(el.search.hidden, true);
  assert.equal(el.note.textContent, NOTE_IDLE);

  snapshots.show(search);
  assert.equal(el.search.src, "data:image/jpeg;base64,Y3JvcA==");
  assert.equal(el.camera.src, "data:image/jpeg;base64,ZnJhbWU=");
  assert.equal(el.note.textContent, NOTE_SENT);

  // The next look starts with a new frame and no crop.
  snapshots.show(camera);
  assert.equal(el.search.src, undefined);
  assert.equal(el.search.hidden, true);

  flip(false);
  assert.equal(el.panel.hidden, true);
  assert.equal(el.camera.src, undefined);
  assert.equal(storage.data[STORAGE_KEY], "0");
  assert.equal(sent.at(-1), '{"type":"snapshots","on":false}');
});

test("a remembered preference turns the panel on at load", () => {
  const { el, snapshots } = fakePage(fakeStorage({ [STORAGE_KEY]: "1" }));
  assert.equal(el.toggle.checked, true);
  assert.equal(el.panel.hidden, false);
  assert.equal(snapshots.isOn(), true);
});
