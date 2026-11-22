import { test } from "node:test";
import assert from "node:assert/strict";

import * as photosModule from "../../web/photos.js";

const { MODES, photosMessage, modeFrom, initPhotos } = photosModule;

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

test("the page keeps no choice of its own", () => {
  assert.equal(photosModule.STORAGE_KEY, undefined);
  assert.equal(photosModule.readPreference, undefined);
  assert.equal(photosModule.writePreference, undefined);
});

test("nothing is sent until the user picks, and the server's message sets the radios", () => {
  const rs = radios();
  const sent = [];
  const photos = initPhotos({ radios: rs }, (t) => sent.push(t));
  assert.deepEqual(checked(rs), ["off"]);
  assert.equal(photos.announce, undefined);
  photos.show({ type: "photos", mode: "disk" });
  assert.deepEqual(checked(rs), ["disk"]);
  photos.show({ type: "photos", mode: "bogus" });
  assert.deepEqual(checked(rs), ["disk"]);
  assert.deepEqual(sent, []);
});

test("picking a mode sends it; the server's answer decides what shows", () => {
  const rs = radios();
  const sent = [];
  const photos = initPhotos({ radios: rs }, (t) => sent.push(t));
  rs[1].checked = true;
  rs[1].fire();
  assert.deepEqual(sent, [photosMessage("followup")]);
  photos.show({ type: "photos", mode: "off" });
  assert.deepEqual(checked(rs), ["off"]);
});
