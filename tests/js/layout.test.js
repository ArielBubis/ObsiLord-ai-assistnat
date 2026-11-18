import { test } from "node:test";
import assert from "node:assert/strict";

import { faceWidth, sideReserve } from "../../web/layout.js";

test("faceWidth leaves room for the side panels and the card chrome", () => {
  assert.equal(faceWidth(1600, 760, 72), 768);
});

test("faceWidth uses the whole window when the panels are stacked", () => {
  assert.equal(faceWidth(900, 0, 72), 828);
});

test("faceWidth never goes negative", () => {
  assert.equal(faceWidth(500, 900, 72), 0);
});

test("sideReserve takes both panels and their gaps on a wide window", () => {
  assert.equal(sideReserve(1920), 752);
  assert.equal(sideReserve(1300), 752);
});

test("sideReserve is 0 once the columns stack", () => {
  assert.equal(sideReserve(1299), 0);
  assert.equal(sideReserve(900), 0);
});

test("sideReserve is 0 in demo mode at any width", () => {
  assert.equal(sideReserve(1920, undefined, undefined, undefined, true), 0);
  assert.equal(sideReserve(1300, undefined, undefined, undefined, true), 0);
});

test("sideReserve honours explicit arguments", () => {
  assert.equal(sideReserve(1000, 100, 10, 900), 220);
});
