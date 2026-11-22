import { test } from "node:test";
import assert from "node:assert/strict";

import {
  rowsFromMessage, modeMessage, cloudMessage, activityRows, CLOUD_LINE, CLOUD_TIP,
} from "../../web/permissions.js";

const tools = [
  { name: "look", label: "Look", mode: "ask", internet: false, private: true, cloud: false, available: true, missing: "", note: "Uses the chat model." },
  { name: "search_image", label: "Image search", mode: "on", internet: true, private: false, cloud: false, available: false, missing: "Needs the [vision] files.", note: "" },
  { name: "recall", label: "Recall", mode: "on", internet: false, private: true, cloud: true, available: true, missing: "" },
  { name: "calculate", label: "Calculator", mode: "off", internet: false, private: false, cloud: false, available: true, missing: "", note: "" },
];
const msg = { type: "permissions", tools };

test("rows follow the server order and badge by internet", () => {
  const rows = rowsFromMessage(msg);
  assert.deepEqual(rows.map((r) => r.name), ["look", "search_image", "recall", "calculate"]);
  assert.deepEqual(rows.map((r) => r.badge), ["local", "internet", "local", "local"]);
});

test("an available tool has Off, Ask, On with the current one active", () => {
  const look = rowsFromMessage(msg)[0];
  assert.deepEqual(look.modes.map((m) => m.mode), ["off", "ask", "on"]);
  assert.deepEqual(look.modes.filter((m) => m.active).map((m) => m.mode), ["ask"]);
});

test("a tool that is not set up shows its missing text and has no buttons or cloud switch", () => {
  const row = rowsFromMessage(msg)[1];
  assert.equal(row.available, false);
  assert.equal(row.missing, "Needs the [vision] files.");
  assert.deepEqual(row.modes, []);
  assert.equal(row.showCloud, false);
});

test("the cloud switch is only for private-data tools", () => {
  const rows = rowsFromMessage(msg);
  assert.deepEqual(rows.map((r) => r.showCloud), [true, false, true, false]);
  assert.equal(rows[2].cloud, true);
  assert.equal(rows[0].cloud, false);
  assert.equal(rows[0].cloudLine, CLOUD_LINE);
  assert.equal(CLOUD_LINE, "When on, its results go to Google.");
});

test("the note shows only when non-empty; a missing note is fine", () => {
  const rows = rowsFromMessage(msg);
  assert.equal(rows[0].note, "Uses the chat model.");
  assert.equal(rows[1].note, "");
  assert.equal(rows[2].note, "");
});

test("bad messages give no rows; bad entries are skipped; bad modes read as off", () => {
  for (const bad of [null, {}, { type: "state" }, { type: "permissions" }, { type: "permissions", tools: "x" }]) {
    assert.equal(rowsFromMessage(bad), null);
  }
  const rows = rowsFromMessage({ type: "permissions", tools: [null, { label: "x" }, { name: "t", mode: "weird" }] });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].mode, "off");
});

test("a mode click sends a permission message", () => {
  assert.deepEqual(JSON.parse(modeMessage("look", "on")), { type: "permission", tool: "look", mode: "on" });
  assert.equal(modeMessage("look", "always"), null);
  assert.equal(modeMessage("", "on"), null);
});

test("the cloud switch sends a permission message with cloud", () => {
  assert.deepEqual(JSON.parse(cloudMessage("recall", true)), { type: "permission", tool: "recall", cloud: true });
  assert.deepEqual(JSON.parse(cloudMessage("recall", false)), { type: "permission", tool: "recall", cloud: false });
  assert.equal(cloudMessage("recall", "yes"), null);
});

test("activity rows keep the order, show the time of day and cap at 50", () => {
  const entries = [
    { ts: "2026-11-19T14:03:05+02:00", tool: "look", mode: "ask", outcome: "ran" },
    { ts: "2026-11-19T14:01:00+02:00", tool: "recall", mode: "on", outcome: "declined" },
    { tool: 5 },
  ];
  const rows = activityRows({ entries });
  assert.deepEqual(rows.map((r) => [r.time, r.tool, r.outcome]), [
    ["14:03:05", "look", "ran"],
    ["14:01:00", "recall", "declined"],
  ]);
  assert.equal(activityRows({ entries: Array.from({ length: 80 }, () => entries[0]) }).length, 50);
  assert.deepEqual(activityRows(null), []);
  assert.deepEqual(activityRows({ entries: "x" }), []);
});

test("the cloud switch carries a tooltip that says it is about Gemini", () => {
  const rows = rowsFromMessage(msg);
  assert.equal(rows[0].cloudTip, CLOUD_TIP);
  assert.match(CLOUD_TIP, /Gemini/);
  assert.match(CLOUD_TIP, /never the photo/);
  assert.match(CLOUD_TIP, /local model/);
});
