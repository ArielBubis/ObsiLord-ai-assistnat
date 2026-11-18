import { test } from "node:test";
import assert from "node:assert/strict";

import {
  formatWhen, conversationLabel, conversationMessage, conversationNote, listItems, editLabel,
  reconnectMessage, MAX_TITLE,
} from "../../web/conversations.js";
import { checksView, MSG_NO_DEMO } from "../../web/memory_checks.js";

// Local time, so the text does not depend on the time zone.
const T = new Date(2026, 10, 17, 14, 2).getTime() / 1000;

test("formatWhen gives day, month and time", () => {
  assert.equal(formatWhen(T), "17 Nov 14:02");
  assert.equal(formatWhen(new Date(2026, 0, 5, 9, 7).getTime() / 1000), "5 Jan 09:07");
  assert.equal(formatWhen(null), "");
  assert.equal(formatWhen("x"), "");
});

test("conversationLabel has date, title and turns", () => {
  assert.equal(conversationLabel({ last_at: T, title: "about Lisbon", turns: 3 }), "17 Nov 14:02  about Lisbon (3 turns)");
  assert.equal(conversationLabel({ last_at: T, title: "", turns: 1 }), "17 Nov 14:02  (no title) (1 turn)");
  const long = conversationLabel({ last_at: T, title: "x".repeat(500), turns: 2 });
  assert.ok(long.includes("x".repeat(MAX_TITLE)) && !long.includes("x".repeat(MAX_TITLE + 1)));
});

test("conversationMessage", () => {
  assert.equal(conversationMessage("abc"), '{"type":"conversation","id":"abc"}');
  assert.equal(conversationMessage(null), '{"type":"conversation","id":null}');
  assert.equal(conversationMessage(undefined), '{"type":"conversation","id":null}');
});

test("conversationNote per reason", () => {
  const info = { type: "conversation", id: "a", turns: 12, last_at: T };
  assert.equal(conversationNote({ ...info, reason: "resume" }), "Continuing the conversation from 17 Nov 14:02 (12 turns).");
  assert.equal(conversationNote({ ...info, reason: "switch" }), "Continuing the conversation from 17 Nov 14:02 (12 turns).");
  assert.equal(conversationNote({ type: "conversation", id: null, reason: "new" }), "New conversation.");
  assert.equal(conversationNote({ type: "conversation", id: null, reason: "start" }), "New conversation.");
  assert.equal(conversationNote({ ...info, reason: "saved" }), null);
  assert.equal(conversationNote({ type: "other" }), null);
});

test("listItems marks the current one and skips bad entries", () => {
  const body = {
    enabled: true,
    conversations: [
      { id: "a", title: "one", last_at: T, turns: 1 },
      { id: 5, title: "bad" },
      null,
      { id: "b", title: "two", last_at: T, turns: 2 },
    ],
  };
  assert.deepEqual(listItems(body, "b"), [
    { id: "a", label: "17 Nov 14:02  one (1 turn)", current: false },
    { id: "b", label: "17 Nov 14:02  two (2 turns)", current: true },
  ]);
  assert.deepEqual(listItems({ enabled: false, conversations: [] }, null), []);
  assert.deepEqual(listItems(null, null), []);
});

test("reconnectMessage only after the server has said where the page is", () => {
  assert.equal(reconnectMessage({ known: false, id: null }), null);
  assert.equal(reconnectMessage({ known: true, id: null }), '{"type":"conversation","id":null}');
  assert.equal(reconnectMessage({ known: true, id: "a" }), '{"type":"conversation","id":"a"}');
});

test("checksView groups by label in order and caps text", () => {
  const view = checksView({
    checks: [
      { date: "2026-11-18", label: "today", question: "q1", expected: "e1", kind: "recall" },
      { date: "2026-11-17", label: "yesterday", question: "q2", expected: "e2", kind: "recall" },
      { date: "2026-11-18", label: "today", question: "q3", expected: "e3", kind: "recall" },
      { date: "", label: "any time", question: "y".repeat(900), expected: "e4", kind: "inline" },
      { bad: true },
    ],
  });
  assert.deepEqual(view.map((g) => [g.title, g.items.length]), [
    ["today (2026-11-18)", 2],
    ["yesterday (2026-11-17)", 1],
    ["any time", 1],
  ]);
  assert.equal(view[0].items[1].question, "q3");
  assert.equal(view[2].items[0].question.length, 300);
  assert.equal(view[2].items[0].kind, "inline");
  assert.deepEqual(checksView(null), []);
  assert.ok(MSG_NO_DEMO.includes("demo_memory = true"));
});

test("conversationNote treats edited like switch", () => {
  assert.equal(
    conversationNote({ type: "conversation", reason: "edited", last_at: T, turns: 2 }),
    conversationNote({ type: "conversation", reason: "switch", last_at: T, turns: 2 }),
  );
  assert.ok(conversationNote({ type: "conversation", reason: "edited", last_at: T, turns: 2 }).startsWith("Continuing"));
});

test("editLabel says Save for the conversation open in the editor", () => {
  assert.equal(editLabel("a", "a"), "Save");
  assert.equal(editLabel("a", "b"), "Edit");
  assert.equal(editLabel("a", null), "Edit");
});
