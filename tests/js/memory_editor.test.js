import { test } from "node:test";
import assert from "node:assert/strict";

import { memoryMessage, editorRows, mergeDrafts, MAX_EDIT } from "../../web/memory_editor.js";

test("memoryMessage edit trims and validates", () => {
  const ok = JSON.parse(memoryMessage("edit", { message: 3, text: "  hi " }));
  assert.deepEqual(ok, { type: "memory", action: "edit", message: 3, text: "hi" });
  assert.equal(memoryMessage("edit", { message: 3, text: "" }), null);
  assert.equal(memoryMessage("edit", { message: 3, text: "   " }), null);
  assert.equal(memoryMessage("edit", { message: 3, text: "x".repeat(MAX_EDIT + 1) }), null);
  assert.notEqual(memoryMessage("edit", { message: 3, text: "x".repeat(MAX_EDIT) }), null);
  assert.equal(memoryMessage("edit", { message: "3", text: "hi" }), null);
  assert.equal(memoryMessage("bogus", { message: 3, text: "hi" }), null);
});

test("memoryMessage other actions", () => {
  assert.deepEqual(JSON.parse(memoryMessage("delete_turn", { conversation: "c", turn: 2 })),
    { type: "memory", action: "delete_turn", conversation: "c", turn: 2 });
  assert.equal(memoryMessage("delete_turn", { conversation: "c" }), null);
  assert.deepEqual(JSON.parse(memoryMessage("delete_conversation", { conversation: "c" })),
    { type: "memory", action: "delete_conversation", conversation: "c" });
  assert.equal(memoryMessage("delete_conversation", {}), null);
  assert.deepEqual(JSON.parse(memoryMessage("rename", { conversation: "c", title: " T " })),
    { type: "memory", action: "rename", conversation: "c", title: "T" });
  assert.equal(memoryMessage("rename", { conversation: "c", title: "" }), null);
  assert.deepEqual(JSON.parse(memoryMessage("add_turn", { conversation: "c", user: " a ", reply: "b" })),
    { type: "memory", action: "add_turn", conversation: "c", user: "a", reply: "b" });
  assert.equal(memoryMessage("add_turn", { conversation: "c", user: "a", reply: "" }), null);
});

test("editorRows labels and editable", () => {
  const body = {
    enabled: true, id: "c", title: "t",
    turns: [
      { turn: 0, ts: 1000, messages: [
        { id: 1, role: "user", text: "q", tool_name: null, tool_arg: null },
        { id: 2, role: "tool", text: "r", tool_name: "look", tool_arg: "x" },
        { id: 3, role: "assistant", text: "a", tool_name: null, tool_arg: null },
      ] },
      { turn: 1, ts: 2000, messages: [
        { id: 4, role: "user", text: "q2" },
        { id: 5, role: "assistant", text: "a2" },
      ] },
    ],
  };
  const rows = editorRows(body);
  assert.equal(rows.length, 2);
  assert.deepEqual(rows[0].messages.map((m) => m.label), ["You", "Tool: look", "Obsi-Lord"]);
  assert.ok(rows.every((r) => r.messages.every((m) => m.editable)));
  assert.equal(rows[0].turn, 0);
  assert.equal(rows[0].messages[0].id, 1);
  assert.equal(rows[0].messages[0].text, "q");
  assert.deepEqual(editorRows({ enabled: false }), []);
  assert.deepEqual(editorRows(null), []);
  assert.deepEqual(editorRows({ enabled: true, turns: "x" }), []);
});

test("mergeDrafts keeps unsaved text, drops confirmed and vanished ones", () => {
  const rows = [
    { turn: 0, messages: [{ id: 1, text: "one" }, { id: 2, text: "two" }, { id: 3, text: "hi" }] },
  ];
  const drafts = new Map([[1, "one edited"], [2, "two"], [3, "  hi "], [9, "gone"]]);
  const { shown, drafts: kept } = mergeDrafts(rows, drafts);
  assert.equal(shown.get(1), "one edited");
  assert.equal(shown.get(2), "two");
  assert.equal(shown.get(3), "hi");
  assert.deepEqual([...kept.entries()], [[1, "one edited"]]);
  assert.equal(mergeDrafts(rows, new Map()).shown.get(1), "one");
});
