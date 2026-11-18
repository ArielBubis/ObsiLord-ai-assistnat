import { test } from "node:test";
import assert from "node:assert/strict";

import { editorRows, emptyStage, hasChanges, saveMessage, MAX_EDIT } from "../../web/memory_editor.js";

test("editorRows labels and text", () => {
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
  assert.equal(rows[0].turn, 0);
  assert.equal(rows[0].messages[0].id, 1);
  assert.equal(rows[0].messages[0].text, "q");
  assert.deepEqual(editorRows({ enabled: false }), []);
  assert.deepEqual(editorRows(null), []);
  assert.deepEqual(editorRows({ enabled: true, turns: "x" }), []);
});

const ROWS = [
  { turn: 1, messages: [{ id: 10, text: "q1" }, { id: 11, text: "a1" }] },
  { turn: 2, messages: [{ id: 12, text: "q2" }, { id: 13, text: "a2" }] },
  { turn: 3, messages: [{ id: 14, text: "q3" }, { id: 15, text: "a3" }] },
];

test("hasChanges", () => {
  assert.equal(hasChanges(emptyStage(), ROWS, "T"), false);
  const s = emptyStage();
  s.edits.set(11, "new");
  assert.equal(hasChanges(s, ROWS, "T"), true);
  const same = emptyStage();
  same.edits.set(11, "  a1  ");
  assert.equal(hasChanges(same, ROWS, "T"), false);
  const del = emptyStage();
  del.deletes.add(2);
  assert.equal(hasChanges(del, ROWS, "T"), true);
  const t = emptyStage();
  t.title = "Other";
  assert.equal(hasChanges(t, ROWS, "T"), true);
  t.title = " T ";
  assert.equal(hasChanges(t, ROWS, "T"), false);
  const add = emptyStage();
  add.adds.push({ user: "", reply: "" });
  assert.equal(hasChanges(add, ROWS, "T"), false);
});

test("saveMessage builds one save", () => {
  const s = emptyStage();
  s.edits.set(11, " new ");
  s.edits.set(13, "a2");
  s.deletes.add(2);
  s.adds.push({ user: " u ", reply: "r" });
  assert.deepEqual(JSON.parse(saveMessage("c", 3, s, ROWS, "T")), {
    type: "memory", action: "save", conversation: "c", expect_turns: 3,
    edits: [{ message: 11, text: "new" }], delete_turns: [2],
    add_turns: [{ user: "u", reply: "r" }],
  });
});

test("saveMessage title and refusals", () => {
  const t = emptyStage();
  t.title = " New ";
  assert.equal(JSON.parse(saveMessage("c", 3, t, ROWS, "T")).title, "New");
  const half = emptyStage();
  half.adds.push({ user: "u", reply: " " });
  assert.equal(saveMessage("c", 3, half, ROWS, "T"), null);
  assert.equal(saveMessage("c", 3, emptyStage(), ROWS, "T"), null);
  const big = emptyStage();
  big.edits.set(11, "x".repeat(MAX_EDIT + 1));
  assert.equal(saveMessage("c", 3, big, ROWS, "T"), null);
  const d = emptyStage();
  d.deletes.add(3);
  d.deletes.add(1);
  assert.deepEqual(JSON.parse(saveMessage("c", 3, d, ROWS, "T")).delete_turns, [1, 3]);
});

// A small DOM stand-in, enough for the editor to render and be clicked.
class FakeEl {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.listeners = {};
    this.attrs = {};
    this.disabled = false;
    this.hidden = false;
    this.value = "";
    this.textContent = "";
    this.className = "";
  }
  append(...c) { this.children.push(...c); }
  replaceChildren(...c) { this.children = c; }
  addEventListener(n, f) { this.listeners[n] = f; }
  setAttribute(k, v) { this.attrs[k] = v; }
  querySelector() { return null; }
  all() { return this.children.flatMap((c) => [c, ...c.all()]); }
  querySelectorAll(sel) { return sel === "button" ? this.all().filter((c) => c.tag === "button") : []; }
  texts() { return this.all().map((c) => c.textContent).filter(Boolean); }
}

// The Save / Revert / Keep editing prompt: three buttons found by data-act.
class FakeDialog extends FakeEl {
  constructor() {
    super("dialog");
    this.open = false;
    this.opened = 0;
    this.btn = {};
    for (const act of ["save", "revert", "keep"]) {
      const b = new FakeEl("button");
      b.attrs["data-act"] = act;
      this.btn[act] = b;
      this.children.push(b);
    }
  }
  querySelector(sel) {
    const m = /\[data-act="(\w+)"\]/.exec(sel);
    return m ? this.btn[m[1]] : null;
  }
  showModal() { this.open = true; this.opened += 1; }
  close() { this.open = false; }
  removeAttribute() { this.open = false; }
  click(act) { this.btn[act].listeners.click(); }
  escape() { this.listeners.cancel(); this.open = false; }
}

function bodyOf(title, ids) {
  return { enabled: true, title, turns: ids.map((id, i) => ({ turn: i + 1, messages: [{ id, role: "user", text: `q${id}` }] })) };
}

async function openedEditor(send, dialog = null) {
  globalThis.document = { createElement: (t) => new FakeEl(t) };
  const el = new FakeEl("div");
  const bodies = { c: bodyOf("T", [5]), other: bodyOf("O", [9]) };
  const { initMemoryEditor } = await import("../../web/memory_editor.js");
  const ed = initMemoryEditor(el, send, async (id) => bodies[id] || null, dialog);
  ed.open("c");
  await new Promise((r) => setTimeout(r, 0));
  return { el, ed };
}

function stageEdit(el) {
  const ta = el.all().find((c) => c.tag === "textarea");
  ta.value = "changed";
  ta.listeners.input();
}

const tick = () => new Promise((r) => setTimeout(r, 0));

test("Delete conversation with the socket closed keeps buttons on and the stage", async () => {
  const { el, ed } = await openedEditor(() => false);
  stageEdit(el);
  globalThis.confirm = () => true;
  el.all().find((c) => c.textContent === "Delete conversation").listeners.click();
  assert.ok(ed.isDirty());
  assert.ok(el.querySelectorAll("button").every((b) => !b.disabled));
  assert.ok(el.texts().some((t) => t.startsWith("Not connected")));
});

test("onDisconnect releases a request in flight and keeps the stage", async () => {
  const sent = [];
  const { el, ed } = await openedEditor((t) => { sent.push(t); return true; });
  el.all().find((c) => c.textContent === "Delete turn").listeners.click();
  // Delete conversation is the one request that goes out without the dialog.
  globalThis.confirm = () => true;
  el.all().find((c) => c.textContent === "Delete conversation").listeners.click();
  assert.equal(sent.length, 1);
  assert.ok(el.querySelectorAll("button").every((b) => b.disabled));
  ed.onDisconnect();
  assert.ok(el.querySelectorAll("button").every((b) => !b.disabled));
  assert.ok(ed.isDirty());
  assert.ok(el.texts().some((t) => t.startsWith("Not connected")));
});

test("Save sends one save and keeps the buttons off until a reply or error", async () => {
  const sent = [];
  const dlg = new FakeDialog();
  const { el, ed } = await openedEditor((t) => { sent.push(t); return true; }, dlg);
  stageEdit(el);
  ed.requestSave();
  assert.equal(dlg.opened, 1);
  dlg.click("save");
  assert.equal(sent.length, 1);
  const msg = JSON.parse(sent[0]);
  assert.equal(msg.type, "memory");
  assert.equal(msg.action, "save");
  assert.deepEqual(msg.edits, [{ message: 5, text: "changed" }]);
  assert.ok(el.querySelectorAll("button").every((b) => b.disabled));
  assert.ok(Object.values(dlg.btn).every((b) => b.disabled));
  ed.requestSave();
  assert.equal(dlg.opened, 1);
  ed.onMessage({ type: "memory", action: "save", conversation: "c" });
  assert.equal(ed.openId(), null);
});

test("Revert sends nothing, drops the stage and opens the pending conversation", async () => {
  const sent = [];
  const dlg = new FakeDialog();
  const { el, ed } = await openedEditor((t) => { sent.push(t); return true; }, dlg);
  stageEdit(el);
  ed.open("other");
  assert.equal(dlg.opened, 1);
  assert.equal(ed.openId(), "c");
  dlg.click("revert");
  await tick();
  assert.equal(sent.length, 0);
  assert.equal(ed.openId(), "other");
  assert.equal(ed.isDirty(), false);
});

test("Keep editing keeps the stage and the open conversation", async () => {
  const sent = [];
  const dlg = new FakeDialog();
  const { el, ed } = await openedEditor((t) => { sent.push(t); return true; }, dlg);
  stageEdit(el);
  ed.open("other");
  dlg.click("keep");
  assert.equal(sent.length, 0);
  assert.equal(ed.openId(), "c");
  assert.ok(ed.isDirty());
  assert.equal(dlg.open, false);
});

test("open of another conversation while dirty prompts and keeps the stage", async () => {
  const dlg = new FakeDialog();
  const { el, ed } = await openedEditor(() => true, dlg);
  stageEdit(el);
  ed.open("other");
  assert.equal(dlg.open, true);
  assert.equal(ed.openId(), "c");
  assert.ok(ed.isDirty());
});

test("Escape on the prompt clears the pending conversation", async () => {
  const sent = [];
  const dlg = new FakeDialog();
  const { el, ed } = await openedEditor((t) => { sent.push(t); return true; }, dlg);
  stageEdit(el);
  ed.open("other");
  dlg.escape();
  // A later plain Save must not jump to "other".
  ed.requestSave();
  dlg.click("save");
  ed.onMessage({ type: "memory", action: "save", conversation: "c" });
  await tick();
  assert.equal(ed.openId(), null);
});

test("an error while sending shows its text; while idle it changes nothing", async () => {
  const dlg = new FakeDialog();
  const { el, ed } = await openedEditor(() => true, dlg);
  stageEdit(el);
  ed.onError("Unrelated failure");
  assert.ok(!el.texts().some((t) => t === "Unrelated failure"));
  ed.requestSave();
  dlg.click("save");
  assert.ok(el.querySelectorAll("button").every((b) => b.disabled));
  ed.onError("This conversation changed since you opened it. Reopen it to edit.");
  assert.ok(el.texts().some((t) => t.includes("Reopen it to edit. Revert discards your changes.")));
  assert.ok(el.querySelectorAll("button").every((b) => !b.disabled));
  assert.ok(ed.isDirty());
});
