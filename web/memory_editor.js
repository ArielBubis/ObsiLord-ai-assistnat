// Editor for one stored conversation, in Settings > Conversations: rename it,
// edit or delete its turns, add turns by hand, delete the whole conversation.
//
// The page reads the turns with GET /conversations/{id}. Every change is
// staged in the page (edits, deleted turns, added turns, a new title) and goes
// to the server as one {"type": "memory", "action": "save"} after the user
// answers the Save / Revert / Keep editing prompt, so a save applies all of it
// or none. Delete conversation is separate and immediate (its own confirm).
// Stored text only goes into .value and textContent, never innerHTML.
//
// The pure helpers are tested with node --test.

export const MAX_EDIT = 4000;
export const MAX_TITLE_EDIT = 80;

function isId(value) {
  return typeof value === "string" && value !== "";
}

export function emptyStage() {
  return { title: null, edits: new Map(), deletes: new Set(), adds: [] };
}

function storedTexts(rows) {
  const stored = new Map();
  for (const row of rows) for (const m of row.messages) stored.set(m.id, m.text.trim());
  return stored;
}

// [[message id, trimmed text]] for the edits that differ from the stored text.
function changedEdits(stage, rows) {
  const stored = storedTexts(rows);
  const out = [];
  for (const [id, text] of stage.edits) {
    if (stored.has(id) && text.trim() !== stored.get(id)) out.push([id, text.trim()]);
  }
  return out;
}

// A blank title keeps the current one, so it is not a change.
function titleChanged(stage, storedTitle) {
  if (typeof stage.title !== "string") return false;
  const t = stage.title.trim();
  return t !== "" && t !== String(storedTitle || "").trim();
}

function addHasText(a) {
  return a.user.trim() !== "" || a.reply.trim() !== "";
}

export function hasChanges(stage, rows, storedTitle) {
  return (
    changedEdits(stage, rows).length > 0 ||
    stage.deletes.size > 0 ||
    stage.adds.some(addHasText) ||
    titleChanged(stage, storedTitle)
  );
}

// JSON text for the WebSocket, or null when nothing changed, an add is half
// filled, or a text is empty or over the caps.
export function saveMessage(conversation, expectTurns, stage, rows, storedTitle) {
  if (!isId(conversation) || !Number.isInteger(expectTurns) || expectTurns < 1) return null;
  if (!hasChanges(stage, rows, storedTitle)) return null;
  const turnOf = new Map();
  for (const row of rows) for (const m of row.messages) turnOf.set(m.id, row.turn);
  const edits = [];
  for (const [id, text] of changedEdits(stage, rows)) {
    if (stage.deletes.has(turnOf.get(id))) continue;
    if (text === "" || text.length > MAX_EDIT) return null;
    edits.push({ message: id, text });
  }
  const adds = [];
  for (const a of stage.adds) {
    if (!addHasText(a)) continue;
    const user = a.user.trim();
    const reply = a.reply.trim();
    if (!user || !reply || user.length > MAX_EDIT || reply.length > MAX_EDIT) return null;
    adds.push({ user, reply });
  }
  const body = { type: "memory", action: "save", conversation, expect_turns: expectTurns };
  if (titleChanged(stage, storedTitle)) {
    const title = stage.title.trim();
    if (title.length > MAX_TITLE_EDIT) return null;
    body.title = title;
  }
  body.edits = edits;
  body.delete_turns = [...stage.deletes].sort((x, y) => x - y);
  body.add_turns = adds;
  return JSON.stringify(body);
}

function roleLabel(m) {
  if (m.role === "user") return "You";
  if (m.role === "assistant") return "Obsi-Lord";
  const name = typeof m.tool_name === "string" && m.tool_name ? m.tool_name : "tool";
  return `Tool: ${name}`;
}

// [{ turn, when, messages: [{ id, role, label, text }] }] from the
// GET body; [] for a disabled or malformed one.
export function editorRows(body) {
  if (!body || body.enabled !== true || !Array.isArray(body.turns)) return [];
  return body.turns
    .filter((t) => t && Number.isInteger(t.turn) && Array.isArray(t.messages))
    .map((t) => ({
      turn: t.turn,
      when: typeof t.ts === "number" ? t.ts : null,
      messages: t.messages
        .filter((m) => m && Number.isInteger(m.id))
        .map((m) => ({
          id: m.id,
          role: m.role,
          label: roleLabel(m),
          text: typeof m.text === "string" ? m.text : "",
        })),
    }));
}

function button(text, onClick) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = text;
  b.addEventListener("click", onClick);
  return b;
}

function field(labelText, value, multiline) {
  const wrap = document.createElement("label");
  wrap.className = "mem-field";
  const span = document.createElement("span");
  span.textContent = labelText;
  const input = document.createElement(multiline ? "textarea" : "input");
  if (multiline) input.rows = 2;
  input.value = value;
  wrap.append(span, input);
  return { wrap, input };
}

// el: the container element (hidden until a conversation is opened).
// send(text) sends on the socket and returns false when it could not.
// fetchTurns(id) resolves to the GET body, or null when the conversation is gone. confirmDialog: the <dialog> with buttons
// marked data-act="save" | "revert" | "keep". onChange(), when given, runs when
// the open conversation changes (the list relabels its button).
const STALE_PREFIX = "This conversation changed since you opened it.";
const NOT_CONNECTED = "Not connected. Your changes are kept; try Save again when the link is back.";

export function initMemoryEditor(el, send, fetchTurns, confirmDialog, onChange) {
  let openId = null;
  let seq = 0;
  let rows = [];
  let storedTitle = "";
  let stage = emptyStage();
  let addDraft = { user: "", reply: "" };
  let note = "";
  // The conversation to open once the prompt is answered (Edit clicked on
  // another one while this one has changes).
  let pending = null;
  // A save or delete was sent and no answer has come yet.
  let sending = false;

  function changed() {
    if (typeof onChange === "function") onChange();
  }

  function reset() {
    stage = emptyStage();
    addDraft = { user: "", reply: "" };
    note = "";
    rows = [];
    storedTitle = "";
    pending = null;
    sending = false;
  }

  function hidePrompt() {
    if (!confirmDialog) return;
    if (typeof confirmDialog.close === "function" && confirmDialog.open) confirmDialog.close();
    else confirmDialog.removeAttribute("open");
  }

  function showPrompt() {
    if (!confirmDialog) return;
    if (typeof confirmDialog.showModal === "function") {
      if (!confirmDialog.open) confirmDialog.showModal();
    } else {
      confirmDialog.setAttribute("open", "");
    }
  }

  function close() {
    openId = null;
    reset();
    seq += 1;
    el.hidden = true;
    el.replaceChildren();
    hidePrompt();
    changed();
  }

  function dirty() {
    return openId !== null && (hasChanges(stage, rows, storedTitle) || addHasText(addDraft));
  }

  function setButtons(disabled) {
    sending = disabled;
    for (const b of el.querySelectorAll("button")) b.disabled = disabled;
    if (confirmDialog) for (const b of confirmDialog.querySelectorAll("button")) b.disabled = disabled;
  }

  // Move a filled add form into the staged adds. false when it is half filled.
  function commitDraft() {
    if (!addHasText(addDraft)) return true;
    if (!addDraft.user.trim() || !addDraft.reply.trim()) return false;
    stage.adds.push({ user: addDraft.user, reply: addDraft.reply });
    addDraft = { user: "", reply: "" };
    return true;
  }

  function doSave() {
    if (openId === null || sending) return;
    if (!commitDraft()) {
      note = "Fill both boxes of the turn you are adding, or clear them.";
      render();
      return;
    }
    if (!hasChanges(stage, rows, storedTitle)) {
      const next = pending;
      close();
      if (next !== null) api.open(next);
      return;
    }
    const text = saveMessage(openId, rows.length, stage, rows, storedTitle);
    if (text === null) {
      note = `Each changed message needs text (under ${MAX_EDIT} characters), and each added turn needs both boxes.`;
      pending = null;
      render();
      return;
    }
    note = "";
    // The buttons stay off until the answer closes the editor or an error
    // comes back (onError).
    trySend(text);
  }

  // Send and lock the buttons until the answer. send() returns false when the
  // socket is not open: then nothing is locked, the stage stays, a note says so.
  function trySend(text) {
    setButtons(true);
    if (send(text) === false) {
      setButtons(false);
      note = NOT_CONNECTED;
      pending = null;
      render();
      return false;
    }
    return true;
  }

  function revert() {
    const next = pending;
    close();
    if (next !== null) api.open(next);
  }

  function requestSave() {
    if (openId !== null && !sending) showPrompt();
  }

  function render() {
    const parts = [];
    const head = document.createElement("div");
    head.className = "mem-head";
    const title = field("Title", stage.title !== null ? stage.title : storedTitle, false);
    title.input.maxLength = MAX_TITLE_EDIT;
    title.input.addEventListener("input", () => {
      stage.title = title.input.value;
    });
    head.append(
      title.wrap,
      button("Save", requestSave),
      button("Close", () => {
        if (dirty()) requestSave();
        else close();
      }),
    );
    parts.push(head);

    if (note) {
      const p = document.createElement("p");
      p.className = "mem-note";
      p.setAttribute("role", "alert");
      p.textContent = note;
      parts.push(p);
    }

    for (const row of rows) {
      const gone = stage.deletes.has(row.turn);
      const box = document.createElement("div");
      box.className = gone ? "mem-turn mem-deleted" : "mem-turn";
      for (const m of row.messages) {
        const shown = stage.edits.has(m.id) ? stage.edits.get(m.id) : m.text;
        const f = field(m.label, shown, true);
        f.input.maxLength = MAX_EDIT;
        f.input.readOnly = gone;
        f.input.addEventListener("input", () => {
          stage.edits.set(m.id, f.input.value);
        });
        box.append(f.wrap);
      }
      box.append(
        gone
          ? button("Undo", () => {
              stage.deletes.delete(row.turn);
              render();
            })
          : button("Delete turn", () => {
              stage.deletes.add(row.turn);
              render();
            }),
      );
      parts.push(box);
    }

    const add = document.createElement("div");
    add.className = "mem-add";
    const user = field("Add turn: your message", addDraft.user, true);
    const reply = field("Add turn: his reply", addDraft.reply, true);
    user.input.maxLength = MAX_EDIT;
    reply.input.maxLength = MAX_EDIT;
    user.input.addEventListener("input", () => {
      addDraft.user = user.input.value;
    });
    reply.input.addEventListener("input", () => {
      addDraft.reply = reply.input.value;
    });
    add.append(
      user.wrap,
      reply.wrap,
      button("Add turn", () => {
        if (!addDraft.user.trim() || !addDraft.reply.trim()) {
          note = "Fill both boxes to add a turn.";
        } else {
          commitDraft();
          note = "";
        }
        render();
      }),
    );
    parts.push(add);

    stage.adds.forEach((a, i) => {
      const item = document.createElement("div");
      item.className = "mem-added";
      const text = document.createElement("p");
      text.textContent = `Added: ${a.user.trim()} / ${a.reply.trim()}`;
      item.append(
        text,
        button("Remove", () => {
          stage.adds.splice(i, 1);
          render();
        }),
      );
      parts.push(item);
    });

    parts.push(
      button("Delete conversation", () => {
        if (confirm("Delete this whole conversation? This cannot be undone.")) {
          trySend(JSON.stringify({ type: "memory", action: "delete_conversation", conversation: openId }));
        }
      }),
    );

    el.replaceChildren(...parts);
    el.hidden = false;
    setButtons(sending);
  }

  async function load() {
    const id = openId;
    const mine = ++seq;
    let body = null;
    try {
      body = await fetchTurns(id);
    } catch {
      body = null;
    }
    if (mine !== seq || id !== openId) return;
    if (!body || body.enabled !== true) {
      close();
      return;
    }
    rows = editorRows(body);
    storedTitle = typeof body.title === "string" ? body.title : "";
    render();
    changed();
  }

  if (confirmDialog) {
    const on = (act, fn) => {
      const b = confirmDialog.querySelector(`[data-act="${act}"]`);
      if (b) b.addEventListener("click", fn);
    };
    on("save", () => {
      hidePrompt();
      doSave();
    });
    on("revert", () => {
      hidePrompt();
      revert();
    });
    const keep = () => {
      pending = null;
      hidePrompt();
    };
    on("keep", keep);
    // Escape closes the dialog: same as Keep editing.
    confirmDialog.addEventListener("cancel", () => {
      pending = null;
    });
  }

  el.hidden = true;

  const api = {
    open(id) {
      if (!isId(id)) return;
      if (openId !== null && id !== openId && dirty()) {
        pending = id;
        showPrompt();
        return;
      }
      if (id !== openId) {
        reset();
        openId = id;
        changed();
      }
      load();
    },
    isOpen(id) {
      return openId !== null && (id === undefined || id === openId);
    },
    isDirty() {
      return dirty();
    },
    openId() {
      return openId;
    },
    requestSave,
    // The server refused or failed the save or delete: keep the stage, show
    // the reason, allow a retry. An error with nothing in flight is not ours.
    onError(message) {
      if (openId === null || !sending) return;
      pending = null;
      note = typeof message === "string" && message ? message : "The server refused the change.";
      // The stale refusal (app.py MSG_MEMORY_STALE) says to reopen; the stage
      // is still here, so say how to get out of it.
      if (note.startsWith(STALE_PREFIX)) note += " Revert discards your changes.";
      setButtons(false);
      render();
    },
    // The socket closed: release an unanswered save, keep the stage.
    onDisconnect() {
      if (openId === null) return;
      pending = null;
      if (sending) {
        note = NOT_CONNECTED;
        sending = false;
        render();
      }
    },
    onMessage(msg) {
      if (!msg || msg.type !== "memory" || openId === null) return;
      if (msg.conversation !== openId) return;
      if (msg.action === "delete_conversation") {
        close();
        return;
      }
      if (msg.action === "save") {
        const next = pending;
        close();
        if (next !== null) api.open(next);
      }
    },
  };
  return api;
}
