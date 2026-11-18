// Editor for one stored conversation, in Settings > Conversations: rename it,
// edit or delete its turns, add a turn by hand, delete the whole conversation.
//
// The page reads the turns with GET /conversations/{id} and changes them with
// {"type": "memory", "action", ...} over the WebSocket. The server answers a
// change with {"type": "memory", "action", "conversation"}; the editor then
// reads the turns again. Stored text only goes into .value and textContent,
// never innerHTML.
//
// The pure helpers at the top are tested with node --test.

export const MAX_EDIT = 4000;
export const MAX_TITLE_EDIT = 80;

function cleanText(value, max) {
  if (typeof value !== "string") return null;
  const text = value.trim();
  return text && text.length <= max ? text : null;
}

function isId(value) {
  return typeof value === "string" && value !== "";
}

// JSON text for the WebSocket, or null when a field is missing, has the wrong
// type, is empty after trimming, or is too long.
export function memoryMessage(action, fields) {
  const f = fields || {};
  let body;
  switch (action) {
    case "edit": {
      const text = cleanText(f.text, MAX_EDIT);
      if (!Number.isInteger(f.message) || text === null) return null;
      body = { message: f.message, text };
      break;
    }
    case "delete_turn":
      if (!isId(f.conversation) || !Number.isInteger(f.turn)) return null;
      body = { conversation: f.conversation, turn: f.turn };
      break;
    case "delete_conversation":
      if (!isId(f.conversation)) return null;
      body = { conversation: f.conversation };
      break;
    case "rename": {
      const title = cleanText(f.title, MAX_TITLE_EDIT);
      if (!isId(f.conversation) || title === null) return null;
      body = { conversation: f.conversation, title };
      break;
    }
    case "add_turn": {
      const user = cleanText(f.user, MAX_EDIT);
      const reply = cleanText(f.reply, MAX_EDIT);
      if (!isId(f.conversation) || user === null || reply === null) return null;
      body = { conversation: f.conversation, user, reply };
      break;
    }
    default:
      return null;
  }
  return JSON.stringify({ type: "memory", action, ...body });
}

function roleLabel(m) {
  if (m.role === "user") return "You";
  if (m.role === "assistant") return "Obsi-Lord";
  const name = typeof m.tool_name === "string" && m.tool_name ? m.tool_name : "tool";
  return `Tool: ${name}`;
}

// [{ turn, when, messages: [{ id, role, label, text, editable }] }] from the
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
          editable: true,
        })),
    }));
}

// What each message field shows after a re-read. drafts: Map of message id to
// unsaved text. A draft stays while it differs from the stored text (compared
// trimmed, since the server stores trimmed text); it is dropped once the save
// is confirmed (same text) or the message is gone.
// Returns { shown: Map id -> text, drafts: Map of the drafts still unsaved }.
export function mergeDrafts(rows, drafts) {
  const shown = new Map();
  const kept = new Map();
  const old = drafts instanceof Map ? drafts : new Map();
  for (const row of rows) {
    for (const m of row.messages) {
      if (old.has(m.id) && old.get(m.id).trim() !== m.text.trim()) {
        shown.set(m.id, old.get(m.id));
        kept.set(m.id, old.get(m.id));
      } else {
        shown.set(m.id, m.text);
      }
    }
  }
  return { shown, drafts: kept };
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
// send(text) sends on the socket. fetchTurns(id) resolves to the GET body, or
// null when the conversation is gone.
export function initMemoryEditor(el, send, fetchTurns) {
  let openId = null;
  let seq = 0;
  // Unsaved text, kept across re-renders: per message id, the title, the add form.
  let drafts = new Map();
  let titleDraft = null;
  let addDraft = { user: "", reply: "" };

  function close() {
    openId = null;
    drafts = new Map();
    titleDraft = null;
    addDraft = { user: "", reply: "" };
    seq += 1;
    el.hidden = true;
    el.replaceChildren();
  }

  function act(action, fields) {
    const text = memoryMessage(action, { conversation: openId, ...fields });
    if (text === null) return;
    // One change at a time: the buttons stay off until the answer re-renders
    // the editor or an error comes back (onError). Stops a double click from
    // sending a position-based change twice.
    setButtons(true);
    send(text);
  }

  function setButtons(disabled) {
    for (const b of el.querySelectorAll("button")) b.disabled = disabled;
  }

  function render(body) {
    const parts = [];
    const rows = editorRows(body);
    const merged = mergeDrafts(rows, drafts);
    drafts = merged.drafts;
    const stored = typeof body.title === "string" ? body.title : "";
    if (titleDraft !== null && titleDraft.trim() === stored.trim()) titleDraft = null;

    const head = document.createElement("div");
    head.className = "mem-head";
    const title = field("Title", titleDraft !== null ? titleDraft : stored, false);
    title.input.maxLength = MAX_TITLE_EDIT;
    title.input.addEventListener("input", () => {
      titleDraft = title.input.value;
    });
    head.append(
      title.wrap,
      button("Rename", () => act("rename", { title: title.input.value })),
      button("Close", close),
    );
    parts.push(head);

    for (const row of rows) {
      const box = document.createElement("div");
      box.className = "mem-turn";
      for (const m of row.messages) {
        const f = field(m.label, merged.shown.get(m.id), true);
        f.input.maxLength = MAX_EDIT;
        f.input.addEventListener("input", () => {
          drafts.set(m.id, f.input.value);
        });
        f.input.readOnly = !m.editable;
        const save = button("Save", () => act("edit", { message: m.id, text: f.input.value }));
        save.disabled = !m.editable;
        box.append(f.wrap, save);
      }
      box.append(
        button("Delete turn", () => {
          if (confirm("Delete this turn? This cannot be undone.")) {
            act("delete_turn", { turn: row.turn });
          }
        }),
      );
      parts.push(box);
    }

    const add = document.createElement("div");
    add.className = "mem-add";
    const user = field("Add turn: your message", addDraft.user, true);
    const reply = field("Add turn: his reply", addDraft.reply, true);
    user.input.addEventListener("input", () => {
      addDraft.user = user.input.value;
    });
    reply.input.addEventListener("input", () => {
      addDraft.reply = reply.input.value;
    });
    user.input.maxLength = MAX_EDIT;
    reply.input.maxLength = MAX_EDIT;
    add.append(
      user.wrap,
      reply.wrap,
      button("Add turn", () => act("add_turn", { user: user.input.value, reply: reply.input.value })),
    );
    parts.push(add);

    parts.push(
      button("Delete conversation", () => {
        if (confirm("Delete this whole conversation? This cannot be undone.")) {
          act("delete_conversation", {});
        }
      }),
    );

    el.replaceChildren(...parts);
    el.hidden = false;
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
    render(body);
  }

  el.hidden = true;

  return {
    open(id) {
      if (!isId(id)) return;
      if (id !== openId) {
        drafts = new Map();
        titleDraft = null;
        addDraft = { user: "", reply: "" };
      }
      openId = id;
      load();
    },
    // The server refused or failed a change: let the user try again.
    onError() {
      if (openId !== null) setButtons(false);
    },
    onMessage(msg) {
      if (!msg || msg.type !== "memory" || openId === null) return;
      if (msg.conversation !== openId) return;
      if (msg.action === "delete_conversation") {
        close();
        return;
      }
      if (msg.action === "add_turn") addDraft = { user: "", reply: "" };
      load();
    },
  };
}
