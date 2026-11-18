// Conversations in the Settings panel: the stored conversations (GET
// /conversations, newest first), a button to start a new one, and a line that
// says which one this page is in.
//
// The server says where the page is with {"type": "conversation", "id",
// "title", "turns", "last_at", "reason"}: on connect (it continues the last
// conversation when it is recent), after a switch, and after each stored turn
// (reason "saved", which also gives a new conversation its id). Clicking an
// entry sends {"type": "conversation", "id"}; "New conversation" sends id
// null. After a reconnect (not a reload) the page asks for the conversation
// it was in, so a dropped socket does not move it.
//
// The pure helpers at the top are tested with node --test.

export const MAX_TITLE = 80;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function pad(n) {
  return String(n).padStart(2, "0");
}

// "17 Nov 14:02" in local time from unix seconds, "" when not a number.
export function formatWhen(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "";
  const d = new Date(seconds * 1000);
  return `${d.getDate()} ${MONTHS[d.getMonth()]} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function turnsText(n) {
  const turns = Number.isInteger(n) && n >= 0 ? n : 0;
  return `${turns} ${turns === 1 ? "turn" : "turns"}`;
}

export function conversationLabel(c) {
  const title = typeof c.title === "string" && c.title ? c.title.slice(0, MAX_TITLE) : "(no title)";
  return `${formatWhen(c.last_at)}  ${title} (${turnsText(c.turns)})`;
}

export function conversationMessage(id) {
  return JSON.stringify({ type: "conversation", id: typeof id === "string" ? id : null });
}

// The line under the list for a conversation message, or null to keep it.
export function conversationNote(msg) {
  if (!msg || msg.type !== "conversation") return null;
  switch (msg.reason) {
    case "resume":
    case "switch":
    case "edited":
      return `Continuing the conversation from ${formatWhen(msg.last_at)} (${turnsText(msg.turns)}).`;
    case "new":
    case "start":
      return "New conversation.";
    default:
      return null;
  }
}

// "Save" for the conversation open in the editor, "Edit" for the others.
export function editLabel(id, openId) {
  return typeof id === "string" && id === openId ? "Save" : "Edit";
}

// [{ id, label, current }] from the /conversations body.
export function listItems(body, currentId) {
  if (!body || body.enabled !== true || !Array.isArray(body.conversations)) return [];
  return body.conversations
    .filter((c) => c && typeof c.id === "string")
    .map((c) => ({ id: c.id, label: conversationLabel(c), current: c.id === currentId }));
}

// What to send when a socket opens: nothing before the server has said where
// the page is (a fresh page lets the server continue a recent one).
export function reconnectMessage(state) {
  return state && state.known ? conversationMessage(state.id) : null;
}

// el: { block, list, newBtn, note }. send(text) sends if the socket is open.
// fetchList() resolves to the /conversations JSON.
// onEdit(id), when given, adds an Edit button to each entry (labelled Save for
// the one editOpenId() names; the caller decides what the click does).
// Returns { onMessage, announce, setDisabled, refresh, rerender }.
export function initConversations(el, send, fetchList, onEdit, editOpenId) {
  const state = { known: false, id: null };
  let body = null;
  let disabled = false;

  function render() {
    el.newBtn.disabled = disabled;
    el.list.replaceChildren(
      ...listItems(body, state.id).map((item) => {
        const li = document.createElement("li");
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = item.label;
        btn.disabled = disabled || item.current;
        if (item.current) {
          btn.classList.add("current");
          btn.setAttribute("aria-current", "true");
        }
        btn.addEventListener("click", () => {
          if (!disabled) send(conversationMessage(item.id));
        });
        li.appendChild(btn);
        if (typeof onEdit === "function") {
          const edit = document.createElement("button");
          edit.type = "button";
          edit.className = "conv-edit";
          edit.textContent = editLabel(item.id, typeof editOpenId === "function" ? editOpenId() : null);
          edit.addEventListener("click", () => onEdit(item.id));
          li.appendChild(edit);
        }
        return li;
      }),
    );
  }

  async function load() {
    try {
      body = await fetchList();
    } catch {
      body = null;
    }
    if (body && body.enabled === false) {
      el.block.hidden = true;
      return;
    }
    el.block.hidden = false;
    render();
  }

  el.newBtn.addEventListener("click", () => {
    if (!disabled) send(conversationMessage(null));
  });

  load();

  return {
    onMessage(msg) {
      if (!msg || msg.type !== "conversation") return;
      state.known = true;
      state.id = typeof msg.id === "string" ? msg.id : null;
      const note = conversationNote(msg);
      if (note !== null) el.note.textContent = note;
      load();
    },
    announce() {
      const msg = reconnectMessage(state);
      if (msg !== null) send(msg);
    },
    refresh() {
      load();
    },
    rerender() {
      render();
    },
    setDisabled(value) {
      disabled = Boolean(value);
      render();
    },
  };
}
