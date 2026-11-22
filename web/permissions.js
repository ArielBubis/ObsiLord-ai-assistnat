// Settings, Permissions and Activity. The server owns the modes: it sends
// {"type": "permissions", "tools": [...]} on connect and after every change
// (from any page), and the page only sends a change when the user clicks.
// Nothing is kept in this browser.
//
// The pure helpers at the top are tested with node --test; the DOM code at
// the bottom is thin.

export const MODES = ["off", "ask", "on"];
export const MODE_LABELS = { off: "Off", ask: "Ask", on: "On" };
export const CLOUD_LINE = "When on, its results go to Google.";
// The tooltip of the cloud switch (user, 2026-11-22: keep the short text,
// explain it on hover).
export const CLOUD_TIP =
  "Only matters when you chat with Gemini (Google's cloud model). On: Gemini may use this tool, " +
  "and what it finds is sent to Google with the conversation (for the camera: Qwenny's description, " +
  "never the photo). Off: Gemini is not offered this tool. No effect with a local model.";
export const MAX_TEXT = 300;
export const MAX_ACTIVITY = 50;

function text(value) {
  return typeof value === "string" ? value.slice(0, MAX_TEXT) : "";
}

// One row per tool from a server permissions message, in the server's order.
// Null if the message is not a permissions message. A tool that is not set
// up (available false) shows its missing text and has no mode buttons and no
// cloud switch. The cloud switch is only for private-data tools. The note
// is optional: missing or empty shows nothing.
export function rowsFromMessage(msg) {
  if (!msg || typeof msg !== "object" || msg.type !== "permissions" || !Array.isArray(msg.tools)) {
    return null;
  }
  const rows = [];
  for (const tool of msg.tools) {
    if (!tool || typeof tool !== "object" || typeof tool.name !== "string" || !tool.name) continue;
    const available = tool.available !== false;
    const mode = MODES.includes(tool.mode) ? tool.mode : "off";
    rows.push({
      name: tool.name,
      label: text(tool.label) || tool.name,
      badge: tool.internet === true ? "internet" : "local",
      available,
      missing: available ? "" : text(tool.missing),
      note: text(tool.note),
      mode,
      modes: available ? MODES.map((m) => ({ mode: m, label: MODE_LABELS[m], active: m === mode })) : [],
      showCloud: available && tool.private === true,
      cloud: tool.cloud === true,
      cloudLine: CLOUD_LINE,
      cloudTip: CLOUD_TIP,
    });
  }
  return rows;
}

// The message a mode click sends, or null for a bad tool or mode.
export function modeMessage(tool, mode) {
  if (typeof tool !== "string" || !tool || !MODES.includes(mode)) return null;
  return JSON.stringify({ type: "permission", tool, mode });
}

// The message the "With cloud model" switch sends.
export function cloudMessage(tool, cloud) {
  if (typeof tool !== "string" || !tool || typeof cloud !== "boolean") return null;
  return JSON.stringify({ type: "permission", tool, cloud });
}

// Activity rows from GET /activity ({"entries": [...]}), newest first as the
// server sends them. ts is an ISO time; the page shows the time of day.
export function activityRows(data) {
  if (!data || typeof data !== "object" || !Array.isArray(data.entries)) return [];
  const rows = [];
  for (const e of data.entries) {
    if (!e || typeof e !== "object" || typeof e.tool !== "string") continue;
    const ts = typeof e.ts === "string" ? e.ts : "";
    const match = /^\d{4}-\d{2}-\d{2}[T ](\d{2}:\d{2}:\d{2})/.exec(ts);
    rows.push({
      time: match ? match[1] : ts.slice(0, 19),
      tool: e.tool.slice(0, 60),
      mode: text(e.mode),
      outcome: text(e.outcome),
    });
    if (rows.length >= MAX_ACTIVITY) break;
  }
  return rows;
}

function node(tag, className, content) {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (content !== undefined) n.textContent = content;
  return n;
}

// el: { list, activity, activityNote }. send(text) sends if the socket is
// open. fetchActivity() resolves to the GET /activity JSON.
// Returns { show, refreshActivity }.
export function initPermissions(el, send, fetchActivity) {
  function renderRow(row) {
    const li = node("li", "perm-row");
    li.dataset.tool = row.name;
    if (!row.available) li.classList.add("perm-unavailable");
    const head = node("div", "perm-head");
    head.append(node("span", "perm-label", row.label));
    const badge = node("span", "perm-badge", row.badge);
    badge.dataset.kind = row.badge;
    head.append(badge);
    li.append(head);
    if (row.note) li.append(node("p", "settings-dim perm-note", row.note));
    if (!row.available) {
      li.append(node("p", "settings-dim perm-missing", row.missing || "Not set up."));
      return li;
    }
    const modes = node("div", "perm-modes");
    modes.setAttribute("role", "group");
    modes.setAttribute("aria-label", `${row.label} mode`);
    for (const m of row.modes) {
      const b = node("button", "perm-mode", m.label);
      b.type = "button";
      b.dataset.mode = m.mode;
      b.setAttribute("aria-pressed", String(m.active));
      b.addEventListener("click", () => {
        const msg = modeMessage(row.name, m.mode);
        if (msg) send(msg);
      });
      modes.append(b);
    }
    li.append(modes);
    if (row.showCloud) {
      const label = node("label", "settings-choice perm-cloud");
      const box = document.createElement("input");
      box.type = "checkbox";
      box.checked = row.cloud;
      box.addEventListener("change", () => {
        const msg = cloudMessage(row.name, box.checked);
        if (msg) send(msg);
      });
      label.append(box, node("span", "", "With cloud model"));
      label.title = row.cloudTip;
      li.append(label);
      const line = node("p", "settings-dim perm-cloud-line", row.cloudLine);
      line.title = row.cloudTip;
      li.append(line);
    }
    return li;
  }

  async function refreshActivity() {
    try {
      const rows = activityRows(await fetchActivity());
      el.activity.replaceChildren(
        ...rows.map((r) => {
          const li = node("li", "act-row");
          li.append(node("span", "act-time", r.time), node("span", "act-tool", r.tool), node("span", "act-outcome", r.outcome));
          return li;
        }),
      );
      el.activityNote.textContent = rows.length ? "" : "Nothing yet.";
    } catch {
      el.activityNote.textContent = "Could not load the activity.";
    }
  }

  return {
    show(msg) {
      const rows = rowsFromMessage(msg);
      if (rows) el.list.replaceChildren(...rows.map(renderRow));
    },
    refreshActivity,
  };
}
