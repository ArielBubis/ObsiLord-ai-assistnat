// "Memory tests" in the debug panel (?debug=1): what to ask the assistant
// to check conversation memory, from GET /debug/memory-checks. The server
// serves it only with [debug] demo_memory = true, which loads made-up history
// with dates relative to the server start. "use" puts the question into the
// text box.
//
// checksView is tested with node --test.

export const MSG_NO_DEMO =
  "Set [debug] demo_memory = true in config.toml and restart to load the test history.";
const MAX_TEXT = 300;

function text(value) {
  return typeof value === "string" ? value.slice(0, MAX_TEXT) : "";
}

// [{ title, items: [{ question, expected, kind }] }], grouped by label in the
// order the server gave.
export function checksView(body) {
  const checks = body && Array.isArray(body.checks) ? body.checks : [];
  const groups = [];
  const byTitle = new Map();
  for (const c of checks) {
    if (!c || typeof c.question !== "string" || typeof c.label !== "string") continue;
    const title = c.date ? `${text(c.label)} (${text(c.date)})` : text(c.label);
    let group = byTitle.get(title);
    if (!group) {
      group = { title, items: [] };
      byTitle.set(title, group);
      groups.push(group);
    }
    group.items.push({ question: text(c.question), expected: text(c.expected), kind: text(c.kind) });
  }
  return groups;
}

// el: { list, note }. fetchChecks() resolves to the JSON, or null when the
// server has no demo history (404). fill(question) puts it in the text box.
export async function initMemoryChecks(el, fetchChecks, fill) {
  let body = null;
  try {
    body = await fetchChecks();
  } catch {
    body = null;
  }
  if (body === null) {
    el.note.textContent = MSG_NO_DEMO;
    return;
  }
  el.note.textContent = "Ask these in Local mode. Expected answers are what a good reply says.";
  el.list.replaceChildren(
    ...checksView(body).map((group) => {
      const section = document.createElement("div");
      section.className = "memory-group";
      const head = document.createElement("p");
      head.className = "memory-group-title";
      head.textContent = group.title;
      const ul = document.createElement("ul");
      for (const item of group.items) {
        const li = document.createElement("li");
        const q = document.createElement("span");
        q.className = "memory-q";
        q.textContent = item.question;
        const e = document.createElement("span");
        e.className = "memory-e";
        e.textContent = `${item.kind}: ${item.expected}`;
        const use = document.createElement("button");
        use.type = "button";
        use.textContent = "use";
        use.addEventListener("click", () => fill(item.question));
        li.append(use, q, e);
        ul.appendChild(li);
      }
      section.append(head, ul);
      return section;
    }),
  );
}
