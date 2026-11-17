// Settings panel: which chat model this page talks to, Gemini or a model on
// the local OpenAI-compatible server (LM Studio, Ollama).
//
// GET /config/llm says whether Gemini has a key and lists the local server's
// models (the server fetches them; the page never talks to the local server).
// The choice is remembered in this browser (localStorage) and sent as
// {"type": "llm", "provider", "model", "announce"}: on a change with
// announce false (the server answers with a notice about what changes and,
// for a local model, llm_status messages while it loads), and after every
// connect with announce true (applied quietly). The local server address is
// only shown here; it is set in config.toml.
//
// The pure helpers at the top are tested with node --test.

export const STORAGE_KEY = "obsilord.llm";
export const PROVIDERS = ["gemini", "local"];
export const MAX_NOTICE_LINES = 12;
const MAX_TEXT = 400;

function isChoice(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    PROVIDERS.includes(value.provider) &&
    typeof value.model === "string" &&
    value.model.length <= 200
  );
}

// The remembered choice, or null. Storage may be missing or blocked.
export function readChoice(storage) {
  try {
    const value = JSON.parse(storage.getItem(STORAGE_KEY));
    return isChoice(value) ? { provider: value.provider, model: value.model } : null;
  } catch {
    return null;
  }
}

export function writeChoice(storage, choice) {
  try {
    storage.setItem(STORAGE_KEY, JSON.stringify(choice));
  } catch {
    // Not remembered.
  }
}

export function choiceMessage(choice, announce) {
  return JSON.stringify({
    type: "llm",
    provider: choice.provider,
    model: choice.provider === "local" ? choice.model : "",
    announce: Boolean(announce),
  });
}

// What the panel starts with: the remembered choice, unless it asks for
// Gemini without a key; otherwise the server's default.
export function initialChoice(stored, config) {
  const geminiOk = Boolean(config && config.gemini && config.gemini.available);
  const localDefault = (config && config.local && config.local.default_model) || "";
  if (stored && (stored.provider === "local" || geminiOk)) return stored;
  if (stored && stored.provider === "gemini" && !geminiOk) {
    return { provider: "local", model: localDefault };
  }
  const provider = config && config.default_provider === "local" ? "local" : geminiOk ? "gemini" : "local";
  return { provider, model: localDefault };
}

// Options for the model list. The selected model stays in the list even when
// the server does not list it (server down, model removed), marked as such.
export function modelOptions(config, selected) {
  const local = (config && config.local) || {};
  const models = Array.isArray(local.models) ? local.models : [];
  const options = [];
  if (!local.default_model) options.push({ value: "", label: "Server default (first listed)" });
  for (const m of models) {
    if (!m || typeof m.id !== "string") continue;
    options.push({ value: m.id, label: m.loaded === true ? `${m.id} (loaded)` : m.id });
  }
  if (selected && !options.some((o) => o.value === selected)) {
    options.push({ value: selected, label: `${selected} (not listed)` });
  }
  return options;
}

// One line about the local server for under the model list.
export function serverLine(config) {
  const local = (config && config.local) || {};
  const url = typeof local.base_url === "string" ? local.base_url : "?";
  if (!local.reachable) return `Local server not reachable at ${url}. Start it and press Refresh.`;
  const n = Array.isArray(local.models) ? local.models.length : 0;
  return `Local server ${url}: ${n} ${n === 1 ? "model" : "models"}.`;
}

// The text for an llm_status message, or null.
export function statusText(msg) {
  if (!msg || msg.type !== "llm_status") return null;
  const model = typeof msg.model === "string" && msg.model ? msg.model : "the model";
  switch (msg.state) {
    case "loading":
      return `Loading ${model}...`;
    case "ready":
      return typeof msg.seconds === "number" ? `${model} ready (${msg.seconds} s).` : `${model} ready.`;
    case "failed":
      return `Could not load ${model}: ${String(msg.message || "unknown error").slice(0, MAX_TEXT)}`;
    default:
      return null;
  }
}

// { title, lines } from a notice message (strings only, capped), or null.
export function noticeView(msg) {
  if (!msg || msg.type !== "notice" || typeof msg.title !== "string") return null;
  const lines = Array.isArray(msg.lines) ? msg.lines.filter((l) => typeof l === "string") : [];
  return {
    title: msg.title.slice(0, MAX_TEXT),
    lines: lines.slice(0, MAX_NOTICE_LINES).map((l) => l.slice(0, MAX_TEXT)),
  };
}

// el: { panel, gemini, local, geminiNote, model, refresh, server, status,
// notice, noticeTitle, noticeLines, noticeClose }. send(text) sends if the
// socket is open. fetchConfig() resolves to the /config/llm JSON.
// Returns { announce, showNotice, showStatus, setDisabled }.
export function initSettings(el, storage, send, fetchConfig) {
  let config = null;
  let choice = readChoice(storage);
  let disabled = false;

  function render() {
    const geminiOk = Boolean(config && config.gemini && config.gemini.available);
    el.gemini.disabled = disabled || !geminiOk;
    el.geminiNote.textContent = config
      ? geminiOk
        ? `(${config.gemini.model})`
        : "(no GEMINI_API_KEY in .env)"
      : "";
    el.local.disabled = disabled;
    if (choice) {
      el.gemini.checked = choice.provider === "gemini";
      el.local.checked = choice.provider === "local";
    }
    // The local model is kept while Gemini is picked, for switching back.
    const selected = choice ? choice.model : "";
    el.model.replaceChildren(
      ...modelOptions(config, selected).map((o) => {
        const opt = document.createElement("option");
        opt.value = o.value;
        opt.textContent = o.label;
        return opt;
      }),
    );
    el.model.value = selected;
    el.model.disabled = disabled || !(choice && choice.provider === "local");
    el.refresh.disabled = disabled;
    el.server.textContent = config ? serverLine(config) : "Loading model list...";
  }

  function change(next) {
    choice = next;
    writeChoice(storage, choice);
    render();
    el.status.textContent = "";
    send(choiceMessage(choice, false));
  }

  // initial: the load at page start. Only then is the choice announced again
  // (it may have been adjusted, Gemini without a key); a Refresh only updates
  // the list. A fresh page uses the server's default, so nothing is sent.
  async function load(initial) {
    try {
      config = await fetchConfig();
    } catch {
      config = { gemini: { available: false }, local: { reachable: false, models: [] } };
    }
    const fresh = choice === null;
    const before = choice ? choiceMessage(choice, true) : null;
    choice = initialChoice(choice, config);
    render();
    if (initial && !fresh && choiceMessage(choice, true) !== before) {
      writeChoice(storage, choice);
      send(choiceMessage(choice, true));
    }
  }

  el.gemini.addEventListener("change", () => {
    if (el.gemini.checked) change({ provider: "gemini", model: el.model.value || "" });
  });
  el.local.addEventListener("change", () => {
    if (el.local.checked) change({ provider: "local", model: el.model.value || "" });
  });
  el.model.addEventListener("change", () => {
    if (choice && choice.provider === "local") change({ provider: "local", model: el.model.value });
  });
  el.refresh.addEventListener("click", () => {
    if (disabled) return;
    el.server.textContent = "Loading model list...";
    load(false);
  });
  el.noticeClose.addEventListener("click", () => {
    el.notice.hidden = true;
  });

  render();
  load(true);

  return {
    // Tell a newly opened socket what this page uses (quietly).
    announce() {
      if (choice) send(choiceMessage(choice, true));
    },
    showNotice(msg) {
      const view = noticeView(msg);
      if (view === null) return;
      el.noticeTitle.textContent = view.title;
      el.noticeLines.replaceChildren(
        ...view.lines.map((line) => {
          const li = document.createElement("li");
          li.textContent = line;
          return li;
        }),
      );
      el.notice.hidden = false;
    },
    showStatus(msg) {
      const text = statusText(msg);
      if (text !== null) el.status.textContent = text;
    },
    // No switching while a turn runs (the server would refuse it anyway).
    setDisabled(value) {
      disabled = Boolean(value);
      render();
    },
  };
}
