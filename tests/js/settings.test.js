import { test } from "node:test";
import assert from "node:assert/strict";

import {
  readChoice, writeChoice, choiceMessage, initialChoice, modelOptions, serverLine, statusText,
  noticeView, STORAGE_KEY, MAX_NOTICE_LINES,
} from "../../web/settings.js";

function fakeStorage(initial = {}) {
  const data = { ...initial };
  return {
    data,
    getItem: (k) => (k in data ? data[k] : null),
    setItem: (k, v) => { data[k] = String(v); },
  };
}

const throwing = {
  getItem: () => { throw new Error("blocked"); },
  setItem: () => { throw new Error("blocked"); },
};

const CONFIG = {
  default_provider: "gemini",
  gemini: { available: true, model: "gemini-3.5-flash-lite" },
  local: {
    base_url: "http://127.0.0.1:1234/v1",
    default_model: "",
    reachable: true,
    error: null,
    models: [{ id: "qwen3-14b", loaded: true }, { id: "gemma-12b", loaded: false }],
  },
};

test("the choice round-trips through storage", () => {
  const storage = fakeStorage();
  assert.equal(readChoice(storage), null);
  writeChoice(storage, { provider: "local", model: "qwen3-14b" });
  assert.deepEqual(readChoice(storage), { provider: "local", model: "qwen3-14b" });
  assert.equal(storage.data[STORAGE_KEY], '{"provider":"local","model":"qwen3-14b"}');
});

test("bad or blocked storage gives no choice", () => {
  for (const raw of ["not json", "null", '{"provider":"openai","model":""}', '{"provider":"local"}',
    `{"provider":"local","model":"${"x".repeat(201)}"}`]) {
    assert.equal(readChoice(fakeStorage({ [STORAGE_KEY]: raw })), null, raw);
  }
  assert.equal(readChoice(throwing), null);
  writeChoice(throwing, { provider: "gemini", model: "" }); // does not throw
});

test("choiceMessage is the JSON the server expects", () => {
  assert.equal(
    choiceMessage({ provider: "local", model: "qwen3-14b" }, false),
    '{"type":"llm","provider":"local","model":"qwen3-14b","announce":false}',
  );
  // The local model is kept in the page while Gemini is picked, but not sent.
  assert.equal(
    choiceMessage({ provider: "gemini", model: "qwen3-14b" }, 1),
    '{"type":"llm","provider":"gemini","model":"","announce":true}',
  );
});

test("initialChoice keeps the stored choice unless Gemini has no key", () => {
  const local = { provider: "local", model: "gemma-12b" };
  assert.deepEqual(initialChoice(local, CONFIG), local);
  assert.deepEqual(initialChoice({ provider: "gemini", model: "" }, CONFIG), { provider: "gemini", model: "" });
  const noKey = { ...CONFIG, gemini: { available: false } };
  assert.deepEqual(initialChoice({ provider: "gemini", model: "" }, noKey), { provider: "local", model: "" });
});

test("initialChoice without a stored choice follows the server default", () => {
  assert.deepEqual(initialChoice(null, CONFIG), { provider: "gemini", model: "" });
  const localFirst = { ...CONFIG, default_provider: "local", local: { ...CONFIG.local, default_model: "gemma-12b" } };
  assert.deepEqual(initialChoice(null, localFirst), { provider: "local", model: "gemma-12b" });
  assert.deepEqual(initialChoice(null, { gemini: { available: false }, local: {} }), { provider: "local", model: "" });
  // Local is the main provider: Gemini only when the server names it.
  assert.deepEqual(initialChoice(null, { gemini: { available: true }, local: {} }), { provider: "local", model: "" });
  assert.deepEqual(
    initialChoice(null, { default_provider: "gemini", gemini: { available: false }, local: {} }),
    { provider: "local", model: "" },
  );
  assert.deepEqual(initialChoice(null, null), { provider: "local", model: "" });
});

test("modelOptions marks loaded models and keeps an unlisted selection", () => {
  assert.deepEqual(modelOptions(CONFIG, "gemma-12b"), [
    { value: "", label: "Server default (first listed)" },
    { value: "qwen3-14b", label: "qwen3-14b (loaded)" },
    { value: "gemma-12b", label: "gemma-12b" },
  ]);
  const down = { local: { reachable: false, models: [], default_model: "" } };
  assert.deepEqual(modelOptions(down, "qwen3-14b"), [
    { value: "", label: "Server default (first listed)" },
    { value: "qwen3-14b", label: "qwen3-14b (not listed)" },
  ]);
  const withDefault = { local: { default_model: "qwen3-14b", models: [{ id: "qwen3-14b" }, { id: 3 }, null] } };
  assert.deepEqual(modelOptions(withDefault, "qwen3-14b"), [{ value: "qwen3-14b", label: "qwen3-14b" }]);
});

test("modelOptions marks models without tool support", () => {
  const config = {
    local: {
      default_model: "x",
      models: [
        { id: "a", loaded: true, tools: false },
        { id: "b", tools: false },
        { id: "c", tools: true },
        { id: "d" },
      ],
    },
  };
  assert.deepEqual(modelOptions(config, "a"), [
    { value: "a", label: "a (loaded, no tools)" },
    { value: "b", label: "b (no tools)" },
    { value: "c", label: "c" },
    { value: "d", label: "d" },
  ]);
});

test("serverLine says how many models or that the server is down", () => {
  assert.equal(serverLine(CONFIG), "Local server http://127.0.0.1:1234/v1: 2 models.");
  assert.equal(
    serverLine({ local: { base_url: "http://127.0.0.1:1234/v1", reachable: true, models: [{ id: "a" }] } }),
    "Local server http://127.0.0.1:1234/v1: 1 model.",
  );
  assert.match(serverLine({ local: { base_url: "http://x/v1", reachable: false } }), /not reachable at http:\/\/x\/v1/);
});

test("statusText for each warm-up state", () => {
  assert.equal(statusText({ type: "llm_status", state: "loading", model: "qwen3-14b" }), "Loading qwen3-14b...");
  assert.equal(statusText({ type: "llm_status", state: "loading", model: "" }), "Loading the model...");
  assert.equal(statusText({ type: "llm_status", state: "ready", model: "m", seconds: 3.2 }), "m ready (3.2 s).");
  assert.equal(
    statusText({ type: "llm_status", state: "failed", model: "m", message: "HTTP 404" }),
    "Could not load m: HTTP 404",
  );
  assert.equal(statusText({ type: "llm_status", state: "other" }), null);
  assert.equal(statusText({ type: "notice" }), null);
});

test("noticeView keeps strings only and caps the lines", () => {
  const lines = Array.from({ length: 20 }, (_, i) => `line ${i}`);
  const view = noticeView({ type: "notice", title: "Chat runs on X", lines: [...lines, 5, null] });
  assert.equal(view.title, "Chat runs on X");
  assert.equal(view.lines.length, MAX_NOTICE_LINES);
  assert.equal(view.lines[0], "line 0");
  assert.equal(noticeView({ type: "notice" }), null);
  assert.equal(noticeView({ type: "error", title: "x" }), null);
  assert.deepEqual(noticeView({ type: "notice", title: "t" }).lines, []);
});
