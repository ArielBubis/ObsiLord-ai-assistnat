import { test } from "node:test";
import assert from "node:assert/strict";

import {
  createMic, micErrorMessage, MicError, MIC_CONSTRAINTS, MIC_MESSAGES, WORKLET_URL, WORKLET_NAME,
} from "../../web/audio_in.js";

const tick = () => new Promise((r) => setTimeout(r, 0));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function domError(name) {
  const e = new Error(name);
  e.name = name;
  return e;
}

// Fake mic, AudioContext, and AudioWorkletNode. log records what happened.
function fakes({ gumError = null, moduleError = null } = {}) {
  const log = { gum: [], modules: [], nodes: [], posted: [], tracks: [], sources: [] };
  const ctx = {
    sampleRate: 48000,
    destination: { name: "destination" },
    audioWorklet: {
      addModule(url) {
        log.modules.push(url);
        return moduleError ? Promise.reject(moduleError) : Promise.resolve();
      },
    },
    createMediaStreamSource(stream) {
      const src = {
        stream, connectedTo: null, disconnected: false,
        connect(n) { src.connectedTo = n; },
        disconnect() { src.disconnected = true; },
      };
      log.sources.push(src);
      return src;
    },
  };
  class FakeNode {
    constructor(c, name, opts) {
      this.ctx = c;
      this.name = name;
      this.opts = opts;
      this.connectedTo = null;
      this.port = { onmessage: null, postMessage: (msg) => log.posted.push(msg) };
      log.nodes.push(this);
    }
    connect(to) {
      this.connectedTo = to;
    }
    // Simulates the worklet posting to the main thread.
    emit(data) {
      this.port.onmessage({ data });
    }
  }
  function getUserMedia(constraints) {
    log.gum.push(constraints);
    if (gumError) return Promise.reject(gumError);
    const track = { readyState: "live", stop() { track.readyState = "ended"; } };
    log.tracks.push(track);
    return Promise.resolve({ getTracks: () => [track] });
  }
  let t = 0;
  const now = () => (t += 10);
  return { log, ctx, FakeNode, getUserMedia, now };
}

function makeMic(f, extra = {}) {
  return createMic({
    getContext: () => f.ctx, getUserMedia: f.getUserMedia, NodeCtor: f.FakeNode, now: f.now, ...extra,
  });
}

const frame = (n = 3200) => new ArrayBuffer(n);

test("mic constraints: echo cancellation, noise suppression, auto gain, mono", () => {
  assert.deepEqual(MIC_CONSTRAINTS, {
    audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
  });
  assert.equal(WORKLET_URL, "/static/capture-worklet.js");
  assert.equal(WORKLET_NAME, "capture-16k");
});

test("getUserMedia errors map to clear messages", () => {
  assert.equal(micErrorMessage(domError("NotAllowedError")), "Microphone blocked. Allow it in the browser and try again.");
  assert.equal(micErrorMessage(domError("SecurityError")), MIC_MESSAGES.blocked);
  assert.equal(micErrorMessage(domError("NotFoundError")), MIC_MESSAGES.missing);
  assert.equal(micErrorMessage(domError("OverconstrainedError")), MIC_MESSAGES.missing);
  assert.equal(micErrorMessage(domError("NotReadableError")), MIC_MESSAGES.inUse);
  assert.equal(micErrorMessage(domError("AbortError")), MIC_MESSAGES.inUse);
  assert.equal(micErrorMessage(new TypeError("no mediaDevices")), MIC_MESSAGES.unsupported);
  assert.equal(micErrorMessage(domError("Weird")), MIC_MESSAGES.failed);
  assert.equal(micErrorMessage(null), MIC_MESSAGES.failed);
  assert.equal(micErrorMessage(new MicError("custom")), "custom");
});

test("begin opens the mic, loads the worklet once, connects, and starts the worklet", async () => {
  const f = fakes();
  const mic = makeMic(f);
  const handle = await mic.begin({ onFrame: () => {} });
  assert.deepEqual(f.log.gum, [MIC_CONSTRAINTS]);
  assert.deepEqual(f.log.modules, ["/static/capture-worklet.js"]);
  assert.equal(f.log.nodes.length, 1);
  const node = f.log.nodes[0];
  assert.equal(node.name, "capture-16k");
  assert.equal(node.opts.channelCount, 1);
  assert.equal(node.opts.channelCountMode, "explicit");
  assert.equal(node.connectedTo, f.ctx.destination);
  assert.equal(f.log.sources[0].connectedTo, node);
  assert.deepEqual(f.log.posted, [{ type: "start", id: handle.id }]);
  assert.equal(mic.active, true);
  assert.equal(mic.stats.opens, 1);
  assert.ok(mic.stats.gumMs > 0);
  handle.abort();
  // A second recording reuses the node.
  await mic.begin({ onFrame: () => {} });
  assert.equal(f.log.modules.length, 1);
  assert.equal(f.log.nodes.length, 1);
});

test("frames reach onFrame; end() flushes, then stops the tracks", async () => {
  const f = fakes();
  const mic = makeMic(f);
  const got = [];
  let firsts = 0;
  const handle = await mic.begin({ onFrame: (b) => got.push(b), onFirstAudio: () => firsts++ });
  const node = f.log.nodes[0];
  assert.equal(firsts, 0);
  node.emit({ type: "first", id: handle.id + 99 });
  assert.equal(firsts, 0, "first audio of another recording is ignored");
  node.emit({ type: "first", id: handle.id });
  assert.equal(firsts, 1);
  assert.ok(mic.stats.firstAudioMs > 0);
  node.emit({ type: "frame", id: handle.id, buf: frame() });
  node.emit({ type: "frame", id: handle.id + 99, buf: frame() });
  assert.equal(got.length, 1, "frames from another recording are dropped");
  let done = false;
  const ending = handle.end().then(() => { done = true; });
  assert.deepEqual(f.log.posted.at(-1), { type: "stop", id: handle.id });
  await tick();
  assert.equal(done, false);
  assert.equal(f.log.tracks[0].readyState, "live", "mic stays open until the last frame is out");
  node.emit({ type: "frame", id: handle.id, buf: frame(640) });
  node.emit({ type: "end", id: handle.id });
  await ending;
  assert.equal(got.length, 2);
  assert.equal(got[1].byteLength, 640);
  assert.equal(f.log.tracks[0].readyState, "ended");
  assert.equal(f.log.sources[0].disconnected, true);
  assert.equal(mic.active, false);
  assert.ok(mic.stats.stopMs >= 0);
  // Late frames are dropped.
  node.emit({ type: "frame", id: handle.id, buf: frame() });
  assert.equal(got.length, 2);
  // end() again is the same promise and harmless.
  await handle.end();
});

test("abort() stops the tracks at once and drops later frames", async () => {
  const f = fakes();
  const mic = makeMic(f);
  const got = [];
  const handle = await mic.begin({ onFrame: (b) => got.push(b) });
  handle.abort();
  assert.equal(f.log.tracks[0].readyState, "ended");
  assert.deepEqual(f.log.posted.at(-1), { type: "stop", id: handle.id });
  f.log.nodes[0].emit({ type: "frame", id: handle.id, buf: frame() });
  assert.equal(got.length, 0);
  assert.equal(mic.active, false);
  await handle.end();
});

test("a worklet that never answers stop still releases the mic", async () => {
  const f = fakes();
  const mic = makeMic(f, { stopTimeoutMs: 5 });
  const handle = await mic.begin({ onFrame: () => {} });
  const warn = console.warn;
  const warned = [];
  console.warn = (...a) => warned.push(a.join(" "));
  try {
    await handle.end();
  } finally {
    console.warn = warn;
  }
  assert.equal(f.log.tracks[0].readyState, "ended");
  assert.equal(warned.length, 1);
});

test("permission denied: MicError with the blocked message, nothing left open", async () => {
  const f = fakes({ gumError: domError("NotAllowedError") });
  const mic = makeMic(f);
  await assert.rejects(mic.begin({ onFrame: () => {} }), (e) => e instanceof MicError && e.message === MIC_MESSAGES.blocked);
  assert.equal(mic.active, false);
  assert.equal(f.log.sources.length, 0);
});

test("worklet module failure stops the new tracks, and the next press retries", async () => {
  const f = fakes({ moduleError: new Error("404") });
  const mic = makeMic(f);
  await assert.rejects(mic.begin({ onFrame: () => {} }), (e) => e.message === MIC_MESSAGES.failed);
  assert.equal(f.log.tracks[0].readyState, "ended");
  await assert.rejects(mic.begin({ onFrame: () => {} }));
  assert.equal(f.log.modules.length, 2);
});

test("no AudioWorklet or no context: unsupported", async () => {
  const f = fakes();
  delete f.ctx.audioWorklet;
  const mic = makeMic(f);
  await assert.rejects(mic.begin({ onFrame: () => {} }), (e) => e.message === MIC_MESSAGES.unsupported);
  assert.equal(f.log.tracks[0].readyState, "ended");
  const none = createMic({ getContext: () => null, getUserMedia: f.getUserMedia, NodeCtor: f.FakeNode });
  await assert.rejects(none.begin({ onFrame: () => {} }), (e) => e.message === MIC_MESSAGES.unsupported);
});

test("without navigator.mediaDevices the default getUserMedia reports unsupported", async () => {
  const f = fakes();
  const mic = createMic({ getContext: () => f.ctx, NodeCtor: f.FakeNode });
  const saved = Object.getOwnPropertyDescriptor(globalThis, "navigator");
  Object.defineProperty(globalThis, "navigator", { value: {}, configurable: true });
  try {
    await assert.rejects(mic.begin({ onFrame: () => {} }), (e) => e.message === MIC_MESSAGES.unsupported);
  } finally {
    if (saved) Object.defineProperty(globalThis, "navigator", saved);
  }
});

test("a new begin while a recording is open closes the old one", async () => {
  const f = fakes();
  const mic = makeMic(f);
  const got = [];
  const first = await mic.begin({ onFrame: (b) => got.push(["a", b]) });
  const second = await mic.begin({ onFrame: (b) => got.push(["b", b]) });
  assert.equal(f.log.tracks[0].readyState, "ended");
  assert.equal(f.log.tracks[1].readyState, "live");
  f.log.nodes[0].emit({ type: "frame", id: first.id, buf: frame() });
  f.log.nodes[0].emit({ type: "frame", id: second.id, buf: frame() });
  assert.deepEqual(got.map((g) => g[0]), ["b"]);
  await sleep(0);
});

test("a context rate the resampler cannot take is refused up front, mic released", async () => {
  const f = fakes();
  f.ctx.sampleRate = 384000;
  const mic = makeMic(f);
  await assert.rejects(mic.begin({ onFrame: () => {} }), (e) => e.message === MIC_MESSAGES.rate);
  assert.equal(f.log.modules.length, 0);
  assert.equal(f.log.tracks[0].readyState, "ended");
  assert.match(MIC_MESSAGES.rate, /8000 and 192000 Hz/);
});

test("a begin() that waits while a newer one starts is superseded and closes its stream", async () => {
  const f = fakes();
  // getUserMedia answers only when told (permission prompt open).
  const gates = [];
  const slowGum = (c) => {
    f.log.gum.push(c);
    return new Promise((resolve) => gates.push(() => {
      const track = { readyState: "live", stop() { track.readyState = "ended"; } };
      f.log.tracks.push(track);
      resolve({ getTracks: () => [track] });
    }));
  };
  const mic = createMic({ getContext: () => f.ctx, getUserMedia: slowGum, NodeCtor: f.FakeNode, now: f.now });
  const first = mic.begin({ onFrame: () => {} });
  const second = mic.begin({ onFrame: () => {} });
  await tick();
  gates[0]();
  await assert.rejects(first, (e) => e instanceof MicError && e.superseded === true);
  assert.equal(f.log.tracks[0].readyState, "ended", "the superseded stream is closed");
  gates[1]();
  const handle = await second;
  assert.equal(f.log.tracks[1].readyState, "live");
  assert.deepEqual(f.log.posted.filter((m) => m.type === "start"), [{ type: "start", id: handle.id }],
    "only the newest call starts the worklet");
  handle.abort();
});

test("a superseded begin() whose getUserMedia fails is marked superseded too", async () => {
  const f = fakes();
  const gates = [];
  const gum = () => new Promise((resolve, reject) => gates.push([resolve, reject]));
  const mic = createMic({ getContext: () => f.ctx, getUserMedia: gum, NodeCtor: f.FakeNode, now: f.now });
  const first = mic.begin({ onFrame: () => {} });
  mic.begin({ onFrame: () => {} }).catch(() => {});
  await tick();
  gates[0][1](domError("NotAllowedError"));
  await assert.rejects(first, (e) => e.superseded === true && e.message === MIC_MESSAGES.blocked);
  gates[1][1](domError("NotAllowedError"));
});
