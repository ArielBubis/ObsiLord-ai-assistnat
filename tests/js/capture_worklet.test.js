// Runs web/capture-worklet.js in node with stand-ins for the AudioWorklet
// globals (AudioWorkletProcessor, registerProcessor, sampleRate).
import { test } from "node:test";
import assert from "node:assert/strict";

const registered = new Map();

class FakePort {
  constructor() {
    this.sent = [];
    this.onmessage = null;
  }
  postMessage(msg, transfer = []) {
    for (const buf of transfer) assert.ok(msg.buf === buf, "only the frame buffer is transferred");
    this.sent.push(msg);
  }
  deliver(data) {
    this.onmessage({ data });
  }
}

globalThis.AudioWorkletProcessor = class {
  constructor() {
    this.port = new FakePort();
  }
};
globalThis.registerProcessor = (name, cls) => registered.set(name, cls);

await import("../../web/capture-worklet.js");

function makeProcessor(rate) {
  globalThis.sampleRate = rate;
  const Cls = registered.get("capture-16k");
  return new Cls();
}

function sine(rate, freq, n, amp = 0.5, offset = 0) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * (i + offset)) / rate);
  return out;
}

// Feeds `seconds` of a sine in 128-sample render quanta.
function feed(proc, rate, freq, seconds, channels = 1) {
  const total = Math.round(rate * seconds);
  for (let off = 0; off < total; off += 128) {
    const n = Math.min(128, total - off);
    const block = sine(rate, freq, n, 0.5, off);
    const chans = [];
    for (let c = 0; c < channels; c++) chans.push(block);
    assert.equal(proc.process([chans], [[new Float32Array(128)]], {}), true);
  }
  return total;
}

function frames(proc) {
  return proc.port.sent.filter((m) => m.type === "frame");
}

function decode(bufs) {
  const total = bufs.reduce((a, b) => a + b.byteLength / 2, 0);
  const out = new Float32Array(total);
  let k = 0;
  for (const b of bufs) {
    const v = new DataView(b);
    for (let i = 0; i < b.byteLength / 2; i++) out[k++] = v.getInt16(i * 2, true) / 32768;
  }
  return out;
}

function upwardCrossingsHz(x, rate) {
  let first = -1;
  let last = -1;
  let count = 0;
  for (let i = 1; i < x.length; i++) {
    if (x[i - 1] < 0 && x[i] >= 0) {
      const t = i - 1 + x[i - 1] / (x[i - 1] - x[i]);
      if (first < 0) first = t;
      else count++;
      last = t;
    }
  }
  return (count * rate) / (last - first);
}

test("registers capture-16k", () => {
  assert.equal(typeof registered.get("capture-16k"), "function");
});

test("48 kHz: 1 s in 128-sample quanta gives ten 3200-byte frames of 440 Hz", () => {
  const proc = makeProcessor(48000);
  proc.port.deliver({ type: "start", id: 7 });
  feed(proc, 48000, 440, 1);
  const sent = proc.port.sent;
  assert.equal(sent[0].type, "first");
  assert.equal(sent.filter((m) => m.type === "first").length, 1);
  // 16000 samples minus the resampler lag: 9 full frames so far.
  assert.equal(frames(proc).length, 9);
  proc.port.deliver({ type: "stop", id: 7 });
  const all = frames(proc);
  assert.equal(all.length, 10);
  for (const f of all) {
    assert.equal(f.id, 7);
    assert.equal(f.buf.byteLength, 3200);
  }
  assert.deepEqual(sent.at(-1), { type: "end", id: 7 });
  const pcm = decode(all.map((f) => f.buf));
  assert.equal(pcm.length, 16000);
  const hz = upwardCrossingsHz(pcm.subarray(200, 15800), 16000);
  assert.ok(Math.abs(hz - 440) < 0.1, `${hz} Hz`);
});

test("44.1 kHz stereo is mixed to mono; the last frame is partial", () => {
  const proc = makeProcessor(44100);
  proc.port.deliver({ type: "start", id: 1 });
  feed(proc, 44100, 1000, 0.25, 2);
  proc.port.deliver({ type: "stop", id: 1 });
  const all = frames(proc);
  // 0.25 s = 4000 samples: two full frames and 800 samples.
  assert.deepEqual(all.map((f) => f.buf.byteLength), [3200, 3200, 1600]);
  const pcm = decode(all.map((f) => f.buf));
  const hz = upwardCrossingsHz(pcm.subarray(100, 3900), 16000);
  assert.ok(Math.abs(hz - 1000) < 0.5, `${hz} Hz`);
  let peak = 0;
  for (const v of pcm.subarray(100, 3900)) peak = Math.max(peak, Math.abs(v));
  assert.ok(Math.abs(peak - 0.5) < 0.01, `peak ${peak}`);
});

test("nothing is posted before start, after stop, or without input channels", () => {
  const proc = makeProcessor(48000);
  feed(proc, 48000, 440, 0.2);
  assert.equal(proc.port.sent.length, 0);
  proc.port.deliver({ type: "start", id: 2 });
  assert.equal(proc.process([[]], [[new Float32Array(128)]], {}), true);
  assert.equal(proc.process([], [], {}), true);
  assert.equal(proc.port.sent.length, 0);
  proc.port.deliver({ type: "stop", id: 2 });
  // Stop with nothing captured: no frame, just end.
  assert.deepEqual(proc.port.sent, [{ type: "end", id: 2 }]);
  feed(proc, 48000, 440, 0.2);
  assert.equal(proc.port.sent.length, 1);
});

test("a stale stop does not end the current recording", () => {
  const proc = makeProcessor(48000);
  proc.port.deliver({ type: "start", id: 3 });
  feed(proc, 48000, 440, 0.15);
  proc.port.deliver({ type: "stop", id: 2 });
  assert.deepEqual(proc.port.sent.at(-1), { type: "end", id: 2 });
  feed(proc, 48000, 440, 0.15);
  proc.port.deliver({ type: "stop", id: 3 });
  const all = frames(proc);
  assert.ok(all.every((f) => f.id === 3));
  assert.equal(decode(all.map((f) => f.buf)).length, 4800);
});

test("a new start resets: no samples leak from the previous recording", () => {
  const proc = makeProcessor(48000);
  proc.port.deliver({ type: "start", id: 4 });
  feed(proc, 48000, 440, 0.07);
  proc.port.deliver({ type: "start", id: 5 });
  feed(proc, 48000, 440, 0.1);
  proc.port.deliver({ type: "stop", id: 5 });
  const mine = frames(proc).filter((f) => f.id === 5);
  assert.equal(decode(mine.map((f) => f.buf)).length, 1600);
  assert.equal(frames(proc).filter((f) => f.id === 4).length, 0);
});

// Feeds n samples of silence in 128-sample quanta.
function feedSilence(proc, n) {
  for (let off = 0; off < n; off += 128) {
    proc.process([[new Float32Array(Math.min(128, n - off))]], [[new Float32Array(128)]], {});
  }
}

test("first is sent on the first non-silent block, not on pre-roll silence", () => {
  const proc = makeProcessor(48000);
  proc.port.deliver({ type: "start", id: 8 });
  feedSilence(proc, 4800); // 100 ms of zeros
  assert.equal(proc.port.sent.filter((m) => m.type === "first").length, 0);
  const block = new Float32Array(128);
  block[100] = 0.01;
  proc.process([[block]], [[new Float32Array(128)]], {});
  assert.deepEqual(proc.port.sent.filter((m) => m.type === "first"), [{ type: "first", id: 8 }]);
  feed(proc, 48000, 440, 0.1);
  assert.equal(proc.port.sent.filter((m) => m.type === "first").length, 1, "only once");
});

test("all-zero input still sends first after 300 ms, so the hint never stalls", () => {
  const proc = makeProcessor(44100);
  proc.port.deliver({ type: "start", id: 9 });
  feedSilence(proc, 13056); // 102 quanta, just under 0.3 s
  assert.equal(proc.port.sent.filter((m) => m.type === "first").length, 0);
  feedSilence(proc, 256);
  assert.deepEqual(proc.port.sent.filter((m) => m.type === "first"), [{ type: "first", id: 9 }]);
  // A new recording waits for its own first audio.
  proc.port.deliver({ type: "start", id: 10 });
  feedSilence(proc, 1280);
  assert.equal(proc.port.sent.filter((m) => m.type === "first" && m.id === 10).length, 0);
});

test("malformed commands are ignored", () => {
  const proc = makeProcessor(48000);
  proc.port.deliver(null);
  proc.port.deliver("start");
  proc.port.deliver({ type: "bogus" });
  feed(proc, 48000, 440, 0.2);
  assert.equal(proc.port.sent.length, 0);
});
