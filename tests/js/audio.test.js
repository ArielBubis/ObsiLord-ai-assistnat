import { test } from "node:test";
import assert from "node:assert/strict";

import {
  PLAYBACK, pcm16ToFloat32, createPcm16Decoder, validSampleRate, planStart, remainingS,
  applyFadeIn, fadeSamples, fadeOutRamp, rms, createPlayer, headroomDb, dbToGain, eqParamValues,
} from "../../web/audio.js";
import { AUDIO_EQ_DEFAULTS } from "../../web/config.js";

function le16(values) {
  const buf = new Uint8Array(values.length * 2);
  const view = new DataView(buf.buffer);
  values.forEach((v, i) => view.setInt16(i * 2, v, true));
  return buf;
}

test("pcm16ToFloat32 maps the extremes and zero", () => {
  const { samples, rest } = pcm16ToFloat32(le16([-32768, 32767, 0, 16384, -1]));
  assert.equal(samples.length, 5);
  assert.equal(samples[0], -1);
  assert.equal(samples[1], 32767 / 32768);
  assert.ok(samples[1] < 1);
  assert.equal(samples[2], 0);
  assert.equal(samples[3], 0.5);
  assert.equal(samples[4], -1 / 32768);
  assert.equal(rest.length, 0);
});

test("pcm16ToFloat32 reads little-endian bytes", () => {
  // 0x0102 little-endian is bytes 02 01.
  const { samples } = pcm16ToFloat32(new Uint8Array([0x02, 0x01]));
  assert.equal(samples[0], 0x0102 / 32768);
});

test("pcm16ToFloat32 accepts an ArrayBuffer and a Uint8Array view with an offset", () => {
  const bytes = le16([1000, -1000]);
  assert.deepEqual([...pcm16ToFloat32(bytes.buffer).samples], [1000 / 32768, -1000 / 32768]);
  const padded = new Uint8Array(5);
  padded.set(bytes, 1);
  const view = new Uint8Array(padded.buffer, 1, 4);
  assert.deepEqual([...pcm16ToFloat32(view).samples], [1000 / 32768, -1000 / 32768]);
});

test("odd byte length: the last byte is returned, not decoded", () => {
  const bytes = new Uint8Array([...le16([100, 200]), 0x7f]);
  const { samples, rest } = pcm16ToFloat32(bytes);
  assert.equal(samples.length, 2);
  assert.deepEqual([...rest], [0x7f]);
  assert.equal(pcm16ToFloat32(new Uint8Array([5])).samples.length, 0);
  assert.equal(pcm16ToFloat32(new Uint8Array(0)).samples.length, 0);
});

test("decoder carries an odd byte into the next chunk", () => {
  const all = le16([-32768, 1234, 32767, -5]);
  const dec = createPcm16Decoder();
  const a = dec.decode(all.slice(0, 3));
  assert.equal(a.length, 1);
  assert.equal(dec.pending, 1);
  const b = dec.decode(all.slice(3, 8).buffer);
  assert.equal(dec.pending, 0);
  assert.deepEqual([...a, ...b], [-1, 1234 / 32768, 32767 / 32768, -5 / 32768]);
  dec.decode(new Uint8Array([1]));
  dec.reset();
  assert.equal(dec.pending, 0);
});

test("validSampleRate", () => {
  assert.ok(validSampleRate(22050));
  assert.ok(validSampleRate(16000));
  // Pitch-shifted rates announced by the server, for example 22050 * 1.06.
  assert.ok(validSampleRate(23373));
  assert.ok(validSampleRate(8000));
  assert.ok(validSampleRate(96000));
  for (const bad of [0, -1, 22050.5, "22050", null, undefined, 1e6, NaN]) {
    assert.equal(validSampleRate(bad), false, String(bad));
  }
});

test("planStart: first chunk of a burst gets the lead, later chunks follow back to back", () => {
  const lead = PLAYBACK.leadS;
  assert.ok(lead >= 0.02 && lead <= 0.1);
  let next = 0;
  let now = 10;
  const first = planStart(next, now, lead);
  assert.deepEqual(first, { start: now + lead, burst: true });
  const dur = 2205 / 22050;
  next = first.start + dur;
  // Chunks arrive faster than real time: the clock barely moves.
  for (let k = 1; k < 20; k++) {
    now += 0.001;
    const p = planStart(next, now, lead);
    assert.equal(p.burst, false);
    assert.equal(p.start, next);
    next = p.start + dur;
  }
  assert.ok(Math.abs(next - (10 + lead + 20 * dur)) < 1e-9);
});

test("planStart: after an underrun a new burst starts from now plus the lead", () => {
  const p = planStart(5, 5.2, 0.05);
  assert.equal(p.burst, true);
  assert.ok(Math.abs(p.start - 5.25) < 1e-12);
  // Exactly at the boundary counts as drained.
  assert.equal(planStart(5, 5, 0.05).burst, true);
});

test("remainingS gives the drain time", () => {
  assert.equal(remainingS(12.5, 12), 0.5);
  assert.equal(remainingS(12, 12.5), 0);
  assert.equal(remainingS(0, 3), 0);
});

test("applyFadeIn ramps the first n samples from 0 and leaves the rest", () => {
  const s = new Float32Array([1, 1, 1, 1, 1, 1]);
  applyFadeIn(s, 4);
  assert.deepEqual([...s], [0, 0.25, 0.5, 0.75, 1, 1]);
  const short = new Float32Array([1, 1]);
  applyFadeIn(short, 10);
  assert.deepEqual([...short], [0, 0.5]);
  const none = new Float32Array([1]);
  applyFadeIn(none, 0);
  assert.deepEqual([...none], [1]);
});

test("fade length is a few ms", () => {
  assert.ok(PLAYBACK.fadeS >= 0.002 && PLAYBACK.fadeS <= 0.005);
  assert.equal(fadeSamples(22050, 0.003), 66);
  assert.equal(fadeSamples(8000, 0.00001), 1);
});

test("fadeOutRamp ends at the sentence end", () => {
  assert.deepEqual(fadeOutRamp(2, 1, 0.003), { from: 2 - 0.003, to: 2 });
  assert.deepEqual(fadeOutRamp(2, 1.999, 0.003), { from: 1.999, to: 2 });
  assert.equal(fadeOutRamp(2, 2, 0.003), null);
  assert.equal(fadeOutRamp(0, 1, 0.003), null);
});

test("rms", () => {
  assert.equal(rms(new Float32Array(0)), 0);
  assert.equal(rms(new Float32Array([0.5, -0.5, 0.5, -0.5])), 0.5);
  const n = 22050;
  const sine = new Float32Array(n);
  for (let i = 0; i < n; i++) sine[i] = 0.8 * Math.sin((2 * Math.PI * 220 * i) / n);
  assert.ok(Math.abs(rms(sine) - 0.8 / Math.SQRT2) < 1e-3);
});

// Minimal Web Audio stand-in: records start times and lets the test move the
// clock and fire ended events.
function fakeAudio() {
  // ramps: sentence gain automation; master: the master gain's values;
  // filters: biquad nodes with their last set values; graph: connections.
  const log = { starts: [], ramps: [], gainsDisconnected: 0, master: null, filters: [], graph: [], targets: [] };
  let gainCount = 0;
  // setTargetAtTime calls are logged in log.targets as [name, value, t, tau].
  const param = (name) => ({ value: 0, setTargetAtTime(v, t, tau) { log.targets.push([name, v, t, tau]); } });
  const sources = [];
  class FakeContext {
    constructor() {
      this.currentTime = 0;
      this.state = "running";
      this.destination = {};
    }
    resume() {
      this.state = "running";
      return Promise.resolve();
    }
    createBiquadFilter() {
      const node = { name: `filter${log.filters.length}`, type: "", connect(to) { log.graph.push([node.name, to.name]); } };
      node.frequency = param(`${node.name}.frequency`);
      node.gain = param(`${node.name}.gain`);
      log.filters.push(node);
      return node;
    }
    createAnalyser() {
      return {
        name: "analyser",
        fftSize: 2048,
        connect(to) { log.graph.push(["analyser", to.name || "destination"]); },
        getFloatTimeDomainData(arr) {
          arr.fill(0.25);
        },
      };
    }
    createGain() {
      // The player creates the master gain first, with the context.
      if (gainCount++ === 0) {
        const m = { name: "master", connect(to) { log.graph.push(["master", to.name]); } };
        m.gain = param("master.gain");
        log.master = m;
        return m;
      }
      return {
        name: "sentence",
        gain: {
          setValueAtTime: (v, t) => log.ramps.push(["set", v, t]),
          linearRampToValueAtTime: (v, t) => log.ramps.push(["ramp", v, t]),
        },
        connect(to) { log.graph.push(["sentence", to.name]); },
        disconnect() {
          log.gainsDisconnected++;
        },
      };
    }
    createBuffer(channels, length, rate) {
      const data = new Float32Array(length);
      return {
        length, sampleRate: rate, duration: length / rate, data,
        copyToChannel(src) {
          data.set(src);
        },
      };
    }
    createBufferSource() {
      const src = {
        buffer: null, onended: null, stopped: false,
        connect() {},
        disconnect() {},
        start(t) {
          log.starts.push({ t, dur: src.buffer.duration, first: src.buffer.data[0], n: src.buffer.length });
          src.at = t;
        },
        stop() {
          src.stopped = true;
        },
      };
      sources.push(src);
      return src;
    }
  }
  const ctxHolder = {};
  globalThis.AudioContext = function AudioContextShim(opts) {
    const c = new FakeContext(opts);
    ctxHolder.ctx = c;
    return c;
  };
  // Advance the clock and fire ended for every source that finished.
  function advance(to) {
    ctxHolder.ctx.currentTime = to;
    for (const s of sources) {
      if (!s.ended && s.at !== undefined && s.at + s.buffer.duration <= to + 1e-12) {
        s.ended = true;
        if (s.onended) s.onended();
      }
    }
  }
  return { log, sources, ctxHolder, advance };
}

function toneChunk(samples, value = 10000) {
  return le16(new Array(samples).fill(value)).buffer;
}

test("player schedules chunks back to back and reports drain once", () => {
  const fa = fakeAudio();
  let drained = 0;
  const player = createPlayer({ onDrained: () => drained++ });
  player.unlock();
  fa.ctxHolder.ctx.currentTime = 1;
  assert.equal(player.isBusy(), false);
  player.startSentence(1, 22050);
  for (let k = 0; k < 5; k++) player.pushChunk(toneChunk(2205));
  player.endSentence(1);
  player.startSentence(2, 22050);
  for (let k = 0; k < 3; k++) player.pushChunk(toneChunk(2205));
  player.endSentence(2);

  const starts = fa.log.starts;
  assert.equal(starts.length, 8);
  assert.ok(Math.abs(starts[0].t - (1 + PLAYBACK.leadS)) < 1e-12);
  for (let k = 1; k < starts.length; k++) {
    const gapSamples = (starts[k].t - (starts[k - 1].t + starts[k - 1].dur)) * 22050;
    assert.ok(Math.abs(gapSamples) < 1, `gap ${gapSamples} samples before chunk ${k}`);
  }
  // Fade-in only on the first chunk of each sentence.
  assert.equal(starts[0].first, 0);
  assert.equal(starts[5].first, 0);
  assert.ok(starts[1].first > 0 && starts[4].first > 0 && starts[6].first > 0);
  // Fade-out ramps end exactly at each sentence end.
  const end1 = starts[4].t + starts[4].dur;
  const end2 = starts[7].t + starts[7].dur;
  const rampEnds = fa.log.ramps.filter((r) => r[0] === "ramp").map((r) => r[2]);
  assert.deepEqual(rampEnds, [end1, end2]);

  assert.equal(player.isBusy(), true);
  assert.ok(Math.abs(player.remainingMs() - (end2 - 1) * 1000) < 1e-6);
  assert.equal(player.level(), 0.25);
  fa.advance(end1);
  assert.equal(drained, 0);
  assert.equal(player.isBusy(), true);
  fa.advance(end2);
  assert.equal(drained, 1);
  assert.equal(player.isBusy(), false);
  assert.equal(player.level(), 0);
  assert.equal(fa.log.gainsDisconnected, 2);
});

test("player starts a new burst with the lead after the queue ran dry", () => {
  const fa = fakeAudio();
  const player = createPlayer();
  player.unlock();
  player.startSentence(1, 22050);
  player.pushChunk(toneChunk(2205));
  const end = fa.log.starts[0].t + 0.1;
  fa.advance(end + 0.5);
  player.pushChunk(toneChunk(2205));
  assert.ok(Math.abs(fa.log.starts[1].t - (end + 0.5 + PLAYBACK.leadS)) < 1e-12);
  assert.equal(fa.log.starts[1].first, 0, "burst restart fades in");
});

test("stopAll stops everything and ignores late ended events", () => {
  const fa = fakeAudio();
  let drained = 0;
  const player = createPlayer({ onDrained: () => drained++ });
  player.unlock();
  player.startSentence(1, 22050);
  player.pushChunk(toneChunk(2205));
  player.pushChunk(toneChunk(2205));
  player.stopAll();
  assert.ok(fa.sources.every((s) => s.stopped));
  assert.equal(player.isBusy(), false);
  fa.advance(100);
  assert.equal(drained, 0);
  // Frames after stopAll without a new tts_start are dropped.
  const warn = console.warn;
  console.warn = () => {};
  player.pushChunk(toneChunk(2205));
  console.warn = warn;
  assert.equal(fa.log.starts.length, 2);
  // The next sentence starts fresh from now plus the lead.
  player.startSentence(1, 22050);
  player.pushChunk(toneChunk(2205));
  assert.ok(Math.abs(fa.log.starts[2].t - (100 + PLAYBACK.leadS)) < 1e-12);
});

test("player ignores a sentence with a bad sample rate", () => {
  fakeAudio();
  const player = createPlayer();
  player.unlock();
  const warn = console.warn;
  console.warn = () => {};
  assert.equal(player.startSentence(1, 0), false);
  player.pushChunk(toneChunk(10));
  console.warn = warn;
  assert.equal(player.isBusy(), false);
});

test("stopAll disconnects the gains of earlier sentences that are still playing", () => {
  const fa = fakeAudio();
  const player = createPlayer();
  player.unlock();
  player.startSentence(1, 22050);
  player.pushChunk(toneChunk(2205));
  player.endSentence(1);
  player.startSentence(2, 22050);
  player.pushChunk(toneChunk(2205));
  assert.equal(player.liveGains, 2);
  player.stopAll();
  assert.equal(player.liveGains, 0);
  assert.equal(fa.log.gainsDisconnected, 2);
  // Late ended events do not disconnect anything twice.
  fa.advance(100);
  assert.equal(fa.log.gainsDisconnected, 2);
});

test("gains are released after a normal drain", () => {
  const fa = fakeAudio();
  const player = createPlayer();
  player.unlock();
  player.startSentence(1, 22050);
  player.pushChunk(toneChunk(2205));
  player.endSentence(1);
  fa.advance(10);
  assert.equal(player.liveGains, 0);
  assert.equal(fa.log.gainsDisconnected, 1);
});

test("closeOpenSentence fades a sentence without tts_end out at its scheduled end", () => {
  const fa = fakeAudio();
  const player = createPlayer();
  player.unlock();
  player.startSentence(3, 22050);
  player.pushChunk(toneChunk(2205));
  player.pushChunk(toneChunk(2205));
  const last = fa.log.starts[1];
  const end = last.t + last.dur;
  assert.equal(fa.log.ramps.length, 0);
  player.closeOpenSentence();
  assert.deepEqual(fa.log.ramps, [["set", 1, end - PLAYBACK.fadeS], ["ramp", 0, end]]);
  // Closing again, or a late tts_end, does nothing more.
  player.closeOpenSentence();
  player.endSentence(3);
  assert.equal(fa.log.ramps.length, 2);
  fa.advance(end);
  assert.equal(player.liveGains, 0);
});

test("headroomDb subtracts only positive shelf gains", () => {
  assert.equal(headroomDb({ low_shelf_db: -6, high_shelf_db: 0 }), -0);
  assert.equal(headroomDb({ low_shelf_db: -6, high_shelf_db: 4 }), -4);
  assert.equal(headroomDb({ low_shelf_db: 3, high_shelf_db: 24 }), -27);
  assert.equal(dbToGain(0), 1);
  assert.ok(Math.abs(dbToGain(-6) - 0.501187) < 1e-6);
  assert.ok(Math.abs(dbToGain(-20) - 0.1) < 1e-12);
});

test("player builds sentence -> lowshelf -> highshelf -> master -> analyser -> destination", () => {
  const fa = fakeAudio();
  const player = createPlayer();
  player.unlock();
  player.startSentence(1, 22050);
  const [low, high] = fa.log.filters;
  assert.equal(low.type, "lowshelf");
  assert.equal(high.type, "highshelf");
  for (const edge of [["filter0", "filter1"], ["filter1", "master"], ["master", "analyser"],
    ["analyser", "destination"], ["sentence", "filter0"]]) {
    assert.ok(fa.log.graph.some((e) => e[0] === edge[0] && e[1] === edge[1]), `missing ${edge}`);
  }
  assert.ok(!fa.log.graph.some((e) => e[0] === "sentence" && e[1] === "analyser"));
});

test("EQ settings made before the context exists apply on creation, later ones live", () => {
  const fa = fakeAudio();
  const player = createPlayer();
  assert.equal(player.getEq(), AUDIO_EQ_DEFAULTS);
  player.setEq({ low_shelf_hz: 200, low_shelf_db: -8, high_shelf_hz: 4000, high_shelf_db: 3 });
  assert.equal(fa.log.filters.length, 0);
  player.unlock();
  const [low, high] = fa.log.filters;
  assert.equal(low.frequency.value, 200);
  assert.equal(low.gain.value, -8);
  assert.equal(high.frequency.value, 4000);
  assert.equal(high.gain.value, 3);
  assert.ok(Math.abs(fa.log.master.gain.value - dbToGain(-3)) < 1e-12);
  // The initial apply uses .value, not automation.
  assert.equal(fa.log.targets.length, 0);
  fa.ctxHolder.ctx.currentTime = 4;
  player.setEq({ ...player.getEq(), high_shelf_db: 0 });
  assert.equal(player.getEq().high_shelf_db, 0);
  const tau = PLAYBACK.eqTimeConstantS;
  assert.deepEqual(fa.log.targets, [
    ["filter0.frequency", 200, 4, tau], ["filter0.gain", -8, 4, tau],
    ["filter1.frequency", 4000, 4, tau], ["filter1.gain", 0, 4, tau],
    ["master.gain", 1, 4, tau],
  ]);
  // Params are left alone by a glide; the stored eq is the source of truth.
  assert.equal(high.gain.value, 3);
  // One set of filters, however often settings change.
  assert.equal(fa.log.filters.length, 2);
});

test("default EQ on a fresh player cuts bass and needs no headroom", () => {
  const fa = fakeAudio();
  const player = createPlayer();
  player.unlock();
  const [low, high] = fa.log.filters;
  assert.deepEqual([low.frequency.value, low.gain.value, high.frequency.value, high.gain.value], [250, -6, 3500, 0]);
  assert.equal(fa.log.master.gain.value, 1);
});

test("eqParamValues and the glide time constant", () => {
  assert.deepEqual(eqParamValues({ low_shelf_hz: 250, low_shelf_db: -6.5, high_shelf_hz: 3500, high_shelf_db: 2 }),
    { lowHz: 250, lowDb: -6.5, highHz: 3500, highDb: 2, master: dbToGain(-2) });
  assert.ok(PLAYBACK.eqTimeConstantS >= 0.005 && PLAYBACK.eqTimeConstantS <= 0.05);
});

test("EQ changes on a suspended context are applied directly", () => {
  const fa = fakeAudio();
  const player = createPlayer();
  player.unlock();
  fa.ctxHolder.ctx.state = "suspended";
  player.setEq({ low_shelf_hz: 300, low_shelf_db: -6.5, high_shelf_hz: 3500, high_shelf_db: 0 });
  assert.equal(fa.log.targets.length, 0);
  assert.equal(fa.log.filters[0].gain.value, -6.5);
  assert.equal(fa.log.filters[0].frequency.value, 300);
});
