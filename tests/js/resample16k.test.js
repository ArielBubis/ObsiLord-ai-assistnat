import { test } from "node:test";
import assert from "node:assert/strict";

import {
  createResampler, createFramer, floatToInt16, downmix, kernelTable, besselI0, validInputRate,
  createFirstAudioDetector, FIRST_AUDIO_FALLBACK_S, TARGET_RATE, FRAME_SAMPLES, RESAMPLER,
} from "../../web/resample16k.js";
import { pcm16ToFloat32 } from "../../web/audio.js";

function sine(rate, freq, seconds, amp = 0.5, phase = 0.3) {
  const n = Math.round(rate * seconds);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) out[i] = amp * Math.sin((2 * Math.PI * freq * i) / rate + phase);
  return out;
}

// Deterministic block sizes (LCG), 1 to maxBlock samples each.
function blockSizes(total, maxBlock, seed = 12345) {
  const sizes = [];
  let s = seed;
  let left = total;
  while (left > 0) {
    s = (s * 1103515245 + 12345) % 2147483648;
    const size = Math.min(left, 1 + (s % maxBlock));
    sizes.push(size);
    left -= size;
  }
  return sizes;
}

// Feeds input in blocks, copies every output block, then flushes.
function run(rs, input, sizes) {
  const parts = [];
  let off = 0;
  for (const size of sizes) {
    parts.push(rs.process(input.subarray(off, off + size)).slice());
    off += size;
  }
  parts.push(rs.flush());
  const total = parts.reduce((a, p) => a + p.length, 0);
  const out = new Float32Array(total);
  let pos = 0;
  for (const p of parts) {
    out.set(p, pos);
    pos += p.length;
  }
  return out;
}

// Fixed-size blocks (the last one shorter), like render quanta.
function fixedBlocks(total, size) {
  const sizes = [];
  for (let left = total; left > 0; left -= size) sizes.push(Math.min(size, left));
  return sizes;
}

function resampleAll(inRate, input, block = 128) {
  return run(createResampler(inRate, TARGET_RATE), input, fixedBlocks(input.length, block));
}

// Frequency from upward zero crossings over the steady middle part.
function zeroCrossingHz(samples, rate, skip) {
  const end = samples.length - skip;
  let first = -1;
  let last = -1;
  let count = 0;
  for (let i = skip + 1; i < end; i++) {
    if (samples[i - 1] < 0 && samples[i] >= 0) {
      // Linear interpolation of the crossing point.
      const t = i - 1 + samples[i - 1] / (samples[i - 1] - samples[i]);
      if (first < 0) first = t;
      else count++;
      last = t;
    }
  }
  return (count * rate) / (last - first);
}

// Magnitude of the DFT at one frequency (Goertzel), normalized to amplitude.
function toneAmplitude(samples, rate, freq) {
  const w = (2 * Math.PI * freq) / rate;
  let re = 0;
  let im = 0;
  for (let i = 0; i < samples.length; i++) {
    re += samples[i] * Math.cos(w * i);
    im -= samples[i] * Math.sin(w * i);
  }
  return (2 * Math.hypot(re, im)) / samples.length;
}

function rmsOf(samples) {
  let sum = 0;
  for (const v of samples) sum += v * v;
  return Math.sqrt(sum / samples.length);
}

// Middle part, away from the start and end transients.
function middle(samples, edge) {
  return samples.subarray(edge, samples.length - edge);
}

const db = (ratio) => 20 * Math.log10(ratio);

test("constants: 16 kHz output, 100 ms frames, 7 kHz cutoff for 16 kHz", () => {
  assert.equal(TARGET_RATE, 16000);
  assert.equal(FRAME_SAMPLES, 1600);
  assert.equal(RESAMPLER.cutoffRatio * TARGET_RATE, 7000);
});

test("Bessel I0 and the kernel table", () => {
  assert.equal(besselI0(0), 1);
  assert.ok(Math.abs(besselI0(1) - 1.2660658777520082) < 1e-12);
  assert.ok(Math.abs(besselI0(7) - 168.59390851028968) < 1e-9);
  const t = kernelTable();
  assert.equal(t[0], 1);
  // Zero crossings of the sinc at whole multiples of tableRes.
  for (let z = 1; z < RESAMPLER.zeroCrossings; z++) assert.ok(Math.abs(t[z * RESAMPLER.tableRes]) < 1e-6);
  assert.equal(t[t.length - 1], 0);
  assert.equal(t[t.length - 2], 0);
  assert.equal(kernelTable(), t, "cached");
});

test("input rates outside 8 to 192 kHz are rejected", () => {
  for (const r of [7999, 192001, 0, -48000, NaN, Infinity, "48000"]) {
    assert.equal(validInputRate(r), false);
    assert.throws(() => createResampler(r), RangeError);
  }
  for (const r of [8000, 16000, 22050, 44100, 48000, 96000, 192000]) assert.equal(validInputRate(r), true);
});

for (const inRate of [48000, 44100]) {
  for (const freq of [440, 1000]) {
    test(`${inRate} Hz input: a ${freq} Hz sine keeps its frequency and level`, () => {
      const out = resampleAll(inRate, sine(inRate, freq, 1));
      const steady = middle(out, 400);
      const zc = zeroCrossingHz(steady, TARGET_RATE, 0);
      assert.ok(Math.abs(zc - freq) < 0.05, `zero crossings say ${zc} Hz`);
      // The DFT peak is at freq, not 1 Hz either side.
      const a = toneAmplitude(steady, TARGET_RATE, freq);
      assert.ok(a > 10 * toneAmplitude(steady, TARGET_RATE, freq * 1.05));
      assert.ok(Math.abs(db(a / 0.5)) < 0.05, `level ${db(a / 0.5).toFixed(3)} dB`);
      assert.ok(Math.abs(db(rmsOf(steady) / (0.5 / Math.SQRT2))) < 0.05);
    });
  }
}

test("8 kHz input is upsampled: 440 Hz stays 440 Hz", () => {
  const out = resampleAll(8000, sine(8000, 440, 1));
  assert.equal(out.length, 16000);
  const steady = middle(out, 400);
  assert.ok(Math.abs(zeroCrossingHz(steady, TARGET_RATE, 0) - 440) < 0.05);
  assert.ok(Math.abs(db(rmsOf(steady) / (0.5 / Math.SQRT2))) < 0.05);
});

test("a 12 kHz tone at 48 kHz is attenuated by at least 30 dB (no alias at 4 kHz)", () => {
  const out = resampleAll(48000, sine(48000, 12000, 1));
  const steady = middle(out, 400);
  const level = db(rmsOf(steady) / (0.5 / Math.SQRT2));
  assert.ok(level < -30, `${level.toFixed(1)} dB`);
  // In practice far more: the alias would land at 4 kHz.
  assert.ok(db(toneAmplitude(steady, TARGET_RATE, 4000) / 0.5) < -60);
});

test("everything from 8.5 kHz up is at least 60 dB down at 44.1 and 48 kHz", () => {
  for (const inRate of [44100, 48000]) {
    for (const freq of [8500, 9000, 10000, 12000, 15000, 20000]) {
      const out = resampleAll(inRate, sine(inRate, freq, 0.5));
      const level = db(rmsOf(middle(out, 400)) / (0.5 / Math.SQRT2));
      assert.ok(level < -60, `${inRate} Hz in, ${freq} Hz: ${level.toFixed(1)} dB`);
    }
  }
});

test("the passband is flat to 5 kHz (within 0.1 dB)", () => {
  for (const freq of [100, 300, 2000, 4000, 5000]) {
    const out = resampleAll(48000, sine(48000, freq, 0.5));
    const level = db(rmsOf(middle(out, 400)) / (0.5 / Math.SQRT2));
    assert.ok(Math.abs(level) < 0.1, `${freq} Hz: ${level.toFixed(3)} dB`);
  }
});

test("output length matches the input duration within 1 sample, across many small blocks", () => {
  for (const inRate of [8000, 16000, 22050, 32000, 44100, 48000, 96000, 192000]) {
    for (const seconds of [0.001, 0.1, 1.2345, 3]) {
      const n = Math.round(inRate * seconds);
      const input = sine(inRate, 300, seconds);
      for (const maxBlock of [1, 37, 128, 1000]) {
        if (maxBlock === 1 && n > 5000) continue;
        const out = run(createResampler(inRate), input, blockSizes(n, maxBlock));
        const exact = (n * TARGET_RATE) / inRate;
        assert.equal(out.length, Math.ceil(exact), `${inRate} Hz, ${n} samples, blocks <= ${maxBlock}`);
        assert.ok(Math.abs(out.length - exact) < 1);
      }
    }
  }
});

test("block size does not change the output: state carries across process() calls", () => {
  for (const inRate of [44100, 48000, 22050]) {
    const input = sine(inRate, 523.25, 0.75, 0.7);
    const whole = run(createResampler(inRate), input, [input.length]);
    const by128 = run(createResampler(inRate), input, fixedBlocks(input.length, 128));
    const random = run(createResampler(inRate), input, blockSizes(input.length, 300, 99));
    const tiny = run(createResampler(inRate), input, blockSizes(input.length, 3, 5));
    assert.deepEqual(by128, whole);
    assert.deepEqual(random, whole);
    assert.deepEqual(tiny, whole);
  }
});

test("no discontinuity at block boundaries: sample steps stay within the sine's slope", () => {
  const out = run(createResampler(44100), sine(44100, 1000, 1, 0.5), blockSizes(44100, 200, 3));
  const steady = middle(out, 200);
  const maxStep = 0.5 * 2 * Math.sin((Math.PI * 1000) / TARGET_RATE) * 1.001;
  for (let i = 1; i < steady.length; i++) {
    assert.ok(Math.abs(steady[i] - steady[i - 1]) <= maxStep, `step at ${i}`);
  }
});

test("output trails the input by the kernel half-width only", () => {
  const rs = createResampler(48000);
  const input = sine(48000, 440, 1);
  let produced = 0;
  let fed = 0;
  for (const size of blockSizes(input.length, 128, 11)) {
    produced += rs.process(input.subarray(fed, fed + size)).length;
    fed += size;
    const expected = (fed * TARGET_RATE) / 48000;
    const lag = expected - produced;
    assert.ok(lag >= 0 && lag <= ((rs.latencySamples + 1) * TARGET_RATE) / 48000 + 1, `lag ${lag}`);
  }
  assert.equal(rs.latencySamples, 55);
});

test("DC passes at exactly unity gain for 44.1 kHz (every fractional position)", () => {
  const input = new Float32Array(44100).fill(0.5);
  const out = resampleAll(44100, input);
  for (const v of middle(out, 100)) assert.ok(Math.abs(v - 0.5) < 1e-6);
});

test("flush resets: the next recording matches a fresh resampler", () => {
  const rs = createResampler(48000);
  run(rs, sine(48000, 200, 0.3), [5000, 9400]);
  assert.equal(rs.inputCount, 0);
  const again = run(rs, sine(48000, 700, 0.2), [128, 128, 9344]);
  const fresh = run(createResampler(48000), sine(48000, 700, 0.2), [9600]);
  assert.deepEqual(again, fresh);
  assert.equal(rs.flush().length, 0, "flush with no input is empty");
});

test("empty and missing blocks are harmless", () => {
  const rs = createResampler(48000);
  assert.equal(rs.process(new Float32Array(0)).length, 0);
  assert.equal(rs.process(null).length, 0);
  assert.equal(rs.flush().length, 0);
});

test("floatToInt16 clips and rounds", () => {
  assert.equal(floatToInt16(0), 0);
  assert.equal(floatToInt16(1), 32767);
  assert.equal(floatToInt16(-1), -32768);
  assert.equal(floatToInt16(2.5), 32767);
  assert.equal(floatToInt16(-3), -32768);
  assert.equal(floatToInt16(Infinity), 32767);
  assert.equal(floatToInt16(-Infinity), -32768);
  assert.equal(floatToInt16(NaN), 0);
  assert.equal(floatToInt16(0.5), 16384);
  assert.equal(floatToInt16(-0.5), -16384);
  assert.equal(floatToInt16(0.25 / 32768), 0);
  assert.equal(floatToInt16(0.75 / 32768), 1);
});

test("floatToInt16 inverts pcm16ToFloat32 for every 16-bit value", () => {
  const bytes = new Uint8Array(65536 * 2);
  const view = new DataView(bytes.buffer);
  for (let k = -32768; k <= 32767; k++) view.setInt16((k + 32768) * 2, k, true);
  const { samples } = pcm16ToFloat32(bytes);
  for (let k = -32768; k <= 32767; k++) {
    const back = floatToInt16(samples[k + 32768]);
    if (back !== k) assert.fail(`${k} came back as ${back}`);
  }
});

test("framer: 1600-sample frames, remainder carried, partial last frame on flush", () => {
  const f = createFramer();
  assert.deepEqual(f.push(new Float32Array(1000).fill(0.5)), []);
  assert.equal(f.pending, 1000);
  const frames = f.push(new Float32Array(700).fill(-0.5));
  assert.equal(frames.length, 1);
  assert.ok(frames[0] instanceof ArrayBuffer);
  assert.equal(frames[0].byteLength, 3200);
  const view = new DataView(frames[0]);
  assert.equal(view.getInt16(0, true), 16384);
  assert.equal(view.getInt16(999 * 2, true), 16384);
  assert.equal(view.getInt16(1000 * 2, true), -16384);
  assert.equal(view.getInt16(1599 * 2, true), -16384);
  assert.equal(f.pending, 100);
  const last = f.flush();
  assert.equal(last.byteLength, 200);
  assert.equal(new DataView(last).getInt16(0, true), -16384);
  assert.equal(f.flush(), null);
  assert.equal(f.pending, 0);
});

test("framer writes little-endian bytes", () => {
  const f = createFramer(2);
  const [frame] = f.push(new Float32Array([0.5, -1]));
  // 16384 = 0x4000, -32768 = 0x8000.
  assert.deepEqual([...new Uint8Array(frame)], [0x00, 0x40, 0x00, 0x80]);
});

test("framer: a big block gives several frames, exact multiples leave nothing", () => {
  const f = createFramer();
  const frames = f.push(new Float32Array(1600 * 3 + 5));
  assert.equal(frames.length, 3);
  assert.equal(f.pending, 5);
  f.reset();
  assert.equal(f.flush(), null);
  assert.equal(f.push(new Float32Array(1600)).length, 1);
  assert.equal(f.flush(), null);
});

test("framer frames are separate buffers, so transferring one leaves the rest intact", () => {
  const f = createFramer(4);
  const frames = f.push(new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]));
  assert.equal(frames.length, 2);
  structuredClone(frames[0], { transfer: [frames[0]] });
  assert.equal(frames[0].byteLength, 0);
  assert.equal(frames[1].byteLength, 8);
  assert.equal(new DataView(frames[1]).getInt16(0, true), floatToInt16(0.5));
});

test("downmix: one channel as is, several averaged, none empty", () => {
  const left = new Float32Array([1, 0.5, -1]);
  const right = new Float32Array([0, 0.5, 1]);
  assert.equal(downmix([left]), left);
  assert.deepEqual([...downmix([left, right])], [0.5, 0.5, 0]);
  const scratch = new Float32Array(8);
  const mono = downmix([left, right], scratch);
  assert.equal(mono.buffer, scratch.buffer);
  assert.equal(mono.length, 3);
  assert.equal(downmix([]).length, 0);
  assert.equal(downmix(undefined).length, 0);
});

test("first-audio detector: any non-zero sample in any channel, else after the fallback time", () => {
  assert.equal(FIRST_AUDIO_FALLBACK_S, 0.3);
  const zero = new Float32Array(128);
  const d = createFirstAudioDetector(16000);
  assert.equal(d.update([zero]), false);
  const right = new Float32Array(128);
  right[5] = -1e-6;
  assert.equal(d.update([zero, right]), true, "a tiny value in the second channel counts");
  assert.equal(d.update([right]), false, "true only once");
  d.reset();
  // 0.3 s at 16 kHz = 4800 samples of zeros.
  let fired = -1;
  for (let i = 0; i < 40; i++) {
    if (d.update([zero])) {
      fired = i;
      break;
    }
  }
  assert.equal(fired, Math.ceil(4800 / 128) - 1);
  assert.equal(d.update([]), false);
  assert.equal(d.update(null), false);
});
