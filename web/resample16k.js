// Mic capture DSP, pure (no DOM, no Web Audio). Used by capture-worklet.js on
// the audio thread and by node:test.
//
// Pipeline per render block: downmix to mono, low-pass and resample from the
// context rate to 16 kHz in one step, convert to PCM16 little-endian, cut
// into 1600-sample (100 ms, 3200-byte) frames.
//
// Resampler: band-limited interpolation with a Kaiser-windowed sinc kernel.
// Each output sample sits at a fractional input position; the kernel is read
// from a precomputed table at that offset (linear interpolation between
// table entries), so any input rate works (44.1 kHz and 48 kHz alike) and
// the low-pass is part of the same filter. The cutoff is 7 kHz for a 16 kHz
// output (0.4375 of the lower rate), about 6 kHz flat, at least 70 dB down
// from 8 kHz up. The output position is tracked as an exact integer
// fraction, so long recordings do not drift. State (recent input and the
// fractional position) carries across process() calls.

export const TARGET_RATE = 16000;
export const FRAME_SAMPLES = 1600;
export const MIN_INPUT_RATE = 8000;
export const MAX_INPUT_RATE = 192000;

export const RESAMPLER = Object.freeze({
  // Cutoff as a fraction of the lower of the two rates: 7000 Hz at 16 kHz out.
  cutoffRatio: 0.4375,
  // Kernel half-width in zero crossings of the sinc. With beta 7 this gives
  // a transition band of about 2 kHz around the cutoff (6 to 8 kHz).
  zeroCrossings: 16,
  kaiserBeta: 7,
  // Table entries per zero crossing.
  tableRes: 512,
});

// Modified Bessel function of the first kind, order 0 (Kaiser window).
export function besselI0(x) {
  const q = (x * x) / 4;
  let sum = 1;
  let term = 1;
  for (let k = 1; k < 100; k++) {
    term *= q / (k * k);
    sum += term;
    if (term < sum * 1e-15) break;
  }
  return sum;
}

const tables = new Map();

// Kernel values at distances 0, 1/res, 2/res, ... zero crossings, plus two
// zero guard entries so interpolation past the end reads 0.
export function kernelTable(opts = RESAMPLER) {
  const key = `${opts.zeroCrossings}/${opts.kaiserBeta}/${opts.tableRes}`;
  let table = tables.get(key);
  if (table) return table;
  const n = opts.zeroCrossings * opts.tableRes;
  table = new Float32Array(n + 2);
  const i0Beta = besselI0(opts.kaiserBeta);
  for (let k = 0; k < n; k++) {
    const u = k / opts.tableRes;
    const sinc = k === 0 ? 1 : Math.sin(Math.PI * u) / (Math.PI * u);
    const r = u / opts.zeroCrossings;
    table[k] = (sinc * besselI0(opts.kaiserBeta * Math.sqrt(1 - r * r))) / i0Beta;
  }
  tables.set(key, table);
  return table;
}

export function validInputRate(rate) {
  return Number.isFinite(rate) && rate >= MIN_INPUT_RATE && rate <= MAX_INPUT_RATE;
}

// Stateful resampler. process(block) returns the output samples that are
// ready (a view into an internal buffer, valid until the next call; copy it
// to keep it). Output lags the input by the kernel half-width (about 1 ms).
// flush() pads the end with silence, returns the remaining samples as a new
// array, and resets for the next recording. The total output for N input
// samples is ceil(N * outRate / inRate).
export function createResampler(inRate, outRate = TARGET_RATE, opts = RESAMPLER) {
  if (!validInputRate(inRate)) throw new RangeError(`unsupported input rate: ${inRate}`);
  if (!(Number.isFinite(outRate) && outRate > 0)) throw new RangeError(`bad output rate: ${outRate}`);
  const inR = Math.round(inRate);
  const outR = Math.round(outRate);
  const table = kernelTable(opts);
  const limit = opts.zeroCrossings * opts.tableRes;
  // Cutoff in cycles per input sample.
  const fc = (opts.cutoffRatio * Math.min(inR, outR)) / inR;
  // Table index per input sample of distance.
  const scale = 2 * fc * opts.tableRes;
  // Taps per side, in input samples.
  const half = Math.ceil(opts.zeroCrossings / (2 * fc));
  const taps = 2 * half;

  // Input history. buf[0] holds input sample number bufStart (negative
  // numbers are the silence before the first sample).
  let buf = new Float32Array(Math.max(4096, 4 * taps));
  let bufStart = 0;
  let bufLen = 0;
  // Position of the next output sample in input samples: ipos + fnum / outR.
  let ipos = 0;
  let fnum = 0;
  let inCount = 0;
  let out = new Float32Array(1024);

  function reset() {
    buf.fill(0, 0, half);
    bufStart = -half;
    bufLen = half;
    ipos = 0;
    fnum = 0;
    inCount = 0;
  }

  function append(block, count) {
    if (bufLen + count > buf.length) {
      const bigger = new Float32Array(Math.max(buf.length * 2, bufLen + count));
      bigger.set(buf.subarray(0, bufLen));
      buf = bigger;
    }
    if (block) buf.set(block.subarray(0, count), bufLen);
    else buf.fill(0, bufLen, bufLen + count);
    bufLen += count;
  }

  // One output sample at ipos + frac. Taps are input samples
  // ipos - half + 1 .. ipos + half. Dividing by the weight sum keeps the DC
  // gain at exactly 1 for every fractional position.
  function sampleAt(frac) {
    const base = ipos - half + 1 - bufStart;
    let acc = 0;
    let wsum = 0;
    for (let t = 0; t < taps; t++) {
      let d = frac + half - 1 - t;
      if (d < 0) d = -d;
      const x = d * scale;
      const k = x | 0;
      if (k >= limit) continue;
      const w = table[k] + (table[k + 1] - table[k]) * (x - k);
      acc += buf[base + t] * w;
      wsum += w;
    }
    return wsum !== 0 ? acc / wsum : 0;
  }

  // Emits every output whose taps are all buffered and whose position is
  // before endPos, then drops input no later output needs.
  function emit(endPos) {
    const avail = bufStart + bufLen;
    let n = 0;
    while (ipos + half < avail && ipos < endPos) {
      if (n === out.length) {
        const bigger = new Float32Array(out.length * 2);
        bigger.set(out);
        out = bigger;
      }
      out[n++] = sampleAt(fnum / outR);
      fnum += inR;
      if (fnum >= outR) {
        const whole = Math.floor(fnum / outR);
        ipos += whole;
        fnum -= whole * outR;
      }
    }
    const drop = Math.min(bufLen, ipos - half + 1 - bufStart);
    if (drop > 0) {
      buf.copyWithin(0, drop, bufLen);
      bufLen -= drop;
      bufStart += drop;
    }
    return out.subarray(0, n);
  }

  reset();

  return {
    inRate: inR,
    outRate: outR,
    // Input samples between the newest input and the newest output.
    latencySamples: half,
    process(block) {
      const count = block ? block.length : 0;
      if (count > 0) {
        append(block, count);
        inCount += count;
      }
      return emit(Infinity);
    },
    flush() {
      append(null, half);
      const tail = emit(inCount).slice();
      reset();
      return tail;
    },
    reset,
    // Total input samples since the last reset (test aid).
    get inputCount() {
      return inCount;
    },
  };
}

// Mono mix of one render block (array of per-channel Float32Arrays). One
// channel is returned as is; more are averaged into out (reused if big
// enough). No channels gives an empty array.
export function downmix(channels, out = null) {
  if (!channels || channels.length === 0) return new Float32Array(0);
  if (channels.length === 1) return channels[0];
  const n = channels[0].length;
  const mono = out && out.length >= n ? out.subarray(0, n) : new Float32Array(n);
  const k = 1 / channels.length;
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < channels.length; c++) s += channels[c][i];
    mono[i] = s * k;
  }
  return mono;
}

// Float sample to a PCM16 value, clipped. The scale is 32768, the inverse of
// pcm16ToFloat32 in audio.js, so k / 32768 converts back to k exactly.
export function floatToInt16(v) {
  if (!(v === v)) return 0; // NaN
  const s = Math.round(v * 32768);
  if (s > 32767) return 32767;
  if (s < -32768) return -32768;
  return s;
}

// Cuts a stream of float samples into PCM16 little-endian frames of
// frameSamples samples. push() returns the frames completed by this block
// (each its own ArrayBuffer, so it can be transferred); the remainder
// carries over. flush() returns the partial last frame (shorter, not
// padded) or null if empty.
export function createFramer(frameSamples = FRAME_SAMPLES) {
  const frameBytes = frameSamples * 2;
  let buf = new ArrayBuffer(frameBytes);
  let view = new DataView(buf);
  let count = 0;
  return {
    push(samples) {
      const frames = [];
      for (let i = 0; i < samples.length; i++) {
        view.setInt16(count * 2, floatToInt16(samples[i]), true);
        count++;
        if (count === frameSamples) {
          frames.push(buf);
          buf = new ArrayBuffer(frameBytes);
          view = new DataView(buf);
          count = 0;
        }
      }
      return frames;
    },
    flush() {
      if (count === 0) return null;
      const last = buf.slice(0, count * 2);
      count = 0;
      return last;
    },
    reset() {
      count = 0;
    },
    // Samples waiting for the current frame to fill.
    get pending() {
      return count;
    },
  };
}

// Decides when real audio starts: the first block with a non-zero sample in
// any channel, or, if the input stays all zeros (pre-roll silence or a
// muted mic), once fallbackS of input has passed, so callers waiting for it
// never stall. update() returns true exactly once.
export const FIRST_AUDIO_FALLBACK_S = 0.3;

export function createFirstAudioDetector(rate, fallbackS = FIRST_AUDIO_FALLBACK_S) {
  const limit = Math.max(1, Math.round(rate * fallbackS));
  let seen = 0;
  let done = false;
  return {
    update(channels) {
      if (done || !channels || channels.length === 0) return false;
      const n = channels[0].length;
      for (let c = 0; c < channels.length; c++) {
        const ch = channels[c];
        for (let i = 0; i < ch.length; i++) {
          if (ch[i] !== 0) {
            done = true;
            return true;
          }
        }
      }
      seen += n;
      if (seen >= limit) {
        done = true;
        return true;
      }
      return false;
    },
    reset() {
      seen = 0;
      done = false;
    },
  };
}
