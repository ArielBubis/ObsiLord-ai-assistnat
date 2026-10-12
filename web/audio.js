// Speech playback with Web Audio. The server streams each sentence as
// tts_start (with sample_rate), binary PCM16 mono little-endian frames, then
// tts_end. Every frame becomes an AudioBuffer at the announced rate (the
// context resamples) and is scheduled right after the previous one, so the
// chunks play back to back without gaps.
//
// Graph: source -> per-sentence gain (fades) -> lowshelf -> highshelf ->
// master (headroom) -> analyser -> destination. The analyser sits after the
// EQ so the mouth tracks what is heard (see face.js); the backend sends no
// mouth data. EQ settings come from GET /config/audio (see config.js).
//
// The top part of this file is pure (no DOM, no Web Audio) so node:test can
// cover it. createPlayer() only touches Web Audio when unlock() is called.

import { AUDIO_EQ_DEFAULTS } from "./config.js";

export const PLAYBACK = Object.freeze({
  // Delay before the first chunk of a burst, so the audio thread has time to
  // pick it up and the next chunks can queue behind it.
  leadS: 0.05,
  // Short ramp at the start and end of each sentence to avoid clicks.
  fadeS: 0.003,
  // Time constant for EQ changes while audio runs (slider moves), so a
  // jump in filter gain does not click.
  eqTimeConstantS: 0.02,
  // Analyser window: 1024 samples is about 21 ms at 48 kHz.
  fftSize: 1024,
  minRate: 8000,
  maxRate: 96000,
});

// Converts PCM16 little-endian bytes to Float32 samples in -1..1.
// -32768 maps to -1 exactly, 32767 to just under 1. An odd trailing byte
// cannot form a sample; it is returned in rest for the next chunk.
export function pcm16ToFloat32(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  const count = u8.length >> 1;
  const samples = new Float32Array(count);
  const view = new DataView(u8.buffer, u8.byteOffset, count * 2);
  for (let i = 0; i < count; i++) {
    samples[i] = view.getInt16(i * 2, true) / 32768;
  }
  return { samples, rest: u8.slice(count * 2) };
}

// Stateful wrapper that carries an odd byte from one chunk into the next, so
// a frame boundary in the middle of a sample does not shift the stream.
export function createPcm16Decoder() {
  let carry = new Uint8Array(0);
  return {
    decode(data) {
      const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
      let input = u8;
      if (carry.length > 0) {
        input = new Uint8Array(carry.length + u8.length);
        input.set(carry, 0);
        input.set(u8, carry.length);
      }
      const { samples, rest } = pcm16ToFloat32(input);
      carry = rest;
      return samples;
    },
    // Bytes held back waiting for their pair (0 or 1).
    get pending() {
      return carry.length;
    },
    reset() {
      carry = new Uint8Array(0);
    },
  };
}

export function validSampleRate(rate, cfg = PLAYBACK) {
  return Number.isInteger(rate) && rate >= cfg.minRate && rate <= cfg.maxRate;
}

// Where the next chunk starts. While earlier audio is still queued
// (nextTime in the future) the chunk goes right after it, gapless. Otherwise
// this is the first chunk of a burst and it starts leadS from now.
export function planStart(nextTime, now, leadS = PLAYBACK.leadS) {
  if (nextTime > now) return { start: nextTime, burst: false };
  return { start: now + leadS, burst: true };
}

// Seconds of queued audio left to play.
export function remainingS(nextTime, now) {
  return Math.max(0, nextTime - now);
}

// Linear fade-in over the first n samples, in place.
export function applyFadeIn(samples, n) {
  const len = Math.min(samples.length, Math.max(0, Math.floor(n)));
  for (let i = 0; i < len; i++) samples[i] *= i / len;
  return samples;
}

export function fadeSamples(rate, fadeS = PLAYBACK.fadeS) {
  return Math.max(1, Math.round(rate * fadeS));
}

// Gain ramp to 0 that ends exactly when the sentence ends. Returns null if
// the sentence already ended; starts the ramp now if it is partly past.
export function fadeOutRamp(endTime, now, fadeS = PLAYBACK.fadeS) {
  if (!(endTime > now)) return null;
  return { from: Math.max(now, endTime - fadeS), to: endTime };
}

// Simple headroom for the EQ: a shelf boost of +N dB can push peaks past
// full scale and clip, so the master gain comes down by the sum of all
// positive shelf gains. Cuts need no compensation. Returns dB (<= 0).
export function headroomDb(eq) {
  return -(Math.max(0, eq.low_shelf_db) + Math.max(0, eq.high_shelf_db));
}

export function dbToGain(db) {
  return 10 ** (db / 20);
}

// Target values for the EQ params. Pure.
export function eqParamValues(eq) {
  return {
    lowHz: eq.low_shelf_hz,
    lowDb: eq.low_shelf_db,
    highHz: eq.high_shelf_hz,
    highDb: eq.high_shelf_db,
    master: dbToGain(headroomDb(eq)),
  };
}

export function rms(samples) {
  const n = samples.length;
  if (n === 0) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += samples[i] * samples[i];
  return Math.sqrt(sum / n);
}

// Web Audio player. onDrained fires when the last scheduled chunk finishes
// (and again after each later burst drains).
export function createPlayer({ onDrained = () => {}, cfg = PLAYBACK } = {}) {
  let ctx = null;
  let analyser = null;
  let timeData = null;
  let nextTime = 0;
  // Bumped by stopAll so ended events from stopped sources are ignored.
  let gen = 0;
  const active = new Set();
  // Every sentence gain node still connected to the analyser.
  const gains = new Set();
  // Current sentence: { id, rate, gain, decoder, first, sources, ended, endTime }
  let sentence = null;
  let warnedOrphan = false;
  // EQ nodes, created once with the context.
  let lowShelf = null;
  let highShelf = null;
  let master = null;
  let eq = AUDIO_EQ_DEFAULTS;

  // Initial apply (context just created, nothing playing yet): plain .value
  // assignment. Later changes while the context runs glide there with
  // setTargetAtTime (eqTimeConstantS) so slider moves do not click. Scheduled
  // values do not reliably read back from AudioParam.value in Edge, so
  // getEq() reports the stored settings, never the params.
  function applyEq(smooth) {
    if (!ctx) return;
    const v = eqParamValues(eq);
    const pairs = [
      [lowShelf.frequency, v.lowHz], [lowShelf.gain, v.lowDb],
      [highShelf.frequency, v.highHz], [highShelf.gain, v.highDb],
      [master.gain, v.master],
    ];
    const glide = smooth && ctx.state === "running";
    for (const [param, value] of pairs) {
      if (glide) param.setTargetAtTime(value, ctx.currentTime, cfg.eqTimeConstantS);
      else param.value = value;
    }
  }

  function ensureContext() {
    if (ctx) return ctx;
    const Ctor = globalThis.AudioContext || globalThis.webkitAudioContext;
    if (!Ctor) return null;
    ctx = new Ctor({ latencyHint: "interactive" });
    analyser = ctx.createAnalyser();
    analyser.fftSize = cfg.fftSize;
    analyser.connect(ctx.destination);
    timeData = new Float32Array(analyser.fftSize);
    lowShelf = ctx.createBiquadFilter();
    lowShelf.type = "lowshelf";
    highShelf = ctx.createBiquadFilter();
    highShelf.type = "highshelf";
    master = ctx.createGain();
    lowShelf.connect(highShelf);
    highShelf.connect(master);
    master.connect(analyser);
    // Settings that arrived before the context existed apply now.
    applyEq(false);
    return ctx;
  }

  function releaseSentence(s) {
    if (s.ended && s.sources === 0 && s.gain) {
      s.gain.disconnect();
      gains.delete(s.gain);
      s.gain = null;
    }
  }

  function startSentence(id, rate) {
    endSentence(sentence ? sentence.id : null);
    if (!validSampleRate(rate, cfg)) {
      console.warn("audio: ignoring sentence with bad sample_rate", rate);
      sentence = null;
      return false;
    }
    if (!ensureContext()) return false;
    const gain = ctx.createGain();
    gain.connect(lowShelf);
    gains.add(gain);
    sentence = {
      id, rate, gain, decoder: createPcm16Decoder(), first: true, sources: 0, ended: false, endTime: 0,
    };
    return true;
  }

  function pushChunk(data) {
    if (!sentence || !sentence.gain) {
      if (!warnedOrphan) console.warn("audio: binary frame outside a sentence, dropped");
      warnedOrphan = true;
      return;
    }
    const s = sentence;
    const samples = s.decoder.decode(data);
    if (samples.length === 0) return;
    const { start, burst } = planStart(nextTime, ctx.currentTime, cfg.leadS);
    if (s.first || burst) applyFadeIn(samples, fadeSamples(s.rate, cfg.fadeS));
    s.first = false;
    const buffer = ctx.createBuffer(1, samples.length, s.rate);
    buffer.copyToChannel(samples, 0);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(s.gain);
    const myGen = gen;
    active.add(src);
    s.sources++;
    src.onended = () => {
      src.disconnect();
      if (myGen !== gen) return;
      active.delete(src);
      s.sources--;
      releaseSentence(s);
      if (active.size === 0) onDrained();
    };
    src.start(start);
    nextTime = start + buffer.duration;
    s.endTime = nextTime;
  }

  // Fade the tail of the sentence out so its last sample lands on 0.
  // id null closes whatever sentence is open (the server cancels TTS on error
  // paths without tts_end; the caller closes it on state idle or error).
  function endSentence(id) {
    if (!sentence || (id !== null && id !== sentence.id)) return;
    const s = sentence;
    sentence = null;
    s.ended = true;
    if (s.gain && ctx) {
      const ramp = fadeOutRamp(s.endTime, ctx.currentTime, cfg.fadeS);
      if (ramp) {
        s.gain.gain.setValueAtTime(1, ramp.from);
        s.gain.gain.linearRampToValueAtTime(0, ramp.to);
      }
    }
    releaseSentence(s);
  }

  function stopAll() {
    gen++;
    for (const src of active) {
      try {
        src.stop();
      } catch {
        // Already stopped.
      }
      src.disconnect();
    }
    active.clear();
    for (const g of gains) g.disconnect();
    gains.clear();
    sentence = null;
    nextTime = 0;
    warnedOrphan = false;
  }

  return {
    // Call from a user gesture (Send click, Enter, or a push-to-talk press)
    // so autoplay rules allow the context to run.
    unlock() {
      const c = ensureContext();
      if (c && c.state === "suspended") {
        c.resume().catch((err) => console.warn("audio: resume failed", err));
      }
    },
    // The shared AudioContext, created if needed (null without Web Audio).
    // Mic capture (audio_in.js) runs in it too, at its native rate.
    context() {
      return ensureContext();
    },
    startSentence,
    pushChunk,
    endSentence,
    stopAll,
    // EQ settings ({low_shelf_hz, low_shelf_db, high_shelf_hz, high_shelf_db},
    // already validated). Applied now if the context exists, else on creation.
    setEq(next) {
      eq = Object.freeze({ ...next });
      applyEq(true);
    },
    getEq() {
      return eq;
    },
    // Closes the open sentence, if any, as if its tts_end had arrived.
    closeOpenSentence() {
      endSentence(null);
    },
    // Test and debug aid: gain nodes still connected.
    get liveGains() {
      return gains.size;
    },
    // True while any chunk is scheduled or playing.
    isBusy() {
      return active.size > 0;
    },
    remainingMs() {
      return ctx ? remainingS(nextTime, ctx.currentTime) * 1000 : 0;
    },
    // Current playback RMS, 0 when nothing is playing.
    level() {
      if (!analyser || active.size === 0) return 0;
      analyser.getFloatTimeDomainData(timeData);
      return rms(timeData);
    },
  };
}
