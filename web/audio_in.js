// Microphone capture for push-to-talk.
//
// Each recording opens the mic with getUserMedia (echo cancellation, noise
// suppression, and auto gain on), connects it to the capture worklet in the
// shared playback AudioContext (audio.js), and stops the tracks again when
// the recording ends, so the browser's mic indicator is only on while
// talking. The worklet (capture-worklet.js) resamples to 16 kHz and posts
// PCM16 frames; this module hands them to the caller and keeps nothing.
//
// The first press shows the browser's permission prompt. Errors reach the
// caller as MicError with a message meant for the user.
//
// Web Audio and getUserMedia are injected (createMic options) so node:test
// can drive this with fakes. Importing the module touches nothing.

import { validInputRate, MIN_INPUT_RATE, MAX_INPUT_RATE } from "./resample16k.js";

export const WORKLET_URL = "/static/capture-worklet.js";
export const WORKLET_NAME = "capture-16k";

export const MIC_CONSTRAINTS = Object.freeze({
  audio: Object.freeze({
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
    channelCount: 1,
  }),
});

// A stop the worklet never answers (for example a suspended context) still
// ends the recording after this long.
export const STOP_TIMEOUT_MS = 500;

export const MIC_MESSAGES = Object.freeze({
  blocked: "Microphone blocked. Allow it in the browser and try again.",
  missing: "No microphone found. Connect one and try again.",
  inUse: "The microphone is in use or not working. Close other apps using it and try again.",
  unsupported: "Voice input is not supported in this browser.",
  failed: "The microphone failed to start.",
  rate: `Voice input needs an audio output rate between ${MIN_INPUT_RATE} and ${MAX_INPUT_RATE} Hz.`,
});

export class MicError extends Error {
  constructor(message, cause, { superseded = false } = {}) {
    super(message);
    this.name = "MicError";
    this.cause = cause;
    // A newer begin() was called while this one waited for the mic.
    this.superseded = superseded;
  }
}

// User-facing text for a getUserMedia (or setup) failure.
export function micErrorMessage(err) {
  if (err instanceof MicError) return err.message;
  switch (err && err.name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
    case "SecurityError":
      return MIC_MESSAGES.blocked;
    case "NotFoundError":
    case "DevicesNotFoundError":
    case "OverconstrainedError":
      return MIC_MESSAGES.missing;
    case "NotReadableError":
    case "TrackStartError":
    case "AbortError":
      return MIC_MESSAGES.inUse;
    case "NotSupportedError":
    case "TypeError":
      return MIC_MESSAGES.unsupported;
    default:
      return MIC_MESSAGES.failed;
  }
}

function stopTracks(stream) {
  if (!stream || typeof stream.getTracks !== "function") return;
  for (const track of stream.getTracks()) track.stop();
}

function defaultGetUserMedia(constraints) {
  const md = globalThis.navigator && globalThis.navigator.mediaDevices;
  if (!md || typeof md.getUserMedia !== "function") {
    return Promise.reject(new MicError(MIC_MESSAGES.unsupported));
  }
  return md.getUserMedia(constraints);
}

// getContext() returns the shared AudioContext (or null without Web Audio).
// begin({onFrame, onFirstAudio}) opens the mic and starts a recording;
// onFrame(ArrayBuffer) gets each PCM16 frame and onFirstAudio() runs once
// when the first audio reaches the worklet (from then on speech is
// captured). It resolves to a handle:
//   end()    flush the last partial frame through onFrame, then release the
//            mic; resolves when done
//   abort()  release the mic now; frames still in flight are dropped
// If begin() is called again before an earlier call got its stream, the
// earlier call closes its stream and rejects with err.superseded true.
export function createMic({
  getContext,
  getUserMedia = defaultGetUserMedia,
  NodeCtor = globalThis.AudioWorkletNode,
  workletUrl = WORKLET_URL,
  now = () => performance.now(),
  stopTimeoutMs = STOP_TIMEOUT_MS,
} = {}) {
  let ctx = null;
  let node = null;
  let nodeReady = null;
  let seq = 0;
  let session = null;
  // Latency of the last recording, for the report and debugging:
  //   gumMs        getUserMedia call to stream
  //   firstAudioMs begin() call to the first block reaching the worklet
  //   stopMs       end() or abort() call to the tracks being stopped
  const stats = { opens: 0, gumMs: null, firstAudioMs: null, stopMs: null };

  function onPortMessage(event) {
    const msg = event.data;
    const s = session;
    if (!msg || typeof msg !== "object" || !s || msg.id !== s.id) return;
    if (msg.type === "frame") {
      if (!s.closed && msg.buf instanceof ArrayBuffer) s.onFrame(msg.buf);
    } else if (msg.type === "first") {
      stats.firstAudioMs = now() - s.t0;
      if (!s.closed && s.onFirstAudio) s.onFirstAudio();
    } else if (msg.type === "end") {
      release(s);
    }
  }

  // Loads the worklet module and creates the node once per context. A
  // failure is not cached, so the next press tries again.
  function ensureNode(c) {
    if (nodeReady && ctx === c) return nodeReady;
    ctx = c;
    node = null;
    const ready = (async () => {
      if (!c.audioWorklet || typeof NodeCtor !== "function") {
        throw new MicError(MIC_MESSAGES.unsupported);
      }
      // The worklet's resampler would refuse this rate on the audio thread.
      if (!validInputRate(c.sampleRate)) throw new MicError(MIC_MESSAGES.rate);
      await c.audioWorklet.addModule(workletUrl);
      const n = new NodeCtor(c, WORKLET_NAME, {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [1],
        channelCount: 1,
        channelCountMode: "explicit",
        channelInterpretation: "speakers",
      });
      n.port.onmessage = onPortMessage;
      n.onprocessorerror = (event) => console.warn("mic: capture worklet failed", event);
      // The output is silent; the connection keeps the node in the rendered
      // graph in every browser.
      n.connect(c.destination);
      node = n;
      return n;
    })();
    nodeReady = ready;
    ready.catch(() => {
      if (nodeReady === ready) nodeReady = null;
    });
    return ready;
  }

  function release(s) {
    if (s.closed) return;
    s.closed = true;
    if (s.timer !== null) clearTimeout(s.timer);
    s.timer = null;
    try {
      s.source.disconnect();
    } catch {
      // Already disconnected.
    }
    stopTracks(s.stream);
    s.stream = null;
    if (session === s) session = null;
    stats.stopMs = now() - s.tStop;
    if (s.resolve) s.resolve();
  }

  function end(s) {
    if (s.ending) return s.ending;
    s.tStop = now();
    s.ending = new Promise((resolve) => {
      s.resolve = resolve;
    });
    if (s.closed) {
      s.resolve();
      return s.ending;
    }
    s.timer = setTimeout(() => {
      console.warn("mic: capture did not flush in time");
      release(s);
    }, stopTimeoutMs);
    if (node) node.port.postMessage({ type: "stop", id: s.id });
    return s.ending;
  }

  function abort(s) {
    if (s.closed) return;
    if (s.tStop === undefined) s.tStop = now();
    // The worklet stops converting; anything it still posts is ignored.
    if (node && !s.ending) node.port.postMessage({ type: "stop", id: s.id });
    release(s);
  }

  async function begin({ onFrame, onFirstAudio = null }) {
    const c = getContext();
    if (!c) throw new MicError(MIC_MESSAGES.unsupported);
    // Only the newest call may start the worklet: a press can come while
    // an earlier getUserMedia still waits (permission prompt).
    const id = ++seq;
    const t0 = now();
    let gum;
    try {
      gum = Promise.resolve(getUserMedia(MIC_CONSTRAINTS));
    } catch (err) {
      gum = Promise.reject(err);
    }
    const [streamResult, nodeResult] = await Promise.allSettled([
      gum.then((stream) => {
        stats.gumMs = now() - t0;
        return stream;
      }),
      ensureNode(c),
    ]);
    if (streamResult.status === "rejected") {
      throw new MicError(micErrorMessage(streamResult.reason), streamResult.reason, { superseded: id !== seq });
    }
    const stream = streamResult.value;
    if (id !== seq) {
      stopTracks(stream);
      throw new MicError(MIC_MESSAGES.failed, null, { superseded: true });
    }
    if (nodeResult.status === "rejected") {
      stopTracks(stream);
      const reason = nodeResult.reason;
      throw new MicError(reason instanceof MicError ? reason.message : MIC_MESSAGES.failed, reason);
    }
    const n = nodeResult.value;
    let source;
    try {
      source = c.createMediaStreamSource(stream);
    } catch (err) {
      stopTracks(stream);
      throw new MicError(MIC_MESSAGES.failed, err);
    }
    if (session) abort(session);
    const s = {
      id, stream, source, onFrame, onFirstAudio, t0,
      closed: false, ending: null, resolve: null, timer: null, tStop: undefined,
    };
    session = s;
    stats.opens++;
    stats.firstAudioMs = null;
    stats.stopMs = null;
    n.port.postMessage({ type: "start", id: s.id });
    source.connect(n);
    return {
      id: s.id,
      end: () => end(s),
      abort: () => abort(s),
    };
  }

  return {
    begin,
    stats,
    // True while a recording holds the mic.
    get active() {
      return session !== null;
    },
  };
}
