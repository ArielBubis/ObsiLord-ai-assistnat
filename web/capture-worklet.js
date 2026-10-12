// AudioWorklet processor for push-to-talk capture. Loaded by audio_in.js with
// audioWorklet.addModule("/static/capture-worklet.js"); runs on the audio
// thread at the context's own rate (Firefox will not connect a mic to a
// context at a different rate, so there is no 16 kHz context). The DSP lives
// in resample16k.js (pure, unit tested).
//
// Messages from the main thread:
//   {type: "start", id}  reset and start converting input
//   {type: "stop", id}   flush the resampler and the partial last frame, stop
// Messages to the main thread:
//   {type: "first", id}        first non-silent input after start (or
//                              after 300 ms of all-zero input)
//   {type: "frame", id, buf}   PCM16 LE mono 16 kHz, 3200 bytes (the last
//                              frame of a recording may be shorter); buf is
//                              transferred, not copied
//   {type: "end", id}          after stop, once every frame has been posted
// Audio is only held in memory until it is posted; nothing is stored.

import {
  createResampler, createFramer, createFirstAudioDetector, downmix, TARGET_RATE, FRAME_SAMPLES,
} from "./resample16k.js";

class Capture16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.resampler = createResampler(sampleRate, TARGET_RATE);
    this.framer = createFramer(FRAME_SAMPLES);
    this.mix = new Float32Array(128);
    this.first = createFirstAudioDetector(sampleRate);
    this.active = false;
    this.id = 0;
    this.port.onmessage = (event) => this.command(event.data);
  }

  command(msg) {
    if (!msg || typeof msg !== "object") return;
    if (msg.type === "start") {
      this.resampler.reset();
      this.framer.reset();
      this.first.reset();
      this.id = msg.id;
      this.active = true;
    } else if (msg.type === "stop") {
      if (!this.active || msg.id !== this.id) {
        this.port.postMessage({ type: "end", id: msg.id });
        return;
      }
      this.active = false;
      this.post(this.framer.push(this.resampler.flush()));
      const last = this.framer.flush();
      if (last) this.post([last]);
      this.port.postMessage({ type: "end", id: this.id });
    }
  }

  post(frames) {
    for (const buf of frames) this.port.postMessage({ type: "frame", id: this.id, buf }, [buf]);
  }

  process(inputs) {
    if (!this.active) return true;
    const channels = inputs[0];
    if (!channels || channels.length === 0) return true;
    if (this.first.update(channels)) this.port.postMessage({ type: "first", id: this.id });
    if (this.mix.length < channels[0].length) this.mix = new Float32Array(channels[0].length);
    const mono = downmix(channels, this.mix);
    this.post(this.framer.push(this.resampler.process(mono)));
    // Output stays silent. Returning true keeps the node alive between
    // recordings, when it has no input.
    return true;
  }
}

registerProcessor("capture-16k", Capture16k);
