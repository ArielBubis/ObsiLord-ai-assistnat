"""Kokoro implementation of TTSClient (kokoro-onnx 0.6.1, Kokoro-82M v1.0, local, CPU).

This is the only module that imports kokoro_onnx or onnxruntime. API checked
against the installed kokoro-onnx 0.6.1 source (kokoro_onnx/__init__.py):
- Kokoro.from_session(session, voices_path) takes a ready onnxruntime
  InferenceSession, so the thread count can be set here.
- kokoro.create(text, voice=, speed=, lang=) is blocking and returns
  (float32 ndarray in -1..1, 24000). It raises ValueError for a speed outside
  0.5..2.0 (config.py checks the range at startup).
- The voices file is an npz (np.load); its keys are the voice names.
- lang is an espeak-ng language: en-gb for the British voices (bf_, bm_),
  en-us for the American ones (af_, am_). Other languages are not supported.

Files: the model (kokoro-v1.0.onnx) and VOICES_FILE sit in one folder. fp32
only: on the test PC (2026-11-18) the int8 model was 10x slower and DirectML
failed on a ConvTranspose node.

Pitch without a protocol change: with pitch p the voice is synthesized at
speed / p (a slower, longer take) and tts_start announces sample_rate =
round(24000 * p). The browser plays the PCM at the announced rate, so the pitch
rises by p while the tempo comes back to speed. Frames keep the size of 100 ms
at 24000 Hz (4800 bytes), so they last 100 / p ms.

Threads: intra_op_num_threads = os.cpu_count(). On a 12-thread Ryzen 5 7600X
that was about 20 percent faster than onnxruntime's default.

Synthesis is blocking, so it runs in a worker thread (asyncio.to_thread) and
the event loop never blocks. A threading lock keeps one synthesis at a time,
also when a cancelled turn's thread is still finishing in the background.
Audio stays in memory; nothing is written to disk.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Collection

import numpy as np

from obsilord.tts.base import FRAME_MS, frame_bytes, split_frames

log = logging.getLogger(__name__)

KOKORO_RATE = 24000
VOICES_FILE = "voices-v1.0.bin"
ENGLISH_PREFIXES = ("af_", "am_", "bf_", "bm_")
WARMUP_TEXT = "Hello."

if sys.byteorder != "little":  # pragma: no cover - all supported hosts are little-endian
    raise RuntimeError("KokoroTTS assumes a little-endian host (PCM16 LE on the wire)")


def voice_lang(name: str) -> str:
    """espeak-ng language for an English Kokoro voice name.

    Raises ValueError for any other voice.
    """
    if name.startswith(("bf_", "bm_")):
        return "en-gb"
    if name.startswith(("af_", "am_")):
        return "en-us"
    raise ValueError(
        f"config: tts.voice {name!r} is not an English Kokoro voice; "
        f"only names starting with {', '.join(ENGLISH_PREFIXES)} are supported"
    )


def resolve_voice(names: Collection[str], name: str) -> str:
    """Voice name from config, checked against the voices file.

    Raises ValueError listing the English voices if name is unknown.
    """
    name = name.strip()
    if name in names:
        return name
    english = sorted(n for n in names if n.startswith(ENGLISH_PREFIXES))
    raise ValueError(
        f"config: tts.voice {name!r} is not in the Kokoro voices file; "
        f"English voices: {', '.join(english)}"
    )


def read_voice_names(path: Path) -> list[str]:
    """Sorted voice names in a Kokoro voices file (an npz archive)."""
    with np.load(path) as data:
        return sorted(data.files)


def float_to_pcm16(samples: np.ndarray) -> bytes:
    """Float samples in -1..1 to PCM16 little-endian bytes; values outside are
    clipped and NaN becomes silence (so the int16 cast is always defined)."""
    finite = np.nan_to_num(samples, nan=0.0, posinf=1.0, neginf=-1.0)
    scaled = np.round(np.clip(finite, -1.0, 1.0) * 32767.0)
    return scaled.astype("<i2").tobytes()


class KokoroTTS:
    """TTSClient backed by a local Kokoro model. Create with KokoroTTS.load()."""

    def __init__(
        self,
        kokoro: Any,
        voice: str,
        speed: float,
        pitch: float = 1.0,
        frame_ms: int = FRAME_MS,
    ) -> None:
        self._kokoro = kokoro
        self.voice = voice
        self.lang: str = voice_lang(voice)
        self.native_rate: int = KOKORO_RATE
        # sample_rate is the rate announced in tts_start; speed is what Kokoro
        # synthesizes at (see the module docstring).
        self.sample_rate: int = round(KOKORO_RATE * pitch)
        self.speed: float = speed / pitch
        self._frame_bytes = frame_bytes(KOKORO_RATE, frame_ms)
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls, model_path: Path, voice: str, speed: float, pitch: float = 1.0
    ) -> "KokoroTTS":
        """Check the files, resolve the voice, load the model, and return a ready client.

        Raises FileNotFoundError if the model or the voices file is missing and
        ValueError if the voice is unknown or not English. Both checks run
        before the slow model load.
        """
        voices_path = model_path.parent / VOICES_FILE
        for path in (model_path, voices_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"tts: Kokoro file not found: {path} (download it into the "
                    "model's folder, see [tts] in config.example.toml, or set "
                    '[tts] model = "" to disable speech)'
                )
        voice = resolve_voice(read_voice_names(voices_path), voice)
        voice_lang(voice)

        import onnxruntime as rt
        from kokoro_onnx import Kokoro

        t0 = time.monotonic()
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = os.cpu_count() or 0
        session = rt.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        kokoro = Kokoro.from_session(session, str(voices_path))
        tts = cls(kokoro, voice, speed, pitch)
        log.info(
            "tts: loaded %s (voice %s, speed %.3f, %d Hz announced) in %d ms",
            model_path.name,
            voice,
            tts.speed,
            tts.sample_rate,
            int((time.monotonic() - t0) * 1000),
        )
        return tts

    def warm_up(self) -> None:
        """Synthesize a short phrase once (blocking). The first call is slow."""
        t0 = time.monotonic()
        self._synth_blocking(WARMUP_TEXT)
        log.info("tts: warm-up took %d ms", int((time.monotonic() - t0) * 1000))

    def _synth_blocking(self, text: str) -> bytes:
        with self._lock:
            samples, rate = self._kokoro.create(
                text, voice=self.voice, speed=self.speed, lang=self.lang
            )
        if rate != KOKORO_RATE:
            raise RuntimeError(f"tts: Kokoro returned {rate} Hz, expected {KOKORO_RATE}")
        return float_to_pcm16(samples)

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        pcm = await asyncio.to_thread(self._synth_blocking, text)
        for frame in split_frames(pcm, self._frame_bytes):
            yield frame
