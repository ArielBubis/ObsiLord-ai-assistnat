"""Piper implementation of TTSClient (piper-tts 1.8.0, local, CPU).

This is the only module that imports piper. API checked against the installed
piper-tts 1.8.0 source (.venv/Lib/site-packages/piper/voice.py, config.py):
- PiperVoice.load(model_path, config_path=None, use_cuda=False); the config
  defaults to model_path + ".json" and is read as UTF-8.
- voice.config.sample_rate (from audio.sample_rate in the .onnx.json),
  voice.config.speaker_id_map (name -> id), voice.config.num_speakers.
- voice.synthesize(text, syn_config=SynthesisConfig(speaker_id=, length_scale=))
  is a blocking generator of AudioChunk, one per sentence piper finds in text.
  AudioChunk.audio_int16_bytes is PCM16 mono in native byte order (numpy
  int16.tobytes()), which is little-endian on x86 and ARM Windows/Linux/macOS.
- normalize_audio defaults to True (each chunk is scaled to full range).

Synthesis is blocking, so it runs in a worker thread (asyncio.to_thread) and
the event loop never blocks. A threading lock keeps one synthesis at a time,
also when a cancelled turn's thread is still finishing in the background.
Audio stays in memory; nothing is written to disk.

Pitch without a protocol change: with pitch p, the voice is synthesized with
length_scale * p (p times as many samples, so a slower, longer take) and
tts_start announces sample_rate = round(native_rate * p). The browser plays the
PCM at the announced rate, so the pitch rises by p while the tempo comes back
to normal. sample_rate below is the announced rate. Frames keep the size of
100 ms at the native rate (4410 bytes for 22050 Hz), so they last 100 / p ms.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, AsyncIterator, Mapping

from obsilord.tts.base import FRAME_MS, frame_bytes, split_frames

log = logging.getLogger(__name__)

WARMUP_TEXT = "Hello."

if sys.byteorder != "little":  # pragma: no cover - all supported hosts are little-endian
    raise RuntimeError("PiperTTS assumes a little-endian host (PCM16 LE on the wire)")


def pitch_adjust(native_rate: int, length_scale: float, pitch: float) -> tuple[int, float]:
    """(announced sample rate, effective length_scale) for a pitch factor.

    pitch 1.0 leaves both unchanged.
    """
    return round(native_rate * pitch), length_scale * pitch


def resolve_speaker(speaker_id_map: Mapping[str, int], name: str) -> int | None:
    """Speaker name -> id. Empty name means the voice default (None).

    Raises ValueError listing the valid names if name is unknown.
    """
    name = name.strip()
    if not name:
        return None
    if name in speaker_id_map:
        return int(speaker_id_map[name])
    if not speaker_id_map:
        raise ValueError(
            f'config: tts.speaker is {name!r} but this voice has a single speaker; '
            'set speaker = ""'
        )
    raise ValueError(
        f"config: tts.speaker {name!r} is not in this voice; "
        f"valid names: {', '.join(sorted(speaker_id_map))}"
    )


def read_speaker_id_map(config_path: Path) -> dict[str, int]:
    """speaker_id_map from a Piper .onnx.json (empty for single-speaker voices)."""
    data: Any = json.loads(config_path.read_text(encoding="utf-8"))
    raw = data.get("speaker_id_map") if isinstance(data, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {str(k): int(v) for k, v in raw.items()}


class PiperTTS:
    """TTSClient backed by a local Piper voice. Create with PiperTTS.load()."""

    def __init__(
        self,
        voice: Any,
        speaker_id: int | None,
        length_scale: float,
        pitch: float = 1.0,
        frame_ms: int = FRAME_MS,
    ) -> None:
        from piper import SynthesisConfig

        self._voice = voice
        self.native_rate: int = int(voice.config.sample_rate)
        # sample_rate is the rate announced in tts_start (see the module docstring).
        self.sample_rate, self.length_scale_effective = pitch_adjust(
            self.native_rate, length_scale, pitch
        )
        self._syn_config = SynthesisConfig(
            speaker_id=speaker_id, length_scale=self.length_scale_effective
        )
        self._frame_bytes = frame_bytes(self.native_rate, frame_ms)
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls, model_path: Path, speaker: str, length_scale: float, pitch: float = 1.0
    ) -> "PiperTTS":
        """Load the voice, resolve the speaker name, and return a ready client.

        Raises FileNotFoundError if the model or its .onnx.json is missing and
        ValueError if the speaker name is unknown.
        """
        config_path = Path(f"{model_path}.json")
        for path in (model_path, config_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"tts: voice file not found: {path} "
                    '(download it, or set [tts] voice = "" to disable speech)'
                )
        # Resolve the speaker before the slow model load so a typo fails fast.
        speaker_id = resolve_speaker(read_speaker_id_map(config_path), speaker)

        from piper import PiperVoice

        t0 = time.monotonic()
        voice = PiperVoice.load(model_path, config_path=config_path)
        log.info(
            "tts: loaded %s (speaker id %s, %d Hz) in %d ms",
            model_path.name,
            speaker_id if speaker_id is not None else "default",
            voice.config.sample_rate,
            int((time.monotonic() - t0) * 1000),
        )
        tts = cls(voice, speaker_id, length_scale, pitch)
        if pitch != 1.0:
            log.info(
                "tts: pitch %.3f (length_scale %.3f, announced rate %d Hz)",
                pitch,
                tts.length_scale_effective,
                tts.sample_rate,
            )
        return tts

    def warm_up(self) -> None:
        """Synthesize a short phrase once (blocking). The first call is slow."""
        t0 = time.monotonic()
        self._synth_blocking(WARMUP_TEXT)
        log.info("tts: warm-up took %d ms", int((time.monotonic() - t0) * 1000))

    def _synth_blocking(self, text: str) -> bytes:
        with self._lock:
            return b"".join(
                chunk.audio_int16_bytes
                for chunk in self._voice.synthesize(text, syn_config=self._syn_config)
            )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        pcm = await asyncio.to_thread(self._synth_blocking, text)
        for frame in split_frames(pcm, self._frame_bytes):
            yield frame
