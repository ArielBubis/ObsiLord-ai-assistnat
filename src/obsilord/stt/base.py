"""Provider-neutral STT interface (see planning/ARCHITECTURE.md, Interfaces).

Nothing in this module may import a provider package.

Audio format from the browser: PCM16, mono, little-endian, at SAMPLE_RATE
(16 kHz), sent as binary WebSocket frames between audio_start and audio_end.
The pipeline buffers the whole utterance in memory and passes it to
STTClient.transcribe in one call. Audio is never written to disk or logged.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # PCM16


class STTClient(Protocol):
    async def transcribe(self, pcm16: bytes, sample_rate: int, language: str | None) -> str:
        """Return the transcript of one utterance ("" if nothing was recognized).

        language: a language code such as "en", or None to detect it.
        """
        ...


def pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    """PCM16 little-endian bytes -> float32 samples in [-1, 1).

    Each sample is divided by 32768. An odd trailing byte (half a sample) is
    dropped.
    """
    usable = len(pcm16) - len(pcm16) % BYTES_PER_SAMPLE
    samples = np.frombuffer(pcm16, dtype="<i2", count=usable // BYTES_PER_SAMPLE)
    return samples.astype(np.float32) / np.float32(32768.0)


def pcm16_duration_ms(num_bytes: int, sample_rate: int = SAMPLE_RATE) -> int:
    """Length in whole milliseconds of num_bytes of PCM16 mono audio."""
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    samples = max(0, num_bytes) // BYTES_PER_SAMPLE
    return samples * 1000 // sample_rate


def pcm16_bytes_for_ms(ms: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Bytes of PCM16 mono audio in ms milliseconds (whole samples, rounded down)."""
    samples = int(ms * sample_rate // 1000)
    return max(0, samples) * BYTES_PER_SAMPLE
