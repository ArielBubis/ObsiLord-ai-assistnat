"""Provider-neutral TTS interface (see ARCHITECTURE.md, Interfaces).

Nothing in this module may import a provider package.

Audio format on the wire: PCM16, mono, little-endian, at sample_rate. The
pipeline sends each chunk as one binary WebSocket frame, so implementations
yield short frames (about FRAME_MS each) and the browser can start playing
before a whole sentence has arrived.
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol

BYTES_PER_SAMPLE = 2  # PCM16
FRAME_MS = 100


class TTSClient(Protocol):
    # Output sample rate in Hz, announced to the browser in tts_start.
    sample_rate: int

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """Yield PCM16 mono little-endian chunks for text (one sentence)."""
        ...


def frame_bytes(sample_rate: int, frame_ms: int = FRAME_MS) -> int:
    """Bytes in one frame of frame_ms at sample_rate (always a whole sample count).

    22050 Hz and 100 ms give 2205 samples, 4410 bytes.
    """
    samples = max(1, sample_rate * frame_ms // 1000)
    return samples * BYTES_PER_SAMPLE


def split_frames(pcm: bytes, size: int) -> list[bytes]:
    """Split PCM16 bytes into frames of size bytes; the last one may be shorter.

    size must be a positive even number so no sample is split across frames.
    """
    if size <= 0 or size % BYTES_PER_SAMPLE:
        raise ValueError(f"frame size must be a positive even number, got {size}")
    return [pcm[i : i + size] for i in range(0, len(pcm), size)]
