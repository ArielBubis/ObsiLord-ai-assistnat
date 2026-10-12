"""faster-whisper implementation of STTClient (faster-whisper 1.2.1, local).

This is the only module that imports faster_whisper, and it does so lazily
(inside functions), so importing this module is cheap. API checked against the
installed faster-whisper 1.2.1 source (.venv/Lib/site-packages/faster_whisper/
transcribe.py, utils.py) on 2026-10-12:
- WhisperModel(model_size_or_path, device, compute_type=, download_root=,
  local_files_only=). A size name such as "small" maps to the Hugging Face repo
  Systran/faster-whisper-small; download_root is passed to
  huggingface_hub.snapshot_download as cache_dir.
- utils.download_model(size_or_id, local_files_only=True, cache_dir=) returns
  the cached snapshot path or raises LocalEntryNotFoundError (a
  FileNotFoundError) without network access. Unknown sizes raise ValueError.
- model.transcribe(audio: float32 mono 16 kHz ndarray, language=, beam_size=,
  vad_filter=, condition_on_previous_text=, without_timestamps=) returns
  (lazy segment generator, info). Decoding happens while the generator is
  consumed, so it is consumed inside the lock below.
- vad_filter uses the Silero VAD ONNX model bundled in the package
  (faster_whisper/assets/silero_vad_v6.onnx, run with onnxruntime).
- The "faster_whisper" logger logs durations at INFO and decoding details at
  DEBUG, never transcript text.

Transcription is blocking, so it runs in a worker thread (asyncio.to_thread)
under a threading lock: one transcription at a time, also when a cancelled or
timed-out turn's thread is still finishing in the background. Audio stays in
memory; nothing is written to disk and no transcript text is logged.

When the model is already in download_root it is loaded with
local_files_only=True, so a normal start makes no network request. A local
model directory given as a relative path is resolved against the repo root
(resolve_model), like download_root and tts.voice.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from obsilord.stt.base import SAMPLE_RATE, pcm16_to_float32

log = logging.getLogger(__name__)

WARMUP_SECONDS = 1.0

# Approximate download sizes of the size names this project has looked up.
# Systran/faster-whisper-small on Hugging Face: about 486 MB (2026-10-12).
KNOWN_SIZES_MB = {"small": 486}


def resolve_model(model: str, base_dir: Path) -> str:
    """A model directory relative to base_dir (the repo root) -> absolute path.

    Size names ("small") and Hugging Face repo ids ("Systran/faster-whisper-small")
    are returned unchanged unless a directory of that name exists under
    base_dir, which is the same rule faster-whisper applies (a directory wins).
    """
    path = Path(model).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return str(path) if path.is_dir() else model


def model_is_cached(model: str, download_root: Path) -> bool:
    """True if model is a local directory or already in download_root.

    Pass model through resolve_model first so relative directories resolve
    against the repo root. Makes no network request. Raises ValueError for an
    unknown size name.
    """
    if Path(model).is_dir():
        return True
    from faster_whisper.utils import download_model

    try:
        download_model(model, local_files_only=True, cache_dir=str(download_root))
    except FileNotFoundError:
        # huggingface_hub's LocalEntryNotFoundError (and IncompleteSnapshotError).
        return False
    return True


class FasterWhisperSTT:
    """STTClient backed by a local faster-whisper model. Create with load()."""

    def __init__(self, model: Any, beam_size: int = 1) -> None:
        self._model = model
        self.beam_size = beam_size
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        model: str,
        device: str,
        compute_type: str,
        download_root: Path,
        beam_size: int = 1,
    ) -> "FasterWhisperSTT":
        """Load the model, downloading it into download_root on first use.

        Raises ValueError for an unknown model size and the provider's errors
        (network, ctranslate2) if the download or load fails.
        """
        cached = model_is_cached(model, download_root)
        if not cached:
            size = KNOWN_SIZES_MB.get(model)
            log.info(
                "stt: model %s is not in %s yet, downloading %s (first run only)",
                model,
                download_root,
                f"about {size} MB" if size else "it",
            )

        from faster_whisper import WhisperModel

        t0 = time.monotonic()
        whisper = WhisperModel(
            model,
            device=device,
            compute_type=compute_type,
            download_root=str(download_root),
            local_files_only=cached,
        )
        log.info(
            "stt: loaded faster-whisper %s (%s, %s, beam %d) in %d ms%s",
            model,
            device,
            compute_type,
            beam_size,
            int((time.monotonic() - t0) * 1000),
            "" if cached else " including the download",
        )
        return cls(whisper, beam_size)

    def warm_up(self, language: str | None) -> None:
        """Run the model and the VAD once (blocking). The first calls are slow.

        Uses the configured language, so a code the model rejects fails at
        startup instead of on every turn. Two calls on 1 s of silence:
        - VAD off, so the decoder runs (with VAD on, silence is removed before
          the model sees it);
        - VAD on, so the Silero VAD session is loaded now and not on the first
          turn. It removes all of the silence and returns no text.
        """
        t0 = time.monotonic()
        silence = np.zeros(int(SAMPLE_RATE * WARMUP_SECONDS), dtype=np.float32)
        self._transcribe_blocking(silence, language, vad_filter=False)
        self._transcribe_blocking(silence, language, vad_filter=True)
        log.info("stt: warm-up took %d ms", int((time.monotonic() - t0) * 1000))

    def _transcribe_blocking(
        self, audio: np.ndarray, language: str | None, vad_filter: bool = True
    ) -> str:
        with self._lock:
            segments, _info = self._model.transcribe(
                audio,
                language=language,
                beam_size=self.beam_size,
                vad_filter=vad_filter,
                condition_on_previous_text=False,
                without_timestamps=True,
            )
            # The generator decodes lazily, so consume it while holding the lock.
            return "".join(segment.text for segment in segments).strip()

    async def transcribe(self, pcm16: bytes, sample_rate: int, language: str | None) -> str:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"stt: expected {SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
        audio = pcm16_to_float32(pcm16)
        if audio.size == 0:
            return ""
        return await asyncio.to_thread(self._transcribe_blocking, audio, language)
