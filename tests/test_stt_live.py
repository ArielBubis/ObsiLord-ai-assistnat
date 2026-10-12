"""Real faster-whisper on speech synthesized by the real Piper voice. Local CPU,
no API cost, no microphone.

Skipped by default; run with: pytest -m hardware -s tests/test_stt_live.py
Needs the Piper voice in voices/ (gitignored). The first run downloads the STT
model from config.example.toml ([stt] model, "small" is about 486 MB) into
models/ (gitignored). Audio stays in memory; nothing is written to disk.
"""

import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from obsilord.config import REPO_ROOT, load_settings
from obsilord.stt.base import SAMPLE_RATE
from obsilord.stt.faster_whisper import FasterWhisperSTT, model_is_cached, resolve_model
from obsilord.tts.piper import PiperTTS

pytestmark = pytest.mark.hardware

QUESTION = "What is the capital of France?"


def resample_pcm16(pcm16: bytes, rate_in: int, rate_out: int) -> bytes:
    """Linear-interpolation resample of PCM16 mono LE (good enough for a test)."""
    x = np.frombuffer(pcm16, dtype="<i2").astype(np.float64)
    n_out = int(round(len(x) * rate_out / rate_in))
    t_out = np.arange(n_out) * (rate_in / rate_out)
    y = np.interp(t_out, np.arange(len(x)), x)
    return np.clip(np.round(y), -32768, 32767).astype("<i2").tobytes()


def synthesize_question_16k(text: str = QUESTION) -> bytes:
    """Speak text with the configured Piper voice, resampled to 16 kHz PCM16."""
    settings = load_settings(
        REPO_ROOT / "config.example.toml", env_path=REPO_ROOT / "no-such.env", environ={}
    )
    tts = PiperTTS.load(
        REPO_ROOT / settings.tts.voice,
        settings.tts.speaker,
        settings.tts.length_scale,
        settings.tts.pitch,
    )

    async def collect() -> bytes:
        return b"".join([frame async for frame in tts.synthesize(text)])

    pcm = asyncio.run(collect())
    # The PCM is meant to be played at the announced rate (see tts/piper.py).
    return resample_pcm16(pcm, tts.sample_rate, SAMPLE_RATE)


def test_resample_length_and_range() -> None:
    pcm = (np.sin(np.arange(22050) / 10) * 20000).astype("<i2").tobytes()
    out = resample_pcm16(pcm, 22050, 16000)
    assert len(out) == 16000 * 2


def test_real_stt_transcribes_piper_speech(tmp_path: Path) -> None:
    settings = load_settings(
        REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
    )
    stt_cfg = settings.stt
    root = REPO_ROOT / stt_cfg.download_root
    model = resolve_model(stt_cfg.model, REPO_ROOT)
    was_cached = model_is_cached(model, root)

    t0 = time.monotonic()
    stt = FasterWhisperSTT.load(
        model, stt_cfg.device, stt_cfg.compute_type, root, stt_cfg.beam_size
    )
    load_ms = (time.monotonic() - t0) * 1000
    t1 = time.monotonic()
    stt.warm_up(stt_cfg.language_or_none)
    warm_ms = (time.monotonic() - t1) * 1000

    pcm = synthesize_question_16k()
    audio_s = len(pcm) / 2 / SAMPLE_RATE

    t2 = time.monotonic()
    text = asyncio.run(stt.transcribe(pcm, SAMPLE_RATE, stt_cfg.language_or_none))
    stt_ms = (time.monotonic() - t2) * 1000
    t3 = time.monotonic()
    again = asyncio.run(stt.transcribe(pcm, SAMPLE_RATE, stt_cfg.language_or_none))
    stt2_ms = (time.monotonic() - t3) * 1000

    lowered = text.lower()
    assert "capital" in lowered
    assert "france" in lowered
    assert again == text
    print(
        f"\nmodel {stt_cfg.model} ({stt_cfg.device}, {stt_cfg.compute_type}, "
        f"beam {stt_cfg.beam_size}), cached before: {was_cached}\n"
        f"load {load_ms:.0f} ms, warm-up {warm_ms:.0f} ms\n"
        f"audio {audio_s:.2f} s, stt {stt_ms:.0f} ms, second run {stt2_ms:.0f} ms\n"
        f"transcript: {text!r}"
    )


def test_real_model_rejects_unknown_language_at_warm_up() -> None:
    """A language the model does not know fails at warm-up (startup)."""
    settings = load_settings(
        REPO_ROOT / "config.example.toml", env_path=REPO_ROOT / "no-such.env", environ={}
    )
    stt_cfg = settings.stt
    stt = FasterWhisperSTT.load(
        resolve_model(stt_cfg.model, REPO_ROOT),
        stt_cfg.device,
        stt_cfg.compute_type,
        REPO_ROOT / stt_cfg.download_root,
        stt_cfg.beam_size,
    )
    with pytest.raises(ValueError, match="not a valid language code"):
        stt.warm_up("xx")
