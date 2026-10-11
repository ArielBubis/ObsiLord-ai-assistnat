"""Loads the real Piper voice from config.example.toml. Local CPU, no API cost.

Skipped by default; run with: pytest -m hardware tests/test_piper_live.py
Needs voices/en_GB-semaine-medium.onnx and its .onnx.json (gitignored).
"""

import asyncio
import time
from pathlib import Path

import pytest

from obsilord.config import REPO_ROOT, load_settings
from obsilord.tts.piper import PiperTTS

pytestmark = pytest.mark.hardware


def test_real_voice_synthesizes_one_sentence(tmp_path: Path) -> None:
    settings = load_settings(
        REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
    )
    t0 = time.monotonic()
    tts = PiperTTS.load(
        REPO_ROOT / settings.tts.voice, settings.tts.speaker,
        settings.tts.length_scale,
        settings.tts.pitch,
    )
    load_ms = (time.monotonic() - t0) * 1000
    tts.warm_up()
    assert tts.native_rate == 22050
    assert tts.sample_rate == round(22050 * settings.tts.pitch)

    async def collect() -> list[bytes]:
        return [frame async for frame in tts.synthesize("Hello, I am Obsi-Lord.")]

    t1 = time.monotonic()
    frames = asyncio.run(collect())
    synth_ms = (time.monotonic() - t1) * 1000
    pcm = b"".join(frames)
    assert len(pcm) > 0
    assert len(pcm) % 2 == 0
    assert all(len(f) <= 4410 and len(f) % 2 == 0 for f in frames)
    assert all(len(f) == 4410 for f in frames[:-1])
    seconds = len(pcm) / 2 / tts.sample_rate
    print(
        f"load {load_ms:.0f} ms, synth {synth_ms:.0f} ms, "
        f"{len(frames)} frames, {len(pcm)} bytes, {seconds:.2f} s audio"
    )
