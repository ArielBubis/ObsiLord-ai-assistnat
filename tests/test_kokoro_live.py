"""Loads the real Kokoro model from config.example.toml. Local CPU, no API cost.

Skipped by default; run with: pytest -m hardware -s tests/test_kokoro_live.py
Needs voices/kokoro/kokoro-v1.0.onnx and voices-v1.0.bin (gitignored).
"""

import asyncio
import time
from pathlib import Path

import pytest

from obsilord.config import REPO_ROOT, load_settings
from obsilord.tts.kokoro import KokoroTTS

pytestmark = pytest.mark.hardware

# About 600 characters: past Kokoro's 510-phoneme window, which kokoro-onnx
# splits into batches internally.
LONG = (
    "I had a look around the room. There is a desk by the window with a lamp, "
    "two monitors and a keyboard, and a mug of coffee that has gone cold. "
    "Behind you there is a shelf with books, a small plant that could use some "
    "water, and a framed photo I cannot quite make out from here. The light is "
    "coming from the left, so your face is a little in shadow on the right "
    "side. If you want me to read the titles on the shelf, move the camera a "
    "bit closer and hold it still for a second. Otherwise, I can tell you that "
    "it looks like a comfortable place to work, if slightly untidy."
)


@pytest.fixture(scope="module")
def tts() -> KokoroTTS:
    settings = load_settings(
        REPO_ROOT / "config.example.toml",
        env_path=Path(REPO_ROOT / "missing.env"),
        environ={},
    )
    cfg = settings.tts
    t0 = time.monotonic()
    client = KokoroTTS.load(REPO_ROOT / cfg.model, cfg.voice, cfg.speed, cfg.pitch)
    load_ms = (time.monotonic() - t0) * 1000
    t1 = time.monotonic()
    client.warm_up()
    print(f"load {load_ms:.0f} ms, warm-up {(time.monotonic() - t1) * 1000:.0f} ms")
    return client


def _collect(tts: KokoroTTS, text: str) -> tuple[list[bytes], float]:
    async def run() -> list[bytes]:
        return [frame async for frame in tts.synthesize(text)]

    t0 = time.monotonic()
    frames = asyncio.run(run())
    return frames, (time.monotonic() - t0) * 1000


def test_real_kokoro_synthesizes_one_sentence(tts: KokoroTTS) -> None:
    assert tts.native_rate == 24000
    assert tts.voice == "bm_lewis"
    assert tts.speed == 1.15
    frames, synth_ms = _collect(tts, "Hello, I am Obsi-Lord.")
    pcm = b"".join(frames)
    assert len(pcm) > 0 and len(pcm) % 2 == 0
    assert all(len(f) == 4800 for f in frames[:-1])
    assert 0 < len(frames[-1]) <= 4800 and len(frames[-1]) % 2 == 0
    seconds = len(pcm) / 2 / tts.sample_rate
    print(f"short: synth {synth_ms:.0f} ms, {seconds:.2f} s audio")


def test_real_kokoro_long_text(tts: KokoroTTS) -> None:
    frames, synth_ms = _collect(tts, LONG)
    seconds = len(b"".join(frames)) / 2 / tts.sample_rate
    print(f"long: synth {synth_ms:.0f} ms, {seconds:.2f} s audio, RTF {synth_ms / 1000 / seconds:.2f}")
    assert seconds > 20
