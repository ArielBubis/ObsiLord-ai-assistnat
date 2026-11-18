"""Kokoro TTS: voice names, PCM conversion, file checks, speed and pitch, framing.

No model is loaded here; a fake stands in for kokoro_onnx.Kokoro.
"""

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from obsilord.tts.kokoro import (
    KokoroTTS,
    float_to_pcm16,
    read_voice_names,
    resolve_voice,
    voice_lang,
)


class _FakeKokoro:
    """Stands in for kokoro_onnx.Kokoro: records create() calls."""

    def __init__(self, rate: int = 24000, delay_s: float = 0.0) -> None:
        self.rate = rate
        self.delay_s = delay_s
        self.calls: list[tuple[str, str, float, str]] = []
        self.active = 0
        self.max_active = 0
        self._guard = threading.Lock()

    def create(self, text: str, voice: str, speed: float, lang: str) -> tuple[np.ndarray, int]:
        with self._guard:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.delay_s)
        with self._guard:
            self.active -= 1
        self.calls.append((text, voice, speed, lang))
        return np.full(5000, 0.5, dtype=np.float32), self.rate


def _write_voices(path: Path, *names: str) -> None:
    with path.open("wb") as f:
        np.savez(f, **{n: np.zeros((2, 256), dtype=np.float32) for n in names})


async def _collect(tts: KokoroTTS, text: str) -> list[bytes]:
    return [f async for f in tts.synthesize(text)]


def test_voice_lang() -> None:
    assert voice_lang("bm_lewis") == "en-gb"
    assert voice_lang("bf_emma") == "en-gb"
    assert voice_lang("am_michael") == "en-us"
    assert voice_lang("af_heart") == "en-us"


@pytest.mark.parametrize("name", ["jf_alpha", "zf_xiaobei", "ff_siwis", "lewis"])
def test_voice_lang_rejects_non_english(name: str) -> None:
    with pytest.raises(ValueError, match="af_, am_, bf_, bm_"):
        voice_lang(name)


NAMES = ["af_heart", "bm_george", "bm_lewis", "jf_alpha"]


def test_resolve_voice() -> None:
    assert resolve_voice(NAMES, " bm_george ") == "bm_george"
    with pytest.raises(ValueError, match="''"):
        resolve_voice(NAMES, "")


def test_resolve_voice_unknown_lists_english_names() -> None:
    with pytest.raises(ValueError, match=r"'nobody'.*af_heart, bm_george, bm_lewis$"):
        resolve_voice(NAMES, "nobody")


def test_read_voice_names(tmp_path: Path) -> None:
    path = tmp_path / "voices-v1.0.bin"
    _write_voices(path, "bm_lewis", "af_heart")
    assert read_voice_names(path) == ["af_heart", "bm_lewis"]


def test_float_to_pcm16_scales() -> None:
    pcm = float_to_pcm16(np.array([0.0, 0.5, -0.5], dtype=np.float32))
    assert np.frombuffer(pcm, "<i2").tolist() == [0, 16384, -16384]


def test_float_to_pcm16_clips() -> None:
    pcm = float_to_pcm16(np.array([1.5, -2.0, 1.0], dtype=np.float32))
    assert np.frombuffer(pcm, "<i2").tolist() == [32767, -32767, 32767]


def test_float_to_pcm16_non_finite() -> None:
    pcm = float_to_pcm16(np.array([np.nan, np.inf, -np.inf], dtype=np.float32))
    assert np.frombuffer(pcm, "<i2").tolist() == [0, 32767, -32767]


def test_load_missing_model(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r'kokoro-v1.0.onnx.*model = ""'):
        KokoroTTS.load(tmp_path / "kokoro-v1.0.onnx", "bm_lewis", 1.0)


def test_load_missing_voices_file(tmp_path: Path) -> None:
    model = tmp_path / "kokoro-v1.0.onnx"
    model.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="voices-v1.0.bin"):
        KokoroTTS.load(model, "bm_lewis", 1.0)


def test_load_unknown_voice_fails_before_model_load(tmp_path: Path) -> None:
    # The model is not valid ONNX: reaching the model load would raise something else.
    model = tmp_path / "kokoro-v1.0.onnx"
    model.write_bytes(b"not onnx")
    _write_voices(tmp_path / "voices-v1.0.bin", "bm_lewis")
    with pytest.raises(ValueError, match="nobody"):
        KokoroTTS.load(model, "nobody", 1.0)


def test_load_non_english_voice_fails_before_model_load(tmp_path: Path) -> None:
    model = tmp_path / "kokoro-v1.0.onnx"
    model.write_bytes(b"not onnx")
    _write_voices(tmp_path / "voices-v1.0.bin", "bm_lewis", "jf_alpha")
    with pytest.raises(ValueError, match="af_, am_, bf_, bm_"):
        KokoroTTS.load(model, "jf_alpha", 1.0)


def test_kokoro_speed_115_and_frames() -> None:
    fake = _FakeKokoro()
    tts = KokoroTTS(fake, "bm_lewis", speed=1.15)
    assert tts.sample_rate == tts.native_rate == 24000
    assert tts.speed == 1.15
    frames = asyncio.run(_collect(tts, "Hi."))
    # 5000 samples are 10000 bytes; 100 ms at 24000 Hz is 4800 bytes.
    assert [len(f) for f in frames] == [4800, 4800, 400]
    assert fake.calls == [("Hi.", "bm_lewis", 1.15, "en-gb")]


def test_kokoro_pitch_106_announces_rate_and_slows_speed() -> None:
    tts = KokoroTTS(_FakeKokoro(), "am_michael", speed=1.15, pitch=1.06)
    assert tts.sample_rate == 25440
    assert tts.native_rate == 24000
    assert tts.speed == pytest.approx(1.15 / 1.06)
    assert tts.lang == "en-us"


def test_kokoro_wrong_sample_rate_raises() -> None:
    tts = KokoroTTS(_FakeKokoro(rate=22050), "bm_lewis", speed=1.0)
    with pytest.raises(RuntimeError, match="24000"):
        asyncio.run(_collect(tts, "Hi."))


def test_kokoro_synthesis_is_serialized() -> None:
    fake = _FakeKokoro(delay_s=0.05)
    tts = KokoroTTS(fake, "bm_lewis", speed=1.0)

    async def both() -> None:
        await asyncio.gather(_collect(tts, "One."), _collect(tts, "Two."))

    asyncio.run(both())
    assert len(fake.calls) == 2
    assert fake.max_active == 1
