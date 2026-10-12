"""STT helpers and FasterWhisperSTT with a fake model. No model is loaded."""

import asyncio
import logging
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest

import obsilord.stt.faster_whisper as fw
from obsilord.stt.base import pcm16_bytes_for_ms, pcm16_duration_ms, pcm16_to_float32
from obsilord.stt.faster_whisper import FasterWhisperSTT, model_is_cached, resolve_model
from obsilord.stt.whisper_languages import WHISPER_LANGUAGE_CODES


def _pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def test_pcm16_to_float32_scales_little_endian() -> None:
    out = pcm16_to_float32(_pcm(0, 16384, -16384, 32767, -32768))
    assert out.dtype == np.float32
    assert out.tolist() == [0.0, 0.5, -0.5, 32767 / 32768, -1.0]


def test_pcm16_to_float32_is_little_endian() -> None:
    # 0x0100 little-endian is 1; big-endian would read 256.
    assert pcm16_to_float32(b"\x01\x00").tolist() == [1 / 32768]


def test_pcm16_to_float32_drops_odd_trailing_byte() -> None:
    out = pcm16_to_float32(_pcm(16384, -16384) + b"\x7f")
    assert out.tolist() == [0.5, -0.5]
    assert pcm16_to_float32(b"\x01").size == 0


def test_pcm16_to_float32_empty() -> None:
    out = pcm16_to_float32(b"")
    assert out.dtype == np.float32 and out.size == 0


def test_pcm16_to_float32_accepts_bytearray_and_is_writable_copy() -> None:
    out = pcm16_to_float32(bytearray(_pcm(100, 200)))
    out[0] = 0.0  # astype returns a new, writable array
    assert out.size == 2


def test_pcm16_durations() -> None:
    assert pcm16_duration_ms(32000) == 1000
    assert pcm16_duration_ms(3200) == 100
    assert pcm16_duration_ms(3201) == 100
    assert pcm16_duration_ms(0) == 0
    assert pcm16_duration_ms(44100, 22050) == 1000
    with pytest.raises(ValueError):
        pcm16_duration_ms(10, 0)
    assert pcm16_bytes_for_ms(300) == 9600
    assert pcm16_bytes_for_ms(30_000) == 960_000
    assert pcm16_bytes_for_ms(0.5 * 1000) == 16000


@dataclass
class _Segment:
    text: str


class _FakeWhisper:
    """Stands in for faster_whisper.WhisperModel.transcribe."""

    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.calls: list[tuple[np.ndarray, dict[str, Any]]] = []
        self.consumed_in_lock: list[bool] = []
        self.owner: FasterWhisperSTT | None = None

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Iterator[_Segment], Any]:
        self.calls.append((audio, kwargs))

        def gen() -> Iterator[_Segment]:
            for t in self.texts:
                assert self.owner is not None
                self.consumed_in_lock.append(self.owner._lock.locked())
                yield _Segment(t)

        return gen(), object()


def _client(texts: list[str], beam_size: int = 1) -> tuple[FasterWhisperSTT, _FakeWhisper]:
    model = _FakeWhisper(texts)
    stt = FasterWhisperSTT(model, beam_size=beam_size)
    model.owner = stt
    return stt, model


def test_transcribe_joins_segments_and_passes_settings() -> None:
    stt, model = _client([" What is", " the capital of France? "], beam_size=3)
    pcm = _pcm(*([1000] * 1600))
    text = asyncio.run(stt.transcribe(pcm, 16000, "en"))
    assert text == "What is the capital of France?"
    [(audio, kwargs)] = model.calls
    assert audio.dtype == np.float32 and audio.shape == (1600,)
    assert kwargs == {
        "language": "en",
        "beam_size": 3,
        "vad_filter": True,
        "condition_on_previous_text": False,
        "without_timestamps": True,
    }
    # The lazy generator is consumed while the lock is held.
    assert model.consumed_in_lock == [True, True]


def test_transcribe_language_none_is_passed_through() -> None:
    stt, model = _client(["hi"])
    asyncio.run(stt.transcribe(_pcm(1, 2, 3), 16000, None))
    assert model.calls[0][1]["language"] is None


def test_transcribe_no_segments_is_empty() -> None:
    stt, _ = _client([])
    assert asyncio.run(stt.transcribe(_pcm(1, 2), 16000, "en")) == ""


def test_transcribe_rejects_other_sample_rates() -> None:
    stt, model = _client(["x"])
    with pytest.raises(ValueError, match="16000"):
        asyncio.run(stt.transcribe(_pcm(1, 2), 22050, "en"))
    assert model.calls == []


def test_transcribe_empty_audio_skips_model() -> None:
    stt, model = _client(["x"])
    assert asyncio.run(stt.transcribe(b"\x01", 16000, "en")) == ""
    assert model.calls == []


def test_transcribe_runs_off_the_event_loop() -> None:
    loop_thread: list[int] = []
    model_thread: list[int] = []

    class Model(_FakeWhisper):
        def transcribe(self, audio: np.ndarray, **kwargs: Any) -> Any:
            model_thread.append(threading.get_ident())
            return iter([_Segment("ok")]), None

    stt = FasterWhisperSTT(Model([]))

    async def main() -> str:
        loop_thread.append(threading.get_ident())
        return await stt.transcribe(_pcm(5), 16000, "en")

    assert asyncio.run(main()) == "ok"
    assert model_thread[0] != loop_thread[0]


def test_warm_up_runs_model_and_vad_with_configured_language() -> None:
    stt, model = _client([])
    stt.warm_up("he")
    [(audio1, kw1), (audio2, kw2)] = model.calls
    assert audio1.shape == audio2.shape == (16000,)
    assert not audio1.any() and not audio2.any()
    # First call runs the decoder, second loads the VAD session.
    assert (kw1["vad_filter"], kw2["vad_filter"]) == (False, True)
    assert kw1["language"] == kw2["language"] == "he"


def test_warm_up_error_propagates() -> None:
    class Rejecting(_FakeWhisper):
        def transcribe(self, audio: np.ndarray, **kwargs: Any) -> Any:
            raise ValueError("'xx' is not a valid language code")

    stt = FasterWhisperSTT(Rejecting([]))
    with pytest.raises(ValueError, match="language"):
        stt.warm_up("xx")


def test_language_list_matches_installed_faster_whisper() -> None:
    from faster_whisper.tokenizer import _LANGUAGE_CODES

    assert WHISPER_LANGUAGE_CODES == tuple(_LANGUAGE_CODES)


def test_resolve_model_relative_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "models" / "mine").mkdir(parents=True)
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)  # resolution must not depend on the working directory
    assert resolve_model("models/mine", tmp_path) == str(tmp_path / "models" / "mine")
    assert resolve_model(str(tmp_path / "models" / "mine"), other) == str(
        tmp_path / "models" / "mine"
    )


def test_resolve_model_names_unchanged(tmp_path: Path) -> None:
    assert resolve_model("small", tmp_path) == "small"
    assert resolve_model("Systran/faster-whisper-small", tmp_path) == (
        "Systran/faster-whisper-small"
    )
    # A missing directory is passed through (faster-whisper then treats it as a name).
    assert resolve_model("models/missing", tmp_path) == "models/missing"


def test_model_is_cached_false_for_empty_dir(tmp_path: Path) -> None:
    # local_files_only: no network request is made.
    assert model_is_cached("small", tmp_path) is False


def test_model_is_cached_true_for_model_dir(tmp_path: Path) -> None:
    assert model_is_cached(str(tmp_path), tmp_path / "unused") is True


def test_model_is_cached_unknown_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        model_is_cached("no-such-size", tmp_path)


class _RecordingWhisperModel:
    instances: list["_RecordingWhisperModel"] = []

    def __init__(self, model: str, **kwargs: Any) -> None:
        self.model = model
        self.kwargs = kwargs
        _RecordingWhisperModel.instances.append(self)


@pytest.mark.parametrize("cached", [False, True])
def test_load_logs_download_and_sets_local_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    cached: bool,
) -> None:
    import faster_whisper

    _RecordingWhisperModel.instances = []
    monkeypatch.setattr(faster_whisper, "WhisperModel", _RecordingWhisperModel)
    monkeypatch.setattr(fw, "model_is_cached", lambda model, root: cached)
    with caplog.at_level(logging.INFO, logger="obsilord.stt.faster_whisper"):
        stt = FasterWhisperSTT.load("small", "cpu", "int8", tmp_path / "models", beam_size=2)
    [inst] = _RecordingWhisperModel.instances
    assert inst.model == "small"
    assert inst.kwargs == {
        "device": "cpu",
        "compute_type": "int8",
        "download_root": str(tmp_path / "models"),
        "local_files_only": cached,
    }
    assert stt.beam_size == 2
    downloading = [r.getMessage() for r in caplog.records if "downloading" in r.getMessage()]
    if cached:
        assert downloading == []
    else:
        [line] = downloading
        assert "about 486 MB" in line and "first run only" in line


class _BlockingWhisper:
    """transcribe() blocks until release is set; records enter/exit order."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.events: list[str] = []

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Iterator[_Segment], Any]:
        n = self.events.count("enter") + 1
        self.events.append("enter")
        self.entered.set()

        def gen() -> Iterator[_Segment]:
            assert self.release.wait(timeout=10)
            self.events.append("exit")
            yield _Segment(f"call {n}")

        return gen(), None


@pytest.mark.parametrize("how", ["cancel", "timeout"])
def test_abandoned_transcription_holds_lock_for_next_call(how: str) -> None:
    model = _BlockingWhisper()
    stt = FasterWhisperSTT(model)
    pcm = _pcm(1, 2, 3)

    async def main() -> str:
        first = asyncio.create_task(stt.transcribe(pcm, 16000, "en"))
        while not model.entered.is_set():
            await asyncio.sleep(0.01)
        if how == "cancel":
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
        else:
            with pytest.raises(TimeoutError):
                async with asyncio.timeout(0.05):
                    await first
        # The abandoned call's thread still runs and still holds the lock.
        assert stt._lock.locked()
        second = asyncio.create_task(stt.transcribe(pcm, 16000, "en"))
        await asyncio.sleep(0.1)
        assert not second.done()
        assert model.events == ["enter"]  # the second call has not reached the model
        model.release.set()
        return await second

    text = asyncio.run(asyncio.wait_for(main(), 10))
    assert text == "call 2"
    assert model.events == ["enter", "exit", "enter", "exit"]
