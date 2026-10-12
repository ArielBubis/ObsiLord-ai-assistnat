"""Push-to-talk over /ws with FakeSTT, FakeLLM, FakeTTS. No model, no network."""

import asyncio
import dataclasses
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import obsilord.stt.faster_whisper as fw
from fakes import FakeLLM, FakeSTT, FakeTTS, text_only
from obsilord.app import (
    MSG_BUSY,
    MSG_NO_KEY,
    MSG_NO_VOICE,
    MSG_NOT_RECORDING,
    MSG_UNEXPECTED_AUDIO,
    MSG_UNKNOWN,
    create_app,
    msg_too_long_audio,
)
from obsilord.config import REPO_ROOT, Settings, load_settings
from obsilord.pipeline import MSG_NOT_CAUGHT
from obsilord.usage import UsageTracker

LISTENING = {"type": "state", "state": "listening"}
THINKING = {"type": "state", "state": "thinking"}
SPEAKING = {"type": "state", "state": "speaking"}
IDLE = {"type": "state", "state": "idle"}
FRAME = bytes(3200)  # 100 ms of PCM16 at 16 kHz


def _settings(tmp_path: Path, **stt: Any) -> Settings:
    base = text_only(
        load_settings(
            REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
        )
    )
    return dataclasses.replace(base, stt=dataclasses.replace(base.stt, **stt))


def _client(
    tmp_path: Path,
    stt: FakeSTT | None,
    llm: FakeLLM | None = None,
    tts: FakeTTS | None = None,
    **stt_settings: Any,
) -> TestClient:
    settings = _settings(tmp_path, **stt_settings)
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    llm = llm if llm is not None else FakeLLM(["Paris."])
    return TestClient(create_app(settings, llm=llm, usage=usage, tts=tts, stt=stt))


def _no_key_client(tmp_path: Path, stt: FakeSTT) -> TestClient:
    settings = _settings(tmp_path)
    app = create_app(settings, llm=FakeLLM(["x"]), stt=stt)
    app.state.llm = None
    return TestClient(app)


def _collect(ws: Any) -> list[dict[str, Any] | bytes]:
    msgs: list[dict[str, Any] | bytes] = []
    while True:
        raw = ws.receive()
        if raw.get("bytes") is not None:
            msgs.append(raw["bytes"])
            continue
        msg = json.loads(raw["text"])
        msgs.append(msg)
        if msg == IDLE:
            return msgs


def _assert_quiet(ws: Any) -> None:
    """Nothing is pending: ping gets no reply, so the next message answers bogus."""
    ws.send_json({"type": "ping"})
    ws.send_json({"type": "bogus"})
    assert ws.receive_json() == {"type": "error", "message": MSG_UNKNOWN}


def _speak(ws: Any, frames: int) -> None:
    ws.send_json({"type": "audio_start"})
    assert ws.receive_json() == LISTENING
    for _ in range(frames):
        ws.send_bytes(FRAME)
    ws.send_json({"type": "audio_end"})


def test_voice_turn_full_sequence(tmp_path: Path) -> None:
    stt = FakeSTT("What is the capital of France?")
    llm = FakeLLM(["Paris."])
    client = _client(tmp_path, stt, llm=llm, tts=FakeTTS(frames=1))
    with client.websocket_connect("/ws") as ws:
        _speak(ws, frames=5)
        msgs = _collect(ws)
    assert msgs == [
        THINKING,
        {"type": "transcript", "role": "user", "text": "What is the capital of France?",
         "final": True},
        {"type": "transcript", "role": "assistant", "text": "Paris.", "final": False},
        {"type": "transcript", "role": "assistant", "text": "Paris.", "final": True},
        {"type": "tts_start", "sentence_id": 1, "sample_rate": 22050},
        SPEAKING,
        bytes([1, 0]) * 2,
        {"type": "tts_end", "sentence_id": 1},
        IDLE,
    ]
    assert stt.calls == [(FRAME * 5, 16000, "en")]
    assert len(llm.calls) == 1


def test_audio_start_when_voice_disabled(tmp_path: Path) -> None:
    client = _client(tmp_path, None)
    assert client.app.state.stt is None  # type: ignore[attr-defined]
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NO_VOICE}
        # Not recording: frames and audio_end are rejected like any stray input.
        ws.send_bytes(FRAME)
        assert ws.receive_json() == {"type": "error", "message": MSG_UNEXPECTED_AUDIO}
        ws.send_json({"type": "audio_end"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NOT_RECORDING}
        _assert_quiet(ws)


def test_audio_start_without_api_key(tmp_path: Path) -> None:
    stt = FakeSTT()
    with _no_key_client(tmp_path, stt).websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NO_KEY}
        _assert_quiet(ws)
    assert stt.calls == []


def test_audio_start_while_turn_running_is_busy(tmp_path: Path) -> None:
    gate = threading.Event()
    client = _client(tmp_path, FakeSTT(), llm=FakeLLM(["First ", "reply."], gate=gate))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "one"})
        assert ws.receive_json() == THINKING
        assert ws.receive_json() == SPEAKING
        assert ws.receive_json()["text"] == "First "
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == {"type": "error", "message": MSG_BUSY}
        gate.set()
        rest = _collect(ws)
        assert LISTENING not in rest
        # After the turn, recording works again.
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == LISTENING


def test_audio_start_during_voice_turn_is_busy(tmp_path: Path) -> None:
    gate = threading.Event()
    stt = FakeSTT("Hello", gate=gate)
    client = _client(tmp_path, stt)
    with client.websocket_connect("/ws") as ws:
        _speak(ws, frames=5)
        assert ws.receive_json() == THINKING
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == {"type": "error", "message": MSG_BUSY}
        ws.send_json({"type": "text", "text": "typed"})
        assert ws.receive_json() == {"type": "error", "message": MSG_BUSY}
        gate.set()
        msgs = _collect(ws)
        assert msgs[0]["role"] == "user"  # type: ignore[index]
    assert len(stt.calls) == 1


def test_audio_start_while_recording_is_busy(tmp_path: Path) -> None:
    stt = FakeSTT("Hello")
    client = _client(tmp_path, stt)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == LISTENING
        ws.send_bytes(FRAME * 3)
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == {"type": "error", "message": MSG_BUSY}
        # The recording continued: the frames before and after are both kept.
        ws.send_bytes(FRAME * 2)
        ws.send_json({"type": "audio_end"})
        _collect(ws)
    assert stt.calls[0][0] == FRAME * 5


def test_text_while_recording_is_busy(tmp_path: Path) -> None:
    llm = FakeLLM(["Paris."])
    stt = FakeSTT("Hello")
    client = _client(tmp_path, stt, llm=llm)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == LISTENING
        ws.send_json({"type": "text", "text": "typed question"})
        assert ws.receive_json() == {"type": "error", "message": MSG_BUSY}
        ws.send_bytes(FRAME * 4)
        ws.send_json({"type": "audio_end"})
        msgs = _collect(ws)
    assert msgs[1] == {"type": "transcript", "role": "user", "text": "Hello", "final": True}
    assert [[m.text for m in call] for call in llm.calls] == [["Hello"]]


def test_too_long_recording_is_truncated_with_one_error(tmp_path: Path) -> None:
    stt = FakeSTT("Hello")
    client = _client(tmp_path, stt, max_utterance_s=1.0)  # 32000 bytes
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == LISTENING
        for i in range(10):
            ws.send_bytes(bytes([i]) * 3200)
        # Exactly at the limit: nothing dropped yet, so no error.
        _assert_quiet(ws)
        for i in range(10, 15):
            ws.send_bytes(bytes([i]) * 3200)
        assert ws.receive_json() == {"type": "error", "message": msg_too_long_audio(1.0)}
        _assert_quiet(ws)  # only one error for the five dropped frames
        ws.send_json({"type": "audio_end"})
        msgs = _collect(ws)
    assert msgs[0] == THINKING
    pcm = stt.calls[0][0]
    assert len(pcm) == 32000
    assert pcm == b"".join(bytes([i]) * 3200 for i in range(10))


def test_too_long_message_text() -> None:
    assert msg_too_long_audio(30.0) == "Recording too long, using the first 30 s."
    assert msg_too_long_audio(12.5) == "Recording too long, using the first 12.5 s."


def test_partial_frame_at_limit_is_cut(tmp_path: Path) -> None:
    stt = FakeSTT("Hello")
    client = _client(tmp_path, stt, max_utterance_s=1.0)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == LISTENING
        ws.send_bytes(b"\x01" * 30000)
        ws.send_bytes(b"\x02" * 3000)
        assert ws.receive_json() == {"type": "error", "message": msg_too_long_audio(1.0)}
        ws.send_json({"type": "audio_end"})
        _collect(ws)
    assert stt.calls[0][0] == b"\x01" * 30000 + b"\x02" * 2000


def test_too_long_error_resets_per_recording(tmp_path: Path) -> None:
    stt = FakeSTT("Hello")
    client = _client(tmp_path, stt, max_utterance_s=1.0)
    too_long = {"type": "error", "message": msg_too_long_audio(1.0)}
    with client.websocket_connect("/ws") as ws:
        for _ in range(2):
            ws.send_json({"type": "audio_start"})
            assert ws.receive_json() == LISTENING
            for _ in range(11):
                ws.send_bytes(FRAME)
            assert ws.receive_json() == too_long
            ws.send_json({"type": "audio_end"})
            assert _collect(ws)[0] == THINKING
    assert [len(c[0]) for c in stt.calls] == [32000, 32000]


def test_stray_frame_error_once_per_connection(tmp_path: Path) -> None:
    client = _client(tmp_path, FakeSTT())
    with client.websocket_connect("/ws") as ws:
        ws.send_bytes(FRAME)
        assert ws.receive_json() == {"type": "error", "message": MSG_UNEXPECTED_AUDIO}
        ws.send_bytes(FRAME)
        ws.send_bytes(FRAME)
        _assert_quiet(ws)
        # A later recording does not include the stray frames.
        _speak(ws, frames=4)
        _collect(ws)
        ws.send_bytes(FRAME)  # stray again after the turn: still no second error
        _assert_quiet(ws)


def test_audio_end_without_start(tmp_path: Path) -> None:
    stt = FakeSTT()
    with _client(tmp_path, stt).websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_end"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NOT_RECORDING}
        _assert_quiet(ws)
    assert stt.calls == []


def test_audio_end_twice(tmp_path: Path) -> None:
    gate = threading.Event()
    stt = FakeSTT("Hi", gate=gate)
    with _client(tmp_path, stt).websocket_connect("/ws") as ws:
        _speak(ws, frames=4)
        assert ws.receive_json() == THINKING
        ws.send_json({"type": "audio_end"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NOT_RECORDING}
        gate.set()
        msgs = _collect(ws)
        assert msgs.count(IDLE) == 1
    assert len(stt.calls) == 1


def test_too_short_recording(tmp_path: Path) -> None:
    stt = FakeSTT()
    llm = FakeLLM(["never"])
    client = _client(tmp_path, stt, llm=llm, min_utterance_ms=300)  # 9600 bytes
    with client.websocket_connect("/ws") as ws:
        _speak(ws, frames=2)  # 200 ms
        assert ws.receive_json() == {"type": "error", "message": MSG_NOT_CAUGHT}
        assert ws.receive_json() == IDLE
        _assert_quiet(ws)
        # 300 ms exactly is long enough.
        _speak(ws, frames=3)
        assert ws.receive_json() == THINKING
        _collect(ws)
    assert [len(c[0]) for c in stt.calls] == [9600]


def test_disconnect_while_recording_discards_buffer(tmp_path: Path) -> None:
    stt = FakeSTT()
    llm = FakeLLM(["never"])
    client = _client(tmp_path, stt, llm=llm)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == LISTENING
        for _ in range(5):
            ws.send_bytes(FRAME)
        _assert_quiet(ws)  # the frames were received
    # A new connection starts clean: there is no recording to end.
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_end"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NOT_RECORDING}
    time.sleep(0.1)
    assert stt.calls == []
    assert llm.calls == []
    assert not stt.cancelled.is_set()


def test_disconnect_during_stt_cancels_turn(tmp_path: Path) -> None:
    gate = threading.Event()  # never set
    stt = FakeSTT(gate=gate)
    llm = FakeLLM(["never"])
    client = _client(tmp_path, stt, llm=llm)
    with client.websocket_connect("/ws") as ws:
        _speak(ws, frames=5)
        assert ws.receive_json() == THINKING
    assert stt.cancelled.wait(timeout=5)
    assert llm.calls == []


def test_voice_and_text_share_history(tmp_path: Path) -> None:
    llm = FakeLLM(["ok"])
    client = _client(tmp_path, FakeSTT("spoken"), llm=llm)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "typed"})
        _collect(ws)
        _speak(ws, frames=4)
        _collect(ws)
    assert [[m.text for m in call] for call in llm.calls] == [
        ["typed"],
        ["typed", "ok", "spoken"],
    ]


def test_language_from_config(tmp_path: Path) -> None:
    stt = FakeSTT("Hi")
    with _client(tmp_path, stt, language="").websocket_connect("/ws") as ws:
        _speak(ws, frames=4)
        _collect(ws)
    assert stt.calls[0][2] is None


def test_empty_model_disables_stt_without_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the STT model must not be loaded")

    monkeypatch.setattr(fw.FasterWhisperSTT, "load", boom)
    settings = _settings(tmp_path, model="")
    with caplog.at_level(logging.INFO, logger="obsilord.app"):
        app = create_app(settings, llm=FakeLLM(["x"]))
    assert app.state.stt is None
    assert any("voice input is off" in r.getMessage() for r in caplog.records)


def test_configured_model_is_loaded_and_warmed_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []

    class Loaded:
        def warm_up(self, language: str | None) -> None:
            calls.append(("warm_up", language))

    def fake_load(*args: Any) -> Loaded:
        calls.append(args)
        return Loaded()

    monkeypatch.setattr(fw.FasterWhisperSTT, "load", staticmethod(fake_load))
    settings = _settings(tmp_path, model="small", beam_size=2, download_root="models")
    app = create_app(settings, llm=FakeLLM(["x"]))
    assert isinstance(app.state.stt, Loaded)
    assert calls == [
        ("small", "cpu", "int8", settings.repo_root / "models", 2),
        ("warm_up", "en"),
    ]


def test_relative_model_dir_resolves_against_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "models" / "my-whisper").mkdir(parents=True)
    (repo / "web").mkdir()  # StaticFiles requires the directory to exist
    calls: list[Any] = []

    class Loaded:
        def warm_up(self, language: str | None) -> None:
            pass

    def fake_load(*args: Any) -> Loaded:
        calls.append(args)
        return Loaded()

    monkeypatch.setattr(fw.FasterWhisperSTT, "load", staticmethod(fake_load))
    # The working directory is not the repo root, as when started elsewhere.
    monkeypatch.chdir(tmp_path)
    base = _settings(tmp_path, model="models/my-whisper")
    settings = dataclasses.replace(base, repo_root=repo)
    create_app(settings, llm=FakeLLM(["x"]))
    assert calls[0][0] == str(repo / "models" / "my-whisper")


def test_failed_model_load_stops_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(*args: Any) -> Any:
        raise ConnectionError("offline")

    monkeypatch.setattr(fw.FasterWhisperSTT, "load", staticmethod(fail))
    settings = _settings(tmp_path, model="small")
    with caplog.at_level(logging.ERROR, logger="obsilord.app"):
        with pytest.raises(ConnectionError):
            create_app(settings, llm=FakeLLM(["x"]))
    assert any('model = ""' in r.getMessage() for r in caplog.records)


# Recording safety timer: a recording with no audio_end is closed by the server.
# The watchdog's sleep is replaced by FakeTimers, so the tests decide when a
# timer fires and see when one is cancelled, without wall-clock margins.


class FakeTimers:
    """Replaces obsilord.app._watchdog_sleep. Timer i (0-based, in start order)
    returns once release(i) is called; cancellations are recorded."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self.seconds: list[float] = []
        self._release: list[threading.Event] = []
        self._cancelled: list[threading.Event] = []

    async def sleep(self, seconds: float) -> None:
        with self._guard:
            index = len(self.seconds)
            self.seconds.append(seconds)
            self._release.append(threading.Event())
            self._cancelled.append(threading.Event())
        try:
            while not self._release[index].is_set():
                await asyncio.sleep(0.005)
        except asyncio.CancelledError:
            self._cancelled[index].set()
            raise

    def count(self) -> int:
        with self._guard:
            return len(self.seconds)

    def wait_started(self, n: int, timeout: float = 10.0) -> None:
        """Block until at least n timers have started."""
        deadline = time.monotonic() + timeout
        while self.count() < n:
            assert time.monotonic() < deadline, f"timer {n} never started"
            time.sleep(0.005)

    def release(self, index: int) -> None:
        self._release[index].set()

    def wait_cancelled(self, index: int, timeout: float = 10.0) -> bool:
        return self._cancelled[index].wait(timeout)

    def is_cancelled(self, index: int) -> bool:
        return self._cancelled[index].is_set()


@pytest.fixture()
def timers(monkeypatch: pytest.MonkeyPatch) -> FakeTimers:
    import obsilord.app as app_module

    fake = FakeTimers()
    monkeypatch.setattr(app_module, "_watchdog_sleep", fake.sleep)
    return fake


def _watchdog_fired(caplog: pytest.LogCaptureFixture) -> int:
    return sum("no audio_end within" in r.getMessage() for r in caplog.records)


def test_recording_timeout_is_max_utterance_plus_grace(
    tmp_path: Path, timers: FakeTimers
) -> None:
    from obsilord.app import RECORDING_GRACE_S

    assert RECORDING_GRACE_S == 5.0
    client = _client(tmp_path, FakeSTT(), max_utterance_s=12.5)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == LISTENING
        timers.wait_started(1)
    assert timers.seconds == [17.5]


def test_stuck_recording_runs_voice_turn(
    tmp_path: Path, timers: FakeTimers, caplog: pytest.LogCaptureFixture
) -> None:
    stt = FakeSTT("Hello")
    client = _client(tmp_path, stt)
    with caplog.at_level(logging.WARNING, logger="obsilord.app"):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "audio_start"})
            assert ws.receive_json() == LISTENING
            for _ in range(5):
                ws.send_bytes(FRAME)
            _assert_quiet(ws)  # all frames are in before the timer fires
            timers.wait_started(1)
            timers.release(0)  # no audio_end: the server closes the recording
            msgs = _collect(ws)
            assert msgs[:2] == [
                THINKING,
                {"type": "transcript", "role": "user", "text": "Hello", "final": True},
            ]
            # The socket is usable again.
            ws.send_json({"type": "audio_start"})
            assert ws.receive_json() == LISTENING
    assert stt.calls[0][0] == FRAME * 5
    assert _watchdog_fired(caplog) == 1


def test_stuck_short_recording_says_didnt_catch(tmp_path: Path, timers: FakeTimers) -> None:
    llm = FakeLLM(["typed reply"])
    client = _client(tmp_path, FakeSTT(), llm=llm)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == LISTENING
        timers.wait_started(1)
        timers.release(0)
        assert ws.receive_json() == {"type": "error", "message": MSG_NOT_CAUGHT}
        assert ws.receive_json() == IDLE
        # Text is no longer rejected as busy, and a late audio_end is harmless.
        ws.send_json({"type": "audio_end"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NOT_RECORDING}
        ws.send_json({"type": "text", "text": "hi"})
        assert _collect(ws)[0] == THINKING
    assert len(llm.calls) == 1


def test_audio_end_cancels_timer(
    tmp_path: Path, timers: FakeTimers, caplog: pytest.LogCaptureFixture
) -> None:
    stt = FakeSTT("Hello")
    client = _client(tmp_path, stt)
    with caplog.at_level(logging.WARNING, logger="obsilord.app"):
        with client.websocket_connect("/ws") as ws:
            _speak(ws, frames=4)
            _collect(ws)
            timers.wait_started(1)
            assert timers.wait_cancelled(0)
            _assert_quiet(ws)
    assert _watchdog_fired(caplog) == 0
    assert len(stt.calls) == 1


def test_stale_timer_does_not_close_next_recording(tmp_path: Path, timers: FakeTimers) -> None:
    stt = FakeSTT("Hello")
    client = _client(tmp_path, stt)
    with client.websocket_connect("/ws") as ws:
        _speak(ws, frames=0)  # recording A, ended at once (too short)
        assert ws.receive_json() == {"type": "error", "message": MSG_NOT_CAUGHT}
        assert ws.receive_json() == IDLE
        timers.wait_started(1)
        assert timers.wait_cancelled(0)
        ws.send_json({"type": "audio_start"})  # recording B
        assert ws.receive_json() == LISTENING
        timers.wait_started(2)
        timers.release(0)  # A's timer was cancelled; releasing it changes nothing
        for _ in range(4):
            ws.send_bytes(FRAME)
        ws.send_json({"type": "audio_end"})
        msgs = _collect(ws)
        assert msgs[0] == THINKING
        assert timers.wait_cancelled(1)
    assert stt.calls[0][0] == FRAME * 4


def test_disconnect_cancels_timer(
    tmp_path: Path, timers: FakeTimers, caplog: pytest.LogCaptureFixture
) -> None:
    stt = FakeSTT("Hello")
    client = _client(tmp_path, stt)
    with caplog.at_level(logging.WARNING, logger="obsilord.app"):
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "audio_start"})
            assert ws.receive_json() == LISTENING
            for _ in range(5):
                ws.send_bytes(FRAME)
            timers.wait_started(1)
        assert timers.wait_cancelled(0)
    assert _watchdog_fired(caplog) == 0
    assert stt.calls == []


def test_watchdog_close_is_atomic_with_audio_start(
    tmp_path: Path, timers: FakeTimers, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The server pauses right after sending the watchdog's error (before its
    # idle) until the test has sent an audio_start. Without the recording lock
    # that audio_start would be accepted in the gap and "listening" would land
    # between the error and the idle. With it, the audio_start waits and the
    # error and idle stay together. A slow machine can only make this pass for
    # the wrong reason, never fail for the right one.
    from starlette.websockets import WebSocket

    caught = {"type": "error", "message": MSG_NOT_CAUGHT}
    resume = threading.Event()
    real_send_json = WebSocket.send_json

    async def pausing_send_json(self: WebSocket, data: Any, mode: str = "text") -> None:
        await real_send_json(self, data, mode)
        if data == caught and not resume.is_set():
            while not resume.is_set():
                await asyncio.sleep(0.005)

    monkeypatch.setattr(WebSocket, "send_json", pausing_send_json)
    client = _client(tmp_path, FakeSTT())
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == LISTENING
        timers.wait_started(1)
        timers.release(0)
        assert ws.receive_json() == caught  # the watchdog is now paused before idle
        ws.send_json({"type": "audio_start"})
        time.sleep(0.1)  # let the server read it while the watchdog is paused
        resume.set()
        assert ws.receive_json() == IDLE
        assert ws.receive_json() == LISTENING
