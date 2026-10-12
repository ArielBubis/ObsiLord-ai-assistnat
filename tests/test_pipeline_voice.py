"""Voice turns: FakeSTT + FakeLLM (+ FakeTTS). No model, no network."""

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeLLM, FakeSTT, FakeTTS
from obsilord.llm.base import Message, UsageUpdate
from obsilord.pipeline import (
    MSG_CAP,
    MSG_NOT_CAUGHT,
    MSG_STT_FAILED,
    MSG_TIMEOUT,
    MSG_UNEXPECTED,
    run_voice_turn,
)
from obsilord.usage import CapExceeded, UsageTracker

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
THINKING = {"type": "state", "state": "thinking"}
SPEAKING = {"type": "state", "state": "speaking"}
IDLE = {"type": "state", "state": "idle"}
PCM = bytes(16000)  # 0.5 s of PCM16 silence at 16 kHz

Sent = list[dict[str, Any] | bytes]


def _usage(tmp_path: Path, cap: int = 1000) -> UsageTracker:
    return UsageTracker(tmp_path / "usage.jsonl", cap, clock=lambda: NOW)


def _user(text: str) -> dict[str, Any]:
    return {"type": "transcript", "role": "user", "text": text, "final": True}


def _partial(text: str) -> dict[str, Any]:
    return {"type": "transcript", "role": "assistant", "text": text, "final": False}


def _final(text: str) -> dict[str, Any]:
    return {"type": "transcript", "role": "assistant", "text": text, "final": True}


def _run(
    stt: Any,
    llm: Any,
    usage: Any,
    tts: Any = None,
    history: list[Message] | None = None,
    timeout_s: float = 60.0,
    pcm: bytes = PCM,
    language: str | None = "en",
) -> tuple[str | None, Sent]:
    sent: Sent = []

    async def send(msg: dict[str, Any] | bytes) -> None:
        sent.append(msg)

    reply = asyncio.run(
        asyncio.wait_for(
            run_voice_turn(
                pcm,
                history if history is not None else [],
                llm,
                usage,
                send,
                stt,
                language=language,
                sample_rate=16000,
                turn_timeout_s=timeout_s,
                tts=tts,
            ),
            10,
        )
    )
    return reply, sent


def test_sequence_with_tts(tmp_path: Path) -> None:
    stt = FakeSTT("  What is up?  ")
    llm = FakeLLM(["Not much. ", "You?"])
    tts = FakeTTS(frames=1)
    history: list[Message] = []
    reply, sent = _run(stt, llm, _usage(tmp_path), tts=tts, history=history)
    assert reply == "Not much. You?"
    assert sent == [
        THINKING,
        _user("What is up?"),
        _partial("Not much. "),
        _partial("Not much. You?"),
        _final("Not much. You?"),
        {"type": "tts_start", "sentence_id": 1, "sample_rate": 22050},
        SPEAKING,
        bytes([1, 0]) * 2,
        {"type": "tts_end", "sentence_id": 1},
        {"type": "tts_start", "sentence_id": 2, "sample_rate": 22050},
        bytes([2, 0]) * 2,
        {"type": "tts_end", "sentence_id": 2},
        IDLE,
    ]
    assert sent.count(THINKING) == 1
    assert stt.calls == [(PCM, 16000, "en")]
    # The transcript (stripped) is the user turn sent to the LLM and kept in history.
    assert llm.calls == [[Message("user", "What is up?")]]
    assert history == [Message("user", "What is up?"), Message("assistant", "Not much. You?")]


def test_sequence_without_tts(tmp_path: Path) -> None:
    reply, sent = _run(FakeSTT("Hello"), FakeLLM(["Hi."]), _usage(tmp_path))
    assert reply == "Hi."
    assert sent == [
        THINKING,
        _user("Hello"),
        SPEAKING,
        _partial("Hi."),
        _final("Hi."),
        IDLE,
    ]


def test_language_none_passed_to_stt(tmp_path: Path) -> None:
    stt = FakeSTT("Hola")
    _run(stt, FakeLLM(["x"]), _usage(tmp_path), language=None)
    assert stt.calls[0][2] is None


def test_cap_reached_skips_stt(tmp_path: Path) -> None:
    usage = _usage(tmp_path, cap=10)
    usage.record("m", 10, 0, 0, 1)
    stt = FakeSTT()
    llm = FakeLLM(["never"])
    reply, sent = _run(stt, llm, usage)
    assert reply is None
    assert sent == [THINKING, {"type": "error", "message": MSG_CAP}, IDLE]
    assert stt.calls == []
    assert llm.calls == []


def test_cap_checked_again_after_stt(tmp_path: Path) -> None:
    class CapAfterFirstCheck:
        today_total = 0

        def __init__(self) -> None:
            self.checks = 0

        def check(self) -> None:
            self.checks += 1
            if self.checks > 1:
                raise CapExceeded("cap")

    usage = CapAfterFirstCheck()
    stt = FakeSTT("Hello")
    llm = FakeLLM(["never"])
    reply, sent = _run(stt, llm, usage)
    assert reply is None
    assert sent == [THINKING, _user("Hello"), {"type": "error", "message": MSG_CAP}, IDLE]
    assert usage.checks == 2
    assert len(stt.calls) == 1
    assert llm.calls == []


def test_stt_failure(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM(["never"])
    history: list[Message] = []
    with caplog.at_level("DEBUG"):
        reply, sent = _run(FakeSTT(fail=True), llm, _usage(tmp_path), history=history)
    assert reply is None
    assert sent == [THINKING, {"type": "error", "message": MSG_STT_FAILED}, IDLE]
    assert llm.calls == []
    assert history == []
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("RuntimeError" in r.getMessage() for r in errors)
    # Type name only: the exception message is not logged.
    assert "secret words" not in caplog.text


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_transcript(tmp_path: Path, text: str) -> None:
    llm = FakeLLM(["never"])
    reply, sent = _run(FakeSTT(text), llm, _usage(tmp_path))
    assert reply is None
    assert sent == [THINKING, {"type": "error", "message": MSG_NOT_CAUGHT}, IDLE]
    assert llm.calls == []


def test_timeout_during_stt(tmp_path: Path) -> None:
    stt = FakeSTT(delay_s=5)
    llm = FakeLLM(["never"])
    t0 = time.monotonic()
    reply, sent = _run(stt, llm, _usage(tmp_path), timeout_s=0.2)
    assert time.monotonic() - t0 < 3
    assert reply is None
    assert sent == [THINKING, {"type": "error", "message": MSG_TIMEOUT}, IDLE]
    assert stt.cancelled.is_set()
    assert llm.calls == []


def test_timeout_covers_stt_and_llm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # One deadline for the whole turn: the STT phase and the LLM phase must use
    # the same asyncio.timeout_at deadline, set once at turn start.
    deadlines: list[float] = []
    real_timeout_at = asyncio.timeout_at

    def recording_timeout_at(when: float | None) -> asyncio.Timeout:
        assert when is not None
        deadlines.append(when)
        return real_timeout_at(when)

    monkeypatch.setattr(asyncio, "timeout_at", recording_timeout_at)
    stt = FakeSTT("Hello", delay_s=0.2)
    llm = FakeLLM(["Slow ", "reply."], usage=UsageUpdate(4, 1, 0), hang_s=30)
    usage = _usage(tmp_path)
    t0 = time.monotonic()
    reply, sent = _run(stt, llm, usage, timeout_s=0.5)
    assert time.monotonic() - t0 < 5  # far below the LLM's 30 s hang
    assert reply is None
    assert len(deadlines) == 2 and deadlines[0] == deadlines[1]
    assert sent[:2] == [THINKING, _user("Hello")]
    assert sent[-2:] == [{"type": "error", "message": MSG_TIMEOUT}, IDLE]
    assert sent.count(THINKING) == 1
    assert llm.cancelled.is_set()
    assert usage.today_total == 5


def test_cancel_during_stt_sends_nothing_more(tmp_path: Path) -> None:
    stt = FakeSTT(delay_s=5)
    sent: Sent = []

    async def send(msg: dict[str, Any] | bytes) -> None:
        sent.append(msg)

    async def main() -> None:
        task = asyncio.create_task(
            run_voice_turn(PCM, [], FakeLLM(["x"]), _usage(tmp_path), send, stt)
        )
        while not stt.calls:
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0.05)

    asyncio.run(asyncio.wait_for(main(), 10))
    assert sent == [THINKING]
    assert stt.cancelled.is_set()


def test_send_failure_is_contained(tmp_path: Path) -> None:
    async def send(msg: dict[str, Any] | bytes) -> None:
        if isinstance(msg, dict) and msg.get("role") == "user":
            raise ConnectionError("socket closed")

    llm = FakeLLM(["x"])
    reply = asyncio.run(
        run_voice_turn(PCM, [], llm, _usage(tmp_path), send, FakeSTT("Hello"))
    )
    assert reply is None
    assert llm.calls == []


def test_unexpected_error_reports_generic(tmp_path: Path) -> None:
    class BrokenUsage:
        today_total = 0

        def check(self) -> None:
            raise OSError("disk gone")

    stt = FakeSTT()
    reply, sent = _run(stt, FakeLLM(["x"]), BrokenUsage())
    assert reply is None
    assert sent == [THINKING, {"type": "error", "message": MSG_UNEXPECTED}, IDLE]
    assert stt.calls == []


def test_latency_line_has_stt_and_no_text(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    stt = FakeSTT("Secret question here", delay_s=0.05)
    with caplog.at_level("DEBUG"):
        _run(stt, FakeLLM(["Secret answer."]), _usage(tmp_path), tts=FakeTTS(frames=1))
    line = next(r.getMessage() for r in caplog.records if "turn latency" in r.getMessage())
    fields = dict(part.split("=", 1) for part in line.split(": ", 1)[1].split())
    assert int(fields["stt_ms"]) >= 1
    assert fields["audio_ms"] == "500"
    for key in ("llm_ttft_ms", "llm_done_ms", "tts_first_chunk_ms", "total_ms"):
        assert int(fields[key]) >= int(fields["stt_ms"])
    assert "Secret" not in caplog.text


def test_text_turn_latency_line_says_na(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from obsilord.pipeline import run_text_turn

    async def send(msg: dict[str, Any] | bytes) -> None:
        pass

    with caplog.at_level("INFO"):
        asyncio.run(run_text_turn("hi", [], FakeLLM(["ok"]), _usage(tmp_path), send))
    line = next(r.getMessage() for r in caplog.records if "turn latency" in r.getMessage())
    assert "stt_ms=n/a audio_ms=n/a" in line
