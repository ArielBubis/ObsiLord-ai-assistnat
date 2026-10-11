"""Pipeline with speech output: FakeLLM + FakeTTS, no model, no network."""

import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeLLM, FakeTTS
from obsilord.llm.base import Message, UsageUpdate
from obsilord.pipeline import (
    MSG_LLM_FAILED,
    MSG_TIMEOUT,
    MSG_TTS_FAILED,
    MSG_UNEXPECTED,
    run_text_turn,
)
from obsilord.usage import UsageTracker

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
THINKING = {"type": "state", "state": "thinking"}
SPEAKING = {"type": "state", "state": "speaking"}
IDLE = {"type": "state", "state": "idle"}

Sent = list[dict[str, Any] | bytes]


def _usage(tmp_path: Path, cap: int = 1000) -> UsageTracker:
    return UsageTracker(tmp_path / "usage.jsonl", cap, clock=lambda: NOW)


def _partial(text: str) -> dict[str, Any]:
    return {"type": "transcript", "role": "assistant", "text": text, "final": False}


def _final(text: str) -> dict[str, Any]:
    return {"type": "transcript", "role": "assistant", "text": text, "final": True}


def _start(n: int) -> dict[str, Any]:
    return {"type": "tts_start", "sentence_id": n, "sample_rate": 22050}


def _end(n: int) -> dict[str, Any]:
    return {"type": "tts_end", "sentence_id": n}


def _frame(call: int, j: int) -> bytes:
    return bytes([call, j]) * 2


async def _turn(
    llm: Any,
    tts: Any,
    usage: Any,
    sent: Sent,
    history: list[Message] | None = None,
    timeout_s: float = 60.0,
    on_send: Any = None,
) -> str | None:
    async def send(msg: dict[str, Any] | bytes) -> None:
        sent.append(msg)
        if on_send is not None:
            on_send(msg)

    return await run_text_turn(
        "hello",
        history if history is not None else [],
        llm,
        usage,
        send,
        turn_timeout_s=timeout_s,
        tts=tts,
    )


def _run(llm: Any, tts: Any, usage: Any, **kw: Any) -> tuple[str | None, Sent]:
    sent: Sent = []
    reply = asyncio.run(asyncio.wait_for(_turn(llm, tts, usage, sent, **kw), 10))
    return reply, sent


def _assert_no_binary_after_idle(sent: Sent) -> None:
    assert sent[-1] == IDLE
    assert sent.count(IDLE) == 1
    assert not any(isinstance(m, bytes) for m in sent[sent.index(IDLE):])


def test_exact_sequence_with_tts(tmp_path: Path) -> None:
    llm = FakeLLM(["Hi there. ", "Bye now."])
    tts = FakeTTS(frames=2)
    history: list[Message] = []
    reply, sent = _run(llm, tts, _usage(tmp_path), history=history)
    assert reply == "Hi there. Bye now."
    # The fakes never suspend, so the LLM finishes before the worker runs.
    assert sent == [
        THINKING,
        _partial("Hi there. "),
        _partial("Hi there. Bye now."),
        _final("Hi there. Bye now."),
        _start(1),
        SPEAKING,
        _frame(1, 0),
        _frame(1, 1),
        _end(1),
        _start(2),
        _frame(2, 0),
        _frame(2, 1),
        _end(2),
        IDLE,
    ]
    assert tts.texts == ["Hi there.", "Bye now."]
    assert history == [Message("user", "hello"), Message("assistant", "Hi there. Bye now.")]


def test_tts_runs_concurrently_with_llm(tmp_path: Path) -> None:
    gate = threading.Event()  # the LLM stalls after its first delta until audio is sent

    def on_send(msg: dict[str, Any] | bytes) -> None:
        if isinstance(msg, bytes):
            gate.set()

    llm = FakeLLM(["First sentence. ", "Second one."], gate=gate)
    tts = FakeTTS(frames=3)
    reply, sent = _run(llm, tts, _usage(tmp_path), on_send=on_send)
    assert reply == "First sentence. Second one."
    first_audio = next(i for i, m in enumerate(sent) if isinstance(m, bytes))
    final_at = sent.index(_final("First sentence. Second one."))
    assert first_audio < final_at
    assert sent[first_audio - 2 : first_audio] == [_start(1), SPEAKING]
    assert sent.count(SPEAKING) == 1
    # Speaking is not sent on the first text delta when TTS is on.
    assert sent[1] == _partial("First sentence. ")
    assert [m for m in sent if isinstance(m, dict) and m["type"] == "tts_start"] == [
        _start(1),
        _start(2),
    ]
    _assert_no_binary_after_idle(sent)
    assert sent[-2] == _end(2)


def test_sentence_ids_and_frames_are_ordered(tmp_path: Path) -> None:
    llm = FakeLLM(["One. Two. ", "Three. Four."])
    tts = FakeTTS(frames=2, frame_delay_s=0.001)
    _, sent = _run(llm, tts, _usage(tmp_path))
    tts_msgs = [m for m in sent if isinstance(m, bytes) or m["type"] in ("tts_start", "tts_end")]
    expected: Sent = []
    for n in range(1, 5):
        expected += [_start(n), _frame(n, 0), _frame(n, 1), _end(n)]
    assert tts_msgs == expected
    assert tts.texts == ["One.", "Two.", "Three.", "Four."]


def test_unspeakable_sentences_are_skipped(tmp_path: Path) -> None:
    llm = FakeLLM(["**\n", "---\n", "**Hello** there."])
    tts = FakeTTS(frames=1)
    _, sent = _run(llm, tts, _usage(tmp_path))
    assert tts.texts == ["Hello there."]
    assert [m for m in sent if isinstance(m, dict) and m["type"].startswith("tts")] == [
        _start(1),
        _end(1),
    ]
    # The transcript keeps the raw text.
    assert _final("**\n---\n**Hello** there.") in sent


def test_tts_failure_mid_sentence_continues(tmp_path: Path) -> None:
    llm = FakeLLM(["One. ", "Two. ", "Three."])
    tts = FakeTTS(frames=2, fail_on={1, 2}, fail_after_frames=1)
    reply, sent = _run(llm, tts, _usage(tmp_path))
    assert reply == "One. Two. Three."
    tail = sent[sent.index(_final("One. Two. Three.")) + 1 :]
    assert tail == [
        _start(1),
        SPEAKING,
        _frame(1, 0),
        _end(1),
        {"type": "error", "message": MSG_TTS_FAILED},
        _start(2),
        _frame(2, 0),
        _end(2),
        _start(3),
        _frame(3, 0),
        _frame(3, 1),
        _end(3),
        IDLE,
    ]


def test_tts_failure_before_audio_sends_no_start(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    llm = FakeLLM(["One. ", "Two."])
    tts = FakeTTS(frames=2, fail_on={1}, fail_after_frames=0)
    with caplog.at_level("ERROR"):
        reply, sent = _run(llm, tts, _usage(tmp_path))
    assert reply == "One. Two."
    tail = sent[sent.index(_final("One. Two.")) + 1 :]
    assert tail == [
        {"type": "error", "message": MSG_TTS_FAILED},
        _start(1),
        SPEAKING,
        _frame(2, 0),
        _frame(2, 1),
        _end(1),
        IDLE,
    ]
    assert "TTS failed: RuntimeError" in caplog.text
    assert "One." not in caplog.text


def test_all_tts_failing_still_ends_idle(tmp_path: Path) -> None:
    llm = FakeLLM(["One. ", "Two."])
    tts = FakeTTS(frames=2, fail_on={1, 2})
    reply, sent = _run(llm, tts, _usage(tmp_path))
    assert reply == "One. Two."
    assert sent.count({"type": "error", "message": MSG_TTS_FAILED}) == 1
    assert not any(isinstance(m, bytes) for m in sent)
    assert SPEAKING not in sent
    assert sent[-1] == IDLE


def test_tts_disabled_keeps_phase2_sequence(tmp_path: Path) -> None:
    llm = FakeLLM(["Hi ", "there."])
    _, sent = _run(llm, None, _usage(tmp_path))
    assert sent == [
        THINKING,
        SPEAKING,
        _partial("Hi "),
        _partial("Hi there."),
        _final("Hi there."),
        IDLE,
    ]


def test_cancellation_stops_tts(tmp_path: Path) -> None:
    llm = FakeLLM(["Long sentence here. ", "More."])
    tts = FakeTTS(frames=200, frame_delay_s=0.01)
    usage = _usage(tmp_path)
    sent: Sent = []

    async def main() -> None:
        task = asyncio.create_task(_turn(llm, tts, usage, sent))
        for _ in range(500):
            if any(isinstance(m, bytes) for m in sent):
                break
            await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        count = len(sent)
        await asyncio.sleep(0.1)
        assert len(sent) == count  # nothing sent after the turn was cancelled

    asyncio.run(asyncio.wait_for(main(), 10))
    assert tts.cancelled.is_set()
    assert tts.closed == 1
    assert usage.today_total == 17  # usage recorded (Done arrived before audio)


def test_llm_error_cancels_tts_and_no_binary_after_idle(tmp_path: Path) -> None:
    gate = threading.Event()

    def on_send(msg: dict[str, Any] | bytes) -> None:
        if isinstance(msg, bytes):
            gate.set()

    llm = FakeLLM(["Spoken first. ", "then", " more"], gate=gate, fail_after=1)
    tts = FakeTTS(frames=200, frame_delay_s=0.01)
    history: list[Message] = []
    reply, sent = _run(llm, tts, _usage(tmp_path), on_send=on_send, history=history)
    assert reply is None
    assert sent[-2:] == [{"type": "error", "message": MSG_LLM_FAILED}, IDLE]
    assert any(isinstance(m, bytes) for m in sent)
    assert not any(isinstance(m, dict) and m.get("final") for m in sent)
    _assert_no_binary_after_idle(sent)
    assert tts.cancelled.is_set()
    assert history == []


def test_timeout_during_llm_cancels_tts(tmp_path: Path) -> None:
    llm = FakeLLM(["Spoken first. ", "never"], usage=UsageUpdate(5, 1, 0), hang_s=5)
    tts = FakeTTS(frames=200, frame_delay_s=0.01)
    usage = _usage(tmp_path)
    reply, sent = _run(llm, tts, usage, timeout_s=0.3)
    assert reply is None
    assert sent[-2:] == [{"type": "error", "message": MSG_TIMEOUT}, IDLE]
    assert any(isinstance(m, bytes) for m in sent)
    _assert_no_binary_after_idle(sent)
    assert tts.cancelled.is_set()
    assert usage.today_total == 6


def test_timeout_during_speech_after_llm_done(tmp_path: Path) -> None:
    llm = FakeLLM(["A long answer."])
    tts = FakeTTS(frames=200, frame_delay_s=0.01)
    history: list[Message] = []
    reply, sent = _run(llm, tts, _usage(tmp_path), timeout_s=0.3, history=history)
    assert reply is None
    assert _final("A long answer.") in sent
    assert sent[-2:] == [{"type": "error", "message": MSG_TIMEOUT}, IDLE]
    _assert_no_binary_after_idle(sent)
    assert tts.cancelled.is_set()
    # The text reply was complete and shown, so it stays in the history.
    assert history == [Message("user", "hello"), Message("assistant", "A long answer.")]


def test_binary_send_failure_ends_turn(tmp_path: Path) -> None:
    llm = FakeLLM(["One. ", "Two."])
    tts = FakeTTS(frames=2)
    usage = _usage(tmp_path)
    sent: Sent = []

    async def send(msg: dict[str, Any] | bytes) -> None:
        if isinstance(msg, bytes):
            raise ConnectionError("socket closed")
        sent.append(msg)

    reply = asyncio.run(run_text_turn("hello", [], llm, usage, send, tts=tts))
    assert reply is None
    assert sent[-2:] == [{"type": "error", "message": MSG_UNEXPECTED}, IDLE]
    assert usage.today_total == 17


def test_latency_log_has_no_text(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM(["Secret words here."])
    with caplog.at_level("INFO"):
        _run(llm, FakeTTS(frames=1), _usage(tmp_path))
    line = next(r.getMessage() for r in caplog.records if "turn latency" in r.getMessage())
    assert "stt_ms=n/a" in line
    assert "llm_ttft_ms=" in line and "tts_first_chunk_ms=" in line
    assert "llm_done_ms=" in line and "llm_done_ms=-" not in line
    assert "total_ms=" in line and "sentences=1" in line and "spoken=1" in line
    assert "tts_first_chunk_ms=-" not in line
    assert "Secret" not in caplog.text
