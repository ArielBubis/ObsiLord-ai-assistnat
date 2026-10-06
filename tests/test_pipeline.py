import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeLLM
from pixelbuddy.llm.base import Done, Message, UsageUpdate
from pixelbuddy.pipeline import (
    MSG_CAP,
    MSG_LLM_FAILED,
    MSG_TIMEOUT,
    MSG_UNEXPECTED,
    estimate_tokens,
    run_text_turn,
    trim_history,
)
from pixelbuddy.usage import UsageTracker

NOW = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)


def _usage(tmp_path: Path, cap: int = 1000) -> UsageTracker:
    return UsageTracker(tmp_path / "usage.jsonl", cap, clock=lambda: NOW)


def _run(
    text: str,
    history: list[Message],
    llm: Any,
    usage: Any,
    turns: int = 10,
    timeout_s: float = 60.0,
) -> tuple[str | None, list[dict[str, Any]]]:
    sent: list[dict[str, Any]] = []

    async def send(msg: dict[str, Any]) -> None:
        sent.append(msg)

    reply = asyncio.run(
        run_text_turn(
            text,
            history,
            llm,
            usage,
            send,
            history_turns=turns,
            max_output_tokens=200,
            turn_timeout_s=timeout_s,
        )
    )
    return reply, sent


def _entries(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "usage.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_message_sequence(tmp_path: Path) -> None:
    llm = FakeLLM(["Hi ", "there. ", "Bye."])
    usage = _usage(tmp_path)
    history: list[Message] = []
    reply, sent = _run("hello", history, llm, usage)
    assert reply == "Hi there. Bye."
    assert sent == [
        {"type": "state", "state": "thinking"},
        {"type": "state", "state": "speaking"},
        {"type": "transcript", "role": "assistant", "text": "Hi ", "final": False},
        {"type": "transcript", "role": "assistant", "text": "Hi there. ", "final": False},
        {"type": "transcript", "role": "assistant", "text": "Hi there. Bye.", "final": False},
        {"type": "transcript", "role": "assistant", "text": "Hi there. Bye.", "final": True},
        {"type": "state", "state": "idle"},
    ]
    assert history == [Message("user", "hello"), Message("assistant", "Hi there. Bye.")]
    assert llm.calls == [[Message("user", "hello")]]
    assert usage.today_total == 17
    entry = json.loads((tmp_path / "usage.jsonl").read_text(encoding="utf-8"))
    assert entry["model"] == "fake-model"
    assert (entry["tokens_in"], entry["tokens_out"], entry["tokens_thinking"]) == (10, 5, 2)


def test_history_passed_to_llm(tmp_path: Path) -> None:
    llm = FakeLLM(["ok"])
    history = [Message("user", "a"), Message("assistant", "b")]
    _run("c", history, llm, _usage(tmp_path))
    assert llm.calls[0] == [Message("user", "a"), Message("assistant", "b"), Message("user", "c")]


def test_cap_blocks_without_calling_llm(tmp_path: Path) -> None:
    usage = _usage(tmp_path, cap=10)
    usage.record("m", 10, 0, 0, 1)
    llm = FakeLLM(["never"])
    history: list[Message] = []
    reply, sent = _run("hello", history, llm, usage)
    assert reply is None
    assert llm.calls == []
    assert sent == [
        {"type": "state", "state": "thinking"},
        {"type": "error", "message": MSG_CAP},
        {"type": "state", "state": "idle"},
    ]
    assert history == []


def test_llm_error_before_text_records_estimate(tmp_path: Path) -> None:
    llm = FakeLLM(["x"], fail_after=0)
    usage = _usage(tmp_path)
    history: list[Message] = []
    reply, sent = _run("hello", history, llm, usage)
    assert reply is None
    assert sent == [
        {"type": "state", "state": "thinking"},
        {"type": "error", "message": MSG_LLM_FAILED},
        {"type": "state", "state": "idle"},
    ]
    assert history == []
    [entry] = _entries(tmp_path)
    assert entry["estimated"] is True and entry["partial"] is True
    assert entry["tokens_in"] == estimate_tokens([Message("user", "hello")]) == 2
    assert entry["tokens_out"] == 200
    assert usage.today_total == 202
    assert llm.closed.is_set()


def test_llm_error_mid_stream_records_last_usage(tmp_path: Path) -> None:
    llm = FakeLLM(["Partial ", "reply"], fail_after=1, usage=UsageUpdate(30, 4, 1))
    usage = _usage(tmp_path)
    history: list[Message] = []
    reply, sent = _run("hello", history, llm, usage)
    assert reply is None
    assert sent[-2:] == [
        {"type": "error", "message": MSG_LLM_FAILED},
        {"type": "state", "state": "idle"},
    ]
    assert {"type": "state", "state": "speaking"} in sent
    assert not any(m.get("final") is True for m in sent)
    assert history == []
    [entry] = _entries(tmp_path)
    assert entry["partial"] is True and entry["estimated"] is False
    assert (entry["tokens_in"], entry["tokens_out"], entry["tokens_thinking"]) == (30, 4, 1)
    assert usage.today_total == 35


def test_done_preferred_over_usage_update(tmp_path: Path) -> None:
    llm = FakeLLM(["Hi."], usage=UsageUpdate(1, 1, 0), done=Done(10, 5, 2))
    _run("hello", [], llm, _usage(tmp_path))
    [entry] = _entries(tmp_path)
    assert entry["partial"] is False and entry["estimated"] is False
    assert entry["total"] == 17


def test_timeout_records_usage_and_reports(tmp_path: Path) -> None:
    llm = FakeLLM(["Slow ", "reply."], usage=UsageUpdate(20, 2, 0), hang_s=5)
    usage = _usage(tmp_path)
    history: list[Message] = []
    reply, sent = _run("hello", history, llm, usage, timeout_s=0.1)
    assert reply is None
    assert sent[-2:] == [
        {"type": "error", "message": MSG_TIMEOUT},
        {"type": "state", "state": "idle"},
    ]
    assert llm.cancelled.is_set() and llm.closed.is_set()
    [entry] = _entries(tmp_path)
    assert entry["partial"] is True
    assert usage.today_total == 22
    assert history == []


def test_unexpected_error_in_check(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    class BrokenUsage:
        today_total = 0

        def check(self) -> None:
            raise OSError("disk gone")

    llm = FakeLLM(["never"])
    with caplog.at_level("ERROR"):
        reply, sent = _run("hello", [], llm, BrokenUsage())
    assert reply is None
    assert llm.calls == []
    assert sent == [
        {"type": "state", "state": "thinking"},
        {"type": "error", "message": MSG_UNEXPECTED},
        {"type": "state", "state": "idle"},
    ]
    assert "OSError" in caplog.text
    assert "disk gone" not in caplog.text


def test_send_failure_is_contained_and_usage_recorded(tmp_path: Path) -> None:
    llm = FakeLLM(["Hi ", "there."], usage=UsageUpdate(9, 1, 0))
    usage = _usage(tmp_path)
    calls = 0

    async def send(msg: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        if msg.get("type") == "transcript":
            raise ConnectionError("socket closed")

    reply = asyncio.run(run_text_turn("hello", [], llm, usage, send))
    assert reply is None
    assert llm.closed.is_set()
    assert usage.today_total == 10


def test_error_log_has_no_prompt(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    llm = FakeLLM(["x"], fail_after=0)
    with caplog.at_level("DEBUG"):
        _run("secret prompt words", [], llm, _usage(tmp_path))
    assert "RuntimeError" in caplog.text
    assert "secret prompt words" not in caplog.text


def test_missing_done_records_estimate(tmp_path: Path) -> None:
    llm = FakeLLM(["Hi."], done=None)
    usage = _usage(tmp_path)
    reply, sent = _run("hello", [], llm, usage)
    assert reply == "Hi."
    assert sent[-1] == {"type": "state", "state": "idle"}
    [entry] = _entries(tmp_path)
    assert entry["estimated"] is True
    assert usage.today_total == 202


def test_history_trimming(tmp_path: Path) -> None:
    history: list[Message] = []
    usage = _usage(tmp_path)
    for i in range(4):
        _run(f"q{i}", history, FakeLLM([f"a{i}"]), usage, turns=2)
    assert history == [
        Message("user", "q2"),
        Message("assistant", "a2"),
        Message("user", "q3"),
        Message("assistant", "a3"),
    ]


def test_trim_history_zero_turns() -> None:
    history = [Message("user", "a"), Message("assistant", "b")]
    trim_history(history, 0)
    assert history == []


def test_empty_reply_not_added_to_history(tmp_path: Path) -> None:
    history: list[Message] = []
    reply, sent = _run("hello", history, FakeLLM([], done=Done(3, 0, 0)), _usage(tmp_path))
    assert reply == ""
    assert sent == [
        {"type": "state", "state": "thinking"},
        {"type": "transcript", "role": "assistant", "text": "", "final": True},
        {"type": "state", "state": "idle"},
    ]
    assert history == []


def test_estimate_includes_prompt_overhead(tmp_path: Path) -> None:
    llm = FakeLLM(["x"], fail_after=0)
    llm.prompt_overhead_chars = 400  # type: ignore[attr-defined]
    usage = _usage(tmp_path)
    _run("hello", [], llm, usage)
    [entry] = _entries(tmp_path)
    assert entry["estimated"] is True
    assert entry["tokens_in"] == estimate_tokens([Message("user", "hello")], 400) == 102
    assert entry["tokens_out"] == 200
