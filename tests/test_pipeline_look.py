"""Pipeline with look(): scripted LLM requests and a fake look tool. No camera,
no model, no network."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fakes import FakeLLM, FakeLooker, FakeToolLLM, FakeTTS
from obsilord.llm.base import Done, Message, TextDelta, ToolCall, ToolResult, UsageUpdate
from obsilord.look import LOOK_TOOL, MSG_ONE_LOOK
from obsilord.pipeline import MSG_CAP, MSG_LLM_FAILED, MSG_TIMEOUT, run_text_turn
from obsilord.usage import UsageTracker

NOW = datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc)
THINKING = {"type": "state", "state": "thinking"}
LOOKING = {"type": "state", "state": "looking"}
SPEAKING = {"type": "state", "state": "speaking"}
IDLE = {"type": "state", "state": "idle"}

Sent = list[dict[str, Any] | bytes]


def _usage(tmp_path: Path, cap: int = 1000) -> UsageTracker:
    return UsageTracker(tmp_path / "usage.jsonl", cap, clock=lambda: NOW)


def _partial(text: str) -> dict[str, Any]:
    return {"type": "transcript", "role": "assistant", "text": text, "final": False}


def _final(text: str) -> dict[str, Any]:
    return {"type": "transcript", "role": "assistant", "text": text, "final": True}


def _look(question: str = "What am I holding?", call_id: str = "c1") -> ToolCall:
    return ToolCall("look", {"question": question}, call_id=call_id, raw=["raw parts"])


def _run(
    llm: Any,
    looker: Any,
    usage: UsageTracker,
    tts: Any = None,
    history: list[Message] | None = None,
    timeout_s: float = 60.0,
) -> tuple[str | None, Sent]:
    sent: Sent = []

    async def send(msg: dict[str, Any] | bytes) -> None:
        sent.append(msg)

    reply = asyncio.run(
        run_text_turn(
            "hello",
            history if history is not None else [],
            llm,
            usage,
            send,
            turn_timeout_s=timeout_s,
            tts=tts,
            looker=looker,
        )
    )
    return reply, sent


def _log_lines(tmp_path: Path) -> list[dict[str, Any]]:
    text = (tmp_path / "usage.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def test_look_then_follow_up_text_only(tmp_path: Path) -> None:
    call = _look()
    llm = FakeToolLLM(
        [
            [UsageUpdate(10, 1, 0), call, Done(10, 2, 0)],
            [TextDelta("You are holding "), TextDelta("a red mug."), Done(20, 5, 1)],
        ]
    )
    looker = FakeLooker({"output": "A red mug."})
    usage = _usage(tmp_path)
    history: list[Message] = []
    reply, sent = _run(llm, looker, usage, history=history)

    assert reply == "You are holding a red mug."
    assert sent == [
        THINKING,
        LOOKING,
        THINKING,
        SPEAKING,
        _partial("You are holding "),
        _partial("You are holding a red mug."),
        _final("You are holding a red mug."),
        IDLE,
    ]
    assert looker.questions == ["What am I holding?"]
    first, second = llm.requests
    assert first.tools == [LOOK_TOOL] and first.tool_results is None
    assert first.allow_tool_calls is True
    assert second.tools == [LOOK_TOOL]
    assert second.tool_results == [ToolResult(call, {"output": "A red mug."})]
    assert second.tool_results[0].call.raw == ["raw parts"]
    assert second.allow_tool_calls is False
    # Both requests see the same history: the tool exchange is not stored in it.
    assert first.history == second.history == [Message("user", "hello")]
    assert history == [Message("user", "hello"), Message("assistant", "You are holding a red mug.")]
    # One usage line per request.
    assert [line["total"] for line in _log_lines(tmp_path)] == [12, 26]
    assert usage.today_total == 38


def test_speech_before_the_call_is_sent_before_looking(tmp_path: Path) -> None:
    llm = FakeToolLLM(
        [
            [TextDelta("Let me look"), _look(), Done(5, 2, 0)],
            [TextDelta("It is a red mug."), Done(9, 4, 0)],
        ]
    )
    tts = FakeTTS(frames=1)
    reply, sent = _run(llm, FakeLooker(), _usage(tmp_path), tts=tts)

    # The two requests' text is joined with a space.
    assert reply == "Let me look It is a red mug."
    # "Let me look" has no full stop, so only the flush before the look
    # releases it: it is spoken first, then looking, then the answer.
    assert tts.texts == ["Let me look", "It is a red mug."]
    looking = sent.index(LOOKING)
    assert sent[looking - 1] == {"type": "tts_end", "sentence_id": 1}
    assert sent[looking + 1] == THINKING
    states = [m["state"] for m in sent if isinstance(m, dict) and m.get("type") == "state"]
    assert states == ["thinking", "speaking", "looking", "thinking", "speaking", "idle"]
    assert _final("Let me look It is a red mug.") in sent
    assert sent[-1] == IDLE


def test_no_extra_space_when_the_text_already_has_one(tmp_path: Path) -> None:
    llm = FakeToolLLM(
        [
            [TextDelta("One moment. "), _look(), Done(5, 2, 0)],
            [TextDelta("A mug."), Done(9, 4, 0)],
        ]
    )
    reply, _ = _run(llm, FakeLooker(), _usage(tmp_path))
    assert reply == "One moment. A mug."


def test_cap_reached_before_follow_up(tmp_path: Path) -> None:
    llm = FakeToolLLM([[_look(), Done(10, 2, 0)], [TextDelta("never"), Done(1, 1, 0)]])
    history: list[Message] = []
    reply, sent = _run(llm, FakeLooker(), _usage(tmp_path, cap=10), history=history)

    assert reply is None
    assert sent == [THINKING, LOOKING, THINKING, {"type": "error", "message": MSG_CAP}, IDLE]
    assert len(llm.requests) == 1
    assert history == []
    assert [line["total"] for line in _log_lines(tmp_path)] == [12]


def test_only_first_look_runs(tmp_path: Path) -> None:
    first, extra = _look("one", "a"), _look("two", "b")
    llm = FakeToolLLM([[first, extra, Done(1, 1, 0)], [TextDelta("Done."), Done(1, 1, 0)]])
    looker = FakeLooker()
    _run(llm, looker, _usage(tmp_path))

    assert looker.questions == ["one"]
    results = llm.requests[1].tool_results
    assert results == [
        ToolResult(first, {"output": "A red mug."}),
        ToolResult(extra, {"error": MSG_ONE_LOOK}),
    ]


def test_unknown_tool_gets_an_error_result(tmp_path: Path) -> None:
    call = ToolCall("open_door", {}, call_id="x")
    llm = FakeToolLLM([[call, Done(1, 1, 0)], [TextDelta("I cannot."), Done(1, 1, 0)]])
    looker = FakeLooker()
    reply, _ = _run(llm, looker, _usage(tmp_path))

    assert reply == "I cannot."
    assert looker.questions == []
    assert llm.requests[1].tool_results == [ToolResult(call, {"error": "Unknown tool: open_door."})]


def test_look_refusal_goes_back_to_the_model(tmp_path: Path) -> None:
    refusal = {"error": "Camera cooldown: try again in 7 s."}
    llm = FakeToolLLM([[_look(), Done(1, 1, 0)], [TextDelta("Give me a moment."), Done(1, 1, 0)]])
    reply, _ = _run(llm, FakeLooker(refusal), _usage(tmp_path))

    assert reply == "Give me a moment."
    assert llm.requests[1].tool_results[0].response == refusal


def test_no_looker_declares_no_tools_and_ignores_calls(tmp_path: Path) -> None:
    llm = FakeToolLLM([[TextDelta("Hi."), _look(), Done(1, 1, 0)]])
    reply, sent = _run(llm, None, _usage(tmp_path))

    assert reply == "Hi."
    assert len(llm.requests) == 1 and llm.requests[0].tools == []
    assert LOOKING not in sent


def test_plain_fake_llm_gets_the_tool_declared(tmp_path: Path) -> None:
    llm = FakeLLM(["Hello."])
    reply, _ = _run(llm, FakeLooker(), _usage(tmp_path))

    assert reply == "Hello."
    assert llm.tools == [[LOOK_TOOL]]


def test_follow_up_failure_is_reported_and_recorded(tmp_path: Path) -> None:
    llm = FakeToolLLM([[_look(), Done(4, 1, 0)], [RuntimeError("network")]])
    history: list[Message] = []
    reply, sent = _run(llm, FakeLooker(), _usage(tmp_path), history=history)

    assert reply is None
    assert sent[-2:] == [{"type": "error", "message": MSG_LLM_FAILED}, IDLE]
    assert history == []
    lines = _log_lines(tmp_path)
    assert len(lines) == 2
    assert lines[1]["estimated"] is True  # no counts from the failed request


def test_timeout_during_look(tmp_path: Path) -> None:
    llm = FakeToolLLM([[_look(), Done(1, 1, 0)], [TextDelta("late"), Done(1, 1, 0)]])
    reply, sent = _run(llm, FakeLooker(delay_s=2.0), _usage(tmp_path), timeout_s=0.2)

    assert reply is None
    assert sent[-2:] == [{"type": "error", "message": MSG_TIMEOUT}, IDLE]
    assert len(llm.requests) == 1
