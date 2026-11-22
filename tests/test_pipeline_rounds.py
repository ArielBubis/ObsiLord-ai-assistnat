"""Tool rounds in one turn: the model calls, reads the results, calls again,
up to max_rounds, then answers. Fakes only."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fakes import FakeLooker, FakeToolLLM, registry
from obsilord.llm.base import Done, Message, TextDelta, ToolCall, ToolNote, ToolResult
from obsilord.pipeline import MSG_CAP, run_text_turn
from obsilord.tools.calculate import calculate
from obsilord.tools.registry import Tool, ToolPolicy, tool
from obsilord.usage import UsageTracker

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
LOOKING = {"type": "state", "state": "looking"}


async def _fake_recall(query: str, *, answers: list[str]) -> str:
    """Search earlier conversations.

    Args:
        query: Keywords.
    """
    answers.append(query)
    return "2026-11-18: the bike cost 2400."


FAKE_RECALL = tool(ToolPolicy(private_data=True, untrusted=True), name="recall")(_fake_recall)


def _recall(queries: list[str]) -> Tool:
    return FAKE_RECALL.bind(answers=queries)


def _run(
    llm: Any,
    tmp_path: Path,
    *tools: Tool,
    looker: Any = None,
    history: list[Message] | None = None,
    max_rounds: int = 3,
    cap: int = 1000,
) -> tuple[str | None, list[Any], list[Message]]:
    sent: list[Any] = []
    history = history if history is not None else []

    async def send(msg: Any) -> None:
        sent.append(msg)

    reply = asyncio.run(
        run_text_turn(
            "What is 17% of what I paid for the bike?",
            history,
            llm,
            UsageTracker(tmp_path / "usage.jsonl", cap, clock=lambda: NOW),
            send,
            tools=registry(looker, *tools),
            max_rounds=max_rounds,
        )
    )
    return reply, sent, history


def _states(sent: list[Any]) -> list[str]:
    return [m["state"] for m in sent if isinstance(m, dict) and m.get("type") == "state"]


def test_two_rounds_then_a_reply(tmp_path: Path) -> None:
    c1 = ToolCall("recall", {"query": "bike"}, call_id="r1")
    c2 = ToolCall("calculate", {"expression": "17% of 2400"}, call_id="k1")
    llm = FakeToolLLM(
        [
            [c1, Done(1, 1, 0)],
            [c2, Done(1, 1, 0)],
            [TextDelta("That is 408."), Done(1, 1, 0)],
        ]
    )
    queries: list[str] = []
    reply, _, history = _run(llm, tmp_path, _recall(queries), calculate)

    assert reply == "That is 408."
    assert queries == ["bike"]
    r1 = ToolResult(c1, {"output": "2026-11-18: the bike cost 2400."})
    r2 = ToolResult(c2, {"output": "17% of 2400 = 408"})
    assert [r.rounds for r in llm.requests] == [[], [[r1]], [[r1], [r2]]]
    assert [r.allow_tool_calls for r in llm.requests] == [True, True, True]
    # Every request sees the same history; the rounds go separately.
    assert llm.requests[0].history == llm.requests[2].history
    # The notes of every round, in order.
    assert [n.name for n in history[-1].tools] == ["recall", "calculate"]
    assert history[-1].tools[1] == ToolNote("calculate", "17% of 2400", "17% of 2400 = 408")


def test_the_request_after_the_last_round_forbids_calls(tmp_path: Path) -> None:
    def calc(i: int) -> ToolCall:
        return ToolCall("calculate", {"expression": f"{i} + 1"}, call_id=f"k{i}")

    llm = FakeToolLLM(
        [
            [calc(1), Done(1, 1, 0)],
            [calc(2), Done(1, 1, 0)],
            [TextDelta("Done."), calc(3), Done(1, 1, 0)],
        ]
    )
    calls_tool = tool(ToolPolicy(per_turn=5), name="calculate")(calculate.fn)
    reply, _, history = _run(llm, tmp_path, calls_tool, max_rounds=2)

    assert reply == "Done."
    assert len(llm.requests) == 3
    assert [r.allow_tool_calls for r in llm.requests] == [True, True, False]
    # The call in the last request is ignored: two rounds, two notes.
    assert [n.arg for n in history[-1].tools] == ["1 + 1", "2 + 1"]


def test_cap_is_checked_before_every_later_request(tmp_path: Path) -> None:
    c1 = ToolCall("calculate", {"expression": "1 + 1"}, call_id="k1")
    c2 = ToolCall("recall", {"query": "x"}, call_id="r1")
    llm = FakeToolLLM(
        [
            [c1, Done(1, 1, 0)],
            [c2, Done(1, 1, 0)],
            [TextDelta("never"), Done(1, 1, 0)],
        ]
    )
    # 2 tokens after the first request (under 3), 4 after the second.
    reply, sent, history = _run(llm, tmp_path, calculate, _recall([]), cap=3)

    assert reply is None
    assert len(llm.requests) == 2
    assert {"type": "error", "message": MSG_CAP} in sent
    assert history == []


def test_looking_only_for_the_camera_round(tmp_path: Path) -> None:
    c1 = ToolCall("recall", {"query": "mug"}, call_id="r1")
    c2 = ToolCall("look", {"question": "Is it the same mug?"}, call_id="c1")
    llm = FakeToolLLM(
        [
            [c1, Done(1, 1, 0)],
            [c2, Done(1, 1, 0)],
            [TextDelta("Same mug."), Done(1, 1, 0)],
        ]
    )
    looker = FakeLooker()
    _, sent, _ = _run(llm, tmp_path, _recall([]), looker=looker)

    assert looker.questions == ["Is it the same mug?"]
    assert _states(sent) == ["thinking", "thinking", "looking", "thinking", "speaking", "idle"]
    assert sent.count(LOOKING) == 1


def test_cloud_model_gets_no_private_tool_and_no_private_notes(tmp_path: Path) -> None:
    earlier = [
        Message("user", "what did we say?"),
        Message(
            "assistant",
            "We said bikes.",
            tools=(ToolNote("recall", "bike", "old turn"), ToolNote("look", "q", "A mug.")),
        ),
    ]
    llm = FakeToolLLM([[TextDelta("Hi."), Done(1, 1, 0)]], cloud=True)
    _run(llm, tmp_path, _recall([]), calculate, looker=FakeLooker(), history=earlier)

    (request,) = llm.requests
    # Private tools are off with a cloud model until the user allows them.
    assert [t.name for t in request.tools] == ["calculate"]
    assert request.history[1].tools == ()
    # The kept history is unchanged; only what went to the cloud is filtered.
    assert len(earlier[1].tools) == 2


async def _ignore(msg: Any) -> None:
    return None


class _CloudFor:
    """Permissions: every tool On; the named ones allowed with a cloud model."""

    def __init__(self, *names: str) -> None:
        self.names = set(names)

    def mode(self, name: str) -> str:
        return "on"

    def cloud(self, name: str) -> bool | None:
        return True if name in self.names else None


def test_a_private_tool_the_user_allows_reaches_a_cloud_model_with_its_notes(tmp_path: Path) -> None:
    earlier = [
        Message("user", "what did we say?"),
        Message(
            "assistant",
            "We said bikes.",
            tools=(ToolNote("recall", "bike", "old turn"), ToolNote("look", "q", "A mug.")),
        ),
    ]
    llm = FakeToolLLM([[TextDelta("Hi."), Done(1, 1, 0)]], cloud=True)
    asyncio.run(
        run_text_turn(
            "Hi.",
            earlier,
            llm,
            UsageTracker(tmp_path / "usage.jsonl", 1000, clock=lambda: NOW),
            _ignore,
            tools=registry(FakeLooker(), _recall([]), calculate, permissions=_CloudFor("look")),
        )
    )
    (request,) = llm.requests
    assert [t.name for t in request.tools] == ["look", "calculate"]
    assert request.history[1].tools == (ToolNote("look", "q", "A mug."),)


def test_local_model_gets_recall_and_its_notes(tmp_path: Path) -> None:
    earlier = [
        Message("user", "what did we say?"),
        Message("assistant", "We said bikes.", tools=(ToolNote("recall", "bike", "old turn"),)),
    ]
    llm = FakeToolLLM([[TextDelta("Hi."), Done(1, 1, 0)]])
    _run(llm, tmp_path, _recall([]), calculate, history=earlier)

    (request,) = llm.requests
    assert [t.name for t in request.tools] == ["recall", "calculate"]
    assert request.history[1].tools == (ToolNote("recall", "bike", "old turn"),)


def test_a_failing_tool_source_leaves_the_turn_without_tools(tmp_path: Path) -> None:
    async def broken(info: Any) -> list[Tool]:
        raise RuntimeError("source broke")

    llm = FakeToolLLM([[TextDelta("Hi."), Done(1, 1, 0)]])
    sent: list[Any] = []

    async def send(msg: Any) -> None:
        sent.append(msg)

    reply = asyncio.run(
        run_text_turn(
            "hello",
            [],
            llm,
            UsageTracker(tmp_path / "usage.jsonl", 1000, clock=lambda: NOW),
            send,
            tools=registry(None, calculate, sources=[broken]),
        )
    )
    assert reply == "Hi."
    assert llm.requests[0].tools == []
