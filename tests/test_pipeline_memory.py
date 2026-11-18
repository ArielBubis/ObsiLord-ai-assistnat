"""Pipeline with memory: tool notes kept in history, non-camera tools, and the
per-turn callback that writes the conversation store. No network, no model."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fakes import FakeLooker, FakeToolLLM
from obsilord.llm.base import Done, Message, TextDelta, ToolCall, ToolNote, ToolSpec, note_from_result
from obsilord.llm.base import ToolResult
from obsilord.pipeline import MSG_ONE_CALL, run_text_turn
from obsilord.tools import Tool
from obsilord.usage import UsageTracker

NOW = datetime(2026, 11, 18, 12, 0, tzinfo=timezone.utc)
LOOKING = {"type": "state", "state": "looking"}

RECALL_SPEC = ToolSpec(
    "recall",
    "Search earlier conversations.",
    {"type": "object", "properties": {"query": {"type": "string"}}},
)


class FakeRecall:
    def __init__(self, response: dict[str, Any] | None = None, fail: bool = False) -> None:
        self.args: list[dict[str, Any]] = []
        self.response = response if response is not None else {"output": "Yesterday: shakshuka."}
        self.fail = fail

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        self.args.append(args)
        if self.fail:
            raise RuntimeError("store broke")
        return dict(self.response)

    def tool(self) -> Tool:
        return Tool(RECALL_SPEC, self.run)


def _usage(tmp_path: Path) -> UsageTracker:
    return UsageTracker(tmp_path / "usage.jsonl", 1000, clock=lambda: NOW)


def _run(
    llm: Any,
    tmp_path: Path,
    looker: Any = None,
    extra_tools: list[Tool] | None = None,
    history: list[Message] | None = None,
    on_turn: Any = None,
) -> tuple[str | None, list[Any]]:
    sent: list[Any] = []

    async def send(msg: Any) -> None:
        sent.append(msg)

    reply = asyncio.run(
        run_text_turn(
            "hello",
            history if history is not None else [],
            llm,
            _usage(tmp_path),
            send,
            looker=looker,
            extra_tools=extra_tools or [],
            on_turn=on_turn,
        )
    )
    return reply, sent


def _recall_call(query: str = "shakshuka", call_id: str = "r1") -> ToolCall:
    return ToolCall("recall", {"query": query}, call_id=call_id)


def test_note_from_result_is_one_capped_line() -> None:
    call = ToolCall("look", {"question": "What  is\nthis?"})
    note = note_from_result(ToolResult(call, {"output": "A red\n\nmug. " + "x" * 600}))
    assert note.name == "look"
    assert note.arg == "What is this?"
    assert note.ok
    assert "\n" not in note.result
    assert note.result.startswith("A red mug.")
    assert len(note.result) == 400
    err = note_from_result(ToolResult(call, {"error": "Camera cooldown."}))
    assert not err.ok and err.result == "Camera cooldown."


def test_tool_results_are_kept_in_history_as_notes(tmp_path: Path) -> None:
    llm = FakeToolLLM(
        [
            [ToolCall("search_image", {"target": "the figure"}, call_id="c1"), Done(1, 1)],
            [TextDelta("It is a Funko Pop."), Done(1, 1)],
        ]
    )
    looker = FakeLooker({"output": "Online lookup of the cut-out: Funko Pop Miku #39"}, search=True)
    history: list[Message] = []
    _run(llm, tmp_path, looker=looker, history=history)
    assert history == [
        Message("user", "hello"),
        Message(
            "assistant",
            "It is a Funko Pop.",
            tools=(ToolNote("search_image", "the figure", "Online lookup of the cut-out: Funko Pop Miku #39"),),
        ),
    ]


def test_extra_tool_runs_without_looking_state(tmp_path: Path) -> None:
    recall = FakeRecall()
    llm = FakeToolLLM(
        [[_recall_call(), Done(1, 1)], [TextDelta("We planned shakshuka."), Done(1, 1)]]
    )
    reply, sent = _run(llm, tmp_path, extra_tools=[recall.tool()])
    assert reply == "We planned shakshuka."
    assert recall.args == [{"query": "shakshuka"}]
    assert LOOKING not in sent
    assert [t.name for t in llm.requests[0].tools] == ["recall"]
    results = llm.requests[1].tool_results
    assert results is not None and results[0].response == {"output": "Yesterday: shakshuka."}


def test_extra_tool_is_declared_after_the_camera_tools(tmp_path: Path) -> None:
    llm = FakeToolLLM([[TextDelta("Hi."), Done(1, 1)]])
    _run(llm, tmp_path, looker=FakeLooker(), extra_tools=[FakeRecall().tool()])
    assert [t.name for t in llm.requests[0].tools] == ["look", "recall"]


def test_second_call_to_the_same_extra_tool_gets_an_error(tmp_path: Path) -> None:
    recall = FakeRecall()
    llm = FakeToolLLM(
        [
            [_recall_call("a", "r1"), _recall_call("b", "r2"), Done(1, 1)],
            [TextDelta("Done."), Done(1, 1)],
        ]
    )
    _run(llm, tmp_path, extra_tools=[recall.tool()])
    assert recall.args == [{"query": "a"}]
    results = llm.requests[1].tool_results
    assert results is not None
    assert results[1].response == {"error": MSG_ONE_CALL}


def test_look_and_recall_in_one_turn_both_run(tmp_path: Path) -> None:
    recall = FakeRecall()
    looker = FakeLooker({"output": "A blue owl."})
    llm = FakeToolLLM(
        [
            [ToolCall("look", {"question": "q"}, call_id="c1"), _recall_call(), Done(1, 1)],
            [TextDelta("Both."), Done(1, 1)],
        ]
    )
    _, sent = _run(llm, tmp_path, looker=looker, extra_tools=[recall.tool()])
    assert looker.questions == ["q"]
    assert recall.args == [{"query": "shakshuka"}]
    assert LOOKING in sent


def test_failing_extra_tool_answers_with_an_error(tmp_path: Path) -> None:
    llm = FakeToolLLM([[_recall_call(), Done(1, 1)], [TextDelta("Sorry."), Done(1, 1)]])
    reply, _ = _run(llm, tmp_path, extra_tools=[FakeRecall(fail=True).tool()])
    assert reply == "Sorry."
    results = llm.requests[1].tool_results
    assert results is not None and "error" in results[0].response


def test_on_turn_gets_the_finished_turn_once(tmp_path: Path) -> None:
    got: list[tuple[str, str, tuple[ToolNote, ...]]] = []

    async def on_turn(user: str, reply: str, notes: tuple[ToolNote, ...]) -> None:
        got.append((user, reply, notes))

    llm = FakeToolLLM([[_recall_call(), Done(1, 1)], [TextDelta("Shakshuka."), Done(1, 1)]])
    _run(llm, tmp_path, extra_tools=[FakeRecall().tool()], on_turn=on_turn)
    assert got == [("hello", "Shakshuka.", (ToolNote("recall", "shakshuka", "Yesterday: shakshuka."),))]


def test_on_turn_is_not_called_when_the_llm_fails(tmp_path: Path) -> None:
    got: list[Any] = []

    async def on_turn(*args: Any) -> None:
        got.append(args)

    llm = FakeToolLLM([[RuntimeError("boom")]])
    reply, _ = _run(llm, tmp_path, on_turn=on_turn)
    assert reply is None
    assert got == []


def test_on_turn_failure_does_not_break_the_turn(tmp_path: Path) -> None:
    async def on_turn(*args: Any) -> None:
        raise OSError("disk full")

    llm = FakeToolLLM([[TextDelta("Hi."), Done(1, 1)]])
    reply, sent = _run(llm, tmp_path, on_turn=on_turn)
    assert reply == "Hi."
    assert not any(isinstance(m, dict) and m.get("type") == "error" for m in sent)
    assert sent[-1] == {"type": "state", "state": "idle"}
