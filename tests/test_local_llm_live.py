"""Live check against the local LLM server in config.toml ([llm_local]).

Needs the server running with a chat model it can load (LM Studio with
just-in-time loading, or the model loaded by hand). Pick the model with the
OBSILORD_LOCAL_MODEL environment variable, else the server's first model.
Skipped by default (hardware); run with: pytest -m hardware tests/test_local_llm_live.py
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from obsilord.config import load_settings
from obsilord.llm.base import Done, Message, TextDelta, ToolCall, ToolNote, ToolResult
from obsilord.llm.openai_compat import OpenAICompatServer
from obsilord.look import LOOK_TOOL, SEARCH_IMAGE_TOOL
from obsilord.memory.demo import seed_demo
from obsilord.memory.recall import RECALL_TOOL, RecallTool
from obsilord.memory.store import SqliteStore

pytestmark = pytest.mark.hardware


@pytest.mark.parametrize("mode", ["router", "native"])
def test_local_tool_choice_and_reply_live(mode: str) -> None:
    ask = [Message("user", "What am I holding?")]

    async def go() -> tuple[list[object], list[object], list[object], str]:
        server = OpenAICompatServer(load_settings().llm_local)
        try:
            model = os.environ.get("OBSILORD_LOCAL_MODEL", "") or await server.default_model()
            seconds, _ = await server.warm_up(model)
            print(f"\n{model} ({mode}): warm-up {seconds:.1f} s")
            client = server.client(model, tool_mode=mode)
            tools = [LOOK_TOOL, SEARCH_IMAGE_TOOL]
            look = [e async for e in client.stream_reply(ask, tools)]
            calls = [e for e in look if isinstance(e, ToolCall)]
            results = [ToolResult(c, {"output": "A red mug with a chipped handle."}) for c in calls]
            follow = [
                e
                async for e in client.stream_reply(ask, tools, tool_results=results, allow_tool_calls=False)
            ] if calls else []
            joke = [e async for e in client.stream_reply([Message("user", "Tell me a short joke.")], tools)]
            return look, follow, joke, model
        finally:
            await server.aclose()

    look, follow, joke, model = asyncio.run(go())
    calls = [e for e in look if isinstance(e, ToolCall)]
    answer = "".join(e.text for e in follow if isinstance(e, TextDelta))
    text = "".join(e.text for e in joke if isinstance(e, TextDelta))
    print(f"look: {calls}\nanswer: {answer!r}\njoke: {text!r}\ndone: {[e for e in joke if isinstance(e, Done)]}")
    assert [c.name for c in calls] == ["look"], f"{model} did not call look()"
    assert "mug" in answer.lower(), "the answer does not use the tool result"
    assert text.strip() and "<think>" not in text and "<function_calls>" not in text
    assert not any(isinstance(e, ToolCall) for e in joke)


def _text(events: list[object]) -> str:
    return "".join(e.text for e in events if isinstance(e, TextDelta))


@pytest.mark.parametrize("mode", ["router", "native"])
def test_earlier_tool_result_answers_a_follow_up_live(mode: str) -> None:
    """A follow-up about an earlier look is answered from its note in history."""
    history = [
        Message("user", "Look, what do you think of this?"),
        Message(
            "assistant",
            "A lovely little owl. It suits the desk.",
            tools=(ToolNote("look", "What is the user showing?", "A small blue ceramic owl figurine on a wooden desk."),),
        ),
        Message("user", "What color was it? Answer in one short sentence."),
    ]

    async def go() -> list[object]:
        server = OpenAICompatServer(load_settings().llm_local)
        try:
            model = os.environ.get("OBSILORD_LOCAL_MODEL", "") or await server.default_model()
            client = server.client(model, tool_mode=mode)
            return [e async for e in client.stream_reply(history, [LOOK_TOOL, SEARCH_IMAGE_TOOL])]
        finally:
            await server.aclose()

    events = asyncio.run(go())
    answer = _text(events)
    print(f"\n{mode}: {answer!r} calls {[e for e in events if isinstance(e, ToolCall)]}")
    assert not any(isinstance(e, ToolCall) for e in events), "it looked again instead"
    assert "blue" in answer.lower()


@pytest.mark.parametrize("mode", ["router", "native"])
def test_recall_of_demo_history_live(mode: str, tmp_path: Path) -> None:
    """The model calls recall() for an earlier conversation, the demo store
    answers, and the reply uses it."""
    ask = [Message("user", "When is my dentist appointment? I told you some days ago.")]

    async def go() -> tuple[list[object], list[object]]:
        store = SqliteStore(tmp_path / "demo.sqlite3")
        await store.open()
        await seed_demo(store, time.time())
        server = OpenAICompatServer(load_settings().llm_local)
        try:
            model = os.environ.get("OBSILORD_LOCAL_MODEL", "") or await server.default_model()
            client = server.client(model, tool_mode=mode)
            tools = [LOOK_TOOL, RECALL_TOOL]
            first = [e async for e in client.stream_reply(ask, tools)]
            calls = [e for e in first if isinstance(e, ToolCall)]
            recall = RecallTool(store)
            results = [ToolResult(c, await recall.run(c.args)) for c in calls if c.name == "recall"]
            follow = [
                e async for e in client.stream_reply(ask, tools, tool_results=results, allow_tool_calls=False)
            ] if results else []
            return first, follow
        finally:
            await server.aclose()
            await store.close()

    first, follow = asyncio.run(go())
    calls = [e for e in first if isinstance(e, ToolCall)]
    answer = _text(follow)
    print(f"\n{mode}: calls {calls}\nanswer: {answer!r}")
    assert [c.name for c in calls] == ["recall"]
    # Replies are spoken, so the time may be written out ("four thirty").
    spoken = answer.lower().replace("-", " ")
    assert any(t in spoken for t in ("16:30", "4:30", "four thirty", "half past four")), answer
