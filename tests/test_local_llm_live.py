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
from obsilord.memory.demo import demo_checks, seed_demo
from obsilord.memory.recall import RECALL_TOOL, RecallTool
from obsilord.memory.store import SqliteStore

pytestmark = pytest.mark.hardware


def test_local_tool_choice_and_reply_live() -> None:
    ask = [Message("user", "What am I holding?")]

    async def go() -> tuple[list[object], list[object], list[object], str]:
        server = OpenAICompatServer(load_settings().llm_local)
        try:
            model = os.environ.get("OBSILORD_LOCAL_MODEL", "") or await server.default_model()
            seconds, _ = await server.warm_up(model)
            print(f"\n{model}: warm-up {seconds:.1f} s")
            client = server.client(model)
            tools = [LOOK_TOOL.spec, SEARCH_IMAGE_TOOL.spec]
            look = [e async for e in client.stream_reply(ask, tools)]
            calls = [e for e in look if isinstance(e, ToolCall)]
            results = [ToolResult(c, {"output": "A red mug with a chipped handle."}) for c in calls]
            follow = [
                e
                async for e in client.stream_reply(ask, tools, rounds=[results], allow_tool_calls=False)
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


def test_earlier_tool_result_answers_a_follow_up_live() -> None:
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
            client = server.client(model)
            return [e async for e in client.stream_reply(history, [LOOK_TOOL.spec, SEARCH_IMAGE_TOOL.spec])]
        finally:
            await server.aclose()

    events = asyncio.run(go())
    answer = _text(events)
    print(f"\n{answer!r} calls {[e for e in events if isinstance(e, ToolCall)]}")
    assert not any(isinstance(e, ToolCall) for e in events), "it looked again instead"
    assert "blue" in answer.lower()


def test_recall_by_date_live(tmp_path: Path) -> None:
    """A question about a date (the demo conversation 13 days ago) finds it."""
    now = time.time()
    day = next(c for c in demo_checks(now) if c["kind"] == "recall-date")["question"].split("on the ")[1]
    ask = [Message("user", f"what did we talk about in {day}")]

    async def go() -> tuple[list[ToolCall], str, str]:
        store = SqliteStore(tmp_path / "demo.sqlite3")
        await store.open()
        await seed_demo(store, now)
        server = OpenAICompatServer(load_settings().llm_local)
        try:
            model = os.environ.get("OBSILORD_LOCAL_MODEL", "") or await server.default_model()
            client = server.client(model)
            recall = RecallTool(store)
            tools = [LOOK_TOOL.spec, RECALL_TOOL.spec]
            first = [e async for e in client.stream_reply(ask, tools)]
            calls = [e for e in first if isinstance(e, ToolCall)]
            results = [ToolResult(c, await recall.run(c.args)) for c in calls if c.name == "recall"]
            follow = [
                e async for e in client.stream_reply(ask, tools, rounds=[results], allow_tool_calls=False)
            ] if results else []
            return calls, _text(first), _text(follow)
        finally:
            await server.aclose()
            await store.close()

    calls, said, answer = asyncio.run(go())
    print(f"\ncalls {calls}\nfirst: {said!r}\nanswer: {answer!r}")
    assert [c.name for c in calls] == ["recall"]
    assert "pita" in answer.lower() or "cat" in answer.lower(), answer


def _sample_photo() -> bytes:
    """The live tests' sample photo at the camera's 768 px edge, as a JPEG."""
    import cv2

    from obsilord.vision.fetch import SAMPLE_DIR

    path = SAMPLE_DIR / "bee.jpg"
    if not path.is_file():
        pytest.skip("sample photo missing: python -m obsilord.vision.fetch --sample")
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    scale = 768 / max(h, w)
    ok, buf = cv2.imencode(".jpg", cv2.resize(img, (int(w * scale), int(h * scale))), [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return buf.tobytes()


def test_a_vision_model_sees_the_look_photo_live() -> None:
    """Phase 5f: the photo goes to the chat model itself, and a kept photo
    answers a follow-up."""
    from dataclasses import replace

    from obsilord.look import LOOK_PHOTO_TOOL, MSG_PHOTO_TAKEN
    from obsilord.pipeline import PHOTO_NOTE_TEXT

    photo = _sample_photo()
    ask = [Message("user", "What am I holding? One short sentence.")]

    async def go() -> tuple[bool, str, str, float]:
        server = OpenAICompatServer(load_settings().llm_local)
        try:
            model = os.environ.get("OBSILORD_LOCAL_MODEL", "") or await server.default_model()
            client = server.client(model)
            if not await client.sees_images():
                pytest.skip(f"{model} does not see photos (LM Studio type is not vlm)")
            call = ToolCall("look", {"question": "What am I holding?"}, call_id="call_1")
            result = ToolResult(call, {"output": MSG_PHOTO_TAKEN}, image=photo)
            t0 = time.perf_counter()
            follow = [
                e async for e in client.stream_reply(ask, [LOOK_PHOTO_TOOL.spec], rounds=[[result]], allow_tool_calls=False)
            ]
            ms = (time.perf_counter() - t0) * 1000
            note = ToolNote("look", "What am I holding?", PHOTO_NOTE_TEXT, image=photo)
            history = [
                ask[0],
                Message("assistant", _text(follow), tools=(note,)),
                Message("user", "What color is the flower? One word."),
            ]
            later = [e async for e in client.stream_reply(history, [LOOK_PHOTO_TOOL.spec])]
            return True, _text(follow), _text(later), ms
        finally:
            await server.aclose()

    _, answer, color, ms = asyncio.run(go())
    print(f"\n{ms:.0f} ms answer {answer!r}\nfollow-up color {color!r}")
    assert any(w in answer.lower() for w in ("bee", "flower", "cosmos")), answer
    assert any(w in color.lower() for w in ("pink", "magenta", "purple", "rose")), color


def test_recall_of_demo_history_live(tmp_path: Path) -> None:
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
            client = server.client(model)
            tools = [LOOK_TOOL.spec, RECALL_TOOL.spec]
            first = [e async for e in client.stream_reply(ask, tools)]
            calls = [e for e in first if isinstance(e, ToolCall)]
            recall = RecallTool(store)
            results = [ToolResult(c, await recall.run(c.args)) for c in calls if c.name == "recall"]
            follow = [
                e async for e in client.stream_reply(ask, tools, rounds=[results], allow_tool_calls=False)
            ] if results else []
            return first, follow
        finally:
            await server.aclose()
            await store.close()

    first, follow = asyncio.run(go())
    calls = [e for e in first if isinstance(e, ToolCall)]
    answer = _text(follow)
    print(f"\ncalls {calls}\nanswer: {answer!r}")
    assert [c.name for c in calls] == ["recall"]
    # Replies are spoken, so the time may be written out ("four thirty").
    spoken = answer.lower().replace("-", " ")
    assert any(t in spoken for t in ("16:30", "4:30", "four thirty", "sixteen thirty", "half past four")), answer
