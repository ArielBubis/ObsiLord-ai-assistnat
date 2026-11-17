"""Live check against the local LLM server in config.toml ([llm_local]).

Needs the server running with a chat model it can load (LM Studio with
just-in-time loading, or the model loaded by hand). Pick the model with the
OBSILORD_LOCAL_MODEL environment variable, else the server's first model.
Skipped by default (hardware); run with: pytest -m hardware tests/test_local_llm_live.py
"""

from __future__ import annotations

import asyncio
import os

import pytest

from obsilord.config import load_settings
from obsilord.llm.base import Done, Message, TextDelta, ToolCall
from obsilord.llm.openai_compat import OpenAICompatServer
from obsilord.look import LOOK_TOOL, SEARCH_IMAGE_TOOL

pytestmark = pytest.mark.hardware


def test_local_router_and_reply_live() -> None:
    async def go() -> tuple[list[object], list[object], str]:
        server = OpenAICompatServer(load_settings().llm_local)
        try:
            model = os.environ.get("OBSILORD_LOCAL_MODEL", "") or await server.default_model()
            seconds, _ = await server.warm_up(model)
            print(f"\n{model}: warm-up {seconds:.1f} s")
            client = server.client(model)
            tools = [LOOK_TOOL, SEARCH_IMAGE_TOOL]
            look = [e async for e in client.stream_reply([Message("user", "What am I holding?")], tools)]
            joke = [e async for e in client.stream_reply([Message("user", "Tell me a short joke.")], tools)]
            return look, joke, model
        finally:
            await server.aclose()

    look, joke, model = asyncio.run(go())
    calls = [e for e in look if isinstance(e, ToolCall)]
    text = "".join(e.text for e in joke if isinstance(e, TextDelta))
    print(f"look route: {calls}\njoke: {text!r}\ndone: {[e for e in joke if isinstance(e, Done)]}")
    assert [c.name for c in calls] == ["look"], f"{model} did not route to look()"
    assert text.strip() and "<think>" not in text
    assert not any(isinstance(e, ToolCall) for e in joke)
