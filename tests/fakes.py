"""Test doubles for the LLM, STT, TTS, and look() interfaces, and a minimal
.glb builder. No network, no model, no camera."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import struct
import threading
from typing import Any, AsyncIterator

from obsilord.config import Settings
from obsilord.llm.base import (
    Done,
    LLMEvent,
    Message,
    TextDelta,
    ToolResult,
    ToolSpec,
    UsageUpdate,
)


def text_only(settings: Settings) -> Settings:
    """Settings with TTS, STT, and look() disabled, so create_app loads no
    voice, no STT model, and no vision files."""
    return dataclasses.replace(
        settings,
        tts=dataclasses.replace(settings.tts, voice=""),
        stt=dataclasses.replace(settings.stt, model=""),
        vision=dataclasses.replace(settings.vision, model=""),
    )


GLB_CHUNK_JSON = 0x4E4F534A
GLB_CHUNK_BIN = 0x004E4942


def make_glb_raw(raw: bytes, version: int = 2, chunk_type: int = GLB_CHUNK_JSON) -> bytes:
    """A binary glTF file: 12-byte header plus one chunk holding raw (padded to 4)."""
    raw += b" " * (-len(raw) % 4)
    body = struct.pack("<II", len(raw), chunk_type) + raw
    return b"glTF" + struct.pack("<II", version, 12 + len(body)) + body


def make_glb(doc: Any) -> bytes:
    """A minimal binary glTF 2.0 file whose JSON chunk is doc."""
    return make_glb_raw(json.dumps(doc).encode("utf-8"))


class FakeLLM:
    """Yields the given deltas, then Done.

    usage: if set, a UsageUpdate is yielded before the first delta.
    fail_after: raise RuntimeError before delta number fail_after
        (or after all deltas if it equals len(deltas)).
    gate: after the first delta, wait (polling) until the event is set.
    hang_s: sleep this long after the first delta (for timeout tests).
    closed / cancelled record whether the stream was closed or cancelled.
    """

    def __init__(
        self,
        deltas: list[str],
        done: Done | None = Done(10, 5, 2),
        fail_after: int | None = None,
        gate: threading.Event | None = None,
        usage: UsageUpdate | None = None,
        hang_s: float | None = None,
    ) -> None:
        self.model = "fake-model"
        self.deltas = deltas
        self.done = done
        self.fail_after = fail_after
        self.gate = gate
        self.usage = usage
        self.hang_s = hang_s
        self.calls: list[list[Message]] = []
        self.tools: list[list[ToolSpec]] = []
        self.closed = threading.Event()
        self.cancelled = threading.Event()

    async def stream_reply(
        self,
        history: list[Message],
        tools: list[ToolSpec],
        tool_results: list[ToolResult] | None = None,
        allow_tool_calls: bool = True,
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append(list(history))
        self.tools.append(list(tools))
        try:
            if self.usage is not None:
                yield self.usage
            for i, d in enumerate(self.deltas):
                if self.fail_after is not None and i == self.fail_after:
                    raise RuntimeError("boom")
                yield TextDelta(d)
                if i == 0 and self.gate is not None:
                    while not self.gate.is_set():
                        await asyncio.sleep(0.01)
                if i == 0 and self.hang_s is not None:
                    await asyncio.sleep(self.hang_s)
            if self.fail_after is not None and self.fail_after >= len(self.deltas):
                raise RuntimeError("boom")
            if self.done is not None:
                yield self.done
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        finally:
            self.closed.set()


@dataclasses.dataclass
class LLMRequest:
    history: list[Message]
    tools: list[ToolSpec]
    tool_results: list[ToolResult] | None
    allow_tool_calls: bool


class FakeToolLLM:
    """Scripted per request: rounds[i] holds the events of request i, in order.

    An Exception in a round is raised at that point instead of yielded.
    requests records what each request was called with.
    """

    def __init__(self, rounds: list[list[Any]]) -> None:
        self.model = "fake-model"
        self.rounds = rounds
        self.requests: list[LLMRequest] = []

    async def stream_reply(
        self,
        history: list[Message],
        tools: list[ToolSpec],
        tool_results: list[ToolResult] | None = None,
        allow_tool_calls: bool = True,
    ) -> AsyncIterator[LLMEvent]:
        i = len(self.requests)
        self.requests.append(LLMRequest(list(history), list(tools), tool_results, allow_tool_calls))
        for event in self.rounds[i]:
            if isinstance(event, Exception):
                raise event
            yield event


class FakeLooker:
    """Returns `response` for every look; questions records the arguments.

    delay_s: sleep before answering (lets other tasks run).
    """

    def __init__(self, response: dict[str, Any] | None = None, delay_s: float = 0.0) -> None:
        self.response = response if response is not None else {"output": "A red mug."}
        self.delay_s = delay_s
        self.questions: list[Any] = []

    async def look(self, question: Any) -> dict[str, Any]:
        self.questions.append(question)
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return dict(self.response)


class FakeTTS:
    """Yields `frames` PCM16 frames per sentence.

    Frame j of call n (1-based) is bytes([n, j]) * 2, so tests can tell frames
    apart. texts records the (cleaned) text of each call.
    fail_on: call numbers that raise RuntimeError after fail_after_frames frames.
    frame_delay_s: sleep before each frame (lets other tasks run).
    cancelled is set if a synthesis is cancelled; closed counts finished streams.
    """

    def __init__(
        self,
        frames: int = 2,
        sample_rate: int = 22050,
        fail_on: set[int] | None = None,
        fail_after_frames: int = 0,
        frame_delay_s: float = 0.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.frames = frames
        self.fail_on = fail_on or set()
        self.fail_after_frames = fail_after_frames
        self.frame_delay_s = frame_delay_s
        self.texts: list[str] = []
        self.cancelled = threading.Event()
        self.closed = 0

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.texts.append(text)
        n = len(self.texts)
        try:
            for j in range(self.frames):
                if n in self.fail_on and j == self.fail_after_frames:
                    raise RuntimeError("tts boom")
                if self.frame_delay_s:
                    await asyncio.sleep(self.frame_delay_s)
                yield bytes([n, j]) * 2
            if n in self.fail_on and self.fail_after_frames >= self.frames:
                raise RuntimeError("tts boom")
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        finally:
            self.closed += 1


class FakeSTT:
    """Returns `text` for every utterance.

    calls records (pcm16 bytes, sample_rate, language) per call.
    fail: raise RuntimeError instead of returning.
    delay_s: sleep before returning (for timeout tests).
    gate: wait (polling) until the event is set before returning.
    cancelled is set if a transcription is cancelled.
    """

    def __init__(
        self,
        text: str = "What is the capital of France?",
        fail: bool = False,
        delay_s: float = 0.0,
        gate: threading.Event | None = None,
    ) -> None:
        self.text = text
        self.fail = fail
        self.delay_s = delay_s
        self.gate = gate
        self.calls: list[tuple[bytes, int, str | None]] = []
        self.cancelled = threading.Event()

    async def transcribe(self, pcm16: bytes, sample_rate: int, language: str | None) -> str:
        self.calls.append((pcm16, sample_rate, language))
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if self.gate is not None:
                while not self.gate.is_set():
                    await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        if self.fail:
            raise RuntimeError("stt boom with secret words")
        return self.text
