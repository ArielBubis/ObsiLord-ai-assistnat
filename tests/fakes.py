"""Test doubles for the LLM, STT, TTS, and look() interfaces, and a minimal
.glb builder. No network, no model, no camera."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import struct
import threading
from typing import Any, AsyncIterator, Sequence

from fastapi.testclient import TestClient

from obsilord.config import Settings
from obsilord.look import camera_source
from obsilord.llm.base import (
    Done,
    LLMEvent,
    Message,
    TextDelta,
    ToolResult,
    ToolRound,
    ToolSpec,
    UsageUpdate,
)
from obsilord.tools.registry import Registry, Tool, ToolSource, static_source


def text_only(settings: Settings) -> Settings:
    """Settings with TTS, STT, and look() disabled, so create_app loads no
    voice, no STT model, and no vision files. Pages start on the "gemini"
    provider, so turns go to the fake passed as create_app(llm=...), never to a
    local server."""
    return dataclasses.replace(
        settings,
        llm=dataclasses.replace(settings.llm, provider="gemini"),
        tts=dataclasses.replace(settings.tts, model=""),
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
        cloud: bool = False,
    ) -> None:
        self.model = "fake-model"
        self.cloud = cloud
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
        rounds: Sequence[ToolRound] = (),
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
    rounds: list[list[ToolResult]]
    allow_tool_calls: bool

    @property
    def tool_results(self) -> list[ToolResult] | None:
        """The results of the last round (None before any round)."""
        return self.rounds[-1] if self.rounds else None


class FakeToolLLM:
    """Scripted per request: rounds[i] holds the events of request i, in order.

    An Exception in a round is raised at that point instead of yielded.
    requests records what each request was called with. cloud: what the
    pipeline sees as llm.cloud (a cloud model is not offered private tools).
    """

    def __init__(self, rounds: list[list[Any]], cloud: bool = False) -> None:
        self.model = "fake-model"
        self.cloud = cloud
        self.rounds = rounds
        self.requests: list[LLMRequest] = []

    async def stream_reply(
        self,
        history: list[Message],
        tools: list[ToolSpec],
        rounds: Sequence[ToolRound] = (),
        allow_tool_calls: bool = True,
    ) -> AsyncIterator[LLMEvent]:
        i = len(self.requests)
        self.requests.append(
            LLMRequest(list(history), list(tools), [list(r) for r in rounds], allow_tool_calls)
        )
        for event in self.rounds[i]:
            if isinstance(event, Exception):
                raise event
            yield event


class SeeingToolLLM(FakeToolLLM):
    """FakeToolLLM for a local model that sees photos (sees_images)."""

    def __init__(self, rounds: list[list[Any]], sees: bool | Exception = True) -> None:
        super().__init__(rounds)
        self.sees = sees

    async def sees_images(self) -> bool:
        if isinstance(self.sees, Exception):
            raise self.sees
        return self.sees


class FakeLooker:
    """Returns `response` for every look or search; questions and targets
    record the arguments. search: search_image() is on. vision: Qwenny is set
    up (without it only take_photo() is offered, to a chat model that sees).

    delay_s: sleep before answering (lets other tasks run).
    """

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        delay_s: float = 0.0,
        search: bool = False,
        vision: bool = True,
    ) -> None:
        self.vision_on = vision
        self.response = response if response is not None else {"output": "A red mug."}
        self.delay_s = delay_s
        self.questions: list[Any] = []
        self.targets: list[Any] = []
        self.search_on = search
        # Sent through the snapshot callback of each call, as (kind, jpeg).
        self.snapshots: list[tuple[str, bytes]] = []
        # take_photo(): the frame it returns and how often it ran.
        self.photo: bytes | None = b"photo-jpeg"
        self.photos_taken = 0

    async def _answer(self, snapshot: Any) -> dict[str, Any]:
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if snapshot is not None:
            for kind, jpeg in self.snapshots:
                await snapshot(kind, jpeg)
        return dict(self.response)

    async def look(self, question: Any, snapshot: Any = None) -> dict[str, Any]:
        self.questions.append(question)
        return await self._answer(snapshot)

    async def search_image(self, target: Any, snapshot: Any = None) -> dict[str, Any]:
        self.targets.append(target)
        return await self._answer(snapshot)

    async def take_photo(self, snapshot: Any = None) -> tuple[dict[str, Any], bytes | None]:
        self.photos_taken += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if snapshot is not None and self.photo is not None:
            await snapshot("camera", self.photo)
        if self.photo is None:
            return {"error": "The camera could not take a photo."}, None
        return {"output": "Photo taken."}, self.photo


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


class AllOn:
    """Permissions with every tool On; the cloud choice stays the policy's.
    Pipeline tests are about the pipeline, not the permission defaults
    (search_image asks by default; test_tool_registry covers the defaults)."""

    def mode(self, name: str) -> str:
        return "on"

    def cloud(self, name: str) -> None:
        return None


def registry(
    looker: Any = None,
    *tools: Tool,
    sources: Sequence[ToolSource] = (),
    permissions: Any = None,
    on_run: Any = None,
) -> Registry:
    """A registry with the camera tools of looker (if given), the given tools
    and sources; every tool On unless permissions are given."""
    srcs: list[ToolSource] = []
    if looker is not None:
        srcs.append(camera_source(looker))
    if tools:
        srcs.append(static_source(*tools))
    srcs.extend(sources)
    return Registry(srcs, permissions=permissions if permissions is not None else AllOn(), on_run=on_run)


class _SettledSession:
    """A /ws session entered past the two messages every page gets on
    connect (permissions, then photos), which are checked, not returned."""

    def __init__(self, session: Any) -> None:
        self._session = session

    def __enter__(self) -> Any:
        ws = self._session.__enter__()
        first, second = ws.receive_json(), ws.receive_json()
        assert first.get("type") == "permissions", first
        assert second.get("type") == "photos", second
        return ws

    def __exit__(self, *exc: Any) -> Any:
        return self._session.__exit__(*exc)


class AppClient(TestClient):
    """TestClient for tests about other things than the settings: /ws opens
    past the permissions and photos messages (test_app_permissions covers
    them)."""

    def websocket_connect(self, url: str, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
        return _SettledSession(super().websocket_connect(url, *args, **kwargs))
