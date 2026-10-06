"""Offline checks for GeminiClient construction. No network calls."""

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from obsilord.config import REPO_ROOT, LLMSettings, load_settings
from obsilord.llm.base import Done, Message, TextDelta, UsageUpdate
from obsilord.llm.gemini import GeminiClient, MissingApiKey, load_system_prompt


def _llm(tmp_path: Path) -> LLMSettings:
    s = load_settings(
        REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
    )
    return s.llm


def test_missing_key_raises(tmp_path: Path) -> None:
    with pytest.raises(MissingApiKey):
        GeminiClient(_llm(tmp_path), None)
    with pytest.raises(MissingApiKey):
        GeminiClient(_llm(tmp_path), "")


def test_bad_thinking_level_raises(tmp_path: Path) -> None:
    llm = replace(_llm(tmp_path), thinking_level="extreme")
    with pytest.raises(ValueError, match="thinking_level"):
        GeminiClient(llm, "dummy-key-for-test")


def test_empty_model_raises(tmp_path: Path) -> None:
    llm = replace(_llm(tmp_path), model="")
    with pytest.raises(ValueError, match="llm.model"):
        GeminiClient(llm, "dummy-key-for-test")


def test_constructs_with_dummy_key(tmp_path: Path) -> None:
    client = GeminiClient(_llm(tmp_path), "dummy-key-for-test", system_prompt="sys")
    assert client.model == "gemini-3.5-flash-lite"
    assert "dummy-key-for-test" not in repr(client)


def test_automatic_function_calling_disabled(tmp_path: Path) -> None:
    from google.genai import _extra_utils

    client = GeminiClient(_llm(tmp_path), "dummy-key-for-test", system_prompt="sys")
    afc = client._config.automatic_function_calling
    assert afc is not None and afc.disable is True
    assert _extra_utils.should_disable_afc(client._config) is True


def test_system_prompt_loaded_from_repo() -> None:
    text = load_system_prompt()
    assert text.startswith("You are Obsi-Lord")


class _FakeSdkStream:
    """Stands in for the SDK's async generator; records aclose()."""

    def __init__(self, chunks: list[Any], fail_at: int | None = None) -> None:
        self.chunks = chunks
        self.fail_at = fail_at
        self.closed = False
        self._i = 0

    def __aiter__(self) -> "_FakeSdkStream":
        return self

    async def __anext__(self) -> Any:
        if self.fail_at is not None and self._i == self.fail_at:
            raise RuntimeError("network")
        if self._i >= len(self.chunks):
            raise StopAsyncIteration
        self._i += 1
        return self.chunks[self._i - 1]

    async def aclose(self) -> None:
        self.closed = True


def _chunk(text: str, tokens: tuple[int, int, int] | None = None) -> Any:
    from google.genai import types

    usage = None
    if tokens is not None:
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=tokens[0],
            candidates_token_count=tokens[1],
            thoughts_token_count=tokens[2],
        )
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(text="thinking", thought=True), types.Part(text=text)],
                )
            )
        ],
        usage_metadata=usage,
    )


def _client_with_stream(tmp_path: Path, stream: _FakeSdkStream) -> GeminiClient:
    client = GeminiClient(_llm(tmp_path), "dummy-key-for-test", system_prompt="sys prompt")

    async def generate_content_stream(**kwargs: Any) -> _FakeSdkStream:
        return stream

    client._client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate_content_stream))
    )
    return client


def test_prompt_overhead_chars(tmp_path: Path) -> None:
    client = GeminiClient(_llm(tmp_path), "dummy-key-for-test", system_prompt="sys prompt")
    assert client.prompt_overhead_chars == len("sys prompt")


def test_stream_events_and_close(tmp_path: Path) -> None:
    stream = _FakeSdkStream([_chunk("Hi ", (5, 1, 0)), _chunk("there.", (5, 3, 2))])
    client = _client_with_stream(tmp_path, stream)

    async def go() -> list[Any]:
        return [ev async for ev in client.stream_reply([Message("user", "hello")], [])]

    events = asyncio.run(go())
    assert events == [
        UsageUpdate(5, 1, 0),
        TextDelta("Hi "),
        UsageUpdate(5, 3, 2),
        TextDelta("there."),
        Done(5, 3, 2),
    ]
    assert stream.closed


def test_stream_closed_when_consumer_stops_early(tmp_path: Path) -> None:
    stream = _FakeSdkStream([_chunk("a"), _chunk("b"), _chunk("c")])
    client = _client_with_stream(tmp_path, stream)

    async def go() -> None:
        gen = client.stream_reply([Message("user", "hello")], [])
        assert await anext(gen) == TextDelta("a")
        await gen.aclose()  # type: ignore[attr-defined]

    asyncio.run(go())
    assert stream.closed


def test_stream_closed_on_error(tmp_path: Path) -> None:
    stream = _FakeSdkStream([_chunk("a"), _chunk("b")], fail_at=1)
    client = _client_with_stream(tmp_path, stream)

    async def go() -> None:
        async for _ in client.stream_reply([Message("user", "hello")], []):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(go())
    assert stream.closed
