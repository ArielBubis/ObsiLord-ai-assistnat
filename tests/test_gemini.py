"""Offline checks for GeminiClient construction. No network calls."""

import asyncio
import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from obsilord.config import REPO_ROOT, LLMSettings, load_settings
from obsilord.llm.base import Done, Message, TextDelta, ToolCall, ToolResult, UsageUpdate
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


def _capturing_client(
    tmp_path: Path, stream: _FakeSdkStream
) -> tuple[GeminiClient, list[dict[str, Any]]]:
    client = GeminiClient(_llm(tmp_path), "dummy-key-for-test", system_prompt="sys prompt")
    seen: list[dict[str, Any]] = []

    async def generate_content_stream(**kwargs: Any) -> _FakeSdkStream:
        seen.append(kwargs)
        return stream

    client._client = SimpleNamespace(  # type: ignore[assignment]
        aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate_content_stream))
    )
    return client, seen


def _parts_chunk(parts: list[Any], tokens: tuple[int, int, int] | None = None) -> Any:
    from google.genai import types

    usage = None
    if tokens is not None:
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=tokens[0],
            candidates_token_count=tokens[1],
            thoughts_token_count=tokens[2],
        )
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=parts))],
        usage_metadata=usage,
    )


def _look_call_stream() -> _FakeSdkStream:
    from google.genai import types

    call = types.Part(
        function_call=types.FunctionCall(
            id="call-1", name="look", args={"question": "What am I holding?"}
        ),
        thought_signature=b"sig-call",
    )
    # A signature may also arrive on an empty part after the call.
    trailing = types.Part(text="", thought_signature=b"sig-final")
    return _FakeSdkStream(
        [
            _parts_chunk([types.Part(text="Let me look. ")]),
            _parts_chunk([call]),
            _parts_chunk([trailing], (7, 3, 0)),
        ]
    )


def test_tool_call_comes_after_text_with_the_raw_parts(tmp_path: Path) -> None:
    from obsilord.look import LOOK_TOOL

    client, seen = _capturing_client(tmp_path, _look_call_stream())

    async def go() -> list[Any]:
        return [ev async for ev in client.stream_reply([Message("user", "hi")], [LOOK_TOOL])]

    events = asyncio.run(go())
    text, usage, call, done = events
    assert text == TextDelta("Let me look. ")
    assert usage == UsageUpdate(7, 3, 0)
    assert isinstance(call, ToolCall)
    assert (call.name, call.args, call.call_id) == (
        "look",
        {"question": "What am I holding?"},
        "call-1",
    )
    # Every part as received, signatures included, in order.
    assert [p.thought_signature for p in call.raw] == [None, b"sig-call", b"sig-final"]
    assert done == Done(7, 3, 0)

    config = seen[0]["config"]
    declaration = config.tools[0].function_declarations[0]
    assert declaration.name == "look"
    assert declaration.parameters_json_schema == LOOK_TOOL.parameters
    assert config.tool_config is None  # calls allowed
    assert config.system_instruction == "sys prompt"


def test_follow_up_sends_the_parts_back_with_the_result(tmp_path: Path) -> None:
    from google.genai import types

    from obsilord.look import LOOK_TOOL

    raw = [types.Part(text="Let me look. "), types.Part(
        function_call=types.FunctionCall(id="call-1", name="look", args={"question": "q"}),
        thought_signature=b"sig-call",
    )]
    call = ToolCall("look", {"question": "q"}, call_id="call-1", raw=raw)
    client, seen = _capturing_client(tmp_path, _FakeSdkStream([_chunk("A red mug.", (9, 4, 0))]))

    async def go() -> list[Any]:
        return [
            ev
            async for ev in client.stream_reply(
                [Message("user", "hi")],
                [LOOK_TOOL],
                tool_results=[ToolResult(call, {"output": "A red mug."})],
                allow_tool_calls=False,
            )
        ]

    events = asyncio.run(go())
    assert TextDelta("A red mug.") in events
    contents = seen[0]["contents"]
    assert [c.role for c in contents] == ["user", "model", "user"]
    assert contents[1].parts == raw  # unchanged, signature included
    response = contents[2].parts[0].function_response
    assert (response.id, response.name, response.response) == (
        "call-1",
        "look",
        {"output": "A red mug."},
    )
    mode = seen[0]["config"].tool_config.function_calling_config.mode
    assert mode == types.FunctionCallingConfigMode.NONE
    assert seen[0]["config"].tools[0].function_declarations[0].name == "look"


def test_results_without_raw_rebuild_the_call() -> None:
    from obsilord.llm.gemini import _to_contents

    call = ToolCall("look", {"question": "q"}, call_id="")
    contents = _to_contents([Message("user", "hi")], [ToolResult(call, {"error": "busy"})])
    fc = contents[1].parts[0].function_call
    assert (fc.id, fc.name, fc.args) == (None, "look", {"question": "q"})
    assert contents[2].parts[0].function_response.response == {"error": "busy"}


def test_results_from_two_responses_are_rejected() -> None:
    from obsilord.llm.gemini import _to_contents

    a = ToolCall("look", {}, raw=["a"])
    b = ToolCall("look", {}, raw=["b"])
    with pytest.raises(ValueError, match="one model response"):
        _to_contents([], [ToolResult(a, {}), ToolResult(b, {})])


def test_no_tools_keeps_the_base_config(tmp_path: Path) -> None:
    client, seen = _capturing_client(tmp_path, _FakeSdkStream([_chunk("Hi.")]))

    async def go() -> None:
        async for _ in client.stream_reply([Message("user", "hi")], []):
            pass

    asyncio.run(go())
    assert seen[0]["config"] is client._config
    assert seen[0]["config"].tools is None


def test_stream_closed_on_error(tmp_path: Path) -> None:
    stream = _FakeSdkStream([_chunk("a"), _chunk("b")], fail_at=1)
    client = _client_with_stream(tmp_path, stream)

    async def go() -> None:
        async for _ in client.stream_reply([Message("user", "hello")], []):
            pass

    with pytest.raises(RuntimeError):
        asyncio.run(go())
    assert stream.closed


def _prompt_records(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "obsilord.prompts"]


def test_prompt_log_prints_the_request_and_the_response(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    from google.genai import types

    from obsilord.look import LOOK_TOOL

    caplog.set_level(logging.INFO, logger="obsilord.prompts")
    raw = [
        types.Part(text="Let me look. "),
        types.Part(
            function_call=types.FunctionCall(id="call-1", name="look", args={"question": "q"}),
            thought_signature=b"sig-call",
        ),
    ]
    call = ToolCall("look", {"question": "q"}, call_id="call-1", raw=raw)
    client, _ = _capturing_client(tmp_path, _FakeSdkStream([_chunk("A red mug.", (9, 4, 0))]))

    async def go() -> None:
        async for _ in client.stream_reply(
            [Message("user", "what am I holding?")],
            [LOOK_TOOL],
            tool_results=[ToolResult(call, {"output": "The user holds a red mug."})],
            allow_tool_calls=False,
        ):
            pass

    asyncio.run(go())
    request, response = _prompt_records(caplog)
    lines = request.splitlines()
    assert lines[0].startswith("gemini #1 request: model ")
    assert '"mode": "NONE"' in lines[0] and '"max_output_tokens"' in lines[0]
    assert lines[1:3] == ["--- system instruction ---", "sys prompt"]
    assert "--- tool look ---" in lines
    assert lines[-4:] == [
        "[user] what am I holding?",
        "[model] Let me look. ",
        '[model] (thought signature, 8 bytes) function_call look id=call-1 args={"question": "q"}',
        '[user] function_response look id=call-1 {"output": "The user holds a red mug."}',
    ]
    assert "sig-call" not in request
    head, *parts = response.splitlines()
    assert head.startswith("gemini #1 response: first chunk ")
    assert "tokens in 9 out 4 thinking 0" in head
    assert parts == ["[model thought] thinking", "[model] A red mug."]


def test_prompt_log_is_silent_when_off(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)
    client, _ = _capturing_client(tmp_path, _FakeSdkStream([_chunk("Hi.", (5, 1, 0))]))

    async def go() -> None:
        async for _ in client.stream_reply([Message("user", "hello")], []):
            pass

    asyncio.run(go())
    assert _prompt_records(caplog) == []
