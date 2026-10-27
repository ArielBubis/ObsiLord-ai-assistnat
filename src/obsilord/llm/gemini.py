"""Gemini implementation of LLMClient (google-genai SDK, async streaming).

This is the only module that imports google.genai. SDK details were checked
against the installed google-genai 2.28.0 source:
- await client.aio.models.generate_content_stream(model=, contents=, config=)
  returns an async iterator of GenerateContentResponse chunks.
- GenerateContentConfig(system_instruction=, max_output_tokens=, thinking_config=)
- ThinkingConfig(thinking_level=ThinkingLevel.MINIMAL | LOW | MEDIUM | HIGH)
- usage_metadata: prompt_token_count, candidates_token_count, thoughts_token_count
- Parts with part.thought set are thought summaries and are skipped.
- Without retry_options the SDK makes a single attempt (no hidden retries).
- generate_content_stream returns an async generator (stream_generator in
  models.py), so it supports aclose(); the innermost generator closes the httpx
  response in a finally (_api_client.py _aiter_response_stream).
- AutomaticFunctionCallingConfig(disable=True) makes
  _extra_utils.should_disable_afc() return True, so the stream is a single
  request with no SDK-driven tool loop.
- Tools: GenerateContentConfig(tools=[Tool(function_declarations=[
  FunctionDeclaration(name=, description=, parameters_json_schema=)])],
  tool_config=ToolConfig(function_calling_config=FunctionCallingConfig(
  mode=FunctionCallingConfigMode.NONE))). NONE: "Model will not predict any
  function calls" (types.py), with the declarations kept.
- part.function_call has name, args and an optional id. The result goes back
  as Part(function_response=FunctionResponse(id=, name=, response=)) in a
  "user" Content; per the FunctionResponse docs, response uses an "output" key
  for the result and "error" for error details. Part.from_function_response
  takes no id, so the Part is built directly.

Thought signatures (checked 2026-10-27): Gemini 3 rejects a follow-up request
with 400 INVALID_ARGUMENT when a function call of the current turn comes back
without its thought_signature (reported on discuss.ai.google.dev and in
several GitHub issues; the official thinking guide only says signatures "can
be attached to any part, such as living inside functionCall parts or the final
part of a response"). So every part of the model's response, thoughts and
empty signature-only parts included, is kept exactly as received and sent
back unchanged as the model Content on the follow-up request. ToolCall.raw
carries that list.

The API key is passed to the SDK and never logged. With [debug] log_prompts on,
each request is printed as it goes to the SDK (model, config, system
instruction, tool declarations, every part of contents) and each response as
received; thought signatures and inline data show as byte counts.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator

from google import genai
from google.genai import types

from obsilord import promptlog
from obsilord.config import REPO_ROOT, LLMSettings
from obsilord.llm.base import (
    Done,
    LLMEvent,
    Message,
    TextDelta,
    ToolCall,
    ToolResult,
    ToolSpec,
    UsageUpdate,
)

log = logging.getLogger(__name__)

SYSTEM_PROMPT_PATH: Path = REPO_ROOT / "prompts" / "system.md"
REQUEST_TIMEOUT_MS = 30_000

_THINKING_LEVELS = {
    "minimal": types.ThinkingLevel.MINIMAL,
    "low": types.ThinkingLevel.LOW,
    "medium": types.ThinkingLevel.MEDIUM,
    "high": types.ThinkingLevel.HIGH,
}


class MissingApiKey(RuntimeError):
    """Raised when no Gemini API key is configured."""


def load_system_prompt(path: Path | None = None) -> str:
    """Read the system prompt (prompts/system.md in the repo root by default)."""
    path = path if path is not None else SYSTEM_PROMPT_PATH
    return path.read_text(encoding="utf-8").strip()


def _counts(usage: types.GenerateContentResponseUsageMetadata) -> tuple[int, int, int]:
    """(prompt, candidates, thoughts) token counts, 0 where missing."""
    return (
        usage.prompt_token_count or 0,
        usage.candidates_token_count or 0,
        usage.thoughts_token_count or 0,
    )


def _to_contents(
    history: list[Message], tool_results: list[ToolResult] | None = None
) -> list[types.Content]:
    """History as Contents, then the model's tool call and the results, if any.

    All results answer calls from one model response, so they share one raw
    parts list (the response as received). Without raw (calls not made by this
    client) the call parts are rebuilt from name and args, with no signature.
    """
    contents = [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part.from_text(text=m.text)],
        )
        for m in history
    ]
    if not tool_results:
        return contents
    raw = tool_results[0].call.raw
    if raw is not None and any(r.call.raw is not raw for r in tool_results):
        raise ValueError("tool results must answer calls from one model response")
    if raw is not None:
        model_parts = list(raw)
    else:
        model_parts = [
            types.Part(
                function_call=types.FunctionCall(
                    id=r.call.call_id or None, name=r.call.name, args=r.call.args
                )
            )
            for r in tool_results
        ]
    contents.append(types.Content(role="model", parts=model_parts))
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part(
                    function_response=types.FunctionResponse(
                        id=r.call.call_id or None, name=r.call.name, response=r.response
                    )
                )
                for r in tool_results
            ],
        )
    )
    return contents


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _format_parts(role: str, parts: list[types.Part]) -> list[str]:
    """One line per part, for the prompt log. Signatures and inline data as sizes."""
    lines: list[str] = []
    for part in parts:
        tag = f"[{role} thought]" if part.thought else f"[{role}]"
        if part.thought_signature:
            tag += f" (thought signature, {len(part.thought_signature)} bytes)"
        if part.text is not None:
            lines.append(f"{tag} {part.text}")
        elif part.function_call is not None:
            fc = part.function_call
            lines.append(f"{tag} function_call {fc.name} id={fc.id} args={_json(fc.args)}")
        elif part.function_response is not None:
            fr = part.function_response
            lines.append(f"{tag} function_response {fr.name} id={fr.id} {_json(fr.response)}")
        elif part.inline_data is not None:
            data = part.inline_data.data or b""
            lines.append(f"{tag} inline_data {part.inline_data.mime_type}, {len(data)} bytes")
        else:
            other = part.model_dump(mode="json", exclude_none=True, exclude={"thought_signature"})
            lines.append(f"{tag} {_json(other)}")
    return lines


def _format_request(
    n: int, model: str, contents: list[types.Content], config: types.GenerateContentConfig
) -> str:
    """The request as passed to the SDK, for the prompt log."""
    # automatic_function_calling is SDK-side only and not sent.
    settings = config.model_dump(
        mode="json",
        exclude_none=True,
        exclude={"system_instruction", "tools", "automatic_function_calling"},
    )
    lines = [
        f"gemini #{n} request: model {model}, config {_json(settings)}",
        "--- system instruction ---",
        str(config.system_instruction),
    ]
    for tool in config.tools or []:
        for fd in getattr(tool, "function_declarations", None) or []:
            lines.append(f"--- tool {fd.name} ---")
            lines.append(f"{fd.description}")
            lines.append(f"parameters: {_json(fd.parameters_json_schema)}")
    lines.append("--- contents ---")
    for content in contents:
        lines.extend(_format_parts(content.role or "?", content.parts or []))
    return "\n".join(lines)


def _format_response(
    n: int,
    parts: list[types.Part],
    counts: tuple[int, int, int] | None,
    finish: Any,
    first_ms: float | None,
    total_ms: float,
) -> str:
    tokens_in, tokens_out, tokens_thinking = counts or (0, 0, 0)
    first = f"{first_ms:.0f} ms" if first_ms is not None else "-"
    lines = [
        f"gemini #{n} response: first chunk {first}, done {total_ms:.0f} ms, "
        f"finish {finish}, tokens in {tokens_in} out {tokens_out} thinking {tokens_thinking}"
    ]
    lines.extend(_format_parts("model", parts))
    return "\n".join(lines)


def _tool_config(
    base: types.GenerateContentConfig, tools: list[ToolSpec], allow_tool_calls: bool
) -> types.GenerateContentConfig:
    """base plus the tool declarations; mode NONE when calls are not allowed."""
    if not tools:
        return base
    declarations = [
        types.FunctionDeclaration(
            name=t.name, description=t.description, parameters_json_schema=t.parameters
        )
        for t in tools
    ]
    update: dict[str, object] = {"tools": [types.Tool(function_declarations=declarations)]}
    if not allow_tool_calls:
        update["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=types.FunctionCallingConfigMode.NONE
            )
        )
    return base.model_copy(update=update)


class GeminiClient:
    def __init__(
        self,
        llm: LLMSettings,
        api_key: str | None,
        system_prompt: str | None = None,
    ) -> None:
        if not api_key:
            raise MissingApiKey("GEMINI_API_KEY is not set in .env")
        if not llm.model:
            raise ValueError("config: llm.model is empty")
        level = llm.thinking_level.strip().lower()
        if level and level not in _THINKING_LEVELS:
            raise ValueError(
                f"config: llm.thinking_level must be one of {sorted(_THINKING_LEVELS)} or empty"
            )
        self.model = llm.model
        prompt = system_prompt if system_prompt is not None else load_system_prompt()
        # Sent with every request; used for the usage estimate on failed streams.
        self.prompt_overhead_chars = len(prompt)
        thinking = (
            types.ThinkingConfig(thinking_level=_THINKING_LEVELS[level]) if level else None
        )
        self._config = types.GenerateContentConfig(
            system_instruction=prompt,
            max_output_tokens=llm.max_output_tokens,
            thinking_config=thinking,
            # AFC would let the SDK run tools and make extra API calls that
            # bypass UsageTracker. Tool calls must come back as ToolCall events.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        self._client = genai.Client(
            api_key=api_key, http_options=types.HttpOptions(timeout=REQUEST_TIMEOUT_MS)
        )
        self._request_ids = itertools.count(1)  # numbers requests in the prompt log

    async def stream_reply(
        self,
        history: list[Message],
        tools: list[ToolSpec],
        tool_results: list[ToolResult] | None = None,
        allow_tool_calls: bool = True,
    ) -> AsyncIterator[LLMEvent]:
        counts: tuple[int, int, int] | None = None
        # Every part of the response as received, for ToolCall.raw.
        received: list[types.Part] = []
        calls: list[types.FunctionCall] = []
        contents = _to_contents(history, tool_results)
        config = _tool_config(self._config, tools, allow_tool_calls)
        n = next(self._request_ids)
        promptlog.emit(_format_request, n, self.model, contents, config)
        t0 = time.perf_counter()
        first_ms: float | None = None
        finish: Any = None
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model, contents=contents, config=config
        )
        # Close the SDK stream (and its HTTP response) as soon as we stop, also
        # on error, timeout, or cancellation, instead of waiting for GC.
        async with contextlib.aclosing(stream):
            async for chunk in stream:
                if first_ms is None:
                    first_ms = (time.perf_counter() - t0) * 1000
                if chunk.usage_metadata is not None:
                    # Cumulative counts; report them right away so a stream that
                    # fails later can still be recorded. The final chunk has totals.
                    counts = _counts(chunk.usage_metadata)
                    yield UsageUpdate(*counts)
                if not chunk.candidates:
                    continue
                finish = chunk.candidates[0].finish_reason or finish
                content = chunk.candidates[0].content
                if content is None or not content.parts:
                    continue
                received.extend(content.parts)
                for part in content.parts:
                    if part.thought:
                        continue
                    if part.text:
                        yield TextDelta(part.text)
                    elif part.function_call is not None:
                        calls.append(part.function_call)
        # After the stream, so raw holds the whole response (a signature may
        # come in a later part than the call it belongs to).
        for fc in calls:
            yield ToolCall(
                name=fc.name or "",
                args=dict(fc.args or {}),
                call_id=fc.id or "",
                raw=received,
            )
        total_ms = (time.perf_counter() - t0) * 1000
        promptlog.emit(_format_response, n, received, counts, finish, first_ms, total_ms)
        if counts is None:
            log.warning("gemini: stream ended without usage metadata")
            counts = (0, 0, 0)
        yield Done(*counts)

    async def aclose(self) -> None:
        await self._client.aio.aclose()
        self._client.close()
