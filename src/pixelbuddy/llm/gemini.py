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

The API key is passed to the SDK and never logged.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import AsyncIterator

from google import genai
from google.genai import types

from pixelbuddy.config import REPO_ROOT, LLMSettings
from pixelbuddy.llm.base import (
    Done,
    LLMEvent,
    Message,
    TextDelta,
    ToolCall,
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


def _to_contents(history: list[Message]) -> list[types.Content]:
    return [
        types.Content(
            role="model" if m.role == "assistant" else "user",
            parts=[types.Part.from_text(text=m.text)],
        )
        for m in history
    ]


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

    async def stream_reply(
        self, history: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        if tools:
            raise NotImplementedError("tools are not wired until Phase 5")
        counts: tuple[int, int, int] | None = None
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model, contents=_to_contents(history), config=self._config
        )
        # Close the SDK stream (and its HTTP response) as soon as we stop, also
        # on error, timeout, or cancellation, instead of waiting for GC.
        async with contextlib.aclosing(stream):
            async for chunk in stream:
                if chunk.usage_metadata is not None:
                    # Cumulative counts; report them right away so a stream that
                    # fails later can still be recorded. The final chunk has totals.
                    counts = _counts(chunk.usage_metadata)
                    yield UsageUpdate(*counts)
                if not chunk.candidates:
                    continue
                content = chunk.candidates[0].content
                if content is None or not content.parts:
                    continue
                for part in content.parts:
                    if part.thought:
                        continue
                    if part.text:
                        yield TextDelta(part.text)
                    elif part.function_call is not None:
                        yield ToolCall(
                            name=part.function_call.name or "",
                            args=dict(part.function_call.args or {}),
                        )
        if counts is None:
            log.warning("gemini: stream ended without usage metadata")
            counts = (0, 0, 0)
        yield Done(*counts)

    async def aclose(self) -> None:
        await self._client.aio.aclose()
        self._client.close()
