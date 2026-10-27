"""Provider-neutral LLM interface (see planning/ARCHITECTURE.md, Interfaces).

Nothing in this module may import a provider SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Literal, Protocol, Union


@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant"]
    text: str


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. parameters is a JSON Schema object."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCall:
    """A tool call from the model. Comes after the last TextDelta, before Done."""

    name: str
    args: dict[str, Any] = field(default_factory=dict)
    # The provider's id for the call, echoed back with the result ("" if none).
    call_id: str = ""
    # Provider data the implementation needs to send the call back on the
    # follow-up request (Gemini: the model's parts as received, with their
    # thought signatures). Opaque to callers, passed through untouched.
    raw: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class ToolResult:
    """The answer to one ToolCall, sent with the follow-up request.

    response is a JSON object: {"output": ...} on success or {"error": ...}.
    """

    call: ToolCall
    response: dict[str, Any]


@dataclass(frozen=True)
class Done:
    """Last event of a stream. Token counts as reported by the provider."""

    tokens_in: int
    tokens_out: int
    tokens_thinking: int = 0


@dataclass(frozen=True)
class UsageUpdate:
    """Latest token counts seen mid-stream (cumulative, not increments).

    Lets the caller record usage for a stream that fails, times out, or is
    cancelled before Done arrives.
    """

    tokens_in: int
    tokens_out: int
    tokens_thinking: int = 0


LLMEvent = Union[TextDelta, ToolCall, UsageUpdate, Done]


class LLMClient(Protocol):
    # Model id, used for usage logging.
    model: str
    # Characters sent with every request besides the history (system prompt,
    # fixed instructions). Used to estimate usage when a stream reports no
    # counts. Callers read it with getattr(..., 0), so it may be omitted.
    prompt_overhead_chars: int

    # tool_results: answers to the ToolCalls of the previous request in this
    # turn; the implementation appends the model's call and these results to
    # history. allow_tool_calls=False forbids further calls (the tools stay
    # declared). One API request per call.
    def stream_reply(
        self,
        history: list[Message],
        tools: list[ToolSpec],
        tool_results: list[ToolResult] | None = None,
        allow_tool_calls: bool = True,
    ) -> AsyncIterator[LLMEvent]: ...
