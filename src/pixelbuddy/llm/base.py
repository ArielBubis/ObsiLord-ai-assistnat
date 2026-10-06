"""Provider-neutral LLM interface (see ARCHITECTURE.md, Interfaces).

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
    """Placeholder for tool declarations (look() arrives in Phase 5)."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


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

    def stream_reply(
        self, history: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]: ...
