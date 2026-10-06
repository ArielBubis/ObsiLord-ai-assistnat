"""One conversation turn. Phase 2: typed text in, streamed text out.

Message contract with the browser (see ARCHITECTURE.md, WebSocket protocol):
  state thinking -> [cap check] -> state speaking on the first text delta ->
  transcript (assistant, cumulative, final false) per delta ->
  transcript (assistant, full text, final true) -> state idle.
Errors send {"type": "error"} and then state idle.

Usage accounting: once a request has been started, usage is always recorded,
in a finally block, so failed, timed-out, and cancelled streams count toward
the cap:
- Done arrived: its counts.
- Otherwise the last UsageUpdate seen, flagged partial.
- Otherwise a conservative estimate, flagged estimated: (history characters +
  the client's prompt_overhead_chars, i.e. the system prompt) / 4 for input and
  max_output_tokens for output.

History is only updated after a successful, non-empty reply, so failed turns
leave it unchanged.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from pixelbuddy.llm.base import Done, LLMClient, LLMEvent, Message, TextDelta, UsageUpdate
from pixelbuddy.sentences import SentenceSplitter
from pixelbuddy.usage import CapExceeded, UsageTracker

log = logging.getLogger(__name__)

Send = Callable[[dict[str, Any]], Awaitable[None]]

MSG_CAP = "Daily token cap reached. Try again tomorrow."
MSG_LLM_FAILED = "LLM request failed."
MSG_TIMEOUT = "Reply timed out."
MSG_UNEXPECTED = "Something went wrong."


def trim_history(history: list[Message], history_turns: int) -> None:
    """Keep the last history_turns turns (one turn = user + assistant) in place."""
    keep = max(0, history_turns) * 2
    if len(history) > keep:
        del history[: len(history) - keep]


def estimate_tokens(messages: list[Message], overhead_chars: int = 0) -> int:
    """Rough input token estimate: characters / 4, rounded up, at least 1."""
    chars = sum(len(m.text) for m in messages) + max(0, overhead_chars)
    return max(1, math.ceil(chars / 4))


async def _close_stream(stream: AsyncIterator[LLMEvent]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the result
            log.debug("closing LLM stream failed: %s", type(exc).__name__)


async def _send_quietly(send: Send, msg: dict[str, Any]) -> None:
    """Best-effort send for error paths (the socket may already be gone)."""
    try:
        await send(msg)
    except Exception as exc:  # noqa: BLE001
        log.debug("send failed: %s", type(exc).__name__)


@dataclass
class _TurnState:
    parts: list[str] = field(default_factory=list)
    done: Done | None = None
    last_usage: UsageUpdate | None = None
    ttft_ms: int | None = None


def _record_usage(
    usage: UsageTracker,
    llm: LLMClient,
    st: _TurnState,
    messages: list[Message],
    max_output_tokens: int,
    total_ms: int,
) -> None:
    model = llm.model
    try:
        if st.done is not None:
            d = st.done
            usage.record(model, d.tokens_in, d.tokens_out, d.tokens_thinking, total_ms)
        elif st.last_usage is not None:
            u = st.last_usage
            usage.record(
                model, u.tokens_in, u.tokens_out, u.tokens_thinking, total_ms, partial=True
            )
            log.warning("usage: stream incomplete, recorded last reported counts")
        else:
            usage.record(
                model,
                estimate_tokens(messages, getattr(llm, "prompt_overhead_chars", 0)),
                max_output_tokens,
                0,
                total_ms,
                partial=True,
                estimated=True,
            )
            log.warning("usage: no counts reported, recorded a conservative estimate")
    except Exception as exc:  # noqa: BLE001 - never let accounting break the turn
        log.error("usage: recording failed: %s", type(exc).__name__)


async def run_text_turn(
    text: str,
    history: list[Message],
    llm: LLMClient,
    usage: UsageTracker,
    send: Send,
    history_turns: int = 10,
    max_output_tokens: int = 200,
    turn_timeout_s: float = 60.0,
) -> str | None:
    """Run one turn. Returns the reply text, or None if the turn failed.

    Never raises except CancelledError. Unexpected errors are logged (type name
    only) and reported to the client as a generic error, best effort.
    """
    try:
        return await _run_turn(
            text, history, llm, usage, send, history_turns, max_output_tokens, turn_timeout_s
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.error("turn failed unexpectedly: %s", type(exc).__name__)
        await _send_quietly(send, {"type": "error", "message": MSG_UNEXPECTED})
        await _send_quietly(send, {"type": "state", "state": "idle"})
        return None


async def _run_turn(
    text: str,
    history: list[Message],
    llm: LLMClient,
    usage: UsageTracker,
    send: Send,
    history_turns: int,
    max_output_tokens: int,
    turn_timeout_s: float,
) -> str | None:
    await send({"type": "state", "state": "thinking"})
    try:
        usage.check()
    except CapExceeded:
        log.warning("turn blocked: daily token cap reached (%d tokens today)", usage.today_total)
        await send({"type": "error", "message": MSG_CAP})
        await send({"type": "state", "state": "idle"})
        return None

    messages = [*history, Message(role="user", text=text)]
    splitter = SentenceSplitter()
    sentence_count = 0
    st = _TurnState()
    failed: Exception | None = None
    timed_out = False
    started = False
    t0 = time.monotonic()

    try:
        async with asyncio.timeout(turn_timeout_s):
            stream = llm.stream_reply(messages, [])
            started = True
            try:
                while True:
                    try:
                        event = await anext(stream)
                    except StopAsyncIteration:
                        break
                    except Exception as exc:  # noqa: BLE001 - any provider error ends the turn
                        failed = exc
                        break
                    if isinstance(event, TextDelta):
                        if not event.text:
                            continue
                        if st.ttft_ms is None:
                            st.ttft_ms = int((time.monotonic() - t0) * 1000)
                            await send({"type": "state", "state": "speaking"})
                        st.parts.append(event.text)
                        await send(
                            {
                                "type": "transcript",
                                "role": "assistant",
                                "text": "".join(st.parts),
                                "final": False,
                            }
                        )
                        sentence_count += len(splitter.feed(event.text))
                    elif isinstance(event, UsageUpdate):
                        st.last_usage = event
                    elif isinstance(event, Done):
                        st.done = event
                    # ToolCall is ignored until Phase 5 (no tools are passed).
            finally:
                await _close_stream(stream)
    except TimeoutError:
        timed_out = True
    finally:
        # Runs on success, failure, timeout, and cancellation alike.
        total_ms = int((time.monotonic() - t0) * 1000)
        if started:
            _record_usage(usage, llm, st, messages, max_output_tokens, total_ms)

    if timed_out:
        log.error("LLM turn timed out after %d ms", total_ms)
        await send({"type": "error", "message": MSG_TIMEOUT})
        await send({"type": "state", "state": "idle"})
        return None

    if failed is not None:
        log.error("LLM request failed: %s (after %d ms)", type(failed).__name__, total_ms)
        await send({"type": "error", "message": MSG_LLM_FAILED})
        await send({"type": "state", "state": "idle"})
        return None

    if splitter.flush():
        sentence_count += 1
    reply = "".join(st.parts)
    d = st.done
    log.debug("turn: %d sentences", sentence_count)
    log.info(
        "turn: ttft_ms=%s total_ms=%d tokens_in=%s tokens_out=%s tokens_thinking=%s",
        st.ttft_ms if st.ttft_ms is not None else "-",
        total_ms,
        d.tokens_in if d else "-",
        d.tokens_out if d else "-",
        d.tokens_thinking if d else "-",
    )

    await send({"type": "transcript", "role": "assistant", "text": reply, "final": True})
    await send({"type": "state", "state": "idle"})

    if reply.strip():
        history.append(Message(role="user", text=text))
        history.append(Message(role="assistant", text=reply))
        trim_history(history, history_turns)
    return reply
