"""Test doubles for the LLM interface. No network."""

from __future__ import annotations

import asyncio
import threading
from typing import AsyncIterator

from obsilord.llm.base import Done, LLMEvent, Message, TextDelta, ToolSpec, UsageUpdate


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
        self.closed = threading.Event()
        self.cancelled = threading.Event()

    async def stream_reply(
        self, history: list[Message], tools: list[ToolSpec]
    ) -> AsyncIterator[LLMEvent]:
        self.calls.append(list(history))
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
