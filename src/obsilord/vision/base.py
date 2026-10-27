"""Provider-neutral vision interface (see planning/ARCHITECTURE.md, Interfaces).

A VisionClient answers a question about one JPEG on this PC. session() loads
the model and unloads it on exit, so an implementation can keep it loaded for
one look only. Implementations never log the image, the question or the
answer. Nothing in this module may import a provider package.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol


class VisionError(RuntimeError):
    """The vision model could not start or answer. The message is safe to log."""


class VisionSession(Protocol):
    async def describe(self, jpeg: bytes, question: str) -> str: ...


class VisionClient(Protocol):
    def session(self) -> AbstractAsyncContextManager[VisionSession]: ...
