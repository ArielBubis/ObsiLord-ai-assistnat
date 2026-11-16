"""Provider-neutral vision interface (see planning/ARCHITECTURE.md, Interfaces).

A VisionClient answers a question about one JPEG on this PC. session() loads
the model and unloads it on exit, so an implementation can keep it loaded for
one look only. locate() finds one thing in the JPEG and returns its box, used
to crop the frame for an image search. Implementations never log the image,
the question, the target or the answer. Nothing in this module may import a
provider package.
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager
from typing import Protocol


class VisionError(RuntimeError):
    """The vision model could not start or answer. The message is safe to log."""


# x1, y1, x2, y2 as fractions (0 to 1) of the image width and height, with
# x1 < x2 and y1 < y2.
Box = tuple[float, float, float, float]


class VisionSession(Protocol):
    async def describe(self, jpeg: bytes, question: str) -> str: ...

    async def locate(self, jpeg: bytes, target: str) -> Box | None: ...


class VisionClient(Protocol):
    def session(self) -> AbstractAsyncContextManager[VisionSession]: ...
