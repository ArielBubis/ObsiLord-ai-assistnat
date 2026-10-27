"""The look() tool: one webcam frame, described by the local vision model.

Gemini only ever gets text. Looker.look(question) returns the tool response:
{"output": answer} or {"error": message}, where message is short and readable
so the model can pass it on. It never raises except CancelledError.

LookLimiter allows one look per cooldown_s and at most max_per_hour per
rolling hour. Refused calls do not count; attempts that open the camera do,
also when they fail. The counts live in memory and reset on restart.

The camera capture and the vision session (which starts llama-server) run as
two tasks in one TaskGroup, so the model loads during the camera warm-up, and
a failure of either cancels the other: a camera that fails at once does not
wait for the model to load. Frames stay in memory. The question and the answer
are never logged; the log gets the outcome and timings.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from typing import Any, Callable, Protocol

from obsilord.camera import CameraError
from obsilord.llm.base import ToolSpec
from obsilord.vision.base import VisionClient, VisionError

log = logging.getLogger(__name__)

HOUR_S = 3600.0
DEFAULT_QUESTION = "What is in front of the camera?"
MAX_QUESTION_CHARS = 300

MSG_ONE_LOOK = "Only one look per turn."
MSG_CAMERA_FAILED = "The camera could not take a photo."
MSG_VISION_FAILED = "The local vision model failed."
MSG_NO_ANSWER = "The local vision model gave no answer."
MSG_LOOK_FAILED = "Looking failed."

LOOK_TOOL = ToolSpec(
    name="look",
    description=(
        "Take one photo with the user's webcam. Qwenny, a small vision model on "
        "this computer, answers the question about it and you get Qwenny's text "
        "answer, not the photo."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": (
                    "What to find out about the view, for example "
                    "'What am I holding?' or 'Read the label.'"
                ),
            }
        },
        "required": ["question"],
    },
)


class Camera(Protocol):
    async def capture_jpeg(self) -> bytes: ...


class LookTool(Protocol):
    async def look(self, question: Any) -> dict[str, Any]: ...


def msg_cooldown(wait_s: float) -> str:
    return f"Camera cooldown: try again in {max(1, math.ceil(wait_s))} s."


def msg_hourly(max_per_hour: int, wait_s: float) -> str:
    return (
        f"Camera limit reached ({max_per_hour} looks per hour). "
        f"Try again in {max(1, math.ceil(wait_s / 60))} min."
    )


class LookLimiter:
    def __init__(
        self,
        cooldown_s: float,
        max_per_hour: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cooldown_s = cooldown_s
        self.max_per_hour = max_per_hour
        self._clock = clock
        self._times: deque[float] = deque()

    def refusal(self) -> str | None:
        """Why a look is not allowed now, or None if it is."""
        now = self._clock()
        while self._times and now - self._times[0] >= HOUR_S:
            self._times.popleft()
        if self._times and now - self._times[-1] < self.cooldown_s:
            return msg_cooldown(self.cooldown_s - (now - self._times[-1]))
        if len(self._times) >= self.max_per_hour:
            return msg_hourly(self.max_per_hour, HOUR_S - (now - self._times[0]))
        return None

    def record(self) -> None:
        self._times.append(self._clock())


def clean_question(question: Any) -> str:
    """The model's question argument as a short single-line string."""
    text = " ".join(question.split()) if isinstance(question, str) else ""
    return text[:MAX_QUESTION_CHARS] or DEFAULT_QUESTION


def _cause(exc: Exception) -> Exception:
    """The error to report from a TaskGroup failure: a CameraError first, then
    a VisionError, else the first error."""
    if not isinstance(exc, ExceptionGroup):
        return exc
    leaves: list[Exception] = []
    stack: list[Exception] = [exc]
    while stack:
        e = stack.pop()
        if isinstance(e, ExceptionGroup):
            stack.extend(reversed(e.exceptions))
        else:
            leaves.append(e)
    for kind in (CameraError, VisionError):
        for e in leaves:
            if isinstance(e, kind):
                return e
    return leaves[0]


class Looker:
    def __init__(self, camera: Camera, vision: VisionClient, limiter: LookLimiter) -> None:
        self._camera = camera
        self._vision = vision
        self._limiter = limiter

    async def look(self, question: Any) -> dict[str, Any]:
        refusal = self._limiter.refusal()
        if refusal is not None:
            log.info("look: refused (cooldown or hourly cap)")
            return {"error": refusal}
        # Checked and recorded with no await in between, so two looks cannot
        # both pass.
        self._limiter.record()
        t0 = time.perf_counter()
        try:
            answer = await self._look(clean_question(question))
        except asyncio.CancelledError:
            raise
        except Exception as group:  # noqa: BLE001 - the turn goes on with an error result
            exc = _cause(group)
            if isinstance(exc, CameraError):
                log.warning("look: camera failed: %s", exc)
                return {"error": MSG_CAMERA_FAILED}
            if isinstance(exc, VisionError):
                log.warning("look: vision failed: %s", exc)
                return {"error": MSG_VISION_FAILED}
            log.error("look: failed unexpectedly: %s", type(exc).__name__)
            return {"error": MSG_LOOK_FAILED}
        ms = (time.perf_counter() - t0) * 1000
        if not answer:
            log.warning("look: empty answer after %d ms", ms)
            return {"error": MSG_NO_ANSWER}
        log.info("look: answered in %d ms", ms)
        return {"output": answer}

    async def _look(self, question: str) -> str:
        # A cancelled capture only stops waiting: its thread finishes and closes
        # the camera on its own. A cancelled session stops llama-server.
        async with asyncio.TaskGroup() as tg:
            capture = tg.create_task(self._camera.capture_jpeg())
            answer = tg.create_task(self._describe(capture, question))
        return answer.result()

    async def _describe(self, capture: asyncio.Task[bytes], question: str) -> str:
        async with self._vision.session() as session:
            jpeg = await capture
            return await session.describe(jpeg, question)
