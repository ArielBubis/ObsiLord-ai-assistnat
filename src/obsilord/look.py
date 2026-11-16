"""The camera tools: look() and, when [image_search] is on, search_image().

look(): one webcam frame, described by the local vision model. Gemini only
ever gets text. Looker.look(question) returns the tool response:
{"output": answer} or {"error": message}, where message is short and readable
so the model can pass it on. It never raises except CancelledError.

search_image(target): one webcam frame; the local vision model returns a box
for the target, the app crops the frame to it, the model describes the
target (from the whole frame: a small crop alone was too little for it, see
planning/PLAN.md), llama-server stops, and then only the crop is uploaded to
the image search provider (Gemini itself or a web image search). This is the only place a piece of a frame leaves the PC. Nothing is
uploaded when no box comes back, when the box covers more than
max_crop_fraction of the frame (the model boxed the whole view), or when the
crop is under MIN_CROP_EDGE_PX. The search has its own caps (SearchBudget),
checked before the camera opens and again, with the upload recorded, right
before it is sent. A search also counts as a look for the camera limiter.

Snapshots: both tools take an optional async snapshot(kind, jpeg) callback and
call it with "camera" (the frame) and "search" (the crop, right before the
upload). The app forwards them to the page only while its Snapshots switch is
on. A failing callback is logged and ignored.

LookLimiter allows one look per cooldown_s and at most max_per_hour per
rolling hour. Refused calls do not count; attempts that open the camera do,
also when they fail. The counts live in memory and reset on restart.

The camera capture and the vision session (which starts llama-server) run as
two tasks in one TaskGroup, so the model loads during the camera warm-up, and
a failure of either cancels the other: a camera that fails at once does not
wait for the model to load. Frames stay in memory. The question, the target,
the answer and the search results are never logged; the log gets the outcome
and timings.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from obsilord.camera import CameraError, crop_jpeg
from obsilord.imagesearch.base import ImageSearchClient, ImageSearchError, format_matches
from obsilord.imagesearch.budget import SearchBudget
from obsilord.llm.base import ToolSpec
from obsilord.vision.base import Box, VisionClient, VisionError

log = logging.getLogger(__name__)

HOUR_S = 3600.0
DEFAULT_QUESTION = "What is in front of the camera?"
MAX_QUESTION_CHARS = 300

MSG_ONE_LOOK = "Only one look per turn."
MSG_CAMERA_FAILED = "The camera could not take a photo."
MSG_VISION_FAILED = "The local vision model failed."
MSG_NO_ANSWER = "The local vision model gave no answer."
MSG_LOOK_FAILED = "Looking failed."
MSG_SEARCH_OFF = "Image search is turned off in the settings."
MSG_NOT_FOUND = "Qwenny could not find that in the photo, so nothing was searched."
MSG_TOO_LARGE = (
    "Qwenny could not pick that out from the rest of the view, so nothing was "
    "searched. Holding it clear of the background may help."
)
MSG_SEARCH_FAILED = "The online image lookup failed."

DEFAULT_TARGET = "the object the user is holding"
MAX_TARGET_CHARS = 100
MIN_CROP_EDGE_PX = 48


def target_question(target: str) -> str:
    """What the local model is asked about the target, as a fallback for the
    web matches."""
    return f"Describe {target} in one or two sentences. Quote any text on it."

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


SEARCH_IMAGE_TOOL = ToolSpec(
    name="search_image",
    description=(
        "Take one photo with the user's webcam and identify one thing in it with "
        "an online lookup. Qwenny cuts the target out of the photo and only "
        "that cut-out is uploaded. You get the identification and Qwenny's own "
        "description as text, not the photo. Use it instead of look() when the "
        "user asks what exactly something is and a description would not be "
        "enough: an album cover, a book, a product, a logo, a poster. It uploads "
        "part of a photo, so use it only when the user asks to identify "
        "something, never on your own, and use look() for everything else. One "
        "camera tool per turn: look() or search_image(), not both. The "
        "identification is a best guess: say what it most likely is, and say so "
        "when it is unsure or nothing was identified."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": (
                    "The one thing to cut out and search for, as a short noun "
                    "phrase, for example 'the album cover' or 'the box in my hand'."
                ),
            }
        },
        "required": ["target"],
    },
)

# kind is "camera" (the frame) or "search" (the crop about to be uploaded).
Snapshot = Callable[[str, bytes], Awaitable[None]]


class Camera(Protocol):
    async def capture_jpeg(self) -> bytes: ...


class LookTool(Protocol):
    # The tools to declare: look(), and search_image() when it is on.
    tools: list[ToolSpec]

    async def look(self, question: Any, snapshot: Snapshot | None = None) -> dict[str, Any]: ...

    async def search_image(
        self, target: Any, snapshot: Snapshot | None = None
    ) -> dict[str, Any]: ...


@dataclass(frozen=True)
class ImageSearch:
    """What search_image() needs besides the camera and the vision model."""

    client: ImageSearchClient
    budget: SearchBudget
    max_crop_fraction: float


class _NoUpload(Exception):
    """The search stopped before the upload. The message goes to the model."""


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


def clean_target(target: Any) -> str:
    """The model's target argument as a short single-line string."""
    text = " ".join(target.split()) if isinstance(target, str) else ""
    return text[:MAX_TARGET_CHARS] or DEFAULT_TARGET


def box_fraction(box: Box) -> float:
    """The share of the frame the box covers."""
    return (box[2] - box[0]) * (box[3] - box[1])


async def _emit(snapshot: Snapshot | None, kind: str, jpeg: bytes) -> None:
    if snapshot is None:
        return
    try:
        await snapshot(kind, jpeg)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - a preview must not fail the tool
        log.warning("look: snapshot not sent: %s", type(exc).__name__)


def _cause(exc: Exception) -> Exception:
    """The error to report from a TaskGroup failure: a CameraError first, then
    a VisionError, then a stopped search, else the first error."""
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
    for kind in (CameraError, VisionError, _NoUpload):
        for e in leaves:
            if isinstance(e, kind):
                return e
    return leaves[0]


class Looker:
    def __init__(
        self,
        camera: Camera,
        vision: VisionClient,
        limiter: LookLimiter,
        search: ImageSearch | None = None,
    ) -> None:
        self._camera = camera
        self._vision = vision
        self._limiter = limiter
        self._search = search
        self.tools = [LOOK_TOOL] if search is None else [LOOK_TOOL, SEARCH_IMAGE_TOOL]

    async def look(self, question: Any, snapshot: Snapshot | None = None) -> dict[str, Any]:
        refusal = self._limiter.refusal()
        if refusal is not None:
            log.info("look: refused (cooldown or hourly cap)")
            return {"error": refusal}
        # Checked and recorded with no await in between, so two looks cannot
        # both pass.
        self._limiter.record()
        t0 = time.perf_counter()
        try:
            answer = await self._look(clean_question(question), snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as group:  # noqa: BLE001 - the turn goes on with an error result
            return self._failure(group)
        ms = (time.perf_counter() - t0) * 1000
        if not answer:
            log.warning("look: empty answer after %d ms", ms)
            return {"error": MSG_NO_ANSWER}
        log.info("look: answered in %d ms", ms)
        return {"output": answer}

    @staticmethod
    def _failure(group: Exception) -> dict[str, Any]:
        exc = _cause(group)
        if isinstance(exc, CameraError):
            log.warning("look: camera failed: %s", exc)
            return {"error": MSG_CAMERA_FAILED}
        if isinstance(exc, VisionError):
            log.warning("look: vision failed: %s", exc)
            return {"error": MSG_VISION_FAILED}
        if isinstance(exc, _NoUpload):
            log.info("search_image: nothing uploaded (no usable box)")
            return {"error": str(exc)}
        log.error("look: failed unexpectedly: %s", type(exc).__name__)
        return {"error": MSG_LOOK_FAILED}

    async def _look(self, question: str, snapshot: Snapshot | None) -> str:
        # A cancelled capture only stops waiting: its thread finishes and closes
        # the camera on its own. A cancelled session stops llama-server.
        async with asyncio.TaskGroup() as tg:
            capture = tg.create_task(self._camera.capture_jpeg())
            answer = tg.create_task(self._describe(capture, question, snapshot))
        return answer.result()

    async def _describe(
        self, capture: asyncio.Task[bytes], question: str, snapshot: Snapshot | None
    ) -> str:
        async with self._vision.session() as session:
            jpeg = await capture
            await _emit(snapshot, "camera", jpeg)
            return await session.describe(jpeg, question)

    async def search_image(
        self, target: Any, snapshot: Snapshot | None = None
    ) -> dict[str, Any]:
        search = self._search
        if search is None:
            return {"error": MSG_SEARCH_OFF}
        refusal = search.budget.refusal() or self._limiter.refusal()
        if refusal is not None:
            log.info("search_image: refused (cooldown or cap)")
            return {"error": refusal}
        # Checked and recorded with no await in between, as in look().
        self._limiter.record()
        t0 = time.perf_counter()
        try:
            crop, description = await self._crop(clean_target(target), search, snapshot)
        except asyncio.CancelledError:
            raise
        except Exception as group:  # noqa: BLE001 - the turn goes on with an error result
            return self._failure(group)
        crop_ms = (time.perf_counter() - t0) * 1000
        # llama-server has stopped. The cap is checked again and the upload
        # recorded with no await before the request goes out.
        refusal = search.budget.refusal()
        if refusal is not None:
            log.info("search_image: refused (cap) after the crop")
            return {"error": refusal}
        search.budget.record(search.client.provider)
        await _emit(snapshot, "search", crop)
        t1 = time.perf_counter()
        try:
            matches = await search.client.search(crop)
        except ImageSearchError as exc:
            log.warning("search_image: search failed: %s", exc)
            return {"error": MSG_SEARCH_FAILED}
        log.info(
            "search_image: crop in %d ms, search in %d ms, %s",
            crop_ms,
            (time.perf_counter() - t1) * 1000,
            "no matches" if matches.empty else "matches",
        )
        found = format_matches(matches) or "Nothing was identified."
        local = f" Qwenny's own description: {description}" if description else ""
        return {"output": f"Online lookup of the cut-out: {found}{local}"}

    async def _crop(
        self, target: str, search: ImageSearch, snapshot: Snapshot | None
    ) -> tuple[bytes, str]:
        async with asyncio.TaskGroup() as tg:
            capture = tg.create_task(self._camera.capture_jpeg())
            result = tg.create_task(self._locate(capture, target, search, snapshot))
        return result.result()

    async def _locate(
        self,
        capture: asyncio.Task[bytes],
        target: str,
        search: ImageSearch,
        snapshot: Snapshot | None,
    ) -> tuple[bytes, str]:
        """The crop for the target and the model's description of the target.
        Raises _NoUpload when there is no usable box."""
        async with self._vision.session() as session:
            jpeg = await capture
            await _emit(snapshot, "camera", jpeg)
            box = await session.locate(jpeg, target)
            if box is None:
                raise _NoUpload(MSG_NOT_FOUND)
            if box_fraction(box) > search.max_crop_fraction:
                raise _NoUpload(MSG_TOO_LARGE)
            try:
                crop = await asyncio.to_thread(crop_jpeg, jpeg, box, MIN_CROP_EDGE_PX)
            except ValueError:
                raise _NoUpload(MSG_NOT_FOUND) from None
            try:
                description = await session.describe(jpeg, target_question(target))
            except VisionError as exc:
                # The search can still go ahead without the local description.
                log.warning("search_image: local description failed: %s", exc)
                description = ""
            return crop, description
