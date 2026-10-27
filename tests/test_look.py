"""look(): LookLimiter and Looker with a fake camera and a fake vision client."""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import pytest

from obsilord.camera import CameraError
from obsilord.look import (
    DEFAULT_QUESTION,
    MAX_QUESTION_CHARS,
    MSG_CAMERA_FAILED,
    MSG_NO_ANSWER,
    MSG_VISION_FAILED,
    LookLimiter,
    Looker,
    _cause,
    clean_question,
    msg_cooldown,
    msg_hourly,
)
from obsilord.vision.base import VisionError


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_limiter_cooldown() -> None:
    clock = Clock()
    lim = LookLimiter(cooldown_s=10, max_per_hour=30, clock=clock)
    assert lim.refusal() is None
    lim.record()
    clock.t += 3.2
    assert lim.refusal() == "Camera cooldown: try again in 7 s."
    clock.t += 6.8
    assert lim.refusal() is None


def test_limiter_hourly_cap_is_rolling() -> None:
    clock = Clock()
    lim = LookLimiter(cooldown_s=0, max_per_hour=2, clock=clock)
    lim.record()
    clock.t += 60
    lim.record()
    clock.t += 60
    assert lim.refusal() == "Camera limit reached (2 looks per hour). Try again in 58 min."
    clock.t = 1000.0 + 3600  # the first look is an hour old
    assert lim.refusal() is None


def test_refused_calls_do_not_count() -> None:
    clock = Clock()
    lim = LookLimiter(cooldown_s=10, max_per_hour=30, clock=clock)
    lim.record()
    for _ in range(5):
        clock.t += 1
        assert lim.refusal() is not None
    clock.t = 1000.0 + 10
    assert lim.refusal() is None


def test_messages_never_say_zero() -> None:
    assert msg_cooldown(0.01) == "Camera cooldown: try again in 1 s."
    assert msg_hourly(5, 1.0).endswith("Try again in 1 min.")


def test_clean_question() -> None:
    assert clean_question("  What   am\nI holding? ") == "What am I holding?"
    assert clean_question("") == DEFAULT_QUESTION
    assert clean_question(None) == DEFAULT_QUESTION
    assert clean_question(42) == DEFAULT_QUESTION
    assert len(clean_question("x" * 1000)) == MAX_QUESTION_CHARS


class FakeCamera:
    def __init__(self, fail: bool = False, wait_for: asyncio.Event | None = None) -> None:
        self.fail = fail
        self.wait_for = wait_for
        self.calls = 0
        self.started = asyncio.Event()
        self.cancelled = False

    async def capture_jpeg(self) -> bytes:
        self.calls += 1
        self.started.set()
        try:
            if self.wait_for is not None:
                await self.wait_for.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.fail:
            raise CameraError("camera 0 did not open (dshow)")
        return b"jpeg-bytes"


class FakeSession:
    def __init__(self, vision: "FakeVision") -> None:
        self.vision = vision

    async def describe(self, jpeg: bytes, question: str) -> str:
        self.vision.described.append((jpeg, question))
        if self.vision.describe_error:
            raise VisionError("llama-server request failed: URLError")
        return self.vision.answer


class FakeVision:
    def __init__(
        self,
        answer: str = "The user holds a red mug.",
        start_error: bool = False,
        describe_error: bool = False,
        while_loading: Any = None,
    ) -> None:
        self.answer = answer
        self.start_error = start_error
        self.describe_error = describe_error
        self.while_loading = while_loading
        self.described: list[tuple[bytes, str]] = []
        self.entered = 0
        self.exited = 0

    @asynccontextmanager
    async def session(self) -> AsyncIterator[FakeSession]:
        self.entered += 1
        if self.while_loading is not None:
            await self.while_loading()
        if self.start_error:
            raise VisionError("llama-server did not start (vulkan: exited with code 1)")
        try:
            yield FakeSession(self)
        finally:
            self.exited += 1


def _looker(camera: Any, vision: Any, cooldown_s: float = 10) -> Looker:
    return Looker(camera, vision, LookLimiter(cooldown_s, 30))


def test_look_returns_the_answer_as_output() -> None:
    camera, vision = FakeCamera(), FakeVision()
    result = asyncio.run(_looker(camera, vision).look("What am I holding?"))
    assert result == {"output": "The user holds a red mug."}
    assert vision.described == [(b"jpeg-bytes", "What am I holding?")]
    assert vision.exited == 1


def test_camera_and_model_load_run_at_the_same_time() -> None:
    async def go() -> dict[str, Any]:
        loading = asyncio.Event()
        camera = FakeCamera(wait_for=loading)

        async def while_loading() -> None:
            # The model "loads" until the camera has started; the camera waits
            # until the model is loading. Run one after the other, this deadlocks.
            loading.set()
            await camera.started.wait()

        looker = _looker(camera, FakeVision(while_loading=while_loading))
        return await asyncio.wait_for(looker.look("q"), 2)

    assert asyncio.run(go()) == {"output": "The user holds a red mug."}


def test_second_look_inside_cooldown_is_refused() -> None:
    camera = FakeCamera()
    looker = _looker(camera, FakeVision())

    async def go() -> tuple[dict[str, Any], dict[str, Any]]:
        return await looker.look("one"), await looker.look("two")

    first, second = asyncio.run(go())
    assert "output" in first
    assert second == {"error": "Camera cooldown: try again in 10 s."}
    assert camera.calls == 1


def test_camera_failure() -> None:
    vision = FakeVision()
    result = asyncio.run(_looker(FakeCamera(fail=True), vision).look("q"))
    assert result == {"error": MSG_CAMERA_FAILED}
    assert vision.exited == vision.entered  # any server started is stopped


def test_camera_failure_does_not_wait_for_the_model() -> None:
    async def go() -> tuple[dict[str, Any], FakeVision]:
        loading = asyncio.Event()
        camera = FakeCamera(fail=True, wait_for=loading)

        async def while_loading() -> None:
            loading.set()  # the camera fails once the model is loading
            await asyncio.sleep(60)  # a load that would take a minute

        vision = FakeVision(while_loading=while_loading)
        return await asyncio.wait_for(_looker(camera, vision).look("q"), 2), vision

    result, vision = asyncio.run(go())
    assert result == {"error": MSG_CAMERA_FAILED}
    assert vision.entered == 1


def test_camera_error_is_reported_before_a_vision_error() -> None:
    group = ExceptionGroup(
        "look", [VisionError("v"), ExceptionGroup("inner", [CameraError("c")])]
    )
    assert isinstance(_cause(group), CameraError)
    assert isinstance(_cause(ExceptionGroup("look", [ValueError("x")])), ValueError)


def test_vision_start_failure_cancels_the_capture() -> None:
    async def go() -> tuple[dict[str, Any], FakeCamera]:
        camera = FakeCamera(wait_for=asyncio.Event())  # never delivers on its own
        # The server fails while the capture is running.
        vision = FakeVision(start_error=True, while_loading=camera.started.wait)
        return await _looker(camera, vision).look("q"), camera

    result, camera = asyncio.run(go())
    assert result == {"error": MSG_VISION_FAILED}
    assert camera.cancelled


def test_describe_failure() -> None:
    vision = FakeVision(describe_error=True)
    result = asyncio.run(_looker(FakeCamera(), vision).look("q"))
    assert result == {"error": MSG_VISION_FAILED}
    assert vision.exited == 1


def test_empty_answer() -> None:
    result = asyncio.run(_looker(FakeCamera(), FakeVision(answer="")).look("q"))
    assert result == {"error": MSG_NO_ANSWER}


def test_failed_look_still_counts_toward_the_cooldown() -> None:
    camera = FakeCamera(fail=True)
    looker = _looker(camera, FakeVision())

    async def go() -> dict[str, Any]:
        await looker.look("q")
        return await looker.look("q")

    assert asyncio.run(go())["error"].startswith("Camera cooldown")
    assert camera.calls == 1


def test_question_and_answer_are_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    vision = FakeVision(answer="SECRET-ANSWER on the label")
    asyncio.run(_looker(FakeCamera(), vision).look("SECRET-QUESTION about the label"))
    assert "SECRET" not in caplog.text
    assert "look: answered" in caplog.text
