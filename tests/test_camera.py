"""CameraService with a fake capture device, and the JPEG helpers. No webcam."""

import asyncio
import threading
from typing import Any

import cv2
import numpy
import pytest

from obsilord.camera import CameraError, CameraService, encode_frame, shrink_jpeg
from obsilord.config import CameraSettings


def _settings(**kw: Any) -> CameraSettings:
    values: dict[str, Any] = dict(
        device_index=0,
        max_edge_px=768,
        warmup_frames=3,
        cooldown_s=10.0,
        max_per_hour=30,
        api="dshow",
        timeout_s=2.0,
    )
    values.update(kw)
    return CameraSettings(**values)


def _frame(w: int = 640, h: int = 480) -> Any:
    return numpy.full((h, w, 3), 128, numpy.uint8)


def _size(jpeg: bytes) -> tuple[int, int]:
    img = cv2.imdecode(numpy.frombuffer(jpeg, numpy.uint8), cv2.IMREAD_COLOR)
    return img.shape[1], img.shape[0]


class FakeCapture:
    def __init__(
        self,
        opened: bool = True,
        read_ok: bool = True,
        frame: Any = None,
        block: threading.Event | None = None,
    ) -> None:
        self.opened = opened
        self.read_ok = read_ok
        self.frame = frame if frame is not None else _frame()
        self.block = block
        self.reads = 0
        self.released = threading.Event()

    def isOpened(self) -> bool:  # noqa: N802 - cv2.VideoCapture API
        if self.block is not None:
            self.block.wait(10)
        return self.opened

    def read(self) -> tuple[bool, Any]:
        self.reads += 1
        return (self.read_ok, self.frame if self.read_ok else None)

    def release(self) -> None:
        self.released.set()


def _service(cap: FakeCapture, **kw: Any) -> tuple[CameraService, list[tuple[int, str]]]:
    opened: list[tuple[int, str]] = []

    def open_capture(index: int, api: str) -> FakeCapture:
        opened.append((index, api))
        return cap

    return CameraService(_settings(**kw), open_capture=open_capture), opened


def test_encode_frame_shrinks_to_max_edge() -> None:
    assert _size(encode_frame(_frame(1280, 720), 768)) == (768, 432)
    assert _size(encode_frame(_frame(640, 480), 768)) == (640, 480)


def test_shrink_jpeg() -> None:
    small = encode_frame(_frame(640, 480), 1000)
    assert shrink_jpeg(small, 640) is small
    assert _size(shrink_jpeg(small, 512)) == (512, 384)
    with pytest.raises(ValueError):
        shrink_jpeg(b"not a jpeg", 512)


def test_capture_success() -> None:
    cap = FakeCapture(frame=_frame(1280, 720))
    service, opened = _service(cap, device_index=1, api="dshow")
    jpeg = asyncio.run(service.capture_jpeg())
    assert _size(jpeg) == (768, 432)
    assert opened == [(1, "dshow")]
    assert cap.reads == 3 + 1  # warm-up frames, then the one kept
    assert cap.released.is_set()


def test_not_opened() -> None:
    cap = FakeCapture(opened=False)
    service, _ = _service(cap)
    with pytest.raises(CameraError, match="did not open"):
        asyncio.run(service.capture_jpeg())
    assert cap.released.is_set()


def test_read_failure() -> None:
    cap = FakeCapture(read_ok=False)
    service, _ = _service(cap)
    with pytest.raises(CameraError, match="read failed"):
        asyncio.run(service.capture_jpeg())
    assert cap.released.is_set()


def test_hung_camera_times_out_and_stays_busy_until_it_returns() -> None:
    block = threading.Event()
    cap = FakeCapture(block=block)
    service, _ = _service(cap, timeout_s=0.2)

    async def go() -> None:
        with pytest.raises(CameraError, match="did not respond within 0.2 s"):
            await service.capture_jpeg()
        # The stuck thread still holds the camera: fail fast, no second thread.
        with pytest.raises(CameraError, match="busy"):
            await service.capture_jpeg()

    asyncio.run(go())
    block.set()
    assert cap.released.wait(5)  # the thread finishes and closes the camera


def test_msmf_sets_the_slow_open_workaround(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv first so monkeypatch restores the original state afterwards.
    monkeypatch.setenv("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "x")
    monkeypatch.delenv("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS")
    CameraService(_settings(api="dshow"))
    import os

    assert "OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS" not in os.environ
    CameraService(_settings(api="msmf"))
    assert os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] == "0"
