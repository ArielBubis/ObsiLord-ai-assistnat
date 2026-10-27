"""CameraService: one webcam frame per look(), as a JPEG in memory (OpenCV).

opencv-python 5.0.0.93 (pinned). cv2 is imported lazily, inside functions.
Frames are never written to disk or logged; the log gets timings and the frame
size only. Checked on the dev machine 2026-10-27 (Logitech Brio 100) with a
local bench script (not in the repo):
- cv2.VideoCapture(0) with OpenCV's default Windows backend (Media
  Foundation) hung inside the call; Ctrl+C did not reach it.
- cv2.VideoCapture(0, cv2.CAP_DSHOW): open 191 to 233 ms, 5 warm-up frames
  plus one grab 517 ms, 640x480 frames by default.
- OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS=0 is the usual workaround for slow
  Media Foundation opens. It is set for api = "msmf" (before cv2 is used) but
  was not tested here.

Each capture runs in its own daemon thread, not asyncio.to_thread: a camera
call stuck in native code would hold a default executor thread, which the
interpreter joins at exit, so the app could not shut down. After a timeout the
thread is left to finish on its own; until it does, further captures fail fast
with CameraError instead of piling up threads. The camera is released in the
thread, so it closes even when the caller has stopped waiting.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from typing import Any, Callable

from obsilord.config import CameraSettings

log = logging.getLogger(__name__)

JPEG_QUALITY = 85

_CV2_APIS = {"dshow": "CAP_DSHOW", "msmf": "CAP_MSMF", "any": "CAP_ANY"}

# (device index, api name) -> an object with isOpened(), read() and release(),
# like cv2.VideoCapture. Replaced in tests.
OpenCapture = Callable[[int, str], Any]


class CameraError(RuntimeError):
    """No frame: the camera did not open, a read failed, it timed out, or it is busy.

    The message is safe to log (it never holds frame data).
    """


def encode_frame(frame: Any, max_edge_px: int, quality: int = JPEG_QUALITY) -> bytes:
    """Shrink a BGR frame so its longest edge is at most max_edge_px, then
    JPEG-encode it in memory. Raises ValueError if encoding fails."""
    import cv2

    h, w = frame.shape[:2]
    scale = max_edge_px / max(h, w)
    if scale < 1:
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def shrink_jpeg(jpeg: bytes, max_edge_px: int) -> bytes:
    """The JPEG with its longest edge at most max_edge_px (returned unchanged
    if it is already that small). Raises ValueError if it cannot be decoded."""
    import cv2
    import numpy

    frame = cv2.imdecode(numpy.frombuffer(jpeg, numpy.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("JPEG decoding failed")
    if max(frame.shape[:2]) <= max_edge_px:
        return jpeg
    return encode_frame(frame, max_edge_px)


def _open_cv2(index: int, api: str) -> Any:
    import cv2

    return cv2.VideoCapture(index, getattr(cv2, _CV2_APIS[api]))


class CameraService:
    def __init__(self, settings: CameraSettings, open_capture: OpenCapture | None = None) -> None:
        self._settings = settings
        self._open = open_capture if open_capture is not None else _open_cv2
        self._thread: threading.Thread | None = None
        if settings.api == "msmf":
            os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")

    async def capture_jpeg(self) -> bytes:
        """Open, drop the warm-up frames, grab one frame, close. Returns it as a
        JPEG with its longest edge at most max_edge_px. Raises CameraError."""
        if self._thread is not None and self._thread.is_alive():
            raise CameraError("camera busy: the previous capture has not finished")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bytes] = loop.create_future()

        def deliver(jpeg: bytes | None, error: BaseException | None) -> None:
            if future.done():  # the caller timed out or was cancelled
                return
            if error is not None:
                future.set_exception(error)
            else:
                future.set_result(jpeg or b"")

        def run() -> None:
            try:
                result: tuple[bytes | None, BaseException | None] = (self._grab(), None)
            except CameraError as exc:
                result = (None, exc)
            except Exception as exc:  # noqa: BLE001 - reported as a CameraError
                result = (None, CameraError(f"capture failed: {type(exc).__name__}"))
            try:
                loop.call_soon_threadsafe(deliver, *result)
            except RuntimeError:
                pass  # the event loop is closed (app shutting down)

        self._thread = threading.Thread(target=run, name="camera-capture", daemon=True)
        self._thread.start()
        timeout_s = self._settings.timeout_s
        try:
            return await asyncio.wait_for(future, timeout_s)
        except TimeoutError:
            log.warning("camera: no frame within %g s", timeout_s)
            raise CameraError(f"camera did not respond within {timeout_s:g} s") from None

    def _grab(self) -> bytes:
        s = self._settings
        t0 = time.perf_counter()
        cap = self._open(s.device_index, s.api)
        try:
            if not cap.isOpened():
                raise CameraError(f"camera {s.device_index} did not open ({s.api})")
            t1 = time.perf_counter()
            for _ in range(s.warmup_frames):
                cap.read()
            ok, frame = cap.read()
            if not ok or frame is None:
                raise CameraError("camera read failed")
        finally:
            cap.release()
        t2 = time.perf_counter()
        jpeg = encode_frame(frame, s.max_edge_px)
        h, w = frame.shape[:2]
        log.info(
            "camera: open %d ms, %d warm-up frames + grab %d ms, %dx%d, closed",
            (t1 - t0) * 1000,
            s.warmup_frames,
            (t2 - t1) * 1000,
            w,
            h,
        )
        return jpeg
