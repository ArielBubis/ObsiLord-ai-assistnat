"""Live look(): the real webcam and the real llama-server. Needs the files and
the sample photo from `python -m obsilord.vision.fetch --sample`. Run with:

    .venv/Scripts/python.exe -m pytest -m hardware tests/test_look_live.py -q -s

The camera tests turn the webcam on for about a second. Nothing is written to
disk. -s shows timings and the answers.
"""

import asyncio
import dataclasses
import time
from pathlib import Path

import pytest

from obsilord.camera import CameraService, shrink_jpeg
from obsilord.config import REPO_ROOT, Settings, load_settings
from obsilord.look import LookLimiter, Looker
from obsilord.vision import fetch
from obsilord.vision.llama_server import LlamaServerVision, ServerProcess, check_files

pytestmark = pytest.mark.hardware

SAMPLE = fetch.SAMPLE_DIR / fetch.SAMPLE.name


def _settings() -> Settings:
    settings = load_settings()
    try:
        check_files(settings.vision, settings.repo_root)
    except ValueError as exc:
        pytest.skip(str(exc))
    return settings


def test_camera_capture() -> None:
    settings = _settings()
    t0 = time.perf_counter()
    jpeg = asyncio.run(CameraService(settings.camera).capture_jpeg())
    print(f"\ncamera: {len(jpeg) / 1024:.0f} KB JPEG in {(time.perf_counter() - t0) * 1000:.0f} ms")
    assert jpeg[:2] == b"\xff\xd8"


@pytest.mark.parametrize("backend", ["cuda", "cpu"])
def test_vision_describes_the_sample_and_stops(backend: str) -> None:
    if not SAMPLE.is_file():
        pytest.skip("sample photo missing")
    settings = _settings()
    vision_cfg = dataclasses.replace(settings.vision, backend=backend)
    servers: list[ServerProcess] = []

    def launcher(name: str, exe: Path) -> ServerProcess:
        s = vision_cfg
        server = ServerProcess(name, exe, REPO_ROOT / s.model, REPO_ROOT / s.mmproj, s.ctx)
        servers.append(server)
        return server

    vision = LlamaServerVision(vision_cfg, settings.repo_root, launcher=launcher)
    # Sized like a camera frame. The full 5184x3456 sample is more image tokens
    # than ctx holds, and the camera never sends more than max_edge_px.
    jpeg = shrink_jpeg(SAMPLE.read_bytes(), settings.camera.max_edge_px)

    async def go() -> str:
        async with vision.session() as session:
            return await session.describe(jpeg, "What is in the photo?")

    t0 = time.perf_counter()
    answer = asyncio.run(go())
    print(f"\n{backend}: {(time.perf_counter() - t0) * 1000:.0f} ms, servers {[s.backend for s in servers]}: {answer}")
    assert "bee" in answer.lower()
    # Every server started for the look has exited.
    assert servers and all(s.exit_code() is not None for s in servers)


def test_full_look() -> None:
    settings = _settings()
    looker = Looker(
        CameraService(settings.camera),
        LlamaServerVision(settings.vision, settings.repo_root),
        LookLimiter(settings.camera.cooldown_s, settings.camera.max_per_hour),
    )

    async def go() -> tuple[dict[str, str], dict[str, str]]:
        return await looker.look("What is in front of the camera?"), await looker.look("again")

    t0 = time.perf_counter()
    first, second = asyncio.run(go())
    print(f"\nlook: {(time.perf_counter() - t0) * 1000:.0f} ms: {first}")
    assert first.get("output")
    assert second["error"].startswith("Camera cooldown")


def test_search_image_crops_the_sample_with_the_real_model(tmp_path: Path) -> None:
    """The real vision model boxes the bee, the app crops it, and a fake search
    client gets only the crop. Nothing is uploaded and no camera is used."""
    import cv2
    import numpy

    from obsilord.imagesearch.base import ImageMatches
    from obsilord.imagesearch.budget import SearchBudget
    from obsilord.look import ImageSearch

    if not SAMPLE.is_file():
        pytest.skip("sample photo missing")
    settings = _settings()
    frame = shrink_jpeg(SAMPLE.read_bytes(), settings.camera.max_edge_px)

    class SampleCamera:
        async def capture_jpeg(self) -> bytes:
            return frame

    class RecordingSearch:
        provider = "fake"
        uploads: list[bytes] = []

        async def search(self, jpeg: bytes) -> ImageMatches:
            self.uploads.append(jpeg)
            return ImageMatches(best_guesses=("bumblebee",))

    def size(jpeg: bytes) -> tuple[int, int]:
        image = cv2.imdecode(numpy.frombuffer(jpeg, numpy.uint8), cv2.IMREAD_COLOR)
        return image.shape[1], image.shape[0]

    client = RecordingSearch()
    looker = Looker(
        SampleCamera(),
        LlamaServerVision(settings.vision, settings.repo_root),
        LookLimiter(0, 30),
        ImageSearch(client, SearchBudget(tmp_path / "image_search.jsonl", 10, 10), 0.8),
    )
    snaps: list[tuple[str, tuple[int, int]]] = []

    async def snapshot(kind: str, jpeg: bytes) -> None:
        snaps.append((kind, size(jpeg)))

    t0 = time.perf_counter()
    result = asyncio.run(looker.search_image("the bee", snapshot))
    print(f"\nsearch_image: {(time.perf_counter() - t0) * 1000:.0f} ms, snapshots {snaps}: {result}")
    assert "bumblebee" in result["output"]
    (upload,) = client.uploads
    full, crop = size(frame), size(upload)
    # The bee is a small part of the photo.
    assert crop[0] < full[0] / 3 and crop[1] < full[1] / 3
    assert snaps == [("camera", full), ("search", crop)]
