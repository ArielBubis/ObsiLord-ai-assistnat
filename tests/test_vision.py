"""llama-server vision client with a fake server process, plus the request,
prompt, file check, and Windows job object helpers. No model is started."""

import asyncio
import base64
import dataclasses
import http.server
import importlib
import logging
import json
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy
import pytest

from obsilord.camera import encode_frame
from obsilord.config import REPO_ROOT, VisionSettings
from obsilord.vision.base import VisionError
from obsilord.vision.llama_server import (
    LlamaServerVision,
    ServerProcess,
    build_request,
    check_files,
    close_job,
    http_error_text,
    kill_on_close_job,
    load_prompt,
    parse_answer,
    trim_answer,
)


def test_repo_prompt_has_the_question() -> None:
    assert "{question}" in load_prompt()


def test_prompt_without_placeholder_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "vision.md"
    path.write_text("Describe the photo.", encoding="utf-8")
    with pytest.raises(ValueError, match="question"):
        load_prompt(path)


def test_build_request() -> None:
    body = build_request(b"\xff\xd8jpeg", "What am I holding?", "Q: {question}", 120)
    content = body["messages"][0]["content"]
    assert content[0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(b"\xff\xd8jpeg").decode()},
    }
    assert content[1] == {"type": "text", "text": "Q: What am I holding?"}
    assert body["max_tokens"] == 120
    assert body["cache_prompt"] is False and body["stream"] is False


def test_parse_answer() -> None:
    assert parse_answer({"choices": [{"message": {"content": "  A mug.\n"}}]}) == "A mug."
    # No sentence end within the cap: a hard cut.
    assert len(parse_answer({"choices": [{"message": {"content": "x" * 5000}}]})) == 1000
    long = "A bee sits on a flower. " * 60  # 1440 chars
    answer = parse_answer({"choices": [{"message": {"content": long}}]})
    assert answer.endswith("flower.") and len(answer) <= 1000
    cut = {
        "message": {"content": "A bee on a flower. The light is soft and"},
        "finish_reason": "length",
    }
    assert parse_answer({"choices": [cut]}) == "A bee on a flower."
    for bad in ({}, {"choices": []}, {"choices": [{"message": {"content": None}}]}, None):
        with pytest.raises(VisionError):
            parse_answer(bad)


def test_trim_answer() -> None:
    assert trim_answer("Short and complete.", 100, cut_off=False) == "Short and complete."
    # Cut off by max_tokens: the unfinished last sentence goes.
    assert trim_answer("It weighs 3.5 kg. It is red and", 100, cut_off=True) == "It weighs 3.5 kg."
    # A sentence that ends exactly at the limit is kept whole.
    assert trim_answer("Ab. Cd. Ef", 7, cut_off=False) == "Ab. Cd."
    # Quotes after the full stop stay with the sentence.
    assert trim_answer('It says "Demon Days." More text here', 21, cut_off=False) == (
        'It says "Demon Days."'
    )
    # Cut off with no sentence end: kept as it is.
    assert trim_answer("A framed picture of", 100, cut_off=True) == "A framed picture of"


def _files(tmp_path: Path, **missing: bool) -> VisionSettings:
    names = {
        "server_vulkan": "vk/llama-server.exe",
        "server_cpu": "cpu/llama-server.exe",
        "model": "m.gguf",
        "mmproj": "mm.gguf",
    }
    for key, rel in names.items():
        if not missing.get(key):
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / rel).write_bytes(b"x")
    return VisionSettings(**names)


def test_check_files(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    check_files(_files(tmp_path), tmp_path)
    with pytest.raises(ValueError, match="model not found"):
        check_files(_files(tmp_path / "a", model=True), tmp_path / "a")
    with pytest.raises(ValueError, match="server_vulkan not found"):
        check_files(_files(tmp_path / "b", server_vulkan=True), tmp_path / "b")
    # A missing CPU build is only a warning for the vulkan backend.
    check_files(_files(tmp_path / "c", server_cpu=True), tmp_path / "c")
    assert "no CPU fallback" in caplog.text


class FakeServer:
    """Stands in for ServerProcess. ready_after: health checks until healthy
    (None never); exit_code: reported once started (None keeps running)."""

    def __init__(
        self,
        backend: str,
        ready_after: int | None = 1,
        exit_code: int | None = None,
        answer: str = "A red mug.",
        start_error: bool = False,
    ) -> None:
        self.backend = backend
        self.ready_after = ready_after
        self._exit_code = exit_code
        self.answer = answer
        self.start_error = start_error
        self.started = False
        self.stopped = False
        self.health_checks = 0
        self.posts: list[dict[str, Any]] = []

    def start(self) -> None:
        if self.start_error:
            raise OSError("not a valid Win32 application")
        self.started = True

    def exit_code(self) -> int | None:
        return self._exit_code if self.started else None

    def healthy(self) -> bool:
        self.health_checks += 1
        return self.ready_after is not None and self.health_checks >= self.ready_after

    def post(self, body: dict[str, Any], timeout_s: float) -> Any:
        self.posts.append(body)
        return {"choices": [{"message": {"content": self.answer}}]}

    def log_tail(self, n: int) -> list[str]:
        return ["ggml_vulkan: out of memory"][-n:]

    def stop(self) -> None:
        self.stopped = True


def _vision(
    tmp_path: Path, servers: dict[str, FakeServer], **kw: Any
) -> tuple[LlamaServerVision, list[str]]:
    settings = dataclasses.replace(_files(tmp_path), **{"start_timeout_s": 0.5, **kw})
    launched: list[str] = []

    def launcher(backend: str, exe: Path) -> FakeServer:
        launched.append(backend)
        return servers[backend]

    return LlamaServerVision(settings, tmp_path, prompt="{question}", launcher=launcher), launched


def _jpeg(w: int = 640, h: int = 480) -> bytes:
    return encode_frame(numpy.full((h, w, 3), 90, numpy.uint8), 4096)


def _sent_size(body: dict[str, Any]) -> tuple[int, int]:
    url = body["messages"][0]["content"][0]["image_url"]["url"]
    raw = base64.b64decode(url.split(",", 1)[1])
    img = cv2.imdecode(numpy.frombuffer(raw, numpy.uint8), cv2.IMREAD_COLOR)
    return img.shape[1], img.shape[0]


async def _describe(vision: LlamaServerVision, jpeg: bytes, question: str = "q") -> str:
    async with vision.session() as session:
        return await session.describe(jpeg, question)


def test_vulkan_session_starts_answers_and_stops(tmp_path: Path) -> None:
    vk = FakeServer("vulkan", ready_after=3)
    vision, launched = _vision(tmp_path, {"vulkan": vk})
    answer = asyncio.run(_describe(vision, _jpeg(), "What am I holding?"))
    assert answer == "A red mug."
    assert launched == ["vulkan"]
    assert vk.stopped
    # Vulkan gets the frame at full size; the question is in the prompt.
    assert _sent_size(vk.posts[0]) == (640, 480)
    assert vk.posts[0]["messages"][0]["content"][1]["text"] == "What am I holding?"


def test_vulkan_exit_falls_back_to_cpu_with_a_smaller_frame(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    vk, cpu = FakeServer("vulkan", ready_after=None, exit_code=1), FakeServer("cpu")
    vision, launched = _vision(tmp_path, {"vulkan": vk, "cpu": cpu}, cpu_max_edge_px=512)
    answer = asyncio.run(_describe(vision, _jpeg()))
    assert answer == "A red mug."
    assert launched == ["vulkan", "cpu"]
    assert vk.stopped and cpu.stopped
    assert _sent_size(cpu.posts[0]) == (512, 384)
    assert "exited with code 1" in caplog.text
    assert "out of memory" in caplog.text  # the server's own last log lines


def test_vulkan_too_slow_falls_back(tmp_path: Path) -> None:
    vk, cpu = FakeServer("vulkan", ready_after=None), FakeServer("cpu")
    vision, launched = _vision(tmp_path, {"vulkan": vk, "cpu": cpu})
    t0 = time.monotonic()
    asyncio.run(_describe(vision, _jpeg()))
    assert launched == ["vulkan", "cpu"]
    assert time.monotonic() - t0 < 3


def test_vulkan_that_cannot_run_falls_back(tmp_path: Path) -> None:
    vk, cpu = FakeServer("vulkan", start_error=True), FakeServer("cpu")
    vision, launched = _vision(tmp_path, {"vulkan": vk, "cpu": cpu})
    assert asyncio.run(_describe(vision, _jpeg())) == "A red mug."
    assert launched == ["vulkan", "cpu"]


def test_cpu_backend_does_not_try_vulkan(tmp_path: Path) -> None:
    cpu = FakeServer("cpu")
    vision, launched = _vision(tmp_path, {"cpu": cpu}, backend="cpu")
    asyncio.run(_describe(vision, _jpeg()))
    assert launched == ["cpu"]


def test_both_fail(tmp_path: Path) -> None:
    servers = {
        "vulkan": FakeServer("vulkan", ready_after=None, exit_code=1),
        "cpu": FakeServer("cpu", ready_after=None, exit_code=3),
    }
    vision, _ = _vision(tmp_path, servers)
    with pytest.raises(VisionError, match="vulkan: exited with code 1; cpu: exited with code 3"):
        asyncio.run(_describe(vision, _jpeg()))
    assert all(s.stopped for s in servers.values())


def test_request_failure_is_a_vision_error(tmp_path: Path) -> None:
    class Broken(FakeServer):
        def post(self, body: dict[str, Any], timeout_s: float) -> Any:
            raise OSError("connection reset")

    vk = Broken("vulkan")
    vision, _ = _vision(tmp_path, {"vulkan": vk})
    with pytest.raises(VisionError, match="OSError"):
        asyncio.run(_describe(vision, _jpeg()))
    assert vk.stopped


def test_cancel_while_loading_stops_the_server(tmp_path: Path) -> None:
    vk = FakeServer("vulkan", ready_after=None)
    vision, _ = _vision(tmp_path, {"vulkan": vk}, start_timeout_s=10.0)

    async def go() -> None:
        task = asyncio.create_task(_describe(vision, _jpeg()))
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert vk.stopped


def test_http_error_text() -> None:
    body = json.dumps({"error": {"code": 400, "message": "the request\nexceeds the context"}})
    assert http_error_text(400, body.encode()) == "HTTP 400: the request exceeds the context"
    assert http_error_text(500, b"not json") == "HTTP 500"
    assert http_error_text(500, b'{"error": "flat"}') == "HTTP 500"
    long = json.dumps({"error": {"message": "x" * 999}}).encode()
    assert http_error_text(400, long) == "HTTP 400: " + "x" * 200


CONTEXT_ERROR = {"error": {"code": 400, "message": "the request exceeds the available context size"}}


@contextmanager
def _error_server() -> Iterator[int]:
    """A local HTTP server on 127.0.0.1 that answers every POST with a 400."""

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            body = json.dumps(CONTEXT_ERROR).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: Any) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield int(httpd.server_address[1])
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_http_error_is_a_vision_error_with_the_server_message() -> None:
    proc = ServerProcess("vulkan", Path("server.exe"), Path("m.gguf"), Path("mm.gguf"), 2048)
    with _error_server() as port:
        proc.port = port
        with pytest.raises(VisionError, match="returned HTTP 400: the request exceeds"):
            proc.post({"messages": []}, 5)


def test_requests_ignore_proxy_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a dead proxy set before the module builds its opener, requests
    still go straight to 127.0.0.1 (through a proxy they would be refused)."""
    from obsilord.vision import llama_server

    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    try:
        module = importlib.reload(llama_server)
        proc = module.ServerProcess("cpu", Path("s.exe"), Path("m.gguf"), Path("mm.gguf"), 2048)
        with _error_server() as port:
            proc.port = port
            with pytest.raises(VisionError, match="HTTP 400"):
                proc.post({"messages": []}, 5)
    finally:
        monkeypatch.undo()
        importlib.reload(llama_server)


def test_server_args_bind_loopback_only() -> None:
    proc = ServerProcess("vulkan", Path("server.exe"), Path("m.gguf"), Path("mm.gguf"), 2048)
    args = proc.args
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert args[args.index("-c") + 1] == "2048"
    assert "--no-webui" in args and args[-2:] == ["-ngl", "99"]
    cpu = ServerProcess("cpu", Path("server.exe"), Path("m.gguf"), Path("mm.gguf"), 2048)
    assert "-ngl" not in cpu.args


def test_default_paths_match_fetch() -> None:
    from obsilord.vision import fetch

    defaults = VisionSettings()
    assert REPO_ROOT / defaults.model == fetch.MODEL_ROOT / fetch.MODEL_KEY / fetch.MODEL.name
    assert REPO_ROOT / defaults.mmproj == fetch.MODEL_ROOT / fetch.MODEL_KEY / fetch.MMPROJ.name
    for backend in ("vulkan", "cpu"):
        exe = REPO_ROOT / getattr(defaults, f"server_{backend}")
        assert exe == fetch.binary_dir(backend) / "llama-server.exe"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
def test_job_object_kills_the_child_when_closed() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        job = kill_on_close_job(child.pid)
        assert job is not None
        assert child.poll() is None
        close_job(job)  # the last handle: the job kills its processes
        assert child.wait(10) is not None
    finally:
        if child.poll() is None:
            child.kill()


def test_prompt_log_shows_the_prompt_but_not_the_image(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="obsilord.prompts")
    vk = FakeServer("vulkan")
    vision, _ = _vision(tmp_path, {"vulkan": vk})
    jpeg = _jpeg()
    asyncio.run(_describe(vision, jpeg, "What am I holding?"))
    request, response = [r.getMessage() for r in caplog.records if r.name == "obsilord.prompts"]
    assert request.splitlines() == [
        'vision request (vulkan): {"max_tokens": 120, "temperature": 0.2, '
        '"cache_prompt": false, "stream": false}',
        "[user image_url] data:image/jpeg;base64, "
        f"{len(jpeg) / 1024:.0f} KB (not shown)",
        "[user text]",
        "What am I holding?",
    ]
    b64 = base64.b64encode(jpeg).decode("ascii")
    assert b64[:40] not in caplog.text
    assert response.startswith("vision response (vulkan) after ")
    assert response.endswith('"usage": null, "timings": null}\nA red mug.')
