"""llama.cpp llama-server implementation of VisionClient (prebuilt build b11919).

This is the only module that starts llama-server. Checked against build b11919
with a local bench script (not in the repo) on 2026-10-27 (Windows 11, RX 6600 XT):
- `llama-server -m MODEL --mmproj MMPROJ --host 127.0.0.1 --port N -c CTX
  --no-webui [-ngl 99]` starts, answers GET /health with 200 once the model is
  loaded (503 before), and serves POST /v1/chat/completions with OpenAI-style
  messages. The image goes in as {"type": "image_url", "image_url": {"url":
  "data:image/jpeg;base64,..."}}; "cache_prompt": false turns off reuse of the
  previous prompt; the answer is choices[0].message.content.
- Qwen3-VL-2B Q4_K_M, started per look: load 1.3 s plus 0.7 s per answer on
  Vulkan (1.6 s with LM Studio's 7B model in VRAM); on the CPU build about
  3.8 s per answer at 768 px and 1.7 s at 512 px.
- The server log did not contain the image data. It is kept in memory anyway
  and only its last lines are logged, when a server fails to start (before any
  image was sent).
- A full-size 5184x3456 photo on the Vulkan build with ctx 2048 got an HTTP
  error (too many image tokens for the context, presumably). Camera frames are
  at most [camera] max_edge_px, 768 by default; the error text, status and
  llama-server's message, goes to the log.

session() starts the server for [vision] backend, waits for /health, and stops
it on exit, so nothing stays loaded between looks. If the Vulkan server exits
or is not ready within start_timeout_s, the CPU build is tried for that look
and the frame is shrunk to cpu_max_edge_px first. On Windows each server is
put in a job object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, so it dies with
this process, also after a crash.

HTTP goes through an opener without proxies, so a system proxy setting can
never route a frame off the machine. With [debug] log_prompts on, each request
is printed with its image as type and size only, and each reply with
llama-server's usage and timings. Blocking work (HTTP, JPEG resize) runs in
worker threads; the process is started with subprocess.Popen, which works on
any event loop (asyncio subprocesses need the Proactor loop on Windows).
"""

from __future__ import annotations

import asyncio
import base64
import collections
import ctypes
import http.client
import json
import logging
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Protocol

from obsilord import promptlog
from obsilord.camera import shrink_jpeg
from obsilord.config import REPO_ROOT, VisionSettings
from obsilord.vision.base import VisionError, VisionSession

log = logging.getLogger(__name__)

PROMPT_PATH: Path = REPO_ROOT / "prompts" / "vision.md"
QUESTION_PLACEHOLDER = "{question}"
HOST = "127.0.0.1"
HEALTH_POLL_S = 0.1
LOG_LINES = 200
LOG_TAIL_ON_FAILURE = 5
MAX_ANSWER_CHARS = 1000
MAX_ERROR_BYTES = 4096
MAX_ERROR_CHARS = 200
STOP_WAIT_S = 5.0

# No proxies, ever: requests go to llama-server on 127.0.0.1 only.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def load_prompt(path: Path | None = None) -> str:
    """Read the vision prompt (prompts/vision.md); it must contain {question}."""
    path = path if path is not None else PROMPT_PATH
    text = path.read_text(encoding="utf-8").strip()
    if QUESTION_PLACEHOLDER not in text:
        raise ValueError(f"{path.name} must contain {QUESTION_PLACEHOLDER}")
    return text


def resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def check_files(vision: VisionSettings, repo_root: Path) -> None:
    """Raise ValueError naming the first missing file. A missing CPU build is
    only a warning when the backend is vulkan (no fallback then)."""
    required = [("model", vision.model), ("mmproj", vision.mmproj)]
    required.append((f"server_{vision.backend}", getattr(vision, f"server_{vision.backend}")))
    for key, value in required:
        if not resolve(repo_root, value).is_file():
            raise ValueError(f"vision: [vision] {key} not found: {value}")
    if vision.backend == "vulkan" and not resolve(repo_root, vision.server_cpu).is_file():
        log.warning(
            "vision: [vision] server_cpu not found (%s), no CPU fallback if Vulkan fails",
            vision.server_cpu,
        )


def build_request(jpeg: bytes, question: str, prompt: str, max_tokens: int) -> dict[str, Any]:
    """The /v1/chat/completions body: the image first, then the prompt text."""
    b64 = base64.b64encode(jpeg).decode("ascii")
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt.replace(QUESTION_PLACEHOLDER, question)},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        # Every look is a new frame: nothing to reuse from the last prompt.
        "cache_prompt": False,
        "stream": False,
    }


# A sentence end: . ! or ? (plus closing quotes or brackets) before whitespace
# or the end of the text, so "3.5 kg" and "v1.2" do not count.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*(?=\s|$)")


def trim_answer(text: str, limit: int, cut_off: bool) -> str:
    """text unchanged if it fits in limit chars and was not cut off. Otherwise
    it ends at the last whole sentence within limit; with no sentence end to
    cut at, it is cut hard at limit."""
    if len(text) <= limit and not cut_off:
        return text
    # One extra char so a sentence ending exactly at limit sees what follows.
    window = text[: limit + 1]
    ends = [m.end() for m in _SENTENCE_END.finditer(window) if m.end() <= limit]
    if ends:
        return text[: ends[-1]]
    return text[:limit].rstrip()


def parse_answer(data: Any) -> str:
    """choices[0].message.content, stripped and trimmed to whole sentences
    within MAX_ANSWER_CHARS. Raises VisionError."""
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise VisionError("unexpected response from llama-server") from None
    if not isinstance(content, str):
        raise VisionError("unexpected response from llama-server")
    # finish_reason "length" means max_tokens stopped the answer (the OpenAI
    # meaning; unverified for llama-server b11919, the prompt log shows it).
    cut_off = isinstance(choice, dict) and choice.get("finish_reason") == "length"
    return trim_answer(content.strip(), MAX_ANSWER_CHARS, cut_off)


def format_request(backend: str, body: dict[str, Any]) -> str:
    """The request for the prompt log ([debug] log_prompts): every setting and
    every text exactly, each image as its type and size only."""
    settings = {k: v for k, v in body.items() if k != "messages"}
    lines = [f"vision request ({backend}): {json.dumps(settings)}"]
    for message in body["messages"]:
        role = message["role"]
        for item in message["content"]:
            if item.get("type") == "text":
                lines.append(f"[{role} text]")
                lines.append(item["text"])
            else:
                url = str(item.get("image_url", {}).get("url", ""))
                head, _, data = url.partition(",")
                kb = len(data) * 3 / 4 / 1024
                lines.append(f"[{role} {item.get('type')}] {head}, {kb:.0f} KB (not shown)")
    return "\n".join(lines)


def format_response(backend: str, data: Any, ms: float) -> str:
    """The reply for the prompt log: llama-server's counts and timings, then the
    answer as received (before strip and cap)."""
    choice = data["choices"][0]
    info = {
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "timings": data.get("timings"),
    }
    head = f"vision response ({backend}) after {ms:.0f} ms: {json.dumps(info)}"
    return f"{head}\n{choice['message']['content']}"


def http_error_text(status: int, body: bytes) -> str:
    """'HTTP <status>' plus llama-server's error message, if the body has one
    (for example a context overflow). Short, for the log."""
    try:
        message = json.loads(body)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        message = None
    text = " ".join(message.split())[:MAX_ERROR_CHARS] if isinstance(message, str) else ""
    return f"HTTP {status}: {text}" if text else f"HTTP {status}"


# Another program could take the port before llama-server binds it. The server
# then exits, and that counts as a failed start like any other.
def _free_port() -> int:
    with socket.socket() as s:
        s.bind((HOST, 0))
        return int(s.getsockname()[1])


# --- Windows job object ----------------------------------------------------

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_longlong),
        ("PerJobUserTimeLimit", ctypes.c_longlong),
        ("LimitFlags", ctypes.c_ulong),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_ulong),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_ulong),
        ("SchedulingClass", ctypes.c_ulong),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_ulonglong)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _kernel32() -> Any:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    k32.CreateJobObjectW.restype = ctypes.c_void_p
    k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    k32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong
    ]
    k32.OpenProcess.restype = ctypes.c_void_p
    k32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    k32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    k32.CloseHandle.argtypes = [ctypes.c_void_p]
    return k32


def kill_on_close_job(pid: int) -> int | None:
    """Put process pid in a new job object that kills it when the job handle
    closes: on close_job, or when this process ends for any reason. Returns
    the job handle, or None (not Windows, or a call failed; logged)."""
    if sys.platform != "win32":
        return None
    k32 = _kernel32()
    job = k32.CreateJobObjectW(None, None)
    if not job:
        log.warning("vision: CreateJobObject failed (error %d)", ctypes.get_last_error())
        return None
    info = _ExtendedLimit()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = k32.SetInformationJobObject(
        job, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info), ctypes.sizeof(info)
    )
    process = k32.OpenProcess(_PROCESS_TERMINATE | _PROCESS_SET_QUOTA, 0, pid) if ok else None
    if not ok or not process or not k32.AssignProcessToJobObject(job, process):
        log.warning("vision: job object setup failed (error %d)", ctypes.get_last_error())
        if process:
            k32.CloseHandle(process)
        k32.CloseHandle(job)
        return None
    k32.CloseHandle(process)
    return int(job)


def close_job(job: int | None) -> None:
    if job is not None and sys.platform == "win32":
        _kernel32().CloseHandle(job)


# --- server process --------------------------------------------------------


class Server(Protocol):
    """One llama-server process. Blocking methods, called from worker threads
    except start (quick) and stop."""

    backend: str

    def start(self) -> None: ...
    def exit_code(self) -> int | None: ...
    def healthy(self) -> bool: ...
    def post(self, body: dict[str, Any], timeout_s: float) -> Any: ...
    def log_tail(self, n: int) -> list[str]: ...
    def stop(self) -> None: ...


class ServerProcess:
    def __init__(
        self, backend: str, exe: Path, model: Path, mmproj: Path, ctx: int
    ) -> None:
        self.backend = backend
        self.port = _free_port()
        self.args = [
            str(exe),
            "-m", str(model),
            "--mmproj", str(mmproj),
            "--host", HOST,
            "--port", str(self.port),
            "-c", str(ctx),
            "--no-webui",
        ]
        if backend == "vulkan":
            self.args += ["-ngl", "99"]
        self._log: collections.deque[str] = collections.deque(maxlen=LOG_LINES)
        self._proc: subprocess.Popen[str] | None = None
        self._job: int | None = None

    def start(self) -> None:
        self._proc = subprocess.Popen(
            self.args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._job = kill_on_close_job(self._proc.pid)
        threading.Thread(target=self._drain, name="llama-server-log", daemon=True).start()

    def _drain(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        with proc.stdout:  # closed at EOF, when the server has exited
            for line in proc.stdout:
                self._log.append(line.rstrip())

    def exit_code(self) -> int | None:
        return None if self._proc is None else self._proc.poll()

    def healthy(self) -> bool:
        try:
            with _OPENER.open(f"http://{HOST}:{self.port}/health", timeout=1) as resp:
                return bool(resp.status == 200)
        except (OSError, http.client.HTTPException):  # HTTPError 503 while loading
            return False

    def post(self, body: dict[str, Any], timeout_s: float) -> Any:
        request = urllib.request.Request(
            f"http://{HOST}:{self.port}/v1/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with _OPENER.open(request, timeout=timeout_s) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            try:
                err = exc.read(MAX_ERROR_BYTES)
            except (OSError, http.client.HTTPException):
                err = b""
            finally:
                exc.close()
            raise VisionError(f"llama-server returned {http_error_text(exc.code, err)}") from None

    def log_tail(self, n: int) -> list[str]:
        return list(self._log)[-n:]

    def stop(self) -> None:
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(STOP_WAIT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(STOP_WAIT_S)
        close_job(self._job)
        self._job = None


# (backend, server exe) -> a Server, not started yet. Replaced in tests.
Launcher = Callable[[str, Path], Server]


class _Session:
    def __init__(self, owner: LlamaServerVision, server: Server) -> None:
        self._owner = owner
        self._server = server

    async def describe(self, jpeg: bytes, question: str) -> str:
        owner = self._owner
        backend = self._server.backend
        if backend == "cpu":
            try:
                jpeg = await asyncio.to_thread(shrink_jpeg, jpeg, owner.settings.cpu_max_edge_px)
            except ValueError as exc:
                raise VisionError(f"could not shrink the frame: {exc}") from None
        body = build_request(jpeg, question, owner.prompt, owner.settings.max_tokens)
        promptlog.emit(format_request, backend, body)
        t0 = time.perf_counter()
        try:
            data = await asyncio.to_thread(
                self._server.post, body, owner.settings.request_timeout_s
            )
        except (OSError, ValueError, http.client.HTTPException) as exc:  # URLError, timeouts, bad JSON
            raise VisionError(f"llama-server request failed: {type(exc).__name__}") from None
        answer = parse_answer(data)
        promptlog.emit(format_response, backend, data, (time.perf_counter() - t0) * 1000)
        log.info(
            "vision: %s answer in %d ms (%d chars)",
            backend,
            (time.perf_counter() - t0) * 1000,
            len(answer),
        )
        return answer


class LlamaServerVision:
    def __init__(
        self,
        settings: VisionSettings,
        repo_root: Path = REPO_ROOT,
        prompt: str | None = None,
        launcher: Launcher | None = None,
    ) -> None:
        self.settings = settings
        self.prompt = prompt if prompt is not None else load_prompt()
        self._repo_root = repo_root
        self._launch = launcher if launcher is not None else self._new_server
        self._order = ["vulkan", "cpu"] if settings.backend == "vulkan" else ["cpu"]
        # One look at a time: two servers would compete for the GPU and RAM.
        self._lock = asyncio.Lock()

    def _new_server(self, backend: str, exe: Path) -> Server:
        s = self.settings
        return ServerProcess(
            backend,
            exe,
            resolve(self._repo_root, s.model),
            resolve(self._repo_root, s.mmproj),
            s.ctx,
        )

    @asynccontextmanager
    async def session(self) -> AsyncIterator[VisionSession]:
        async with self._lock:
            server = await self._start()
            try:
                yield _Session(self, server)
            finally:
                # Synchronous, so it runs even while the task is being
                # cancelled. terminate() is immediate on Windows.
                server.stop()

    async def _start(self) -> Server:
        reasons: list[str] = []
        for backend in self._order:
            exe = resolve(self._repo_root, getattr(self.settings, f"server_{backend}"))
            if not exe.is_file():
                reasons.append(f"{backend}: server not found")
                continue
            server = self._launch(backend, exe)
            t0 = time.perf_counter()
            try:
                server.start()
            except OSError as exc:
                server.stop()
                log.warning("vision: %s server could not run: %s", backend, type(exc).__name__)
                reasons.append(f"{backend}: could not run ({type(exc).__name__})")
                continue
            try:
                reason = await self._wait_ready(server)
            except BaseException:
                server.stop()
                raise
            if reason is None:
                log.info(
                    "vision: %s server ready in %d ms", backend, (time.perf_counter() - t0) * 1000
                )
                return server
            tail = server.log_tail(LOG_TAIL_ON_FAILURE)
            server.stop()
            log.warning(
                "vision: %s server failed to start (%s); last log lines: %s",
                backend,
                reason,
                " | ".join(tail) or "-",
            )
            reasons.append(f"{backend}: {reason}")
        raise VisionError("llama-server did not start (" + "; ".join(reasons) + ")")

    async def _wait_ready(self, server: Server) -> str | None:
        """None once /health answers 200, else why not (exited or too slow)."""
        deadline = time.monotonic() + self.settings.start_timeout_s
        while time.monotonic() < deadline:
            code = server.exit_code()
            if code is not None:
                return f"exited with code {code}"
            if await asyncio.to_thread(server.healthy):
                return None
            await asyncio.sleep(HEALTH_POLL_S)
        return f"not ready after {self.settings.start_timeout_s:g} s"
