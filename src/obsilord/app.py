"""FastAPI app: serves the web page and the /ws WebSocket.

{"type": "text"} runs a streamed LLM turn with spoken output (see
pipeline.py). Push-to-talk voice input: {"type": "audio_start"}, binary PCM16
mono little-endian frames at 16 kHz, then {"type": "audio_end"}, which runs a
voice turn (STT, then the same turn body). Invalid JSON gets an error message
and the connection stays open.

Per connection the socket is idle, recording, or running a turn:
- audio_start: rejected with an error (no state message) if voice input is
  disabled, no API key is configured, or a turn is running or a recording is
  open. Otherwise the buffer is cleared and state listening is sent.
- Binary frame while recording: appended, up to stt.max_utterance_s; the rest
  is dropped and one "Recording too long" error is sent per recording.
  Binary frame otherwise: one "Unexpected audio frame." error per connection,
  the frame is dropped.
- audio_end without a recording: error. Shorter than stt.min_utterance_ms:
  "Didn't catch that." then idle. Otherwise a voice turn starts.
- A text message while recording is rejected as busy.
- Disconnect while recording discards the buffer.
- A recording with no audio_end within stt.max_utterance_s +
  RECORDING_GRACE_S of audio_start is closed as if audio_end had arrived, so a
  client that lost track cannot leave the socket busy forever.
Mic audio stays in memory only: never written to disk or logged.

The LLM client, STT client, TTS client, and UsageTracker are created once in
create_app (app startup) and can be injected for tests. If no API key is
configured the app still starts and turns return an error. If [tts] voice is
empty, TTS is disabled and replies are text only; if [stt] model is empty,
voice input is disabled. A configured voice or STT model that fails to load
(missing file, unknown speaker, failed download) stops startup with an error.
A configured [face] character file is checked first (face_character.py:
missing, not binary glTF 2.0, compressed, or linking external files), and a
failed check also stops startup with an error. GET /face/character.glb serves
it, or answers 404 when the key is empty.

look(): with [vision] model set, startup checks the vision files (server for
the backend, model, mmproj) and builds the Looker (camera, per-look
llama-server, cooldown and hourly cap); a missing file stops startup with a
readable error. Nothing is loaded until a look runs. An empty model turns
look() off: the tool is not declared.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Coroutine

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from obsilord import promptlog
from obsilord.config import Settings
from obsilord.face_character import check_character
from obsilord.llm.base import LLMClient, Message
from obsilord.look import LookTool
from obsilord.pipeline import MSG_NOT_CAUGHT, run_text_turn, run_voice_turn
from obsilord.stt.base import SAMPLE_RATE as STT_SAMPLE_RATE
from obsilord.stt.base import STTClient, pcm16_bytes_for_ms
from obsilord.tts.base import TTSClient
from obsilord.usage import UsageTracker

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 2000

MSG_INVALID_JSON = "invalid JSON"
MSG_NO_KEY = "No API key configured."
MSG_BUSY = "Busy, wait for the current reply."
MSG_EMPTY = "Message is empty."
MSG_TOO_LONG = f"Message is too long (max {MAX_TEXT_CHARS} characters)."
MSG_NO_VOICE = "Voice input is not available."
MSG_UNKNOWN = "Unknown message type."
MSG_UNEXPECTED_AUDIO = "Unexpected audio frame."
MSG_NOT_RECORDING = "No recording in progress."

# Extra time after max_utterance_s before the server closes a recording that
# never got its audio_end.
RECORDING_GRACE_S = 5.0

# The recording watchdog waits with this (looked up at call time, so tests can
# replace it with an event-driven fake instead of waiting on the wall clock).
_watchdog_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


def msg_too_long_audio(max_utterance_s: float) -> str:
    return f"Recording too long, using the first {max_utterance_s:g} s."


def _checked_character(settings: Settings) -> Path | None:
    """The [face] character file after check_character, or None when the key is
    empty (the page then shows the bundled character)."""
    character = settings.face.character.strip()
    if not character:
        log.info("face: [face] character is empty, the page shows the bundled character")
        return None
    path = settings.repo_root / character
    try:
        check_character(path)
    except ValueError as exc:
        log.error('%s; set [face] character = "" to use the bundled character', exc)
        raise
    return path


def _default_llm(settings: Settings) -> LLMClient | None:
    # Imported here so the provider SDK is only loaded when actually used.
    from obsilord.llm.gemini import GeminiClient, MissingApiKey

    try:
        return GeminiClient(settings.llm, settings.secrets.gemini_api_key)
    except MissingApiKey:
        log.warning("no GEMINI_API_KEY configured; text turns will return an error")
        return None


def _default_tts(settings: Settings) -> TTSClient | None:
    """Load and warm up the configured voice, or None if TTS is disabled."""
    if not settings.tts.enabled:
        log.info("tts: disabled ([tts] voice is empty), replies are text only")
        return None
    # Imported here so the provider package is only loaded when used.
    from obsilord.tts.piper import PiperTTS

    tts = PiperTTS.load(
        settings.repo_root / settings.tts.voice,
        settings.tts.speaker,
        settings.tts.length_scale,
        settings.tts.pitch,
    )
    tts.warm_up()
    return tts


def _default_stt(settings: Settings) -> STTClient | None:
    """Load and warm up the configured STT model, or None if voice input is off."""
    if not settings.stt.enabled:
        log.info("stt: disabled ([stt] model is empty), voice input is off")
        return None
    # Imported here so the provider package is only loaded when used.
    from obsilord.stt.faster_whisper import FasterWhisperSTT, resolve_model

    try:
        stt = FasterWhisperSTT.load(
            resolve_model(settings.stt.model, settings.repo_root),
            settings.stt.device,
            settings.stt.compute_type,
            settings.repo_root / settings.stt.download_root,
            settings.stt.beam_size,
        )
        stt.warm_up(settings.stt.language_or_none)
    except Exception as exc:
        log.error(
            'stt: could not load model %r (%s); set [stt] model = "" to start '
            "without voice input",
            settings.stt.model,
            type(exc).__name__,
        )
        raise
    return stt


def _default_looker(settings: Settings) -> LookTool | None:
    """The look() tool from [camera] and [vision], or None if [vision] model is empty."""
    vision = settings.vision
    if not vision.enabled:
        log.info("look: disabled ([vision] model is empty), no camera tool")
        return None
    # Imported here so OpenCV and the vision module are only loaded when used.
    from obsilord.camera import CameraService
    from obsilord.look import LookLimiter, Looker
    from obsilord.vision.llama_server import LlamaServerVision, check_files

    try:
        check_files(vision, settings.repo_root)
    except ValueError as exc:
        log.error(
            '%s; run "python -m obsilord.vision.fetch", or set [vision] model = "" '
            "to start without look()",
            exc,
        )
        raise
    camera = settings.camera
    looker = Looker(
        CameraService(camera),
        LlamaServerVision(vision, settings.repo_root),
        LookLimiter(camera.cooldown_s, camera.max_per_hour),
    )
    log.info(
        "look: enabled (backend %s, camera %d via %s, cooldown %g s, %d per hour)",
        vision.backend,
        camera.device_index,
        camera.api,
        camera.cooldown_s,
        camera.max_per_hour,
    )
    return looker


def create_app(
    settings: Settings,
    llm: LLMClient | None = None,
    usage: UsageTracker | None = None,
    tts: TTSClient | None = None,
    stt: STTClient | None = None,
    looker: LookTool | None = None,
) -> FastAPI:
    """Build the app. Pass llm, usage, tts, stt, and looker to inject fakes (tests).

    tts=None loads the voice from settings (nothing is loaded when [tts] voice
    is empty). stt=None loads the STT model from settings (nothing is loaded
    when [stt] model is empty). looker=None builds look() from settings (off
    when [vision] model is empty).
    """
    web_dir = settings.repo_root / "web"
    index_path = web_dir / "index.html"

    promptlog.set_enabled(settings.debug.log_prompts)
    if settings.debug.log_prompts:
        log.warning(
            "debug.log_prompts is on: every Gemini and vision request and reply is "
            "printed (camera frames as their size only)"
        )
    # Cheap file check first, so a bad character fails before any model loads.
    character_path = _checked_character(settings)
    if llm is None:
        llm = _default_llm(settings)
    if tts is None:
        tts = _default_tts(settings)
    if stt is None:
        stt = _default_stt(settings)
    if looker is None:
        looker = _default_looker(settings)
    if usage is None:
        usage = UsageTracker(
            settings.repo_root / "logs" / "usage.jsonl", settings.llm.daily_token_cap
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        aclose = getattr(app.state.llm, "aclose", None)
        if aclose is not None:
            await aclose()

    app = FastAPI(
        title="Obsi-Lord", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.state.settings = settings
    app.state.llm = llm
    app.state.usage = usage
    app.state.tts = tts
    app.state.stt = stt
    app.state.looker = looker

    stt_cfg = settings.stt
    max_audio_bytes = pcm16_bytes_for_ms(stt_cfg.max_utterance_s * 1000, STT_SAMPLE_RATE)
    min_audio_bytes = pcm16_bytes_for_ms(stt_cfg.min_utterance_ms, STT_SAMPLE_RATE)
    msg_too_long_audio_text = msg_too_long_audio(stt_cfg.max_utterance_s)
    # How long a recording may stay open without audio_end.
    recording_limit_s = stt_cfg.max_utterance_s + RECORDING_GRACE_S

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(index_path, media_type="text/html")

    @app.get("/config/face", include_in_schema=False)
    async def face_config() -> dict[str, Any]:
        # Face settings only: no secrets and no other sections.
        face = settings.face
        return {
            "logical_width": face.logical_width,
            "logical_height": face.logical_height,
            "mouth_thresholds": list(face.mouth_thresholds),
            "skin": face.skin,
        }

    @app.get("/face/character.glb", include_in_schema=False)
    async def face_character() -> FileResponse:
        # 404 when [face] character is empty (or the file was removed after
        # startup); the dither skin then shows its bundled character.
        if character_path is None or not character_path.is_file():
            raise HTTPException(status_code=404)
        # no-cache: the URL stays the same when the setting changes, so the
        # browser must revalidate (ETag) instead of reusing a stale copy.
        return FileResponse(
            character_path,
            media_type="model/gltf-binary",
            headers={"Cache-Control": "no-cache"},
        )

    @app.get("/config/audio", include_in_schema=False)
    async def audio_config() -> dict[str, Any]:
        # Browser playback EQ and the recording limit (the page stops a
        # push-to-talk recording at max_utterance_s). No secrets, nothing else.
        tts_cfg = settings.tts
        return {
            "low_shelf_hz": tts_cfg.low_shelf_hz,
            "low_shelf_db": tts_cfg.low_shelf_db,
            "high_shelf_hz": tts_cfg.high_shelf_hz,
            "high_shelf_db": tts_cfg.high_shelf_db,
            "max_utterance_s": settings.stt.max_utterance_s,
        }

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        log.info("ws connected")
        history: list[Message] = []
        turn: asyncio.Task[Any] | None = None
        send_lock = asyncio.Lock()
        # Push-to-talk state. recording is None when no recording is open.
        recording: bytearray | None = None
        too_long_sent = False  # per recording
        stray_frame_sent = False  # per connection
        recording_timer: asyncio.Task[None] | None = None  # closes a stuck recording
        # Strong references to watchdog tasks until they finish (the event loop
        # itself only keeps weak ones).
        timers: set[asyncio.Task[None]] = set()
        # Serializes the recording bookkeeping and the messages the handlers and
        # the watchdog send right away, so for example an audio_start cannot
        # slip in between a watchdog close's error and its idle. Turns run as
        # separate tasks and never take this lock.
        recording_lock = asyncio.Lock()

        async def send(msg: dict[str, Any] | bytes) -> None:
            # One lock for JSON and binary frames keeps their order intact.
            async with send_lock:
                if isinstance(msg, bytes):
                    await websocket.send_bytes(msg)
                else:
                    await websocket.send_json(msg)

        def on_turn_done(task: asyncio.Task[Any]) -> None:
            # The turn functions handle their own errors; this is a last-resort log.
            if not task.cancelled() and task.exception() is not None:
                log.error("turn task failed: %s", type(task.exception()).__name__)

        def turn_running() -> bool:
            return turn is not None and not turn.done()

        def start_turn(coro: Coroutine[Any, Any, str | None]) -> None:
            nonlocal turn
            turn = asyncio.create_task(coro)
            turn.add_done_callback(on_turn_done)

        async def handle_text(payload: dict[str, Any]) -> None:
            async with recording_lock:
                await _handle_text_locked(payload)

        async def _handle_text_locked(payload: dict[str, Any]) -> None:
            if recording is not None:
                await send({"type": "error", "message": MSG_BUSY})
                return
            raw = payload.get("text")
            text = raw.strip() if isinstance(raw, str) else ""
            if not text:
                await send({"type": "error", "message": MSG_EMPTY})
                return
            if len(text) > MAX_TEXT_CHARS:
                await send({"type": "error", "message": MSG_TOO_LONG})
                return
            if turn_running():
                await send({"type": "error", "message": MSG_BUSY})
                return
            if app.state.llm is None:
                await send({"type": "error", "message": MSG_NO_KEY})
                return
            start_turn(
                run_text_turn(
                    text,
                    history,
                    app.state.llm,
                    app.state.usage,
                    send,
                    history_turns=settings.llm.history_turns,
                    max_output_tokens=settings.llm.max_output_tokens,
                    turn_timeout_s=settings.llm.turn_timeout_s,
                    tts=app.state.tts,
                    looker=app.state.looker,
                )
            )

        async def handle_audio_start() -> None:
            async with recording_lock:
                await _handle_audio_start_locked()

        async def _handle_audio_start_locked() -> None:
            nonlocal recording, too_long_sent
            if app.state.stt is None:
                await send({"type": "error", "message": MSG_NO_VOICE})
                return
            if recording is not None or turn_running():
                await send({"type": "error", "message": MSG_BUSY})
                return
            if app.state.llm is None:
                await send({"type": "error", "message": MSG_NO_KEY})
                return
            recording = bytearray()
            too_long_sent = False
            start_recording_timer(recording)
            await send({"type": "state", "state": "listening"})

        def on_timer_done(task: asyncio.Task[None]) -> None:
            nonlocal recording_timer
            timers.discard(task)
            if recording_timer is task:
                recording_timer = None

        def start_recording_timer(rec: bytearray) -> None:
            nonlocal recording_timer
            cancel_recording_timer()
            task = asyncio.create_task(recording_watchdog(rec))
            timers.add(task)
            task.add_done_callback(on_timer_done)
            recording_timer = task

        def cancel_recording_timer() -> None:
            # The watchdog closes its own recording through
            # _end_recording_locked, which lands here: never cancel the running
            # task from inside. on_timer_done clears the reference when it ends.
            if recording_timer is not None and recording_timer is not asyncio.current_task():
                recording_timer.cancel()

        async def recording_watchdog(rec: bytearray) -> None:
            await _watchdog_sleep(recording_limit_s)
            try:
                async with recording_lock:
                    if recording is not rec:
                        return  # that recording already ended
                    log.warning(
                        "recording: no audio_end within %g s, closing it", recording_limit_s
                    )
                    await _end_recording_locked()
            except Exception as exc:  # noqa: BLE001 - socket gone; the loop cleans up
                log.debug("recording watchdog: %s", type(exc).__name__)

        async def handle_audio_frame(frame: bytes) -> None:
            async with recording_lock:
                await _handle_audio_frame_locked(frame)

        async def _handle_audio_frame_locked(frame: bytes) -> None:
            nonlocal too_long_sent, stray_frame_sent
            if recording is None:
                if not stray_frame_sent:
                    stray_frame_sent = True
                    await send({"type": "error", "message": MSG_UNEXPECTED_AUDIO})
                return
            room = max_audio_bytes - len(recording)
            if len(frame) <= room:
                recording.extend(frame)
                return
            if room > 0:
                recording.extend(memoryview(frame)[:room])
            if not too_long_sent:
                too_long_sent = True
                log.warning(
                    "recording longer than %g s, the rest is dropped", stt_cfg.max_utterance_s
                )
                await send({"type": "error", "message": msg_too_long_audio_text})

        async def handle_audio_end() -> None:
            async with recording_lock:
                await _end_recording_locked()

        async def _end_recording_locked() -> None:
            # Caller holds recording_lock. The voice turn is started as a task,
            # so the lock is not held while it runs.
            nonlocal recording
            if recording is None:
                await send({"type": "error", "message": MSG_NOT_RECORDING})
                return
            pcm16 = bytes(recording)
            recording = None
            cancel_recording_timer()
            if len(pcm16) < min_audio_bytes:
                log.info(
                    "recording too short: %d bytes (min %d ms)",
                    len(pcm16),
                    stt_cfg.min_utterance_ms,
                )
                await send({"type": "error", "message": MSG_NOT_CAUGHT})
                await send({"type": "state", "state": "idle"})
                return
            start_turn(
                run_voice_turn(
                    pcm16,
                    history,
                    app.state.llm,
                    app.state.usage,
                    send,
                    app.state.stt,
                    language=stt_cfg.language_or_none,
                    sample_rate=STT_SAMPLE_RATE,
                    history_turns=settings.llm.history_turns,
                    max_output_tokens=settings.llm.max_output_tokens,
                    turn_timeout_s=settings.llm.turn_timeout_s,
                    tts=app.state.tts,
                    looker=app.state.looker,
                )
            )

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                frame = message.get("bytes")
                if frame is not None:
                    await handle_audio_frame(frame)
                    continue
                data = message.get("text")
                if data is None:
                    continue
                try:
                    payload = json.loads(data)
                except json.JSONDecodeError:
                    await send({"type": "error", "message": MSG_INVALID_JSON})
                    continue
                kind = payload.get("type") if isinstance(payload, dict) else None
                if kind == "text":
                    await handle_text(payload)
                elif kind == "ping":
                    continue
                elif kind == "audio_start":
                    await handle_audio_start()
                elif kind == "audio_end":
                    await handle_audio_end()
                else:
                    await send({"type": "error", "message": MSG_UNKNOWN})
        except WebSocketDisconnect:
            pass
        finally:
            # An open recording is dropped with the connection (never transcribed).
            recording = None
            for task in list(timers):
                task.cancel()
                try:
                    await task
                except BaseException:  # noqa: BLE001 - cancelled on close
                    pass
            if turn is not None and not turn.done():
                turn.cancel()
                try:
                    await turn
                except BaseException:  # noqa: BLE001 - cancelled or failed on close
                    pass
        log.info("ws disconnected")

    app.mount("/static", StaticFiles(directory=web_dir), name="static")
    return app
