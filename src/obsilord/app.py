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

Chat LLM: Gemini, or a local OpenAI-compatible server ([llm_local]). Both are
set up at startup (the local one connects only when used). Each connection
starts on [llm] provider and the page's Settings panel switches it with
{"type": "llm", "provider": "gemini"|"local", "model": "...", "announce":
bool}: refused while a turn or recording runs; answered with a
{"type": "notice"} (what changes and what may work worse) and, for a local
model, {"type": "llm_status"} while the server loads it. announce: true (sent
again after every reconnect) applies the choice quietly. GET /config/llm lists
both providers and the local server's models. The local server address is only
read from config.

Memory ([memory], memory/): every finished turn is stored in a local SQLite
file. On connect the page gets {"type": "conversation", "id", "title",
"turns", "last_at", "resumed", "reason"} (see conversation_message; it is
sent again after each stored turn, reason "saved"): the last web conversation is continued (its
last history_turns turns loaded) when its last turn is at most
resume_minutes old, otherwise a new one starts (its row is written with the
first turn). {"type": "conversation", "id": "<id>" | null} continues another
conversation or starts a new one (refused while a turn or recording runs).
GET /conversations lists them for the Settings panel; GET /conversations/{id}
returns one with its turns and messages (read only). The memory editor sends
{"type": "memory", "action": "edit" | "delete_turn" | "delete_conversation" |
"rename" | "add_turn", ...} (fields: message, text, conversation, turn,
title, user, reply; text capped at MAX_EDIT_CHARS); the reply is {"type":
"memory", "action", "conversation"}, plus a conversation message (reason
"edited", or "new" when the open one was deleted) when the open conversation
changed. Refused while a turn runs in the open conversation. recall() is declared
for the local model only. The store is opened on first use; if it cannot be
opened, memory is off for the session and everything else works. With
[debug] demo_memory the store is a separate demo file, deleted and seeded
with made-up history at startup, and GET /debug/memory-checks lists what to
ask (404 otherwise).

/ws accepts only this app's own page as Origin (or no Origin header), so
another site open in the browser cannot drive the socket. Every route answers
only to Host 127.0.0.1 or localhost (TrustedHostMiddleware), against DNS
rebinding.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Coroutine, TypeVar

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from obsilord import promptlog
from obsilord.config import Settings, memory_db_path, photos_dir_path
from obsilord.face_character import check_character
from obsilord.llm.base import LLMClient, Message, ToolNote
from obsilord.llm.openai_compat import LocalLLMError, OpenAICompatServer
from obsilord.llm_switch import GEMINI, LOCAL, clean_model, switch_notice
from obsilord.look import SEARCH_IMAGE_TOOL, ImageSearch, LookTool
from obsilord.memory.base import ConversationInfo, MemoryStore, TurnRecord
from obsilord.memory.demo import demo_checks, reset_demo_db, seed_demo
from obsilord.memory.recall import RecallTool, notes_to_store
from obsilord.memory.store import SqliteStore
from obsilord.photos import PHOTO_MODES, PhotoStore, TurnPhotos, attach_photos, keep_newest_photos
from obsilord.pipeline import MSG_NOT_CAUGHT, run_text_turn, run_voice_turn
from obsilord.tools import Tool
from obsilord.stt.base import SAMPLE_RATE as STT_SAMPLE_RATE
from obsilord.stt.base import STTClient, pcm16_bytes_for_ms
from obsilord.tts.base import TTSClient
from obsilord.usage import UsageTracker

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 2000

MSG_INVALID_JSON = "invalid JSON"
MSG_NO_KEY = "No API key configured. Pick Local in Settings to use a local model."
MSG_SWITCH_BUSY = "Wait for the current reply before switching the model."
MSG_BAD_PROVIDER = "Unknown provider."
MSG_BAD_MODEL = "Invalid model name."
MSG_GEMINI_OFF = "Gemini is not available: GEMINI_API_KEY is not set in .env."
MSG_BUSY = "Busy, wait for the current reply."
MSG_EMPTY = "Message is empty."
MSG_TOO_LONG = f"Message is too long (max {MAX_TEXT_CHARS} characters)."
MSG_NO_VOICE = "Voice input is not available."
MSG_UNKNOWN = "Unknown message type."
MSG_UNEXPECTED_AUDIO = "Unexpected audio frame."
MSG_NOT_RECORDING = "No recording in progress."
MSG_CONVERSATION_BUSY = "Wait for the current reply before switching the conversation."
MSG_MEMORY_OFF = "Conversation memory is off."
MSG_UNKNOWN_CONVERSATION = "Unknown conversation."
MAX_SQL_INT = 2**63 - 1  # SQLite integers are 64-bit signed
MSG_MEMORY_BAD = "That memory change was not understood."
MSG_MEMORY_MISSING = "That conversation or message no longer exists."
MAX_EDIT_CHARS = 4000
MSG_BAD_PHOTOS = "Unknown Photos setting."

CHANNEL = "web"
_CONVERSATION_ID = re.compile(r"[0-9a-f]{32}")

T = TypeVar("T")

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


# The page and its scripts are checked with the server on every load (the
# browser still gets a quick 304 when a file is unchanged). Without this a
# browser keeps old scripts for days after an update, next to a new
# index.html.
NO_CACHE = {"Cache-Control": "no-cache"}


class _RevalidatedStatic(StaticFiles):
    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers.update(NO_CACHE)
        return response


def _default_llm(settings: Settings) -> LLMClient | None:
    # Imported here so the provider SDK is only loaded when actually used.
    from obsilord.llm.gemini import GeminiClient, MissingApiKey

    try:
        return GeminiClient(settings.llm, settings.secrets.gemini_api_key)
    except MissingApiKey:
        log.warning("no GEMINI_API_KEY configured; Gemini turns will return an error")
        return None


def allowed_origins(settings: Settings) -> frozenset[str]:
    """The page's own origins. /ws refuses any other Origin header."""
    port = settings.server.port
    return frozenset({f"http://127.0.0.1:{port}", f"http://localhost:{port}"})


# Host names the app answers to. Any other Host header (a DNS rebinding page
# under its own domain) gets 400. "testserver" is the name Starlette's
# TestClient sends; it is not a public domain, so it cannot be rebound.
ALLOWED_HOSTS = ["127.0.0.1", "localhost", "testserver"]


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


def _default_image_search(
    settings: Settings, usage: UsageTracker | None = None
) -> ImageSearch | None:
    """What search_image() needs, from [image_search], or None when it is off
    or has no API key (the tool is then not declared). usage: the token
    tracker the "gemini" provider records its requests in."""
    cfg = settings.image_search
    if not cfg.enabled:
        log.info("image search: disabled ([image_search] enabled = false)")
        return None
    gemini = cfg.provider == "gemini"
    key_name = "GEMINI_API_KEY" if gemini else "GOOGLE_VISION_API_KEY"
    key = settings.secrets.gemini_api_key if gemini else settings.secrets.google_vision_api_key
    if not key:
        log.warning(
            "image search: disabled, %s is not set in .env ([image_search] enabled = true)",
            key_name,
        )
        return None
    from obsilord.imagesearch.base import ImageSearchClient
    from obsilord.imagesearch.budget import SearchBudget
    from obsilord.look import ImageSearch

    client: ImageSearchClient
    if gemini:
        from obsilord.imagesearch.gemini import GeminiImageIdentify

        if usage is None:
            usage = _default_usage(settings)
        client = GeminiImageIdentify(key, settings.llm.model, usage, cfg.timeout_s)
    else:
        from obsilord.imagesearch.google_vision import GoogleVisionSearch

        client = GoogleVisionSearch(key, cfg.max_results, cfg.timeout_s)

    budget = SearchBudget(
        settings.repo_root / "logs" / "image_search.jsonl", cfg.max_per_hour, cfg.max_per_month
    )
    log.warning(
        "image search: ENABLED (%s). search_image() uploads a crop of a camera "
        "frame to the provider. Caps: %d per hour, %d per month (%d used this month)",
        cfg.provider,
        cfg.max_per_hour,
        cfg.max_per_month,
        budget.month_total(),
    )
    return ImageSearch(client, budget, cfg.max_crop_fraction)


def _default_usage(settings: Settings) -> UsageTracker:
    return UsageTracker(
        settings.repo_root / "logs" / "usage.jsonl", settings.llm.daily_token_cap
    )


def _default_looker(settings: Settings, usage: UsageTracker | None = None) -> LookTool | None:
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
        _default_image_search(settings, usage),
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


def _memory_from_settings(settings: Settings) -> SqliteStore | None:
    """The conversation store from [memory], not opened yet. With [debug]
    demo_memory: the demo file, deleted here so it is seeded fresh."""
    demo = settings.debug.demo_memory
    if not settings.memory.enabled and not demo:
        log.info("memory: off ([memory] enabled = false)")
        return None
    path = memory_db_path(settings)
    if demo:
        reset_demo_db(path)
        log.warning("memory: demo memory on, made-up history in %s (the real store is not used)", path)
    return SqliteStore(path)


# Looked up at call time, so tests can turn the default store off.
_default_memory = _memory_from_settings


def turn_tools(provider: str, recall: T | None) -> list[T]:
    """The tools besides the camera for a turn: recall for the local model
    only, so earlier conversations never go to a cloud provider through it."""
    return [recall] if provider == LOCAL and recall is not None else []


CONVERSATION_REASONS = ("start", "resume", "switch", "new", "saved", "edited")


def conversation_message(info: ConversationInfo | None, reason: str) -> dict[str, Any]:
    """Which conversation the connection is in. reason: start (connected,
    nothing recent to continue), resume (continued on connect), switch (the
    page picked one), new (the page started one), saved (a turn was stored),
    edited (the memory editor changed the open conversation)."""
    assert reason in CONVERSATION_REASONS
    msg: dict[str, Any] = {"type": "conversation", "id": None, "title": "", "turns": 0, "last_at": None}
    if info is not None:
        msg.update(id=info.id, title=info.title, turns=info.turns, last_at=info.last_at)
    msg.update(resumed=reason == "resume", reason=reason)
    return msg


def create_app(
    settings: Settings,
    llm: LLMClient | None = None,
    usage: UsageTracker | None = None,
    tts: TTSClient | None = None,
    stt: STTClient | None = None,
    looker: LookTool | None = None,
    local: OpenAICompatServer | None = None,
    memory: MemoryStore | None = None,
    photos: PhotoStore | None = None,
) -> FastAPI:
    """Build the app. Pass llm, usage, tts, stt, looker, local, memory, and
    photos to inject fakes (tests). llm is the Gemini client; local the local
    LLM server; memory the conversation store (None builds it from [memory]);
    photos the folder for Photos "disk" (None: [camera] photos_dir).

    tts=None loads the voice from settings (nothing is loaded when [tts] voice
    is empty). stt=None loads the STT model from settings (nothing is loaded
    when [stt] model is empty). looker=None builds look() from settings (off
    when [vision] model is empty). local=None builds the server object from
    [llm_local] (no connection until it is used).
    """
    web_dir = settings.repo_root / "web"
    index_path = web_dir / "index.html"

    promptlog.set_enabled(settings.debug.log_prompts)
    if settings.debug.log_prompts:
        log.warning(
            "debug.log_prompts is on: every Gemini, local LLM and vision request "
            "and reply is printed (camera frames as their size only)"
        )
    # Cheap file check first, so a bad character fails before any model loads.
    character_path = _checked_character(settings)
    if llm is None:
        llm = _default_llm(settings)
    if tts is None:
        tts = _default_tts(settings)
    if stt is None:
        stt = _default_stt(settings)
    if usage is None:
        usage = _default_usage(settings)
    if looker is None:
        looker = _default_looker(settings, usage)
    if local is None:
        local = OpenAICompatServer(settings.llm_local)
    if memory is None:
        memory = _default_memory(settings)
    if photos is None:
        photos = PhotoStore(photos_dir_path(settings))
    photo_store = photos
    origins = allowed_origins(settings)
    log.info(
        "chat LLM: pages start on %s (Gemini %s, local server %s)",
        settings.llm.provider,
        "available" if llm is not None else "off, no key",
        settings.llm_local.base_url,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Saved photos go with [memory] retention_days (0 keeps everything).
        try:
            removed = await photo_store.prune(settings.memory.retention_days)
            if removed:
                log.info("photos: deleted %d old photos", removed)
        except OSError as exc:
            log.warning("photos: could not clean up old photos (%s)", type(exc).__name__)
        yield
        for client in (app.state.llm, app.state.local):
            aclose = getattr(client, "aclose", None)
            if aclose is not None:
                await aclose()
        if memory_state["ready"] is not None:
            await memory_state["ready"].close()

    # The store is opened (and pruned, or seeded in demo mode) on first use,
    # under a lock. A failure turns memory off for the session.
    memory_state: dict[str, Any] = {"ready": None, "failed": memory is None, "demo_at": None}
    memory_lock = asyncio.Lock()

    async def get_memory() -> MemoryStore | None:
        if memory_state["ready"] is not None or memory_state["failed"]:
            return memory_state["ready"]
        async with memory_lock:
            if memory_state["ready"] is not None or memory_state["failed"]:
                return memory_state["ready"]
            assert memory is not None
            try:
                await memory.open()
                if settings.debug.demo_memory:
                    now = time.time()
                    await seed_demo(memory, now)
                    memory_state["demo_at"] = now
                else:
                    pruned = await memory.prune(settings.memory.retention_days)
                    if pruned:
                        log.info("memory: deleted %d old conversations", pruned)
            except Exception as exc:  # noqa: BLE001 - memory is optional
                log.warning("memory: store could not be opened (%s), memory is off", type(exc).__name__)
                memory_state["failed"] = True
                return None
            memory_state["ready"] = memory
            return memory

    app = FastAPI(
        title="Obsi-Lord", docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.state.settings = settings
    app.state.llm = llm
    app.state.usage = usage
    app.state.tts = tts
    app.state.stt = stt
    app.state.looker = looker
    app.state.local = local

    def search_declared() -> bool:
        tools = getattr(app.state.looker, "tools", None) or []
        return any(t.name == SEARCH_IMAGE_TOOL.name for t in tools)

    stt_cfg = settings.stt
    max_audio_bytes = pcm16_bytes_for_ms(stt_cfg.max_utterance_s * 1000, STT_SAMPLE_RATE)
    min_audio_bytes = pcm16_bytes_for_ms(stt_cfg.min_utterance_ms, STT_SAMPLE_RATE)
    msg_too_long_audio_text = msg_too_long_audio(stt_cfg.max_utterance_s)
    # How long a recording may stay open without audio_end.
    recording_limit_s = stt_cfg.max_utterance_s + RECORDING_GRACE_S

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(index_path, media_type="text/html", headers=NO_CACHE)

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

    @app.get("/config/photos", include_in_schema=False)
    async def photos_config() -> dict[str, Any]:
        # The Photos setting's default for a page that has not chosen one.
        return {"default": settings.camera.photos, "keep": settings.camera.photos_keep}

    @app.get("/config/llm", include_in_schema=False)
    async def llm_config() -> dict[str, Any]:
        # Both chat providers for the Settings panel. No secrets: only whether
        # the Gemini key is set. The local model list is fetched from the
        # server here, so the page never talks to it directly.
        models: list[dict[str, Any]] = []
        error: str | None = None
        try:
            models = [
                {"id": m.id, "loaded": m.loaded, "tools": m.tools, "vision": m.vision}
                for m in await app.state.local.list_models()
            ]
        except LocalLLMError as exc:
            error = str(exc)
        return {
            "default_provider": settings.llm.provider,
            "gemini": {"available": app.state.llm is not None, "model": settings.llm.model},
            "local": {
                "base_url": settings.llm_local.base_url,
                "default_model": settings.llm_local.model,
                "reachable": error is None,
                "error": error,
                "models": models,
            },
        }

    @app.get("/conversations", include_in_schema=False)
    async def conversations(limit: int = 0) -> dict[str, Any]:
        # Titles and dates for the Settings list, newest first. Local page only
        # (TrustedHost, 127.0.0.1); no message text besides the first line.
        store = await get_memory()
        if store is None:
            return {"enabled": False, "conversations": []}
        cap = settings.memory.list_limit
        infos = await store.list_conversations(min(limit, cap) if limit > 0 else cap)
        return {
            "enabled": True,
            "conversations": [
                {"id": i.id, "title": i.title, "started_at": i.started_at, "last_at": i.last_at, "turns": i.turns}
                for i in infos
            ],
        }

    @app.get("/conversations/{cid}", include_in_schema=False)
    async def conversation_turns(cid: str) -> dict[str, Any]:
        # One conversation's turns and messages for the memory editor (read only;
        # changes go over /ws). Local page only, like the list above.
        store = await get_memory()
        if store is None:
            return {"enabled": False, "id": "", "title": "", "turns": []}
        info = await store.info(cid) if _CONVERSATION_ID.fullmatch(cid) else None
        if info is None:
            raise HTTPException(status_code=404)
        turns = await store.turns(cid)
        return {
            "enabled": True,
            "id": info.id,
            "title": info.title,
            "turns": [
                {
                    "turn": t.turn,
                    "ts": t.ts,
                    "messages": [
                        {"id": m.id, "role": m.role, "text": m.text, "tool_name": m.tool_name, "tool_arg": m.tool_arg}
                        for m in t.messages
                    ],
                }
                for t in turns
            ],
        }

    @app.get("/debug/memory-checks", include_in_schema=False)
    async def memory_checks() -> dict[str, Any]:
        if not settings.debug.demo_memory:
            raise HTTPException(status_code=404)
        await get_memory()
        if memory_state["demo_at"] is None:
            raise HTTPException(status_code=404)
        return {"checks": demo_checks(memory_state["demo_at"])}

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin is not None and origin not in origins:
            # Another site open in the browser: refused before accept (HTTP 403).
            log.warning("ws refused: foreign origin")
            await websocket.close(code=1008)
            return
        await websocket.accept()
        log.info("ws connected")
        history: list[Message] = []
        # The chat LLM of this connection (Settings panel). local_client is
        # rebuilt when the page picks another local model.
        provider = settings.llm.provider
        local_client: LLMClient = app.state.local.client(settings.llm_local.model)
        background: set[asyncio.Task[None]] = set()  # model warm-ups
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
        # Set by the page's Snapshots switch ({"type": "snapshots", "on": true}).
        # While off, no frame or crop is sent to the page.
        snapshots_on = False
        # Conversation memory: the store (None when off), this connection's
        # conversation (None until its first turn is stored), and recall().
        store = await get_memory()
        conv_id: str | None = None
        recall: RecallTool | None = None
        if store is not None and settings.memory.recall:
            recall = RecallTool(store, settings.memory.recall_limit, settings.memory.recall_max_chars)

        # The Photos setting for this page ({"type": "photos", "mode": ...});
        # [camera] photos until the page says.
        photo_mode = settings.camera.photos
        photo_keep = settings.camera.photos_keep

        async def save_photo(jpeg: bytes) -> str:
            """Photos "disk": write a look's frame next to its conversation,
            creating the conversation row if this is its first turn."""
            nonlocal conv_id
            if store is not None and conv_id is None:
                conv_id = await store.new_conversation(CHANNEL)
            return await photo_store.save(conv_id, jpeg)

        def turn_photos() -> TurnPhotos:
            return TurnPhotos(photo_mode, photo_keep, save_photo if photo_mode == "disk" else None)  # type: ignore[arg-type]

        async def load_photos() -> None:
            """Photos "disk": give the history its saved frames back (a resumed
            conversation). Called right before a turn starts, under
            recording_lock, so no turn changes history meanwhile; frames that
            are already loaded are not read again."""
            if photo_mode != "disk" or conv_id is None:
                return
            try:
                await attach_photos(history, photo_store, conv_id, photo_keep)
            except Exception as exc:  # noqa: BLE001 - photos are optional
                log.warning("photos: could not load saved photos (%s)", type(exc).__name__)

        async def handle_photos(payload: dict[str, Any]) -> None:
            nonlocal photo_mode
            async with recording_lock:
                mode = payload.get("mode")
                if mode not in PHOTO_MODES:
                    await send({"type": "error", "message": MSG_BAD_PHOTOS})
                    return
                photo_mode = mode
                # A running turn may be changing history: then the next turn
                # drops the photos itself (pipeline), and saved ones load
                # before the next turn (load_photos).
                if mode == "off" and not turn_running():
                    keep_newest_photos(history, 0)
                await send({"type": "photos", "mode": photo_mode})

        def recall_tool() -> Tool | None:
            # Built per turn: its description carries today's date, and the
            # turns of this conversation already in history are skipped.
            if recall is None:
                return None
            skip = (conv_id, settings.llm.history_turns) if conv_id is not None else None
            return recall.tool(skip=skip)

        async def send(msg: dict[str, Any] | bytes) -> None:
            if isinstance(msg, dict) and msg.get("type") == "snapshot" and not snapshots_on:
                return
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

        def current_llm() -> LLMClient | None:
            return local_client if provider == LOCAL else app.state.llm

        def current_max_output_tokens() -> int:
            # Used by the pipeline for the usage estimate of a failed request.
            if provider == LOCAL:
                return settings.llm_local.max_output_tokens
            return settings.llm.max_output_tokens

        async def handle_llm(payload: dict[str, Any]) -> None:
            async with recording_lock:
                await _handle_llm_locked(payload)

        async def _handle_llm_locked(payload: dict[str, Any]) -> None:
            nonlocal provider, local_client
            wanted = payload.get("provider")
            model = clean_model(payload.get("model", ""))
            announce = payload.get("announce") is True
            if wanted not in (GEMINI, LOCAL):
                await send({"type": "error", "message": MSG_BAD_PROVIDER})
                return
            if model is None:
                await send({"type": "error", "message": MSG_BAD_MODEL})
                return
            if recording is not None or turn_running():
                # An announce comes from the page itself (reconnect); it is
                # sent again on the next connect, so it is dropped quietly.
                if not announce:
                    await send({"type": "error", "message": MSG_SWITCH_BUSY})
                return
            if wanted == GEMINI:
                if app.state.llm is None:
                    if not announce:
                        await send({"type": "error", "message": MSG_GEMINI_OFF})
                    return
                provider = GEMINI
            else:
                provider = LOCAL
                local_client = app.state.local.client(model)
            log.info("chat LLM for this page: %s%s", provider, f" ({model})" if model else "")
            if announce:
                return
            await send(
                switch_notice(
                    provider,
                    settings,
                    model=model,
                    vision_on=app.state.looker is not None,
                    search_on=search_declared(),
                    today_tokens=app.state.usage.today_total,
                    # From the model list the page loaded; never a request here.
                    tool_mode=app.state.local.known_tool_mode(model),
                )
            )
            if provider == LOCAL:
                task = asyncio.create_task(warm_up(model))
                background.add(task)
                task.add_done_callback(background.discard)

        async def warm_up(model: str) -> None:
            try:
                await _warm_up(model)
            except Exception as exc:  # noqa: BLE001 - socket gone; nothing to tell
                log.debug("warm-up: %s", type(exc).__name__)

        async def _warm_up(model: str) -> None:
            """Load the model now and tell the page how long it took. A failed
            warm-up is only reported: the next turn gets its own error. An
            unknown model id shows up here first."""
            await send({"type": "llm_status", "state": "loading", "model": model})
            try:
                if not model:
                    model = await app.state.local.default_model()
                seconds, counts = await app.state.local.warm_up(model)
            except LocalLLMError as exc:
                log.warning("local LLM warm-up failed: %s", exc)
                await send(
                    {"type": "llm_status", "state": "failed", "model": model, "message": str(exc)}
                )
                return
            if counts is not None:
                app.state.usage.record(
                    "local:" + model,
                    counts.tokens_in,
                    counts.tokens_out,
                    counts.tokens_thinking,
                    int(seconds * 1000),
                    counted=False,
                )
            log.info("local LLM %s ready in %.1f s", model, seconds)
            await send(
                {"type": "llm_status", "state": "ready", "model": model, "seconds": round(seconds, 1)}
            )

        def start_turn(coro: Coroutine[Any, Any, str | None]) -> None:
            nonlocal turn
            turn = asyncio.create_task(coro)
            turn.add_done_callback(on_turn_done)

        def store_turn(
            turn_provider: str, llm: LLMClient
        ) -> Callable[[str, str, tuple[ToolNote, ...]], Awaitable[None]] | None:
            """The pipeline's on_turn: writes the finished turn, creating the
            conversation row with the first one."""
            if store is None:
                return None

            async def on_turn(user: str, reply: str, notes: tuple[ToolNote, ...]) -> None:
                nonlocal conv_id
                if conv_id is None:
                    conv_id = await store.new_conversation(CHANNEL)
                record = TurnRecord(user, reply, notes_to_store(notes), turn_provider, llm.model, time.time())
                try:
                    await store.add_turn(conv_id, record)
                except KeyError:
                    # Deleted meanwhile (another page): keep the turn in a new one.
                    conv_id = await store.new_conversation(CHANNEL)
                    await store.add_turn(conv_id, record)
                # The page keeps the id (a reconnect returns to it) and the count.
                info = await store.info(conv_id)
                try:
                    await send(conversation_message(info, "saved"))
                except Exception as exc:  # noqa: BLE001 - the turn is stored; the page is gone
                    log.debug("conversation saved message: %s", type(exc).__name__)

            return on_turn

        async def start_new() -> None:
            nonlocal conv_id
            conv_id = None
            history.clear()
            log.info("conversation: new")
            await send(conversation_message(None, "new"))

        async def handle_memory(payload: dict[str, Any]) -> None:
            async with recording_lock:
                await _handle_memory_locked(payload)

        async def drop_photos(action: str, cid: str, names: list[str]) -> None:
            """Remove the saved photos of what the editor deleted."""
            try:
                if action == "delete_conversation" or (
                    action == "delete_turn" and store is not None and await store.info(cid) is None
                ):
                    removed = await photo_store.delete_conversation(cid)
                elif action == "delete_turn":
                    removed = await photo_store.delete_files(cid, names)
                else:
                    return
                log.debug("memory: removed %d photos", removed)
            except Exception as exc:  # noqa: BLE001 - the change is stored; photos are cleaned up later
                log.warning("photos: could not remove photos (%s)", type(exc).__name__)

        async def _handle_memory_locked(payload: dict[str, Any]) -> None:
            """The memory editor's changes (edit, delete, rename, add a turn).
            Never logs message text, only the action."""
            if store is None:
                await send({"type": "error", "message": MSG_MEMORY_OFF})
                return
            action = payload.get("action")
            bad = {"type": "error", "message": MSG_MEMORY_BAD}
            missing = {"type": "error", "message": MSG_MEMORY_MISSING}

            def text_field(name: str) -> str | None:
                v = payload.get(name)
                if not isinstance(v, str) or len(v) > MAX_EDIT_CHARS or not v.strip():
                    return None
                return v

            def int_field(name: str) -> int | None:
                v = payload.get(name)
                if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= MAX_SQL_INT:
                    return None
                return v

            def conv_field() -> str | None:
                v = payload.get("conversation")
                return v if isinstance(v, str) and _CONVERSATION_ID.fullmatch(v) else None

            message_id: int | None = None
            turn_no: int | None = None
            cid: str | None = None
            text: str | None = None
            title: str | None = None
            user: str | None = None
            reply: str | None = None
            if action == "edit":
                message_id, text = int_field("message"), text_field("text")
                if message_id is None or text is None:
                    await send(bad)
                    return
                cid = await store.message_conversation(message_id)
            elif action == "delete_turn":
                cid, turn_no = conv_field(), int_field("turn")
                if cid is None or turn_no is None:
                    await send(bad)
                    return
            elif action == "delete_conversation":
                cid = conv_field()
                if cid is None:
                    await send(bad)
                    return
            elif action == "rename":
                cid, title = conv_field(), text_field("title")
                if cid is None or title is None:
                    await send(bad)
                    return
            elif action == "add_turn":
                cid, user, reply = conv_field(), text_field("user"), text_field("reply")
                if cid is None or user is None or reply is None:
                    await send(bad)
                    return
            else:
                await send(bad)
                return
            if cid is None or await store.info(cid) is None:
                await send(missing)
                return
            is_open = cid == conv_id
            if is_open and (recording is not None or turn_running()):
                await send({"type": "error", "message": MSG_CONVERSATION_BUSY})
                return
            ok = True
            gone_photos: list[str] = []
            if action == "delete_turn":
                assert turn_no is not None
                gone_photos = await store.turn_photos(cid, turn_no)
            if action == "edit":
                assert message_id is not None and text is not None
                ok = await store.edit_message(message_id, text) is not None
            elif action == "delete_turn":
                assert turn_no is not None
                ok = await store.delete_turn(cid, turn_no)
            elif action == "delete_conversation":
                ok = await store.delete_conversation(cid)
            elif action == "rename":
                assert title is not None
                ok = await store.rename(cid, title)
            else:
                assert user is not None and reply is not None
                # Keep the conversation's place in the list (and resume).
                existing = await store.info(cid)
                stamp = existing.last_at if existing is not None else time.time()
                await store.add_turn(cid, TurnRecord(user, reply, (), "edited", "", stamp))
            if not ok:
                await send(bad if action == "rename" else missing)
                return
            await drop_photos(action, cid, gone_photos)
            log.info("memory: %s", action)
            await send({"type": "memory", "action": action, "conversation": cid})
            if not is_open:
                return
            info = await store.info(cid)
            if info is None:
                await start_new()
                return
            history[:] = await store.last_turns(cid, settings.llm.history_turns)
            await send(conversation_message(info, "edited"))

        async def handle_conversation(payload: dict[str, Any]) -> None:
            async with recording_lock:
                await _handle_conversation_locked(payload)

        async def _handle_conversation_locked(payload: dict[str, Any]) -> None:
            nonlocal conv_id
            if store is None:
                await send({"type": "error", "message": MSG_MEMORY_OFF})
                return
            if recording is not None or turn_running():
                await send({"type": "error", "message": MSG_CONVERSATION_BUSY})
                return
            wanted = payload.get("id")
            if wanted is None:
                await start_new()
                return
            info = None
            if isinstance(wanted, str) and _CONVERSATION_ID.fullmatch(wanted):
                info = await store.info(wanted)
            if info is None:
                await send({"type": "error", "message": MSG_UNKNOWN_CONVERSATION})
                return
            conv_id = info.id
            history[:] = await store.last_turns(info.id, settings.llm.history_turns)
            log.info("conversation: continued one with %d turns", info.turns)
            await send(conversation_message(info, "switch"))

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
            llm = current_llm()
            if llm is None:
                await send({"type": "error", "message": MSG_NO_KEY})
                return
            await load_photos()
            start_turn(
                run_text_turn(
                    text,
                    history,
                    llm,
                    app.state.usage,
                    send,
                    history_turns=settings.llm.history_turns,
                    max_output_tokens=current_max_output_tokens(),
                    turn_timeout_s=settings.llm.turn_timeout_s,
                    tts=app.state.tts,
                    looker=app.state.looker,
                    extra_tools=turn_tools(provider, recall_tool()),
                    on_turn=store_turn(provider, llm),
                    photos=turn_photos(),
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
            if current_llm() is None:
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
            llm = current_llm()
            if llm is None:  # switched to Gemini without a key mid-recording
                await send({"type": "error", "message": MSG_NO_KEY})
                await send({"type": "state", "state": "idle"})
                return
            await load_photos()
            start_turn(
                run_voice_turn(
                    pcm16,
                    history,
                    llm,
                    app.state.usage,
                    send,
                    app.state.stt,
                    language=stt_cfg.language_or_none,
                    sample_rate=STT_SAMPLE_RATE,
                    history_turns=settings.llm.history_turns,
                    max_output_tokens=current_max_output_tokens(),
                    turn_timeout_s=settings.llm.turn_timeout_s,
                    tts=app.state.tts,
                    looker=app.state.looker,
                    extra_tools=turn_tools(provider, recall_tool()),
                    on_turn=store_turn(provider, llm),
                    photos=turn_photos(),
                )
            )

        async def greet() -> None:
            """Continue the last conversation if it is recent, and tell the page
            which conversation this connection is in."""
            nonlocal conv_id, store
            if store is None:
                return
            info = None
            try:
                if settings.memory.resume_minutes > 0:
                    cid = await store.latest(CHANNEL, settings.memory.resume_minutes * 60)
                    info = await store.info(cid) if cid is not None else None
                if info is not None:
                    history[:] = await store.last_turns(info.id, settings.llm.history_turns)
                    conv_id = info.id
                    log.info("conversation: resumed one with %d turns", info.turns)
            except Exception as exc:  # noqa: BLE001 - memory is optional
                log.warning("memory: could not resume (%s), memory off for this page", type(exc).__name__)
                store = None
                history.clear()
                return
            await send(conversation_message(info, "resume" if info is not None else "start"))

        try:
            await greet()
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
                elif kind == "snapshots":
                    snapshots_on = payload.get("on") is True
                elif kind == "photos":
                    await handle_photos(payload)
                elif kind == "llm":
                    await handle_llm(payload)
                elif kind == "conversation":
                    await handle_conversation(payload)
                elif kind == "memory":
                    await handle_memory(payload)
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
            for task in [*timers, *background]:
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

    app.mount("/static", _RevalidatedStatic(directory=web_dir), name="static")
    return app
