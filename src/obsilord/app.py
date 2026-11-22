"""FastAPI app: serves the web page and the /ws WebSocket.

{"type": "text"} runs a streamed LLM turn with spoken output (see
pipeline.py). Push-to-talk voice input: {"type": "audio_start"}, binary PCM16
mono little-endian frames at 16 kHz, then {"type": "audio_end"}, which runs a
voice turn (STT, then the same turn body). Invalid JSON gets an error message
and the connection stays open.

Per connection the socket is idle, recording, or running a turn:
- audio_start: rejected with an error (no state message) if voice input is
  disabled, or a turn is running or a recording is open. Otherwise the buffer is cleared and state listening is sent.
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
create_app (app startup) and can be injected for tests. Without a Gemini
key the app still starts, pages start on the local model, and switching to
Gemini is refused. If [tts] model is
empty, TTS is disabled and replies are text only; if [stt] model is empty,
voice input is disabled. A configured voice or STT model that fails to load
(missing file, unknown voice, failed download) stops startup with an error.
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
{"type": "memory", "action": "save", "conversation", "expect_turns",
"title"?, "edits": [{"message", "text"}], "delete_turns": [turn],
"add_turns": [{"user", "reply"}]} (all changes in one transaction, at most
MAX_SAVE_EDITS / MAX_SAVE_DELETES / MAX_SAVE_ADDS of each, text capped at
MAX_EDIT_CHARS; refused with MSG_MEMORY_STALE when the conversation's turn
count is not expect_turns) or {"type": "memory", "action":
"delete_conversation", "conversation"}; the reply is {"type": "memory",
"action", "conversation"}, plus a conversation message (reason "edited", or
"new" when the open one is gone) when the open conversation changed. Refused
while a turn runs or audio records in the open conversation. recall() is declared
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
from typing import Any, AsyncIterator, Awaitable, Callable, Coroutine

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from obsilord import promptlog
from obsilord.activity import ActivityLog
from obsilord.config import (
    Settings,
    activity_log_path,
    memory_db_path,
    photos_dir_path,
    settings_file_path,
)
from obsilord.face_character import check_character
from obsilord.llm.base import LLMClient, Message, ToolNote
from obsilord.llm.openai_compat import LocalLLMError, OpenAICompatServer
from obsilord.llm_switch import GEMINI, LOCAL, clean_model, switch_notice
from obsilord.look import LOOK_TOOL, SEARCH_IMAGE_TOOL, ImageSearch, LookTool, camera_source
from obsilord.memory.base import ConversationInfo, MemoryStore, TurnRecord
from obsilord.memory.demo import demo_checks, reset_demo_db, seed_demo
from obsilord.memory.recall import RECALL_TOOL, RecallTool, notes_to_store
from obsilord.memory.store import SqliteStore
from obsilord.permissions import Choices, SettingsStore, ToolChoice
from obsilord.photos import PHOTO_MODES, PhotoStore, TurnPhotos, attach_photos, keep_newest_photos
from obsilord.pipeline import MSG_NOT_CAUGHT, ConfirmReply, Confirms, run_text_turn, run_voice_turn
from obsilord.stt.base import SAMPLE_RATE as STT_SAMPLE_RATE
from obsilord.stt.base import STTClient, pcm16_bytes_for_ms
from obsilord.tools.calculate import calculate
from obsilord.tools.registry import (
    MODES,
    OnRun,
    Permissions,
    Registry,
    Tool,
    ToolSource,
    TurnInfo,
    static_source,
)
from obsilord.tts.base import TTSClient
from obsilord.usage import UsageTracker

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 2000

MSG_INVALID_JSON = "invalid JSON"
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
MAX_SAVE_EDITS = 500
MAX_SAVE_DELETES = 500
MAX_SAVE_ADDS = 50
MSG_MEMORY_STALE = "This conversation changed since you opened it. Reopen it to edit."
MSG_BAD_PHOTOS = "Unknown Photos setting."
MSG_UNKNOWN_TOOL = "Unknown tool."
MSG_BAD_PERMISSION = "That permission change was not understood."
MSG_NO_CLOUD_CHOICE = "That tool has no cloud model choice."
MSG_BAD_CONFIRM = "That answer was not understood."
MSG_NOT_SAVED = "That change could not be saved, so nothing changed."

# What a Permissions row says when its tool is not set up.
MISSING_CAMERA = "needs [camera]"
MISSING_LOOK = "needs [vision] for a chat model that cannot see images"
MISSING_SEARCH = "needs [vision], [image_search] and its key"
# The look row's note: which path the page's current chat model uses.
NOTE_LOOK_SEES = "Your chat model looks at the photo itself"
NOTE_LOOK_QWENNY = "Qwenny looks and tells your chat model"
MISSING_RECALL = "needs [memory] with recall on"
# GET /activity: the newest entries shown in Settings.
ACTIVITY_LIMIT = 50

CHANNEL = "web"
_CONVERSATION_ID = re.compile(r"[0-9a-f]{32}")


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
        log.info("tts: disabled ([tts] model is empty), replies are text only")
        return None
    # Imported here so the provider package is only loaded when used.
    from obsilord.tts.kokoro import KokoroTTS

    cfg = settings.tts
    tts = KokoroTTS.load(settings.repo_root / cfg.model, cfg.voice, cfg.speed, cfg.pitch)
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
    """The camera tools' Looker. The camera is always built ([camera] is
    checked when the config loads); Qwenny only when [vision] model is set and
    its files exist. A [vision] model whose files are missing is one log line
    and the Looker runs without Qwenny: a chat model that sees photos still
    gets look(). Nothing opens at startup."""
    # Imported here so OpenCV and the vision module are only loaded when used.
    from obsilord.camera import CameraService
    from obsilord.look import LookLimiter, Looker
    from obsilord.vision.llama_server import LlamaServerVision, check_files

    vision_cfg = settings.vision
    camera = settings.camera
    qwenny = None
    why = "[vision] model is empty"
    if vision_cfg.enabled:
        try:
            check_files(vision_cfg, settings.repo_root)
        except ValueError as exc:
            why = f'{exc}; run "python -m obsilord.vision.fetch" to get it'
        else:
            qwenny = LlamaServerVision(vision_cfg, settings.repo_root)
    looker = Looker(
        CameraService(camera),
        qwenny,
        LookLimiter(camera.cooldown_s, camera.max_per_hour),
        _default_image_search(settings, usage) if qwenny is not None else None,
    )
    log.info(
        "camera: photo for a chat model that sees images, available; "
        "look via Qwenny %s (camera %d via %s, cooldown %g s, %d per hour)",
        f"on (backend {vision_cfg.backend})" if qwenny is not None else f"off ({why})",
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


# The Permissions panel rows, in this order. look stands for both camera
# variants (Qwenny's and the photo one share its name and policy).
PANEL_TOOLS: tuple[Tool, ...] = (LOOK_TOOL, SEARCH_IMAGE_TOOL, RECALL_TOOL, calculate)
_PANEL_BY_NAME = {t.name: t for t in PANEL_TOOLS}


def tool_defaults() -> dict[str, ToolChoice]:
    """Every panel tool's choice until the user picks: from its policy."""
    return {t.name: ToolChoice(t.policy.default_mode, t.policy.cloud_default) for t in PANEL_TOOLS}


def _settings_store_from(settings: Settings) -> SettingsStore:
    return SettingsStore(settings_file_path(settings), tool_defaults(), settings.camera.photos)


def _activity_from(settings: Settings) -> ActivityLog:
    return ActivityLog(activity_log_path(settings), settings.memory.retention_days)


# Looked up at call time, so tests keep these files in their tmp_path.
_default_settings_store = _settings_store_from
_default_activity = _activity_from


def permissions_message(
    choices: Choices, missing: dict[str, str], notes: dict[str, str] | None = None
) -> dict[str, Any]:
    """The Permissions panel: one row per tool. missing: why a tool is not
    set up (no entry: it is). notes: a short line per tool, "" when none."""
    rows = []
    for t in PANEL_TOOLS:
        choice = choices.tools[t.name]
        why = missing.get(t.name, "")
        rows.append(
            {
                "name": t.name,
                "label": t.policy.label_text(t.name),
                "mode": choice.mode,
                "internet": t.policy.leaves_pc,
                "private": t.policy.private_data,
                "cloud": choice.cloud,
                "available": not why,
                "missing": why,
                "note": (notes or {}).get(t.name, ""),
            }
        )
    return {"type": "permissions", "tools": rows}


def build_registry(
    sources: list[ToolSource], permissions: Permissions | None = None, on_run: OnRun | None = None
) -> Registry:
    """The registry of one connection: the camera tools (when there is a
    camera), recall (when memory is on) and calculate. Which of them a turn
    offers is decided per turn by the registry (the user's modes, cloud model,
    photos)."""
    return Registry([*sources, static_source(calculate)], permissions=permissions, on_run=on_run)


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
    settings_store: SettingsStore | None = None,
    activity: ActivityLog | None = None,
) -> FastAPI:
    """Build the app. Pass llm, usage, tts, stt, looker, local, memory,
    photos, settings_store and activity to inject fakes (tests). llm is the
    Gemini client; local the local LLM server; memory the conversation store
    (None builds it from [memory]); photos the folder for Photos "disk" (None:
    [camera] photos_dir); settings_store the user's choices (None:
    [permissions] settings_path); activity the tool run log (None:
    [permissions] activity_path).

    tts=None loads the voice from settings (nothing is loaded when [tts] model
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
    if settings_store is None:
        settings_store = _default_settings_store(settings)
    choices_store = settings_store
    if activity is None:
        activity = _default_activity(settings)
    activity_log = activity
    origins = allowed_origins(settings)
    # Pages start on Gemini only when it is configured and has a key; the
    # local model otherwise, so no page starts on a provider that cannot answer.
    start_provider = GEMINI if settings.llm.provider == GEMINI and llm is not None else LOCAL
    log.info(
        "chat LLM: pages start on %s (Gemini %s, local server %s)",
        start_provider,
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
        # The activity log goes with [memory] retention_days too.
        await asyncio.to_thread(activity_log.prune)
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
    app.state.settings_store = choices_store
    app.state.activity = activity_log

    # One callback per open page: resend its Permissions rows when the local
    # model list was loaded (the look row depends on it). Sends only changes.
    row_refreshers: set[Callable[[], Awaitable[None]]] = set()

    def search_declared() -> bool:
        return bool(getattr(app.state.looker, "search_on", False))

    stt_cfg = settings.stt
    max_audio_bytes = pcm16_bytes_for_ms(stt_cfg.max_utterance_s * 1000, STT_SAMPLE_RATE)
    min_audio_bytes = pcm16_bytes_for_ms(stt_cfg.min_utterance_ms, STT_SAMPLE_RATE)
    msg_too_long_audio_text = msg_too_long_audio(stt_cfg.max_utterance_s)
    # How long a recording may stay open without audio_end.
    recording_limit_s = stt_cfg.max_utterance_s + RECORDING_GRACE_S

    @app.get("/", include_in_schema=False)
    async def index() -> Response:
        if not settings.face.demo:
            return FileResponse(index_path, media_type="text/html", headers=NO_CACHE)
        # [face] demo: mark the body in the served page so the first paint is
        # already the demo layout (the config fetch arrives later). Fixed
        # string replace, no user text involved.
        html = index_path.read_text(encoding="utf-8").replace(
            "<body>", '<body data-demo="true">', 1
        )
        return Response(html, media_type="text/html", headers=NO_CACHE)

    @app.get("/config/face", include_in_schema=False)
    async def face_config() -> dict[str, Any]:
        # Face settings only: no secrets and no other sections.
        face = settings.face
        return {
            "logical_width": face.logical_width,
            "logical_height": face.logical_height,
            "mouth_thresholds": list(face.mouth_thresholds),
            "skin": face.skin,
            "demo": face.demo,
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

    @app.get("/activity", include_in_schema=False)
    async def activity_entries() -> dict[str, Any]:
        # Settings, Activity: time, tool, mode and outcome only (no content).
        return {"entries": await asyncio.to_thread(activity_log.newest, ACTIVITY_LIMIT)}

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
        else:
            for refresh in list(row_refreshers):
                await refresh()
        return {
            "default_provider": start_provider,
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
        provider = start_provider
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

        # The Photos setting, kept by the app for every page (settings store);
        # [camera] photos until the user picks one.
        photo_keep = settings.camera.photos_keep

        def photo_mode() -> str:
            return choices_store.get().photos

        async def save_photo(jpeg: bytes) -> str:
            """Photos "disk": write a look's frame next to its conversation,
            creating the conversation row if this is its first turn."""
            nonlocal conv_id
            if store is not None and conv_id is None:
                conv_id = await store.new_conversation(CHANNEL)
            return await photo_store.save(conv_id, jpeg)

        def turn_photos() -> TurnPhotos:
            mode = photo_mode()
            return TurnPhotos(mode, photo_keep, save_photo if mode == "disk" else None)  # type: ignore[arg-type]

        async def load_photos() -> None:
            """Photos "disk": give the history its saved frames back (a resumed
            conversation). Called right before a turn starts, under
            recording_lock, so no turn changes history meanwhile; frames that
            are already loaded are not read again."""
            if photo_mode() != "disk" or conv_id is None:
                return
            try:
                await attach_photos(history, photo_store, conv_id, photo_keep)
            except Exception as exc:  # noqa: BLE001 - photos are optional
                log.warning("photos: could not load saved photos (%s)", type(exc).__name__)

        # What this page was last sent, so a change from any page reaches it
        # once (push_settings is the store's subscriber for this page).
        shown: dict[str, Any] = {"photos": None, "permissions": None}

        def chat_model_sees() -> bool:
            """Whether this page's chat model takes photos, from the model
            list the page loaded (never a request here). Gemini does not."""
            if provider != LOCAL:
                return False
            server = app.state.local
            known = getattr(server, "known_sees_images", None)
            if known is None:
                return False
            # [llm_local] model = "" is the server's default until a turn resolves it.
            model = getattr(local_client, "model_id", "") or getattr(server, "known_default_model", lambda: "")()
            return known(model) is True

        def rows_now(choices: Choices) -> tuple[dict[str, str], dict[str, str]]:
            """(missing, notes) of the Permissions rows for this page now. The
            look row depends on the page's chat model."""
            missing: dict[str, str] = {}
            notes: dict[str, str] = {}
            looker = app.state.looker
            if looker is None:
                missing["look"] = MISSING_CAMERA
            elif chat_model_sees():
                notes["look"] = NOTE_LOOK_SEES
            elif looker.vision_on:
                notes["look"] = NOTE_LOOK_QWENNY
            else:
                missing["look"] = MISSING_LOOK
            if not search_declared():
                missing["search_image"] = MISSING_SEARCH
            if recall is None:
                missing["recall"] = MISSING_RECALL
            for t in PANEL_TOOLS:
                needed = _PANEL_BY_NAME.get(t.policy.requires)
                if needed is not None and t.name not in missing and choices.tools[needed.name].mode == "off":
                    missing[t.name] = f"needs {needed.policy.label_text(needed.name)} on"
            return missing, notes

        async def push_settings(
            choices: Choices, photos_too: bool = False, permissions_too: bool = False
        ) -> None:
            """Sends what changed since this page was last told; *_too sends
            it anyway (the answer to an unchanged choice)."""
            missing, notes = rows_now(choices)
            msg = permissions_message(choices, missing, notes)
            if permissions_too or msg != shown["permissions"]:
                shown["permissions"] = msg
                await send(msg)
            if photos_too or choices.photos != shown["photos"]:
                # A running turn may be changing history: then the next turn
                # drops the photos itself (pipeline), and saved ones load
                # before the next turn (load_photos).
                if choices.photos == "off" and shown["photos"] not in (None, "off") and not turn_running():
                    keep_newest_photos(history, 0)
                shown["photos"] = choices.photos
                await send({"type": "photos", "mode": choices.photos})

        async def saved(change: Awaitable[Any], photos_too: bool = False, permissions_too: bool = False) -> None:
            """Runs a store change. A failed write (read-only file, full disk)
            leaves the old file and choices: the page gets an error and the
            unchanged choice, so its switch snaps back, and stays connected."""
            try:
                await change
            except OSError as exc:
                log.warning("settings: a change could not be saved (%s)", type(exc).__name__)
                await send({"type": "error", "message": MSG_NOT_SAVED})
                await push_settings(choices_store.get(), photos_too=photos_too, permissions_too=permissions_too)

        async def handle_photos(payload: dict[str, Any]) -> None:
            async with recording_lock:
                mode = payload.get("mode")
                if mode not in PHOTO_MODES:
                    await send({"type": "error", "message": MSG_BAD_PHOTOS})
                    return
                if mode == photo_mode():
                    await push_settings(choices_store.get(), photos_too=True)
                    return
                # Every open page hears of it through push_settings.
                await saved(choices_store.set_photos(mode), photos_too=True)

        async def handle_permission(payload: dict[str, Any]) -> None:
            async with recording_lock:
                await _handle_permission_locked(payload)

        async def _handle_permission_locked(payload: dict[str, Any]) -> None:
            name = payload.get("tool")
            panel_tool = _PANEL_BY_NAME.get(name) if isinstance(name, str) else None
            if panel_tool is None:
                await send({"type": "error", "message": MSG_UNKNOWN_TOOL})
                return
            current = choices_store.get().tools[panel_tool.name]
            if "mode" in payload:
                mode = payload["mode"]
                if mode not in MODES:
                    await send({"type": "error", "message": MSG_BAD_PERMISSION})
                    return
                if mode == current.mode:
                    await push_settings(choices_store.get(), permissions_too=True)
                    return
                log.info("permissions: %s set to %s", panel_tool.name, mode)
                await saved(choices_store.set_mode(panel_tool.name, mode), permissions_too=True)
                return
            if "cloud" in payload:
                on = payload["cloud"]
                if not isinstance(on, bool):
                    await send({"type": "error", "message": MSG_BAD_PERMISSION})
                    return
                if not panel_tool.policy.private_data:
                    await send({"type": "error", "message": MSG_NO_CLOUD_CHOICE})
                    return
                if on == current.cloud:
                    await push_settings(choices_store.get(), permissions_too=True)
                    return
                log.info("permissions: %s with a cloud model %s", panel_tool.name, "on" if on else "off")
                await saved(choices_store.set_cloud(panel_tool.name, on), permissions_too=True)
                return
            await send({"type": "error", "message": MSG_BAD_PERMISSION})

        # A confirm waiting for this page's answer (pipeline.Confirms): its
        # id and the future its reply() awaits. queued: answers that were
        # neither yes nor no, run as turns after the current one.
        pending: dict[str, Any] = {"id": None, "future": None}
        queued: list[str] = []

        def waiting() -> bool:
            future = pending["future"]
            return future is not None and not future.done()

        def answer(value: ConfirmReply) -> None:
            pending["future"].set_result(value)

        async def confirm_reply(confirm_id: str) -> ConfirmReply:
            future: asyncio.Future[ConfirmReply] = asyncio.get_running_loop().create_future()
            pending["id"], pending["future"] = confirm_id, future
            try:
                return await future
            finally:
                if pending["future"] is future:
                    pending["id"] = pending["future"] = None

        def run_next(text: str) -> None:
            queued.append(text)

        confirms = Confirms(
            reply=confirm_reply,
            run_next=run_next,
            timeout_s=settings.permissions.confirm_timeout_s,
            stt=app.state.stt,
            language=stt_cfg.language_or_none,
        )

        async def handle_confirm_answer(payload: dict[str, Any]) -> None:
            async with recording_lock:
                confirm_id = payload.get("id")
                yes = payload.get("yes")
                if not isinstance(confirm_id, str) or not isinstance(yes, bool):
                    await send({"type": "error", "message": MSG_BAD_CONFIRM})
                    return
                if not waiting() or confirm_id != pending["id"]:
                    # A card from an earlier question (timed out, or another turn).
                    log.info("confirm: an answer to a question that is not waiting, ignored")
                    return
                answer(yes)

        async def recall_source(info: TurnInfo) -> list[Tool]:
            # Per turn: the turns of this conversation already in history are
            # skipped. A cloud model does not get it (private data, registry).
            if recall is None:
                return []
            skip = (conv_id, settings.llm.history_turns) if conv_id is not None else None
            return [RECALL_TOOL.bind(recall=recall, skip=skip)]

        def turn_registry() -> Registry:
            sources: list[ToolSource] = [recall_source]
            if app.state.looker is not None:
                sources.insert(0, camera_source(app.state.looker))
            return build_registry(sources, permissions=choices_store, on_run=activity_log.add)

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

        def current_llm() -> LLMClient:
            gemini = app.state.llm
            return gemini if provider == GEMINI and gemini is not None else local_client

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
                # The only way onto GEMINI after connect: never without a client.
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
                await push_settings(choices_store.get())
                return
            await send(
                switch_notice(
                    provider,
                    settings,
                    model=model,
                    vision_on=bool(getattr(app.state.looker, "vision_on", False)),
                    search_on=search_declared(),
                    today_tokens=app.state.usage.today_total,
                    # From the model list the page loaded; never a request here.
                    tools=app.state.local.known_tools(model),
                )
            )
            # The look row follows the model (sent only when it changed).
            await push_settings(choices_store.get())
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
            turn = asyncio.create_task(turn_chain(coro))
            turn.add_done_callback(on_turn_done)

        async def turn_chain(first: Coroutine[Any, Any, str | None]) -> None:
            """The turn, then each queued answer (neither yes nor no) as a text
            turn of its own, once, in the same task, so nothing else starts
            in between."""
            await first
            await settle_rows()
            while queued:
                text = queued.pop(0)
                async with recording_lock:
                    llm = current_llm()
                    await load_photos()
                    coro = text_turn(text, llm)
                await coro
                await settle_rows()

        async def settle_rows() -> None:
            # A turn may have learned whether the chat model sees images (the
            # model list): the look row follows. Sends nothing when unchanged.
            try:
                await push_settings(choices_store.get())
            except Exception as exc:  # noqa: BLE001 - the page may be gone
                log.debug("permissions after a turn: %s", type(exc).__name__)

        def text_turn(text: str, llm: LLMClient) -> Coroutine[Any, Any, str | None]:
            return run_text_turn(
                text,
                history,
                llm,
                app.state.usage,
                send,
                history_turns=settings.llm.history_turns,
                max_output_tokens=current_max_output_tokens(),
                turn_timeout_s=settings.llm.turn_timeout_s,
                tts=app.state.tts,
                tools=turn_registry(),
                max_rounds=settings.tools.max_rounds,
                on_turn=store_turn(provider, llm),
                photos=turn_photos(),
                confirms=confirms,
            )

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

        async def drop_photos(cid: str, names: tuple[str, ...], whole: bool) -> None:
            """Remove the saved photos of what the editor deleted."""
            try:
                if whole:
                    removed = await photo_store.delete_conversation(cid)
                else:
                    removed = await photo_store.delete_files(cid, list(names))
                log.debug("memory: removed %d photos", removed)
            except Exception as exc:  # noqa: BLE001 - the change is stored; photos are cleaned up later
                log.warning("photos: could not remove photos (%s)", type(exc).__name__)

        async def _handle_memory_locked(payload: dict[str, Any]) -> None:
            """The memory editor's writes: one atomic save, or delete a
            conversation. Never logs message text, only the action."""
            if store is None:
                await send({"type": "error", "message": MSG_MEMORY_OFF})
                return
            action = payload.get("action")
            bad = {"type": "error", "message": MSG_MEMORY_BAD}

            def text_of(v: Any) -> str | None:
                if not isinstance(v, str) or len(v) > MAX_EDIT_CHARS or not v.strip():
                    return None
                return v

            def int_of(v: Any) -> int | None:
                if not isinstance(v, int) or isinstance(v, bool) or not 1 <= v <= MAX_SQL_INT:
                    return None
                return v

            def items(name: str, limit: int) -> list[Any] | None:
                v = payload.get(name, [])
                return v if isinstance(v, list) and len(v) <= limit else None

            v = payload.get("conversation")
            cid = v if isinstance(v, str) and _CONVERSATION_ID.fullmatch(v) else None
            if cid is None or action not in ("save", "delete_conversation"):
                await send(bad)
                return
            expect = 0
            title: str | None = None
            edits: list[tuple[int, str]] = []
            doomed: list[int] = []
            adds: list[tuple[str, str]] = []
            if action == "save":
                raw_edits = items("edits", MAX_SAVE_EDITS)
                raw_deletes = items("delete_turns", MAX_SAVE_DELETES)
                raw_adds = items("add_turns", MAX_SAVE_ADDS)
                expect_in = int_of(payload.get("expect_turns"))
                raw_title = payload.get("title")
                if (
                    raw_edits is None or raw_deletes is None or raw_adds is None or expect_in is None
                    or not (raw_title is None or (isinstance(raw_title, str) and len(raw_title) <= MAX_EDIT_CHARS))
                ):
                    await send(bad)
                    return
                expect, title = expect_in, raw_title
                for e in raw_edits:
                    mid = int_of(e.get("message")) if isinstance(e, dict) else None
                    etext = text_of(e.get("text")) if isinstance(e, dict) else None
                    if mid is None or etext is None:
                        await send(bad)
                        return
                    edits.append((mid, etext))
                for d in raw_deletes:
                    turn_no = int_of(d)
                    if turn_no is None:
                        await send(bad)
                        return
                    doomed.append(turn_no)
                for a in raw_adds:
                    user = text_of(a.get("user")) if isinstance(a, dict) else None
                    reply = text_of(a.get("reply")) if isinstance(a, dict) else None
                    if user is None or reply is None:
                        await send(bad)
                        return
                    adds.append((user, reply))
            if await store.info(cid) is None:
                await send({"type": "error", "message": MSG_MEMORY_MISSING})
                return
            is_open = cid == conv_id
            if is_open and (recording is not None or turn_running()):
                await send({"type": "error", "message": MSG_CONVERSATION_BUSY})
                return
            if action == "delete_conversation":
                if not await store.delete_conversation(cid):
                    await send({"type": "error", "message": MSG_MEMORY_MISSING})
                    return
                await drop_photos(cid, (), True)
            else:
                outcome = await store.save_changes(cid, expect, title, edits, doomed, adds)
                if outcome.status == "stale":
                    await send({"type": "error", "message": MSG_MEMORY_STALE})
                    return
                if outcome.status != "ok":
                    await send({"type": "error", "message": MSG_MEMORY_MISSING})
                    return
                await drop_photos(cid, outcome.photos, outcome.deleted)
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
            if waiting():
                # The answer to the waiting confirm, not a turn of its own.
                answer(text)
                return
            if turn_running():
                await send({"type": "error", "message": MSG_BUSY})
                return
            llm = current_llm()
            await load_photos()
            start_turn(text_turn(text, llm))

        async def handle_audio_start() -> None:
            async with recording_lock:
                await _handle_audio_start_locked()

        async def _handle_audio_start_locked() -> None:
            nonlocal recording, too_long_sent
            if app.state.stt is None:
                await send({"type": "error", "message": MSG_NO_VOICE})
                return
            # While a confirm waits, push-to-talk records the answer.
            if recording is not None or (turn_running() and not waiting()):
                await send({"type": "error", "message": MSG_BUSY})
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
                # A confirm still waits (the face goes back to it); else idle.
                await send({"type": "state", "state": "thinking" if waiting() else "idle"})
                return
            if waiting():
                # The answer to the waiting confirm (the pipeline transcribes it).
                answer(pcm16)
                await send({"type": "state", "state": "thinking"})
                return
            if turn_running():
                # The confirm this recording was for ended before audio_end.
                await send({"type": "error", "message": MSG_BUSY})
                return
            llm = current_llm()
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
                    tools=turn_registry(),
                    max_rounds=settings.tools.max_rounds,
                    on_turn=store_turn(provider, llm),
                    photos=turn_photos(),
                    confirms=confirms,
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

        leave_settings = None
        try:
            await push_settings(choices_store.get())
            leave_settings = choices_store.subscribe(push_settings)
            row_refreshers.add(settle_rows)
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
                elif kind == "permission":
                    await handle_permission(payload)
                elif kind == "confirm_answer":
                    await handle_confirm_answer(payload)
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
            if leave_settings is not None:
                leave_settings()
            row_refreshers.discard(settle_rows)
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
