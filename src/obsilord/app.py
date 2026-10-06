"""FastAPI app: serves the web page and the /ws WebSocket.

Phase 2: {"type": "text"} runs a streamed LLM turn (see pipeline.py). Voice
input is not wired yet. Invalid JSON gets an error message and the connection
stays open.

The LLM client and UsageTracker are created once in create_app (app startup)
and can be injected for tests. If no API key is configured the app still
starts and text turns return an error.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from obsilord.config import Settings
from obsilord.llm.base import LLMClient, Message
from obsilord.pipeline import run_text_turn
from obsilord.usage import UsageTracker

log = logging.getLogger(__name__)

MAX_TEXT_CHARS = 2000

MSG_INVALID_JSON = "invalid JSON"
MSG_NO_KEY = "No API key configured."
MSG_BUSY = "Busy, wait for the current reply."
MSG_EMPTY = "Message is empty."
MSG_TOO_LONG = f"Message is too long (max {MAX_TEXT_CHARS} characters)."
MSG_NO_VOICE = "Voice input is not available yet."
MSG_UNKNOWN = "Unknown message type."


def _default_llm(settings: Settings) -> LLMClient | None:
    # Imported here so the provider SDK is only loaded when actually used.
    from obsilord.llm.gemini import GeminiClient, MissingApiKey

    try:
        return GeminiClient(settings.llm, settings.secrets.gemini_api_key)
    except MissingApiKey:
        log.warning("no GEMINI_API_KEY configured; text turns will return an error")
        return None


def create_app(
    settings: Settings,
    llm: LLMClient | None = None,
    usage: UsageTracker | None = None,
) -> FastAPI:
    """Build the app. Pass llm and usage to inject fakes (tests)."""
    web_dir = settings.repo_root / "web"
    index_path = web_dir / "index.html"

    if llm is None:
        llm = _default_llm(settings)
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

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(index_path, media_type="text/html")

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        log.info("ws connected")
        history: list[Message] = []
        turn: asyncio.Task[Any] | None = None
        send_lock = asyncio.Lock()

        async def send(msg: dict[str, Any]) -> None:
            async with send_lock:
                await websocket.send_json(msg)

        def on_turn_done(task: asyncio.Task[Any]) -> None:
            # run_text_turn handles its own errors; this is a last-resort log.
            if not task.cancelled() and task.exception() is not None:
                log.error("turn task failed: %s", type(task.exception()).__name__)

        async def handle_text(payload: dict[str, Any]) -> None:
            nonlocal turn
            raw = payload.get("text")
            text = raw.strip() if isinstance(raw, str) else ""
            if not text:
                await send({"type": "error", "message": MSG_EMPTY})
                return
            if len(text) > MAX_TEXT_CHARS:
                await send({"type": "error", "message": MSG_TOO_LONG})
                return
            if turn is not None and not turn.done():
                await send({"type": "error", "message": MSG_BUSY})
                return
            if app.state.llm is None:
                await send({"type": "error", "message": MSG_NO_KEY})
                return
            turn = asyncio.create_task(
                run_text_turn(
                    text,
                    history,
                    app.state.llm,
                    app.state.usage,
                    send,
                    history_turns=settings.llm.history_turns,
                    max_output_tokens=settings.llm.max_output_tokens,
                    turn_timeout_s=settings.llm.turn_timeout_s,
                )
            )
            turn.add_done_callback(on_turn_done)

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                data = message.get("text")
                if data is None:
                    # Binary frames (mic audio) arrive in Phase 4.
                    await send({"type": "error", "message": MSG_NO_VOICE})
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
                elif kind in ("audio_start", "audio_end"):
                    await send({"type": "error", "message": MSG_NO_VOICE})
                else:
                    await send({"type": "error", "message": MSG_UNKNOWN})
        except WebSocketDisconnect:
            pass
        finally:
            if turn is not None and not turn.done():
                turn.cancel()
                try:
                    await turn
                except BaseException:  # noqa: BLE001 - cancelled or failed on close
                    pass
        log.info("ws disconnected")

    app.mount("/static", StaticFiles(directory=web_dir), name="static")
    return app
