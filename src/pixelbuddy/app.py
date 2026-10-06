"""FastAPI app: serves the web page and the /ws WebSocket.

Phase 0: /ws echoes every JSON message back. Invalid JSON gets an error
message (see ARCHITECTURE.md protocol) and the connection stays open.
"""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pixelbuddy.config import Settings

log = logging.getLogger(__name__)


def create_app(settings: Settings) -> FastAPI:
    web_dir = settings.repo_root / "web"
    index_path = web_dir / "index.html"

    app = FastAPI(title="Pixel Buddy", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings = settings

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(index_path, media_type="text/html")

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        log.info("ws connected")
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break
                text = message.get("text")
                if text is None:
                    # Binary frames (audio) are not handled until Phase 4.
                    await websocket.send_json(
                        {"type": "error", "message": "binary frames not supported yet"}
                    )
                    continue
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "invalid JSON"})
                    continue
                await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        log.info("ws disconnected")

    app.mount("/static", StaticFiles(directory=web_dir), name="static")
    return app
