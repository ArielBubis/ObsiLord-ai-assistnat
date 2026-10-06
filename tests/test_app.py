from pathlib import Path

import json
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fakes import FakeLLM
from pixelbuddy.app import (
    MAX_TEXT_CHARS,
    MSG_BUSY,
    MSG_EMPTY,
    MSG_NO_KEY,
    MSG_NO_VOICE,
    MSG_TOO_LONG,
    MSG_UNKNOWN,
    create_app,
)
from pixelbuddy.config import REPO_ROOT, Settings, load_settings
from pixelbuddy.llm.base import UsageUpdate
from pixelbuddy.usage import UsageTracker


def _settings(tmp_path: Path) -> Settings:
    return load_settings(
        REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
    )


def _make(tmp_path: Path, llm: FakeLLM | None) -> TestClient:
    settings = _settings(tmp_path)
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    return TestClient(create_app(settings, llm=llm, usage=usage))


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    return _make(tmp_path, FakeLLM(["Hello ", "you."]))


def _collect_turn(ws: Any) -> list[dict[str, Any]]:
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg == {"type": "state", "state": "idle"}:
            return msgs


def test_index_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "<html" in resp.text.lower()


def test_static_js_served(client: TestClient) -> None:
    resp = client.get("/static/ws.js")
    assert resp.status_code == 200
    assert "WebSocket" in resp.text


def test_ws_text_turn_sequence(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "  hi  "})
        assert _collect_turn(ws) == [
            {"type": "state", "state": "thinking"},
            {"type": "state", "state": "speaking"},
            {"type": "transcript", "role": "assistant", "text": "Hello ", "final": False},
            {"type": "transcript", "role": "assistant", "text": "Hello you.", "final": False},
            {"type": "transcript", "role": "assistant", "text": "Hello you.", "final": True},
            {"type": "state", "state": "idle"},
        ]


def test_ws_history_kept_per_connection(tmp_path: Path) -> None:
    llm = FakeLLM(["ok"])
    client = _make(tmp_path, llm)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "one"})
        _collect_turn(ws)
        ws.send_json({"type": "text", "text": "two"})
        _collect_turn(ws)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "three"})
        _collect_turn(ws)
    assert [len(c) for c in llm.calls] == [1, 3, 1]


def test_ws_busy_rejection(tmp_path: Path) -> None:
    gate = threading.Event()
    client = _make(tmp_path, FakeLLM(["First ", "reply."], gate=gate))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "one"})
        assert ws.receive_json() == {"type": "state", "state": "thinking"}
        assert ws.receive_json() == {"type": "state", "state": "speaking"}
        assert ws.receive_json()["text"] == "First "
        ws.send_json({"type": "text", "text": "two"})
        assert ws.receive_json() == {"type": "error", "message": MSG_BUSY}
        gate.set()
        rest = _collect_turn(ws)
        assert rest[-2] == {
            "type": "transcript", "role": "assistant", "text": "First reply.", "final": True,
        }


def test_ws_ping_gets_no_reply(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})
        ws.send_json({"type": "bogus"})
        # The first reply is for the unknown type, so the ping produced nothing.
        assert ws.receive_json() == {"type": "error", "message": MSG_UNKNOWN}


@pytest.mark.parametrize("text", ["", "   ", None, 5])
def test_ws_empty_text_rejected(client: TestClient, text: Any) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": text})
        assert ws.receive_json() == {"type": "error", "message": MSG_EMPTY}


def test_ws_too_long_text_rejected(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "a" * (MAX_TEXT_CHARS + 1)})
        assert ws.receive_json() == {"type": "error", "message": MSG_TOO_LONG}


def test_ws_voice_not_available(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NO_VOICE}
        ws.send_bytes(bytes([0, 1]))
        assert ws.receive_json() == {"type": "error", "message": MSG_NO_VOICE}
        ws.send_json({"type": "audio_end"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NO_VOICE}


def test_ws_non_object_json_is_unknown(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_text("[1, 2]")
        assert ws.receive_json() == {"type": "error", "message": MSG_UNKNOWN}


def test_ws_no_api_key(tmp_path: Path) -> None:
    client = _make(tmp_path, None)
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "hi"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NO_KEY}


def test_ws_invalid_json_returns_error_and_stays_open(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert isinstance(msg["message"], str)
        ws.send_json({"type": "text", "text": "hi"})
        assert ws.receive_json() == {"type": "state", "state": "thinking"}


def test_ws_disconnect_cancels_turn_and_records_usage(tmp_path: Path) -> None:
    gate = threading.Event()  # never set: the turn stays mid-stream
    llm = FakeLLM(["First ", "reply."], gate=gate, usage=UsageUpdate(12, 3, 0))
    settings = _settings(tmp_path)
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    client = TestClient(create_app(settings, llm=llm, usage=usage))
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "one"})
        assert ws.receive_json() == {"type": "state", "state": "thinking"}
        assert ws.receive_json() == {"type": "state", "state": "speaking"}
        assert ws.receive_json()["text"] == "First "
    # Leaving the block closes the socket; the server cancels the turn.
    assert llm.closed.wait(timeout=5)
    assert llm.cancelled.is_set()
    deadline = time.monotonic() + 5
    while usage.today_total == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert usage.today_total == 15
    entry = json.loads((tmp_path / "usage.jsonl").read_text(encoding="utf-8"))
    assert entry["partial"] is True
