from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from pixelbuddy.app import create_app
from pixelbuddy.config import REPO_ROOT, load_settings


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    settings = load_settings(
        REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
    )
    return TestClient(create_app(settings))


def test_index_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert "<html" in resp.text.lower()


def test_static_js_served(client: TestClient) -> None:
    resp = client.get("/static/ws.js")
    assert resp.status_code == 200
    assert "WebSocket" in resp.text


def test_ws_echoes_json(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "ping"}
        ws.send_json({"type": "text", "text": "hello"})
        assert ws.receive_json() == {"type": "text", "text": "hello"}


def test_ws_invalid_json_returns_error_and_stays_open(client: TestClient) -> None:
    with client.websocket_connect("/ws") as ws:
        ws.send_text("{not json")
        msg = ws.receive_json()
        assert msg["type"] == "error"
        assert isinstance(msg["message"], str)
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "ping"}
