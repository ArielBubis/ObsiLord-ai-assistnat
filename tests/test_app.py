from pathlib import Path

import dataclasses
import json
import threading
import time
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fakes import FakeLLM, FakeTTS, text_only
from obsilord.app import (
    MAX_TEXT_CHARS,
    MSG_BUSY,
    MSG_EMPTY,
    MSG_NO_KEY,
    MSG_NO_VOICE,
    MSG_NOT_RECORDING,
    MSG_TOO_LONG,
    MSG_UNEXPECTED_AUDIO,
    MSG_UNKNOWN,
    create_app,
)
from obsilord.config import REPO_ROOT, Settings, load_settings
from obsilord.llm.base import UsageUpdate
from obsilord.usage import UsageTracker


def _settings(tmp_path: Path) -> Settings:
    return text_only(
        load_settings(
            REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
        )
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
    # text_only settings: [stt] model is empty, so voice input is disabled.
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NO_VOICE}
        ws.send_bytes(bytes([0, 1]))
        assert ws.receive_json() == {"type": "error", "message": MSG_UNEXPECTED_AUDIO}
        ws.send_json({"type": "audio_end"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NOT_RECORDING}


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


def _collect_turn_raw(ws: Any) -> list[dict[str, Any] | bytes]:
    msgs: list[dict[str, Any] | bytes] = []
    while True:
        raw = ws.receive()
        if raw.get("bytes") is not None:
            msgs.append(raw["bytes"])
            continue
        msg = json.loads(raw["text"])
        msgs.append(msg)
        if msg == {"type": "state", "state": "idle"}:
            return msgs


def test_ws_turn_with_tts_sends_binary_frames(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    tts = FakeTTS(frames=2)
    app = create_app(settings, llm=FakeLLM(["Hello ", "you."]), usage=usage, tts=tts)
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "hi"})
        msgs = _collect_turn_raw(ws)
    assert msgs == [
        {"type": "state", "state": "thinking"},
        {"type": "transcript", "role": "assistant", "text": "Hello ", "final": False},
        {"type": "transcript", "role": "assistant", "text": "Hello you.", "final": False},
        {"type": "transcript", "role": "assistant", "text": "Hello you.", "final": True},
        {"type": "tts_start", "sentence_id": 1, "sample_rate": 22050},
        {"type": "state", "state": "speaking"},
        bytes([1, 0]) * 2,
        bytes([1, 1]) * 2,
        {"type": "tts_end", "sentence_id": 1},
        {"type": "state", "state": "idle"},
    ]


def test_empty_voice_loads_no_tts(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), llm=FakeLLM(["x"]))
    assert app.state.tts is None


def test_bad_speaker_stops_startup(tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"junk")
    (tmp_path / "voice.onnx.json").write_text(
        json.dumps({"speaker_id_map": {"obadiah": 2}}), encoding="utf-8"
    )
    base = _settings(tmp_path)
    settings = dataclasses.replace(
        base, tts=dataclasses.replace(base.tts, voice=str(model), speaker="nobody")
    )
    with pytest.raises(ValueError, match="valid names: obadiah"):
        create_app(settings, llm=FakeLLM(["x"]))


def test_face_config_endpoint(client: TestClient) -> None:
    resp = client.get("/config/face")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {
        "logical_width": 64,
        "logical_height": 48,
        "mouth_thresholds": [0.05, 0.15, 0.30],
        "skin": "dither",
    }


def test_face_config_follows_settings(tmp_path: Path) -> None:
    base = _settings(tmp_path)
    settings = dataclasses.replace(
        base,
        face=dataclasses.replace(base.face, mouth_thresholds=(0.1, 0.2, 0.4), skin="code"),
    )
    client = TestClient(create_app(settings, llm=FakeLLM(["x"])))
    body = client.get("/config/face").json()
    assert body["mouth_thresholds"] == [0.1, 0.2, 0.4]
    assert body["skin"] == "code"
    # The character path is not exposed: the page loads /face/character.glb.
    assert set(body) == {"logical_width", "logical_height", "mouth_thresholds", "skin"}


def test_audio_config_endpoint(client: TestClient) -> None:
    resp = client.get("/config/audio")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json() == {
        "low_shelf_hz": 250.0,
        "low_shelf_db": -6.0,
        "high_shelf_hz": 3500.0,
        "high_shelf_db": 0.0,
        "max_utterance_s": 30.0,
    }


def test_audio_config_follows_settings(tmp_path: Path) -> None:
    base = _settings(tmp_path)
    settings = dataclasses.replace(
        base,
        tts=dataclasses.replace(base.tts, low_shelf_db=-3.0, high_shelf_db=2.5, pitch=1.1),
        stt=dataclasses.replace(base.stt, max_utterance_s=12.5),
    )
    client = TestClient(create_app(settings, llm=FakeLLM(["x"])))
    body = client.get("/config/audio").json()
    # Exactly these five keys: nothing else from the settings is exposed.
    assert body == {
        "low_shelf_hz": 250.0,
        "low_shelf_db": -3.0,
        "high_shelf_hz": 3500.0,
        "high_shelf_db": 2.5,
        "max_utterance_s": 12.5,
    }
