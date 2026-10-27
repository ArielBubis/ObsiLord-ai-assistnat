"""look() through the app: the WebSocket turn with a fake look tool, and the
startup checks for the vision files. No camera, no model."""

import dataclasses
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from fakes import FakeLooker, FakeToolLLM, text_only
from obsilord.app import _default_looker, create_app
from obsilord.config import REPO_ROOT, Settings, load_settings
from obsilord.llm.base import Done, TextDelta, ToolCall
from obsilord.look import LOOK_TOOL, Looker
from obsilord.usage import UsageTracker


def _settings(tmp_path: Path) -> Settings:
    return text_only(
        load_settings(
            REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
        )
    )


def _collect_turn(ws: Any) -> list[dict[str, Any]]:
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg == {"type": "state", "state": "idle"}:
            return msgs


def test_ws_turn_with_a_look(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeToolLLM(
        [
            [ToolCall("look", {"question": "What am I holding?"}), Done(5, 1, 0)],
            [TextDelta("A red mug."), Done(9, 3, 0)],
        ]
    )
    looker = FakeLooker()
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    app = create_app(settings, llm=llm, usage=usage, looker=looker)
    with TestClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "what am I holding?"})
        msgs = _collect_turn(ws)
    assert [m["state"] for m in msgs if m["type"] == "state"] == [
        "thinking",
        "looking",
        "thinking",
        "speaking",
        "idle",
    ]
    assert msgs[-2] == {"type": "transcript", "role": "assistant", "text": "A red mug.", "final": True}
    assert looker.questions == ["What am I holding?"]
    assert llm.requests[0].tools == [LOOK_TOOL]


def test_vision_off_declares_no_tool(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), llm=FakeToolLLM([[TextDelta("Hi."), Done(1, 1, 0)]]))
    assert app.state.looker is None


def _with_vision_files(tmp_path: Path, skip: str = "") -> Settings:
    base = load_settings(
        REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
    )
    paths = {}
    for key in ("server_vulkan", "server_cpu", "model", "mmproj"):
        path = tmp_path / f"{key}.bin"
        if key != skip:
            path.write_bytes(b"x")
        paths[key] = str(path)  # absolute, so repo_root does not matter
    return dataclasses.replace(base, vision=dataclasses.replace(base.vision, **paths))


def test_default_looker_is_built_without_starting_anything(tmp_path: Path) -> None:
    looker = _default_looker(_with_vision_files(tmp_path))
    assert isinstance(looker, Looker)


def test_missing_vision_file_stops_startup_with_a_hint(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.ERROR)
    with pytest.raises(ValueError, match="mmproj not found"):
        _default_looker(_with_vision_files(tmp_path, skip="mmproj"))
    assert "python -m obsilord.vision.fetch" in caplog.text
    assert '[vision] model = ""' in caplog.text
