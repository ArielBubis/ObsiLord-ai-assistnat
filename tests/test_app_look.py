"""look() through the app: the WebSocket turn with a fake look tool, and the
startup checks for the vision files. No camera, no model."""

import dataclasses
import logging
from pathlib import Path
from typing import Any

import pytest

from fakes import AppClient, FakeLooker, FakeToolLLM, SeeingToolLLM, text_only
from obsilord.app import _default_looker, create_app
from obsilord.config import REPO_ROOT, Settings, load_settings
from obsilord.llm.base import Done, TextDelta, ToolCall
from obsilord.look import LOOK_TOOL, Looker
from obsilord.tools.calculate import calculate
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
    with AppClient(app).websocket_connect("/ws") as ws:
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
    assert llm.requests[0].tools == [LOOK_TOOL.spec, calculate.spec]


def test_vision_off_still_builds_the_camera_for_a_chat_model_that_sees(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path), llm=FakeToolLLM([[TextDelta("Hi."), Done(1, 1, 0)]]))
    assert app.state.looker is not None
    assert app.state.looker.vision_on is False
    assert app.state.looker.search_on is False


def _with_vision_files(tmp_path: Path, skip: str = "") -> Settings:
    base = load_settings(
        REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
    )
    paths = {}
    for key in ("server_cuda", "server_cpu", "model", "mmproj"):
        path = tmp_path / f"{key}.bin"
        if key != skip:
            path.write_bytes(b"x")
        paths[key] = str(path)  # absolute, so repo_root does not matter
    return dataclasses.replace(base, vision=dataclasses.replace(base.vision, **paths))


def test_default_looker_is_built_without_starting_anything(tmp_path: Path) -> None:
    looker = _default_looker(_with_vision_files(tmp_path))
    assert isinstance(looker, Looker)
    assert looker.vision_on is True


def test_no_vision_model_gives_a_camera_without_qwenny(tmp_path: Path) -> None:
    looker = _default_looker(_settings(tmp_path))
    assert isinstance(looker, Looker)
    assert looker.vision_on is False and looker.search_on is False


def test_missing_vision_file_runs_without_qwenny_and_logs_one_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="obsilord.app")
    looker = _default_looker(_with_vision_files(tmp_path, skip="mmproj"))
    assert isinstance(looker, Looker) and looker.vision_on is False
    lines = [r.getMessage() for r in caplog.records if r.name == "obsilord.app"]
    assert len(lines) == 1  # the one startup line, which also says what is missing
    assert "mmproj not found" in lines[0]
    assert "python -m obsilord.vision.fetch" in lines[0]
    assert "photo" in lines[0]


def test_startup_line_names_the_camera_paths_without_content(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="obsilord.app")
    _default_looker(_with_vision_files(tmp_path))
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("camera:")]
    assert len(lines) == 1
    assert "photo" in lines[0] and "Qwenny" in lines[0]


def test_a_vision_chat_model_turn_gets_look_photo_without_qwenny(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = SeeingToolLLM(
        [
            [ToolCall("look", {"question": "q"}), Done(5, 1, 0)],
            [TextDelta("A red mug."), Done(9, 3, 0)],
        ]
    )
    looker = FakeLooker(vision=False)
    app = create_app(
        settings,
        llm=llm,
        usage=UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap),
        looker=looker,
    )
    with AppClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "what am I holding?"})
        _collect_turn(ws)
    assert looker.photos_taken == 1 and looker.questions == []
    assert [t.name for t in llm.requests[0].tools] == ["look", "calculate"]
