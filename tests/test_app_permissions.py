"""Permissions, confirms and the activity log through the app (T2 Task 5).
Fakes only: no camera, no model, no network."""

import dataclasses
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from fakes import FakeLooker, FakeSTT, FakeToolLLM, text_only
from obsilord.activity import ActivityLog
from obsilord.app import (
    MSG_BAD_CONFIRM,
    MSG_BAD_PERMISSION,
    MSG_BUSY,
    MSG_NO_CLOUD_CHOICE,
    MSG_NOT_SAVED,
    MSG_UNKNOWN_TOOL,
    create_app,
    tool_defaults,
)
from obsilord.config import REPO_ROOT, Settings, load_settings
from obsilord.llm.base import Done, TextDelta, ToolCall
from obsilord.llm.openai_compat import OpenAICompatServer
from obsilord import permissions
from obsilord.permissions import SettingsStore
from obsilord.tools.registry import msg_declined
from obsilord.usage import UsageTracker
from test_openai_compat import LMSTUDIO_MODELS

IDLE = {"type": "state", "state": "idle"}
LISTENING = {"type": "state", "state": "listening"}
FRAME = bytes(3200)  # 100 ms of PCM16 at 16 kHz
LOOK = ToolCall("look", {"question": "What am I holding?"}, call_id="c1")


def _settings(tmp_path: Path, **permissions: Any) -> Settings:
    s = text_only(load_settings(REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}))
    return dataclasses.replace(s, permissions=dataclasses.replace(s.permissions, **permissions))


def _store(tmp_path: Path, settings: Settings) -> SettingsStore:
    return SettingsStore(tmp_path / "settings.json", tool_defaults(), settings.camera.photos)


def _look_then_reply(*replies: str) -> list[list[Any]]:
    """A turn that calls look, then answers; then one answer per later turn."""
    return [[LOOK, Done(1, 1)], [TextDelta("A red mug."), Done(1, 1)], *([TextDelta(r), Done(1, 1)] for r in replies)]


class Fixture:
    def __init__(
        self,
        tmp_path: Path,
        llm: FakeToolLLM | None = None,
        looker: FakeLooker | None = None,
        stt: FakeSTT | None = None,
        local: Any = None,
        **permissions: Any,
    ) -> None:
        self.settings = _settings(tmp_path, **permissions)
        self.store = _store(tmp_path, self.settings)
        self.activity = ActivityLog(tmp_path / "activity.jsonl", 0)
        self.llm = llm or FakeToolLLM(_look_then_reply())
        self.looker = looker or FakeLooker()
        self.client = TestClient(
            create_app(
                self.settings,
                llm=self.llm,
                usage=UsageTracker(tmp_path / "usage.jsonl", self.settings.llm.daily_token_cap),
                looker=self.looker,
                local=local,
                stt=stt,
                settings_store=self.store,
                activity=self.activity,
            )
        )


def _connect(ws: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The two messages every page gets on connect: permissions, then photos."""
    return ws.receive_json(), ws.receive_json()


def _until(ws: Any, kind: str) -> list[dict[str, Any]]:
    """Messages up to and including the first of type kind (binary frames skipped)."""
    msgs = []
    while True:
        raw = ws.receive()
        if raw.get("text") is None:
            continue
        msg = json.loads(raw["text"])
        msgs.append(msg)
        if msg.get("type") == kind:
            return msgs


def _until_idle(ws: Any) -> list[dict[str, Any]]:
    msgs = []
    while (msg := ws.receive_json()) != IDLE:
        msgs.append(msg)
    return msgs


def _rows(msg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    assert msg["type"] == "permissions"
    return {row["name"]: row for row in msg["tools"]}


# --- on connect, changes, errors --------------------------------------------------------


def test_every_page_gets_the_permissions_and_photos_on_connect(tmp_path: Path) -> None:
    f = Fixture(tmp_path)
    with f.client.websocket_connect("/ws") as ws:
        permissions, photos = _connect(ws)
    assert [row["name"] for row in permissions["tools"]] == ["look", "search_image", "recall", "calculate"]
    rows = _rows(permissions)
    assert rows["look"] == {
        "name": "look",
        "label": "Camera",
        "mode": "on",
        "internet": False,
        "private": True,
        # Private tools are off with a cloud model until the user allows it
        # (decision 2026-11-18, kept 2026-11-22).
        "cloud": False,
        "available": True,
        "missing": "",
        "note": "Qwenny looks and tells your chat model",
    }
    assert [row["note"] for name, row in rows.items() if name != "look"] == ["", "", ""]
    assert rows["search_image"]["mode"] == "ask"
    assert rows["search_image"]["cloud"] is False and rows["recall"]["cloud"] is False
    assert rows["search_image"]["internet"] is True
    assert rows["search_image"]["available"] is False
    assert rows["search_image"]["missing"] == "needs [vision], [image_search] and its key"
    assert rows["recall"]["available"] is False  # no memory store in this app
    assert rows["recall"]["missing"] == "needs [memory] with recall on"
    assert rows["calculate"]["available"] is True and rows["calculate"]["private"] is False
    assert photos == {"type": "photos", "mode": f.settings.camera.photos}


def test_without_a_camera_look_is_listed_as_not_set_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("obsilord.app._default_looker", lambda settings, usage=None: None)
    settings = _settings(tmp_path)
    app = create_app(
        settings,
        llm=FakeToolLLM([]),
        usage=UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap),
        settings_store=_store(tmp_path, settings),
        activity=ActivityLog(tmp_path / "activity.jsonl", 0),
    )
    with TestClient(app).websocket_connect("/ws") as ws:
        permissions, _ = _connect(ws)
    look = _rows(permissions)["look"]
    assert look["available"] is False and look["missing"] == "needs [camera]"
    assert look["note"] == ""


# --- the look row follows the vision path of the page's chat model ---------------------


LOOK_NOTE_SEES = "Your chat model looks at the photo itself"
LOOK_NOTE_QWENNY = "Qwenny looks and tells your chat model"
LOOK_MISSING_NO_QWENNY = "needs [vision] for a chat model that cannot see images"


def _lmstudio() -> OpenAICompatServer:
    """A local server whose list has a text model and a vision model (vlm)."""
    handler = lambda request: httpx.Response(200, json=LMSTUDIO_MODELS)  # noqa: E731
    return OpenAICompatServer(
        load_settings(REPO_ROOT / "config.example.toml", env_path=Path("nope.env"), environ={}).llm_local,
        transport=httpx.MockTransport(handler),
    )


def _switch(ws: Any, model: str) -> dict[str, Any]:
    """Switch the page to a local model; the permissions message that follows."""
    ws.send_json({"type": "llm", "provider": "local", "model": model})
    return _until(ws, "permissions")[-1]


def test_look_note_for_a_vision_chat_model(tmp_path: Path) -> None:
    f = Fixture(tmp_path, local=_lmstudio())
    f.client.get("/config/llm")  # the page loads the model list first
    with f.client.websocket_connect("/ws") as ws:
        _connect(ws)
        look = _rows(_switch(ws, "qwen3-vl-8b"))["look"]
    assert look["available"] is True and look["missing"] == ""
    assert look["note"] == LOOK_NOTE_SEES


def test_look_note_for_a_text_model_with_qwenny(tmp_path: Path) -> None:
    f = Fixture(tmp_path, local=_lmstudio())
    f.client.get("/config/llm")  # the page loads the model list first
    with f.client.websocket_connect("/ws") as ws:
        first, _ = _connect(ws)  # Gemini: no sees_images
        assert _rows(first)["look"]["note"] == LOOK_NOTE_QWENNY
        ws.send_json({"type": "llm", "provider": "local", "model": "qwen3-vl-8b"})
        _until(ws, "permissions")
        look = _rows(_switch(ws, "qwen3-14b"))["look"]
    assert look["available"] is True and look["note"] == LOOK_NOTE_QWENNY


def test_look_is_unavailable_for_a_text_model_without_qwenny(tmp_path: Path) -> None:
    f = Fixture(tmp_path, looker=FakeLooker(vision=False))
    with f.client.websocket_connect("/ws") as ws:
        permissions, _ = _connect(ws)
    look = _rows(permissions)["look"]
    assert look["available"] is False
    assert look["missing"] == LOOK_MISSING_NO_QWENNY
    assert look["note"] == ""
    assert _rows(permissions)["search_image"]["missing"] == "needs [vision], [image_search] and its key"


def test_the_look_row_changes_after_a_chat_model_switch(tmp_path: Path) -> None:
    f = Fixture(tmp_path, looker=FakeLooker(vision=False), local=_lmstudio())
    f.client.get("/config/llm")  # the page loads the model list first
    with f.client.websocket_connect("/ws") as ws:
        first, _ = _connect(ws)
        assert _rows(first)["look"]["available"] is False
        sees = _rows(_switch(ws, "qwen3-vl-8b"))["look"]
        text = _rows(_switch(ws, "qwen3-14b"))["look"]
    assert (sees["available"], sees["note"]) == (True, LOOK_NOTE_SEES)
    assert (text["available"], text["missing"]) == (False, LOOK_MISSING_NO_QWENNY)


def test_the_look_row_is_sent_again_once_a_turn_learns_the_model_sees(tmp_path: Path) -> None:
    # The model list fails once (the background check after the switch), so
    # only the turn's own check learns that the model sees images.
    lists = {"failed": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models") and lists["failed"] < 2:
            lists["failed"] += 1  # LM Studio's list, then the OpenAI one: one failed list
            return httpx.Response(500)
        return httpx.Response(200, json=LMSTUDIO_MODELS)

    local = OpenAICompatServer(
        load_settings(REPO_ROOT / "config.example.toml", env_path=Path("nope.env"), environ={}).llm_local,
        transport=httpx.MockTransport(handler),
    )
    f = Fixture(tmp_path, looker=FakeLooker(vision=False), local=local)
    with f.client.websocket_connect("/ws") as ws:
        first, _ = _connect(ws)
        ws.send_json({"type": "llm", "provider": "local", "model": "qwen3-vl-8b"})
        _until(ws, "notice")
        ws.send_json({"type": "text", "text": "hi"})
        during = _until_idle(ws)  # every other request gets the list: the reply fails
        ws.send_json({"type": "bogus"})  # so a missing resend cannot hang the test
        after = next(m for m in iter(ws.receive_json, None) if m.get("type") in ("permissions", "error"))
    assert lists["failed"] == 2
    assert _rows(first)["look"]["available"] is False
    assert not any(m.get("type") == "permissions" for m in during)
    assert after["type"] == "permissions"
    assert _rows(after)["look"]["note"] == LOOK_NOTE_SEES


def test_a_page_on_the_default_local_model_gets_its_look_path_when_the_list_loads(tmp_path: Path) -> None:
    # [llm_local] model = "" (the server's default, here the loaded vision
    # model) and no Gemini key: the page starts on local with an unknown id.
    vlm_loaded = {
        "object": "list",
        "data": [
            {"id": "qwen3-14b", "type": "llm", "state": "not-loaded"},
            {"id": "qwen3-vl-8b", "type": "vlm", "state": "loaded"},
        ],
    }
    settings = _settings(tmp_path)
    assert settings.llm_local.model == ""
    local = OpenAICompatServer(
        settings.llm_local, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=vlm_loaded))
    )
    app = create_app(
        settings,
        llm=None,
        usage=UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap),
        looker=FakeLooker(),
        local=local,
        settings_store=_store(tmp_path, settings),
        activity=ActivityLog(tmp_path / "activity.jsonl", 0),
    )
    with TestClient(app) as client:  # one event loop for the page and /config/llm
        with client.websocket_connect("/ws") as ws:
            first, _ = _connect(ws)
            # The page loads the model list when it opens: every open page
            # gets its look row again, without a request of its own.
            client.get("/config/llm")
            ws.send_json({"type": "bogus"})  # so a missing resend cannot hang the test
            later = next(m for m in iter(ws.receive_json, None) if m.get("type") in ("permissions", "error"))
    assert _rows(first)["look"]["note"] == LOOK_NOTE_QWENNY  # not known yet at connect
    assert later["type"] == "permissions"
    assert _rows(later)["look"]["note"] == LOOK_NOTE_SEES


def test_a_change_reaches_every_open_page(tmp_path: Path) -> None:
    f = Fixture(tmp_path)
    with f.client:  # one event loop for every page, as in the app
        with f.client.websocket_connect("/ws") as one, f.client.websocket_connect("/ws") as two:
            _connect(one)
            _connect(two)
            one.send_json({"type": "permission", "tool": "look", "mode": "ask"})
            assert _rows(one.receive_json())["look"]["mode"] == "ask"
            assert _rows(two.receive_json())["look"]["mode"] == "ask"
            two.send_json({"type": "permission", "tool": "recall", "cloud": True})
            assert _rows(one.receive_json())["recall"]["cloud"] is True
            assert _rows(two.receive_json())["recall"]["cloud"] is True
        assert f.store.mode("look") == "ask" and f.store.cloud("recall") is True
        # A new page gets the stored choices.
        with f.client.websocket_connect("/ws") as three:
            permissions, _ = _connect(three)
        assert _rows(permissions)["look"]["mode"] == "ask"


def test_an_unchanged_choice_still_answers_the_page(tmp_path: Path) -> None:
    f = Fixture(tmp_path)
    with f.client.websocket_connect("/ws") as ws:
        _connect(ws)
        ws.send_json({"type": "permission", "tool": "look", "mode": "on"})
        assert _rows(ws.receive_json())["look"]["mode"] == "on"


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"tool": "nope", "mode": "on"}, MSG_UNKNOWN_TOOL),
        ({"tool": "look", "mode": "maybe"}, MSG_BAD_PERMISSION),
        ({"tool": "look", "cloud": "yes"}, MSG_BAD_PERMISSION),
        ({"tool": "look"}, MSG_BAD_PERMISSION),
        ({"tool": "calculate", "cloud": True}, MSG_NO_CLOUD_CHOICE),
    ],
)
def test_bad_permission_messages_get_an_error(tmp_path: Path, payload: dict[str, Any], error: str) -> None:
    f = Fixture(tmp_path)
    with f.client.websocket_connect("/ws") as ws:
        _connect(ws)
        ws.send_json({"type": "permission", **payload})
        assert ws.receive_json() == {"type": "error", "message": error}
    assert not (tmp_path / "settings.json").exists()


def test_off_keeps_the_tool_away_from_the_model(tmp_path: Path) -> None:
    f = Fixture(tmp_path, llm=FakeToolLLM([[TextDelta("Hi."), Done(1, 1)]]))
    with f.client.websocket_connect("/ws") as ws:
        _connect(ws)
        ws.send_json({"type": "permission", "tool": "look", "mode": "off"})
        ws.receive_json()
        ws.send_json({"type": "text", "text": "hello"})
        _until_idle(ws)
    assert "look" not in [t.name for t in f.llm.requests[0].tools]


# --- confirms ---------------------------------------------------------------------------


def _ask_look(f: Fixture, ws: Any) -> dict[str, Any]:
    """Set look to Ask, start a turn, and return the confirm message."""
    _connect(ws)
    ws.send_json({"type": "permission", "tool": "look", "mode": "ask"})
    ws.receive_json()
    ws.send_json({"type": "text", "text": "What am I holding?"})
    return _until(ws, "confirm")[-1]


def test_the_confirm_uses_the_configured_timeout(tmp_path: Path) -> None:
    f = Fixture(tmp_path, confirm_timeout_s=33)
    with f.client.websocket_connect("/ws") as ws:
        confirm = _ask_look(f, ws)
        ws.send_json({"type": "confirm_answer", "id": confirm["id"], "yes": False})
        _until_idle(ws)
    assert confirm["timeout_s"] == 33 and confirm["tool"] == "look"


def test_a_click_answers_the_waiting_confirm_and_a_wrong_id_is_ignored(tmp_path: Path) -> None:
    f = Fixture(tmp_path)
    with f.client.websocket_connect("/ws") as ws:
        confirm = _ask_look(f, ws)
        ws.send_json({"type": "confirm_answer", "id": "0" * 16, "yes": True})
        ws.send_json({"type": "confirm_answer", "id": confirm["id"], "yes": True})
        msgs = _until_idle(ws)
    assert {"type": "confirm_done", "id": confirm["id"], "answer": "yes"} in msgs
    assert f.looker.questions == ["What am I holding?"]
    assert not any(m.get("type") == "error" for m in msgs)


def test_a_bad_confirm_answer_gets_an_error(tmp_path: Path) -> None:
    f = Fixture(tmp_path)
    with f.client.websocket_connect("/ws") as ws:
        confirm = _ask_look(f, ws)
        ws.send_json({"type": "confirm_answer", "id": confirm["id"], "yes": "sure"})
        assert ws.receive_json() == {"type": "error", "message": MSG_BAD_CONFIRM}
        ws.send_json({"type": "confirm_answer", "id": confirm["id"], "yes": False})
        _until_idle(ws)
    assert f.looker.questions == []


def test_typed_yes_while_waiting_answers_it(tmp_path: Path) -> None:
    f = Fixture(tmp_path)
    with f.client.websocket_connect("/ws") as ws:
        _ask_look(f, ws)
        ws.send_json({"type": "text", "text": "Yes please"})
        msgs = _until_idle(ws)
    assert f.looker.questions == ["What am I holding?"]
    assert {"type": "error", "message": MSG_BUSY} not in msgs
    # The answer is not a turn of its own.
    assert len(f.llm.requests) == 2


def test_another_answer_counts_as_no_and_runs_next_exactly_once(tmp_path: Path) -> None:
    f = Fixture(tmp_path, llm=FakeToolLLM(_look_then_reply("It is noon.")))
    with f.client.websocket_connect("/ws") as ws:
        _ask_look(f, ws)
        ws.send_json({"type": "text", "text": "what time is it"})
        first = _until_idle(ws)
        second = _until_idle(ws)
    assert f.looker.questions == []
    assert f.llm.requests[1].tool_results[0].response == {"error": msg_declined("Camera")}
    assert len(f.llm.requests) == 3
    assert f.llm.requests[2].history[-1].text == "what time is it"
    assert any(m.get("text") == "It is noon." and m.get("final") for m in second)
    assert not any(m.get("text") == "It is noon." for m in first)


def test_push_to_talk_while_waiting_answers_it(tmp_path: Path) -> None:
    stt = FakeSTT(text="Yes.")
    f = Fixture(tmp_path, stt=stt)
    with f.client.websocket_connect("/ws") as ws:
        _ask_look(f, ws)
        ws.send_json({"type": "audio_start"})
        assert ws.receive_json() == LISTENING
        for _ in range(5):
            ws.send_bytes(FRAME)
        ws.send_json({"type": "audio_end"})
        msgs = _until_idle(ws)
    assert len(stt.calls) == 1
    assert {"type": "transcript", "role": "user", "text": "Yes.", "final": True} in msgs
    assert f.looker.questions == ["What am I holding?"]
    assert len(f.llm.requests) == 2


def test_outside_a_confirm_text_during_a_turn_is_still_busy(tmp_path: Path) -> None:
    f = Fixture(tmp_path, looker=FakeLooker(delay_s=0.3))
    with f.client.websocket_connect("/ws") as ws:
        _connect(ws)
        ws.send_json({"type": "text", "text": "What am I holding?"})
        _until(ws, "state")  # thinking: the turn runs
        ws.send_json({"type": "text", "text": "and now?"})
        msgs = _until_idle(ws)
    assert {"type": "error", "message": MSG_BUSY} in msgs
    assert len(f.llm.requests) == 2


# --- activity, photos -------------------------------------------------------------------


def test_activity_lists_the_runs_without_content(tmp_path: Path) -> None:
    f = Fixture(tmp_path)
    with f.client.websocket_connect("/ws") as ws:
        _connect(ws)
        ws.send_json({"type": "text", "text": "What am I holding?"})
        _until_idle(ws)
    entries = f.client.get("/activity").json()["entries"]
    assert [(e["tool"], e["mode"], e["outcome"]) for e in entries] == [("look", "on", "ok")]
    assert set(entries[0]) == {"ts", "tool", "mode", "outcome"}
    assert "holding" not in (tmp_path / "activity.jsonl").read_text(encoding="utf-8")


def test_a_declined_ask_is_in_the_activity(tmp_path: Path) -> None:
    f = Fixture(tmp_path)
    with f.client.websocket_connect("/ws") as ws:
        confirm = _ask_look(f, ws)
        ws.send_json({"type": "confirm_answer", "id": confirm["id"], "yes": False})
        _until_idle(ws)
    entries = f.client.get("/activity").json()["entries"]
    assert [(e["tool"], e["mode"], e["outcome"]) for e in entries] == [("look", "ask", "declined")]


def test_activity_is_pruned_at_startup(tmp_path: Path) -> None:
    path = tmp_path / "activity.jsonl"
    now = datetime.now().astimezone()
    old = (now - timedelta(days=40)).isoformat(timespec="seconds")
    new = now.isoformat(timespec="seconds")
    lines = [
        {"ts": old, "tool": "look", "mode": "on", "outcome": "ok"},
        {"ts": new, "tool": "calculate", "mode": "on", "outcome": "ok"},
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    settings = _settings(tmp_path)
    settings = dataclasses.replace(settings, memory=dataclasses.replace(settings.memory, retention_days=30))
    app = create_app(
        settings,
        llm=FakeToolLLM([]),
        usage=UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap),
        settings_store=_store(tmp_path, settings),
        activity=ActivityLog(path, 30),
    )
    with TestClient(app) as client:
        entries = client.get("/activity").json()["entries"]
    assert [e["tool"] for e in entries] == ["calculate"]


def _failing_replace(src: Any, dst: Any) -> None:
    raise PermissionError("read-only")


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        ({"type": "permission", "tool": "look", "mode": "off"}, "permissions"),
        ({"type": "permission", "tool": "recall", "cloud": True}, "permissions"),
        ({"type": "photos", "mode": "disk"}, "photos"),
    ],
)
def test_a_change_that_cannot_be_saved_keeps_the_page_connected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], kind: str
) -> None:
    f = Fixture(tmp_path)
    with f.client.websocket_connect("/ws") as ws:
        before = dict(zip(("permissions", "photos"), _connect(ws)))
        monkeypatch.setattr(permissions.os, "replace", _failing_replace)
        ws.send_json(payload)
        assert ws.receive_json() == {"type": "error", "message": MSG_NOT_SAVED}
        # The page is told the unchanged choice, so its switch snaps back.
        assert ws.receive_json() == before[kind]
        monkeypatch.undo()
        ws.send_json({"type": "permission", "tool": "look", "mode": "ask"})
        assert _rows(ws.receive_json())["look"]["mode"] == "ask"


def test_photos_are_stored_by_the_app_and_sent_to_every_page(tmp_path: Path) -> None:
    f = Fixture(tmp_path)
    with f.client:  # one event loop for every page, as in the app
        with f.client.websocket_connect("/ws") as one, f.client.websocket_connect("/ws") as two:
            _connect(one)
            _connect(two)
            one.send_json({"type": "photos", "mode": "disk"})
            assert one.receive_json() == {"type": "photos", "mode": "disk"}
            assert two.receive_json() == {"type": "photos", "mode": "disk"}
        assert json.loads((tmp_path / "settings.json").read_text(encoding="utf-8"))["photos"] == "disk"
        with f.client.websocket_connect("/ws") as three:
            _, photos = _connect(three)
        assert photos == {"type": "photos", "mode": "disk"}


def test_camera_off_also_keeps_image_search_away(tmp_path: Path) -> None:
    f = Fixture(tmp_path, llm=FakeToolLLM([[TextDelta("Hi."), Done(1, 1)]]), looker=FakeLooker(search=True))
    with f.client.websocket_connect("/ws") as ws:
        first, _ = _connect(ws)
        assert _rows(first)["search_image"]["available"] is True
        ws.send_json({"type": "permission", "tool": "look", "mode": "off"})
        rows = _rows(ws.receive_json())
        ws.send_json({"type": "text", "text": "what do you see?"})
        _until_idle(ws)
    assert rows["search_image"]["available"] is False
    assert rows["search_image"]["missing"] == "needs Camera on"
    assert rows["search_image"]["mode"] == "ask"  # its own choice is kept
    assert [t.name for t in f.llm.requests[0].tools] == ["calculate"]
