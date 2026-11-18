"""The Photos setting through the app (Phase 5f): the WS switch, photos saved
to disk with their conversation, and given back on resume. No camera, no model."""

import dataclasses
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from fakes import FakeLooker, SeeingToolLLM, text_only
from obsilord.app import MSG_BAD_PHOTOS, create_app
from obsilord.config import REPO_ROOT, Settings, load_settings
from obsilord.llm.base import Done, TextDelta, ToolCall
from obsilord.memory.store import SqliteStore
from obsilord.photos import PhotoStore
from obsilord.usage import UsageTracker

IDLE = {"type": "state", "state": "idle"}
FRAME = b"photo-jpeg"


def _settings(tmp_path: Path, **camera: Any) -> Settings:
    s = text_only(load_settings(REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}))
    return dataclasses.replace(s, camera=dataclasses.replace(s.camera, **camera), repo_root=REPO_ROOT)


def _look_rounds(n: int) -> list[list[Any]]:
    rounds: list[list[Any]] = []
    for _ in range(n):
        rounds += [[ToolCall("look", {"question": "q"}, call_id="c1"), Done(1, 1)], [TextDelta("A mug."), Done(1, 1)]]
    return rounds


def _app(tmp_path: Path, llm: Any, settings: Settings | None = None, store: bool = True) -> Any:
    settings = settings or _settings(tmp_path)
    return create_app(
        settings,
        llm=llm,
        usage=UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap),
        looker=FakeLooker(),
        memory=SqliteStore(tmp_path / "m.sqlite3") if store else None,
        photos=PhotoStore(tmp_path / "photos"),
    )


def _turn(ws: Any, text: str = "look at this") -> list[dict[str, Any]]:
    ws.send_json({"type": "text", "text": text})
    msgs = []
    while (msg := ws.receive_json()) != IDLE:
        msgs.append(msg)
    return msgs


def _set(ws: Any, mode: str) -> dict[str, Any]:
    ws.send_json({"type": "photos", "mode": mode})
    return ws.receive_json()


def _saved(tmp_path: Path) -> list[Path]:
    root = tmp_path / "photos"
    return sorted(root.rglob("*.jpg")) if root.exists() else []


def test_the_page_gets_the_configured_default(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path, SeeingToolLLM([]), _settings(tmp_path, photos="followup", photos_keep=3)))
    assert client.get("/config/photos").json() == {"default": "followup", "keep": 3}


def test_the_switch_answers_with_the_mode(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, SeeingToolLLM([]))).websocket_connect("/ws") as ws:
        ws.receive_json()  # conversation
        assert _set(ws, "disk") == {"type": "photos", "mode": "disk"}
        assert _set(ws, "nope") == {"type": "error", "message": MSG_BAD_PHOTOS}


def test_off_saves_nothing(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, SeeingToolLLM(_look_rounds(1)))).websocket_connect("/ws") as ws:
        ws.receive_json()
        _turn(ws)
    assert _saved(tmp_path) == []


def test_disk_saves_the_photo_with_its_conversation(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, SeeingToolLLM(_look_rounds(1)))).websocket_connect("/ws") as ws:
        ws.receive_json()
        _set(ws, "disk")
        msgs = _turn(ws)
    cid = next(m["id"] for m in msgs if m["type"] == "conversation")
    files = _saved(tmp_path)
    assert [p.parent.name for p in files] == [cid]
    assert files[0].read_bytes() == FRAME
    con = sqlite3.connect(tmp_path / "m.sqlite3")
    try:
        meta = con.execute("select meta from messages where role = 'tool'").fetchone()[0]
    finally:
        con.close()
    assert json.loads(meta) == {"photo": files[0].name}


def test_disk_without_memory_uses_its_own_folder(tmp_path: Path) -> None:
    app = _app(tmp_path, SeeingToolLLM(_look_rounds(1)), store=False)
    with TestClient(app).websocket_connect("/ws") as ws:
        _set(ws, "disk")
        _turn(ws)
    assert [p.parent.name for p in _saved(tmp_path)] == ["no-conversation"]


def test_a_resumed_conversation_gets_its_saved_photos_back(tmp_path: Path) -> None:
    llm = SeeingToolLLM([*_look_rounds(1), [TextDelta("Still a mug."), Done(1, 1)]])
    client = TestClient(_app(tmp_path, llm))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _set(ws, "disk")
        _turn(ws)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["resumed"] is True
        _set(ws, "disk")
        _turn(ws, "what color was it?")
    history = llm.requests[-1].history
    assert history[1].tools[0].image == FRAME


def test_switching_to_disk_during_a_turn_loads_photos_for_the_next_turn(tmp_path: Path) -> None:
    llm = SeeingToolLLM([*_look_rounds(2), [TextDelta("Pink."), Done(1, 1)]])
    looker = FakeLooker(delay_s=0.2)  # the second turn is still looking when the switch arrives
    settings = _settings(tmp_path)
    app = create_app(
        settings,
        llm=llm,
        usage=UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap),
        looker=looker,
        memory=SqliteStore(tmp_path / "m.sqlite3"),
        photos=PhotoStore(tmp_path / "photos"),
    )
    client = TestClient(app)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _set(ws, "disk")
        _turn(ws)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()  # resumed, Photos "off" (the default)
        ws.send_json({"type": "text", "text": "and now?"})
        ws.send_json({"type": "photos", "mode": "disk"})
        msgs = []
        while (msg := ws.receive_json()) != IDLE:
            msgs.append(msg)
        assert {"type": "photos", "mode": "disk"} in msgs
        _turn(ws, "what color was it?")
    history = llm.requests[-1].history
    # The turn that ran during the switch kept its own messages in order.
    assert [m.text for m in history] == ["look at this", "A mug.", "and now?", "A mug.", "what color was it?"]
    assert history[1].tools[0].image == FRAME


def test_followup_does_not_bring_photos_back_from_disk(tmp_path: Path) -> None:
    llm = SeeingToolLLM([*_look_rounds(1), [TextDelta("Hm."), Done(1, 1)]])
    client = TestClient(_app(tmp_path, llm))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _set(ws, "disk")
        _turn(ws)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _set(ws, "followup")
        _turn(ws, "what color was it?")
    assert llm.requests[-1].history[1].tools[0].image is None


def test_switching_off_drops_the_kept_photos(tmp_path: Path) -> None:
    llm = SeeingToolLLM([*_look_rounds(1), [TextDelta("Hm."), Done(1, 1)]])
    with TestClient(_app(tmp_path, llm)).websocket_connect("/ws") as ws:
        ws.receive_json()
        _set(ws, "followup")
        _turn(ws)
        _set(ws, "off")
        _turn(ws, "what color was it?")
    assert llm.requests[-1].history[1].tools[0].image is None


def test_startup_removes_photos_past_the_retention(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = dataclasses.replace(settings, memory=dataclasses.replace(settings.memory, retention_days=30))
    old = tmp_path / "photos" / ("a" * 32) / "20260101-000000-000.jpg"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"x")
    os.utime(old, (time.time() - 40 * 86400,) * 2)
    with TestClient(_app(tmp_path, SeeingToolLLM([]), settings)):
        pass
    assert not old.exists()


def _two_look_turns(tmp_path: Path) -> tuple[TestClient, str]:
    client = TestClient(_app(tmp_path, SeeingToolLLM(_look_rounds(2))))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _set(ws, "disk")
        _turn(ws, "first look")
        msgs = _turn(ws, "second look")
    cid = next(m["id"] for m in msgs if m["type"] == "conversation")
    assert len(_saved(tmp_path)) == 2
    return client, str(cid)


def _memory(client: TestClient, **fields: Any) -> dict[str, Any]:
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "memory", **fields})
        return ws.receive_json()  # type: ignore[no-any-return]


def test_deleting_a_conversation_removes_its_photos(tmp_path: Path) -> None:
    client, cid = _two_look_turns(tmp_path)
    assert _memory(client, action="delete_conversation", conversation=cid)["type"] == "memory"
    assert _saved(tmp_path) == [] and not (tmp_path / "photos" / cid).exists()


def test_deleting_a_turn_removes_only_its_photos(tmp_path: Path) -> None:
    client, cid = _two_look_turns(tmp_path)
    first, second = _saved(tmp_path)
    con = sqlite3.connect(tmp_path / "m.sqlite3")
    try:
        metas = con.execute("select turn, meta from messages where role = 'tool' order by turn").fetchall()
    finally:
        con.close()
    names = {t: json.loads(m)["photo"] for t, m in metas}
    assert _memory(client, action="delete_turn", conversation=cid, turn=1)["type"] == "memory"
    assert [p.name for p in _saved(tmp_path)] == [names[2]]


def test_deleting_the_last_turn_removes_the_conversation_folder(tmp_path: Path) -> None:
    client = TestClient(_app(tmp_path, SeeingToolLLM(_look_rounds(1))))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _set(ws, "disk")
        msgs = _turn(ws)
    cid = str(next(m["id"] for m in msgs if m["type"] == "conversation"))
    assert len(_saved(tmp_path)) == 1
    assert _memory(client, action="delete_turn", conversation=cid, turn=1)["type"] == "memory"
    assert not (tmp_path / "photos" / cid).exists()
