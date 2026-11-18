"""The app with a conversation store: resume, the Conversations list, the WS
conversation switch, demo history, and recall offered to the local model only."""

import dataclasses
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from fakes import FakeLLM, text_only
from obsilord.app import (
    MSG_CONVERSATION_BUSY,
    MSG_MEMORY_OFF,
    MSG_UNKNOWN_CONVERSATION,
    _memory_from_settings,
    create_app,
    turn_tools,
)
from obsilord.config import DEMO_DB_PATH, REPO_ROOT, Settings, load_settings
from obsilord.llm.base import Message
from obsilord.memory.demo import DEMO_CONVERSATIONS
from obsilord.memory.recall import RECALL_TOOL
from obsilord.memory.store import SqliteStore
from obsilord.usage import UsageTracker

IDLE = {"type": "state", "state": "idle"}


def _settings(tmp_path: Path, **memory: Any) -> Settings:
    s = text_only(
        load_settings(REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={})
    )
    return dataclasses.replace(s, memory=dataclasses.replace(s.memory, **memory), repo_root=REPO_ROOT)


def _client(
    tmp_path: Path, llm: Any = None, store: Any = None, settings: Settings | None = None
) -> tuple[TestClient, Any]:
    settings = settings or _settings(tmp_path)
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    llm = llm or FakeLLM(["Hi."])
    store = store if store is not None else SqliteStore(tmp_path / "m.sqlite3")
    return TestClient(create_app(settings, llm=llm, usage=usage, memory=store)), llm


def _turn(ws: Any, text: str) -> list[dict[str, Any]]:
    ws.send_json({"type": "text", "text": text})
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg == IDLE:
            return msgs


def _rows(tmp_path: Path, sql: str) -> list[tuple[Any, ...]]:
    con = sqlite3.connect(tmp_path / "m.sqlite3")
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_first_connect_starts_a_new_conversation_and_stores_turns(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello == {
            "type": "conversation", "id": None, "title": "", "turns": 0, "last_at": None,
            "resumed": False, "reason": "start",
        }
        msgs = _turn(ws, "hello there")
        saved = [m for m in msgs if m["type"] == "conversation"]
        # The page learns the new conversation's id once its first turn is stored.
        assert len(saved) == 1 and saved[0]["reason"] == "saved"
        assert isinstance(saved[0]["id"], str) and saved[0]["turns"] == 1
        assert msgs.index(saved[0]) < msgs.index(IDLE)
        later = _turn(ws, "again")
        assert [m["turns"] for m in later if m["type"] == "conversation"] == [2]
    rows = _rows(tmp_path, "select role, text, provider, model from messages order by id")
    assert rows[:2] == [("user", "hello there", None, None), ("assistant", "Hi.", "gemini", "fake-model")]
    assert _rows(tmp_path, "select channel, turns, title from conversations") == [("web", 2, "hello there")]


def test_reconnect_within_the_window_continues(tmp_path: Path) -> None:
    client, llm = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _turn(ws, "my name is Ariel")
    with client.websocket_connect("/ws") as ws:
        hello = ws.receive_json()
        assert hello["resumed"] is True and hello["reason"] == "resume"
        assert hello["turns"] == 1 and hello["title"] == "my name is Ariel"
        assert isinstance(hello["id"], str) and isinstance(hello["last_at"], float)
        _turn(ws, "what is my name?")
    assert llm.calls[-1][:2] == [Message("user", "my name is Ariel"), Message("assistant", "Hi.")]
    assert _rows(tmp_path, "select count(*), max(turns) from conversations") == [(1, 2)]


def test_resume_off_starts_fresh(tmp_path: Path) -> None:
    client, llm = _client(tmp_path, settings=_settings(tmp_path, resume_minutes=0))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _turn(ws, "one")
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["resumed"] is False
        _turn(ws, "two")
    assert llm.calls[-1] == [Message("user", "two")]
    assert _rows(tmp_path, "select count(*) from conversations") == [(2,)]


def test_old_conversation_is_not_resumed(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _turn(ws, "long ago")
    con = sqlite3.connect(tmp_path / "m.sqlite3")
    con.execute("update conversations set last_at = ?", (time.time() - 31 * 60,))
    con.commit()
    con.close()
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["resumed"] is False


def test_conversations_list(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, settings=_settings(tmp_path, resume_minutes=0))
    for text in ("first chat", "second chat"):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            _turn(ws, text)
    body = client.get("/conversations").json()
    assert body["enabled"] is True
    assert [c["title"] for c in body["conversations"]] == ["second chat", "first chat"]
    assert set(body["conversations"][0]) == {"id", "title", "started_at", "last_at", "turns"}
    assert len(client.get("/conversations?limit=1").json()["conversations"]) == 1


def test_switch_conversation_loads_its_history(tmp_path: Path) -> None:
    client, llm = _client(tmp_path, settings=_settings(tmp_path, resume_minutes=0))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _turn(ws, "about Lisbon")
    old_id = client.get("/conversations").json()["conversations"][0]["id"]
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "conversation", "id": old_id})
        msg = ws.receive_json()
        assert msg["type"] == "conversation" and msg["id"] == old_id and msg["turns"] == 1
        assert msg["resumed"] is False and msg["reason"] == "switch"
        _turn(ws, "which city?")
        assert llm.calls[-1][0] == Message("user", "about Lisbon")
        ws.send_json({"type": "conversation", "id": None})
        new = ws.receive_json()
        assert new["id"] is None and new["reason"] == "new"
        _turn(ws, "fresh")
        assert llm.calls[-1] == [Message("user", "fresh")]


def test_switch_to_unknown_conversation(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for bad in ("0" * 32, "../etc", 5):
            ws.send_json({"type": "conversation", "id": bad})
            assert ws.receive_json() == {"type": "error", "message": MSG_UNKNOWN_CONVERSATION}


def test_switch_is_refused_during_a_turn(tmp_path: Path) -> None:
    gate = threading.Event()
    client, _ = _client(tmp_path, llm=FakeLLM(["a", "b"], gate=gate))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "text", "text": "hi"})
        assert ws.receive_json() == {"type": "state", "state": "thinking"}
        ws.send_json({"type": "conversation", "id": None})
        seen = []
        while True:
            msg = ws.receive_json()
            seen.append(msg)
            if msg == {"type": "error", "message": MSG_CONVERSATION_BUSY}:
                break
        gate.set()
        while ws.receive_json() != IDLE:
            pass


def test_memory_off(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    client = TestClient(create_app(settings, llm=FakeLLM(["Hi."]), usage=usage))  # conftest: no store
    assert client.get("/conversations").json() == {"enabled": False, "conversations": []}
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "conversation", "id": None})
        assert ws.receive_json() == {"type": "error", "message": MSG_MEMORY_OFF}


def test_store_that_fails_to_open_leaves_the_app_working(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    client, _ = _client(tmp_path, store=SqliteStore(blocker / "m.sqlite3"))
    with client.websocket_connect("/ws") as ws:
        msgs = _turn(ws, "hi")  # no conversation message first
        assert msgs[0] == {"type": "state", "state": "thinking"}
    assert client.get("/conversations").json()["enabled"] is False


def test_turn_tools_offer_recall_to_the_local_model_only() -> None:
    recall = object()
    assert turn_tools("local", recall) == [recall]  # type: ignore[comparison-overlap]
    assert turn_tools("gemini", recall) == []
    assert turn_tools("local", None) == []


def test_gemini_turn_declares_no_recall(tmp_path: Path) -> None:
    client, llm = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        _turn(ws, "hi")
    assert all(t.name != RECALL_TOOL.name for t in llm.tools[-1])


def test_memory_checks_404_without_demo(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    assert client.get("/debug/memory-checks").status_code == 404


def test_demo_memory_seeds_and_serves_checks(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings = dataclasses.replace(settings, debug=dataclasses.replace(settings.debug, demo_memory=True))
    client, _ = _client(tmp_path, settings=settings)
    checks = client.get("/debug/memory-checks").json()["checks"]
    assert checks and {"date", "label", "question", "expected", "kind"} == set(checks[0])
    assert len(client.get("/conversations").json()["conversations"]) == len(DEMO_CONVERSATIONS)
    with client.websocket_connect("/ws") as ws:
        assert ws.receive_json()["resumed"] is False  # newest demo turn is 2 h old


def test_demo_store_uses_the_demo_file_only(tmp_path: Path) -> None:
    s = _settings(tmp_path)
    s = dataclasses.replace(s, repo_root=tmp_path, debug=dataclasses.replace(s.debug, demo_memory=True))
    demo = tmp_path / DEMO_DB_PATH
    demo.parent.mkdir(parents=True)
    demo.write_bytes(b"old demo file")
    store = _memory_from_settings(s)
    assert isinstance(store, SqliteStore) and store.path == demo
    assert not demo.exists()  # deleted for a fresh seed
    assert not (tmp_path / "data" / "memory.sqlite3").exists()


def test_memory_disabled_in_settings_builds_no_store(tmp_path: Path) -> None:
    assert _memory_from_settings(_settings(tmp_path, enabled=False)) is None
