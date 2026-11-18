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
    MSG_MEMORY_BAD,
    MSG_MEMORY_MISSING,
    MSG_MEMORY_OFF,
    MSG_MEMORY_STALE,
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


# Memory editor: GET /conversations/{id} and the "memory" WebSocket message.


def _saved_id(msgs: list[dict[str, Any]]) -> str:
    return str(next(m["id"] for m in msgs if m["type"] == "conversation" and m["reason"] == "saved"))


def _memory(ws: Any, **fields: Any) -> dict[str, Any]:
    ws.send_json({"type": "memory", **fields})
    return ws.receive_json()  # type: ignore[no-any-return]


def _stored_turns(client: TestClient, cid: str) -> list[dict[str, Any]]:
    return client.get(f"/conversations/{cid}").json()["turns"]  # type: ignore[no-any-return]


def test_conversation_turns_endpoint(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        cid = _saved_id(_turn(ws, "first"))
        _turn(ws, "second")
    body = client.get(f"/conversations/{cid}").json()
    assert body["enabled"] is True and body["id"] == cid and body["title"] == "first"
    assert [t["turn"] for t in body["turns"]] == [1, 2]
    first = body["turns"][0]
    assert isinstance(first["ts"], float)
    assert [(m["role"], m["text"]) for m in first["messages"]] == [("user", "first"), ("assistant", "Hi.")]
    assert set(first["messages"][0]) == {"id", "role", "text", "tool_name", "tool_arg"}
    assert client.get("/conversations/nope").status_code == 404
    assert client.get("/conversations/" + "0" * 32).status_code == 404


def _save(cid: str, expect: int, **changes: Any) -> dict[str, Any]:
    return {"action": "save", "conversation": cid, "expect_turns": expect, **changes}


def test_save_updates_the_open_history_and_sends_edited(tmp_path: Path) -> None:
    client, llm = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        cid = _saved_id(_turn(ws, "my cat is Rex"))
        user_id = _stored_turns(client, cid)[0]["messages"][0]["id"]
        ws.send_json({"type": "memory", **_save(cid, 1, edits=[{"message": user_id, "text": "my cat is Pita"}])})
        assert ws.receive_json() == {"type": "memory", "action": "save", "conversation": cid}
        assert ws.receive_json()["reason"] == "edited"
        _turn(ws, "what is my cat called?")
    sent = " ".join(m.text for m in llm.calls[-1])
    assert "my cat is Pita" in sent and "Rex" not in sent


def test_save_refused_mid_turn_for_the_open_conversation(tmp_path: Path) -> None:
    gate = threading.Event()
    client, _ = _client(tmp_path, llm=FakeLLM(["a", "b", "c"], gate=gate))
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        gate.set()
        cid = _saved_id(_turn(ws, "one"))
        mid = _stored_turns(client, cid)[0]["messages"][0]["id"]
        # A second stored conversation, not the open one.
        ws.send_json({"type": "conversation", "id": None})
        ws.receive_json()
        other_id = _saved_id(_turn(ws, "elsewhere"))
        other_mid = _stored_turns(client, other_id)[0]["messages"][0]["id"]
        ws.send_json({"type": "conversation", "id": cid})
        ws.receive_json()
        gate.clear()
        ws.send_json({"type": "text", "text": "hi"})
        assert ws.receive_json() == {"type": "state", "state": "thinking"}
        ws.send_json({"type": "memory", **_save(cid, 1, edits=[{"message": mid, "text": "changed"}])})
        while (msg := ws.receive_json())["type"] != "error":
            pass
        assert msg == {"type": "error", "message": MSG_CONVERSATION_BUSY}
        ws.send_json({"type": "memory", **_save(other_id, 1, edits=[{"message": other_mid, "text": "changed"}])})
        while (msg := ws.receive_json())["type"] != "memory":
            pass
        assert msg == {"type": "memory", "action": "save", "conversation": other_id}
        gate.set()
        while ws.receive_json() != IDLE:
            pass
    assert _stored_turns(client, cid)[0]["messages"][0]["text"] == "one"
    assert _stored_turns(client, other_id)[0]["messages"][0]["text"] == "changed"


def test_deleting_the_open_conversation_starts_a_new_one(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        cid = _saved_id(_turn(ws, "one"))
        ws.send_json({"type": "memory", "action": "delete_conversation", "conversation": cid})
        assert ws.receive_json() == {"type": "memory", "action": "delete_conversation", "conversation": cid}
        new = ws.receive_json()
        assert new["reason"] == "new" and new["id"] is None
        new_id = _saved_id(_turn(ws, "two"))
    assert new_id != cid
    assert client.get(f"/conversations/{cid}").status_code == 404


def test_save_adds_turns_and_renames(tmp_path: Path) -> None:
    client, llm = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        cid = _saved_id(_turn(ws, "one"))
        ws.send_json({"type": "memory", **_save(cid, 1, title="Tea talk", add_turns=[{"user": "I like tea", "reply": "Noted."}])})
        assert ws.receive_json() == {"type": "memory", "action": "save", "conversation": cid}
        edited = ws.receive_json()
        assert edited["reason"] == "edited" and edited["title"] == "Tea talk" and edited["turns"] == 2
        _turn(ws, "again")
    body = client.get(f"/conversations/{cid}").json()
    assert body["title"] == "Tea talk"
    assert [m["text"] for m in body["turns"][1]["messages"]] == ["I like tea", "Noted."]
    assert Message("user", "I like tea") in llm.calls[-1]
    assert _rows(tmp_path, "select provider, model from messages where text = 'Noted.'") == [("edited", "")]


def test_save_stale_gives_the_stale_message(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        cid = _saved_id(_turn(ws, "keep me"))
        assert _memory(ws, **_save(cid, 2, delete_turns=[1])) == {"type": "error", "message": MSG_MEMORY_STALE}
    turns = _stored_turns(client, cid)
    assert len(turns) == 1 and turns[0]["messages"][0]["text"] == "keep me"


def test_bad_save_payloads_are_refused(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        cid = _saved_id(_turn(ws, "keep me"))
        mid = _stored_turns(client, cid)[0]["messages"][0]["id"]
        ok_edit = {"message": mid, "text": "x"}
        ok_add = {"user": "u", "reply": "r"}
        base: dict[str, Any] = {"action": "save", "conversation": cid, "expect_turns": 1}
        bad: list[dict[str, Any]] = [
            {"action": "explode"},
            {"action": "save", "conversation": cid},
            {**base, "expect_turns": True},
            {**base, "expect_turns": 0},
            {**base, "conversation": "../etc"},
            {**base, "edits": "x"},
            {**base, "edits": [{"message": 2**70, "text": "x"}]},
            {**base, "edits": [{"message": str(mid), "text": "x"}]},
            {**base, "edits": [{"message": mid, "text": " "}]},
            {**base, "edits": [{"message": mid, "text": "x" * 4001}]},
            {**base, "edits": [ok_edit] * 501},
            {**base, "delete_turns": [True]},
            {**base, "delete_turns": [0]},
            {**base, "delete_turns": [2**70]},
            {**base, "delete_turns": [1] * 501},
            {**base, "delete_turns": "1"},
            {**base, "add_turns": [ok_add] * 51},
            {**base, "add_turns": [{"user": "", "reply": "r"}]},
            {**base, "add_turns": [{"user": "u", "reply": "r" * 4001}]},
            {**base, "title": 5},
            {**base, "title": "t" * 4001},
        ]
        for fields in bad:
            assert _memory(ws, **fields) == {"type": "error", "message": MSG_MEMORY_BAD}, fields
        missing = _memory(ws, **_save(cid, 1, edits=[{"message": 99999, "text": "x"}]))
        assert missing == {"type": "error", "message": MSG_MEMORY_MISSING}
        assert _memory(ws, **_save("0" * 32, 1)) == {"type": "error", "message": MSG_MEMORY_MISSING}
        ws.send_json({"type": "conversation", "id": None})
        assert ws.receive_json()["reason"] == "new"
    turns = _stored_turns(client, cid)
    assert [m["text"] for m in turns[0]["messages"]] == ["keep me", "Hi."] and len(turns) == 1


def test_old_per_change_actions_are_unknown(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        cid = _saved_id(_turn(ws, "keep me"))
        mid = _stored_turns(client, cid)[0]["messages"][0]["id"]
        old: list[dict[str, Any]] = [
            {"action": "edit", "message": mid, "text": "x"},
            {"action": "delete_turn", "conversation": cid, "turn": 1},
            {"action": "add_turn", "conversation": cid, "user": "u", "reply": "r"},
            {"action": "rename", "conversation": cid, "title": "t"},
        ]
        for fields in old:
            assert _memory(ws, **fields) == {"type": "error", "message": MSG_MEMORY_BAD}, fields
    turns = _stored_turns(client, cid)
    assert len(turns) == 1 and turns[0]["messages"][0]["text"] == "keep me"


def test_delete_conversation_payload_checks(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert _memory(ws, action="delete_conversation", conversation="../etc") == {
            "type": "error", "message": MSG_MEMORY_BAD,
        }
        assert _memory(ws, action="delete_conversation", conversation="0" * 32) == {
            "type": "error", "message": MSG_MEMORY_MISSING,
        }


def test_memory_messages_with_memory_off(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    client = TestClient(create_app(settings, llm=FakeLLM(["Hi."]), usage=usage))
    assert client.get("/conversations/" + "0" * 32).json() == {
        "enabled": False, "id": "", "title": "", "turns": [],
    }
    with client.websocket_connect("/ws") as ws:
        for fields in (_save("0" * 32, 1), {"action": "bogus"}):
            assert _memory(ws, **fields) == {"type": "error", "message": MSG_MEMORY_OFF}


def test_deleted_open_conversation_is_replaced_on_the_next_turn(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        cid = _saved_id(_turn(ws, "one"))
        # Another tab deletes it.
        with client.websocket_connect("/ws") as other:
            other.receive_json()
            other.send_json({"type": "memory", "action": "delete_conversation", "conversation": cid})
            assert other.receive_json()["type"] == "memory"
        msgs = _turn(ws, "two")
        saved = [m for m in msgs if m["type"] == "conversation"]
        assert len(saved) == 1 and saved[0]["reason"] == "saved"
        assert saved[0]["id"] != cid and saved[0]["turns"] == 1
    assert client.get(f"/conversations/{saved[0]['id']}").json()["turns"][0]["messages"][0]["text"] == "two"


def test_save_added_turns_keep_the_conversation_order(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, settings=_settings(tmp_path, resume_minutes=0))
    ids = []
    for text in ("older chat", "newer chat"):
        with client.websocket_connect("/ws") as ws:
            ws.receive_json()
            ids.append(_saved_id(_turn(ws, text)))
            time.sleep(0.01)
    before = client.get("/conversations").json()["conversations"]
    assert [c["id"] for c in before] == [ids[1], ids[0]]
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "memory", **_save(ids[0], 1, add_turns=[{"user": "u", "reply": "r"}])})
        assert ws.receive_json()["type"] == "memory"
    after = client.get("/conversations").json()["conversations"]
    assert [c["id"] for c in after] == [ids[1], ids[0]]
    assert after[1]["last_at"] == before[1]["last_at"] and after[1]["turns"] == 2


def test_save_deleting_the_last_turn_of_the_open_conversation_starts_a_new_one(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        cid = _saved_id(_turn(ws, "only"))
        ws.send_json({"type": "memory", **_save(cid, 1, delete_turns=[1])})
        assert ws.receive_json() == {"type": "memory", "action": "save", "conversation": cid}
        new = ws.receive_json()
        assert new["reason"] == "new" and new["id"] is None
    assert client.get(f"/conversations/{cid}").status_code == 404
