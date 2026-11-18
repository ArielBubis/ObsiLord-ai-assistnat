"""SqliteStore and fts_query on a temporary database. No network."""

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any, Awaitable, TypeVar

import pytest

from obsilord.llm.base import Message, ToolNote
from obsilord.memory.store import SCHEMA_VERSION, SqliteStore, TurnRecord, fts_query

T = TypeVar("T")
DAY = 86400.0
T0 = 1_763_467_200.0  # 2025-11-18 12:00 UTC


def run(coro: Awaitable[T]) -> T:
    return asyncio.run(coro)  # type: ignore[arg-type]


def turn(user: str, reply: str, ts: float, notes: tuple[ToolNote, ...] = ()) -> TurnRecord:
    return TurnRecord(user, reply, notes, provider="local", model="local:qwen", ts=ts)


async def _store(path: Path) -> SqliteStore:
    store = SqliteStore(path)
    await store.open()
    return store


def test_fts_query_quotes_tokens_as_prefix_terms() -> None:
    assert fts_query("Shakshuka, feta?") == '"shakshuka"* OR "feta"*'


def test_fts_query_drops_operators_and_short_tokens() -> None:
    q = fts_query('a AND "b" NEAR( x* owl')
    assert q == '"and"* OR "near"* OR "owl"*'


def test_fts_query_none_when_nothing_is_left() -> None:
    assert fts_query("") is None
    assert fts_query(" ? ! * ") is None
    assert fts_query(None) is None  # type: ignore[arg-type]


def test_fts_query_caps_tokens_and_keeps_unicode() -> None:
    q = fts_query(" ".join(f"word{i}" for i in range(30)))
    assert q is not None and q.count(" OR ") == 11
    assert fts_query("Pão café") == '"pão"* OR "café"*'


def test_migrations_run_once(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite3"

    async def go() -> None:
        store = await _store(path)
        await store.close()
        store = await _store(path)
        await store.close()

    run(go())
    con = sqlite3.connect(path)
    assert con.execute("pragma user_version").fetchone()[0] == SCHEMA_VERSION == 1
    con.close()


def test_open_creates_the_parent_folder(tmp_path: Path) -> None:
    async def go() -> None:
        store = await _store(tmp_path / "data" / "deep" / "m.sqlite3")
        await store.close()

    run(go())
    assert (tmp_path / "data" / "deep" / "m.sqlite3").exists()


def test_add_turn_then_last_turns_rebuilds_notes(tmp_path: Path) -> None:
    note = ToolNote("look", "what is this", "A blue owl.")
    err = ToolNote("search_image", "the owl", "Search is off.", ok=False)

    async def go() -> list[Message]:
        store = await _store(tmp_path / "m.sqlite3")
        cid = await store.new_conversation("web", now=T0)
        await store.add_turn(cid, turn("hi", "Hello.", T0))
        await store.add_turn(cid, turn("look", "An owl.", T0 + 5, (note, err)))
        await store.add_turn(cid, turn("thanks", "Sure.", T0 + 9))
        got = await store.last_turns(cid, 2)
        await store.close()
        return got

    assert run(go()) == [
        Message("user", "look"),
        Message("assistant", "An owl.", tools=(note, err)),
        Message("user", "thanks"),
        Message("assistant", "Sure."),
    ]


def test_a_photo_name_is_stored_and_the_image_never_is(tmp_path: Path) -> None:
    note = ToolNote("look", "q", "Photo taken.", photo="20261118-204203-250.jpg", image=b"JPEGDATA")
    failed = ToolNote("look", "q", "Camera failed.", ok=False, photo="20261118-204204-000.jpg")

    async def go() -> list[Message]:
        store = await _store(tmp_path / "m.sqlite3")
        cid = await store.new_conversation("web", now=T0)
        await store.add_turn(cid, turn("look", "A mug.", T0, (note, failed)))
        got = await store.last_turns(cid, 1)
        await store.close()
        return got

    notes = run(go())[1].tools
    assert [(n.photo, n.ok, n.image) for n in notes] == [
        ("20261118-204203-250.jpg", True, None),
        ("20261118-204204-000.jpg", False, None),
    ]
    assert b"JPEGDATA" not in (tmp_path / "m.sqlite3").read_bytes()


def test_conversation_info_and_order(tmp_path: Path) -> None:
    async def go() -> list[Any]:
        store = await _store(tmp_path / "m.sqlite3")
        a = await store.new_conversation("web", now=T0)
        await store.add_turn(a, turn("first line " + "x" * 200, "ok", T0))
        await store.add_turn(a, turn("second", "ok", T0 + 60))
        b = await store.new_conversation("web", now=T0 + 100)
        await store.add_turn(b, turn("other", "ok", T0 + 100))
        await store.new_conversation("web", now=T0 + 200)  # empty: not listed
        infos = await store.list_conversations(10)
        await store.close()
        return infos

    infos = run(go())
    assert [i.title for i in infos] == ["other", ("first line " + "x" * 200)[:80]]
    assert infos[1].turns == 2
    assert infos[1].started_at == T0 and infos[1].last_at == T0 + 60


def test_latest_respects_the_window_and_channel(tmp_path: Path) -> None:
    async def go() -> tuple[Any, Any, Any, Any]:
        store = await _store(tmp_path / "m.sqlite3")
        cid = await store.new_conversation("web", now=T0)
        await store.add_turn(cid, turn("hi", "ok", T0))
        tg = await store.new_conversation("telegram", now=T0 + 10)
        await store.add_turn(tg, turn("hi", "ok", T0 + 10))
        inside = await store.latest("web", 1800, now=T0 + 1799)
        outside = await store.latest("web", 1800, now=T0 + 1801)
        other = await store.latest("telegram", 1800, now=T0 + 20)
        exists = await store.exists(cid), await store.exists("nope")
        await store.close()
        return inside, outside, other, exists

    inside, outside, other, exists = run(go())
    assert inside is not None and outside is None and other is not None and inside != other
    assert exists == (True, False)


def _seed(store: SqliteStore) -> Awaitable[None]:
    async def go() -> None:
        a = await store.new_conversation("web", now=T0 - 2 * DAY)
        await store.add_turn(
            a,
            turn(
                "what is this figure?",
                "Looks like a collectible.",
                T0 - 2 * DAY,
                (ToolNote("search_image", "figure", "Online lookup: Funko Pop Miku #39"),),
            ),
        )
        b = await store.new_conversation("web", now=T0 - DAY)
        await store.add_turn(b, turn("I want shakshuka tonight", "You are out of feta.", T0 - DAY))
        c = await store.new_conversation("web", now=T0 - 40 * DAY)
        await store.add_turn(c, turn("trip to Lisbon in spring?", "Lisbon is lovely in April.", T0 - 40 * DAY))

    return go()


def test_search_finds_tool_rows_with_their_turn(tmp_path: Path) -> None:
    async def go() -> list[Any]:
        store = await _store(tmp_path / "m.sqlite3")
        await _seed(store)
        hits = await store.search("funko", 5)
        await store.close()
        return hits

    hits = run(go())
    assert len(hits) == 1
    hit = hits[0]
    assert hit.ts == T0 - 2 * DAY
    assert hit.user == "what is this figure?"
    assert hit.reply == "Looks like a collectible."
    assert hit.tools == ("search_image: Online lookup: Funko Pop Miku #39",)


def test_search_days_back_and_empty_query(tmp_path: Path) -> None:
    async def go() -> tuple[list[Any], list[Any], list[Any]]:
        store = await _store(tmp_path / "m.sqlite3")
        await _seed(store)
        old = await store.search("lisbon", 5, since=T0 - 7 * DAY)
        recent = await store.search("", 5, since=T0 - 3 * DAY)
        everything = await store.search("lisbon shakshuka", 5)
        await store.close()
        return old, recent, everything

    old, recent, everything = run(go())
    assert old == []
    assert [h.user for h in recent] == ["I want shakshuka tonight", "what is this figure?"]
    assert {h.user for h in everything} == {"I want shakshuka tonight", "trip to Lisbon in spring?"}


def test_search_until_bounds_the_window(tmp_path: Path) -> None:
    async def go() -> tuple[list[Any], list[Any]]:
        store = await _store(tmp_path / "m.sqlite3")
        await _seed(store)
        day = await store.search("", 5, since=T0 - 1.5 * DAY, until=T0 - 0.5 * DAY)
        words = await store.search("funko shakshuka", 5, until=T0 - 1.5 * DAY)
        await store.close()
        return day, words

    day, words = run(go())
    assert [h.user for h in day] == ["I want shakshuka tonight"]
    assert [h.user for h in words] == ["what is this figure?"]


def test_search_skips_the_turns_already_inline(tmp_path: Path) -> None:
    # skip=(conversation, n): that conversation's last n turns are in the
    # prompt already, so recall must not return them (they echo the miss).
    async def go() -> tuple[list[Any], list[Any], list[Any]]:
        store = await _store(tmp_path / "m.sqlite3")
        old = await store.new_conversation("web", now=T0 - DAY)
        await store.add_turn(old, turn("my cat is Pita", "Nice name.", T0 - DAY))
        cur = await store.new_conversation("web", now=T0)
        for i in range(3):
            await store.add_turn(cur, turn(f"cat question {i}", "No record of a cat.", T0 + i))
        words = await store.search("cat", 5, skip=(cur, 2))
        listing = await store.search("", 5, skip=(cur, 2))
        other = await store.search("cat", 5, skip=("f" * 32, 2))
        await store.close()
        return words, listing, other

    words, listing, other = run(go())
    assert sorted(h.user for h in words) == ["cat question 0", "my cat is Pita"]
    assert [h.user for h in listing] == ["cat question 0", "my cat is Pita"]
    assert len(other) == 4


def test_search_one_hit_per_turn(tmp_path: Path) -> None:
    async def go() -> list[Any]:
        store = await _store(tmp_path / "m.sqlite3")
        await _seed(store)
        hits = await store.search("lisbon", 5)  # user and reply both match
        await store.close()
        return hits

    assert len(run(go())) == 1


def test_search_with_hostile_text_does_not_raise(tmp_path: Path) -> None:
    async def go() -> list[Any]:
        store = await _store(tmp_path / "m.sqlite3")
        await _seed(store)
        hits = await store.search('") OR 1=1; -- NEAR(*', 5)
        await store.close()
        return hits

    assert isinstance(run(go()), list)


def test_prune_keeps_the_index_in_sync(tmp_path: Path) -> None:
    async def go() -> tuple[list[Any], list[Any], int]:
        store = await _store(tmp_path / "m.sqlite3")
        await _seed(store)
        removed = await store.prune(30, now=T0)
        gone = await store.search("lisbon", 5)
        kept = await store.search("shakshuka", 5)
        await store.close()
        return gone, kept, removed

    gone, kept, removed = run(go())
    assert gone == [] and len(kept) == 1 and removed == 1
    con = sqlite3.connect(tmp_path / "m.sqlite3")
    assert con.execute("select count(*) from messages_fts where messages_fts match 'lisbon'").fetchone()[0] == 0
    assert con.execute("select count(*) from conversations").fetchone()[0] == 2
    con.close()


def test_prune_zero_keeps_everything(tmp_path: Path) -> None:
    async def go() -> int:
        store = await _store(tmp_path / "m.sqlite3")
        await _seed(store)
        removed = await store.prune(0, now=T0)
        await store.close()
        return removed

    assert run(go()) == 0


def test_reopen_keeps_the_data(tmp_path: Path) -> None:
    path = tmp_path / "m.sqlite3"

    async def go() -> list[Any]:
        store = await _store(path)
        await _seed(store)
        await store.close()
        store = await _store(path)
        hits = await store.search("feta", 5)
        await store.close()
        return hits

    assert len(run(go())) == 1


def test_cancelled_call_keeps_the_connection_to_itself(tmp_path: Path) -> None:
    """A call cancelled while its worker thread runs (page closed mid-write)
    must not let the next call use the connection at the same time."""
    first_started = threading.Event()
    release = threading.Event()
    overlap: list[bool] = []
    busy = threading.Event()

    def slow() -> None:
        busy.set()
        first_started.set()
        release.wait(5)
        busy.clear()

    def second() -> None:
        overlap.append(busy.is_set())

    async def go() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        task = asyncio.create_task(store._call(slow))
        await asyncio.to_thread(first_started.wait, 5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        follow = asyncio.create_task(store._call(second))
        await asyncio.sleep(0.1)
        release.set()
        await follow
        await store.close()

    run(go())
    assert overlap == [False]


def test_add_turn_to_an_unknown_conversation_raises(tmp_path: Path) -> None:
    async def go() -> None:
        store = await _store(tmp_path / "m.sqlite3")
        try:
            await store.add_turn("nope", turn("hi", "ok", T0))
        finally:
            await store.close()

    with pytest.raises(KeyError):
        run(go())
