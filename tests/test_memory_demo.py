"""Demo history: seeding, relative dates, and a check list that matches it."""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

from obsilord.memory.demo import (
    DEMO_CONVERSATIONS,
    EXTRA_CHECKS,
    demo_checks,
    reset_demo_db,
    seed_demo,
    when,
)
from obsilord.memory.recall import RecallTool
from obsilord.memory.store import SqliteStore

NOW = datetime(2026, 11, 18, 15, 30).timestamp()


async def _seeded(path: Path) -> SqliteStore:
    store = SqliteStore(path)
    await store.open()
    await seed_demo(store, NOW)
    return store


def test_dates_are_relative_to_now() -> None:
    today = next(c for c in DEMO_CONVERSATIONS if c.days_ago == 0)
    assert when(today, NOW) == NOW - 2 * 3600
    yesterday = next(c for c in DEMO_CONVERSATIONS if c.days_ago == 1)
    expected = (datetime(2026, 11, 18) - timedelta(days=1)).replace(
        hour=yesterday.hour, minute=yesterday.minute
    )
    assert when(yesterday, NOW) == expected.timestamp()


def test_seed_writes_every_conversation(tmp_path: Path) -> None:
    async def go() -> list:
        store = await _seeded(tmp_path / "demo.sqlite3")
        infos = await store.list_conversations(50)
        await store.close()
        return infos

    infos = run(go())
    assert len(infos) == len(DEMO_CONVERSATIONS) >= 7
    assert all(i.channel == "web" for i in infos)
    assert [i.turns for i in reversed(infos)] == [len(c.turns) for c in sorted(DEMO_CONVERSATIONS, key=lambda c: -c.days_ago)]


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_every_recall_check_finds_its_fact(tmp_path: Path) -> None:
    async def go() -> list[tuple[str, list]]:
        store = await _seeded(tmp_path / "demo.sqlite3")
        out = []
        for conv in DEMO_CONVERSATIONS:
            for check in conv.checks:
                if check.search is not None:
                    out.append((check.expect_in, await store.search(check.search, 5)))
        await store.close()
        return out

    results = run(go())
    assert results
    for expect_in, hits in results:
        assert hits, expect_in
        top = hits[0]
        text = " ".join([top.user, top.reply, *top.tools]).lower()
        assert expect_in.lower() in text


def test_negative_check_finds_nothing(tmp_path: Path) -> None:
    negative = [c for c in EXTRA_CHECKS if c.kind == "recall-none"]
    assert negative

    async def go() -> list:
        store = await _seeded(tmp_path / "demo.sqlite3")
        hits = [await store.search(c.search or "", 5) for c in negative]
        await store.close()
        return hits

    assert all(h == [] for h in run(go()))


def test_check_list_has_dates_labels_and_kinds() -> None:
    checks = demo_checks(NOW)
    kinds = {c["kind"] for c in checks}
    assert {"recall", "recall-date", "recall-none", "inline", "tool-note", "resume", "continue"} <= kinds
    by_date = next(c for c in checks if c["kind"] == "recall-date")
    assert by_date["question"] == "What did we talk about on the 5th of November?"  # 13 days before NOW
    assert by_date["date"] == "2026-11-05"
    first = checks[0]
    assert set(first) == {"date", "label", "question", "expected", "kind"}
    assert first["date"] == "2026-11-18" and first["label"] == "today"
    yesterday = [c for c in checks if c["label"] == "yesterday"]
    assert yesterday and yesterday[0]["date"] == "2026-11-17"
    assert all(c["date"] == "" for c in checks if c["kind"] in ("inline", "tool-note", "resume"))


def test_date_check_finds_its_day_through_recall(tmp_path: Path) -> None:
    # The question as a small model passes it: the date inside the query.
    question = next(c for c in demo_checks(NOW) if c["kind"] == "recall-date")["question"]

    async def go() -> dict:
        store = await _seeded(tmp_path / "demo.sqlite3")
        result = await RecallTool(store, clock=lambda: NOW).run({"query": question})
        await store.close()
        return result

    assert "Pita" in run(go())["output"]


def test_reset_removes_the_file_and_its_wal(tmp_path: Path) -> None:
    path = tmp_path / "demo.sqlite3"

    async def go() -> None:
        store = await _seeded(path)
        await store.close()

    run(go())
    (tmp_path / "demo.sqlite3-wal").write_bytes(b"x")
    reset_demo_db(path)
    assert not path.exists() and not (tmp_path / "demo.sqlite3-wal").exists()
    reset_demo_db(path)  # nothing there: no error


def test_reseed_does_not_double(tmp_path: Path) -> None:
    path = tmp_path / "demo.sqlite3"

    async def go() -> int:
        store = await _seeded(path)
        await store.close()
        reset_demo_db(path)
        store = await _seeded(path)
        n = len(await store.list_conversations(50))
        await store.close()
        return n

    assert run(go()) == len(DEMO_CONVERSATIONS)
