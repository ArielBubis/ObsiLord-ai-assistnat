"""The conversation store: one local SQLite file (stdlib sqlite3, WAL) with an
FTS5 index over every message.

Text only: user text, replies, and tool results (vision and lookup text).
Never frames. Nothing here logs message text.

Schema changes go in MIGRATIONS (applied in order, PRAGMA user_version holds
the count). messages_fts is an external-content index kept in sync by
triggers, so it can be rebuilt with
INSERT INTO messages_fts(messages_fts) VALUES('rebuild').
All calls run in a worker thread, one at a time, on one connection.
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from obsilord.llm.base import Message, ToolNote
from obsilord.memory.base import ConversationInfo, Hit, TurnRecord

T = TypeVar("T")

DAY_S = 86400.0
TITLE_MAX_CHARS = 80
QUERY_MAX_TOKENS = 12

MIGRATIONS: list[str] = [
    """
    CREATE TABLE conversations (
        id TEXT PRIMARY KEY,
        channel TEXT NOT NULL,
        started_at REAL NOT NULL,
        last_at REAL NOT NULL,
        title TEXT NOT NULL DEFAULT '',
        turns INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX conversations_last_at ON conversations(channel, last_at);
    CREATE TABLE messages (
        id INTEGER PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id),
        turn INTEGER NOT NULL,
        ts REAL NOT NULL,
        role TEXT NOT NULL,
        text TEXT NOT NULL,
        provider TEXT,
        model TEXT,
        tool_name TEXT,
        tool_arg TEXT,
        meta TEXT
    );
    CREATE INDEX messages_turn ON messages(conversation_id, turn);
    CREATE INDEX messages_ts ON messages(ts);
    CREATE VIRTUAL TABLE messages_fts USING fts5(
        text, content='messages', content_rowid='id',
        tokenize='unicode61 remove_diacritics 2'
    );
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
    END;
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
    END;
    CREATE TRIGGER messages_au AFTER UPDATE ON messages BEGIN
        INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
        INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
    END;
    """,
]
SCHEMA_VERSION = len(MIGRATIONS)

_TOKEN = re.compile(r"\w+")


def fts_query(text: str) -> str | None:
    """Free text as a safe FTS5 query: word tokens (2 or more characters, at
    most QUERY_MAX_TOKENS, no repeats), each quoted as a prefix term, ORed.
    Quoting makes operators in the text (AND, NEAR, *, quotes) plain words.
    None when nothing is left."""
    if not isinstance(text, str):
        return None
    tokens: list[str] = []
    for token in _TOKEN.findall(text.lower()):
        if len(token) >= 2 and token not in tokens:
            tokens.append(token)
        if len(tokens) == QUERY_MAX_TOKENS:
            break
    if not tokens:
        return None
    return " OR ".join(f'"{t}"*' for t in tokens)


def _title(text: str) -> str:
    return " ".join(text.split())[:TITLE_MAX_CHARS]


def _now(now: float | None) -> float:
    return time.time() if now is None else now


class SqliteStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._con: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        # Held by the worker thread itself: a call cancelled while its thread
        # runs releases the asyncio lock at once, but the thread keeps the
        # connection until it is done.
        self._thread_lock = threading.Lock()

    async def _call(self, fn: Callable[..., T], *args: Any) -> T:
        def locked() -> T:
            with self._thread_lock:
                return fn(*args)

        async with self._lock:
            return await asyncio.to_thread(locked)

    def _db(self) -> sqlite3.Connection:
        if self._con is None:
            raise RuntimeError("store is not open")
        return self._con

    # Lifecycle

    async def open(self) -> None:
        await self._call(self._open)

    def _open(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path, check_same_thread=False)
        try:
            con.execute("PRAGMA journal_mode=WAL")
            con.execute("PRAGMA foreign_keys=ON")
            version = con.execute("PRAGMA user_version").fetchone()[0]
            for i in range(version, SCHEMA_VERSION):
                with con:
                    con.executescript(f"BEGIN; {MIGRATIONS[i]} PRAGMA user_version = {i + 1}; COMMIT;")
        except Exception:
            con.close()
            raise
        self._con = con

    async def close(self) -> None:
        if self._con is not None:
            await self._call(self._close)

    def _close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None

    # Conversations

    async def new_conversation(self, channel: str, now: float | None = None) -> str:
        return await self._call(self._new_conversation, channel, _now(now))

    def _new_conversation(self, channel: str, now: float) -> str:
        cid = uuid.uuid4().hex
        with self._db() as con:
            con.execute(
                "INSERT INTO conversations (id, channel, started_at, last_at) VALUES (?, ?, ?, ?)",
                (cid, channel, now, now),
            )
        return cid

    async def exists(self, conversation_id: str) -> bool:
        return await self._call(self._exists, conversation_id)

    def _exists(self, cid: str) -> bool:
        row = self._db().execute("SELECT 1 FROM conversations WHERE id = ?", (cid,)).fetchone()
        return row is not None

    async def info(self, conversation_id: str) -> ConversationInfo | None:
        return await self._call(self._info, conversation_id)

    def _info(self, cid: str) -> ConversationInfo | None:
        row = self._db().execute(
            "SELECT id, channel, title, started_at, last_at, turns FROM conversations WHERE id = ?",
            (cid,),
        ).fetchone()
        return ConversationInfo(*row) if row else None

    async def latest(self, channel: str, within_s: float, now: float | None = None) -> str | None:
        """The conversation on channel with a turn in the last within_s seconds."""
        return await self._call(self._latest, channel, _now(now) - within_s)

    def _latest(self, channel: str, since: float) -> str | None:
        row = self._db().execute(
            "SELECT id FROM conversations WHERE channel = ? AND turns > 0 AND last_at >= ? "
            "ORDER BY last_at DESC LIMIT 1",
            (channel, since),
        ).fetchone()
        return row[0] if row else None

    async def list_conversations(self, limit: int) -> list[ConversationInfo]:
        """Conversations with at least one turn, the most recent first."""
        return await self._call(self._list, limit)

    def _list(self, limit: int) -> list[ConversationInfo]:
        rows = self._db().execute(
            "SELECT id, channel, title, started_at, last_at, turns FROM conversations "
            "WHERE turns > 0 ORDER BY last_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [ConversationInfo(*row) for row in rows]

    # Turns

    async def add_turn(self, conversation_id: str, record: TurnRecord) -> None:
        await self._call(self._add_turn, conversation_id, record)

    def _add_turn(self, cid: str, r: TurnRecord) -> None:
        with self._db() as con:
            row = con.execute("SELECT turns, title FROM conversations WHERE id = ?", (cid,)).fetchone()
            if row is None:
                raise KeyError(cid)
            n = row[0] + 1
            title = row[1] or _title(r.user_text)
            con.execute(
                "INSERT INTO messages (conversation_id, turn, ts, role, text) VALUES (?, ?, ?, 'user', ?)",
                (cid, n, r.ts, r.user_text),
            )
            for note in r.notes:
                con.execute(
                    "INSERT INTO messages (conversation_id, turn, ts, role, text, tool_name, tool_arg, meta) "
                    "VALUES (?, ?, ?, 'tool', ?, ?, ?, ?)",
                    (cid, n, r.ts, note.result, note.name, note.arg, None if note.ok else json.dumps({"ok": False})),
                )
            con.execute(
                "INSERT INTO messages (conversation_id, turn, ts, role, text, provider, model) "
                "VALUES (?, ?, ?, 'assistant', ?, ?, ?)",
                (cid, n, r.ts, r.reply, r.provider, r.model),
            )
            con.execute(
                "UPDATE conversations SET turns = ?, title = ?, last_at = MAX(last_at, ?) WHERE id = ?",
                (n, title, r.ts, cid),
            )

    async def last_turns(self, conversation_id: str, n: int) -> list[Message]:
        """The last n turns as history: user and assistant messages, the
        assistant ones with their tool notes."""
        return await self._call(self._last_turns, conversation_id, n)

    def _last_turns(self, cid: str, n: int) -> list[Message]:
        if n <= 0:
            return []
        rows = self._db().execute(
            "SELECT turn, role, text, tool_name, tool_arg, meta FROM messages "
            "WHERE conversation_id = ? AND turn > "
            "(SELECT MAX(turns) - ? FROM conversations WHERE id = ?) ORDER BY turn, id",
            (cid, n, cid),
        ).fetchall()
        out: list[Message] = []
        notes: list[ToolNote] = []
        for _, role, text, tool_name, tool_arg, meta in rows:
            if role == "user":
                out.append(Message("user", text))
                notes = []
            elif role == "tool":
                ok = not (meta and json.loads(meta).get("ok") is False)
                notes.append(ToolNote(tool_name or "", tool_arg or "", text, ok=ok))
            else:
                out.append(Message("assistant", text, tools=tuple(notes)))
        return out

    # Search

    async def search(
        self,
        query: str,
        limit: int,
        since: float | None = None,
        until: float | None = None,
        skip: tuple[str, int] | None = None,
    ) -> list[Hit]:
        """Matching turns, best match first (newer first on ties). With no
        usable words in query: the newest turns. since and until (unix
        seconds, since inclusive, until exclusive) bound both. skip
        (conversation id, n) leaves out that conversation's last n turns."""
        return await self._call(self._search, fts_query(query), limit, since, until, skip)

    def _search(
        self,
        match: str | None,
        limit: int,
        since: float | None,
        until: float | None,
        skip: tuple[str, int] | None,
    ) -> list[Hit]:
        con = self._db()
        floor = since if since is not None else float("-inf")
        ceiling = until if until is not None else float("inf")
        skip_id, skip_n = skip if skip is not None else ("", 0)
        # The skipped turns: those after the conversation's turns minus n.
        skipped = (
            "NOT (m.conversation_id = ? AND m.turn > "
            "COALESCE((SELECT turns - ? FROM conversations WHERE id = ?), 0))"
        )
        skip_args = (skip_id, max(skip_n, 0), skip_id)
        if match is None:
            keys = con.execute(
                "SELECT m.conversation_id, m.turn FROM messages m "
                f"WHERE m.role = 'user' AND m.ts >= ? AND m.ts < ? AND {skipped} "
                "ORDER BY m.ts DESC, m.id DESC LIMIT ?",
                (floor, ceiling, *skip_args, limit),
            ).fetchall()
        else:
            keys = con.execute(
                "SELECT m.conversation_id, m.turn FROM messages_fts f "
                "JOIN messages m ON m.id = f.rowid "
                f"WHERE messages_fts MATCH ? AND m.ts >= ? AND m.ts < ? AND {skipped} "
                "GROUP BY m.conversation_id, m.turn "
                "ORDER BY MIN(f.rank), MAX(m.ts) DESC LIMIT ?",
                (match, floor, ceiling, *skip_args, limit),
            ).fetchall()
        return [self._hit(cid, turn) for cid, turn in keys]

    def _hit(self, cid: str, turn: int) -> Hit:
        rows = self._db().execute(
            "SELECT ts, role, text, tool_name FROM messages WHERE conversation_id = ? AND turn = ? "
            "ORDER BY id",
            (cid, turn),
        ).fetchall()
        user = next((r[2] for r in rows if r[1] == "user"), "")
        reply = next((r[2] for r in rows if r[1] == "assistant"), "")
        tools = tuple(f"{r[3]}: {r[2]}" for r in rows if r[1] == "tool")
        return Hit(rows[0][0], cid, user, reply, tools)

    # Retention

    async def prune(self, retention_days: int, now: float | None = None) -> int:
        """Delete conversations whose last turn is older than retention_days
        (0 keeps everything). Returns how many were deleted."""
        if retention_days <= 0:
            return 0
        return await self._call(self._prune, _now(now) - retention_days * DAY_S)

    def _prune(self, cutoff: float) -> int:
        with self._db() as con:
            ids = [r[0] for r in con.execute("SELECT id FROM conversations WHERE last_at < ?", (cutoff,))]
            for cid in ids:
                con.execute("DELETE FROM messages WHERE conversation_id = ?", (cid,))
                con.execute("DELETE FROM conversations WHERE id = ?", (cid,))
        return len(ids)
