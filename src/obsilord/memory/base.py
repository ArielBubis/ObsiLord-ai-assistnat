"""Memory records and the interfaces the app and the recall tool use.

The SQLite store implements both protocols. Later sources (an Obsidian vault,
embedding search) implement SearchSource and plug into recall unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from obsilord.llm.base import Message, ToolNote


@dataclass(frozen=True)
class TurnRecord:
    """One finished turn as the pipeline hands it over. ts: unix seconds."""

    user_text: str
    reply: str
    notes: tuple[ToolNote, ...]
    provider: str
    model: str
    ts: float


@dataclass(frozen=True)
class ConversationInfo:
    id: str
    channel: str
    title: str
    started_at: float
    last_at: float
    turns: int


@dataclass(frozen=True)
class Hit:
    """One matching turn: the user text, the reply and its tool results
    (each "name: result")."""

    ts: float
    conversation_id: str
    user: str
    reply: str
    tools: tuple[str, ...] = ()


class SearchSource(Protocol):
    # since/until: unix seconds, since inclusive, until exclusive.
    async def search(
        self,
        query: str,
        limit: int,
        since: float | None = None,
        until: float | None = None,
        skip: tuple[str, int] | None = None,  # (conversation id, its last n turns)
    ) -> list[Hit]: ...


class MemoryStore(SearchSource, Protocol):
    async def open(self) -> None: ...

    async def close(self) -> None: ...

    async def prune(self, retention_days: int, now: float | None = None) -> int: ...

    async def new_conversation(self, channel: str, now: float | None = None) -> str: ...

    async def exists(self, conversation_id: str) -> bool: ...

    async def add_turn(self, conversation_id: str, record: TurnRecord) -> None: ...

    async def last_turns(self, conversation_id: str, n: int) -> list[Message]: ...

    async def latest(self, channel: str, within_s: float, now: float | None = None) -> str | None: ...

    async def info(self, conversation_id: str) -> ConversationInfo | None: ...

    async def list_conversations(self, limit: int) -> list[ConversationInfo]: ...
