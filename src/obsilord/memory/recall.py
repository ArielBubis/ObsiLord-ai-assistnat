"""recall(): the model searches earlier conversations in the local store.

Offered to the local model only (app.py), so old conversations never go to a
cloud provider through it. The result is plain text with dates, capped, and
starts with today's date so "yesterday" can be resolved.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable

from obsilord.llm.base import ToolNote, ToolSpec
from obsilord.memory.base import SearchSource
from obsilord.memory.store import fts_query
from obsilord.tools import Tool

log = logging.getLogger(__name__)

MAX_DAYS_BACK = 3650
TEXT_MAX_CHARS = 300
MSG_RECALL_FAILED = "The conversation store could not be searched."
CUT = "(more results cut)"

RECALL_TOOL = ToolSpec(
    name="recall",
    description=(
        "Search earlier conversations with the user, kept on this computer. Use it only "
        "when the user refers to something from an earlier conversation or asks what you "
        "talked about before (for example yesterday or last week). Do not use it for the "
        "recent messages of this conversation, which you already have. Returns the "
        "matching turns with their dates."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A few keywords to look for, for example 'dentist appointment'. "
                "Empty to list the most recent turns.",
            },
            "days_back": {
                "type": "integer",
                "description": "Only search the last N days (1 is today and yesterday). "
                "Leave out to search everything.",
            },
        },
        "required": ["query"],
    },
)


def notes_to_store(notes: tuple[ToolNote, ...]) -> tuple[ToolNote, ...]:
    """The notes worth keeping in the store: all but recall's own results,
    which hold older turns and would come back as nested copies in the next
    search. They stay in the inline history of the session."""
    return tuple(n for n in notes if n.name != RECALL_TOOL.name)


def _flat(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= TEXT_MAX_CHARS else text[: TEXT_MAX_CHARS - 3] + "..."


def _days_back(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return min(value, MAX_DAYS_BACK)


class RecallTool:
    def __init__(
        self,
        source: SearchSource,
        max_results: int = 5,
        max_chars: int = 1500,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._source = source
        self._max_results = max_results
        self._max_chars = max_chars
        self._clock = clock

    def tool(self) -> Tool:
        return Tool(RECALL_TOOL, self.run)

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query")
        query = query.strip() if isinstance(query, str) else ""
        days_back = _days_back(args.get("days_back"))
        now = self._clock()
        try:
            hits = await self._source.search(query, self._max_results, days_back=days_back, now=now)
        except Exception as exc:  # noqa: BLE001
            log.warning("recall: search failed: %s", type(exc).__name__)
            return {"error": MSG_RECALL_FAILED}
        log.info("recall: %d results", len(hits))
        today = datetime.fromtimestamp(now)
        head = f"Today is {today:%Y-%m-%d} ({today:%A})."
        if not hits:
            if fts_query(query) is not None:
                return {"output": f"{head} No earlier conversation matched '{_flat(query)}'."}
            span = f" in the last {days_back} days" if days_back else ""
            return {"output": f"{head} No earlier conversation{span}."}
        order = "best match first" if fts_query(query) is not None else "newest first"
        out = f"{head} Earlier conversations, {order}:"
        for hit in hits:
            when = datetime.fromtimestamp(hit.ts)
            lines = [f"{when:%Y-%m-%d %H:%M} user: {_flat(hit.user)}"]
            lines += [f"  tool {_flat(t)}" for t in hit.tools]
            lines.append(f"  assistant: {_flat(hit.reply)}")
            block = "\n" + "\n".join(lines)
            if len(out) + len(block) + len(CUT) + 1 > self._max_chars:
                out += "\n" + CUT
                break
            out += block
        return {"output": out[: self._max_chars]}
