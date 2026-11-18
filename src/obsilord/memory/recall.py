"""recall(): the model searches earlier conversations in the local store.

Offered to the local model only (app.py), so old conversations never go to a
cloud provider through it. The spec is rebuilt every turn with today's date in
its description (recall_spec), so the model can turn "on the 5th of November"
or "yesterday" into a date before it calls. date (YYYY-MM-DD) limits the
search to that day, days_back to the last N days; without either, a day
named in query ("5th of November", "yesterday": the router passes one
argument only) is used as the date. Words that match nothing
inside such a window fall back to listing the window, since a small model
often puts the date itself in query. The result is plain text with dates,
capped, and starts with today's date too.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date as Date
from datetime import datetime, timedelta
from typing import Any, Callable

from obsilord.llm.base import ToolNote, ToolSpec
from obsilord.memory.base import Hit, SearchSource
from obsilord.memory.store import fts_query
from obsilord.tools import Tool

log = logging.getLogger(__name__)

DAY_S = 86400.0
MAX_DAYS_BACK = 3650
TEXT_MAX_CHARS = 300
MSG_RECALL_FAILED = "The conversation store could not be searched."
CUT = "(more results cut)"

_DESCRIPTION = (
    "Search earlier conversations with the user, kept on this computer. Use it only "
    "when the user refers to something from an earlier conversation or asks what you "
    "talked about before (for example yesterday, last week, or on a date). Do not use it "
    "for the recent messages of this conversation, which you already have. For a "
    "question about a day, pass date and leave query empty. Returns the matching turns "
    "with their dates."
)
_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A few keywords from what was said, for example 'dentist "
            "appointment'. Never a date. Empty to list the turns of the date or the days.",
        },
        "date": {
            "type": "string",
            "description": "Only that day, as YYYY-MM-DD (work it out from today's date "
            "above, for example 'the 5th' is this month unless that is still ahead).",
        },
        "days_back": {
            "type": "integer",
            "description": "Only the last N days (1 is today and yesterday). Leave out "
            "date and days_back to search everything.",
        },
    },
    "required": ["query"],
}

# Without a date: the declaration for routecheck and tests. Turns use recall_spec.
RECALL_TOOL = ToolSpec(name="recall", description=_DESCRIPTION, parameters=_PARAMETERS)


def _today(now: float) -> str:
    day = datetime.fromtimestamp(now)
    return f"Today is {day:%Y-%m-%d} ({day:%A})."


def recall_spec(now: float) -> ToolSpec:
    """The declaration for one turn, with today's date in the description."""
    return ToolSpec(name=RECALL_TOOL.name, description=f"{_today(now)} {_DESCRIPTION}", parameters=_PARAMETERS)


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


def _day(value: Any) -> Date | None:
    if not isinstance(value, str):
        return None
    try:
        return Date.fromisoformat(value.strip())
    except ValueError:
        return None


_MONTHS = {
    name: i
    for i, names in enumerate(
        (
            ("january", "jan"), ("february", "feb"), ("march", "mar"), ("april", "apr"),
            ("may",), ("june", "jun"), ("july", "jul"), ("august", "aug"),
            ("september", "sep", "sept"), ("october", "oct"), ("november", "nov"),
            ("december", "dec"),
        ),
        start=1,
    )
    for name in names
}
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_DAY_MONTH = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?({_MONTH})\b\.?(?:,?\s+(\d{{4}})\b)?", re.I)
_MONTH_DAY = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\b(?:,?\s+(\d{{4}})\b)?", re.I)


def _dated(year: int | None, month: int, day: int, today: Date) -> Date | None:
    """The date; without a year the latest one that is not after today."""
    try:
        if year is not None:
            return Date(year, month, day)
        found = Date(today.year, month, day)
        return found if found <= today else Date(today.year - 1, month, day)
    except ValueError:
        return None


def date_in_text(text: str, today: Date) -> Date | None:
    """A day named in text: 2026-11-05, 5th of November, Nov 5, 2025,
    yesterday, today. None when there is none (a month alone is no day)."""
    if not isinstance(text, str):
        return None
    m = _ISO.search(text)
    if m:
        return _dated(int(m[1]), int(m[2]), int(m[3]), today)
    m = _DAY_MONTH.search(text)
    if m:
        return _dated(int(m[3]) if m[3] else None, _MONTHS[m[2].lower()], int(m[1]), today)
    m = _MONTH_DAY.search(text)
    if m:
        return _dated(int(m[3]) if m[3] else None, _MONTHS[m[1].lower()], int(m[2]), today)
    lowered = text.lower()
    if re.search(r"\byesterday\b", lowered):
        return today - timedelta(days=1)
    if re.search(r"\btoday\b", lowered):
        return today
    return None


def _day_window(day: Date) -> tuple[float, float] | None:
    """Local midnight to the next one, in unix seconds. None when the
    platform cannot convert the day (before 1970 on Windows, year 9999)."""
    try:
        end = day + timedelta(days=1)
        return (
            datetime(day.year, day.month, day.day).timestamp(),
            datetime(end.year, end.month, end.day).timestamp(),
        )
    except (OverflowError, OSError, ValueError):
        return None


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
        """The tool for one turn (today's date in its description)."""
        return Tool(recall_spec(self._clock()), self.run)

    async def run(self, args: dict[str, Any]) -> dict[str, Any]:
        query = args.get("query")
        query = query.strip() if isinstance(query, str) else ""
        now = self._clock()
        day = _day(args.get("date"))
        if day is None:
            # The router passes one arg, so a date may sit in the query; a
            # day named there beats a guessed days_back.
            day = date_in_text(query, datetime.fromtimestamp(now).date())
        days_back = None if day is not None else _days_back(args.get("days_back"))
        since: float | None = None
        until: float | None = None
        if day is not None:
            window = f" on {day:%Y-%m-%d}"
            bounds = _day_window(day)
            if bounds is None:
                return {"output": f"{_today(now)} No earlier conversation{window}."}
            since, until = bounds
        elif days_back is not None:
            since = now - days_back * DAY_S
            window = f" in the last {days_back} days"
        else:
            window = ""
        words = fts_query(query) is not None
        missed = ""
        try:
            hits = await self._source.search(query, self._max_results, since=since, until=until)
            if not hits and words and window:
                # The words found nothing in the window: list the window.
                missed = f"Nothing matched '{_flat(query)}'. "
                words = False
                hits = await self._source.search("", self._max_results, since=since, until=until)
        except Exception as exc:  # noqa: BLE001
            log.warning("recall: search failed: %s", type(exc).__name__)
            return {"error": MSG_RECALL_FAILED}
        log.info("recall: %d results", len(hits))
        head = _today(now)
        if not hits:
            if words:
                return {"output": f"{head} No earlier conversation matched '{_flat(query)}'."}
            return {"output": f"{head} {missed}No earlier conversation{window}."}
        order = "best match first" if words else "newest first"
        span = window if day is not None else ""
        return {"output": self._format(f"{head} {missed}Earlier conversations{span}, {order}:", hits)}

    def _format(self, out: str, hits: list[Hit]) -> str:
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
        return out[: self._max_chars]
