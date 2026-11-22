"""RecallTool: arguments, output format and caps, against a fake source."""

import asyncio
from datetime import date as Date
from datetime import datetime
from typing import Any

from obsilord.llm.base import ToolNote
from obsilord.memory.base import Hit
from obsilord.tools.registry import TurnContext
from obsilord.memory.recall import (
    MAX_DAYS_BACK,
    MSG_RECALL_FAILED,
    RECALL_TOOL,
    RecallTool,
    date_in_text,
    notes_to_store,
)

# Local time, so the dates in the output do not depend on the time zone.
NOW = datetime(2026, 11, 18, 15, 30).timestamp()
YESTERDAY = datetime(2026, 11, 17, 19, 5).timestamp()
NOV5 = datetime(2026, 11, 5, 22, 10).timestamp()
DAY = 86400.0


class FakeSource:
    """hits: returned for any query. window_hits: returned only for an empty
    query (the listing of a time window)."""

    def __init__(
        self, hits: list[Hit] | None = None, fail: bool = False, window_hits: list[Hit] | None = None
    ) -> None:
        self.hits = hits or []
        self.window_hits = window_hits
        self.fail = fail
        self.calls: list[tuple[str, int, Any, Any]] = []
        self.skips: list[Any] = []

    async def search(
        self,
        query: str,
        limit: int,
        since: float | None = None,
        until: float | None = None,
        skip: tuple[str, int] | None = None,
    ) -> list[Hit]:
        self.calls.append((query, limit, since, until))
        self.skips.append(skip)
        if self.fail:
            raise OSError("disk")
        if self.window_hits is not None and not query:
            return self.window_hits[:limit]
        return self.hits[:limit]


def _tool(source: FakeSource, max_chars: int = 1500) -> RecallTool:
    return RecallTool(source, max_results=5, max_chars=max_chars, clock=lambda: NOW)


def test_spec_declares_query_date_and_days_back() -> None:
    spec = RECALL_TOOL.spec
    assert spec.name == "recall"
    props = spec.parameters["properties"]
    assert set(props) == {"query", "date", "days_back"}
    assert (props["date"]["type"], props["days_back"]["type"]) == ("string", "integer")
    assert spec.parameters["required"] == ["query"]
    assert "earlier conversation" in spec.description
    # The date comes from the system prompt now, not the description.
    assert "Today is" not in spec.description
    assert RECALL_TOOL.policy.private_data and RECALL_TOOL.policy.untrusted
    assert not RECALL_TOOL.policy.cloud_default


def test_tool_of_a_turn_skips_the_inline_turns() -> None:
    source = FakeSource(hits=[], window_hits=[])
    bound = RECALL_TOOL.bind(recall=_tool(source), skip=("c" * 32, 10))
    ctx = TurnContext()
    asyncio.run(bound.call(bound.arguments({"query": "cat", "date": "2026-11-05"}), ctx))  # words, then the window
    plain = RECALL_TOOL.bind(recall=_tool(source))
    asyncio.run(plain.call(plain.arguments({"query": "cat"}), ctx))
    assert source.skips == [("c" * 32, 10), ("c" * 32, 10), None]


def test_output_lists_hits_with_dates() -> None:
    hit = Hit(YESTERDAY, "c1", "I want shakshuka", "You are out of feta.", ("look: a pan",))
    result = asyncio.run(_tool(FakeSource([hit])).run({"query": "shakshuka"}))
    assert result == {
        "output": "Today is 2026-11-18 (Wednesday). Earlier conversations, best match first:\n"
        "2026-11-17 19:05 user: I want shakshuka\n"
        "  tool look: a pan\n"
        "  assistant: You are out of feta."
    }


def test_no_match() -> None:
    result = asyncio.run(_tool(FakeSource()).run({"query": "skiing"}))
    assert result == {"output": "Today is 2026-11-18 (Wednesday). No earlier conversation matched 'skiing'."}


def test_days_back_is_a_window_and_clamped() -> None:
    source = FakeSource()
    tool = _tool(source)
    asyncio.run(tool.run({"query": "", "days_back": 2}))
    asyncio.run(tool.run({"query": "x", "days_back": 10**9}))
    asyncio.run(tool.run({"query": "x", "days_back": "three"}))
    asyncio.run(tool.run({"query": "x", "days_back": 0}))
    assert source.calls[0] == ("", 5, NOW - 2 * DAY, None)
    assert source.calls[1][2] == NOW - MAX_DAYS_BACK * DAY
    assert [c[2] for c in source.calls[2:]] == [None, None]


def test_date_limits_the_search_to_that_day() -> None:
    source = FakeSource([Hit(NOV5, "c", "My cat Pita is hiding.", "Poor Pita.")])
    result = asyncio.run(_tool(source).run({"query": "", "date": "2026-11-05"}))
    query, _, since, until = source.calls[0]
    assert query == ""
    assert since == datetime(2026, 11, 5).timestamp()
    assert until == datetime(2026, 11, 6).timestamp()
    assert "Earlier conversations on 2026-11-05, newest first:" in result["output"]
    assert "2026-11-05 22:10 user: My cat Pita is hiding." in result["output"]


def test_date_in_text() -> None:
    today = Date(2026, 11, 18)
    assert date_in_text("what did we talk about in 5th of november?", today) == Date(2026, 11, 5)
    assert date_in_text("November 5", today) == Date(2026, 11, 5)
    assert date_in_text("what did we talke about on november the 5th?", today) == Date(2026, 11, 5)
    assert date_in_text("the 5th of november", today) == Date(2026, 11, 5)
    assert date_in_text("on nov 5th, 2025", today) == Date(2025, 11, 5)
    assert date_in_text("5 Dec", today) == Date(2025, 12, 5)  # not yet this year
    assert date_in_text("2026-10-01 dentist", today) == Date(2026, 10, 1)
    assert date_in_text("yesterday", today) == Date(2026, 11, 17)
    assert date_in_text("what did we say today", today) == Date(2026, 11, 18)
    for none in ("dentist appointment", "may I ask", "31 february", "november 2025", ""):
        assert date_in_text(none, today) is None, none


def test_a_date_in_the_query_becomes_the_window() -> None:
    # A small model often leaves date empty and puts the day in query.
    day_hit = Hit(NOV5, "c", "My cat Pita is hiding.", "Poor Pita.")
    source = FakeSource(hits=[], window_hits=[day_hit])
    result = asyncio.run(_tool(source).run({"query": "5th of November"}))
    assert source.calls[0][2:] == (datetime(2026, 11, 5).timestamp(), datetime(2026, 11, 6).timestamp())
    assert "My cat Pita is hiding." in result["output"]


def test_a_date_in_the_query_wins_over_days_back() -> None:
    # The shape of the original failure: a day in the query plus a guessed days_back.
    source = FakeSource()
    asyncio.run(_tool(source).run({"query": "5th of November", "days_back": 3}))
    assert source.calls[0][2:] == (datetime(2026, 11, 5).timestamp(), datetime(2026, 11, 6).timestamp())


def test_a_date_out_of_range_finds_nothing() -> None:
    # datetime.timestamp() raises OSError on Windows before 1970.
    for args in (
        {"query": "x", "date": "1960-01-01"},
        {"query": "what did we say on 1 jan 1900"},
        {"query": "", "date": "9999-12-31"},
        {"query": "", "date": "0001-01-01"},
    ):
        source = FakeSource()
        result = asyncio.run(_tool(source).run(args))
        assert source.calls == [], args
        assert "No earlier conversation on " in result["output"], args


def test_bad_dates_are_ignored() -> None:
    source = FakeSource()
    tool = _tool(source)
    for bad in ("November 5", "2026-13-01", 20261105, ""):
        asyncio.run(tool.run({"query": "cat", "date": bad}))
    assert all(c[2] is None and c[3] is None for c in source.calls)


def test_words_that_miss_fall_back_to_the_window() -> None:
    # A small model asks for "November" on that date: no message holds the
    # word, so the turns of the day are listed instead.
    day_hit = Hit(NOV5, "c", "My cat Pita is hiding.", "Poor Pita.")
    source = FakeSource(hits=[], window_hits=[day_hit])
    result = asyncio.run(_tool(source).run({"query": "November 5th", "date": "2026-11-05"}))
    assert [c[0] for c in source.calls] == ["November 5th", ""]
    out = result["output"]
    assert "Nothing matched 'November 5th'. Earlier conversations on 2026-11-05, newest first:" in out
    assert "My cat Pita is hiding." in out


def test_no_fallback_without_a_window() -> None:
    source = FakeSource(hits=[], window_hits=[Hit(NOV5, "c", "u", "r")])
    result = asyncio.run(_tool(source).run({"query": "skiing"}))
    assert len(source.calls) == 1
    assert "No earlier conversation matched 'skiing'." in result["output"]


def test_empty_window() -> None:
    result = asyncio.run(_tool(FakeSource()).run({"query": "", "date": "2026-11-04"}))
    assert result["output"].endswith("No earlier conversation on 2026-11-04.")
    result = asyncio.run(_tool(FakeSource()).run({"query": "", "days_back": 3}))
    assert result["output"].endswith("No earlier conversation in the last 3 days.")


def test_empty_query_with_days_says_recent() -> None:
    hit = Hit(YESTERDAY, "c1", "hi", "hello")
    result = asyncio.run(_tool(FakeSource([hit])).run({"days_back": 1}))
    assert "Earlier conversations, newest first:" in result["output"]


def test_non_string_query_is_empty() -> None:
    source = FakeSource()
    asyncio.run(_tool(source).run({"query": 42}))
    assert source.calls[0][0] == ""


def test_output_is_capped_on_whole_lines() -> None:
    hits = [Hit(YESTERDAY - i, "c", "u" * 100, "r" * 100) for i in range(5)]
    out = asyncio.run(_tool(FakeSource(hits), max_chars=400).run({"query": "u"}))["output"]
    assert len(out) <= 400
    assert out.endswith("(more results cut)")


def test_long_texts_are_shortened_and_flattened() -> None:
    hit = Hit(YESTERDAY, "c", "line one\nline two " + "x" * 500, "ok")
    out = asyncio.run(_tool(FakeSource([hit])).run({"query": "x"}))["output"]
    assert "line one line two" in out
    assert "x" * 301 not in out


def test_recall_notes_are_not_stored() -> None:
    # A stored recall result would hold older turns and be found by the next
    # search as a nested copy.
    look = ToolNote("look", "q", "A blue owl.")
    recall = ToolNote("recall", "owl", "Today is ... user: owl")
    assert notes_to_store((look, recall)) == (look,)
    assert notes_to_store(()) == ()


def test_source_error_is_an_error_result() -> None:
    result = asyncio.run(_tool(FakeSource(fail=True)).run({"query": "x"}))
    assert result == {"error": MSG_RECALL_FAILED}
