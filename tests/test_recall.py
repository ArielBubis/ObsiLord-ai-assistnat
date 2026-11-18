"""RecallTool: arguments, output format and caps, against a fake source."""

import asyncio
from datetime import datetime
from typing import Any

from obsilord.llm.base import ToolNote
from obsilord.memory.base import Hit
from obsilord.memory.recall import (
    MAX_DAYS_BACK,
    MSG_RECALL_FAILED,
    RECALL_TOOL,
    RecallTool,
    notes_to_store,
)

# Local time, so the dates in the output do not depend on the time zone.
NOW = datetime(2026, 11, 18, 15, 30).timestamp()
YESTERDAY = datetime(2026, 11, 17, 19, 5).timestamp()


class FakeSource:
    def __init__(self, hits: list[Hit] | None = None, fail: bool = False) -> None:
        self.hits = hits or []
        self.fail = fail
        self.calls: list[tuple[str, int, Any]] = []

    async def search(
        self, query: str, limit: int, days_back: int | None = None, now: float | None = None
    ) -> list[Hit]:
        self.calls.append((query, limit, days_back))
        if self.fail:
            raise OSError("disk")
        return self.hits[:limit]


def _tool(source: FakeSource, max_chars: int = 1500) -> RecallTool:
    return RecallTool(source, max_results=5, max_chars=max_chars, clock=lambda: NOW)


def test_spec_declares_query_and_days_back() -> None:
    assert RECALL_TOOL.name == "recall"
    props = RECALL_TOOL.parameters["properties"]
    assert set(props) == {"query", "days_back"}
    assert RECALL_TOOL.parameters["required"] == ["query"]
    assert "earlier conversation" in RECALL_TOOL.description


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


def test_days_back_is_passed_and_clamped() -> None:
    source = FakeSource()
    tool = _tool(source)
    asyncio.run(tool.run({"query": "", "days_back": 2}))
    asyncio.run(tool.run({"query": "x", "days_back": 10**9}))
    asyncio.run(tool.run({"query": "x", "days_back": "three"}))
    asyncio.run(tool.run({"query": "x", "days_back": 0}))
    assert [c[2] for c in source.calls] == [2, MAX_DAYS_BACK, None, None]
    assert source.calls[0][1] == 5


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
