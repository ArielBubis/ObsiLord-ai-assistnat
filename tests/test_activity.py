"""The activity log (logs/activity.jsonl): time, tool, mode, outcome, never content."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from obsilord.activity import ActivityLog

START = datetime(2026, 11, 19, 12, 0, 0, tzinfo=timezone.utc)


class Clock:
    def __init__(self) -> None:
        self.now = START

    def __call__(self) -> datetime:
        return self.now


def _lines(path: Path) -> list[dict[str, object]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def test_add_writes_one_line_with_time_tool_mode_and_outcome_only(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "activity.jsonl"  # the folder is made on first write
    log = ActivityLog(path, retention_days=0, now=Clock())
    log.add("look", "ask", "ok")
    log.add("recall", "on", "declined")
    assert _lines(path) == [
        {"ts": "2026-11-19T12:00:00+00:00", "tool": "look", "mode": "ask", "outcome": "ok"},
        {"ts": "2026-11-19T12:00:00+00:00", "tool": "recall", "mode": "on", "outcome": "declined"},
    ]


def test_newest_first_with_a_limit(tmp_path: Path) -> None:
    clock = Clock()
    log = ActivityLog(tmp_path / "activity.jsonl", retention_days=0, now=clock)
    for n in range(5):
        clock.now = START + timedelta(minutes=n)
        log.add(f"tool{n}", "on", "ok")
    assert [e["tool"] for e in log.newest(3)] == ["tool4", "tool3", "tool2"]
    assert len(log.newest(50)) == 5


def test_newest_of_a_missing_file_is_empty(tmp_path: Path) -> None:
    assert ActivityLog(tmp_path / "activity.jsonl", retention_days=0).newest(50) == []


def test_newest_skips_lines_it_cannot_read(tmp_path: Path) -> None:
    path = tmp_path / "activity.jsonl"
    path.write_text('{"ts": "x", "tool": "look", "mode": "on", "outcome": "ok"}\nnot json\n[1]\n', encoding="utf-8")
    assert [e["tool"] for e in ActivityLog(path, retention_days=0).newest(50)] == ["look"]


def test_prune_drops_entries_older_than_retention_days(tmp_path: Path) -> None:
    path = tmp_path / "activity.jsonl"
    clock = Clock()
    log = ActivityLog(path, retention_days=7, now=clock)
    for days in (10, 8, 6, 1):
        clock.now = START - timedelta(days=days)
        log.add(f"d{days}", "on", "ok")
    clock.now = START
    assert log.prune() == 2
    assert [e["tool"] for e in _lines(path)] == ["d6", "d1"]


def test_prune_with_zero_keeps_everything(tmp_path: Path) -> None:
    path = tmp_path / "activity.jsonl"
    clock = Clock()
    log = ActivityLog(path, retention_days=0, now=clock)
    clock.now = START - timedelta(days=4000)
    log.add("old", "on", "ok")
    clock.now = START
    assert log.prune() == 0
    assert len(_lines(path)) == 1


def test_prune_of_a_missing_file_does_nothing(tmp_path: Path) -> None:
    assert ActivityLog(tmp_path / "activity.jsonl", retention_days=3).prune() == 0
    assert not (tmp_path / "activity.jsonl").exists()


@pytest.mark.parametrize("bad", ["sometimes", ""])
def test_a_bad_mode_is_refused(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ValueError):
        ActivityLog(tmp_path / "activity.jsonl", retention_days=0).add("look", bad, "ok")


def test_a_write_failure_is_logged_not_raised(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    folder = tmp_path / "taken"
    folder.write_text("a file where the folder should be", encoding="utf-8")
    ActivityLog(folder / "activity.jsonl", retention_days=0).add("look", "on", "ok")
    assert "activity log" in caplog.text
