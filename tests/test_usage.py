import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from obsilord.usage import CapExceeded, UsageTracker

TZ = timezone(timedelta(hours=3))


class FakeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _tracker(tmp_path: Path, cap: int, clock: FakeClock) -> UsageTracker:
    return UsageTracker(tmp_path / "logs" / "usage.jsonl", cap, clock=clock)


def test_under_cap_allows(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 12, 0, tzinfo=TZ))
    t = _tracker(tmp_path, 100, clock)
    t.check()
    t.record("m", 10, 20, 0, 50)
    t.check()
    assert t.today_total == 30


def test_at_cap_blocks(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 12, 0, tzinfo=TZ))
    t = _tracker(tmp_path, 100, clock)
    t.record("m", 60, 40, 0, 50)
    with pytest.raises(CapExceeded):
        t.check()


def test_over_cap_blocks(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 12, 0, tzinfo=TZ))
    t = _tracker(tmp_path, 100, clock)
    t.record("m", 90, 50, 0, 50)
    with pytest.raises(CapExceeded):
        t.check()


def test_zero_cap_blocks_everything(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 12, 0, tzinfo=TZ))
    with pytest.raises(CapExceeded):
        _tracker(tmp_path, 0, clock).check()


def test_thinking_tokens_count(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 12, 0, tzinfo=TZ))
    t = _tracker(tmp_path, 100, clock)
    t.record("m", 10, 10, 80, 50)
    assert t.today_total == 100
    with pytest.raises(CapExceeded):
        t.check()


def test_rollover_at_local_midnight(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 23, 59, 59, tzinfo=TZ))
    t = _tracker(tmp_path, 100, clock)
    t.record("m", 100, 0, 0, 50)
    with pytest.raises(CapExceeded):
        t.check()
    clock.now = datetime(2026, 10, 7, 0, 0, 0, tzinfo=TZ)
    t.check()
    assert t.today_total == 0
    t.record("m", 5, 5, 0, 50)
    assert t.today_total == 10


def test_rebuild_from_existing_log(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 9, 0, tzinfo=TZ))
    t1 = _tracker(tmp_path, 1000, clock)
    t1.record("m", 100, 50, 10, 50)
    t1.record("m", 20, 20, 0, 50)
    clock.now = datetime(2026, 10, 6, 18, 0, tzinfo=TZ)
    t2 = _tracker(tmp_path, 1000, clock)
    assert t2.today_total == 200
    clock.now = datetime(2026, 10, 7, 9, 0, tzinfo=TZ)
    t3 = _tracker(tmp_path, 1000, clock)
    assert t3.today_total == 0


def test_rebuild_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "usage.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"date": "2026-10-06", "total": 7}\n'
        "not json\n"
        '{"date": "2026-10-06"}\n'
        "[1, 2]\n"
        "\n"
        '{"date": "2026-10-05", "total": 1000}\n',
        encoding="utf-8",
    )
    clock = FakeClock(datetime(2026, 10, 6, 9, 0, tzinfo=TZ))
    assert UsageTracker(path, 100, clock=clock).today_total == 7


def test_log_has_counts_only(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 12, 0, tzinfo=TZ))
    t = _tracker(tmp_path, 100, clock)
    t.record("gemini-x", 1, 2, 3, 456)
    lines = (tmp_path / "logs" / "usage.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert set(entry) == {
        "ts", "date", "model", "tokens_in", "tokens_out", "tokens_thinking", "total", "latency_ms",
        "partial", "estimated",
    }
    assert entry["partial"] is False and entry["estimated"] is False
    assert entry["total"] == 6
    assert entry["date"] == "2026-10-06"
    assert entry["model"] == "gemini-x"
    assert entry["latency_ms"] == 456


def test_no_log_dir_created_until_record(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 12, 0, tzinfo=TZ))
    t = _tracker(tmp_path, 100, clock)
    t.check()
    assert not (tmp_path / "logs").exists()
    t.record("m", 1, 1, 0, 1)
    assert (tmp_path / "logs" / "usage.jsonl").is_file()


def test_partial_and_estimated_flags_logged_and_counted(tmp_path: Path) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 12, 0, tzinfo=TZ))
    t = _tracker(tmp_path, 1000, clock)
    t.record("m", 5, 200, 0, 10, partial=True, estimated=True)
    entry = json.loads((tmp_path / "logs" / "usage.jsonl").read_text(encoding="utf-8"))
    assert entry["partial"] is True and entry["estimated"] is True
    assert _tracker(tmp_path, 1000, clock).today_total == 205


def test_rebuild_skips_undecodable_lines(tmp_path: Path) -> None:
    path = tmp_path / "usage.jsonl"
    path.write_bytes(
        b'{"date": "2026-10-06", "total": 7}' + bytes([10, 255, 254, 32, 128, 10])
        + b'{"date": "2026-10-06", "total": 3}' + bytes([10])
    )
    clock = FakeClock(datetime(2026, 10, 6, 9, 0, tzinfo=TZ))
    assert UsageTracker(path, 100, clock=clock).today_total == 10


def test_failed_write_still_counts(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    clock = FakeClock(datetime(2026, 10, 6, 12, 0, tzinfo=TZ))
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x", encoding="utf-8")
    t = UsageTracker(blocker / "usage.jsonl", 10, clock=clock)
    with caplog.at_level("ERROR"):
        t.record("m", 10, 0, 0, 1)
    assert t.today_total == 10
    with pytest.raises(CapExceeded):
        t.check()
    assert "could not write" in caplog.text
