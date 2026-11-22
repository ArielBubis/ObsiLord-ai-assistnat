"""The activity log: one JSON line per tool call in logs/activity.jsonl with the
time, the tool, its mode and the outcome. Never arguments, results or any
other content. Pruned at startup with [memory] retention_days (0 keeps all);
Settings shows the newest entries.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from obsilord.tools.registry import MODES

log = logging.getLogger(__name__)


def _local_now() -> datetime:
    return datetime.now().astimezone()


class ActivityLog:
    def __init__(
        self,
        path: Path,
        retention_days: int,
        now: Callable[[], datetime] = _local_now,
    ) -> None:
        self.path = path
        self.retention_days = retention_days
        self._now = now

    def add(self, tool: str, mode: str, outcome: str) -> None:
        """One line for one call. A failed write is logged, never raised: the
        turn goes on."""
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        line = json.dumps(
            {"ts": self._now().isoformat(timespec="seconds"), "tool": tool, "mode": mode, "outcome": outcome}
        )
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            log.error("activity log: could not write %s (%s)", self.path.name, type(exc).__name__)

    def newest(self, limit: int) -> list[dict[str, str]]:
        """Up to limit entries, newest first. Lines that cannot be read are
        skipped."""
        return self._entries()[::-1][:limit]

    def prune(self) -> int:
        """Drops entries older than retention_days (0 keeps all) and returns
        how many went. Unreadable lines go too."""
        if self.retention_days <= 0 or not self.path.is_file():
            return 0
        cutoff = self._now() - timedelta(days=self.retention_days)
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.error("activity log: could not read %s (%s)", self.path.name, type(exc).__name__)
            return 0
        kept = [line for line in lines if (ts := _time(line)) is not None and ts >= cutoff]
        dropped = len(lines) - len(kept)
        if not dropped:
            return 0
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            tmp.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            log.error("activity log: could not prune %s (%s)", self.path.name, type(exc).__name__)
            return 0
        log.info("activity log: pruned %d entries older than %d days", dropped, self.retention_days)
        return dropped

    def _entries(self) -> list[dict[str, str]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        except OSError as exc:
            log.error("activity log: could not read %s (%s)", self.path.name, type(exc).__name__)
            return []
        return [entry for line in lines if (entry := _entry(line)) is not None]


_FIELDS = ("ts", "tool", "mode", "outcome")


def _entry(line: str) -> dict[str, str] | None:
    try:
        data = json.loads(line)
    except ValueError:
        return None
    if not isinstance(data, dict) or not all(isinstance(data.get(k), str) for k in _FIELDS):
        return None
    return {k: data[k] for k in _FIELDS}


def _time(line: str) -> datetime | None:
    entry = _entry(line)
    if entry is None:
        return None
    try:
        ts = datetime.fromisoformat(entry["ts"])
    except ValueError:
        return None
    return ts if ts.tzinfo is not None else None
