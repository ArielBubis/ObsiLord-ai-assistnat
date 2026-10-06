"""Token usage logging and the daily token cap.

Each LLM call appends one JSON line to the usage log (logs/usage.jsonl by
default) with counts only: never prompt or reply text. The daily total is the
sum of input, output, and thinking tokens for the current local date. It is
rebuilt from the log at startup and resets at local midnight.

The cap is checked before a call, so a single call that starts under the cap
can finish over it. The next call is then blocked.

check() followed by the LLM call is not atomic: two connections could both
pass check() before either records. That is acceptable for a single-user
localhost app; a shared deployment would need a reservation or a lock.

Records can be flagged "partial" (stream did not finish, counts are the last
ones the provider reported) or "estimated" (no counts were reported at all).
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


class CapExceeded(Exception):
    """Raised by UsageTracker.check() when today's tokens reached the cap."""


def _local_now() -> datetime:
    return datetime.now().astimezone()


class UsageTracker:
    def __init__(
        self,
        log_path: Path,
        daily_token_cap: int,
        clock: Callable[[], datetime] = _local_now,
    ) -> None:
        if daily_token_cap < 0:
            raise ValueError("daily_token_cap must be >= 0")
        self.log_path = log_path
        self.daily_token_cap = daily_token_cap
        self._clock = clock
        self._day: date = clock().date()
        self._total = self._load_total(self._day)
        log.info(
            "usage: %d tokens used today (cap %d)", self._total, self.daily_token_cap
        )

    def _load_total(self, day: date) -> int:
        """Sum today's tokens from the existing log.

        Malformed or undecodable lines are skipped. An unreadable file raises
        (OSError), so the cap is never silently reset.
        """
        if not self.log_path.is_file():
            return 0
        total = 0
        day_str = day.isoformat()
        data = self.log_path.read_bytes()
        for lineno, raw in enumerate(data.split(b"\n"), 1):
            try:
                line = raw.decode("utf-8").strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("date") != day_str:
                    continue
                total += int(entry["total"])
            except (ValueError, KeyError, TypeError, AttributeError):
                # UnicodeDecodeError and JSONDecodeError are ValueErrors.
                log.warning("usage: skipping malformed line %d in %s", lineno, self.log_path)
        return total

    def _roll(self) -> date:
        today = self._clock().date()
        if today != self._day:
            self._day = today
            self._total = self._load_total(today)
        return today

    @property
    def today_total(self) -> int:
        self._roll()
        return self._total

    def check(self) -> None:
        """Raise CapExceeded if today's total is at or above the cap."""
        self._roll()
        if self._total >= self.daily_token_cap:
            raise CapExceeded(
                f"daily token cap reached ({self._total} >= {self.daily_token_cap})"
            )

    def record(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        tokens_thinking: int,
        latency_ms: int,
        partial: bool = False,
        estimated: bool = False,
    ) -> None:
        """Add to today's total, then append one usage line. Counts only.

        The total is updated first so a failed log write still counts toward
        the cap. Write failures are logged at ERROR and not raised.
        """
        now = self._clock()
        self._roll()
        total = int(tokens_in) + int(tokens_out) + int(tokens_thinking)
        self._total += total
        entry = {
            "ts": now.isoformat(timespec="seconds"),
            "date": now.date().isoformat(),
            "model": model,
            "tokens_in": int(tokens_in),
            "tokens_out": int(tokens_out),
            "tokens_thinking": int(tokens_thinking),
            "total": total,
            "latency_ms": int(latency_ms),
            "partial": bool(partial),
            "estimated": bool(estimated),
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")
        except OSError as exc:
            log.error("usage: could not write %s: %s", self.log_path, type(exc).__name__)
