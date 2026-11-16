"""Cost guard for image search: an hourly cap and a monthly cap.

Every upload appends one JSON line to the search log (logs/image_search.jsonl
by default): the time, the provider and whether it worked. Never the image,
the target or the results. The monthly count is read from that log at startup
and when the month changes, so it survives a restart; the hourly count lives
in memory. Failed uploads count too, since a provider may bill them.

refusal() and record() have no await between them in the caller, so two
searches cannot both pass the cap.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

HOUR_S = 3600.0


def _local_now() -> datetime:
    return datetime.now().astimezone()


def msg_hourly(max_per_hour: int, wait_s: float) -> str:
    return (
        f"Image search limit reached ({max_per_hour} per hour). "
        f"Try again in {max(1, math.ceil(wait_s / 60))} min."
    )


def msg_monthly(max_per_month: int) -> str:
    return f"Image search limit reached ({max_per_month} per month)."


class SearchBudget:
    def __init__(
        self,
        log_path: Path,
        max_per_hour: int,
        max_per_month: int,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = _local_now,
    ) -> None:
        self.log_path = log_path
        self.max_per_hour = max_per_hour
        self.max_per_month = max_per_month
        self._clock = clock
        self._now = now
        self._times: deque[float] = deque()
        self._month = self._month_key()
        self._month_count = self._load_month(self._month)

    def _month_key(self) -> str:
        return self._now().strftime("%Y-%m")

    def _load_month(self, month: str) -> int:
        """Uploads logged for month. Malformed lines are skipped."""
        if not self.log_path.is_file():
            return 0
        count = 0
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            log.error("image search: could not read %s: %s", self.log_path, type(exc).__name__)
            return 0
        for lineno, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                ts = json.loads(line)["ts"]
            except (ValueError, KeyError, TypeError):
                log.warning(
                    "image search: skipping malformed line %d in %s", lineno, self.log_path
                )
                continue
            if isinstance(ts, str) and ts.startswith(month):
                count += 1
        return count

    def _roll(self) -> None:
        month = self._month_key()
        if month != self._month:
            self._month = month
            self._month_count = self._load_month(month)

    def month_total(self) -> int:
        self._roll()
        return self._month_count

    def refusal(self) -> str | None:
        """Why a search is not allowed now, or None if it is."""
        self._roll()
        if self._month_count >= self.max_per_month:
            return msg_monthly(self.max_per_month)
        now = self._clock()
        while self._times and now - self._times[0] >= HOUR_S:
            self._times.popleft()
        if len(self._times) >= self.max_per_hour:
            return msg_hourly(self.max_per_hour, HOUR_S - (now - self._times[0]))
        return None

    def record(self, provider: str) -> None:
        """Count one upload, before it is sent."""
        self._roll()
        self._times.append(self._clock())
        self._month_count += 1
        line = json.dumps({"ts": self._now().isoformat(timespec="seconds"), "provider": provider})
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            log.error("image search: could not write %s: %s", self.log_path, type(exc).__name__)
