"""The assistant's system prompt, shared by every LLM implementation, and
the date and time line each client adds to it per request."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from obsilord.config import REPO_ROOT

SYSTEM_PROMPT_PATH: Path = REPO_ROOT / "prompts" / "system.md"


def load_system_prompt(path: Path | None = None) -> str:
    """Read the system prompt (prompts/system.md in the repo root by default)."""
    path = path if path is not None else SYSTEM_PROMPT_PATH
    return path.read_text(encoding="utf-8").strip()


# Room the date line takes, for the usage estimate of a failed request.
NOW_LINE_CHARS = 60


def now_line(ts: float) -> str:
    """The current date and time as a paragraph for the end of the system
    prompt, so the model can work out "yesterday" or "on Friday"."""
    now = datetime.fromtimestamp(ts)
    return f"\n\nCurrent date and time: {now:%A} {now:%Y-%m-%d}, {now:%H:%M}."
