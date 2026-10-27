"""Opt-in log of every request to Gemini and llama-server ([debug] log_prompts).

Off by default: the output holds the system prompt, the conversation, tool
calls and results, and the vision prompt and answer. Camera frames are never
in it, only their size. Records go to the "obsilord.prompts" logger at INFO,
which is held at WARNING unless the setting turns it on. Each provider module
formats its own requests and passes the formatter to emit(), so nothing is
formatted while it is off, and a formatter error never breaks a turn.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

log = logging.getLogger("obsilord.prompts")
log.setLevel(logging.WARNING)


def set_enabled(on: bool) -> None:
    log.setLevel(logging.INFO if on else logging.WARNING)


def enabled() -> bool:
    return log.isEnabledFor(logging.INFO)


def emit(format_fn: Callable[..., str], *args: Any) -> None:
    """Log format_fn(*args) if enabled. A formatter error is logged, never raised."""
    if not enabled():
        return
    try:
        text = format_fn(*args)
    except Exception as exc:  # noqa: BLE001 - debug output must not fail a request
        log.warning("prompt log failed (%s)", type(exc).__name__)
        return
    log.info("%s", text)
