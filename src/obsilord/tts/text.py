"""Text cleanup before speech synthesis (pure, provider-neutral).

The LLM sometimes answers with light markdown. The transcript keeps the raw
text; only what is spoken is cleaned:
- markdown links [label](url) become the label,
- a leading bullet or quote marker ("- ", "* ", "+ ", "> ") is dropped,
- the symbols * _ # and backticks are removed (underscores become spaces, so
  snake_case reads as two words),
- whitespace is collapsed.
A sentence with no letters or digits left (for example "---" or "**") has
nothing to say and comes back as an empty string, so callers skip it.
"""

from __future__ import annotations

import re

_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_LEADING_MARKER = re.compile(r"^\s*(?:[-*+>]\s+)+")
_REMOVE = str.maketrans({"*": "", "#": "", "`": "", "_": " "})
_SPACE = re.compile(r"\s+")


def is_speakable(text: str) -> bool:
    """True if text has at least one letter or digit."""
    return any(c.isalnum() for c in text)


def clean_for_speech(text: str) -> str:
    """Return text fit for TTS, or "" if nothing speakable is left."""
    text = _LINK.sub(r"\1", text)
    text = _LEADING_MARKER.sub("", text)
    text = text.translate(_REMOVE)
    text = _SPACE.sub(" ", text).strip()
    return text if is_speakable(text) else ""
