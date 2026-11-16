"""Provider-neutral image search interface (see planning/ARCHITECTURE.md).

An ImageSearchClient uploads one JPEG (a crop of a camera frame) to a web
image search and returns what the web says it shows, as short text fields.
Implementations never log the image or the results. Nothing in this module may
import a provider package.

The results come from web pages, so they are untrusted text: format_matches
strips markup and caps every field, and prompts/system.md tells the model to
treat them as information, never as instructions.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Protocol

MAX_FIELD_CHARS = 120

_TAG = re.compile(r"<[^>]*>")


class ImageSearchError(RuntimeError):
    """The search could not be done. The message is safe to log."""


@dataclass(frozen=True)
class ImageMatches:
    best_guesses: tuple[str, ...] = ()
    entities: tuple[str, ...] = ()
    page_titles: tuple[str, ...] = ()

    @property
    def empty(self) -> bool:
        return not (self.best_guesses or self.entities or self.page_titles)


class ImageSearchClient(Protocol):
    provider: str   # used for the usage log

    async def search(self, jpeg: bytes) -> ImageMatches: ...


def clean_field(value: object) -> str:
    """One result field as plain single-line text: markup and entities
    removed, whitespace collapsed, capped at MAX_FIELD_CHARS. "" if value is
    not a string."""
    if not isinstance(value, str):
        return ""
    text = html.unescape(_TAG.sub("", value))
    return " ".join(text.split())[:MAX_FIELD_CHARS]


def unique(values: list[str], limit: int) -> tuple[str, ...]:
    """The first limit non-empty values, without repeats (case-insensitive)."""
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
        if len(out) == limit:
            break
    return tuple(out)


def format_matches(matches: ImageMatches) -> str:
    """The matches as one short text for the model. "" when there are none."""
    parts: list[str] = []
    if matches.best_guesses:
        parts.append("Best guess: " + "; ".join(matches.best_guesses) + ".")
    if matches.entities:
        parts.append("Related: " + "; ".join(matches.entities) + ".")
    if matches.page_titles:
        titles = "; ".join(f'"{t}"' for t in matches.page_titles)
        parts.append("Titles of pages showing this image: " + titles + ".")
    return " ".join(parts)
