"""Streaming sentence splitter.

Feed text deltas as they arrive; complete sentences come out as soon as the
boundary is certain. A boundary is one or more terminators (. ! ? or the
ellipsis character), optional closing quotes or brackets, then whitespace.
A newline also ends a sentence.

Not a boundary:
- a period inside a token (3.14, example.com),
- a single period after a known abbreviation (Mr. Dr. vs. ...),
  a dotted abbreviation (e.g. i.e. U.S. a.m.), or a single capital initial (J.),
- an ellipsis followed by a lowercase word ("well... maybe"),
- a list number: a short integer (up to 3 digits) that is the first token of
  the current sentence or follows a colon ("1. Open it." and "Steps: 1. Open
  it." each stay one sentence). A number elsewhere ("I have 3. Then...") still
  ends the sentence.

Because whitespace (and, for ellipses, the next word) must be seen before a
sentence is emitted, the last sentence of a stream comes out of flush().
"""

from __future__ import annotations

import re

ELLIPSIS = "\u2026"
TERMINATORS = frozenset(".!?" + ELLIPSIS)
CLOSERS = frozenset("\"')]}\u201d\u2019")
OPENERS = "\"'([{\u201c\u2018"

ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs",
        "approx", "fig", "dept", "gen", "col", "lt", "sgt", "capt", "rev", "hon",
    }
)
_DOTTED = re.compile(r"^(?:[A-Za-z]\.)+[A-Za-z]$")


def _token_before(buf: str, start: int, end: int) -> str:
    """The whitespace-delimited token ending just before index end."""
    i = end
    while i > start and not buf[i - 1].isspace():
        i -= 1
    return buf[i:end].lstrip(OPENERS)


def _is_list_marker(buf: str, start: int, end: int) -> bool:
    """True if the token before end is a 1 to 3 digit number that starts the
    sentence or directly follows a colon ("Steps: 1. Open it.")."""
    j = end
    while j > start and not buf[j - 1].isspace():
        j -= 1
    token = buf[j:end]
    if not (1 <= len(token) <= 3 and token.isdigit()):
        return False
    before = buf[start:j].rstrip()
    return before == "" or before.endswith(":")


def _is_abbreviation(token: str) -> bool:
    if not token:
        return False
    if token.lower() in ABBREVIATIONS:
        return True
    if _DOTTED.match(token):
        return True
    return len(token) == 1 and token.isupper()


class SentenceSplitter:
    def __init__(self) -> None:
        self._buf = ""

    def feed(self, delta: str) -> list[str]:
        """Add text and return any sentences that are now complete."""
        self._buf += delta
        out: list[str] = []
        buf = self._buf
        n = len(buf)
        start = 0
        i = 0
        while i < n:
            c = buf[i]
            if c == "\n":
                piece = buf[start:i].strip()
                if piece:
                    out.append(piece)
                start = i + 1
                i += 1
                continue
            if c not in TERMINATORS:
                i += 1
                continue
            j = i
            while j < n and buf[j] in TERMINATORS:
                j += 1
            k = j
            while k < n and buf[k] in CLOSERS:
                k += 1
            if k >= n:
                break  # cannot decide yet
            if not buf[k].isspace():
                i = k  # e.g. 3.14 or a.m: not a boundary
                continue
            run = buf[i:j]
            boundary = True
            if run == ".":
                boundary = not (
                    _is_abbreviation(_token_before(buf, start, i))
                    or _is_list_marker(buf, start, i)
                )
            elif set(run) <= {".", ELLIPSIS}:
                m = k
                while m < n and buf[m].isspace() and buf[m] != "\n":
                    m += 1
                if m >= n:
                    break  # need the next word to decide
                if buf[m].islower():
                    boundary = False
            if boundary:
                piece = buf[start:k].strip()
                if piece:
                    out.append(piece)
                start = k
            i = k
        self._buf = buf[start:]
        return out

    def flush(self) -> str:
        """Return whatever is left (stripped) and reset."""
        rest = self._buf.strip()
        self._buf = ""
        return rest
