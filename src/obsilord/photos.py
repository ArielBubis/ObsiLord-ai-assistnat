"""Photos from look(): kept for follow-ups, and saved to disk when asked.

The Photos setting (per page, Settings) has three modes:
- "off": a photo is used for the reply of its turn only.
- "followup": the photos of the newest turns stay in history (in memory) and
  go with later requests to a model that sees photos.
- "disk": as "followup", and every look's frame is also written to
  <root>/<conversation id>/<YYYYMMDD-HHMMSS-mmm>.jpg. The memory store keeps
  the file name, so a resumed conversation gets its photos back.

Nothing else writes frames to disk. File names are made here and checked
again on load, so a stored name cannot point outside the folder. Frames are
never logged.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Awaitable, Callable, Literal

from obsilord.llm.base import Message

log = logging.getLogger(__name__)

PhotoMode = Literal["off", "followup", "disk"]
PHOTO_MODES: tuple[PhotoMode, ...] = ("off", "followup", "disk")
# Folder for photos taken while no conversation is stored (memory off).
NO_CONVERSATION = "no-conversation"

_CONVERSATION = re.compile(r"[0-9a-f]{32}|" + NO_CONVERSATION)
_NAME = re.compile(r"[0-9]{8}-[0-9]{6}-[0-9]{3}(?:-[0-9]{1,4})?\.jpg")
DAY_S = 86400.0
DEFAULT_KEEP = 2


@dataclass(frozen=True)
class TurnPhotos:
    """What the pipeline does with a turn's photo.

    keep: how many of the newest photos stay in history ("followup", "disk").
    save: "disk" only, writes the frame and returns its file name.
    """

    mode: PhotoMode = "off"
    keep: int = DEFAULT_KEEP
    save: Callable[[bytes], Awaitable[str]] | None = None

    @property
    def kept(self) -> int:
        """Photos kept in history: none in "off"."""
        return 0 if self.mode == "off" else max(0, self.keep)


class PhotoStore:
    """The photo folder. All file work runs in a thread."""

    def __init__(self, root: Path, clock: Callable[[], float] = time.time) -> None:
        self.root = root
        self._clock = clock

    async def save(self, conversation_id: str | None, jpeg: bytes) -> str:
        """Write the frame and return its file name. Raises OSError."""
        folder = conversation_id or NO_CONVERSATION
        if not _CONVERSATION.fullmatch(folder):
            raise ValueError("not a conversation id")
        now = self._clock()
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(now)) + f"-{int(now * 1000) % 1000:03d}"
        return await asyncio.to_thread(self._write, self.root / folder, stamp, jpeg)

    @staticmethod
    def _write(folder: Path, stamp: str, jpeg: bytes) -> str:
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(10000):
            name = f"{stamp}.jpg" if i == 0 else f"{stamp}-{i}.jpg"
            try:
                with open(folder / name, "xb") as f:  # never overwrites
                    f.write(jpeg)
                return name
            except FileExistsError:
                continue
        raise OSError("no free photo name")

    async def load(self, conversation_id: str, name: str) -> bytes | None:
        """The saved frame, or None when the name is not one of ours or the
        file is gone."""
        if not _CONVERSATION.fullmatch(conversation_id) or not _NAME.fullmatch(name):
            return None
        path = self.root / conversation_id / name
        try:
            return await asyncio.to_thread(path.read_bytes)
        except OSError:
            return None

    async def prune(self, retention_days: int) -> int:
        """Delete photos older than retention_days (0 keeps everything) and
        the folders left empty. Returns the number of files removed."""
        if retention_days <= 0:
            return 0
        return await asyncio.to_thread(self._prune, self._clock() - retention_days * DAY_S)

    def _prune(self, cutoff: float) -> int:
        if not self.root.is_dir():
            return 0
        removed = 0
        for folder in self.root.iterdir():
            if not folder.is_dir() or not _CONVERSATION.fullmatch(folder.name):
                continue
            for path in folder.iterdir():
                try:
                    if _NAME.fullmatch(path.name) and path.stat().st_mtime < cutoff:
                        path.unlink()
                        removed += 1
                except OSError as exc:
                    log.warning("photos: could not remove an old photo (%s)", type(exc).__name__)
            try:
                folder.rmdir()  # only succeeds when empty
            except OSError:
                pass
        return removed


def keep_newest_photos(history: list[Message], keep: int) -> None:
    """In place: drop the image from every note but the newest keep that have
    one. The notes themselves stay."""
    seen = 0
    for i in range(len(history) - 1, -1, -1):
        m = history[i]
        if not any(n.image is not None for n in m.tools):
            continue
        notes = []
        for n in reversed(m.tools):
            if n.image is not None:
                seen += 1
                if seen > keep:
                    n = replace(n, image=None)
            notes.append(n)
        history[i] = replace(m, tools=tuple(reversed(notes)))


async def attach_photos(
    history: list[Message], store: PhotoStore, conversation_id: str, keep: int
) -> None:
    """In place: load the saved frames of the newest keep notes that name one
    (a resumed conversation in "disk" mode). Missing files are skipped, and a
    note that still holds its image is not read again. The caller makes sure
    no turn changes history meanwhile (the app calls it before a turn)."""
    loaded = 0
    for i in range(len(history) - 1, -1, -1):
        if loaded >= keep:
            return
        m = history[i]
        if not any(n.photo for n in m.tools):
            continue
        notes = []
        for n in reversed(m.tools):
            if n.photo and loaded < keep:
                if n.image is not None:
                    loaded += 1
                else:
                    image = await store.load(conversation_id, n.photo)
                    if image is not None:
                        n = replace(n, image=image)
                        loaded += 1
            notes.append(n)
        history[i] = replace(m, tools=tuple(reversed(notes)))
