"""What the user lets each tool do, and the Photos setting, in one file on
this PC (data/settings.json), so every open page sees the same choices.

The file holds only the choices the user made:
    {"tools": {"look": {"mode": "ask", "cloud": false}}, "photos": "disk"}
Everything else follows the defaults (each tool's policy, [camera] photos),
so a default changed in the code still reaches a tool the user never touched.
Tools in the file that are not offered now (a section switched off) are kept
and written back, so a tool that comes back keeps its choice.

A missing file gives the defaults. An unreadable one also gives the defaults,
logs one line and is kept as settings.json.bad. Writes go to a temp file that
replaces the old one (os.replace), so a failed write leaves the old file and
the old choices. Nothing here is ever logged but the file name and the error
type.

answer_kind reads a spoken or typed reply to a confirm question.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping

from obsilord.photos import PHOTO_MODES
from obsilord.tools.registry import MODES, Mode

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolChoice:
    mode: Mode
    # A private-data tool may run with a cloud model (its results go there).
    cloud: bool


@dataclass(frozen=True)
class Choices:
    """The choices now: every known tool (defaults filled in) and Photos."""

    tools: Mapping[str, ToolChoice]
    photos: str


Subscriber = Callable[[Choices], Awaitable[None]]


class SettingsStore:
    def __init__(self, path: Path, defaults: Mapping[str, ToolChoice], photos_default: str) -> None:
        self.path = path
        self._defaults = dict(defaults)
        self._photos_default = photos_default
        self._lock = asyncio.Lock()
        self._subscribers: list[Subscriber] = []
        self._tools, self._photos = self._load()

    def get(self) -> Choices:
        tools = {}
        for name, default in self._defaults.items():
            chosen = self._tools.get(name, {})
            tools[name] = ToolChoice(chosen.get("mode", default.mode), chosen.get("cloud", default.cloud))
        return Choices(tools, self._photos or self._photos_default)

    # The registry's Permissions: None for a tool the store does not know
    # (the policy default applies then).
    def mode(self, name: str) -> Mode | None:
        choice = self.get().tools.get(name)
        return choice.mode if choice is not None else None

    def cloud(self, name: str) -> bool | None:
        choice = self.get().tools.get(name)
        return choice.cloud if choice is not None else None

    async def set_mode(self, tool: str, mode: str) -> Choices:
        self._known(tool)
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        return await self._change(tool, "mode", mode)

    async def set_cloud(self, tool: str, on: bool) -> Choices:
        self._known(tool)
        if not isinstance(on, bool):
            raise ValueError("cloud must be true or false")
        return await self._change(tool, "cloud", on)

    async def set_photos(self, mode: str) -> Choices:
        if mode not in PHOTO_MODES:
            raise ValueError(f"unknown photos mode {mode!r}")
        return await self._change(None, "photos", mode)

    def subscribe(self, callback: Subscriber) -> Callable[[], None]:
        """callback gets the new Choices after every change. Returns the
        function that unsubscribes it."""
        self._subscribers.append(callback)

        def leave() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return leave

    # --- internals ------------------------------------------------------------------

    def _known(self, tool: str) -> None:
        if tool not in self._defaults:
            raise ValueError(f"unknown tool {tool!r}")

    async def _change(self, tool: str | None, key: str, value: Any) -> Choices:
        async with self._lock:
            tools = {name: dict(fields) for name, fields in self._tools.items()}
            photos = self._photos
            if tool is None:
                photos = value
            else:
                tools.setdefault(tool, {})[key] = value
            data: dict[str, Any] = {"tools": tools}
            if photos is not None:
                data["photos"] = photos
            await asyncio.to_thread(self._write, data)
            self._tools, self._photos = tools, photos
            snapshot = self.get()
        for callback in list(self._subscribers):
            try:
                await callback(snapshot)
            except Exception as exc:  # noqa: BLE001 - one page gone must not stop the rest
                log.debug("settings: a subscriber failed (%s)", type(exc).__name__)
        return snapshot

    def _write(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def _load(self) -> tuple[dict[str, dict[str, Any]], str | None]:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            return {}, None
        except OSError as exc:
            log.warning("settings: cannot read %s (%s), using the defaults", self.path.name, type(exc).__name__)
            return {}, None
        try:
            # utf-8-sig: Notepad saves with a BOM. Not UTF-8 is a ValueError; a
            # very deep nest raises RecursionError: both are a broken file.
            data = json.loads(raw.decode("utf-8-sig"))
            if not isinstance(data, dict) or not isinstance(data.get("tools", {}), dict):
                raise ValueError("not a settings object")
        except (ValueError, RecursionError):
            self._keep_bad()
            return {}, None
        tools: dict[str, dict[str, Any]] = {}
        dropped = 0
        for name, fields in data.get("tools", {}).items():
            if not isinstance(name, str) or not isinstance(fields, dict):
                dropped += 1
                continue
            kept: dict[str, Any] = {}
            for key, value in fields.items():
                if (key == "mode" and value in MODES) or (key == "cloud" and isinstance(value, bool)):
                    kept[key] = value
                else:
                    dropped += 1
            if kept:
                tools[name] = kept
        photos = data.get("photos")
        if photos is not None and photos not in PHOTO_MODES:
            dropped += 1
            photos = None
        if dropped:
            log.warning("settings: %d unreadable choices in %s, using their defaults", dropped, self.path.name)
        return tools, photos

    def _keep_bad(self) -> None:
        bad = self.path.with_name(self.path.name + ".bad")
        try:
            os.replace(self.path, bad)
        except OSError as exc:
            log.warning("settings: %s is unreadable and could not be moved (%s), using the defaults",
                        self.path.name, type(exc).__name__)
            return
        log.warning("settings: %s is unreadable, kept as %s, using the defaults", self.path.name, bad.name)


# --- spoken or typed answers to a confirm question ---------------------------------

YES_WORDS = frozenset({"yes", "yeah", "yep", "yup", "sure", "ok", "okay", "do it", "go ahead", "please do", "confirm"})
NO_WORDS = frozenset({"no", "nope", "dont", "do not", "cancel", "stop", "never mind", "nevermind"})
_APOSTROPHES = re.compile(r"['’]")
_NOT_WORD = re.compile(r"[^\w\s]|_")


def answer_kind(text: str) -> Literal["yes", "no", "other"]:
    """"yes" or "no" when text is one of the answer words alone or with
    "please" (case, punctuation and apostrophes do not matter); "other" for
    anything else, which counts as no and is then handled as a normal turn."""
    words = " ".join(_NOT_WORD.sub(" ", _APOSTROPHES.sub("", text.lower()).replace(".", "")).split())
    for phrase in (words, _without_please(words)):
        if phrase in YES_WORDS:
            return "yes"
        if phrase in NO_WORDS:
            return "no"
    return "other"


def _without_please(words: str) -> str:
    parts = words.split()
    if parts and parts[0] == "please":
        parts = parts[1:]
    if parts and parts[-1] == "please":
        parts = parts[:-1]
    return " ".join(parts)
