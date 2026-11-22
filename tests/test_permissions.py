"""The settings store (data/settings.json) and the spoken yes/no words."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import pytest

from obsilord import permissions
from obsilord.permissions import SettingsStore, Choices, ToolChoice, answer_kind

DEFAULTS = {
    "look": ToolChoice("on", True),
    "recall": ToolChoice("on", False),
    "search_image": ToolChoice("ask", True),
}


def _store(path: Path, photos: str = "followup") -> SettingsStore:
    return SettingsStore(path, DEFAULTS, photos)


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


# --- loading ----------------------------------------------------------------------


def test_a_missing_file_gives_the_defaults(tmp_path: Path) -> None:
    snap = _store(tmp_path / "settings.json").get()
    assert snap.tools == DEFAULTS
    assert snap.photos == "followup"
    assert not (tmp_path / "settings.json").exists()  # nothing written until a change


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", '"text"', "", '{"tools": 3}'])
def test_a_broken_file_gives_the_defaults_and_one_log_line(
    tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(content, encoding="utf-8")
    caplog.set_level(logging.WARNING, logger="obsilord.permissions")
    snap = _store(path).get()
    assert snap.tools == DEFAULTS and snap.photos == "followup"
    assert len(caplog.records) == 1
    # The unreadable file is kept next to it, never silently overwritten.
    assert (tmp_path / "settings.json.bad").read_text(encoding="utf-8") == content


@pytest.mark.parametrize("raw", [b"\xff\xfe{}", b"[" * 100_000], ids=["not utf-8", "deep nest"])
def test_a_garbled_file_never_stops_startup(
    tmp_path: Path, raw: bytes, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "settings.json"
    path.write_bytes(raw)
    caplog.set_level(logging.WARNING, logger="obsilord.permissions")
    snap = _store(path).get()
    assert snap.tools == DEFAULTS and snap.photos == "followup"
    assert len(caplog.records) == 1
    assert (tmp_path / "settings.json.bad").read_bytes() == raw


def test_a_file_saved_with_a_bom_keeps_its_choices(tmp_path: Path) -> None:
    # Notepad writes UTF-8 with a BOM.
    path = tmp_path / "settings.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"tools": {"look": {"mode": "ask"}}, "photos": "disk"}).encode())
    snap = _store(path).get()
    assert snap.tools["look"] == ToolChoice("ask", True)
    assert snap.photos == "disk"
    assert not (tmp_path / "settings.json.bad").exists()


def test_bad_fields_fall_back_to_their_defaults_and_keep_the_rest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "tools": {
                    "look": {"mode": "sometimes", "cloud": False},
                    "recall": {"mode": "ask", "cloud": "yes"},
                    "search_image": "on",
                },
                "photos": "everywhere",
            }
        ),
        encoding="utf-8",
    )
    caplog.set_level(logging.WARNING, logger="obsilord.permissions")
    snap = _store(path).get()
    assert snap.tools["look"] == ToolChoice("on", False)
    assert snap.tools["recall"] == ToolChoice("ask", False)
    assert snap.tools["search_image"] == DEFAULTS["search_image"]
    assert snap.photos == "followup"
    assert len(caplog.records) == 1


# --- changes ----------------------------------------------------------------------


def test_set_and_reload(tmp_path: Path) -> None:
    path = tmp_path / "data" / "settings.json"  # the folder is made on first write
    store = _store(path)

    async def go() -> Choices:
        await store.set_mode("look", "ask")
        await store.set_cloud("look", False)
        return await store.set_photos("disk")

    snap = _run(go())
    assert snap.tools["look"] == ToolChoice("ask", False)
    assert snap.photos == "disk"
    again = _store(path).get()
    assert again == snap
    # Only the choices the user made are stored; the rest follow the defaults.
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "tools": {"look": {"mode": "ask", "cloud": False}},
        "photos": "disk",
    }


def test_photos_follow_the_config_default_until_picked(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    _run(_store(path).set_mode("recall", "off"))
    assert _store(path, photos="disk").get().photos == "disk"


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda s: s.set_mode("look", "maybe"), "mode"),
        (lambda s: s.set_mode("delete_files", "on"), "tool"),
        (lambda s: s.set_cloud("delete_files", True), "tool"),
        (lambda s: s.set_cloud("look", "yes"), "cloud"),
        (lambda s: s.set_photos("everywhere"), "photos"),
    ],
)
def test_bad_changes_are_refused(tmp_path: Path, call: object, message: str) -> None:
    path = tmp_path / "settings.json"
    store = _store(path)
    with pytest.raises(ValueError, match=message):
        _run(call(store))  # type: ignore[operator]
    assert store.get().tools == DEFAULTS
    assert not path.exists()


def test_unknown_tools_in_the_file_are_kept(tmp_path: Path) -> None:
    # A tool that is gone for now (its section switched off) keeps its choice.
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"tools": {"weather": {"mode": "off"}}}), encoding="utf-8")
    store = _store(path)
    assert "weather" not in store.get().tools
    _run(store.set_mode("look", "ask"))
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["tools"]["weather"] == {"mode": "off"}
    assert stored["tools"]["look"] == {"mode": "ask"}


def test_a_failed_write_leaves_the_old_file_and_the_old_choices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "settings.json"
    store = _store(path)
    _run(store.set_mode("look", "ask"))
    before = path.read_text(encoding="utf-8")

    def broken_replace(src: object, dst: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(permissions.os, "replace", broken_replace)
    with pytest.raises(OSError):
        _run(store.set_mode("look", "off"))
    monkeypatch.setattr(permissions.os, "replace", os.replace)
    assert path.read_text(encoding="utf-8") == before
    assert store.get().tools["look"].mode == "ask"
    assert [p.name for p in tmp_path.iterdir()] == ["settings.json"]  # no temp file left


def test_subscribers_get_every_change_and_can_leave(tmp_path: Path) -> None:
    store = _store(tmp_path / "settings.json")
    seen: list[str] = []
    other: list[str] = []

    async def first(snap: Choices) -> None:
        seen.append(snap.tools["look"].mode)

    async def broken(snap: Choices) -> None:
        raise RuntimeError("socket gone")

    async def second(snap: Choices) -> None:
        other.append(snap.photos)

    leave = store.subscribe(first)
    store.subscribe(broken)  # a failing subscriber does not stop the others
    store.subscribe(second)

    async def go() -> None:
        await store.set_mode("look", "ask")
        leave()
        await store.set_photos("off")

    _run(go())
    assert seen == ["ask"]
    assert other == ["followup", "off"]


def test_changes_from_two_pages_at_once_both_land(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = _store(path)

    async def go() -> None:
        await asyncio.gather(store.set_mode("look", "off"), store.set_mode("recall", "ask"))

    _run(go())
    again = _store(path).get()
    assert again.tools["look"].mode == "off" and again.tools["recall"].mode == "ask"


# --- spoken answers ------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["yes", "yeah", "yep", "yup", "sure", "ok", "okay", "do it", "go ahead", "please do", "confirm",
     "Yes.", "YES!", "yes please", "Please, go ahead.", "  okay  ", "O.K.", "Go-ahead", "Sure, please."],
)
def test_yes_words(text: str) -> None:
    assert answer_kind(text) == "yes"


@pytest.mark.parametrize(
    "text",
    ["no", "nope", "don't", "do not", "cancel", "stop", "never mind",
     "No.", "NO!", "no please", "Please don’t.", "dont", "Cancel, please", "Never mind!", "nevermind"],
)
def test_no_words(text: str) -> None:
    assert answer_kind(text) == "no"


@pytest.mark.parametrize(
    "text",
    ["what time is it", "yes and also tell me a joke", "no thanks", "please", "", "   ", "...",
     "yes no", "I think so", "sure, but what is it for?"],
)
def test_anything_else_is_other(text: str) -> None:
    assert answer_kind(text) == "other"


# --- as the registry's Permissions ------------------------------------------------


def test_the_store_answers_the_registry_per_tool(tmp_path: Path) -> None:
    store = _store(tmp_path / "settings.json")
    _run(store.set_mode("look", "off"))
    _run(store.set_cloud("recall", True))
    assert (store.mode("look"), store.cloud("look")) == ("off", True)
    assert (store.mode("recall"), store.cloud("recall")) == ("on", True)
    assert store.mode("search_image") == "ask"


def test_a_tool_the_store_does_not_know_gives_the_policy_default(tmp_path: Path) -> None:
    store = _store(tmp_path / "settings.json")
    assert store.mode("calculate") is None and store.cloud("calculate") is None
