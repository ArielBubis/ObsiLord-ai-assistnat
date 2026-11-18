"""Photos kept for follow-ups and saved to disk (Phase 5f)."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from obsilord.llm.base import Message, ToolNote
from obsilord.photos import (
    NO_CONVERSATION,
    PHOTO_MODES,
    PhotoStore,
    attach_photos,
    keep_newest_photos,
)

CID = "0123456789abcdef0123456789abcdef"
T = time.mktime((2026, 11, 18, 20, 42, 3, 0, 0, -1)) + 0.25


def _store(tmp_path: Path) -> PhotoStore:
    return PhotoStore(tmp_path / "photos", clock=lambda: T)


def test_modes() -> None:
    assert PHOTO_MODES == ("off", "followup", "disk")


def test_save_writes_the_jpeg_under_the_conversation(tmp_path: Path) -> None:
    store = _store(tmp_path)
    name = asyncio.run(store.save(CID, b"jpeg-1"))
    assert name == "20261118-204203-250.jpg"
    assert (tmp_path / "photos" / CID / name).read_bytes() == b"jpeg-1"
    assert asyncio.run(store.load(CID, name)) == b"jpeg-1"


def test_two_photos_in_the_same_millisecond_get_their_own_files(tmp_path: Path) -> None:
    store = _store(tmp_path)

    async def go() -> list[str]:
        return [await store.save(CID, b"a"), await store.save(CID, b"b")]

    first, second = asyncio.run(go())
    assert first != second
    assert asyncio.run(store.load(CID, second)) == b"b"


def test_without_a_conversation_photos_go_to_their_own_folder(tmp_path: Path) -> None:
    store = _store(tmp_path)
    name = asyncio.run(store.save(None, b"x"))
    assert (tmp_path / "photos" / NO_CONVERSATION / name).exists()


@pytest.mark.parametrize(
    "cid, name",
    [
        (CID, "../../config.toml"),
        (CID, "20261118-204203-250.png"),
        ("..", "20261118-204203-250.jpg"),
        ("not-an-id", "20261118-204203-250.jpg"),
        (CID, "missing-20261118.jpg"),
        (CID, "٢٠٢٦١١١٨-204203-250.jpg"),  # non-ASCII digits
    ],
)
def test_load_refuses_names_it_did_not_make(tmp_path: Path, cid: str, name: str) -> None:
    (tmp_path / "config.toml").write_text("secret")
    assert asyncio.run(_store(tmp_path).load(cid, name)) is None


def test_a_missing_photo_loads_as_none(tmp_path: Path) -> None:
    assert asyncio.run(_store(tmp_path).load(CID, "20261118-204203-250.jpg")) is None


def test_prune_removes_photos_older_than_the_retention(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = asyncio.run(store.save(CID, b"old"))
    new = asyncio.run(store.save("fedcba9876543210fedcba9876543210", b"new"))
    old_path = tmp_path / "photos" / CID / old
    os.utime(old_path, (T - 40 * 86400, T - 40 * 86400))
    removed = asyncio.run(store.prune(30))
    assert removed == 1
    assert not (tmp_path / "photos" / CID).exists()  # the empty folder goes too
    assert asyncio.run(store.load("fedcba9876543210fedcba9876543210", new)) == b"new"


def test_prune_with_zero_keeps_everything(tmp_path: Path) -> None:
    store = _store(tmp_path)
    name = asyncio.run(store.save(CID, b"x"))
    os.utime(tmp_path / "photos" / CID / name, (0, 0))
    assert asyncio.run(store.prune(0)) == 0
    assert asyncio.run(store.prune(30)) == 1


def test_prune_without_a_folder_is_fine(tmp_path: Path) -> None:
    assert asyncio.run(_store(tmp_path).prune(30)) == 0


def _turn(image: bytes | None, photo: str = "") -> list[Message]:
    return [
        Message("user", "look"),
        Message("assistant", "A mug.", tools=(ToolNote("look", "q", "Photo taken.", photo=photo, image=image),)),
    ]


def test_keep_newest_photos_drops_the_older_images() -> None:
    history = [*_turn(b"1"), *_turn(b"2"), *_turn(None), *_turn(b"3")]
    keep_newest_photos(history, 2)
    images = [n.image for m in history for n in m.tools]
    assert images == [None, b"2", None, b"3"]
    # The rest of the note stays.
    assert history[1].tools[0].result == "Photo taken."


def test_keep_zero_drops_all_images() -> None:
    history = [*_turn(b"1")]
    keep_newest_photos(history, 0)
    assert history[1].tools[0].image is None


def test_attach_photos_loads_the_newest_saved_ones(tmp_path: Path) -> None:
    store = _store(tmp_path)

    async def go() -> list[Message]:
        a = await store.save(CID, b"a")
        b = await store.save(CID, b"b")
        c = await store.save(CID, b"c")
        history = [*_turn(None, a), *_turn(None, b), *_turn(None, "20991231-000000-000.jpg"), *_turn(None, c)]
        await attach_photos(history, store, CID, 2)
        return history

    history = asyncio.run(go())
    # The newest two notes with a file name: the missing one is skipped, so b and c load.
    assert [n.image for m in history for n in m.tools] == [None, b"b", None, b"c"]


def test_attach_photos_does_not_read_photos_already_in_memory(tmp_path: Path) -> None:
    class NoReads(PhotoStore):
        async def load(self, conversation_id: str, name: str) -> bytes | None:
            raise AssertionError("read a photo that was already loaded")

    history = [*_turn(b"kept", "20261118-204203-250.jpg")]
    asyncio.run(attach_photos(history, NoReads(tmp_path), CID, 2))
    assert history[1].tools[0].image == b"kept"


def test_images_are_not_part_of_note_equality_or_repr() -> None:
    a = ToolNote("look", "q", "r", photo="p.jpg", image=b"x" * 100)
    b = ToolNote("look", "q", "r", photo="p.jpg")
    assert a == b
    assert "xxxx" not in repr(a)
