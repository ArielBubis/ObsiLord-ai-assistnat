"""Pure TTS helpers: speech text cleanup and frame splitting."""

import pytest

from obsilord.tts.base import FRAME_MS, frame_bytes, split_frames
from obsilord.tts.text import clean_for_speech, is_speakable


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("Hello there.", "Hello there."),
        ("**Bold** and *italic* words.", "Bold and italic words."),
        ("Use `pip install` now.", "Use pip install now."),
        ("## Heading", "Heading"),
        ("_emph_ text", "emph text"),
        ("snake_case name", "snake case name"),
        ("- a bullet item", "a bullet item"),
        ("* star bullet", "star bullet"),
        ("> quoted line", "quoted line"),
        ("See [the docs](https://example.com) here.", "See the docs here."),
        ("  lots   of\t\tspace \n here ", "lots of space here"),
        ("Pi is 3.14, right?", "Pi is 3.14, right?"),
        ("Café au lait.", "Café au lait."),
    ],
)
def test_clean_for_speech(raw: str, spoken: str) -> None:
    assert clean_for_speech(raw) == spoken


@pytest.mark.parametrize("raw", ["", "   ", "**", "---", "...", "# ", "`` ``", "- ", "!?"])
def test_unspeakable_becomes_empty(raw: str) -> None:
    assert clean_for_speech(raw) == ""


def test_is_speakable() -> None:
    assert is_speakable("a")
    assert is_speakable("7")
    assert not is_speakable(" .,!-*")


def test_frame_bytes_100ms_at_24000() -> None:
    assert FRAME_MS == 100
    assert frame_bytes(24000) == 4800
    assert frame_bytes(16000) == 3200
    assert frame_bytes(24000, 20) == 960
    assert frame_bytes(44100, 7) == 616  # 308.7 samples floor to 308
    assert frame_bytes(10, 1) == 2  # never zero


def test_split_frames_sizes_and_content() -> None:
    pcm = bytes(range(256)) * 40  # 10240 bytes
    frames = split_frames(pcm, 4800)
    assert [len(f) for f in frames] == [4800, 4800, 640]
    assert b"".join(frames) == pcm
    assert all(len(f) % 2 == 0 for f in frames)


def test_split_frames_exact_and_empty() -> None:
    assert split_frames(b"\x00" * 9600, 4800) == [b"\x00" * 4800] * 2
    assert split_frames(b"", 4800) == []


@pytest.mark.parametrize("size", [0, -2, 3])
def test_split_frames_rejects_bad_size(size: int) -> None:
    with pytest.raises(ValueError):
        split_frames(b"\x00\x00", size)
