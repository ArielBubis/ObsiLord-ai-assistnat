"""Pipeline with the Photos setting (Phase 5f): a model that sees photos gets
the frame, photos are kept for follow-ups or saved, and nothing else changes
for a model that does not see them."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fakes import FakeLooker, FakeToolLLM, registry
from fakes import SeeingToolLLM as SeeingLLM
from obsilord.llm.base import Done, Message, TextDelta, ToolCall, ToolNote
from obsilord.look import LOOK_PHOTO_TOOL, LOOK_TOOL
from obsilord.photos import TurnPhotos
from obsilord.pipeline import PHOTO_NOTE_TEXT, run_text_turn
from obsilord.usage import UsageTracker

NOW = datetime(2026, 11, 18, 20, 0, tzinfo=timezone.utc)
FRAME = b"photo-jpeg"


def _look() -> ToolCall:
    return ToolCall("look", {"question": "What am I holding?"}, call_id="c1")


def _rounds() -> list[list[Any]]:
    return [[_look(), Done(10, 2)], [TextDelta("A red mug."), Done(20, 4)]]


def _run(
    tmp_path: Path,
    llm: Any,
    looker: Any,
    photos: TurnPhotos | None = None,
    history: list[Message] | None = None,
) -> tuple[str | None, list[Message], list[tuple[ToolNote, ...]]]:
    history = history if history is not None else []
    stored: list[tuple[ToolNote, ...]] = []

    async def send(msg: Any) -> None:
        pass

    async def on_turn(user: str, reply: str, notes: tuple[ToolNote, ...]) -> None:
        stored.append(notes)

    reply = asyncio.run(
        run_text_turn(
            "What am I holding?",
            history,
            llm,
            UsageTracker(tmp_path / "usage.jsonl", 1000, clock=lambda: NOW),
            send,
            tools=registry(looker),
            on_turn=on_turn,
            photos=photos,
        )
    )
    return reply, history, stored


def test_a_model_that_sees_photos_gets_the_frame_and_not_qwenny(tmp_path: Path) -> None:
    llm, looker = SeeingLLM(_rounds()), FakeLooker()
    reply, _, _ = _run(tmp_path, llm, looker)
    assert reply == "A red mug."
    assert looker.photos_taken == 1 and looker.questions == []  # no Qwenny look
    first, follow = llm.requests
    assert [t.description for t in first.tools] == [LOOK_PHOTO_TOOL.spec.description]
    assert follow.tool_results is not None and follow.tool_results[0].image == FRAME


def test_off_drops_the_photo_after_the_reply(tmp_path: Path) -> None:
    _, history, stored = _run(tmp_path, SeeingLLM(_rounds()), FakeLooker(), TurnPhotos("off"))
    note = history[-1].tools[0]
    assert note.image is None and note.photo == ""
    assert note.result == PHOTO_NOTE_TEXT


def test_followup_keeps_the_photo_in_history(tmp_path: Path) -> None:
    _, history, _ = _run(tmp_path, SeeingLLM(_rounds()), FakeLooker(), TurnPhotos("followup"))
    assert history[-1].tools[0].image == FRAME


def test_followup_keeps_only_the_newest_photos(tmp_path: Path) -> None:
    def turn(image: bytes) -> list[Message]:
        return [Message("user", "look"), Message("assistant", "ok", tools=(ToolNote("look", "q", "r", image=image),))]

    history = [*turn(b"old"), *turn(b"newer")]
    _, history, _ = _run(tmp_path, SeeingLLM(_rounds()), FakeLooker(), TurnPhotos("followup", keep=2), history)
    assert [n.image for m in history for n in m.tools] == [None, b"newer", FRAME]


def test_off_drops_photos_kept_by_an_earlier_turn_before_the_request(tmp_path: Path) -> None:
    # Photos was switched to "off" while a "followup" turn was running.
    kept = [Message("user", "look"), Message("assistant", "ok", tools=(ToolNote("look", "q", "r", image=b"old"),))]
    llm = SeeingLLM([[TextDelta("Fine."), Done(1, 1)]])
    _, history, _ = _run(tmp_path, llm, FakeLooker(), TurnPhotos("off"), kept)
    assert llm.requests[0].history[1].tools[0].image is None
    assert history[1].tools[0].image is None


def test_disk_saves_the_frame_and_names_it_in_the_note(tmp_path: Path) -> None:
    saved: list[bytes] = []

    async def save(jpeg: bytes) -> str:
        saved.append(jpeg)
        return "20261118-200000-000.jpg"

    _, history, stored = _run(tmp_path, SeeingLLM(_rounds()), FakeLooker(), TurnPhotos("disk", save=save))
    assert saved == [FRAME]
    assert history[-1].tools[0].photo == "20261118-200000-000.jpg"
    assert history[-1].tools[0].image == FRAME
    assert stored[0][0].photo == "20261118-200000-000.jpg"


def test_disk_saves_qwenny_looks_too(tmp_path: Path) -> None:
    saved: list[bytes] = []

    async def save(jpeg: bytes) -> str:
        saved.append(jpeg)
        return "20261118-200000-000.jpg"

    looker = FakeLooker()
    looker.snapshots = [("camera", FRAME)]
    llm = FakeToolLLM(_rounds())  # Gemini or a text model: no sees_images
    _, history, _ = _run(tmp_path, llm, looker, TurnPhotos("disk", save=save))
    assert looker.questions == ["What am I holding?"] and looker.photos_taken == 0
    assert saved == [FRAME]
    note = history[-1].tools[0]
    assert note.photo == "20261118-200000-000.jpg"
    assert note.image is None  # this model never gets photos
    assert note.result == "A red mug."  # Qwenny's answer, as before


def test_a_failed_save_does_not_end_the_turn(tmp_path: Path) -> None:
    async def save(jpeg: bytes) -> str:
        raise OSError("disk full")

    reply, history, _ = _run(tmp_path, SeeingLLM(_rounds()), FakeLooker(), TurnPhotos("disk", save=save))
    assert reply == "A red mug."
    assert history[-1].tools[0].photo == ""


def test_a_model_that_does_not_see_photos_keeps_qwenny(tmp_path: Path) -> None:
    llm, looker = SeeingLLM(_rounds(), sees=False), FakeLooker()
    _, history, _ = _run(tmp_path, llm, looker, TurnPhotos("followup"))
    assert looker.questions == ["What am I holding?"] and looker.photos_taken == 0
    assert [t.description for t in llm.requests[0].tools] == [LOOK_TOOL.spec.description]
    assert history[-1].tools[0].image is None


def test_a_failing_image_check_keeps_qwenny(tmp_path: Path) -> None:
    llm, looker = SeeingLLM(_rounds(), sees=RuntimeError("list failed")), FakeLooker()
    reply, _, _ = _run(tmp_path, llm, looker)
    assert reply == "A red mug." and looker.photos_taken == 0


def test_a_failed_photo_keeps_no_image(tmp_path: Path) -> None:
    looker = FakeLooker()
    looker.photo = None  # the camera failed
    _, history, _ = _run(tmp_path, SeeingLLM(_rounds()), looker, TurnPhotos("followup"))
    note = history[-1].tools[0]
    assert note.ok is False and note.image is None
