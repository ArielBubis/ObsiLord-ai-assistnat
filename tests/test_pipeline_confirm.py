"""Ask inside a turn: the spoken question, the confirm messages, the wait, and
what each answer does. Fakes only (planning/phases/T2, Task 4)."""

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from fakes import FakeLooker, FakeSTT, FakeToolLLM, FakeTTS, registry
from obsilord.llm.base import Done, Message, TextDelta, ToolCall
from obsilord.pipeline import MSG_NOT_CAUGHT, MSG_TIMEOUT, Confirms, run_text_turn, run_voice_turn
from obsilord.tools.registry import msg_declined, msg_no_answer, msg_no_page
from obsilord.usage import UsageTracker

NOW = datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)
LOOK = ToolCall("look", {"question": "What am I holding?"}, call_id="c1")
QUESTION = "Look through the camera?"


class AskFor:
    """Permissions: the named tools ask (and the user allowed them with a
    cloud model), every other tool is On with its policy's cloud default."""

    def __init__(self, *names: str) -> None:
        self.names = set(names)

    def mode(self, name: str) -> str:
        return "ask" if name in self.names else "on"

    def cloud(self, name: str) -> bool | None:
        return True if name in self.names else None


class Answers:
    """The app's side of a confirm: gives the scripted replies in order.

    A reply of None never comes (the wait runs into the confirm timeout).
    delay_s: wait this long before each reply. ids: the confirm ids asked for.
    looks_at_ask: how many looks had run when each reply was asked for.
    """

    def __init__(self, *replies: Any, delay_s: float = 0.0, looker: FakeLooker | None = None) -> None:
        self.replies = list(replies)
        self.delay_s = delay_s
        self.looker = looker
        self.ids: list[str] = []
        self.looks_at_ask: list[int] = []
        self.next_turns: list[str] = []
        self.cancelled = False

    async def reply(self, confirm_id: str) -> Any:
        self.ids.append(confirm_id)
        if self.looker is not None:
            self.looks_at_ask.append(len(self.looker.questions))
        value = self.replies.pop(0)
        try:
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            if value is None:
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return value

    def run_next(self, text: str) -> None:
        self.next_turns.append(text)

    def confirms(self, timeout_s: float = 5.0, stt: Any = None) -> Confirms:
        return Confirms(reply=self.reply, run_next=self.run_next, timeout_s=timeout_s, stt=stt)


def _llm(cloud: bool = False, before: str = "") -> FakeToolLLM:
    first: list[Any] = [TextDelta(before)] if before else []
    return FakeToolLLM([[*first, LOOK, Done(1, 1, 0)], [TextDelta("A red mug."), Done(1, 1, 0)]], cloud=cloud)


def _run(
    llm: FakeToolLLM,
    tmp_path: Path,
    confirms: Confirms | None,
    *,
    looker: FakeLooker | None = None,
    tts: FakeTTS | None = None,
    turn_timeout_s: float = 10.0,
) -> tuple[str | None, list[Any]]:
    sent: list[Any] = []

    async def send(msg: Any) -> None:
        sent.append(msg)

    reply = asyncio.run(
        run_text_turn(
            "What am I holding?",
            [],
            llm,
            UsageTracker(tmp_path / "usage.jsonl", 100000, clock=lambda: NOW),
            send,
            tts=tts,
            turn_timeout_s=turn_timeout_s,
            tools=registry(looker or FakeLooker(), permissions=AskFor("look")),
            confirms=confirms,
        )
    )
    return reply, sent


def _of(sent: list[Any], kind: str) -> list[dict[str, Any]]:
    return [m for m in sent if isinstance(m, dict) and m.get("type") == kind]


def _states(sent: list[Any]) -> list[str]:
    return [m["state"] for m in _of(sent, "state")]


def _look_result(llm: FakeToolLLM) -> dict[str, Any]:
    (result,) = llm.requests[1].tool_results or []
    return result.response


def test_the_question_is_spoken_after_the_speech_before_the_call(tmp_path: Path) -> None:
    tts = FakeTTS()
    answers = Answers(True)
    _run(_llm(before="Let me see."), tmp_path, answers.confirms(), tts=tts)
    assert tts.texts == ["Let me see.", QUESTION, "A red mug."]


def test_the_confirm_message_goes_out_after_the_question_audio(tmp_path: Path) -> None:
    tts = FakeTTS()
    answers = Answers(True)
    _, sent = _run(_llm(before="Let me see."), tmp_path, answers.confirms(timeout_s=20), tts=tts)
    (confirm,) = _of(sent, "confirm")
    assert confirm == {
        "type": "confirm",
        "id": answers.ids[0],
        "tool": "look",
        "label": "Camera",
        "question": QUESTION,
        "details": "The model's words: question: What am I holding?",
        "timeout_s": 20,
    }
    # The question is sentence 2; its audio has gone out before the card.
    assert sent.index({"type": "tts_end", "sentence_id": 2}) < sent.index(confirm)
    assert _of(sent, "confirm_done") == [{"type": "confirm_done", "id": answers.ids[0], "answer": "yes"}]


def test_the_turn_waits_and_yes_runs_the_tool(tmp_path: Path) -> None:
    looker = FakeLooker()
    answers = Answers(True, looker=looker)
    llm = _llm()
    reply, _ = _run(llm, tmp_path, answers.confirms(), looker=looker)
    assert answers.looks_at_ask == [0]
    assert looker.questions == ["What am I holding?"]
    assert _look_result(llm) == {"output": "A red mug."}
    assert reply == "A red mug."


@pytest.mark.parametrize("cloud", [False, True])
def test_no_gives_the_refusal_to_the_model(tmp_path: Path, cloud: bool) -> None:
    looker = FakeLooker()
    answers = Answers(False)
    llm = _llm(cloud=cloud)
    reply, sent = _run(llm, tmp_path, answers.confirms(), looker=looker)
    assert looker.questions == []
    assert _look_result(llm) == {"error": msg_declined("Camera")}
    assert _of(sent, "confirm_done")[0]["answer"] == "no"
    assert reply == "A red mug."


@pytest.mark.parametrize("cloud", [False, True])
def test_a_timeout_counts_as_no_answer(tmp_path: Path, cloud: bool) -> None:
    looker = FakeLooker()
    answers = Answers(None)
    llm = _llm(cloud=cloud)
    _, sent = _run(llm, tmp_path, answers.confirms(timeout_s=0.05), looker=looker)
    assert looker.questions == []
    assert _look_result(llm) == {"error": msg_no_answer("Camera")}
    assert _of(sent, "confirm_done")[0]["answer"] == "timeout"


def test_no_page_counts_as_no(tmp_path: Path) -> None:
    llm = _llm()
    _, sent = _run(llm, tmp_path, None)
    assert _look_result(llm) == {"error": msg_no_page("Camera")}
    assert _of(sent, "confirm") == []


@pytest.mark.parametrize("typed", ["yes", "Yes please.", "ok", "go ahead"])
def test_typed_yes_words_run_the_tool(tmp_path: Path, typed: str) -> None:
    looker = FakeLooker()
    answers = Answers(typed)
    _run(_llm(), tmp_path, answers.confirms(), looker=looker)
    assert looker.questions == ["What am I holding?"]
    assert answers.next_turns == []


def test_typed_no_declines(tmp_path: Path) -> None:
    looker = FakeLooker()
    answers = Answers("nope")
    llm = _llm()
    _run(llm, tmp_path, answers.confirms(), looker=looker)
    assert _look_result(llm) == {"error": msg_declined("Camera")}
    assert answers.next_turns == []


def test_another_answer_counts_as_no_and_runs_next(tmp_path: Path) -> None:
    looker = FakeLooker()
    answers = Answers("what time is it")
    llm = _llm()
    _, sent = _run(llm, tmp_path, answers.confirms(), looker=looker)
    assert looker.questions == []
    assert _look_result(llm) == {"error": msg_declined("Camera")}
    assert _of(sent, "confirm_done")[0]["answer"] == "no"
    assert answers.next_turns == ["what time is it"]


def test_a_spoken_answer_is_transcribed_and_shown(tmp_path: Path) -> None:
    looker = FakeLooker()
    stt = FakeSTT(text="Yeah.")
    answers = Answers(b"\x00\x00" * 1600)
    _, sent = _run(_llm(), tmp_path, answers.confirms(stt=stt), looker=looker)
    assert len(stt.calls) == 1
    assert {"type": "transcript", "role": "user", "text": "Yeah.", "final": True} in sent
    assert looker.questions == ["What am I holding?"]


def test_a_spoken_answer_that_is_not_caught_keeps_waiting(tmp_path: Path) -> None:
    looker = FakeLooker()
    stt = FakeSTT(text="  ")
    answers = Answers(b"\x00\x00" * 1600, True)
    _, sent = _run(_llm(), tmp_path, answers.confirms(stt=stt), looker=looker)
    assert {"type": "error", "message": MSG_NOT_CAUGHT} in sent
    assert len(answers.ids) == 2 and answers.ids[0] == answers.ids[1]
    assert looker.questions == ["What am I holding?"]


def test_the_turn_deadline_moves_out_by_the_waiting_time(tmp_path: Path) -> None:
    looker = FakeLooker()
    answers = Answers(True, delay_s=0.4)
    reply, sent = _run(_llm(), tmp_path, answers.confirms(), looker=looker, turn_timeout_s=0.25)
    assert {"type": "error", "message": MSG_TIMEOUT} not in sent
    assert reply == "A red mug."


def test_the_latency_line_counts_the_wait_apart_from_the_tools(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="obsilord.pipeline")
    answers = Answers(True, delay_s=0.3)
    _run(_llm(), tmp_path, answers.confirms())
    line = next(r.getMessage() for r in caplog.records if "turn latency" in r.getMessage())
    fields = dict(part.split("=") for part in line.split(": ", 1)[1].split())
    # Windows timers can wake a sleep a few ms early: margins, not exact values.
    assert int(fields["ask_ms"]) >= 250
    assert int(fields["tools_ms"]) < 250


def _logged(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records]


@pytest.mark.parametrize(
    ("reply", "line"),
    [
        (True, "confirm look: yes (click)"),
        (False, "confirm look: no (click)"),
        ("yes please", "confirm look: yes (typed)"),
        ("what time is it", "confirm look: no (typed, other words, run next)"),
        (None, "confirm look: timeout (no reply)"),
    ],
)
def test_each_confirm_logs_its_answer_without_the_words(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, reply: Any, line: str
) -> None:
    caplog.set_level("INFO", logger="obsilord.pipeline")
    answers = Answers(reply)
    _run(_llm(), tmp_path, answers.confirms(timeout_s=0.05))
    logged = _logged(caplog)
    assert "confirm look: asked (timeout 0.05 s)" in logged
    assert any(m.startswith(line + " after ") and m.endswith(" ms") for m in logged), logged
    assert not any("what time" in m for m in logged)


def test_a_spoken_confirm_logs_how_it_came(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="obsilord.pipeline")
    answers = Answers(b"\x00\x00" * 1600, b"\x00\x00" * 1600, True)
    stt = FakeSTT(text="")
    _run(_llm(), tmp_path, answers.confirms(timeout_s=5, stt=stt))
    logged = _logged(caplog)
    assert logged.count("confirm look: a spoken answer was not caught, still waiting") == 2
    assert any(m.startswith("confirm look: yes (click) after ") for m in logged)


def test_the_turn_logs_its_face_states_once_at_idle(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger="obsilord.pipeline")
    answers = Answers(True)
    _run(_llm(before="Let me see."), tmp_path, answers.confirms(), tts=FakeTTS())
    faces = [m for m in _logged(caplog) if m.startswith("turn face:")]
    assert faces == ["turn face: thinking > speaking > thinking > looking > thinking > speaking > idle"]


def test_the_deadline_still_ends_a_turn_after_a_confirm(tmp_path: Path) -> None:
    # The tool after the yes takes longer than the whole turn may.
    looker = FakeLooker(delay_s=0.5)
    answers = Answers(True)
    reply, sent = _run(_llm(), tmp_path, answers.confirms(), looker=looker, turn_timeout_s=0.2)
    assert reply is None
    assert {"type": "error", "message": MSG_TIMEOUT} in sent


def test_the_face_is_thinking_while_waiting_and_looking_after_the_yes(tmp_path: Path) -> None:
    tts = FakeTTS()
    answers = Answers(True)
    _, sent = _run(_llm(before="Let me see."), tmp_path, answers.confirms(), tts=tts)
    confirm = sent.index(_of(sent, "confirm")[0])
    # Speaking (the speech and the question), thinking for the wait, then the card.
    assert _states(sent[:confirm]) == ["thinking", "speaking", "thinking"]
    assert _states(sent[confirm:]) == ["looking", "thinking", "speaking", "idle"]


def test_a_declined_look_never_shows_looking(tmp_path: Path) -> None:
    answers = Answers(False)
    _, sent = _run(_llm(), tmp_path, answers.confirms())
    assert "looking" not in _states(sent)


def test_a_cancelled_turn_cancels_the_wait(tmp_path: Path) -> None:
    answers = Answers(None)

    async def go() -> None:
        async def send(msg: Any) -> None:
            pass

        task = asyncio.create_task(
            run_text_turn(
                "What am I holding?",
                [],
                _llm(),
                UsageTracker(tmp_path / "usage.jsonl", 100000, clock=lambda: NOW),
                send,
                tools=registry(FakeLooker(), permissions=AskFor("look")),
                confirms=answers.confirms(timeout_s=30),
            )
        )
        for _ in range(1000):
            if answers.ids or task.done():
                break
            await asyncio.sleep(0)
        assert answers.ids, "the turn never asked"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(go())
    assert answers.cancelled


def test_a_voice_turn_asks_the_same_way(tmp_path: Path) -> None:
    looker = FakeLooker()
    answers = Answers(True)
    llm = _llm()
    sent: list[Any] = []

    async def send(msg: Any) -> None:
        sent.append(msg)

    reply = asyncio.run(
        run_voice_turn(
            b"\x00\x00" * 16000,
            [Message("user", "hi"), Message("assistant", "Hello.")],
            llm,
            UsageTracker(tmp_path / "usage.jsonl", 100000, clock=lambda: NOW),
            send,
            FakeSTT(text="What am I holding?"),
            tools=registry(looker, permissions=AskFor("look")),
            confirms=answers.confirms(),
        )
    )
    assert reply == "A red mug."
    assert len(_of(sent, "confirm")) == 1
    assert looker.questions == ["What am I holding?"]
