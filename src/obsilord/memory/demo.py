"""Made-up conversation history for testing memory by hand.

With [debug] demo_memory = true the app deletes and reseeds a separate file
(data/memory-demo.sqlite3) at every start, so dates stay relative ("yesterday"
is always yesterday) and the real store is never touched. demo_checks() is
the list the debug panel shows: what to ask and what a good answer says.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from obsilord.llm.base import ToolNote
from obsilord.memory.base import MemoryStore, TurnRecord

DEMO_PROVIDER = "demo"
DEMO_MODEL = "demo"


@dataclass(frozen=True)
class DemoTurn:
    user: str
    reply: str
    notes: tuple[ToolNote, ...] = ()


@dataclass(frozen=True)
class DemoCheck:
    """kind: recall (the answer needs recall), recall-none (recall finds
    nothing), inline, tool-note, resume, continue (no recall needed).
    search: keywords a model would pass to recall, with expect_in a text the
    top hit must hold (both checked by the tests); None for other kinds."""

    question: str
    expected: str
    kind: str = "recall"
    search: str | None = None
    expect_in: str = ""


@dataclass(frozen=True)
class DemoConversation:
    days_ago: int
    hour: int
    minute: int
    label: str
    turns: tuple[DemoTurn, ...]
    checks: tuple[DemoCheck, ...] = ()


DEMO_CONVERSATIONS: tuple[DemoConversation, ...] = (
    DemoConversation(
        0, 0, 0, "today",
        (
            DemoTurn(
                "Look, what do you think of this?",
                "That is a lovely blue ceramic owl. It suits the desk.",
                (ToolNote("look", "What is the user showing?", "A small blue ceramic owl figurine on a wooden desk, next to a keyboard."),),
            ),
            DemoTurn("My sister made it.", "She has a good eye. The glaze looks great."),
        ),
        (
            DemoCheck(
                "What did I show you earlier today?",
                "A blue ceramic owl, made by your sister.",
                search="owl",
                expect_in="blue ceramic owl",
            ),
        ),
    ),
    DemoConversation(
        1, 19, 5, "yesterday",
        (
            DemoTurn("I want to make shakshuka tonight.", "Good choice. Do you have eggs, tomatoes and feta?"),
            DemoTurn("Eggs and tomatoes yes, but I'm out of feta.", "Then skip it or use a bit of cottage cheese. It still works."),
        ),
        (
            DemoCheck(
                "What did I want to cook yesterday?",
                "Shakshuka, and you were out of feta.",
                search="cook shakshuka",
                expect_in="shakshuka",
            ),
        ),
    ),
    DemoConversation(
        2, 21, 40, "2 days ago",
        (
            DemoTurn(
                "Can you find out what this figure is?",
                "It is a Funko Pop of Hatsune Miku, number 39.",
                (ToolNote("search_image", "the figure in the user's hand", "Online lookup of the cut-out: Funko Pop! Vocaloid Hatsune Miku #39, vinyl figure."),),
            ),
        ),
        (
            DemoCheck(
                "What was the figure we looked up a few days ago?",
                "A Funko Pop of Hatsune Miku, number 39 (found by the online lookup).",
                search="figure",
                expect_in="hatsune miku",
            ),
        ),
    ),
    DemoConversation(
        6, 9, 15, "6 days ago",
        (
            DemoTurn("Remind me, I booked the dentist for Thursday at 16:30.", "Noted: dentist on Thursday at 16:30."),
        ),
        (
            DemoCheck(
                "When is my dentist appointment?",
                "Thursday at 16:30.",
                search="dentist",
                expect_in="16:30",
            ),
        ),
    ),
    DemoConversation(
        13, 22, 10, "13 days ago",
        (
            DemoTurn("My cat Pita is hiding under the bed again.", "Poor Pita. Did something scare her?"),
            DemoTurn("The vacuum. She hates it.", "Most cats do. Maybe vacuum when she is in another room."),
        ),
        (
            DemoCheck(
                "What's my cat's name, and what does she hate?",
                "Pita, and she hates the vacuum.",
                search="cat",
                expect_in="pita",
            ),
        ),
    ),
    DemoConversation(
        40, 18, 0, "40 days ago",
        (
            DemoTurn("I'm thinking about a trip to Lisbon in spring.", "Lisbon is lovely in April: warm, and before the summer crowds."),
        ),
        (
            DemoCheck(
                "Where did I want to travel in spring?",
                "Lisbon, and April was suggested.",
                search="trip spring",
                expect_in="lisbon",
            ),
        ),
    ),
    DemoConversation(
        200, 14, 20, "200 days ago",
        (
            DemoTurn("My favorite synth is still the Korg minilogue.", "A classic. Four voices of analog warmth."),
        ),
        (
            DemoCheck(
                "What's my favorite synth?",
                "The Korg minilogue.",
                search="synth",
                expect_in="minilogue",
            ),
        ),
    ),
)

EXTRA_CHECKS: tuple[DemoCheck, ...] = (
    DemoCheck(
        "Did we ever talk about skiing?",
        "No, nothing about skiing (recall finds nothing; it should not make something up).",
        kind="recall-none",
        search="skiing",
    ),
    DemoCheck(
        "Ask for a short joke, then: what did you just say?",
        "It repeats the joke without calling recall (this conversation is in context).",
        kind="inline",
    ),
    DemoCheck(
        "Ask it to look at something you hold, then: what color was it?",
        "It answers from the earlier look result, without a new look.",
        kind="tool-note",
    ),
    DemoCheck(
        "Say something, reload the page within 30 minutes, then: what did I just tell you?",
        "The conversation continued after the reload, so it knows.",
        kind="resume",
    ),
    DemoCheck(
        "In Settings, Conversations, continue the Lisbon one, then: which month did you suggest?",
        "April, from the continued conversation (no recall needed).",
        kind="continue",
    ),
)


def when(conv: DemoConversation, now: float) -> float:
    """The conversation's time: two hours ago for today, otherwise its time
    of day, days_ago days back (local time)."""
    if conv.days_ago == 0:
        return now - 2 * 3600
    day = datetime.fromtimestamp(now) - timedelta(days=conv.days_ago)
    return day.replace(hour=conv.hour, minute=conv.minute, second=0, microsecond=0).timestamp()


async def seed_demo(store: MemoryStore, now: float) -> None:
    """Write the demo conversations, oldest first (turns a minute apart)."""
    for conv in sorted(DEMO_CONVERSATIONS, key=lambda c: -c.days_ago):
        t = when(conv, now)
        cid = await store.new_conversation("web", now=t)
        for i, turn in enumerate(conv.turns):
            record = TurnRecord(turn.user, turn.reply, turn.notes, DEMO_PROVIDER, DEMO_MODEL, t + 60 * i)
            await store.add_turn(cid, record)


def reset_demo_db(path: Path) -> None:
    """Delete the demo file and its WAL side files, if any."""
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def demo_checks(now: float) -> list[dict[str, Any]]:
    """The debug panel's list: per demo conversation its date and label, then
    the checks that are not tied to one date."""
    out: list[dict[str, Any]] = []
    for conv in DEMO_CONVERSATIONS:
        date = f"{datetime.fromtimestamp(when(conv, now)):%Y-%m-%d}"
        for check in conv.checks:
            out.append(
                {"date": date, "label": conv.label, "question": check.question, "expected": check.expected, "kind": check.kind}
            )
    for check in EXTRA_CHECKS:
        out.append({"date": "", "label": "any time", "question": check.question, "expected": check.expected, "kind": check.kind})
    return out
