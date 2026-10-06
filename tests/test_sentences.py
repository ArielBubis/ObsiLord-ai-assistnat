import pytest

from pixelbuddy.sentences import SentenceSplitter


def split_all(text: str) -> list[str]:
    s = SentenceSplitter()
    out = s.feed(text)
    rest = s.flush()
    return out + ([rest] if rest else [])


def split_chars(text: str) -> list[str]:
    s = SentenceSplitter()
    out: list[str] = []
    for ch in text:
        out.extend(s.feed(ch))
    rest = s.flush()
    return out + ([rest] if rest else [])


CASES = [
    ("Hello there. How are you?", ["Hello there.", "How are you?"]),
    ("Wow! Really? Yes.", ["Wow!", "Really?", "Yes."]),
    ("What?! No way.", ["What?!", "No way."]),
    ("Pi is 3.14 roughly. Next.", ["Pi is 3.14 roughly.", "Next."]),
    ("Pi is 3.14. Next.", ["Pi is 3.14.", "Next."]),
    ("Version 2.0.1 is out. Get it.", ["Version 2.0.1 is out.", "Get it."]),
    ("Ask Mr. Smith and Dr. Who. Then go.", ["Ask Mr. Smith and Dr. Who.", "Then go."]),
    ("Fruit, e.g. apples, i.e. red ones. Done.", ["Fruit, e.g. apples, i.e. red ones.", "Done."]),
    ("Meet at 5 p.m. Bring snacks.", ["Meet at 5 p.m. Bring snacks."]),
    ("J. R. R. Tolkien wrote it. Cool.", ["J. R. R. Tolkien wrote it.", "Cool."]),
    ("Well... maybe not. Okay.", ["Well... maybe not.", "Okay."]),
    ("I waited... Then it came.", ["I waited...", "Then it came."]),
    ("I waited\u2026 Then it came.", ["I waited\u2026", "Then it came."]),
    ('He said "hi." Then left.', ['He said "hi."', "Then left."]),
    ("It works (mostly.) Fine.", ["It works (mostly.)", "Fine."]),
    ("She asked \u201cwhy?\u201d Nobody knew.", ["She asked \u201cwhy?\u201d", "Nobody knew."]),
    ("Visit example.com today. Bye.", ["Visit example.com today.", "Bye."]),
    ("No trailing punctuation", ["No trailing punctuation"]),
    ("Line one\nLine two.", ["Line one", "Line two."]),
    ("Spaces.   Lots of them.  ", ["Spaces.", "Lots of them."]),
    ("", []),
    ("   ", []),
    ("No. I refuse.", ["No.", "I refuse."]),
    ("1. Open it. 2. Close it.", ["1. Open it.", "2. Close it."]),
    ("Steps: 1. Open it. 10. Done.", ["Steps: 1. Open it.", "10. Done."]),
    ("I have 3. Then I left.", ["I have 3.", "Then I left."]),
    ("Pick 2. 3. Wait.", ["Pick 2.", "3. Wait."]),
    ("The year 2026. Nice.", ["The year 2026.", "Nice."]),
    ("1234. Big number.", ["1234.", "Big number."]),
]


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_split_whole(text: str, expected: list[str]) -> None:
    assert split_all(text) == expected


@pytest.mark.parametrize(("text", "expected"), CASES)
def test_split_char_by_char_matches(text: str, expected: list[str]) -> None:
    assert split_chars(text) == expected


def test_sentence_emitted_only_after_whitespace() -> None:
    s = SentenceSplitter()
    assert s.feed("Hello.") == []
    assert s.feed(" Wor") == ["Hello."]
    assert s.feed("ld!") == []
    assert s.flush() == "World!"


def test_decimal_split_across_deltas() -> None:
    s = SentenceSplitter()
    assert s.feed("It is 3.") == []
    assert s.feed("14 now. ") == ["It is 3.14 now."]
    assert s.flush() == ""


def test_ellipsis_waits_for_next_word() -> None:
    s = SentenceSplitter()
    assert s.feed("Hmm... ") == []
    assert s.feed("so") == []
    assert s.feed(" yes. ") == ["Hmm... so yes."]


def test_flush_resets() -> None:
    s = SentenceSplitter()
    s.feed("partial")
    assert s.flush() == "partial"
    assert s.flush() == ""
    assert s.feed("New one. ") == ["New one."]
