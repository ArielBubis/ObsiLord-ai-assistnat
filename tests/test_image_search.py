"""search_image(): result parsing, the Google Vision client with a fake opener,
the search budget, box parsing, cropping, the Looker flow with fakes, the
pipeline routing and the snapshot messages. No camera, no model, no network."""

import asyncio
import dataclasses
import io
import json
import logging
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import cv2
import numpy
import pytest

from fakes import AppClient, FakeLooker, FakeToolLLM, registry, text_only
from obsilord.app import _default_image_search, create_app
from obsilord.camera import crop_jpeg, encode_frame
from obsilord.config import REPO_ROOT, ImageSearchSettings, Secrets, Settings, load_settings
from obsilord.imagesearch.base import (
    MAX_FIELD_CHARS,
    ImageMatches,
    ImageSearchError,
    clean_field,
    format_matches,
    unique,
)
from google.genai import types

from obsilord.imagesearch.budget import SearchBudget
from obsilord.imagesearch.gemini import PROMPT as GEMINI_PROMPT
from obsilord.imagesearch.gemini import MSG_CAP as GEMINI_MSG_CAP
from obsilord.imagesearch.gemini import GeminiImageIdentify, clean_answer
from obsilord.imagesearch.google_vision import (
    ENDPOINT,
    GoogleVisionSearch,
    build_request,
    parse_response,
)
from obsilord.llm.base import Done, TextDelta, ToolCall, ToolResult
from obsilord.look import (
    DEFAULT_TARGET,
    LOOK_TOOL,
    MAX_TARGET_CHARS,
    MSG_NOT_FOUND,
    MSG_SEARCH_FAILED,
    MSG_SEARCH_OFF,
    MSG_TOO_LARGE,
    MSG_VISION_FAILED,
    SEARCH_IMAGE_TOOL,
    ImageSearch,
    LookLimiter,
    Looker,
    clean_target,
)
from obsilord.pipeline import run_text_turn
from obsilord.tools.registry import msg_limit, msg_unknown
from obsilord.usage import UsageTracker
from obsilord.vision.base import Box, VisionError
from obsilord.vision.llama_server import LOCATE_PROMPT, QUESTION_PLACEHOLDER, parse_box

NOW = datetime(2026, 11, 14, 12, 0, tzinfo=timezone.utc)


# Result text


def test_clean_field_strips_markup_and_caps() -> None:
    assert clean_field("Gorillaz - <b>Demon Days</b> &amp; more\n here") == (
        "Gorillaz - Demon Days & more here"
    )
    assert clean_field(None) == "" and clean_field(42) == ""
    assert len(clean_field("x" * 500)) == MAX_FIELD_CHARS


def test_unique_keeps_order_and_limit() -> None:
    assert unique(["A", "", "a", "B", "C"], 2) == ("A", "B")


def test_format_matches() -> None:
    matches = ImageMatches(("demon days",), ("Gorillaz", "Album"), ("Demon Days - Wikipedia",))
    assert format_matches(matches) == (
        "Best guess: demon days. Related: Gorillaz; Album. "
        'Titles of pages showing this image: "Demon Days - Wikipedia".'
    )
    assert format_matches(ImageMatches()) == "" and ImageMatches().empty


# Google Vision


def test_build_request() -> None:
    body = build_request(b"\xff\xd8abc", 5)
    request = body["requests"][0]
    assert request["image"] == {"content": "/9hhYmM="}
    assert request["features"] == [{"type": "WEB_DETECTION", "maxResults": 5}]


ANSWER = {
    "responses": [
        {
            "webDetection": {
                "webEntities": [
                    {"entityId": "/m/1", "score": 0.4, "description": "Album"},
                    {"entityId": "/m/2", "score": 1.2, "description": "Gorillaz"},
                    {"entityId": "/m/3", "score": 0.9},
                ],
                "bestGuessLabels": [{"label": "demon days", "languageCode": "en"}],
                "pagesWithMatchingImages": [
                    {"url": "https://example.com/a", "pageTitle": "<b>Demon Days</b> - Wikipedia"},
                    {"url": "https://example.com/b"},
                ],
                "visuallySimilarImages": [{"url": "https://example.com/x.jpg"}],
            }
        }
    ]
}


def test_parse_response_orders_entities_by_score() -> None:
    assert parse_response(ANSWER, 5) == ImageMatches(
        best_guesses=("demon days",),
        entities=("Gorillaz", "Album"),
        page_titles=("Demon Days - Wikipedia",),
    )
    assert parse_response(ANSWER, 1).entities == ("Gorillaz",)


def test_parse_response_without_matches_is_empty() -> None:
    assert parse_response({"responses": [{}]}, 5).empty


@pytest.mark.parametrize(
    "data",
    [
        {"error": {"code": 403, "message": "API key not valid."}},
        {"responses": [{"error": {"message": "Bad image data."}}]},
        {"responses": []},
        {"responses": ["x"]},
        [],
    ],
)
def test_parse_response_errors(data: Any) -> None:
    with pytest.raises(ImageSearchError):
        parse_response(data, 5)


class FakeOpener:
    """Stands in for urllib's opener: records the request, returns `answer`
    or raises `error`."""

    def __init__(self, answer: Any = None, error: Exception | None = None) -> None:
        self.answer = answer
        self.error = error
        self.requests: list[urllib.request.Request] = []

    def open(self, request: urllib.request.Request, timeout: float) -> Any:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return io.BytesIO(json.dumps(self.answer).encode("utf-8"))


def test_google_vision_search_sends_the_key_in_a_header() -> None:
    opener = FakeOpener(ANSWER)
    client = GoogleVisionSearch("SECRET-KEY", 5, 15.0, opener=opener)  # type: ignore[arg-type]
    matches = asyncio.run(client.search(b"crop"))
    assert matches.best_guesses == ("demon days",)
    (request,) = opener.requests
    assert request.full_url == ENDPOINT and "SECRET-KEY" not in request.full_url
    assert request.get_header("X-goog-api-key") == "SECRET-KEY"
    assert json.loads(request.data)["requests"][0]["image"]["content"] == "Y3JvcA=="


def test_google_vision_http_error_has_the_status_and_message_only() -> None:
    body = io.BytesIO(json.dumps({"error": {"message": "API key not valid."}}).encode())
    error = urllib.error.HTTPError(ENDPOINT, 400, "Bad Request", {}, body)  # type: ignore[arg-type]
    client = GoogleVisionSearch("SECRET-KEY", 5, 15.0, opener=FakeOpener(error=error))  # type: ignore[arg-type]
    with pytest.raises(ImageSearchError, match="HTTP 400: API key not valid.") as info:
        asyncio.run(client.search(b"crop"))
    assert "SECRET-KEY" not in str(info.value)


def test_google_vision_network_error() -> None:
    client = GoogleVisionSearch("k", 5, 15.0, opener=FakeOpener(error=urllib.error.URLError("x")))  # type: ignore[arg-type]
    with pytest.raises(ImageSearchError, match="request failed: URLError"):
        asyncio.run(client.search(b"crop"))


# Budget


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.now = NOW

    def __call__(self) -> float:
        return self.t


def _budget(tmp_path: Path, clock: Clock, per_hour: int = 2, per_month: int = 5) -> SearchBudget:
    return SearchBudget(
        tmp_path / "image_search.jsonl", per_hour, per_month, clock=clock, now=lambda: clock.now
    )


def test_budget_hourly_cap_is_rolling(tmp_path: Path) -> None:
    clock = Clock()
    budget = _budget(tmp_path, clock)
    assert budget.refusal() is None
    budget.record("google_vision")
    clock.t += 60
    budget.record("google_vision")
    assert budget.refusal() == "Image search limit reached (2 per hour). Try again in 59 min."
    clock.t = 1000.0 + 3600
    assert budget.refusal() is None


def test_budget_monthly_cap_survives_a_restart(tmp_path: Path) -> None:
    clock = Clock()
    budget = _budget(tmp_path, clock, per_hour=100, per_month=3)
    for _ in range(3):
        budget.record("google_vision")
    assert budget.refusal() == "Image search limit reached (3 per month)."
    lines = (tmp_path / "image_search.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"ts": "2026-11-14T12:00:00+00:00", "provider": "google_vision"}
    restarted = _budget(tmp_path, Clock(), per_hour=100, per_month=3)
    assert restarted.month_total() == 3 and restarted.refusal() is not None


def test_budget_new_month_starts_at_zero(tmp_path: Path) -> None:
    clock = Clock()
    budget = _budget(tmp_path, clock, per_hour=100, per_month=1)
    budget.record("google_vision")
    assert budget.refusal() is not None
    clock.now = datetime(2026, 12, 1, 0, 0, tzinfo=timezone.utc)
    clock.t += 7200
    assert budget.refusal() is None and budget.month_total() == 0


def test_budget_skips_malformed_lines(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "image_search.jsonl"
    path.write_text('not json\n{"ts": "2026-11-01T00:00:00+00:00"}\n{"x": 1}\n\n', encoding="utf-8")
    assert _budget(tmp_path, Clock()).month_total() == 1
    assert "malformed line 1" in caplog.text


# Box and crop


def test_parse_box() -> None:
    fenced = '```json\n{\n  "bbox_2d": [454, 468, 552, 615],\n  "label": "bee"\n}\n```'
    assert parse_box(fenced) == (0.454, 0.468, 0.552, 0.615)
    assert parse_box("[0, 0, 1000, 1000]") == (0.0, 0.0, 1.0, 1.0)
    assert parse_box("[10.5, 20, 30, 40]") == (0.0105, 0.02, 0.03, 0.04)


@pytest.mark.parametrize(
    "text",
    [
        "I cannot see an album cover.",
        '{"bbox_2d": null}',
        "[1, 2, 3]",
        "[500, 100, 400, 200]",  # x2 before x1
        "[100, 200, 300, 200]",  # no height
        "[100, 100, 1200, 900]",  # off the 0 to 1000 grid
    ],
)
def test_parse_box_rejects(text: str) -> None:
    assert parse_box(text) is None


def test_locate_prompt_takes_the_target() -> None:
    assert QUESTION_PLACEHOLDER in LOCATE_PROMPT and "bbox_2d" in LOCATE_PROMPT


def _jpeg(width: int = 640, height: int = 480) -> bytes:
    frame = numpy.zeros((height, width, 3), numpy.uint8)
    frame[:, width // 2 :] = (255, 255, 255)
    return encode_frame(frame, max(width, height))


def _size(jpeg: bytes) -> tuple[int, int]:
    frame = cv2.imdecode(numpy.frombuffer(jpeg, numpy.uint8), cv2.IMREAD_COLOR)
    return frame.shape[1], frame.shape[0]


def test_crop_jpeg() -> None:
    assert _size(crop_jpeg(_jpeg(), (0.25, 0.5, 0.75, 1.0), 48)) == (320, 240)
    # Values past the edge are clamped.
    assert _size(crop_jpeg(_jpeg(), (0.5, 0.5, 1.5, 1.5), 48)) == (320, 240)


def test_crop_jpeg_errors() -> None:
    with pytest.raises(ValueError, match="too small"):
        crop_jpeg(_jpeg(), (0.0, 0.0, 0.05, 0.5), 48)
    with pytest.raises(ValueError, match="decoding"):
        crop_jpeg(b"not a jpeg", (0.0, 0.0, 1.0, 1.0), 48)


def test_clean_target() -> None:
    assert clean_target("  the  album\ncover ") == "the album cover"
    assert clean_target("") == DEFAULT_TARGET and clean_target(None) == DEFAULT_TARGET
    assert len(clean_target("x" * 500)) == MAX_TARGET_CHARS


# Looker.search_image


class FakeCamera:
    def __init__(self) -> None:
        self.calls = 0
        self.jpeg = _jpeg()

    async def capture_jpeg(self) -> bytes:
        self.calls += 1
        return self.jpeg


class FakeSession:
    def __init__(self, vision: "FakeVision") -> None:
        self.vision = vision

    async def locate(self, jpeg: bytes, target: str) -> Box | None:
        self.vision.located.append((jpeg, target))
        return self.vision.box

    async def describe(self, jpeg: bytes, question: str) -> str:
        self.vision.described.append((jpeg, question))
        if self.vision.describe_error:
            raise VisionError("llama-server request failed: URLError")
        return self.vision.answer


class FakeVision:
    def __init__(
        self,
        box: Box | None = (0.25, 0.25, 0.75, 0.75),
        answer: str = "An album cover with four cartoon faces.",
        describe_error: bool = False,
    ) -> None:
        self.box = box
        self.answer = answer
        self.describe_error = describe_error
        self.located: list[tuple[bytes, str]] = []
        self.described: list[tuple[bytes, str]] = []
        self.open = False
        self.exited = 0

    @asynccontextmanager
    async def session(self) -> AsyncIterator[FakeSession]:
        self.open = True
        try:
            yield FakeSession(self)
        finally:
            self.open = False
            self.exited += 1


class FakeSearchClient:
    provider = "fake"

    def __init__(self, vision: FakeVision, matches: ImageMatches | None = None, fail: bool = False) -> None:
        self.vision = vision
        self.matches = matches if matches is not None else ImageMatches(("demon days",), ("Gorillaz",))
        self.fail = fail
        self.uploads: list[bytes] = []
        self.vision_open_during_upload: bool | None = None

    async def search(self, jpeg: bytes) -> ImageMatches:
        self.uploads.append(jpeg)
        self.vision_open_during_upload = self.vision.open
        if self.fail:
            raise ImageSearchError("Vision API HTTP 500")
        return self.matches


class Rig:
    def __init__(self, tmp_path: Path, vision: FakeVision | None = None, **search: Any) -> None:
        self.camera = FakeCamera()
        self.vision = vision if vision is not None else FakeVision()
        self.client = FakeSearchClient(self.vision, **search)
        self.clock = Clock()
        self.budget = _budget(tmp_path, self.clock)
        self.looker = Looker(
            self.camera,
            self.vision,
            LookLimiter(10, 30, clock=self.clock),
            ImageSearch(self.client, self.budget, 0.8),
        )
        self.snaps: list[tuple[str, bytes]] = []

    async def snapshot(self, kind: str, jpeg: bytes) -> None:
        self.snaps.append((kind, jpeg))

    def search(self, target: Any = "the album cover") -> dict[str, Any]:
        return asyncio.run(self.looker.search_image(target, self.snapshot))


def test_search_uploads_only_the_crop(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    result = rig.search()
    assert result == {
        "output": "Online lookup of the cut-out: Best guess: demon days. Related: Gorillaz. "
        "Qwenny's own description: An album cover with four cartoon faces."
    }
    assert rig.vision.located == [(rig.camera.jpeg, "the album cover")]
    (upload,) = rig.client.uploads
    assert upload != rig.camera.jpeg and _size(upload) == (320, 240)
    # The model described the target from the whole frame, and llama-server
    # had stopped before the upload.
    assert rig.vision.described == [
        (
            rig.camera.jpeg,
            "Describe the album cover in one or two sentences. Quote any text on it.",
        )
    ]
    assert rig.client.vision_open_during_upload is False and rig.vision.exited == 1
    assert rig.snaps == [("camera", rig.camera.jpeg), ("search", upload)]
    assert rig.budget.month_total() == 1
    assert rig.looker.search_on is True


@pytest.mark.parametrize(
    ("box", "message"),
    [
        (None, MSG_NOT_FOUND),
        ((0.0, 0.0, 1.0, 0.9), MSG_TOO_LARGE),  # 90% of the frame
        ((0.5, 0.5, 0.52, 0.9), MSG_NOT_FOUND),  # 13 px wide
    ],
)
def test_no_usable_box_means_no_upload(tmp_path: Path, box: Box | None, message: str) -> None:
    rig = Rig(tmp_path, FakeVision(box=box))
    assert rig.search() == {"error": message}
    assert rig.client.uploads == [] and rig.budget.month_total() == 0
    # The frame was shown, no crop was.
    assert [kind for kind, _ in rig.snaps] == ["camera"]
    assert rig.vision.exited == 1


def test_search_budget_refusal_does_not_open_the_camera(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    rig.budget.record("fake")
    rig.budget.record("fake")
    assert rig.search() == {
        "error": "Image search limit reached (2 per hour). Try again in 60 min."
    }
    assert rig.camera.calls == 0 and rig.client.uploads == []


def test_search_client_refusal_is_asked_before_the_camera(tmp_path: Path) -> None:
    # A spent daily token cap (GeminiImageIdentify.refusal) costs no photo,
    # no crop and no search slot.
    rig = Rig(tmp_path)
    rig.client.refusal = lambda: "token cap"  # type: ignore[attr-defined]
    assert rig.search() == {"error": "token cap"}
    assert rig.camera.calls == 0 and rig.client.uploads == [] and rig.snaps == []
    assert rig.budget.month_total() == 0


def test_search_client_refusal_after_the_crop_uploads_nothing(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    answers = iter([None, "token cap"])
    rig.client.refusal = lambda: next(answers)  # type: ignore[attr-defined]
    assert rig.search() == {"error": "token cap"}
    assert rig.camera.calls == 1 and rig.client.uploads == []
    assert rig.budget.month_total() == 0
    assert [kind for kind, _ in rig.snaps] == ["camera"]


def test_search_shares_the_camera_cooldown_with_look(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert "output" in rig.search()
    assert rig.search() == {"error": "Camera cooldown: try again in 10 s."}
    assert asyncio.run(rig.looker.look("q")) == {"error": "Camera cooldown: try again in 10 s."}
    assert rig.camera.calls == 1


def test_search_failure_is_reported_and_counted(tmp_path: Path) -> None:
    rig = Rig(tmp_path, fail=True)
    assert rig.search() == {"error": MSG_SEARCH_FAILED}
    assert rig.budget.month_total() == 1


def test_search_without_matches(tmp_path: Path) -> None:
    rig = Rig(tmp_path, matches=ImageMatches())
    assert rig.search()["output"].startswith(
        "Online lookup of the cut-out: Nothing was identified. Qwenny's"
    )


def test_search_goes_ahead_without_the_local_description(tmp_path: Path) -> None:
    rig = Rig(tmp_path, FakeVision(describe_error=True))
    assert rig.search() == {
        "output": "Online lookup of the cut-out: Best guess: demon days. Related: Gorillaz."
    }


def test_vision_failure_before_the_box(tmp_path: Path) -> None:
    class Broken(FakeVision):
        @asynccontextmanager
        async def session(self) -> AsyncIterator[FakeSession]:
            raise VisionError("llama-server did not start")
            yield  # pragma: no cover

    rig = Rig(tmp_path, Broken())
    assert rig.search() == {"error": MSG_VISION_FAILED}
    assert rig.client.uploads == []


def test_search_is_off_without_the_setup() -> None:
    looker = Looker(FakeCamera(), FakeVision(), LookLimiter(10, 30))
    assert looker.search_on is False
    assert asyncio.run(looker.search_image("x")) == {"error": MSG_SEARCH_OFF}


def test_a_failing_snapshot_does_not_stop_the_search(tmp_path: Path) -> None:
    rig = Rig(tmp_path)

    async def broken(kind: str, jpeg: bytes) -> None:
        raise RuntimeError("socket closed")

    assert "output" in asyncio.run(rig.looker.search_image("x", broken))


def test_nothing_about_the_search_is_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)
    rig = Rig(tmp_path)
    rig.search("SECRET-TARGET")
    assert "search_image: crop in" in caplog.text
    for secret in ("SECRET-TARGET", "demon days", "Gorillaz", "cartoon faces"):
        assert secret not in caplog.text


def test_look_sends_the_camera_snapshot(tmp_path: Path) -> None:
    rig = Rig(tmp_path)
    assert asyncio.run(rig.looker.look("q", rig.snapshot)) == {
        "output": "An album cover with four cartoon faces."
    }
    assert rig.snaps == [("camera", rig.camera.jpeg)]


# Pipeline and app


def _search_call(call_id: str = "c1") -> ToolCall:
    return ToolCall("search_image", {"target": "the album cover"}, call_id=call_id, raw=["raw"])


def _run(llm: Any, looker: Any, tmp_path: Path) -> list[Any]:
    sent: list[Any] = []

    async def send(msg: Any) -> None:
        sent.append(msg)

    usage = UsageTracker(tmp_path / "usage.jsonl", 1000, clock=lambda: NOW)
    asyncio.run(run_text_turn("hello", [], llm, usage, send, turn_timeout_s=60.0, tools=registry(looker)))
    return sent


def test_pipeline_routes_search_image(tmp_path: Path) -> None:
    call = _search_call()
    llm = FakeToolLLM([[call, Done(5, 1, 0)], [TextDelta("Demon Days."), Done(9, 3, 0)]])
    looker = FakeLooker({"output": "Best guess: demon days."}, search=True)
    looker.snapshots = [("camera", b"frame"), ("search", b"crop")]
    sent = _run(llm, looker, tmp_path)
    assert looker.targets == ["the album cover"] and looker.questions == []
    assert llm.requests[0].tools == [LOOK_TOOL.spec, SEARCH_IMAGE_TOOL.spec]
    assert llm.requests[1].tool_results == [ToolResult(call, {"output": "Best guess: demon days."})]
    snaps = [m for m in sent if isinstance(m, dict) and m["type"] == "snapshot"]
    assert snaps == [
        {"type": "snapshot", "kind": "camera", "jpeg_b64": "ZnJhbWU="},
        {"type": "snapshot", "kind": "search", "jpeg_b64": "Y3JvcA=="},
    ]
    states = [m["state"] for m in sent if isinstance(m, dict) and m["type"] == "state"]
    assert states == ["thinking", "looking", "thinking", "speaking", "idle"]


def test_pipeline_one_camera_tool_per_turn(tmp_path: Path) -> None:
    first = _search_call("a")
    second = ToolCall("look", {"question": "q"}, call_id="b", raw=["raw"])
    llm = FakeToolLLM([[first, second, Done(5, 1, 0)], [TextDelta("Ok."), Done(1, 1, 0)]])
    looker = FakeLooker(search=True)
    _run(llm, looker, tmp_path)
    assert looker.targets == ["the album cover"] and looker.questions == []
    assert llm.requests[1].tool_results[1] == ToolResult(second, {"error": msg_limit("camera", 1)})


def test_system_prompt_does_not_offer_the_search() -> None:
    # The prompt is sent whether or not the tool exists; the guidance for the
    # search lives in the tool description, which is only sent with the tool.
    prompt = (REPO_ROOT / "prompts" / "system.md").read_text(encoding="utf-8")
    assert "search_image" not in prompt
    assert "only when the user asks" in SEARCH_IMAGE_TOOL.spec.description


def test_pipeline_search_image_is_off_when_not_declared(tmp_path: Path) -> None:
    call = _search_call()
    llm = FakeToolLLM([[call, Done(5, 1, 0)], [TextDelta("Ok."), Done(1, 1, 0)]])
    looker = FakeLooker()
    _run(llm, looker, tmp_path)
    assert looker.targets == []
    assert llm.requests[1].tool_results == [
        ToolResult(call, {"error": msg_unknown("search_image")})
    ]


def _settings(tmp_path: Path) -> Settings:
    return text_only(
        load_settings(
            REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
        )
    )


def _collect_turn(ws: Any) -> list[dict[str, Any]]:
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg == {"type": "state", "state": "idle"}:
            return msgs


@pytest.mark.parametrize("switch", [None, False, True, "yes"])
def test_ws_snapshots_only_when_the_page_asked(tmp_path: Path, switch: Any) -> None:
    settings = _settings(tmp_path)
    llm = FakeToolLLM(
        [[ToolCall("look", {"question": "q"}), Done(5, 1, 0)], [TextDelta("A mug."), Done(9, 3, 0)]]
    )
    looker = FakeLooker()
    looker.snapshots = [("camera", b"frame")]
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    app = create_app(settings, llm=llm, usage=usage, looker=looker)
    with AppClient(app).websocket_connect("/ws") as ws:
        if switch is not None:
            ws.send_json({"type": "snapshots", "on": switch})
        ws.send_json({"type": "text", "text": "look"})
        msgs = _collect_turn(ws)
    snaps = [m for m in msgs if m["type"] == "snapshot"]
    if switch is True:
        assert snaps == [{"type": "snapshot", "kind": "camera", "jpeg_b64": "ZnJhbWU="}]
    else:
        assert snaps == []
    assert not [m for m in msgs if m["type"] == "error"]


def test_ws_snapshots_can_be_switched_off_again(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    llm = FakeToolLLM(
        [[ToolCall("look", {"question": "q"}), Done(5, 1, 0)], [TextDelta("A mug."), Done(9, 3, 0)]]
    )
    looker = FakeLooker()
    looker.snapshots = [("camera", b"frame")]
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    app = create_app(settings, llm=llm, usage=usage, looker=looker)
    with AppClient(app).websocket_connect("/ws") as ws:
        ws.send_json({"type": "snapshots", "on": True})
        ws.send_json({"type": "snapshots", "on": False})
        ws.send_json({"type": "text", "text": "look"})
        msgs = _collect_turn(ws)
    assert not [m for m in msgs if m["type"] == "snapshot"]


# Config and startup


def test_image_search_is_off_by_default(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    assert settings.image_search == ImageSearchSettings()  # the example spells out every default
    assert settings.image_search.enabled is False
    assert _default_image_search(settings) is None


def _write_example(tmp_path: Path, old: str, new: str) -> Path:
    text = (REPO_ROOT / "config.example.toml").read_text(encoding="utf-8")
    assert text.count(old) == 1
    path = tmp_path / "config.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


def test_image_search_section_is_optional(tmp_path: Path) -> None:
    text = (REPO_ROOT / "config.example.toml").read_text(encoding="utf-8")
    start, end = text.index("[image_search]"), text.index("[face]")
    path = tmp_path / "config.toml"
    path.write_text(text[:start] + text[end:], encoding="utf-8")
    s = load_settings(path, env_path=tmp_path / "missing.env", environ={})
    assert s.image_search == ImageSearchSettings()


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        (
            'enabled = false\nprovider = "gemini"',
            'enabled = false\nprovider = "lens"',
            "image_search.provider",
        ),
        ("max_per_hour = 10 ", "max_per_hour = 0 ", "image_search.max_per_hour"),
        ("max_per_month = 900", "max_per_month = 0", "image_search.max_per_month"),
        ("max_crop_fraction = 0.8", "max_crop_fraction = 1.5", "image_search.max_crop_fraction"),
        ("max_results = 5", "max_results = 0", "image_search.max_results"),
        ("timeout_s = 15 ", "timeout_s = 0 ", "image_search.timeout_s"),
    ],
)
def test_image_search_config_is_validated(tmp_path: Path, old: str, new: str, match: str) -> None:
    path = _write_example(tmp_path, old, new)
    with pytest.raises(ValueError, match=match):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_image_search_needs_the_vision_model(tmp_path: Path) -> None:
    text = (REPO_ROOT / "config.example.toml").read_text(encoding="utf-8")
    text = text.replace("enabled = false", "enabled = true")
    model = 'model = "models/vision/qwen3-vl-2b/Qwen3VL-2B-Instruct-Q4_K_M.gguf"'
    assert text.count(model) == 1
    path = tmp_path / "config.toml"
    path.write_text(text.replace(model, 'model = ""'), encoding="utf-8")
    with pytest.raises(ValueError, match="image_search.enabled needs the local vision model"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_google_vision_key_comes_from_env(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=g\nGOOGLE_VISION_API_KEY=v\n", encoding="utf-8")
    s = load_settings(REPO_ROOT / "config.example.toml", env_path=env, environ={})
    assert s.secrets.google_vision_api_key == "v" and s.secrets.gemini_api_key == "g"
    assert "v" not in repr(s.secrets)
    s = load_settings(
        REPO_ROOT / "config.example.toml", env_path=env, environ={"GOOGLE_VISION_API_KEY": "w"}
    )
    assert s.secrets.google_vision_api_key == "w"


def _enabled(tmp_path: Path, provider: str, **keys: str | None) -> Settings:
    base = _settings(tmp_path)
    return dataclasses.replace(
        base,
        image_search=dataclasses.replace(base.image_search, enabled=True, provider=provider),
        secrets=Secrets(**keys),
        repo_root=tmp_path,
    )


@pytest.mark.parametrize(
    ("provider", "key_name"),
    [("gemini", "GEMINI_API_KEY"), ("google_vision", "GOOGLE_VISION_API_KEY")],
)
def test_enabled_without_a_key_stays_off_with_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, provider: str, key_name: str
) -> None:
    assert _default_image_search(_enabled(tmp_path, provider)) is None
    assert f"{key_name} is not set" in caplog.text


def test_a_key_for_the_other_provider_does_not_count(tmp_path: Path) -> None:
    assert _default_image_search(_enabled(tmp_path, "gemini", google_vision_api_key="v")) is None
    assert _default_image_search(_enabled(tmp_path, "google_vision", gemini_api_key="g")) is None


def test_enabled_with_a_key_builds_the_search_and_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    search = _default_image_search(
        _enabled(tmp_path, "google_vision", google_vision_api_key="SECRET-KEY")
    )
    assert search is not None and search.max_crop_fraction == 0.8
    assert isinstance(search.client, GoogleVisionSearch)
    assert search.budget.log_path == tmp_path / "logs" / "image_search.jsonl"
    assert "image search: ENABLED" in caplog.text and "SECRET-KEY" not in caplog.text


def test_gemini_provider_uses_the_llm_model_and_the_usage_tracker(tmp_path: Path) -> None:
    usage = UsageTracker(tmp_path / "usage.jsonl", 1000, clock=lambda: NOW)
    settings = _enabled(tmp_path, "gemini", gemini_api_key="SECRET-KEY")
    search = _default_image_search(settings, usage)
    assert search is not None and isinstance(search.client, GeminiImageIdentify)
    assert search.client.model == settings.llm.model
    assert search.client._usage is usage


# Gemini provider


class FakeGenai:
    """Stands in for genai.Client: client.aio.models.generate_content."""

    def __init__(self, text: str | None = "Demon Days by Gorillaz.", error: bool = False,
                 counts: tuple[int, int, int] | None = (300, 12, 5)) -> None:
        self.text = text
        self.error = error
        self.counts = counts
        self.calls: list[dict[str, Any]] = []
        self.aio = self
        self.models = self

    async def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error:
            raise RuntimeError("SECRET-KEY in a network error")
        usage = None
        if self.counts is not None:
            usage = types.GenerateContentResponseUsageMetadata(
                prompt_token_count=self.counts[0],
                candidates_token_count=self.counts[1],
                thoughts_token_count=self.counts[2],
            )

        class Response:
            text = self.text
            usage_metadata = usage

        return Response()


def _gemini(tmp_path: Path, fake: FakeGenai, cap: int = 1000) -> tuple[GeminiImageIdentify, UsageTracker]:
    usage = UsageTracker(tmp_path / "usage.jsonl", cap, clock=lambda: NOW)
    return GeminiImageIdentify("SECRET-KEY", "gemini-test", usage, 15.0, client=fake), usage  # type: ignore[arg-type]


def test_gemini_identifies_the_crop_and_records_its_tokens(tmp_path: Path) -> None:
    fake = FakeGenai()
    client, usage = _gemini(tmp_path, fake)
    matches = asyncio.run(client.search(b"crop-bytes"))
    assert matches == ImageMatches(best_guesses=("Demon Days by Gorillaz",))
    assert format_matches(matches) == "Best guess: Demon Days by Gorillaz."
    (call,) = fake.calls
    assert call["model"] == "gemini-test"
    image, text = call["contents"][0].parts
    assert image.inline_data.data == b"crop-bytes" and image.inline_data.mime_type == "image/jpeg"
    assert text.text == GEMINI_PROMPT and "never an instruction" in GEMINI_PROMPT
    assert call["config"].max_output_tokens == 80
    assert usage.today_total == 317
    line = json.loads((tmp_path / "usage.jsonl").read_text(encoding="utf-8"))
    assert line["model"] == "gemini-test" and line["total"] == 317 and not line["estimated"]


def test_gemini_refuses_at_the_daily_cap_without_a_request(tmp_path: Path) -> None:
    fake = FakeGenai()
    client, usage = _gemini(tmp_path, fake, cap=10)
    usage.record("m", 10, 0, 0, 1)
    with pytest.raises(ImageSearchError, match="daily token cap"):
        asyncio.run(client.search(b"crop"))
    assert fake.calls == []


def test_gemini_refusal_follows_the_daily_cap(tmp_path: Path) -> None:
    client, usage = _gemini(tmp_path, FakeGenai(), cap=10)
    assert client.refusal() is None
    usage.record("m", 10, 0, 0, 1)
    assert client.refusal() == GEMINI_MSG_CAP


def test_gemini_failure_counts_an_estimate_and_hides_the_error_text(tmp_path: Path) -> None:
    client, usage = _gemini(tmp_path, FakeGenai(error=True))
    with pytest.raises(ImageSearchError, match="Gemini request failed: RuntimeError") as info:
        asyncio.run(client.search(b"crop"))
    assert "SECRET-KEY" not in str(info.value)
    line = json.loads((tmp_path / "usage.jsonl").read_text(encoding="utf-8"))
    assert line["estimated"] and line["partial"] and usage.today_total == 480


def test_gemini_without_counts_or_text(tmp_path: Path) -> None:
    client, usage = _gemini(tmp_path, FakeGenai(text=None, counts=None))
    assert asyncio.run(client.search(b"crop")).empty
    assert usage.today_total == 480


def test_gemini_answer_is_one_capped_line() -> None:
    assert clean_answer("  Demon Days\n by Gorillaz.  ") == "Demon Days by Gorillaz"
    assert clean_answer(None) == "" and len(clean_answer("x" * 1000)) == 300
