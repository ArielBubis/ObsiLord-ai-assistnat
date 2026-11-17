"""Switching the chat LLM from the page: GET /config/llm, the "llm" socket
message, notices, local turns (router, cap), and the /ws Origin check. The
local server is OpenAICompatServer on httpx.MockTransport."""

from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from fakes import FakeLLM, FakeLooker, text_only
from obsilord.app import (
    MSG_BAD_MODEL,
    MSG_BAD_PROVIDER,
    MSG_GEMINI_OFF,
    MSG_NO_KEY,
    MSG_SWITCH_BUSY,
    create_app,
)
from obsilord.config import REPO_ROOT, Settings, load_settings
from obsilord.llm.openai_compat import OpenAICompatServer
from obsilord.llm_switch import clean_model, switch_notice
from obsilord.pipeline import MSG_CAP
from obsilord.usage import UsageTracker
from test_openai_compat import LMSTUDIO_MODELS, Script, delta, route, sse, usage_chunk

ORIGIN = "http://127.0.0.1:8765"


def _settings(tmp_path: Path, **llm: Any) -> Settings:
    s = text_only(
        load_settings(
            REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
        )
    )
    return dataclasses.replace(s, llm=dataclasses.replace(s.llm, **llm)) if llm else s


class Rig:
    def __init__(
        self,
        tmp_path: Path,
        script: Script,
        gemini: FakeLLM | None = None,
        looker: Any = None,
        cap: int | None = None,
        **llm: Any,
    ) -> None:
        self.settings = _settings(tmp_path, **llm)
        self.script = script
        self.usage = UsageTracker(
            tmp_path / "usage.jsonl", cap if cap is not None else self.settings.llm.daily_token_cap
        )
        self.local = OpenAICompatServer(
            self.settings.llm_local, transport=httpx.MockTransport(script)
        )
        self.client = TestClient(
            create_app(self.settings, llm=gemini, usage=self.usage, looker=looker, local=self.local)
        )

    def usage_lines(self) -> list[dict[str, Any]]:
        path = self.usage.log_path
        if not path.is_file():
            return []
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()]


def _turn(ws: Any) -> list[dict[str, Any]]:
    msgs = []
    while True:
        msg = ws.receive_json()
        msgs.append(msg)
        if msg == {"type": "state", "state": "idle"}:
            return msgs


def _final_reply(msgs: list[dict[str, Any]]) -> str | None:
    finals = [m for m in msgs if m.get("type") == "transcript" and m.get("final")]
    return finals[-1]["text"] if finals else None


def _switch_local(ws: Any, model: str = "qwen3-14b") -> list[dict[str, Any]]:
    """Switch to a local model and read the notice and the warm-up statuses."""
    ws.send_json({"type": "llm", "provider": "local", "model": model})
    return [ws.receive_json() for _ in range(3)]


WARM = route("none", usage={"prompt_tokens": 9, "completion_tokens": 1})


# --- GET /config/llm ------------------------------------------------------------


def test_config_lists_both_providers_and_the_local_models(tmp_path: Path) -> None:
    rig = Rig(tmp_path, Script(httpx.Response(200, json=LMSTUDIO_MODELS)), gemini=FakeLLM([]))
    data = rig.client.get("/config/llm").json()
    assert data["default_provider"] == "gemini"
    assert data["gemini"] == {"available": True, "model": rig.settings.llm.model}
    local = data["local"]
    assert local["base_url"] == "http://127.0.0.1:1234/v1" and local["reachable"] is True
    assert local["error"] is None and local["default_model"] == ""
    assert [m["id"] for m in local["models"]] == ["qwen3-14b", "gemma-12b", "qwen3-vl-8b"]
    assert local["models"][0]["loaded"] is True
    assert "key" not in json.dumps(data).lower().replace("api_key", "")


def test_config_when_the_local_server_is_down(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    rig = Rig(tmp_path, Script(refuse, refuse))
    local = rig.client.get("/config/llm").json()["local"]
    assert local["reachable"] is False and local["models"] == []
    assert "cannot reach the local LLM server" in local["error"]


# --- switching ------------------------------------------------------------------


def test_switch_to_local_sends_a_notice_and_loads_the_model(tmp_path: Path) -> None:
    script = Script(WARM, sse(delta("Hello "), delta("from the GPU."), usage_chunk(40, 6)))
    rig = Rig(tmp_path, script, gemini=FakeLLM(["Gemini."]))
    with rig.client.websocket_connect("/ws") as ws:
        notice, loading, ready = _switch_local(ws)
        assert notice["type"] == "notice" and "qwen3-14b" in notice["title"]
        assert any("does not count toward the daily cap" in line for line in notice["lines"])
        assert loading == {"type": "llm_status", "state": "loading", "model": "qwen3-14b"}
        assert ready["state"] == "ready" and ready["model"] == "qwen3-14b"
        ws.send_json({"type": "text", "text": "hi"})
        assert _final_reply(_turn(ws)) == "Hello from the GPU."
    warm, reply = script.bodies()
    assert warm["max_tokens"] == 1 and reply["model"] == "qwen3-14b"
    lines = rig.usage_lines()
    assert [(x["model"], x["counted"]) for x in lines] == [
        ("local:qwen3-14b", False),
        ("local:qwen3-14b", False),
    ]
    assert lines[1]["total"] == 46 and rig.usage.today_total == 0


def test_history_is_kept_across_a_switch(tmp_path: Path) -> None:
    script = Script(WARM, sse(delta("Second."), usage_chunk()))
    rig = Rig(tmp_path, script, gemini=FakeLLM(["First."]))
    with rig.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "one"})
        _turn(ws)
        _switch_local(ws)
        ws.send_json({"type": "text", "text": "two"})
        _turn(ws)
    reply = script.bodies()[-1]
    assert [m["content"] for m in reply["messages"][1:]] == ["one", "First.", "two"]


def test_switch_back_to_gemini(tmp_path: Path) -> None:
    gemini = FakeLLM(["Back on Gemini."])
    rig = Rig(tmp_path, Script(WARM), gemini=gemini)
    with rig.client.websocket_connect("/ws") as ws:
        _switch_local(ws)
        ws.send_json({"type": "llm", "provider": "gemini"})
        notice = ws.receive_json()
        assert notice["type"] == "notice" and "Gemini" in notice["title"]
        assert any("sent to Google" in line for line in notice["lines"])
        ws.send_json({"type": "text", "text": "hi"})
        assert _final_reply(_turn(ws)) == "Back on Gemini."
    assert len(gemini.calls) == 1


def test_announce_applies_quietly(tmp_path: Path) -> None:
    script = Script(sse(delta("Quiet."), usage_chunk()))  # no warm-up request
    rig = Rig(tmp_path, script, gemini=FakeLLM([]))
    with rig.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "llm", "provider": "local", "model": "gemma-12b", "announce": True})
        ws.send_json({"type": "text", "text": "hi"})
        msgs = _turn(ws)
    assert msgs[0] == {"type": "state", "state": "thinking"}
    assert _final_reply(msgs) == "Quiet."
    assert script.bodies()[0]["model"] == "gemma-12b"


MODELS = httpx.Response(200, json=LMSTUDIO_MODELS)


def test_default_provider_local_from_config(tmp_path: Path) -> None:
    # [llm_local] model is empty: the first listed (loaded) model is used.
    script = Script(MODELS, sse(delta("Local by default."), usage_chunk()))
    rig = Rig(tmp_path, script, provider="local")
    with rig.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "hi"})
        assert _final_reply(_turn(ws)) == "Local by default."
    assert script.bodies()[-1]["model"] == "qwen3-14b"
    assert rig.usage_lines()[0]["model"] == "local:qwen3-14b"


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ({"type": "llm", "provider": "openai"}, MSG_BAD_PROVIDER),
        ({"type": "llm"}, MSG_BAD_PROVIDER),
        ({"type": "llm", "provider": "local", "model": "x" * 300}, MSG_BAD_MODEL),
        ({"type": "llm", "provider": "local", "model": "a\nb"}, MSG_BAD_MODEL),
        ({"type": "llm", "provider": "local", "model": 5}, MSG_BAD_MODEL),
    ],
)
def test_bad_switch_messages(tmp_path: Path, payload: dict[str, Any], error: str) -> None:
    rig = Rig(tmp_path, Script(), gemini=FakeLLM([]))
    with rig.client.websocket_connect("/ws") as ws:
        ws.send_json(payload)
        assert ws.receive_json() == {"type": "error", "message": error}


def test_no_switch_during_a_turn(tmp_path: Path) -> None:
    gate = threading.Event()
    rig = Rig(tmp_path, Script(), gemini=FakeLLM(["a ", "b."], gate=gate))
    with rig.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "hi"})
        assert ws.receive_json() == {"type": "state", "state": "thinking"}
        # A reconnect announce during a turn is dropped quietly; a real switch
        # is refused with an error.
        ws.send_json({"type": "llm", "provider": "local", "model": "q", "announce": True})
        ws.send_json({"type": "llm", "provider": "local", "model": "qwen3-14b"})
        while (msg := ws.receive_json()).get("type") != "error":
            pass
        assert msg == {"type": "error", "message": MSG_SWITCH_BUSY}
        gate.set()
        rest = _turn(ws)
    assert not any(m.get("type") == "error" for m in rest)
    assert rig.script.requests == []


def test_without_a_gemini_key_local_still_works(tmp_path: Path) -> None:
    rig = Rig(tmp_path, Script(WARM, sse(delta("Local."), usage_chunk())))
    with rig.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "hi"})
        assert ws.receive_json() == {"type": "error", "message": MSG_NO_KEY}
        ws.send_json({"type": "llm", "provider": "gemini"})
        assert ws.receive_json() == {"type": "error", "message": MSG_GEMINI_OFF}
        ws.send_json({"type": "llm", "provider": "gemini", "announce": True})  # ignored quietly
        _switch_local(ws)
        ws.send_json({"type": "text", "text": "hi"})
        assert _final_reply(_turn(ws)) == "Local."


def test_failed_warm_up_is_reported(tmp_path: Path) -> None:
    script = Script(httpx.Response(404, json={"error": "Model nope not found"}))
    rig = Rig(tmp_path, script, gemini=FakeLLM([]))
    with rig.client.websocket_connect("/ws") as ws:
        notice, loading, failed = _switch_local(ws, "nope")
    assert failed == {
        "type": "llm_status",
        "state": "failed",
        "model": "nope",
        "message": "local LLM server answered HTTP 404: Model nope not found",
    }


def test_unreachable_server_gives_a_readable_turn_error(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    rig = Rig(tmp_path, Script(refuse), provider="local")
    with rig.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "llm", "provider": "local", "model": "m", "announce": True})
        ws.send_json({"type": "text", "text": "hi"})
        errors = [m for m in _turn(ws) if m.get("type") == "error"]
    assert errors == [
        {
            "type": "error",
            "message": "LLM request failed. Cannot reach the local LLM server at "
            "http://127.0.0.1:1234/v1.",
        }
    ]


# --- cap ------------------------------------------------------------------------


def test_local_turns_run_when_the_daily_cap_is_reached(tmp_path: Path) -> None:
    script = Script(sse(delta("Still here."), usage_chunk()))
    rig = Rig(tmp_path, script, gemini=FakeLLM(["no"]), cap=10)
    rig.usage.record("gemini-x", 10, 0, 0, 1)
    with rig.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "hi"})
        assert {"type": "error", "message": MSG_CAP} in _turn(ws)
        ws.send_json({"type": "llm", "provider": "local", "model": "m", "announce": True})
        ws.send_json({"type": "text", "text": "hi"})
        assert _final_reply(_turn(ws)) == "Still here."
    assert rig.usage.today_total == 10


# --- router through the pipeline ------------------------------------------------


def test_router_look_runs_the_camera_and_the_reply_uses_the_result(tmp_path: Path) -> None:
    script = Script(
        MODELS,
        route("look", "what is the user holding"),
        sse(delta("A red mug, how dull."), usage_chunk()),
    )
    looker = FakeLooker({"output": "A red mug."})
    rig = Rig(tmp_path, script, looker=looker, provider="local")
    with rig.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "What am I holding?"})
        msgs = _turn(ws)
    assert {"type": "state", "state": "looking"} in msgs
    assert _final_reply(msgs) == "A red mug, how dull."
    assert looker.questions == ["what is the user holding"]
    router, reply = script.bodies()  # the model list GET has no body
    assert "look" in json.dumps(router["response_format"])
    assert reply["messages"][0]["content"].endswith(
        'look("what is the user holding") returned: A red mug.\n```'
    )
    assert [x["counted"] for x in rig.usage_lines()] == [False, False]


def test_router_none_does_not_open_the_camera(tmp_path: Path) -> None:
    script = Script(MODELS, route("none"), sse(delta("Hi."), usage_chunk()))
    looker = FakeLooker()
    rig = Rig(tmp_path, script, looker=looker, provider="local")
    with rig.client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "text", "text": "tell me a joke"})
        msgs = _turn(ws)
    assert {"type": "state", "state": "looking"} not in msgs
    assert looker.questions == []


# --- origin ---------------------------------------------------------------------


def test_ws_refuses_a_foreign_origin(tmp_path: Path) -> None:
    rig = Rig(tmp_path, Script(), gemini=FakeLLM([]))
    with pytest.raises(WebSocketDisconnect):
        with rig.client.websocket_connect("/ws", headers={"origin": "http://evil.example"}):
            pass
    with pytest.raises(WebSocketDisconnect):
        with rig.client.websocket_connect("/ws", headers={"origin": "http://127.0.0.1:9999"}):
            pass


def test_foreign_host_is_refused(tmp_path: Path) -> None:
    # A DNS rebinding page reaches 127.0.0.1 under its own host name.
    rig = Rig(tmp_path, Script(), gemini=FakeLLM([]))
    assert rig.client.get("/config/llm", headers={"host": "evil.example:8765"}).status_code == 400
    assert rig.client.get("/", headers={"host": "evil.example"}).status_code == 400
    assert rig.script.requests == []  # the model list was never fetched


@pytest.mark.parametrize("host", ["127.0.0.1:8765", "localhost:8765"])
def test_own_host_is_served(tmp_path: Path, host: str) -> None:
    rig = Rig(tmp_path, Script(), gemini=FakeLLM([]))
    assert rig.client.get("/", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("origin", [ORIGIN, "http://localhost:8765"])
def test_ws_accepts_the_page_origin(tmp_path: Path, origin: str) -> None:
    rig = Rig(tmp_path, Script(), gemini=FakeLLM(["Ok."]))
    with rig.client.websocket_connect("/ws", headers={"origin": origin}) as ws:
        ws.send_json({"type": "text", "text": "hi"})
        assert _final_reply(_turn(ws)) == "Ok."


# --- notice text ----------------------------------------------------------------


def _notice(tmp_path: Path, provider: str, **kw: Any) -> dict[str, Any]:
    args = {"model": "qwen3-14b", "vision_on": True, "search_on": False, "today_tokens": 1234}
    args.update(kw)
    return switch_notice(provider, _settings(tmp_path), **args)


def test_local_notice_lists_what_is_lost_and_what_works_worse(tmp_path: Path) -> None:
    text = "\n".join(_notice(tmp_path, "local")["lines"])
    assert "routing step" in text and "look()" in text and "search_image" not in text
    assert "GPU is shared with Qwenny" in text
    assert "weaker general knowledge" in text
    assert "does not count toward the daily cap" in text


def test_local_notice_without_vision_skips_the_camera_lines(tmp_path: Path) -> None:
    text = "\n".join(_notice(tmp_path, "local", vision_on=False)["lines"])
    assert "routing step" not in text and "Qwenny" not in text


def test_local_notice_warns_that_search_still_uploads(tmp_path: Path) -> None:
    text = "\n".join(_notice(tmp_path, "local", search_on=True)["lines"])
    assert "look() or search_image()" in text
    assert "search_image() still sends the cut-out to Google Gemini" in text


def test_local_notice_without_a_model_name(tmp_path: Path) -> None:
    notice = _notice(tmp_path, "local", model="")
    assert "the first model the server lists" in notice["title"]


def test_gemini_notice_has_the_cap(tmp_path: Path) -> None:
    notice = _notice(tmp_path, "gemini")
    text = "\n".join(notice["lines"])
    assert "(1234 used today)" in text and "sent to Google" in text
    assert notice["title"].startswith("Chat runs on Gemini")


def test_clean_model() -> None:
    assert clean_model("  qwen/qwen3-14b ") == "qwen/qwen3-14b"
    assert clean_model("") == ""
    assert clean_model(None) is None and clean_model("x" * 201) is None
    assert clean_model("a\x00b") is None
