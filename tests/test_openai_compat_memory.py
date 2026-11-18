"""Tool notes from earlier turns in local requests, and router wording that
fits tools besides the camera. httpx.MockTransport only."""

from __future__ import annotations

import json

from test_openai_compat import ASK, Script, delta, events, route, sse, usage_chunk

from obsilord.llm.base import Message, ToolNote, ToolSpec
from obsilord.llm.openai_compat import (
    HISTORY_NOTES_HEADER,
    LOCAL_MODE_NOTE,
    ROUTER_INTRO,
    router_instructions,
    router_transcript,
)
from obsilord.look import LOOK_TOOL
from obsilord.memory.recall import RECALL_TOOL

OWL = ToolNote("look", "What is it?", "A blue ceramic owl.")
HISTORY = [
    Message("user", "What is this?"),
    Message("assistant", "An owl.", tools=(OWL,)),
    Message("user", "What color was it?"),
]


def test_native_replays_noted_turns_as_tool_messages() -> None:
    script = Script(sse(delta("Blue."), usage_chunk()))
    events(script, HISTORY, tools=[LOOK_TOOL, RECALL_TOOL], tool_mode="native")
    messages = script.bodies()[0]["messages"]
    assert messages[1:] == [
        {"role": "user", "content": "What is this?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "h1",
                    "type": "function",
                    "function": {"name": "look", "arguments": json.dumps({"question": "What is it?"})},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "h1", "content": json.dumps({"output": "A blue ceramic owl."})},
        {"role": "assistant", "content": "An owl."},
        {"role": "user", "content": "What color was it?"},
    ]


def test_native_replay_of_an_error_and_an_unknown_tool() -> None:
    history = [
        Message("user", "a"),
        Message(
            "assistant",
            "b",
            tools=(ToolNote("look", "q", "Camera cooldown.", ok=False), ToolNote("gone", "x", "y")),
        ),
        Message("user", "c"),
    ]
    script = Script(sse(delta("ok"), usage_chunk()))
    events(script, history, tools=[LOOK_TOOL], tool_mode="native")
    messages = script.bodies()[0]["messages"]
    calls = messages[2]["tool_calls"]
    assert [c["id"] for c in calls] == ["h1", "h2"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"arg": "x"}
    assert messages[3]["content"] == json.dumps({"error": "Camera cooldown."})
    assert messages[4] == {"role": "tool", "tool_call_id": "h2", "content": json.dumps({"output": "y"})}


def test_router_mode_puts_notes_in_the_system_message() -> None:
    script = Script(route("none"), sse(delta("Blue."), usage_chunk()))
    events(script, HISTORY, tools=[LOOK_TOOL], tool_mode="router")
    router_body, reply_body = script.bodies()
    assert "look returned: A blue ceramic owl." in router_body["messages"][1]["content"]
    system = reply_body["messages"][0]["content"]
    assert system.startswith("SYS" + LOCAL_MODE_NOTE + HISTORY_NOTES_HEADER)
    assert 'look("What is it?") returned: A blue ceramic owl.' in system
    assert all(m["role"] in ("system", "user", "assistant") for m in reply_body["messages"])
    assert reply_body["messages"][2] == {"role": "assistant", "content": "An owl."}


def test_no_notes_no_block() -> None:
    script = Script(sse(delta("Hi."), usage_chunk()))
    events(script, ASK, tools=[])
    assert script.bodies()[0]["messages"][0]["content"] == "SYS" + LOCAL_MODE_NOTE


def test_notes_block_without_tools_offered() -> None:
    script = Script(sse(delta("Blue."), usage_chunk()))
    events(script, HISTORY, tools=[])
    assert HISTORY_NOTES_HEADER in script.bodies()[0]["messages"][0]["content"]


def test_router_transcript_shows_earlier_tool_results() -> None:
    text = router_transcript(HISTORY, 4)
    assert "Assistant: An owl. (look returned: A blue ceramic owl.)" in text


def test_router_prompt_is_not_camera_only() -> None:
    assert "camera" not in ROUTER_INTRO
    text = router_instructions([LOOK_TOOL, RECALL_TOOL])
    assert "- recall: Search earlier conversations" in text
    assert "arg: A few keywords" in text


def test_router_picks_recall_with_a_query() -> None:
    script = Script(route("recall", "dentist"))
    got = events(script, [Message("user", "When was my dentist?")], tools=[LOOK_TOOL, RECALL_TOOL])
    calls = [e for e in got if getattr(e, "name", None) == "recall"]
    assert calls and calls[0].args == {"query": "dentist"}


def test_spec_without_parameters_still_renders() -> None:
    bare = ToolSpec("ping", "Ping.")
    assert "- ping: Ping." in router_instructions([bare])
