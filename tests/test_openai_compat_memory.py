"""Tool notes from earlier turns in local requests, with tools declared and
without. httpx.MockTransport only."""

from __future__ import annotations

import json

from test_openai_compat import ASK, Script, delta, events, sse, usage_chunk

from obsilord.llm.base import Message, ToolNote
from obsilord.llm.openai_compat import HISTORY_NOTES_HEADER, NO_TOOLS_NOTE
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
    events(script, HISTORY, tools=[LOOK_TOOL.spec, RECALL_TOOL.spec])
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
    events(script, history, tools=[LOOK_TOOL.spec])
    messages = script.bodies()[0]["messages"]
    calls = messages[2]["tool_calls"]
    assert [c["id"] for c in calls] == ["h1", "h2"]
    assert json.loads(calls[1]["function"]["arguments"]) == {"arg": "x"}
    assert messages[3]["content"] == json.dumps({"error": "Camera cooldown."})
    assert messages[4] == {"role": "tool", "tool_call_id": "h2", "content": json.dumps({"output": "y"})}


def test_a_model_without_tool_support_gets_the_notes_in_the_system_message() -> None:
    script = Script(sse(delta("Blue."), usage_chunk()))
    events(script, HISTORY, tools=[LOOK_TOOL.spec], supports_tools=False)
    (body,) = script.bodies()
    assert "tools" not in body
    system = body["messages"][0]["content"]
    assert system.startswith("SYS" + NO_TOOLS_NOTE + HISTORY_NOTES_HEADER)
    assert 'look("What is it?") returned: A blue ceramic owl.' in system
    assert all(m["role"] in ("system", "user", "assistant") for m in body["messages"])
    assert body["messages"][2] == {"role": "assistant", "content": "An owl."}


def test_no_notes_no_block() -> None:
    script = Script(sse(delta("Hi."), usage_chunk()))
    events(script, ASK, tools=[])
    assert script.bodies()[0]["messages"][0]["content"] == "SYS" + NO_TOOLS_NOTE


def test_notes_block_without_tools_offered() -> None:
    script = Script(sse(delta("Blue."), usage_chunk()))
    events(script, HISTORY, tools=[])
    assert HISTORY_NOTES_HEADER in script.bodies()[0]["messages"][0]["content"]
