"""What the page is told when it switches the chat LLM (Settings panel).

Pure functions: the app passes in what is switched on, and gets back the
{"type": "notice"} message listing what changes, what is lost, and what may
work worse. No provider code here.
"""

from __future__ import annotations

from typing import Any

from obsilord.config import Settings

GEMINI = "gemini"
LOCAL = "local"
MAX_MODEL_CHARS = 200


def switch_notice(
    provider: str,
    settings: Settings,
    *,
    model: str,
    vision_on: bool,
    search_on: bool,
    today_tokens: int,
    tool_mode: str,
) -> dict[str, Any]:
    """The notice for a switch to provider ("gemini" or "local").

    model: the local model id ("" for the server's first model); ignored for
    Gemini. vision_on / search_on: whether look() and search_image() are
    declared. today_tokens: tokens counted toward the daily cap so far.
    tool_mode: "native" or "router", how the local model gets its tools.
    """
    if provider == GEMINI:
        return {
            "type": "notice",
            "title": f"Chat runs on Gemini ({settings.llm.model})",
            "lines": [
                "Your messages, the replies and Qwenny's text answers are sent to Google.",
                f"Tokens count toward the daily cap of {settings.llm.daily_token_cap} "
                f"({today_tokens} used today).",
                "On the free tier Google may use what is sent to improve its products.",
            ],
        }
    name = model or "the first model the server lists"
    lines = [
        f"Chat runs on {name} at {settings.llm_local.base_url}. "
        "The conversation stays on this PC.",
        "Answers depend on the model: expect weaker general knowledge, no recent "
        "facts, and a less steady personality than Gemini.",
    ]
    if vision_on:
        tools = "look() or search_image()" if search_on else "look()"
        if tool_mode == "native":
            lines.append(
                f"Camera: the model decides itself when to use {tools}, like Gemini, "
                "but a small model can miss a request or look when not asked. If it "
                'misses, ask plainly, for example "look at me".'
            )
        else:
            lines.append(
                f"Camera: this model has no tool support, so a short routing step "
                f"decides before each reply whether to use {tools}. It can miss a "
                "request or look when not asked. If it misses, ask plainly, for "
                'example "look at me".'
            )
        lines.append(
            "The GPU is shared with Qwenny, which needs about 2 to 3 GB for each "
            "look. With a large chat model loaded, look() can fail or fall back to "
            "the slower CPU build."
        )
    if search_on:
        where = (
            "Google Gemini (its tokens count toward the daily cap)"
            if settings.image_search.provider == GEMINI
            else "Google Cloud Vision"
        )
        lines.append(f"search_image() still sends the cut-out to {where}.")
    lines.append(
        "The first reply after picking a model can take a while: the server loads "
        "it into memory first."
    )
    lines.append("No token cost: local usage is logged but does not count toward the daily cap.")
    return {"type": "notice", "title": f"Chat runs on a local model ({name})", "lines": lines}


def clean_model(value: Any) -> str | None:
    """A model id from the page: stripped, at most MAX_MODEL_CHARS, or None
    when it is not a usable string."""
    if not isinstance(value, str):
        return None
    model = value.strip()
    if len(model) > MAX_MODEL_CHARS or any(ord(c) < 32 for c in model):
        return None
    return model
