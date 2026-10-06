"""Live Gemini call. Needs network and a real key in .env. Run with:

    .venv/Scripts/python.exe -m pytest -m hardware tests/test_gemini_live.py -q

Usage is written to a temp log, not logs/usage.jsonl. Prints counts and
latency only, never the key.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest

from pixelbuddy.config import REPO_ROOT, load_settings
from pixelbuddy.llm.base import Done, Message, TextDelta
from pixelbuddy.llm.gemini import GeminiClient
from pixelbuddy.pipeline import MSG_CAP, run_text_turn
from pixelbuddy.usage import UsageTracker

pytestmark = pytest.mark.hardware


def _client(config_path: Path | None = None) -> GeminiClient:
    settings = load_settings(config_path)
    if settings.secrets.gemini_api_key is None:
        pytest.skip("no API key configured")
    return GeminiClient(settings.llm, settings.secrets.gemini_api_key)


def test_say_hi_streams_text_and_tokens() -> None:
    client = _client()

    async def go() -> tuple[str, Done | None, float, float]:
        parts: list[str] = []
        done: Done | None = None
        t0 = time.monotonic()
        ttft = -1.0
        try:
            async for ev in client.stream_reply([Message("user", "Say hi in three words.")], []):
                if isinstance(ev, TextDelta):
                    if ttft < 0:
                        ttft = time.monotonic() - t0
                    parts.append(ev.text)
                elif isinstance(ev, Done):
                    done = ev
        finally:
            await client.aclose()
        return "".join(parts), done, ttft, time.monotonic() - t0

    text, done, ttft, total = asyncio.run(go())
    assert done is not None
    print(
        f"\nchars={len(text)} tokens_in={done.tokens_in} tokens_out={done.tokens_out} "
        f"tokens_thinking={done.tokens_thinking} ttft_ms={ttft * 1000:.0f} total_ms={total * 1000:.0f}"
    )
    assert text.strip()
    assert done.tokens_in > 0 and done.tokens_out > 0


def test_cap_blocks_second_live_call(tmp_path: Path) -> None:
    """With a cap of 1, one real turn goes through and the next is blocked locally."""
    base = (REPO_ROOT / "config.example.toml").read_text(encoding="utf-8")
    assert "daily_token_cap = 200000" in base
    cfg = tmp_path / "config.toml"
    cfg.write_text(base.replace("daily_token_cap = 200000", "daily_token_cap = 1"), encoding="utf-8")
    client = _client(cfg)
    settings = load_settings(cfg)
    assert settings.llm.daily_token_cap == 1
    usage = UsageTracker(tmp_path / "usage.jsonl", settings.llm.daily_token_cap)
    calls = 0
    real_stream = client.stream_reply

    def counting_stream(history: list[Message], tools: list[Any]) -> Any:
        nonlocal calls
        calls += 1
        return real_stream(history, tools)

    client.stream_reply = counting_stream  # type: ignore[method-assign]

    async def go() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        first: list[dict[str, Any]] = []
        second: list[dict[str, Any]] = []

        async def send1(m: dict[str, Any]) -> None:
            first.append(m)

        async def send2(m: dict[str, Any]) -> None:
            second.append(m)

        try:
            history: list[Message] = []
            await run_text_turn("Say hi in three words.", history, client, usage, send1)
            await run_text_turn("Say bye in three words.", history, client, usage, send2)
        finally:
            await client.aclose()
        return first, second

    first, second = asyncio.run(go())
    print(f"\nfirst turn total tokens={usage.today_total} api_calls={calls}")
    assert first[-2]["final"] is True and first[-2]["text"].strip()
    assert usage.today_total >= 1
    assert calls == 1
    assert second == [
        {"type": "state", "state": "thinking"},
        {"type": "error", "message": MSG_CAP},
        {"type": "state", "state": "idle"},
    ]
