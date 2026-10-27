"""One conversation turn: typed text or recorded speech in, streamed text and
speech out.

Message contract with the browser (see planning/ARCHITECTURE.md, WebSocket protocol):
  state thinking -> [cap check] ->
  transcript (assistant, cumulative, final false) per text delta ->
  transcript (assistant, full text, final true) when the LLM finishes ->
  state idle.
Errors send {"type": "error"} and then state idle.

Voice turns (run_voice_turn) send state thinking once for the whole turn, then
check the cap before running STT (no STT when the cap is reached), then
transcribe. STT failure sends "Speech recognition failed." and an empty
transcript sends "Didn't catch that.", each followed by idle. Otherwise
{"type": "transcript", "role": "user", "text": ..., "final": true} is sent and
the text turn body follows (cap check again, LLM, speech) without a second
thinking. One turn timeout covers STT and the rest. Transcript text is never
logged.

Latency line (INFO, one per successful turn): stt_ms is the STT duration and
audio_ms the utterance length (both n/a for text turns). llm_ttft_ms,
llm_done_ms, tts_first_chunk_ms, and total_ms are measured from turn start
(text received, or audio_end received), so for voice turns they include
stt_ms.

Speech (when a TTSClient is given): every complete sentence from the
SentenceSplitter, plus the flush remainder, is queued to one TTS worker task
that runs concurrently with the LLM stream, in order. Per spoken sentence:
  {"type": "tts_start", "sentence_id": n, "sample_rate": rate} (n from 1 per
  turn), binary PCM16 frames, {"type": "tts_end", "sentence_id": n}.
Sentences with nothing speakable are skipped without any message. State
speaking is sent right before the first binary frame of the turn. Idle is sent
after the LLM is done and the last tts_end has been sent (the browser holds
idle until its audio queue drains). A TTS failure for one sentence is logged,
closes that sentence with tts_end if tts_start was sent, and sends at most one
{"type": "error", "message": "Voice output failed."} per turn; text continues.
On timeout, LLM error, or cancellation the worker is cancelled and awaited
before error and idle are sent, so no binary frame follows the idle.

Without a TTSClient the Phase 2 sequence is kept: state speaking is sent on the
first text delta.

look() (when a LookTool is given, the tool is declared): if the first request
ends with a ToolCall, the speech queued so far is sent first, then state
looking, then the look (only the first look call runs; others get
MSG_ONE_LOOK, unknown tools an error), then state thinking. A second cap check
follows, then one follow-up request with the tool results and
allow_tool_calls=False. Speaking is sent again on the follow-up's first audio
frame (or text delta without TTS). At most two requests per turn. If the cap
is reached before the follow-up, the turn ends with the cap error.

Usage accounting: once a request has been started, usage is always recorded,
in a finally block, so failed, timed-out, and cancelled streams count toward
the cap. Each request is recorded on its own:
- Done arrived: its counts.
- Otherwise the last UsageUpdate seen, flagged partial.
- Otherwise a conservative estimate, flagged estimated: (history characters +
  the client's prompt_overhead_chars, i.e. the system prompt, + the tool
  results) / 4 for input and max_output_tokens for output.

History is updated once the LLM has finished with a non-empty reply (before
speech ends), so failed LLM turns leave it unchanged. It holds the user text
and the reply text only, never tool calls or results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Awaitable, Callable

from obsilord.llm.base import (
    Done,
    LLMClient,
    Message,
    TextDelta,
    ToolCall,
    ToolResult,
    UsageUpdate,
)
from obsilord.look import LOOK_TOOL, MSG_ONE_LOOK, LookTool
from obsilord.sentences import SentenceSplitter
from obsilord.stt.base import SAMPLE_RATE, STTClient, pcm16_duration_ms
from obsilord.tts.base import TTSClient
from obsilord.tts.text import clean_for_speech
from obsilord.usage import CapExceeded, UsageTracker

log = logging.getLogger(__name__)

# Sends one message: a dict goes out as a JSON text frame, bytes as a binary frame.
Send = Callable[[dict[str, Any] | bytes], Awaitable[None]]

MSG_CAP = "Daily token cap reached. Try again tomorrow."
MSG_LLM_FAILED = "LLM request failed."
MSG_TIMEOUT = "Reply timed out."
MSG_UNEXPECTED = "Something went wrong."
MSG_TTS_FAILED = "Voice output failed."
MSG_STT_FAILED = "Speech recognition failed."
MSG_NOT_CAUGHT = "Didn't catch that."


def trim_history(history: list[Message], history_turns: int) -> None:
    """Keep the last history_turns turns (one turn = user + assistant) in place."""
    keep = max(0, history_turns) * 2
    if len(history) > keep:
        del history[: len(history) - keep]


def estimate_tokens(messages: list[Message], overhead_chars: int = 0) -> int:
    """Rough input token estimate: characters / 4, rounded up, at least 1."""
    chars = sum(len(m.text) for m in messages) + max(0, overhead_chars)
    return max(1, math.ceil(chars / 4))


async def _close_stream(stream: AsyncIterator[Any]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception as exc:  # noqa: BLE001 - cleanup must not mask the result
            log.debug("closing stream failed: %s", type(exc).__name__)


async def _send_quietly(send: Send, msg: dict[str, Any]) -> None:
    """Best-effort send for error paths (the socket may already be gone)."""
    try:
        await send(msg)
    except Exception as exc:  # noqa: BLE001
        log.debug("send failed: %s", type(exc).__name__)


@dataclass
class _RequestUsage:
    """Counts seen during one LLM request."""

    done: Done | None = None
    last_usage: UsageUpdate | None = None


@dataclass
class _TurnState:
    parts: list[str] = field(default_factory=list)
    ttft_ms: int | None = None
    done_ms: int | None = None  # turn start to the end of a successful LLM stream
    requests: int = 0
    # Sums over the requests whose Done arrived, for the turn usage line.
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_thinking: int = 0


def _record_usage(
    usage: UsageTracker,
    llm: LLMClient,
    ru: _RequestUsage,
    messages: list[Message],
    max_output_tokens: int,
    total_ms: int,
    extra_chars: int = 0,
) -> None:
    """Record one request. extra_chars: request input besides the history and
    the system prompt (tool results), for the estimate."""
    model = llm.model
    try:
        if ru.done is not None:
            d = ru.done
            usage.record(model, d.tokens_in, d.tokens_out, d.tokens_thinking, total_ms)
        elif ru.last_usage is not None:
            u = ru.last_usage
            usage.record(
                model, u.tokens_in, u.tokens_out, u.tokens_thinking, total_ms, partial=True
            )
            log.warning("usage: stream incomplete, recorded last reported counts")
        else:
            usage.record(
                model,
                estimate_tokens(
                    messages, getattr(llm, "prompt_overhead_chars", 0) + max(0, extra_chars)
                ),
                max_output_tokens,
                0,
                total_ms,
                partial=True,
                estimated=True,
            )
            log.warning("usage: no counts reported, recorded a conservative estimate")
    except Exception as exc:  # noqa: BLE001 - never let accounting break the turn
        log.error("usage: recording failed: %s", type(exc).__name__)




@dataclass
class _SpeechState:
    sentence_id: int = 0  # last sentence id sent in tts_start (ids start at 1)
    first_audio_ms: int | None = None  # turn start to first binary frame sent
    failures: int = 0
    error_sent: bool = False
    # State speaking was sent since the turn started or the last look.
    speaking_sent: bool = False


# A queue item that the worker sets when it reaches it (see _drain_speech).
SpeechItem = str | asyncio.Event | None


async def _speak_sentence(
    tts: TTSClient, text: str, send: Send, sp: _SpeechState, t0: float
) -> None:
    """Synthesize one sentence and stream it. Synthesis errors are contained;
    send errors propagate (the socket is gone, the turn cannot continue)."""
    sentence_id: int | None = None
    failed = False
    stream = tts.synthesize(text)
    try:
        while True:
            try:
                frame = await anext(stream)
            except StopAsyncIteration:
                break
            except Exception as exc:  # noqa: BLE001 - one bad sentence must not end the turn
                failed = True
                log.error("TTS failed: %s", type(exc).__name__)
                break
            if not frame:
                continue
            if sentence_id is None:
                sp.sentence_id += 1
                sentence_id = sp.sentence_id
                await send(
                    {
                        "type": "tts_start",
                        "sentence_id": sentence_id,
                        "sample_rate": tts.sample_rate,
                    }
                )
            if sp.first_audio_ms is None:
                sp.first_audio_ms = int((time.monotonic() - t0) * 1000)
            if not sp.speaking_sent:
                sp.speaking_sent = True
                await send({"type": "state", "state": "speaking"})
            await send(frame)
    finally:
        await _close_stream(stream)
    if sentence_id is not None:
        await send({"type": "tts_end", "sentence_id": sentence_id})
    if failed:
        sp.failures += 1
        if not sp.error_sent:
            sp.error_sent = True
            await send({"type": "error", "message": MSG_TTS_FAILED})


async def _tts_worker(
    tts: TTSClient,
    queue: asyncio.Queue[SpeechItem],
    send: Send,
    sp: _SpeechState,
    t0: float,
) -> None:
    """Speak queued sentences in order until the None sentinel arrives. An
    Event in the queue is set when reached."""
    while True:
        item = await queue.get()
        if item is None:
            return
        if isinstance(item, asyncio.Event):
            item.set()
            continue
        speech = clean_for_speech(item)
        if speech:
            await _speak_sentence(tts, speech, send, sp, t0)


async def _drain_speech(
    queue: asyncio.Queue[SpeechItem], worker: asyncio.Task[None] | None
) -> None:
    """Wait until the worker has sent every sentence queued so far. Raises if
    the worker failed (a send error: the socket is gone)."""
    if worker is None:
        return
    if not worker.done():
        reached = asyncio.Event()
        queue.put_nowait(reached)
        waiter = asyncio.ensure_future(reached.wait())
        try:
            await asyncio.wait([waiter, worker], return_when=asyncio.FIRST_COMPLETED)
        finally:
            waiter.cancel()
    if worker.done():
        await worker


async def _stop_worker(worker: asyncio.Task[None] | None) -> None:
    """Cancel the worker if it is still running and wait until it has stopped.

    asyncio.wait does not raise the worker's CancelledError here, but a
    cancellation of the calling task still propagates.
    """
    if worker is None:
        return
    if worker.done():
        if not worker.cancelled() and worker.exception() is not None:
            # Retrieve it so asyncio does not warn; the turn is ending anyway.
            log.debug("TTS worker ended with %s", type(worker.exception()).__name__)
        return
    worker.cancel()
    await asyncio.wait([worker])


async def run_text_turn(
    text: str,
    history: list[Message],
    llm: LLMClient,
    usage: UsageTracker,
    send: Send,
    history_turns: int = 10,
    max_output_tokens: int = 200,
    turn_timeout_s: float = 60.0,
    tts: TTSClient | None = None,
    looker: LookTool | None = None,
) -> str | None:
    """Run one turn. Returns the reply text, or None if the turn failed.

    tts: speak the reply sentence by sentence; None keeps text-only behavior.
    looker: declares look() and runs it; None offers no tools.

    Never raises except CancelledError. Unexpected errors are logged (type name
    only) and reported to the client as a generic error, best effort.
    """
    try:
        t0 = time.monotonic()
        deadline = asyncio.get_running_loop().time() + turn_timeout_s
        await send({"type": "state", "state": "thinking"})
        return await _run_turn(
            text,
            history,
            llm,
            usage,
            send,
            history_turns,
            max_output_tokens,
            tts,
            looker,
            t0=t0,
            deadline=deadline,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.error("turn failed unexpectedly: %s", type(exc).__name__)
        await _send_quietly(send, {"type": "error", "message": MSG_UNEXPECTED})
        await _send_quietly(send, {"type": "state", "state": "idle"})
        return None


async def run_voice_turn(
    pcm16: bytes,
    history: list[Message],
    llm: LLMClient,
    usage: UsageTracker,
    send: Send,
    stt: STTClient,
    language: str | None = None,
    sample_rate: int = SAMPLE_RATE,
    history_turns: int = 10,
    max_output_tokens: int = 200,
    turn_timeout_s: float = 60.0,
    tts: TTSClient | None = None,
    looker: LookTool | None = None,
) -> str | None:
    """Run one spoken turn: STT, then the text turn body with the transcript.

    pcm16: the whole utterance, PCM16 mono little-endian at sample_rate.
    language: passed to the STT client (None lets it detect the language).
    Returns the reply text, or None if the turn failed or nothing was heard.

    Never raises except CancelledError, like run_text_turn.
    """
    try:
        return await _run_voice_turn(
            pcm16,
            history,
            llm,
            usage,
            send,
            stt,
            language,
            sample_rate,
            history_turns,
            max_output_tokens,
            turn_timeout_s,
            tts,
            looker,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.error("voice turn failed unexpectedly: %s", type(exc).__name__)
        await _send_quietly(send, {"type": "error", "message": MSG_UNEXPECTED})
        await _send_quietly(send, {"type": "state", "state": "idle"})
        return None


def _cap_reached(usage: UsageTracker) -> bool:
    try:
        usage.check()
    except CapExceeded:
        log.warning("turn blocked: daily token cap reached (%d tokens today)", usage.today_total)
        return True
    return False


async def _cap_allows(usage: UsageTracker, send: Send) -> bool:
    """False (after sending the cap error and idle) if the daily cap is reached."""
    if _cap_reached(usage):
        await send({"type": "error", "message": MSG_CAP})
        await send({"type": "state", "state": "idle"})
        return False
    return True


@dataclass(frozen=True)
class _VoiceTiming:
    stt_ms: int
    audio_ms: int


async def _run_voice_turn(
    pcm16: bytes,
    history: list[Message],
    llm: LLMClient,
    usage: UsageTracker,
    send: Send,
    stt: STTClient,
    language: str | None,
    sample_rate: int,
    history_turns: int,
    max_output_tokens: int,
    turn_timeout_s: float,
    tts: TTSClient | None,
    looker: LookTool | None,
) -> str | None:
    t0 = time.monotonic()
    # One deadline for STT and the rest of the turn.
    deadline = asyncio.get_running_loop().time() + turn_timeout_s
    audio_ms = pcm16_duration_ms(len(pcm16), sample_rate)
    await send({"type": "state", "state": "thinking"})
    # Checked before STT so a capped day does not spend CPU on transcription.
    if not await _cap_allows(usage, send):
        return None

    stt_error: Exception | None = None
    transcript: Any = ""
    t_stt = time.monotonic()
    try:
        async with asyncio.timeout_at(deadline):
            try:
                transcript = await stt.transcribe(pcm16, sample_rate, language)
            except Exception as exc:  # noqa: BLE001 - any STT error ends the turn
                stt_error = exc
    except TimeoutError:
        log.error(
            "turn timed out during STT after %d ms (audio_ms=%d)",
            int((time.monotonic() - t0) * 1000),
            audio_ms,
        )
        await send({"type": "error", "message": MSG_TIMEOUT})
        await send({"type": "state", "state": "idle"})
        return None
    stt_ms = int((time.monotonic() - t_stt) * 1000)

    if stt_error is not None:
        # Type name only: provider messages could echo input details.
        log.error(
            "STT failed: %s (stt_ms=%d audio_ms=%d)", type(stt_error).__name__, stt_ms, audio_ms
        )
        await send({"type": "error", "message": MSG_STT_FAILED})
        await send({"type": "state", "state": "idle"})
        return None

    text = transcript.strip() if isinstance(transcript, str) else ""
    if not text:
        log.info("stt: nothing recognized (stt_ms=%d audio_ms=%d)", stt_ms, audio_ms)
        await send({"type": "error", "message": MSG_NOT_CAUGHT})
        await send({"type": "state", "state": "idle"})
        return None

    await send({"type": "transcript", "role": "user", "text": text, "final": True})
    return await _run_turn(
        text,
        history,
        llm,
        usage,
        send,
        history_turns,
        max_output_tokens,
        tts,
        looker,
        t0=t0,
        deadline=deadline,
        voice=_VoiceTiming(stt_ms, audio_ms),
    )


async def _run_turn(
    text: str,
    history: list[Message],
    llm: LLMClient,
    usage: UsageTracker,
    send: Send,
    history_turns: int,
    max_output_tokens: int,
    tts: TTSClient | None,
    looker: LookTool | None = None,
    *,
    t0: float,
    deadline: float,
    voice: _VoiceTiming | None = None,
) -> str | None:
    """The shared turn body after state thinking has been sent.

    t0: turn start (time.monotonic) for the latency line. deadline: end of the
    turn in event loop time (asyncio.timeout_at). voice: STT timings for the
    latency line, None for text turns.
    """
    if not await _cap_allows(usage, send):
        return None

    messages = [*history, Message(role="user", text=text)]
    tools = [LOOK_TOOL] if looker is not None else []
    splitter = SentenceSplitter()
    sentence_count = 0
    st = _TurnState()
    sp = _SpeechState()
    failed: Exception | None = None
    timed_out = False
    capped = False
    reply: str | None = None
    llm_ms = 0
    look_ms: int | None = None
    calls: list[ToolCall] = []

    queue: asyncio.Queue[SpeechItem] = asyncio.Queue()
    worker: asyncio.Task[None] | None = None
    if tts is not None:
        worker = asyncio.create_task(_tts_worker(tts, queue, send, sp, t0))

    def enqueue(sentences: list[str]) -> None:
        nonlocal sentence_count
        sentence_count += len(sentences)
        if worker is not None:
            for sentence in sentences:
                queue.put_nowait(sentence)

    async def stream_once(tool_results: list[ToolResult] | None) -> Exception | None:
        """One LLM request. Its usage is always recorded. Returns the provider
        error, if any. ToolCalls are collected from the first request only."""
        nonlocal llm_ms
        ru = _RequestUsage()
        error: Exception | None = None
        started = False
        extra_chars = sum(len(json.dumps(r.response)) for r in tool_results or [])
        # The follow-up's text joins the text from before the call: keep a space.
        gap = tool_results is not None and bool(st.parts) and not st.parts[-1][-1:].isspace()
        t_llm = time.monotonic()
        try:
            stream = llm.stream_reply(
                messages, tools, tool_results=tool_results, allow_tool_calls=tool_results is None
            )
            started = True
            try:
                while True:
                    try:
                        event = await anext(stream)
                    except StopAsyncIteration:
                        # Done is the last event, so this is "LLM done".
                        st.done_ms = int((time.monotonic() - t0) * 1000)
                        break
                    except Exception as exc:  # noqa: BLE001 - any provider error ends the turn
                        error = exc
                        break
                    if isinstance(event, TextDelta):
                        if not event.text:
                            continue
                        if st.ttft_ms is None:
                            st.ttft_ms = int((time.monotonic() - t0) * 1000)
                        if tts is None and not sp.speaking_sent:
                            sp.speaking_sent = True
                            await send({"type": "state", "state": "speaking"})
                        if gap:
                            gap = False
                            if not event.text[:1].isspace():
                                st.parts.append(" ")
                        st.parts.append(event.text)
                        await send(
                            {
                                "type": "transcript",
                                "role": "assistant",
                                "text": "".join(st.parts),
                                "final": False,
                            }
                        )
                        enqueue(splitter.feed(event.text))
                    elif isinstance(event, ToolCall):
                        if tool_results is None:
                            calls.append(event)
                    elif isinstance(event, UsageUpdate):
                        ru.last_usage = event
                    elif isinstance(event, Done):
                        ru.done = event
            finally:
                await _close_stream(stream)
        finally:
            # Runs on success, failure, timeout, and cancellation alike.
            ms = int((time.monotonic() - t_llm) * 1000)
            llm_ms += ms
            if started:
                st.requests += 1
                _record_usage(usage, llm, ru, messages, max_output_tokens, ms, extra_chars)
                if ru.done is not None:
                    st.tokens_in += ru.done.tokens_in
                    st.tokens_out += ru.done.tokens_out
                    st.tokens_thinking += ru.done.tokens_thinking
        return error

    async def run_tools(lookup: LookTool) -> list[ToolResult]:
        """Answer every call of the first request; only the first look runs."""
        nonlocal look_ms
        # Speech from before the call (including a sentence the splitter still
        # holds) goes out first, so no speaking state lands in the look.
        rest = splitter.flush()
        if rest:
            enqueue([rest])
        await _drain_speech(queue, worker)
        await send({"type": "state", "state": "looking"})
        sp.speaking_sent = False
        t_look = time.monotonic()
        results: list[ToolResult] = []
        looked = False
        for call in calls:
            if call.name != LOOK_TOOL.name:
                results.append(ToolResult(call, {"error": f"Unknown tool: {call.name}."}))
            elif looked:
                results.append(ToolResult(call, {"error": MSG_ONE_LOOK}))
            else:
                looked = True
                results.append(ToolResult(call, await lookup.look(call.args.get("question"))))
        look_ms = int((time.monotonic() - t_look) * 1000)
        await send({"type": "state", "state": "thinking"})
        return results

    try:
        async with asyncio.timeout_at(deadline):
            failed = await stream_once(None)
            if failed is None and calls and looker is not None:
                results = await run_tools(looker)
                if _cap_reached(usage):
                    capped = True
                else:
                    failed = await stream_once(results)

            if failed is None and not capped:
                rest = splitter.flush()
                if rest:
                    enqueue([rest])
                reply = "".join(st.parts)
                await send(
                    {"type": "transcript", "role": "assistant", "text": reply, "final": True}
                )
                if reply.strip():
                    history.append(Message(role="user", text=text))
                    history.append(Message(role="assistant", text=reply))
                    trim_history(history, history_turns)
                else:
                    log.warning("turn: the model gave no text (%d requests)", st.requests)
                if worker is not None:
                    queue.put_nowait(None)
                    await worker  # raises if a send failed inside the worker
    except TimeoutError:
        timed_out = True
    finally:
        # No binary frame may follow the idle sent below or by the caller.
        await _stop_worker(worker)

    total_ms = int((time.monotonic() - t0) * 1000)
    if timed_out:
        log.error("turn timed out after %d ms (llm %d ms)", total_ms, llm_ms)
        await send({"type": "error", "message": MSG_TIMEOUT})
        await send({"type": "state", "state": "idle"})
        return None

    if failed is not None:
        log.error("LLM request failed: %s (after %d ms)", type(failed).__name__, llm_ms)
        await send({"type": "error", "message": MSG_LLM_FAILED})
        await send({"type": "state", "state": "idle"})
        return None

    if capped:
        await send({"type": "error", "message": MSG_CAP})
        await send({"type": "state", "state": "idle"})
        return None

    log.info(
        "turn latency: stt_ms=%s audio_ms=%s llm_ttft_ms=%s llm_done_ms=%s "
        "tts_first_chunk_ms=%s look_ms=%s total_ms=%d sentences=%d spoken=%d tts_failures=%d",
        voice.stt_ms if voice is not None else "n/a",
        voice.audio_ms if voice is not None else "n/a",
        st.ttft_ms if st.ttft_ms is not None else "-",
        st.done_ms if st.done_ms is not None else "-",
        sp.first_audio_ms if sp.first_audio_ms is not None else "-",
        look_ms if look_ms is not None else "-",
        total_ms,
        sentence_count,
        sp.sentence_id,
        sp.failures,
    )
    log.info(
        "turn usage: llm_ms=%d requests=%d tokens_in=%d tokens_out=%d tokens_thinking=%d",
        llm_ms,
        st.requests,
        st.tokens_in,
        st.tokens_out,
        st.tokens_thinking,
    )

    await send({"type": "state", "state": "idle"})
    return reply
