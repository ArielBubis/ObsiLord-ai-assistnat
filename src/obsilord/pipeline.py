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

Without a TTSClient, state speaking is sent on the first text delta.

Tools (a Registry, see obsilord.tools.registry): the turn's tools come from
the registry once per turn, from whether the model is a cloud model and
whether it sees photos. A request that ends with ToolCalls starts a tool
round: the speech queued so far is sent first, then every call runs in order
through the registry (which applies the modes, the limits, the groups and
the timeouts), with state looking right before a camera tool runs (policy
face_state, through TurnContext.face), then state thinking. The next
request carries every round so far. Up to max_rounds rounds; the request
after the last allowed round forbids further calls (allow_tool_calls=False),
so a turn makes at most max_rounds + 1 requests. Every request after the
first is preceded by a cap check for a counted client; if the cap is reached
then, the turn ends with the cap error. Speaking is sent again on the first
audio frame (or text delta without TTS) after a round.

Ask (a tool in mode Ask, or an acting tool after untrusted output): the
registry calls TurnContext.confirm, which is None without confirms (no page:
Ask counts as no). Otherwise the turn deadline is paused, the fixed question
is spoken after the speech before the call (with TTS), the face goes back to
thinking, and {"type": "confirm", "id", "tool", "label", "question",
"details", "timeout_s"} goes out. The answer is a click, typed text or a
recording (transcribed and sent as a user transcript; not caught sends an
error and the wait goes on); yes and no words count as such, anything else
counts as no and goes to confirms.run_next, and no answer within timeout_s is
"timeout". {"type": "confirm_done", "id", "answer": "yes" | "no" | "timeout"}
follows, and the deadline moves out by the time spent asking. State looking
comes only after a yes.

A cloud model gets the history without the tool notes of tools whose results
may not go to it (TurnTools.cloud_visible): private-data tools that are off
for a cloud model, and any tool not offered now.

Usage accounting: once a request has been started, usage is always recorded,
in a finally block, so failed, timed-out, and cancelled streams count toward
the cap. Each request is recorded on its own:
- Done arrived: its counts.
- Otherwise the last UsageUpdate seen, flagged partial.
- Otherwise a conservative estimate, flagged estimated: (history characters +
  the client's prompt_overhead_chars, i.e. the system prompt, + the tool
  results) / 4 for input and max_output_tokens for output.

Photos (photos, see obsilord.photos): when the chat model sees photos
(llm.sees_images(), a local vision model), the camera source offers the photo
variant of look, whose result carries the frame (ToolResult.image) to the
next request. Its note says the model looked itself and carries the image
while Photos keeps it (the newest photos.keep). In "disk" mode the turn's
camera frame (also Qwenny's and search_image's) is saved after its round and
its file name goes in the note of the call that took it; a failed save is
logged and the turn goes on.

History is updated once the LLM has finished with a non-empty reply (before
speech ends), so failed LLM turns leave it unchanged. It holds the user text
and the reply text, and the reply carries ToolNotes: each tool call of the
turn with its result as one capped line, so a later turn can refer to it.
on_turn (if given) is then awaited with the user text, the reply and the
notes, to write the conversation store. Its errors are logged (type name
only) and never end the turn.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import secrets
import time
from dataclasses import dataclass, field, replace
from typing import Any, AsyncIterator, Awaitable, Callable

from obsilord.llm.base import (
    Done,
    LLMClient,
    Message,
    TextDelta,
    ToolCall,
    ToolNote,
    ToolResult,
    UsageUpdate,
    note_from_result,
)
from obsilord.permissions import answer_kind
from obsilord.photos import TurnPhotos, keep_newest_photos
from obsilord.sentences import SentenceSplitter
from obsilord.stt.base import SAMPLE_RATE, STTClient, pcm16_duration_ms
from obsilord.tools.registry import Answer, ConfirmRequest, Registry, TurnContext, TurnInfo, TurnTools
from obsilord.tts.base import TTSClient
from obsilord.tts.text import clean_for_speech
from obsilord.usage import CapExceeded, UsageTracker

log = logging.getLogger(__name__)

# Sends one message: a dict goes out as a JSON text frame, bytes as a binary frame.
Send = Callable[[dict[str, Any] | bytes], Awaitable[None]]
# Called once per finished turn with (user text, reply, tool notes).
OnTurn = Callable[[str, str, tuple[ToolNote, ...]], Awaitable[None]]
# The page's reply to a confirm: a click (True is Yes), typed text, or a
# push-to-talk recording (PCM16 mono at 16 kHz).
ConfirmReply = bool | str | bytes


@dataclass(frozen=True)
class Confirms:
    """How a turn asks the user before a tool runs (the app's side).

    reply(id): waits for the page's next reply to confirm id; cancelled when
    the confirm times out or the turn ends. run_next(text): an answer that was
    neither yes nor no; it counts as no, and the app runs it as a normal turn
    after this one. stt and language: for spoken replies (None: a recording
    counts as not caught).
    """

    reply: Callable[[str], Awaitable[ConfirmReply]]
    run_next: Callable[[str], None]
    timeout_s: float = 20.0
    stt: STTClient | None = None
    language: str | None = None

# Tool rounds per turn when the caller does not say ([tools] max_rounds).
DEFAULT_MAX_ROUNDS = 3
# History note of a look whose photo the chat model saw itself.
PHOTO_NOTE_TEXT = "A photo was taken and you looked at it yourself."

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
    counted = _counts_toward_cap(llm)
    try:
        if ru.done is not None:
            d = ru.done
            usage.record(
                model, d.tokens_in, d.tokens_out, d.tokens_thinking, total_ms, counted=counted
            )
        elif ru.last_usage is not None:
            u = ru.last_usage
            usage.record(
                model,
                u.tokens_in,
                u.tokens_out,
                u.tokens_thinking,
                total_ms,
                partial=True,
                counted=counted,
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
                counted=counted,
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


class _FaceTrail:
    """Wraps send: keeps the face states of a turn and logs them once, when
    idle goes out ("turn face: thinking > looking > ..."), to check the face
    in a manual test. States only, never content."""

    def __init__(self, send: Send) -> None:
        self._send = send
        self._states: list[str] = []

    async def __call__(self, msg: dict[str, Any] | bytes) -> None:
        if isinstance(msg, dict) and msg.get("type") == "state":
            state = str(msg.get("state"))
            self._states.append(state)
            if state == "idle":
                log.info("turn face: %s", " > ".join(self._states))
                self._states = []
        await self._send(msg)


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
    *,
    tools: Registry | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    on_turn: OnTurn | None = None,
    photos: TurnPhotos | None = None,
    confirms: Confirms | None = None,
) -> str | None:
    """Run one turn. Returns the reply text, or None if the turn failed.

    tts: speak the reply sentence by sentence; None keeps text-only behavior.
    tools: the tools the model may call (see the module docstring); None
    offers none. max_rounds: tool rounds per turn.
    on_turn: awaited once the turn has a reply, to store it.
    photos: the Photos setting (see obsilord.photos); None is "off".

    Never raises except CancelledError. Unexpected errors are logged (type name
    only) and reported to the client as a generic error, best effort.
    """
    send = _FaceTrail(send)
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
            t0=t0,
            deadline=deadline,
            tools=tools,
            max_rounds=max_rounds,
            on_turn=on_turn,
            photos=photos,
            confirms=confirms,
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
    *,
    tools: Registry | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    on_turn: OnTurn | None = None,
    photos: TurnPhotos | None = None,
    confirms: Confirms | None = None,
) -> str | None:
    """Run one spoken turn: STT, then the text turn body with the transcript.

    pcm16: the whole utterance, PCM16 mono little-endian at sample_rate.
    language: passed to the STT client (None lets it detect the language).
    Returns the reply text, or None if the turn failed or nothing was heard.

    Never raises except CancelledError, like run_text_turn.
    """
    send = _FaceTrail(send)
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
            tools=tools,
            max_rounds=max_rounds,
            on_turn=on_turn,
            photos=photos,
            confirms=confirms,
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        log.error("voice turn failed unexpectedly: %s", type(exc).__name__)
        await _send_quietly(send, {"type": "error", "message": MSG_UNEXPECTED})
        await _send_quietly(send, {"type": "state", "state": "idle"})
        return None


async def _sees_images(llm: LLMClient) -> bool:
    """Whether the chat model takes photos (a local vision model). Clients
    without sees_images (Gemini) and a failing check count as no."""
    check = getattr(llm, "sees_images", None)
    if check is None:
        return False
    try:
        return await check() is True
    except Exception as exc:  # noqa: BLE001 - look() falls back to Qwenny
        log.warning("image support check failed: %s", type(exc).__name__)
        return False


def _counts_toward_cap(llm: LLMClient) -> bool:
    """False for clients whose requests cost nothing (a local model)."""
    return getattr(llm, "counts_toward_cap", True) is not False


def _is_cloud(llm: LLMClient) -> bool:
    """Whether the client's requests leave the PC; unknown counts as yes."""
    return getattr(llm, "cloud", True) is not False


def _cap_reached(usage: UsageTracker, llm: LLMClient) -> bool:
    if not _counts_toward_cap(llm):
        return False
    try:
        usage.check()
    except CapExceeded:
        log.warning("turn blocked: daily token cap reached (%d tokens today)", usage.today_total)
        return True
    return False


async def _cap_allows(usage: UsageTracker, llm: LLMClient, send: Send) -> bool:
    """False (after sending the cap error and idle) if the daily cap is reached.
    Always True for a client that does not count toward the cap."""
    if _cap_reached(usage, llm):
        await send({"type": "error", "message": MSG_CAP})
        await send({"type": "state", "state": "idle"})
        return False
    return True


def cloud_history(history: list[Message], visible: Callable[[str], bool]) -> list[Message]:
    """history for a cloud model: tool notes of tools whose results may not go
    to it (private data, or a tool not offered now) are left out."""
    out: list[Message] = []
    for m in history:
        kept = tuple(n for n in m.tools if visible(n.name))
        out.append(m if len(kept) == len(m.tools) else replace(m, tools=kept))
    return out


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
    *,
    tools: Registry | None,
    max_rounds: int,
    on_turn: OnTurn | None = None,
    photos: TurnPhotos | None = None,
    confirms: Confirms | None = None,
) -> str | None:
    t0 = time.monotonic()
    # One deadline for STT and the rest of the turn.
    deadline = asyncio.get_running_loop().time() + turn_timeout_s
    audio_ms = pcm16_duration_ms(len(pcm16), sample_rate)
    await send({"type": "state", "state": "thinking"})
    # Checked before STT so a capped day does not spend CPU on transcription.
    if not await _cap_allows(usage, llm, send):
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
        t0=t0,
        deadline=deadline,
        voice=_VoiceTiming(stt_ms, audio_ms),
        tools=tools,
        max_rounds=max_rounds,
        on_turn=on_turn,
        photos=photos,
        confirms=confirms,
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
    *,
    t0: float,
    deadline: float,
    voice: _VoiceTiming | None = None,
    tools: Registry | None,
    max_rounds: int,
    on_turn: OnTurn | None = None,
    photos: TurnPhotos | None = None,
    confirms: Confirms | None = None,
) -> str | None:
    """The shared turn body after state thinking has been sent.

    t0: turn start (time.monotonic) for the latency line. deadline: end of the
    turn in event loop time (asyncio.timeout_at). voice: STT timings for the
    latency line, None for text turns.
    """
    if not await _cap_allows(usage, llm, send):
        return None

    photos = photos or TurnPhotos()
    if not photos.kept:
        # Photos is "off" (maybe switched while an earlier turn ran): no
        # earlier photo goes with this request.
        keep_newest_photos(history, 0)
    cloud = _is_cloud(llm)

    async def sees_images() -> bool:
        return await _sees_images(llm)

    turn_tools: TurnTools | None = None
    if tools is not None:
        try:
            turn_tools = await tools.for_turn(TurnInfo(cloud, sees_images))
        except Exception as exc:  # noqa: BLE001 - a broken source must not end the turn
            # The turn goes on without tools (and a cloud model gets no notes).
            log.error("turn tools failed, answering without tools: %s", type(exc).__name__)
    specs = turn_tools.specs if turn_tools is not None else []
    sent_history = history
    if cloud:
        sent_history = cloud_history(history, turn_tools.cloud_visible if turn_tools else lambda name: False)
    messages = [*sent_history, Message(role="user", text=text)]
    rounds: list[list[ToolResult]] = []
    finished: tuple[str, tuple[ToolNote, ...]] | None = None
    frame: bytes | None = None  # the camera frame of this turn's look, for "disk"
    camera_index: int | None = None  # which result of the turn ran the camera
    running_index = 0  # the result the running tool call will be
    saved_name = ""

    async def snapshot(kind: str, jpeg: bytes) -> None:
        # The frame (kind "camera") or the crop about to be uploaded ("search")
        # for the page's snapshot panel. The app's send drops these unless the
        # page asked for them. Never logged.
        nonlocal frame, camera_index
        if kind == "camera":
            frame = jpeg
            camera_index = running_index
        b64 = base64.b64encode(jpeg).decode("ascii")
        await send({"type": "snapshot", "kind": kind, "jpeg_b64": b64})

    async def save_frame() -> str:
        """"disk": write this turn's frame. A failure is logged, never fatal."""
        if photos.mode != "disk" or photos.save is None or frame is None:
            return ""
        try:
            return await photos.save(frame)
        except Exception as exc:  # noqa: BLE001 - the turn goes on without it
            log.warning("photo not saved: %s", type(exc).__name__)
            return ""

    def turn_notes() -> tuple[ToolNote, ...]:
        notes = []
        for i, r in enumerate(r for results in rounds for r in results):
            note = note_from_result(r)
            if r.image is not None:
                note = replace(note, result=PHOTO_NOTE_TEXT, image=r.image if photos.kept else None)
            if i == camera_index and saved_name:
                note = replace(note, photo=saved_name)
            notes.append(note)
        return tuple(notes)

    splitter = SentenceSplitter()
    sentence_count = 0
    st = _TurnState()
    sp = _SpeechState()
    failed: Exception | None = None
    timed_out = False
    capped = False
    reply: str | None = None
    llm_ms = 0
    tools_ms: int | None = None
    ask_ms = 0  # time spent asking the user (confirms), apart from tools_ms
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

    async def stream_once(allow: bool) -> Exception | None:
        """One LLM request after the rounds so far. Its usage is always
        recorded. Returns the provider error, if any. ToolCalls are collected
        only when calls are allowed."""
        nonlocal llm_ms
        ru = _RequestUsage()
        error: Exception | None = None
        started = False
        extra_chars = sum(len(json.dumps(r.response)) for results in rounds for r in results)
        # Text after a tool round joins the text from before it: keep a space.
        gap = bool(rounds) and bool(st.parts) and not st.parts[-1][-1:].isspace()
        t_llm = time.monotonic()
        try:
            stream = llm.stream_reply(
                messages, specs, rounds=[list(r) for r in rounds], allow_tool_calls=allow
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
                        if allow:
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

    async def face(state: str) -> None:
        # The state a tool runs in, sent right before it runs: after any
        # confirm, so an asked look shows "looking" only after the yes.
        sp.speaking_sent = False
        await send({"type": "state", "state": state})

    async def hear(pcm16: bytes, tool: str) -> str:
        """A spoken confirm reply as text; "" (after an error message) when
        nothing was caught. Never logged."""
        assert confirms is not None
        heard: Any = ""
        if confirms.stt is not None:
            try:
                async with asyncio.timeout(confirms.timeout_s):
                    heard = await confirms.stt.transcribe(pcm16, SAMPLE_RATE, confirms.language)
            except Exception as exc:  # noqa: BLE001 - the user can answer again
                log.error("STT failed for a confirm answer: %s", type(exc).__name__)
                await send({"type": "error", "message": MSG_STT_FAILED})
                return ""
        heard = heard.strip() if isinstance(heard, str) else ""
        if not heard:
            log.info("confirm %s: a spoken answer was not caught, still waiting", tool)
            await send({"type": "error", "message": MSG_NOT_CAUGHT})
            return ""
        await send({"type": "transcript", "role": "user", "text": heard, "final": True})
        return heard

    async def wait_answer(confirm_id: str, tool: str) -> tuple[Answer, str]:
        """The user's answer within confirms.timeout_s, and how it came (for
        the log). A reply that was not caught keeps the wait going; one that
        is neither yes nor no counts as no and goes to run_next."""
        assert confirms is not None
        end = asyncio.get_running_loop().time() + confirms.timeout_s
        while True:
            try:
                async with asyncio.timeout_at(end):
                    got = await confirms.reply(confirm_id)
            except TimeoutError:
                return "timeout", "no reply"
            if isinstance(got, bool):
                return ("yes" if got else "no"), "click"
            how = "spoken" if isinstance(got, bytes) else "typed"
            said = await hear(got, tool) if isinstance(got, bytes) else got.strip()
            if not said:
                continue
            kind = answer_kind(said)
            if kind == "other":
                confirms.run_next(said)
                return "no", f"{how}, other words, run next"
            return kind, how

    async def confirm(request: ConfirmRequest) -> Answer:
        """Ask the user (TurnContext.confirm). The turn deadline is paused
        meanwhile and then moved out by the time spent asking."""
        nonlocal ask_ms
        assert confirms is not None and turn_timeout is not None
        loop = asyncio.get_running_loop()
        started = loop.time()
        resume = turn_timeout.when()
        turn_timeout.reschedule(None)
        try:
            if worker is not None:
                # A fixed question, spoken after the speech before the call.
                queue.put_nowait(request.question)
                await _drain_speech(queue, worker)
            if sp.speaking_sent:
                # The face waits in thinking.
                sp.speaking_sent = False
                await send({"type": "state", "state": "thinking"})
            confirm_id = secrets.token_hex(8)
            log.info("confirm %s: asked (timeout %s s)", request.tool, confirms.timeout_s)
            await send(
                {
                    "type": "confirm",
                    "id": confirm_id,
                    "tool": request.tool,
                    "label": request.label,
                    "question": request.question,
                    "details": request.details,
                    "timeout_s": confirms.timeout_s,
                }
            )
            answer, how = await wait_answer(confirm_id, request.tool)
            log.info(
                "confirm %s: %s (%s) after %d ms",
                request.tool,
                answer,
                how,
                (loop.time() - started) * 1000,
            )
            await send({"type": "confirm_done", "id": confirm_id, "answer": answer})
            return answer
        finally:
            waited = loop.time() - started
            ask_ms += int(waited * 1000)
            if resume is not None:
                turn_timeout.reschedule(resume + waited)

    context = TurnContext(snapshot=snapshot, confirm=confirm if confirms is not None else None, face=face)
    turn_timeout: asyncio.Timeout | None = None

    async def run_round(turn: TurnTools) -> list[ToolResult]:
        """Run the calls of the last request in order. Every policy (modes,
        limits, groups, timeouts, asking) is applied by the registry."""
        nonlocal tools_ms, running_index
        # Speech from before the calls (including a sentence the splitter
        # still holds) goes out first; only a confirm question speaks in the
        # round.
        rest = splitter.flush()
        if rest:
            enqueue([rest])
        await _drain_speech(queue, worker)
        t_tools = time.monotonic()
        asked_before = ask_ms
        results: list[ToolResult] = []
        for call in calls:
            running_index = sum(len(r) for r in rounds) + len(results)
            results.append(await turn.run(call, context))
        spent = int((time.monotonic() - t_tools) * 1000) - (ask_ms - asked_before)
        tools_ms = (tools_ms or 0) + max(spent, 0)
        sp.speaking_sent = False
        await send({"type": "state", "state": "thinking"})
        return results

    try:
        async with asyncio.timeout_at(deadline) as turn_timeout:
            for request in range(max_rounds + 1):
                allow = bool(specs) and request < max_rounds
                if request > 0 and _cap_reached(usage, llm):
                    capped = True
                    break
                calls.clear()
                failed = await stream_once(allow)
                if failed is not None or not calls or turn_tools is None:
                    break
                rounds.append(await run_round(turn_tools))
                if frame is not None and not saved_name:
                    saved_name = await save_frame()

            if failed is None and not capped:
                rest = splitter.flush()
                if rest:
                    enqueue([rest])
                reply = "".join(st.parts)
                await send(
                    {"type": "transcript", "role": "assistant", "text": reply, "final": True}
                )
                if reply.strip():
                    notes = turn_notes()
                    history.append(Message(role="user", text=text))
                    history.append(Message(role="assistant", text=reply, tools=notes))
                    trim_history(history, history_turns)
                    keep_newest_photos(history, photos.kept)
                    finished = (reply, notes)
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

    if finished is not None and on_turn is not None:
        # The turn is in history, so it is stored too (even if speech then
        # timed out). Never ends the turn.
        try:
            await on_turn(text, *finished)
        except Exception as exc:  # noqa: BLE001
            log.warning("turn not stored: %s", type(exc).__name__)

    total_ms = int((time.monotonic() - t0) * 1000)
    if timed_out:
        log.error("turn timed out after %d ms (llm %d ms)", total_ms, llm_ms)
        await send({"type": "error", "message": MSG_TIMEOUT})
        await send({"type": "state", "state": "idle"})
        return None

    if failed is not None:
        # user_message: a provider error that is known to hold no conversation
        # text and is worth showing (for example "cannot reach the local server").
        detail = getattr(failed, "user_message", None)
        detail = detail.strip().rstrip(".") if isinstance(detail, str) else None
        detail = detail or None
        log.error(
            "LLM request failed: %s%s (after %d ms)",
            type(failed).__name__,
            f" ({detail})" if detail else "",
            llm_ms,
        )
        message = f"{MSG_LLM_FAILED} {detail[:1].upper()}{detail[1:]}." if detail else MSG_LLM_FAILED
        await send({"type": "error", "message": message})
        await send({"type": "state", "state": "idle"})
        return None

    if capped:
        await send({"type": "error", "message": MSG_CAP})
        await send({"type": "state", "state": "idle"})
        return None

    log.info(
        "turn latency: stt_ms=%s audio_ms=%s llm_ttft_ms=%s llm_done_ms=%s "
        "tts_first_chunk_ms=%s tools_ms=%s ask_ms=%d total_ms=%d sentences=%d spoken=%d tts_failures=%d",
        voice.stt_ms if voice is not None else "n/a",
        voice.audio_ms if voice is not None else "n/a",
        st.ttft_ms if st.ttft_ms is not None else "-",
        st.done_ms if st.done_ms is not None else "-",
        sp.first_audio_ms if sp.first_audio_ms is not None else "-",
        tools_ms if tools_ms is not None else "-",
        ask_ms,
        total_ms,
        sentence_count,
        sp.sentence_id,
        sp.failures,
    )
    log.info(
        "turn usage: llm_ms=%d requests=%d rounds=%d tokens_in=%d tokens_out=%d tokens_thinking=%d",
        llm_ms,
        st.requests,
        len(rounds),
        st.tokens_in,
        st.tokens_out,
        st.tokens_thinking,
    )

    await send({"type": "state", "state": "idle"})
    return reply
