"""Score a local model on picking the tools (camera and recall).

    python -m obsilord.llm.routecheck [--model ID] [--no-search] [--no-recall]
                                      [--repeat N] [--mode auto|native|router]

Sends a fixed set of labelled messages (some need look(), some
search_image(), some recall(), most need no tool, a few are follow-ups, one
answered by an earlier tool result in history) against the
[llm_local] server in config.toml, the way the app would: in native mode one
request with the tools declared and the app's system prompt, in router mode
the routing request. Prints the misses, the accuracy per expected tool, how
many calls came back written as text, and the latency. Run it per model and
mode to compare them; --repeat matters, small models vary from run to run.
Nothing is logged to the usage file and no camera is used.

--no-search: search_image() is not offered (as with [image_search] off);
messages that ask to identify something then expect look().
--no-recall: recall() is not offered (as in Gemini mode or with [memory]
recall off); messages about earlier conversations then expect no tool. No
store is searched either way: only the decision is scored.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass

from obsilord.config import TOOL_MODES, load_settings
from obsilord.llm.base import Message, ToolNote, ToolSpec
from obsilord.llm.openai_compat import NO_TOOL, LocalClient, LocalLLMError, OpenAICompatServer
from obsilord.look import LOOK_TOOL, SEARCH_IMAGE_TOOL
from obsilord.memory.recall import RECALL_TOOL

LOOK = LOOK_TOOL.name
SEARCH = SEARCH_IMAGE_TOOL.name
RECALL = RECALL_TOOL.name

_MUG = (Message("user", "What am I holding?"), Message("assistant", "A red mug. Thrilling."))
_CAT = (Message("user", "Look at my cat."), Message("assistant", "A fat grey cat, asleep on a chair."))
_OWL = (
    Message("user", "Look, what do you think of this?"),
    Message(
        "assistant",
        "A lovely little owl. It suits the desk.",
        tools=(ToolNote(LOOK, "What is the user showing?", "A small blue ceramic owl figurine on a wooden desk."),),
    ),
)
_JOKE = (
    Message("user", "Tell me a short joke."),
    Message("assistant", "Why did the scarecrow win an award? He was outstanding in his field."),
)


@dataclass(frozen=True)
class Case:
    text: str
    expected: str
    history: tuple[Message, ...] = ()


CASES: tuple[Case, ...] = (
    Case("What am I holding?", LOOK),
    Case("How do I look today?", LOOK),
    Case("Can you read what this label says?", LOOK),
    Case("What color is my shirt?", LOOK),
    Case("Is anyone standing behind me?", LOOK),
    Case("Look at this and tell me what you think.", LOOK),
    Case("Do I look tired?", LOOK),
    Case("How many fingers am I holding up?", LOOK),
    Case("What's on my desk right now?", LOOK),
    Case("Check out my new haircut.", LOOK),
    Case("And now?", LOOK, _MUG),
    Case("What is she doing now?", LOOK, _CAT),
    Case("What album is this cover from?", SEARCH),
    Case("Can you identify this book for me?", SEARCH),
    Case("What brand is this bottle I'm holding?", SEARCH),
    Case("Look up this logo and tell me which company it is.", SEARCH),
    Case("Which game is this box from?", SEARCH),
    Case("Tell me a joke.", NO_TOOL),
    Case("What is the capital of Australia?", NO_TOOL),
    Case("How are you today?", NO_TOOL),
    Case("Explain how a rainbow forms.", NO_TOOL),
    Case("Who wrote The Hobbit?", NO_TOOL),
    Case("What do you think of the album Demon Days?", NO_TOOL),
    Case("I'm holding a cup of coffee, any tips for staying awake?", NO_TOOL),
    Case("Good night!", NO_TOOL),
    Case("Thanks, that's all.", NO_TOOL),
    Case("Give me three names for a grey cat.", NO_TOOL),
    Case("What should I drink from it?", NO_TOOL, _MUG),
    Case("Why do cats sleep so much?", NO_TOOL, _CAT),
    Case("What did I just ask you?", NO_TOOL, _MUG),
    Case("What color was it?", NO_TOOL, _OWL),
    Case("What did you just say?", NO_TOOL, _JOKE),
    Case("What did we talk about yesterday?", RECALL),
    Case("Remember the figure I showed you last week? What was it?", RECALL),
    Case("When did I say my dentist appointment was?", RECALL),
    Case("I told you my cat's name a while ago. What is it?", RECALL),
    Case("Did we ever talk about skiing?", RECALL),
    Case("What did we talk about on the 5th of November?", RECALL),
    Case("Pita knocked over my plant again.", RECALL),
    Case("I am so tired today.", NO_TOOL),
    Case("I just made coffee.", NO_TOOL),
)


def expected_for(case: Case, search: bool, recall: bool) -> str:
    """The tool the case expects with the offered tools."""
    if case.expected == SEARCH and not search:
        return LOOK
    if case.expected == RECALL and not recall:
        return NO_TOOL
    return case.expected


@dataclass(frozen=True)
class Result:
    case: Case
    expected: str
    got: str  # a tool name, NO_TOOL, or "unusable"
    arg: str
    ms: float
    # Native mode: the call came back written as text, not as a tool call.
    as_text: bool = False


async def check(
    client: LocalClient,
    mode: str,
    tools: list[ToolSpec],
    cases: tuple[Case, ...],
    repeat: int,
) -> list[Result]:
    """Runs every case repeat times in mode ("native" or "router"). Raises
    LocalLLMError when a request itself fails (server down, unknown model, no
    json_schema support); an unparsable router answer counts as "unusable"."""
    search = any(t.name == SEARCH for t in tools)
    recall = any(t.name == RECALL for t in tools)
    results: list[Result] = []
    for _ in range(repeat):
        for case in cases:
            expected = expected_for(case, search, recall)
            history = [*case.history, Message("user", case.text)]
            t0 = time.perf_counter()
            as_text = False
            if mode == "native":
                route, _, as_text = await client.native_route(history, tools)
            else:
                route, _ = await client.route(history, tools)
            ms = (time.perf_counter() - t0) * 1000
            got, arg = route if route is not None else ("unusable", "")
            results.append(Result(case, expected, got, arg, ms, as_text))
    return results


async def _run(model: str, search: bool, recall: bool, repeat: int, mode: str) -> int:
    settings = load_settings()
    server = OpenAICompatServer(settings.llm_local)
    tools = [LOOK_TOOL, *([SEARCH_IMAGE_TOOL] if search else []), *([RECALL_TOOL] if recall else [])]
    try:
        try:
            resolved = model or await server.default_model()
            if mode == "auto":
                mode = await server.tool_mode(resolved)
            # Native mode sends the app's system prompt, as the app does; the
            # router request never uses it.
            client = server.client(resolved, tool_mode=mode)
            print(f"tool check: {resolved} at {settings.llm_local.base_url}, mode {mode}, "
                  f"tools {', '.join(t.name for t in tools)}, {len(CASES)} messages x {repeat}")
            results = await check(client, mode, tools, CASES, repeat)
        except LocalLLMError as exc:
            print(f"error: {exc}")
            return 2
    finally:
        await server.aclose()
    _report(results)
    return 0


def _report(results: list[Result]) -> None:
    misses = [r for r in results if r.got != r.expected]
    if misses:
        print("\nmisses:")
        for r in misses:
            context = " (follow-up)" if r.case.history else ""
            print(f"  expected {r.expected:<12} got {r.got:<12} {r.case.text!r}{context}")
    print("\nby expected tool:")
    for name in (LOOK, SEARCH, RECALL, NO_TOOL):
        group = [r for r in results if r.expected == name]
        if group:
            ok = sum(r.got == r.expected for r in group)
            print(f"  {name:<12} {ok}/{len(group)}")
    unusable = sum(r.got == "unusable" for r in results)
    times = sorted(r.ms for r in results)
    p90 = times[min(len(times) - 1, int(len(times) * 0.9))]
    ok = len(results) - len(misses)
    as_text = sum(r.as_text for r in results)
    print(f"\naccuracy {ok}/{len(results)} ({100 * ok / len(results):.0f}%), "
          f"unusable answers {unusable}, calls written as text {as_text}")
    print(f"latency: median {statistics.median(times):.0f} ms, p90 {p90:.0f} ms")
    if unusable:
        print("Unusable answers: the model may spend its tokens thinking before the JSON, "
              "or the server may not support response_format json_schema. A model without "
              "thinking (or with thinking turned off in the server) usually fixes it.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m obsilord.llm.routecheck",
        description="Score a local model on picking the tools (camera and recall).",
    )
    parser.add_argument("--model", default="", help="model id (default: [llm_local] model, "
                        "or the first model the server lists)")
    parser.add_argument("--no-search", action="store_true", help="do not offer search_image()")
    parser.add_argument("--no-recall", action="store_true", help="do not offer recall()")
    parser.add_argument("--repeat", type=int, default=1, help="run the set N times (1 to 10)")
    parser.add_argument("--mode", choices=TOOL_MODES, default="auto",
                        help="native: tools declared; router: the routing request; "
                        "auto: as the app picks it for the model")
    args = parser.parse_args(argv)
    if not 1 <= args.repeat <= 10:
        parser.error("--repeat must be 1 to 10")
    return asyncio.run(
        _run(args.model.strip(), not args.no_search, not args.no_recall, args.repeat, args.mode)
    )


if __name__ == "__main__":
    sys.exit(main())
