"""Score a local model as the camera tool router.

    python -m obsilord.llm.routecheck [--model ID] [--no-search] [--repeat N]

Sends a fixed set of labelled messages (some need look(), some
search_image(), most need no tool, a few are follow-ups) through the same
router request the app uses, against the [llm_local] server in config.toml,
and prints the misses, the accuracy per expected tool, and the router
latency. Run it once per model to compare them. Nothing is logged to the usage
file and no camera is used.

--no-search: only look() is offered (as with [image_search] off); messages
that ask to identify something then expect look().
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass

from obsilord.config import load_settings
from obsilord.llm.base import Message
from obsilord.llm.openai_compat import NO_TOOL, LocalLLMError, OpenAICompatServer
from obsilord.look import LOOK_TOOL, SEARCH_IMAGE_TOOL

LOOK = LOOK_TOOL.name
SEARCH = SEARCH_IMAGE_TOOL.name

_MUG = (Message("user", "What am I holding?"), Message("assistant", "A red mug. Thrilling."))
_CAT = (Message("user", "Look at my cat."), Message("assistant", "A fat grey cat, asleep on a chair."))


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
)


@dataclass(frozen=True)
class Result:
    case: Case
    expected: str
    got: str  # a tool name, NO_TOOL, or "unusable"
    arg: str
    ms: float


async def _run(model: str, search: bool, repeat: int) -> int:
    settings = load_settings()
    server = OpenAICompatServer(settings.llm_local)
    tools = [LOOK_TOOL, SEARCH_IMAGE_TOOL] if search else [LOOK_TOOL]
    try:
        try:
            resolved = model or await server.default_model()
        except LocalLLMError as exc:
            print(f"error: {exc}")
            return 2
        client = server.client(resolved, system_prompt="")
        print(f"router check: {resolved} at {settings.llm_local.base_url}, "
              f"tools {', '.join(t.name for t in tools)}, {len(CASES)} messages x {repeat}")
        results: list[Result] = []
        for _ in range(repeat):
            for case in CASES:
                expected = LOOK if case.expected == SEARCH and not search else case.expected
                history = [*case.history, Message("user", case.text)]
                t0 = time.perf_counter()
                try:
                    route, _ = await client.route(history, tools)
                except LocalLLMError as exc:
                    # The request itself failed (server down, unknown model, no
                    # json_schema support); an unparsable answer is None instead.
                    print(f"error: {exc}")
                    return 2
                ms = (time.perf_counter() - t0) * 1000
                got, arg = route if route is not None else ("unusable", "")
                results.append(Result(case, expected, got, arg, ms))
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
    for name in (LOOK, SEARCH, NO_TOOL):
        group = [r for r in results if r.expected == name]
        if group:
            ok = sum(r.got == r.expected for r in group)
            print(f"  {name:<12} {ok}/{len(group)}")
    unusable = sum(r.got == "unusable" for r in results)
    times = sorted(r.ms for r in results)
    p90 = times[min(len(times) - 1, int(len(times) * 0.9))]
    ok = len(results) - len(misses)
    print(f"\naccuracy {ok}/{len(results)} ({100 * ok / len(results):.0f}%), "
          f"unusable answers {unusable}")
    print(f"router latency: median {statistics.median(times):.0f} ms, p90 {p90:.0f} ms")
    if unusable:
        print("Unusable answers: the model may spend its tokens thinking before the JSON, "
              "or the server may not support response_format json_schema. A model without "
              "thinking (or with thinking turned off in the server) usually fixes it.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m obsilord.llm.routecheck",
        description="Score a local model as the camera tool router.",
    )
    parser.add_argument("--model", default="", help="model id (default: [llm_local] model, "
                        "or the first model the server lists)")
    parser.add_argument("--no-search", action="store_true", help="offer look() only")
    parser.add_argument("--repeat", type=int, default=1, help="run the set N times (1 to 10)")
    args = parser.parse_args(argv)
    if not 1 <= args.repeat <= 10:
        parser.error("--repeat must be 1 to 10")
    return asyncio.run(_run(args.model.strip(), not args.no_search, args.repeat))


if __name__ == "__main__":
    sys.exit(main())
