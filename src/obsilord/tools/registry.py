"""The tool registry: tools are async Python functions, their declarations
come from the signature and the docstring, and the rules that limit them sit
here in one place (planning/T-tools-roadmap.md, F1 and F6).

Writing a tool:

    @tool(ToolPolicy(per_turn=3))
    async def calculate(expression: str, action: Literal["math", "convert"] = "math") -> str:
        \"\"\"One line or more of description for the model.

        Args:
            expression: What to work out.
            action: math or convert.
        \"\"\"

- Positional parameters are the model's arguments. Their hints give the JSON
  schema: str, int, float, bool, Literal[...] (an enum of strings), each
  optionally "| None". A parameter with a default is optional.
- Keyword-only parameters are not shown to the model: ctx gets the
  TurnContext, the others are filled by Tool.bind(name=value) (the camera,
  the store).
- The docstring up to "Args:" is the description, collapsed to one line;
  each "name: text" line under "Args:" (continued on indented lines)
  describes a parameter.
- The function returns a str (the output), a dict ({"output": ...} or
  {"error": ...}), or a ToolOutput (with a photo). ToolError(message) gives
  {"error": message}; any other exception gives MSG_TOOL_FAILED and a log line
  with the type name only.

A Registry holds ToolSources: async callables that give the tools for one
turn from the TurnInfo (whether the model is a cloud model, whether it sees
photos). Registry.for_turn() returns the TurnTools of one turn: the
declarations to send, and run(), which applies every policy:
- an unknown or unoffered tool is an error;
- per_turn calls per turn, shared by the tools of one group (the camera);
- after a tool with untrusted output ran, a tool that acts is refused for the
  rest of the turn (Ask arrives in T2);
- timeout_s per call;
- a cloud model is never offered a private-data tool unless its policy allows
  it (cloud=True), and cloud_visible() tells the pipeline which tool notes of
  earlier turns may go to a cloud model.

Only the tool name, the outcome and the time are logged, never arguments or
results.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
import types
import typing
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Sequence

from obsilord.llm.base import ToolCall, ToolResult, ToolSpec

log = logging.getLogger(__name__)

MSG_TOOL_FAILED = "The tool failed."
MSG_NEEDS_CONFIRM = (
    "That needs the user's confirmation after reading outside text, and asking "
    "is not available yet. Tell the user."
)

# kind is "camera" (the frame) or "search" (the crop about to be uploaded).
Snapshot = Callable[[str, bytes], Awaitable[None]]


def msg_unknown(name: str) -> str:
    return f"The tool {name} is not available."


def msg_limit(key: str, limit: int) -> str:
    return f"No more {key} calls this turn (limit {limit})."


def msg_timeout(name: str) -> str:
    return f"The {name} tool took too long."


@dataclass(frozen=True)
class ToolPolicy:
    """How a tool may be used. See the module docstring."""

    per_turn: int = 1
    group: str = ""  # tools in one group share per_turn
    face_state: str = ""  # "looking" while it runs, "" for none
    untrusted: bool = False  # output from outside the code (camera text, web, old turns)
    acts: bool = False  # changes something outside the conversation
    leaves_pc: bool = False  # sends data off this PC
    private_data: bool = False  # reads the user's private data
    cloud: bool = False  # a private-data tool that may still go to a cloud model
    timeout_s: float = 30.0


@dataclass(frozen=True)
class ToolOutput:
    """A result with a photo for a model that sees photos."""

    response: dict[str, Any]
    image: bytes | None = field(default=None, repr=False)


class ToolError(Exception):
    """A readable refusal; the message goes to the model."""


@dataclass(frozen=True)
class TurnContext:
    """What a tool gets from the running turn."""

    snapshot: Snapshot | None = None


@dataclass(frozen=True)
class TurnInfo:
    """What the tool sources may base the turn's tools on."""

    cloud: bool
    # Whether the chat model takes photos; asked only by sources that need it.
    sees_images: Callable[[], Awaitable[bool]]


_SCALARS: dict[Any, str] = {str: "string", int: "integer", float: "number", bool: "boolean"}


@dataclass(frozen=True)
class _Param:
    name: str
    kind: Any  # str, int, float, bool, or a tuple of allowed strings (Literal)
    required: bool
    default: Any


def _base_type(hint: Any, name: str) -> Any:
    """The scalar type or the Literal values of a hint, "| None" removed."""
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return _base_type(args[0], name)
    if origin is Literal:
        values = typing.get_args(hint)
        if values and all(isinstance(v, str) for v in values):
            return tuple(values)
    if hint in _SCALARS:
        return hint
    raise TypeError(f"tool parameter {name!r}: unsupported type hint {hint!r}")


def _docstring(fn: Callable[..., Any]) -> tuple[str, dict[str, str]]:
    doc = inspect.getdoc(fn)
    if not doc:
        raise TypeError(f"tool {fn.__name__!r} needs a docstring (its description)")
    head, _, args_block = doc.partition("\nArgs:")
    params: dict[str, str] = {}
    current = ""
    for line in args_block.splitlines():
        match = re.match(r"^\s{0,8}(\w+):\s*(.*)$", line)
        if match and (not current or len(line) - len(line.lstrip()) <= 4):
            current = match.group(1)
            params[current] = match.group(2).strip()
        elif current and line.strip():
            params[current] = f"{params[current]} {line.strip()}"
    return " ".join(head.split()), params


def _coerce(value: Any, kind: Any) -> Any:
    """The model's value as kind; raises ValueError when it does not fit."""
    if isinstance(kind, tuple):
        if isinstance(value, str) and value.strip() in kind:
            return value.strip()
        raise ValueError
    if kind is str:
        if isinstance(value, (dict, list)) or value is None:
            raise ValueError
        return value if isinstance(value, str) else str(value)
    if kind is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in ("true", "false"):
            return value.strip().lower() == "true"
        raise ValueError
    if isinstance(value, bool):
        raise ValueError
    if kind is int:
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"\s*-?\d+\s*", value):
            return int(value)
        raise ValueError
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)  # raises ValueError
    raise ValueError


class Tool:
    """One tool: its declaration, its policy and the function behind it."""

    def __init__(
        self,
        fn: Callable[..., Awaitable[Any]],
        policy: ToolPolicy,
        name: str | None = None,
        deps: dict[str, Any] | None = None,
    ) -> None:
        self.fn = fn
        self.policy = policy
        self.name = name or fn.__name__
        self.deps = dict(deps or {})
        hints = typing.get_type_hints(fn)
        description, arg_docs = _docstring(fn)
        params: list[_Param] = []
        self._injected: set[str] = set()
        for p in inspect.signature(fn).parameters.values():
            if p.kind is inspect.Parameter.KEYWORD_ONLY:
                self._injected.add(p.name)
                continue
            if p.kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
                raise TypeError(f"tool {self.name!r}: *args and **kwargs are not supported")
            kind = _base_type(hints.get(p.name, inspect.Parameter.empty), p.name)
            required = p.default is inspect.Parameter.empty
            params.append(_Param(p.name, kind, required, None if required else p.default))
        self._params = params
        properties: dict[str, Any] = {}
        for p in params:
            prop: dict[str, Any] = (
                {"type": "string", "enum": list(p.kind)}
                if isinstance(p.kind, tuple)
                else {"type": _SCALARS[p.kind]}
            )
            if p.name in arg_docs:
                prop["description"] = arg_docs[p.name]
            properties[p.name] = prop
        self.spec = ToolSpec(
            name=self.name,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": [p.name for p in params if p.required],
            },
        )

    def bind(self, **deps: Any) -> Tool:
        """A copy with keyword-only dependencies filled in."""
        unknown = set(deps) - (self._injected - {"ctx"})
        if unknown:
            raise TypeError(f"tool {self.name!r}: unknown dependencies {sorted(unknown)}")
        return Tool(self.fn, self.policy, self.name, {**self.deps, **deps})

    def arguments(self, args: dict[str, Any]) -> dict[str, Any]:
        """The model's arguments checked against the signature. Raises
        ToolError naming the first missing or bad one; unknown ones are
        dropped. A missing required string is "" (the tool decides what that
        means, for example look() asks its default question)."""
        out: dict[str, Any] = {}
        for p in self._params:
            value = args.get(p.name)
            if value is None or (isinstance(value, str) and not value.strip() and p.kind is not str):
                if p.required and p.kind is str:
                    out[p.name] = ""
                elif p.required:
                    raise ToolError(f"Missing argument: {p.name}.")
                else:
                    out[p.name] = p.default
                continue
            try:
                out[p.name] = _coerce(value, p.kind)
            except ValueError:
                raise ToolError(f"Bad argument: {p.name}.") from None
        return out

    async def call(self, args: dict[str, Any], ctx: TurnContext) -> Any:
        extra: dict[str, Any] = dict(self.deps)
        if "ctx" in self._injected:
            extra["ctx"] = ctx
        return await self.fn(**args, **extra)


def tool(policy: ToolPolicy, *, name: str | None = None) -> Callable[[Callable[..., Awaitable[Any]]], Tool]:
    """Decorator: the function as a Tool with this policy."""

    def wrap(fn: Callable[..., Awaitable[Any]]) -> Tool:
        return Tool(fn, policy, name)

    return wrap


ToolSource = Callable[[TurnInfo], Awaitable[Sequence[Tool]]]


def static_source(*tools: Tool) -> ToolSource:
    """A source that always gives these tools."""

    async def source(info: TurnInfo) -> Sequence[Tool]:
        return tools

    return source


def _response(value: Any) -> tuple[dict[str, Any], bytes | None]:
    if isinstance(value, ToolOutput):
        return value.response, value.image
    if isinstance(value, str):
        return {"output": value}, None
    if isinstance(value, dict) and ("output" in value or "error" in value):
        return value, None
    raise TypeError(f"a tool returned {type(value).__name__}")


class TurnTools:
    """The tools of one turn and their per-turn state."""

    def __init__(self, tools: Sequence[Tool], cloud_ok: frozenset[str]) -> None:
        self._tools = {t.name: t for t in tools}
        self._cloud_ok = cloud_ok
        self._used: Counter[str] = Counter()
        self._untrusted_ran = False
        self.specs: list[ToolSpec] = [t.spec for t in tools]

    def face_state(self, calls: Sequence[ToolCall]) -> str:
        """"looking" when one of the calls uses the camera, else ""."""
        states = {t.policy.face_state for c in calls if (t := self._tools.get(c.name)) is not None}
        return "looking" if "looking" in states else ""

    def cloud_visible(self, name: str) -> bool:
        """Whether a tool note of an earlier turn may go to a cloud model."""
        return name in self._cloud_ok

    async def run(self, call: ToolCall, ctx: TurnContext) -> ToolResult:
        t0 = time.perf_counter()
        response, image, outcome = await self._run(call, ctx)
        log.info("tool %s: %s in %d ms", call.name, outcome, (time.perf_counter() - t0) * 1000)
        return ToolResult(call, response, image=image)

    async def _run(self, call: ToolCall, ctx: TurnContext) -> tuple[dict[str, Any], bytes | None, str]:
        tool_ = self._tools.get(call.name)
        if tool_ is None:
            return {"error": msg_unknown(call.name)}, None, "unknown"
        policy = tool_.policy
        try:
            args = tool_.arguments(dict(call.args))
        except ToolError as exc:
            return {"error": str(exc)}, None, "bad arguments"
        key = policy.group or tool_.name
        if self._used[key] >= policy.per_turn:
            return {"error": msg_limit(key, policy.per_turn)}, None, "refused (limit)"
        if policy.acts and self._untrusted_ran:
            return {"error": MSG_NEEDS_CONFIRM}, None, "refused (after untrusted output)"
        self._used[key] += 1
        if policy.untrusted:
            self._untrusted_ran = True
        try:
            async with asyncio.timeout(policy.timeout_s):
                value = await tool_.call(args, ctx)
            response, image = _response(value)
        except asyncio.CancelledError:
            raise
        except ToolError as exc:
            return {"error": str(exc)}, None, "error"
        except TimeoutError:
            return {"error": msg_timeout(tool_.name)}, None, "timeout"
        except Exception as exc:  # noqa: BLE001 - a tool error is the model's to explain
            log.warning("tool %s failed: %s", tool_.name, type(exc).__name__)
            return {"error": MSG_TOOL_FAILED}, None, "failed"
        return response, image, "ok" if "output" in response else "error"


class Registry:
    """The tool sources of one connection."""

    def __init__(self, sources: Sequence[ToolSource]) -> None:
        self._sources = list(sources)

    async def for_turn(self, info: TurnInfo) -> TurnTools:
        tools: list[Tool] = []
        for source in self._sources:
            tools.extend(await source(info))
        names = Counter(t.name for t in tools)
        dupes = sorted(n for n, c in names.items() if c > 1)
        if dupes:
            raise ValueError(f"tool names used twice: {dupes}")
        cloud_ok = frozenset(t.name for t in tools if not t.policy.private_data or t.policy.cloud)
        if info.cloud:
            tools = [t for t in tools if t.name in cloud_ok]
        return TurnTools(tools, cloud_ok)


__all__ = [
    "MSG_NEEDS_CONFIRM",
    "MSG_TOOL_FAILED",
    "Registry",
    "Snapshot",
    "Tool",
    "ToolError",
    "ToolOutput",
    "ToolPolicy",
    "ToolSource",
    "TurnContext",
    "TurnInfo",
    "TurnTools",
    "msg_limit",
    "msg_timeout",
    "msg_unknown",
    "static_source",
    "tool",
]
