"""Tools besides the camera that the model may call in a turn (recall now,
others later). Camera tools stay on the LookTool: they share its limiter and
the one-look-per-turn rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from obsilord.llm.base import ToolSpec

# Runs one call with the model's arguments. Returns {"output": ...} or
# {"error": ...}; may raise, the pipeline turns that into an error result.
ToolRun = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    run: ToolRun
