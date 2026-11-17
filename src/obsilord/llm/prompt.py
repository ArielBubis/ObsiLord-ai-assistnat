"""The assistant's system prompt, shared by every LLM implementation."""

from __future__ import annotations

from pathlib import Path

from obsilord.config import REPO_ROOT

SYSTEM_PROMPT_PATH: Path = REPO_ROOT / "prompts" / "system.md"


def load_system_prompt(path: Path | None = None) -> str:
    """Read the system prompt (prompts/system.md in the repo root by default)."""
    path = path if path is not None else SYSTEM_PROMPT_PATH
    return path.read_text(encoding="utf-8").strip()
