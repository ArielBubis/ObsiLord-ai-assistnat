"""Typed settings loaded from config.toml (tomllib) and secrets from .env.

config.toml holds non-secret settings. If it is missing, config.example.toml is
used. Secrets come from .env in the repo root, and real environment variables
override values found in .env.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin, get_type_hints

log = logging.getLogger(__name__)

# src/obsilord/config.py -> repo root is two levels above the package dir.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

ALLOWED_HOST = "127.0.0.1"

# The browser gives up on a turn after WATCHDOG_MS = 75000 in web/turn.js. The
# server turn timeout must be shorter so the client receives "Reply timed out."
# instead of its own watchdog firing. Keep the two in sync.
CLIENT_WATCHDOG_S = 75.0


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    max_output_tokens: int
    thinking_level: str
    history_turns: int
    daily_token_cap: int
    turn_timeout_s: float


@dataclass(frozen=True)
class STTSettings:
    provider: str
    model: str
    device: str
    compute_type: str
    language: str


@dataclass(frozen=True)
class TTSSettings:
    provider: str
    voice: str


@dataclass(frozen=True)
class CameraSettings:
    device_index: int
    max_edge_px: int
    warmup_frames: int
    cooldown_s: float
    max_per_hour: int


@dataclass(frozen=True)
class FaceSettings:
    logical_width: int
    logical_height: int
    mouth_thresholds: tuple[float, ...]


@dataclass(frozen=True)
class AudioSettings:
    push_to_talk_key: str


@dataclass(frozen=True)
class Secrets:
    gemini_api_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Settings:
    server: ServerSettings
    llm: LLMSettings
    stt: STTSettings
    tts: TTSSettings
    camera: CameraSettings
    face: FaceSettings
    audio: AudioSettings
    secrets: Secrets = field(default_factory=Secrets, repr=False)
    repo_root: Path = REPO_ROOT


def parse_dotenv(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Blank lines and # comments are ignored.

    An optional leading "export " is accepted. Matching single or double quotes
    around the value are stripped. Lines without "=" are ignored.
    """
    result: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        elif " #" in value:
            # Unquoted value with a trailing inline comment.
            value = value.split(" #", 1)[0].rstrip()
        result[key] = value
    return result


def load_secrets(
    env_path: Path | None = None, environ: Mapping[str, str] | None = None
) -> Secrets:
    """Load secrets from .env, with real environment variables taking precedence."""
    env_path = env_path if env_path is not None else REPO_ROOT / ".env"
    environ = environ if environ is not None else os.environ
    values: dict[str, str] = {}
    if env_path.is_file():
        values.update(parse_dotenv(env_path.read_text(encoding="utf-8")))
    values.update({k: v for k, v in environ.items() if k == "GEMINI_API_KEY"})
    key = values.get("GEMINI_API_KEY") or None
    return Secrets(gemini_api_key=key)


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", None) or str(tp)


def _check_value(value: Any, tp: Any, where: str) -> Any:
    """Check value against a field type and return it (int coerced to float).

    Supports str, int, float, and tuple[float, ...]. bool is rejected where a
    number is expected even though it subclasses int.
    """
    got = type(value).__name__
    if tp is str:
        if isinstance(value, str):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif get_origin(tp) is tuple:
        args = get_args(tp)
        if len(args) != 2 or args[1] is not Ellipsis:
            raise TypeError(f"config: unsupported field type {tp} for {where}")
        if isinstance(value, (list, tuple)):
            return tuple(
                _check_value(item, args[0], f"{where}[{i}]") for i, item in enumerate(value)
            )
        raise ValueError(f"config: {where} expected list of {_type_name(args[0])}, got {got}")
    else:
        raise TypeError(f"config: unsupported field type {tp} for {where}")
    raise ValueError(f"config: {where} expected {_type_name(tp)}, got {got}")


def _build(cls: type, section: str, data: Mapping[str, Any]) -> Any:
    raw = data.get(section)
    if not isinstance(raw, dict):
        raise ValueError(f"config: missing [{section}] section")
    hints = get_type_hints(cls)
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    missing = names - set(raw)
    if unknown:
        raise ValueError(f"config: unknown keys in [{section}]: {sorted(unknown)}")
    if missing:
        raise ValueError(f"config: missing keys in [{section}]: {sorted(missing)}")
    values = {
        name: _check_value(raw[name], hints[name], f"{section}.{name}") for name in names
    }
    return cls(**values)


def default_config_path() -> Path:
    """config.toml in the repo root, falling back to config.example.toml."""
    path = REPO_ROOT / "config.toml"
    if path.is_file():
        return path
    return REPO_ROOT / "config.example.toml"


def load_settings(
    path: Path | None = None,
    env_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load and validate settings. Raises ValueError on invalid config."""
    path = path if path is not None else default_config_path()
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    settings = Settings(
        server=_build(ServerSettings, "server", data),
        llm=_build(LLMSettings, "llm", data),
        stt=_build(STTSettings, "stt", data),
        tts=_build(TTSSettings, "tts", data),
        camera=_build(CameraSettings, "camera", data),
        face=_build(FaceSettings, "face", data),
        audio=_build(AudioSettings, "audio", data),
        secrets=load_secrets(env_path, environ),
    )
    if settings.server.host != ALLOWED_HOST:
        raise ValueError(
            f"config: server.host must be {ALLOWED_HOST}, got {settings.server.host!r}"
        )
    if not 1 <= settings.server.port <= 65535:
        raise ValueError("config: server.port must be an int in 1..65535")
    if not 0 < settings.llm.turn_timeout_s < CLIENT_WATCHDOG_S:
        raise ValueError(
            f"config: llm.turn_timeout_s must be > 0 and < {CLIENT_WATCHDOG_S:g} "
            "(the browser watchdog WATCHDOG_MS in web/turn.js), "
            f"got {settings.llm.turn_timeout_s:g}"
        )
    log.info("loaded config from %s", path)
    return settings
