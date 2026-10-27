"""Typed settings loaded from config.toml (tomllib) and secrets from .env.

config.toml holds non-secret settings. If it is missing, config.example.toml is
used. Keys whose dataclass field has a default (keys added after Phase 0) may
be left out of config.toml; the default is used and one INFO line lists the
defaulted keys. Fields without a default are required. Unknown keys are
rejected, and provided values are type checked. Secrets come from .env in the repo root, and real environment variables
override values found in .env.
"""

from __future__ import annotations

import logging
import os
import tomllib
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping, get_args, get_origin, get_type_hints

from obsilord.stt.whisper_languages import WHISPER_LANGUAGE_CODES

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


# kw_only lets fields with defaults (optional in the TOML) sit anywhere.
@dataclass(frozen=True, kw_only=True)
class LLMSettings:
    provider: str
    model: str
    max_output_tokens: int
    thinking_level: str = "minimal"  # added in Phase 2
    history_turns: int
    daily_token_cap: int
    turn_timeout_s: float = 60.0  # added in Phase 2


STT_PROVIDERS = ("faster-whisper",)
STT_BEAM_SIZE_MIN = 1
STT_BEAM_SIZE_MAX = 10
STT_MAX_UTTERANCE_MIN_S = 1.0
STT_MAX_UTTERANCE_MAX_S = 120.0
STT_MIN_UTTERANCE_MIN_MS = 50
STT_MIN_UTTERANCE_MAX_MS = 5000


@dataclass(frozen=True, kw_only=True)
class STTSettings:
    provider: str
    # Model size (for example "small") or a path to a converted model
    # directory. Empty means voice input is disabled.
    model: str
    device: str
    compute_type: str
    # Spoken language code ("en"). Empty lets the model detect it per utterance.
    language: str
    # Decoding beam width, STT_BEAM_SIZE_MIN..MAX. 1 (greedy) is fastest.
    beam_size: int = 1  # added in Phase 4
    # Longest utterance kept, in seconds; audio past it is dropped.
    max_utterance_s: float = 30.0  # added in Phase 4
    # Shorter utterances are rejected without running STT.
    min_utterance_ms: int = 300  # added in Phase 4
    # Model download and cache directory, relative to the repo root (or absolute).
    download_root: str = "models"  # added in Phase 4

    @property
    def enabled(self) -> bool:
        return bool(self.model.strip())

    @property
    def language_or_none(self) -> str | None:
        """The configured language, or None to let the model detect it."""
        return self.language.strip() or None


TTS_PROVIDERS = ("piper",)
PITCH_MIN = 0.8
PITCH_MAX = 1.25
EQ_HZ_MIN = 20.0
EQ_HZ_MAX = 10000.0
EQ_DB_LIMIT = 24.0


@dataclass(frozen=True, kw_only=True)
class TTSSettings:
    provider: str
    # Path to the voice model, relative to the repo root (or absolute).
    # Empty means TTS is disabled and replies are text only.
    voice: str
    # Speaker name for multi-speaker voices, resolved at load time against the
    # voice's speaker_id_map. Empty means the voice's default speaker.
    speaker: str = ""  # added in Phase 3
    # Phoneme length scale: < 1 speaks faster, > 1 slower. Must be > 0.
    length_scale: float = 1.0  # added in Phase 3
    # Pitch factor, PITCH_MIN..PITCH_MAX. Applied server side by synthesizing
    # with length_scale * pitch and announcing native_rate * pitch.
    pitch: float = 1.0  # added in Phase 3
    # Browser playback EQ (served at /config/audio, applied by Web Audio
    # shelving filters). Frequencies EQ_HZ_MIN..EQ_HZ_MAX, gains within
    # plus or minus EQ_DB_LIMIT dB; 0 dB disables a shelf. Added in Phase 3.
    low_shelf_hz: float = 250.0
    low_shelf_db: float = -6.0
    high_shelf_hz: float = 3500.0
    high_shelf_db: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.voice.strip())


CAMERA_APIS = ("dshow", "msmf", "any")
CAMERA_MAX_EDGE_MIN_PX = 64
CAMERA_MAX_EDGE_MAX_PX = 4096
CAMERA_WARMUP_MAX = 60


@dataclass(frozen=True)
class CameraSettings:
    device_index: int
    max_edge_px: int
    warmup_frames: int
    cooldown_s: float
    max_per_hour: int
    # OpenCV capture backend, one of CAMERA_APIS: "dshow" (DirectShow),
    # "msmf" (Media Foundation, which hung on open on the dev machine), or
    # "any" (OpenCV picks).
    api: str = "dshow"  # added in Phase 5
    # Limit for opening the camera and grabbing one frame, in seconds.
    timeout_s: float = 20.0  # added in Phase 5


VISION_BACKENDS = ("vulkan", "cpu")


@dataclass(frozen=True, kw_only=True)
class VisionSettings:
    """[vision], added in Phase 5. The whole section is optional.

    Local vision model for look(): llama.cpp's llama-server, started for each
    look on 127.0.0.1 and stopped after. Paths are relative to the repo root
    (or absolute); `python -m obsilord.vision.fetch` downloads the defaults.
    An empty model disables look().
    """

    # One of VISION_BACKENDS. "vulkan" falls back to the CPU build for a look
    # whose Vulkan server fails to start.
    backend: str = "vulkan"
    server_vulkan: str = "models/llama.cpp/b11919-vulkan/llama-server.exe"
    server_cpu: str = "models/llama.cpp/b11919-cpu/llama-server.exe"
    model: str = "models/vision/qwen3-vl-2b/Qwen3VL-2B-Instruct-Q4_K_M.gguf"
    mmproj: str = "models/vision/qwen3-vl-2b/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf"
    # Context size in tokens: the image (about 350 to 450 tokens at 640 to
    # 768 px), the prompt, and the answer.
    ctx: int = 2048
    # Longest answer, in tokens.
    max_tokens: int = 120
    # On the CPU build the frame is shrunk to this edge first (about half the
    # time of 768 px in the bench).
    cpu_max_edge_px: int = 512
    # From process start until the server answers /health.
    start_timeout_s: float = 30.0
    # One describe request.
    request_timeout_s: float = 30.0

    @property
    def enabled(self) -> bool:
        return bool(self.model.strip())


FACE_SKINS = ("dither", "code")


@dataclass(frozen=True, kw_only=True)
class FaceSettings:
    logical_width: int
    logical_height: int
    mouth_thresholds: tuple[float, ...]
    # One of FACE_SKINS: "dither" (3D face, WebGL) or "code" (8-bit face).
    skin: str = "dither"  # added in Phase 8
    # Character file (.face.glb, .glb or .vrm), relative to the repo root (or
    # absolute). Checked by app startup. Empty means the bundled character.
    character: str = ""  # added in Phase 8


@dataclass(frozen=True)
class AudioSettings:
    push_to_talk_key: str


@dataclass(frozen=True, kw_only=True)
class DebugSettings:
    """[debug], optional."""

    # Print every Gemini and llama-server request and reply to the console
    # (see obsilord.promptlog). Camera frames show as their size only.
    log_prompts: bool = False


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
    vision: VisionSettings = field(default_factory=VisionSettings)
    debug: DebugSettings = field(default_factory=DebugSettings)
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

    Supports str, bool, int, float, and tuple[float, ...]. bool is rejected
    where a number is expected even though it subclasses int, and a number is
    rejected where a bool is expected.
    """
    got = type(value).__name__
    if tp is str:
        if isinstance(value, str):
            return value
    elif tp is bool:
        if isinstance(value, bool):
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


def _has_default(f: Any) -> bool:
    return f.default is not MISSING or f.default_factory is not MISSING


def _build(
    cls: type,
    section: str,
    data: Mapping[str, Any],
    defaulted: list[str] | None = None,
    optional: bool = False,
) -> Any:
    """Build a settings dataclass from one TOML section.

    Fields with a default may be omitted (their names, as section.key, are
    appended to defaulted); fields without one are required. optional: the
    section itself may be missing (all its fields must have defaults).
    """
    raw = data.get(section)
    if raw is None and optional:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"config: missing [{section}] section")
    hints = get_type_hints(cls)
    all_fields = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(all_fields)
    if unknown:
        raise ValueError(f"config: unknown keys in [{section}]: {sorted(unknown)}")
    absent = set(all_fields) - set(raw)
    missing = sorted(n for n in absent if not _has_default(all_fields[n]))
    if missing:
        raise ValueError(f"config: missing keys in [{section}]: {missing}")
    if defaulted is not None:
        defaulted.extend(f"{section}.{n}" for n in sorted(absent))
    values = {
        name: _check_value(raw[name], hints[name], f"{section}.{name}")
        for name in all_fields
        if name in raw
    }
    return cls(**values)


def _check_tts_voice_controls(tts: TTSSettings) -> None:
    """Pitch in [PITCH_MIN, PITCH_MAX], EQ frequencies and gains in range."""
    if not PITCH_MIN <= tts.pitch <= PITCH_MAX:
        raise ValueError(
            f"config: tts.pitch must be in {PITCH_MIN:g}..{PITCH_MAX:g}, got {tts.pitch:g}"
        )
    for name in ("low_shelf_hz", "high_shelf_hz"):
        value = getattr(tts, name)
        if not EQ_HZ_MIN <= value <= EQ_HZ_MAX:
            raise ValueError(
                f"config: tts.{name} must be in {EQ_HZ_MIN:g}..{EQ_HZ_MAX:g} Hz, got {value:g}"
            )
    for name in ("low_shelf_db", "high_shelf_db"):
        value = getattr(tts, name)
        if not -EQ_DB_LIMIT <= value <= EQ_DB_LIMIT:
            raise ValueError(
                f"config: tts.{name} must be in -{EQ_DB_LIMIT:g}..{EQ_DB_LIMIT:g} dB, "
                f"got {value:g}"
            )


def _check_stt(stt: STTSettings) -> None:
    """Range checks for the [stt] keys. Provider and download_root only matter
    when voice input is enabled."""
    if not STT_BEAM_SIZE_MIN <= stt.beam_size <= STT_BEAM_SIZE_MAX:
        raise ValueError(
            f"config: stt.beam_size must be in {STT_BEAM_SIZE_MIN}..{STT_BEAM_SIZE_MAX}, "
            f"got {stt.beam_size}"
        )
    if not STT_MAX_UTTERANCE_MIN_S <= stt.max_utterance_s <= STT_MAX_UTTERANCE_MAX_S:
        raise ValueError(
            f"config: stt.max_utterance_s must be in "
            f"{STT_MAX_UTTERANCE_MIN_S:g}..{STT_MAX_UTTERANCE_MAX_S:g}, "
            f"got {stt.max_utterance_s:g}"
        )
    if not STT_MIN_UTTERANCE_MIN_MS <= stt.min_utterance_ms <= STT_MIN_UTTERANCE_MAX_MS:
        raise ValueError(
            f"config: stt.min_utterance_ms must be in "
            f"{STT_MIN_UTTERANCE_MIN_MS}..{STT_MIN_UTTERANCE_MAX_MS}, "
            f"got {stt.min_utterance_ms}"
        )
    if stt.min_utterance_ms >= stt.max_utterance_s * 1000:
        raise ValueError(
            "config: stt.min_utterance_ms must be shorter than stt.max_utterance_s, "
            f"got {stt.min_utterance_ms} ms and {stt.max_utterance_s:g} s"
        )
    if not stt.enabled:
        return
    if stt.provider not in STT_PROVIDERS:
        raise ValueError(
            f"config: stt.provider must be one of {list(STT_PROVIDERS)}, "
            f"got {stt.provider!r}"
        )
    if not stt.download_root.strip():
        raise ValueError(
            'config: stt.download_root must not be empty (the default is "models")'
        )
    language = stt.language_or_none
    if language is not None and language not in WHISPER_LANGUAGE_CODES:
        raise ValueError(
            f'config: stt.language must be "" (detect) or a Whisper language code '
            f'such as "en" or "he", got {stt.language!r}'
        )


MOUTH_THRESHOLD_COUNT = 3


def _check_mouth_thresholds(values: tuple[float, ...]) -> None:
    """Exactly 3 RMS cutoffs (mouth frames 1, 2, 3), each in (0, 1], strictly increasing."""
    if len(values) != MOUTH_THRESHOLD_COUNT:
        raise ValueError(
            f"config: face.mouth_thresholds must have exactly {MOUTH_THRESHOLD_COUNT} "
            f"values, got {len(values)}"
        )
    if not all(0 < v <= 1 for v in values):
        raise ValueError("config: face.mouth_thresholds values must each be in (0, 1]")
    if not all(a < b for a, b in zip(values, values[1:])):
        raise ValueError("config: face.mouth_thresholds must be strictly increasing")


def _check_face_skin(face: FaceSettings) -> None:
    """face.skin is one of FACE_SKINS. The character file is checked by app startup."""
    if face.skin not in FACE_SKINS:
        raise ValueError(
            f"config: face.skin must be one of {list(FACE_SKINS)}, got {face.skin!r}"
        )


def _check_camera(camera: CameraSettings) -> None:
    """Range checks for the [camera] keys."""
    if camera.api not in CAMERA_APIS:
        raise ValueError(
            f"config: camera.api must be one of {list(CAMERA_APIS)}, got {camera.api!r}"
        )
    if camera.device_index < 0:
        raise ValueError(f"config: camera.device_index must be >= 0, got {camera.device_index}")
    if not CAMERA_MAX_EDGE_MIN_PX <= camera.max_edge_px <= CAMERA_MAX_EDGE_MAX_PX:
        raise ValueError(
            f"config: camera.max_edge_px must be in "
            f"{CAMERA_MAX_EDGE_MIN_PX}..{CAMERA_MAX_EDGE_MAX_PX}, got {camera.max_edge_px}"
        )
    if not 0 <= camera.warmup_frames <= CAMERA_WARMUP_MAX:
        raise ValueError(
            f"config: camera.warmup_frames must be in 0..{CAMERA_WARMUP_MAX}, "
            f"got {camera.warmup_frames}"
        )
    if camera.cooldown_s < 0:
        raise ValueError(f"config: camera.cooldown_s must be >= 0, got {camera.cooldown_s:g}")
    if camera.max_per_hour < 1:
        raise ValueError(
            "config: camera.max_per_hour must be >= 1 (set [vision] model = \"\" to "
            f"turn look() off), got {camera.max_per_hour}"
        )
    if not camera.timeout_s > 0:
        raise ValueError(f"config: camera.timeout_s must be > 0, got {camera.timeout_s:g}")


def _check_vision(vision: VisionSettings) -> None:
    """Range checks for the [vision] keys. Files are checked by app startup."""
    if vision.backend not in VISION_BACKENDS:
        raise ValueError(
            f"config: vision.backend must be one of {list(VISION_BACKENDS)}, "
            f"got {vision.backend!r}"
        )
    for name in ("ctx", "max_tokens"):
        if getattr(vision, name) < 1:
            raise ValueError(f"config: vision.{name} must be >= 1, got {getattr(vision, name)}")
    if not CAMERA_MAX_EDGE_MIN_PX <= vision.cpu_max_edge_px <= CAMERA_MAX_EDGE_MAX_PX:
        raise ValueError(
            f"config: vision.cpu_max_edge_px must be in "
            f"{CAMERA_MAX_EDGE_MIN_PX}..{CAMERA_MAX_EDGE_MAX_PX}, got {vision.cpu_max_edge_px}"
        )
    for name in ("start_timeout_s", "request_timeout_s"):
        if not getattr(vision, name) > 0:
            raise ValueError(f"config: vision.{name} must be > 0, got {getattr(vision, name):g}")


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

    defaulted: list[str] = []
    settings = Settings(
        server=_build(ServerSettings, "server", data, defaulted),
        llm=_build(LLMSettings, "llm", data, defaulted),
        stt=_build(STTSettings, "stt", data, defaulted),
        tts=_build(TTSSettings, "tts", data, defaulted),
        camera=_build(CameraSettings, "camera", data, defaulted),
        face=_build(FaceSettings, "face", data, defaulted),
        audio=_build(AudioSettings, "audio", data, defaulted),
        vision=_build(VisionSettings, "vision", data, defaulted, optional=True),
        debug=_build(DebugSettings, "debug", data, defaulted, optional=True),
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
    if not settings.tts.length_scale > 0:
        raise ValueError(
            f"config: tts.length_scale must be > 0, got {settings.tts.length_scale:g}"
        )
    if settings.tts.enabled and settings.tts.provider not in TTS_PROVIDERS:
        raise ValueError(
            f"config: tts.provider must be one of {list(TTS_PROVIDERS)}, "
            f"got {settings.tts.provider!r}"
        )
    _check_tts_voice_controls(settings.tts)
    _check_stt(settings.stt)
    _check_mouth_thresholds(settings.face.mouth_thresholds)
    _check_face_skin(settings.face)
    _check_camera(settings.camera)
    _check_vision(settings.vision)
    if defaulted:
        log.info(
            "config: %s not set in %s, using defaults (copy them from "
            "config.example.toml to silence this): %s",
            "key" if len(defaulted) == 1 else "keys",
            path.name,
            ", ".join(defaulted),
        )
    log.info("loaded config from %s", path)
    return settings
