"""Typed settings loaded from config.toml (tomllib) and secrets from .env.

config.toml holds non-secret settings. If it is missing, config.example.toml is
used. Keys whose dataclass field has a default may be left out of
config.toml; the default is used and one INFO line lists the defaulted keys.
Fields without a default are required. Unknown sections and unknown keys are
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
LLM_PROVIDERS = ("gemini", "local")
TOOL_MODES = ("auto", "native", "router")
# OpenAI-style reasoning_effort values; LM Studio listed these in its 400 for
# "off" (checked 2026-11-18). "" means the field is not sent.
REASONING_EFFORTS = ("", "none", "minimal", "low", "medium", "high", "xhigh")


@dataclass(frozen=True, kw_only=True)
class LLMSettings:
    # One of LLM_PROVIDERS: the provider a new page connection starts with. The
    # page's Settings panel can switch each connection between the two. Local
    # is the main one; Gemini is the opt-in alternative.
    provider: str = "local"
    # The Gemini model (also used by [image_search] provider "gemini").
    model: str
    max_output_tokens: int
    thinking_level: str = "minimal"
    # Sampling temperature for Gemini replies, 0..2. 1.0 is the Gemini 3 default.
    temperature: float = 1.0
    history_turns: int
    daily_token_cap: int
    turn_timeout_s: float = 60.0


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
    beam_size: int = 1
    # Longest utterance kept, in seconds; audio past it is dropped.
    max_utterance_s: float = 30.0
    # Shorter utterances are rejected without running STT.
    min_utterance_ms: int = 300
    # Model download and cache directory, relative to the repo root (or absolute).
    download_root: str = "models"

    @property
    def enabled(self) -> bool:
        return bool(self.model.strip())

    @property
    def language_or_none(self) -> str | None:
        """The configured language, or None to let the model detect it."""
        return self.language.strip() or None


TTS_PROVIDERS = ("kokoro",)
# Kokoro synthesizes at speed / pitch; kokoro-onnx 0.6.1 create() refuses
# speeds outside this range.
KOKORO_SPEED_MIN = 0.5
KOKORO_SPEED_MAX = 2.0
PITCH_MIN = 0.8
PITCH_MAX = 1.25
EQ_HZ_MIN = 20.0
EQ_HZ_MAX = 10000.0
EQ_DB_LIMIT = 24.0


@dataclass(frozen=True, kw_only=True)
class TTSSettings:
    provider: str
    # Path to the Kokoro model, relative to the repo root (or absolute); the
    # voices file sits next to it. Empty means TTS is disabled (text only).
    model: str
    # Kokoro voice name, checked at load time against the voices file.
    voice: str = "bm_lewis"
    # Speaking rate: 1.0 normal, > 1 faster. speed / pitch must stay in
    # KOKORO_SPEED_MIN..KOKORO_SPEED_MAX.
    speed: float = 1.15
    # Pitch factor, PITCH_MIN..PITCH_MAX. Applied server side by synthesizing
    # at speed / pitch and announcing native_rate * pitch.
    pitch: float = 1.0
    # Browser playback EQ (served at /config/audio, applied by Web Audio
    # shelving filters). Frequencies EQ_HZ_MIN..EQ_HZ_MAX, gains within
    # plus or minus EQ_DB_LIMIT dB; 0 dB disables a shelf.
    low_shelf_hz: float = 250.0
    low_shelf_db: float = -6.0
    high_shelf_hz: float = 3500.0
    high_shelf_db: float = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.model.strip())


CAMERA_APIS = ("dshow", "msmf", "any")
# The Photos setting (obsilord.photos.PHOTO_MODES, repeated here so config
# imports nothing from the app).
PHOTO_MODES = ("off", "followup", "disk")
PHOTOS_KEEP_MAX = 10
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
    api: str = "dshow"
    # Limit for opening the camera and grabbing one frame, in seconds.
    timeout_s: float = 20.0
    # Photos setting default, one of PHOTO_MODES (see obsilord.photos); the
    # page's choice wins.
    photos: str = "off"
    photos_keep: int = 2
    # Relative to the repo root unless absolute. Written only in "disk" mode.
    photos_dir: str = "data/photos"


VISION_BACKENDS = ("cuda", "cpu")


@dataclass(frozen=True, kw_only=True)
class VisionSettings:
    """[vision]. The whole section is optional.

    Local vision model for look(): llama.cpp's llama-server, started for each
    look on 127.0.0.1 and stopped after. Paths are relative to the repo root
    (or absolute, for example a shared model folder such as LM Studio's);
    `python -m obsilord.vision.fetch` downloads the defaults.
    An empty model disables look().
    """

    # One of VISION_BACKENDS. "cuda" (NVIDIA GPU) falls back to the CPU build
    # for a look whose CUDA server fails to start.
    backend: str = "cuda"
    server_cuda: str = "models/llama.cpp/b11919-cuda/llama-server.exe"
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
    skin: str = "dither"
    # Character file (.face.glb, .glb or .vrm), relative to the repo root (or
    # absolute). Checked by app startup. Empty means the bundled character.
    character: str = ""
    # Show only the face, subtitles, chat row and error line.
    demo: bool = False


IMAGE_SEARCH_PROVIDERS = ("gemini", "google_vision")


@dataclass(frozen=True, kw_only=True)
class ImageSearchSettings:
    """[image_search], optional, off by default.

    The search_image() tool: the local vision model finds the target in one
    webcam frame, the app crops it, and the crop is uploaded to be identified.
    This is the only place a piece of a camera frame leaves the PC. Needs
    [vision], and the API key of the provider in .env.
    """

    enabled: bool = False
    # One of IMAGE_SEARCH_PROVIDERS. "gemini": the [llm] model names what the
    # crop shows (GEMINI_API_KEY, tokens count toward the daily cap).
    # "google_vision": Cloud Vision web detection (GOOGLE_VISION_API_KEY).
    provider: str = "gemini"
    # Cost guards: uploads per rolling hour (in memory) and per calendar month
    # (counted from logs/image_search.jsonl).
    max_per_hour: int = 10
    max_per_month: int = 900
    # A box covering more of the frame than this is not uploaded: it usually
    # means the model did not find the target and boxed the whole view.
    max_crop_fraction: float = 0.8
    # Results kept per kind (best guesses, related names, page titles).
    max_results: int = 5
    # One upload and its answer.
    timeout_s: float = 15.0


LOCAL_ROUTER_CONTEXT_MAX = 20


@dataclass(frozen=True, kw_only=True)
class LocalLLMSettings:
    """[llm_local], optional. A local OpenAI-compatible server (LM Studio,
    Ollama, llama-server) used when the page selects the "local" provider.

    base_url is only read from here: the page can pick a model but never the
    address the server talks to.
    """

    # Ends with /v1. LM Studio's default port is 1234, Ollama's 11434.
    base_url: str = "http://127.0.0.1:1234/v1"
    # Model id as the server lists it. Empty: the first model the server lists.
    model: str = ""
    # Reasoning models spend part of this on thinking before the reply.
    max_output_tokens: int = 400
    temperature: float = 0.7
    # Limit for one request (a model that is not loaded yet loads first).
    request_timeout_s: float = 60.0
    # Recent messages the tool router sees besides the new one (0..20).
    router_context_messages: int = 4
    # "native": tools declared like for Gemini; "router": a routing request
    # before each reply; "auto": per model from the server's model list.
    tool_mode: str = "auto"
    # Sent as reasoning_effort with every chat request. "none" turns thinking
    # off: spoken replies are short, and thinking used up max_output_tokens
    # before any text. "" leaves it to the server.
    reasoning_effort: str = "none"


RECALL_LIMIT_MAX = 50
RECALL_MIN_CHARS = 200
DEMO_DB_PATH = "data/memory-demo.sqlite3"


@dataclass(frozen=True, kw_only=True)
class MemorySettings:
    """[memory], optional. The local conversation store and recall()."""

    enabled: bool = True
    # Relative to the repo root unless absolute.
    db_path: str = "data/memory.sqlite3"
    recall: bool = True
    recall_limit: int = 5
    recall_max_chars: int = 1500
    # A reload continues the last conversation within this many minutes.
    resume_minutes: int = 30
    # 0 keeps everything.
    retention_days: int = 0
    list_limit: int = 50


MAX_ROUNDS_MAX = 6


@dataclass(frozen=True, kw_only=True)
class ToolsSettings:
    """[tools], optional. How the model uses its tools."""

    # Tool rounds per turn: the model may call tools, read the results and
    # call again, up to this many times before it must answer (1..6). Each
    # round is one more model request, checked against the daily token cap.
    max_rounds: int = 3


@dataclass(frozen=True, kw_only=True)
class DebugSettings:
    """[debug], optional."""

    # Print every Gemini and llama-server request and reply to the console
    # (see obsilord.promptlog). Camera frames show as their size only.
    log_prompts: bool = False
    # Made-up history in DEMO_DB_PATH, reseeded at every start, instead of
    # the real store (see obsilord.memory.demo).
    demo_memory: bool = False


@dataclass(frozen=True)
class Secrets:
    gemini_api_key: str | None = field(default=None, repr=False)
    google_vision_api_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class Settings:
    server: ServerSettings
    llm: LLMSettings
    stt: STTSettings
    tts: TTSSettings
    camera: CameraSettings
    face: FaceSettings
    vision: VisionSettings = field(default_factory=VisionSettings)
    image_search: ImageSearchSettings = field(default_factory=ImageSearchSettings)
    llm_local: LocalLLMSettings = field(default_factory=LocalLLMSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    tools: ToolsSettings = field(default_factory=ToolsSettings)
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


SECRET_KEYS = ("GEMINI_API_KEY", "GOOGLE_VISION_API_KEY")


def load_secrets(
    env_path: Path | None = None, environ: Mapping[str, str] | None = None
) -> Secrets:
    """Load secrets from .env, with real environment variables taking precedence."""
    env_path = env_path if env_path is not None else REPO_ROOT / ".env"
    environ = environ if environ is not None else os.environ
    values: dict[str, str] = {}
    if env_path.is_file():
        values.update(parse_dotenv(env_path.read_text(encoding="utf-8")))
    values.update({k: v for k, v in environ.items() if k in SECRET_KEYS})
    return Secrets(
        gemini_api_key=values.get("GEMINI_API_KEY") or None,
        google_vision_api_key=values.get("GOOGLE_VISION_API_KEY") or None,
    )


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
    """Pitch in [PITCH_MIN, PITCH_MAX], speed / pitch in the Kokoro range, EQ
    frequencies and gains in range."""
    if not PITCH_MIN <= tts.pitch <= PITCH_MAX:
        raise ValueError(
            f"config: tts.pitch must be in {PITCH_MIN:g}..{PITCH_MAX:g}, got {tts.pitch:g}"
        )
    effective = tts.speed / tts.pitch
    if not KOKORO_SPEED_MIN <= effective <= KOKORO_SPEED_MAX:
        raise ValueError(
            "config: tts.speed / tts.pitch must be in "
            f"{KOKORO_SPEED_MIN:g}..{KOKORO_SPEED_MAX:g}, got {effective:.3g}"
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
    if camera.photos not in PHOTO_MODES:
        raise ValueError(
            f"config: camera.photos must be one of {list(PHOTO_MODES)}, got {camera.photos!r}"
        )
    if not 0 <= camera.photos_keep <= PHOTOS_KEEP_MAX:
        raise ValueError(
            f"config: camera.photos_keep must be in 0..{PHOTOS_KEEP_MAX}, got {camera.photos_keep}"
        )
    if not camera.photos_dir.strip():
        raise ValueError("config: camera.photos_dir must not be empty")


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


def _check_image_search(search: ImageSearchSettings, vision: VisionSettings) -> None:
    if search.provider not in IMAGE_SEARCH_PROVIDERS:
        raise ValueError(
            f"config: image_search.provider must be one of {list(IMAGE_SEARCH_PROVIDERS)}, "
            f"got {search.provider!r}"
        )
    for key in ("max_per_hour", "max_per_month", "max_results"):
        if getattr(search, key) < 1:
            raise ValueError(
                f"config: image_search.{key} must be >= 1, got {getattr(search, key)} "
                "(set image_search.enabled = false to turn the search off)"
            )
    if not 0 < search.max_crop_fraction <= 1:
        raise ValueError(
            "config: image_search.max_crop_fraction must be > 0 and <= 1, "
            f"got {search.max_crop_fraction:g}"
        )
    if not search.timeout_s > 0:
        raise ValueError(
            f"config: image_search.timeout_s must be > 0, got {search.timeout_s:g}"
        )
    if search.enabled and not vision.enabled:
        raise ValueError(
            "config: image_search.enabled needs the local vision model "
            '([vision] model is empty)'
        )


def _check_llm_local(llm: LLMSettings, local: LocalLLMSettings) -> None:
    if llm.provider not in LLM_PROVIDERS:
        raise ValueError(
            f"config: llm.provider must be one of {list(LLM_PROVIDERS)}, got {llm.provider!r}"
        )
    url = local.base_url.strip()
    if not url.startswith(("http://", "https://")) or url != local.base_url:
        raise ValueError(
            "config: llm_local.base_url must be an http:// or https:// URL without "
            f"surrounding spaces, got {local.base_url!r}"
        )
    if local.max_output_tokens < 1:
        raise ValueError(
            f"config: llm_local.max_output_tokens must be >= 1, got {local.max_output_tokens}"
        )
    if not 0 <= local.temperature <= 2:
        raise ValueError(
            f"config: llm_local.temperature must be in 0..2, got {local.temperature:g}"
        )
    if not 0 < local.request_timeout_s < CLIENT_WATCHDOG_S:
        raise ValueError(
            f"config: llm_local.request_timeout_s must be > 0 and < {CLIENT_WATCHDOG_S:g}, "
            f"got {local.request_timeout_s:g}"
        )
    if not 0 <= local.router_context_messages <= LOCAL_ROUTER_CONTEXT_MAX:
        raise ValueError(
            f"config: llm_local.router_context_messages must be in "
            f"0..{LOCAL_ROUTER_CONTEXT_MAX}, got {local.router_context_messages}"
        )
    if local.tool_mode not in TOOL_MODES:
        raise ValueError(
            f"config: llm_local.tool_mode must be one of {list(TOOL_MODES)}, got {local.tool_mode!r}"
        )
    if local.reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(
            f"config: llm_local.reasoning_effort must be one of {list(REASONING_EFFORTS)}, "
            f"got {local.reasoning_effort!r}"
        )


def _check_memory(memory: MemorySettings) -> None:
    if not memory.db_path.strip():
        raise ValueError("config: memory.db_path must not be empty")
    if Path(memory.db_path) == Path(DEMO_DB_PATH):
        raise ValueError(
            f"config: memory.db_path must not be {DEMO_DB_PATH} (the demo file is deleted at "
            "every start with [debug] demo_memory)"
        )
    if not 1 <= memory.recall_limit <= RECALL_LIMIT_MAX:
        raise ValueError(
            f"config: memory.recall_limit must be in 1..{RECALL_LIMIT_MAX}, "
            f"got {memory.recall_limit}"
        )
    if memory.recall_max_chars < RECALL_MIN_CHARS:
        raise ValueError(
            f"config: memory.recall_max_chars must be >= {RECALL_MIN_CHARS}, "
            f"got {memory.recall_max_chars}"
        )
    for name in ("resume_minutes", "retention_days"):
        if getattr(memory, name) < 0:
            raise ValueError(f"config: memory.{name} must be >= 0, got {getattr(memory, name)}")
    if memory.list_limit < 1:
        raise ValueError(f"config: memory.list_limit must be >= 1, got {memory.list_limit}")


def photos_dir_path(settings: Settings) -> Path:
    """[camera] photos_dir, relative paths against the repo root."""
    path = Path(settings.camera.photos_dir)
    return path if path.is_absolute() else settings.repo_root / path


def memory_db_path(settings: Settings) -> Path:
    """The store file: the demo file with [debug] demo_memory, otherwise
    [memory] db_path (relative paths against the repo root)."""
    raw = DEMO_DB_PATH if settings.debug.demo_memory else settings.memory.db_path
    path = Path(raw)
    return path if path.is_absolute() else settings.repo_root / path


def default_config_path() -> Path:
    """config.toml in the repo root, falling back to config.example.toml."""
    path = REPO_ROOT / "config.toml"
    if path.is_file():
        return path
    return REPO_ROOT / "config.example.toml"


# Every top-level table config.toml may have.
SECTIONS = (
    "server", "llm", "stt", "tts", "camera", "face", "vision", "image_search",
    "llm_local", "memory", "tools", "debug",
)


def load_settings(
    path: Path | None = None,
    env_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Settings:
    """Load and validate settings. Raises ValueError on invalid config."""
    path = path if path is not None else default_config_path()
    with path.open("rb") as fh:
        data = tomllib.load(fh)

    unknown = sorted(set(data) - set(SECTIONS))
    if unknown:
        raise ValueError(f"config: unknown sections: {unknown}")
    defaulted: list[str] = []
    settings = Settings(
        server=_build(ServerSettings, "server", data, defaulted),
        llm=_build(LLMSettings, "llm", data, defaulted),
        stt=_build(STTSettings, "stt", data, defaulted),
        tts=_build(TTSSettings, "tts", data, defaulted),
        camera=_build(CameraSettings, "camera", data, defaulted),
        face=_build(FaceSettings, "face", data, defaulted),
        vision=_build(VisionSettings, "vision", data, defaulted, optional=True),
        image_search=_build(
            ImageSearchSettings, "image_search", data, defaulted, optional=True
        ),
        llm_local=_build(LocalLLMSettings, "llm_local", data, defaulted, optional=True),
        memory=_build(MemorySettings, "memory", data, defaulted, optional=True),
        tools=_build(ToolsSettings, "tools", data, defaulted, optional=True),
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
    _check_image_search(settings.image_search, settings.vision)
    _check_llm_local(settings.llm, settings.llm_local)
    _check_memory(settings.memory)
    if not 1 <= settings.tools.max_rounds <= MAX_ROUNDS_MAX:
        raise ValueError(
            f"config: tools.max_rounds must be in 1..{MAX_ROUNDS_MAX}, "
            f"got {settings.tools.max_rounds}"
        )
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
