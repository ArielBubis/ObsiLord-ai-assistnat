import logging
from pathlib import Path

import pytest

from obsilord.config import (
    REPO_ROOT,
    Secrets,
    Settings,
    VisionSettings,
    load_secrets,
    load_settings,
    parse_dotenv,
)

EXAMPLE = REPO_ROOT / "config.example.toml"


def _write_config(tmp_path: Path, host: str) -> Path:
    text = EXAMPLE.read_text(encoding="utf-8").replace(
        'host = "127.0.0.1"', f'host = "{host}"'
    )
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_loads_example_config_with_types(tmp_path: Path) -> None:
    s = load_settings(EXAMPLE, env_path=tmp_path / "missing.env", environ={})
    assert isinstance(s, Settings)
    assert s.server.host == "127.0.0.1"
    assert s.server.port == 8765
    assert s.llm.provider == "gemini"
    assert isinstance(s.llm.daily_token_cap, int)
    assert s.llm.turn_timeout_s == 60.0 and type(s.llm.turn_timeout_s) is float
    assert s.stt.compute_type == "int8"
    assert s.tts.provider == "piper"
    assert s.camera.max_per_hour == 30
    assert s.face.logical_width == 64 and s.face.logical_height == 48
    assert s.face.mouth_thresholds == (0.05, 0.15, 0.30)
    assert s.audio.push_to_talk_key == "Space"
    assert s.secrets.gemini_api_key is None


def test_rejects_non_loopback_host(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "0.0.0.0")
    with pytest.raises(ValueError, match="127.0.0.1"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        EXAMPLE.read_text(encoding="utf-8") + "\n[extra]\n", encoding="utf-8"
    )
    # Unknown top-level sections are ignored; unknown keys inside a section are not.
    load_settings(path, env_path=tmp_path / "missing.env", environ={})
    bad = EXAMPLE.read_text(encoding="utf-8").replace(
        "port = 8765", "port = 8765\nbogus = 1"
    )
    path.write_text(bad, encoding="utf-8")
    with pytest.raises(ValueError, match="bogus"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_parse_dotenv_comments_and_quotes() -> None:
    text = (
        "# a comment\n"
        "\n"
        "PLAIN=abc\n"
        'DOUBLE="with spaces"\n'
        "SINGLE='single'\n"
        "  SPACED  =  value  \n"
        "export EXPORTED=yes\n"
        "INLINE=val # trailing comment\n"
        "EMPTY=\n"
        "not a pair\n"
    )
    assert parse_dotenv(text) == {
        "PLAIN": "abc",
        "DOUBLE": "with spaces",
        "SINGLE": "single",
        "SPACED": "value",
        "EXPORTED": "yes",
        "INLINE": "val",
        "EMPTY": "",
    }


def test_secrets_from_dotenv(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('# key\nGEMINI_API_KEY="from-file"\n', encoding="utf-8")
    assert load_secrets(env, environ={}).gemini_api_key == "from-file"


def test_env_var_overrides_dotenv(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=from-file\n", encoding="utf-8")
    secrets = load_secrets(env, environ={"GEMINI_API_KEY": "from-env"})
    assert secrets.gemini_api_key == "from-env"


def test_empty_key_is_none(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=\n", encoding="utf-8")
    assert load_secrets(env, environ={}).gemini_api_key is None


def test_api_key_not_in_repr(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("GEMINI_API_KEY=super-secret-value\n", encoding="utf-8")
    s = load_settings(EXAMPLE, env_path=env, environ={})
    assert s.secrets.gemini_api_key == "super-secret-value"
    assert "super-secret-value" not in repr(s)
    assert "super-secret-value" not in repr(s.secrets)
    assert "super-secret-value" not in str(Secrets(gemini_api_key="super-secret-value"))


def _write_replaced(tmp_path: Path, old: str, new: str) -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "config.toml"
    path.write_text(text.replace(old, new), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("value", "got"), [("true", "bool"), ('"8765"', "str"), ("8765.0", "float")]
)
def test_rejects_wrong_port_type(tmp_path: Path, value: str, got: str) -> None:
    path = _write_replaced(tmp_path, "port = 8765", f"port = {value}")
    with pytest.raises(ValueError, match=rf"server\.port expected int, got {got}"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_int_coerced_to_float(tmp_path: Path) -> None:
    s = load_settings(EXAMPLE, env_path=tmp_path / "missing.env", environ={})
    assert s.camera.cooldown_s == 10.0
    assert type(s.camera.cooldown_s) is float


def test_rejects_non_numeric_mouth_threshold(tmp_path: Path) -> None:
    path = _write_replaced(
        tmp_path, "mouth_thresholds = [0.05, 0.15, 0.30]", 'mouth_thresholds = [0.05, "x", 0.30]'
    )
    with pytest.raises(ValueError, match=r"face\.mouth_thresholds\[1\] expected float, got str"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_rejects_bool_in_float_list_and_wrong_str(tmp_path: Path) -> None:
    path = _write_replaced(
        tmp_path, "mouth_thresholds = [0.05, 0.15, 0.30]", "mouth_thresholds = [true]"
    )
    with pytest.raises(ValueError, match=r"face\.mouth_thresholds\[0\] expected float, got bool"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})
    path = _write_replaced(tmp_path, 'language = "en"', "language = 1")
    with pytest.raises(ValueError, match=r"stt\.language expected str, got int"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


@pytest.mark.parametrize("value", ["0", "-5", "75", "75.0", "120"])
def test_rejects_bad_turn_timeout(tmp_path: Path, value: str) -> None:
    path = _write_replaced(tmp_path, "turn_timeout_s = 60", f"turn_timeout_s = {value}")
    with pytest.raises(ValueError, match=r"llm\.turn_timeout_s must be > 0 and < 75.*WATCHDOG_MS"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_accepts_turn_timeout_just_under_watchdog(tmp_path: Path) -> None:
    path = _write_replaced(tmp_path, "turn_timeout_s = 60", "turn_timeout_s = 74.5")
    s = load_settings(path, env_path=tmp_path / "missing.env", environ={})
    assert s.llm.turn_timeout_s == 74.5


def test_tts_settings_from_example(tmp_path: Path) -> None:
    s = load_settings(EXAMPLE, env_path=tmp_path / "missing.env", environ={})
    assert s.tts.provider == "piper"
    assert s.tts.voice == "voices/en_GB-semaine-medium.onnx"
    assert s.tts.speaker == "obadiah"
    assert s.tts.length_scale == 1.0 and type(s.tts.length_scale) is float
    assert s.tts.enabled


@pytest.mark.parametrize("value", ["0", "0.0", "-1.5"])
def test_rejects_non_positive_length_scale(tmp_path: Path, value: str) -> None:
    path = _write_replaced(tmp_path, "length_scale = 1.0", f"length_scale = {value}")
    with pytest.raises(ValueError, match=r"tts\.length_scale must be > 0"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_rejects_non_numeric_length_scale(tmp_path: Path) -> None:
    path = _write_replaced(tmp_path, "length_scale = 1.0", 'length_scale = "fast"')
    with pytest.raises(ValueError, match=r"tts\.length_scale expected float, got str"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_empty_voice_disables_tts(tmp_path: Path) -> None:
    path = _write_replaced(
        tmp_path, 'voice = "voices/en_GB-semaine-medium.onnx"', 'voice = ""'
    )
    s = load_settings(path, env_path=tmp_path / "missing.env", environ={})
    assert s.tts.voice == ""
    assert not s.tts.enabled


def test_rejects_unknown_tts_provider(tmp_path: Path) -> None:
    path = _write_replaced(tmp_path, 'provider = "piper"', 'provider = "other"')
    with pytest.raises(ValueError, match=r"tts\.provider must be one of \['piper'\]"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_missing_tts_speaker_uses_voice_default(tmp_path: Path) -> None:
    path = _write_replaced(tmp_path, 'speaker = "obadiah"', "")
    s = load_settings(path, env_path=tmp_path / "missing.env", environ={})
    assert s.tts.speaker == ""


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("[0.05, 0.15]", "exactly 3 values, got 2"),
        ("[0.05, 0.15, 0.3, 0.5]", "exactly 3 values, got 4"),
        ("[0.0, 0.15, 0.30]", r"in \(0, 1\]"),
        ("[0.05, 0.15, 1.5]", r"in \(0, 1\]"),
        ("[-0.1, 0.15, 0.30]", r"in \(0, 1\]"),
        ("[0.15, 0.15, 0.30]", "strictly increasing"),
        ("[0.30, 0.15, 0.05]", "strictly increasing"),
    ],
)
def test_rejects_bad_mouth_thresholds(tmp_path: Path, value: str, match: str) -> None:
    path = _write_replaced(
        tmp_path, "mouth_thresholds = [0.05, 0.15, 0.30]", f"mouth_thresholds = {value}"
    )
    with pytest.raises(ValueError, match=rf"face\.mouth_thresholds.*{match}"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_accepts_mouth_threshold_of_one(tmp_path: Path) -> None:
    path = _write_replaced(
        tmp_path, "mouth_thresholds = [0.05, 0.15, 0.30]", "mouth_thresholds = [0.1, 0.5, 1]"
    )
    s = load_settings(path, env_path=tmp_path / "missing.env", environ={})
    assert s.face.mouth_thresholds == (0.1, 0.5, 1.0)


def test_voice_controls_from_example(tmp_path: Path) -> None:
    s = load_settings(EXAMPLE, env_path=tmp_path / "missing.env", environ={})
    assert s.tts.pitch == 1.0
    assert (s.tts.low_shelf_hz, s.tts.low_shelf_db) == (250.0, -6.0)
    assert (s.tts.high_shelf_hz, s.tts.high_shelf_db) == (3500.0, 0.0)


@pytest.mark.parametrize("value", ["0.8", "1.0", "1.06", "1.25", "1"])
def test_accepts_pitch_in_range(tmp_path: Path, value: str) -> None:
    path = _write_replaced(tmp_path, "pitch = 1.0", f"pitch = {value}")
    s = load_settings(path, env_path=tmp_path / "missing.env", environ={})
    assert s.tts.pitch == float(value)


@pytest.mark.parametrize("value", ["0.79", "1.26", "0", "-1", "2"])
def test_rejects_pitch_out_of_range(tmp_path: Path, value: str) -> None:
    path = _write_replaced(tmp_path, "pitch = 1.0", f"pitch = {value}")
    with pytest.raises(ValueError, match=r"tts\.pitch must be in 0\.8\.\.1\.25"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("low_shelf_hz = 250.0", "low_shelf_hz = 19.9", r"tts\.low_shelf_hz must be in 20\.\.10000 Hz"),
        ("high_shelf_hz = 3500.0", "high_shelf_hz = 10001", r"tts\.high_shelf_hz must be in 20\.\.10000 Hz"),
        ("low_shelf_db = -6.0", "low_shelf_db = -24.5", r"tts\.low_shelf_db must be in -24\.\.24 dB"),
        ("high_shelf_db = 0.0", "high_shelf_db = 25", r"tts\.high_shelf_db must be in -24\.\.24 dB"),
        ("low_shelf_db = -6.0", 'low_shelf_db = "-6"', r"tts\.low_shelf_db expected float, got str"),
    ],
)
def test_rejects_bad_eq(tmp_path: Path, old: str, new: str, match: str) -> None:
    path = _write_replaced(tmp_path, old, new)
    with pytest.raises(ValueError, match=match):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_accepts_eq_bounds(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    for old, new in [
        ("low_shelf_hz = 250.0", "low_shelf_hz = 20"),
        ("high_shelf_hz = 3500.0", "high_shelf_hz = 10000"),
        ("low_shelf_db = -6.0", "low_shelf_db = -24"),
        ("high_shelf_db = 0.0", "high_shelf_db = 24"),
    ]:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    s = load_settings(path, env_path=tmp_path / "missing.env", environ={})
    assert (s.tts.low_shelf_hz, s.tts.high_shelf_hz) == (20.0, 10000.0)
    assert (s.tts.low_shelf_db, s.tts.high_shelf_db) == (-24.0, 24.0)


OPTIONAL_KEYS = {
    "llm": ["thinking_level", "turn_timeout_s"],
    "stt": ["beam_size", "max_utterance_s", "min_utterance_ms", "download_root"],
    "tts": [
        "speaker",
        "length_scale",
        "pitch",
        "low_shelf_hz",
        "low_shelf_db",
        "high_shelf_hz",
        "high_shelf_db",
    ],
    "face": ["skin", "character"],
}


def _drop_keys(text: str, section: str, keys: list[str]) -> str:
    """Remove `key = ...` lines from one [section] of a TOML text."""
    out: list[str] = []
    current = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            current = stripped[1:-1]
        key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
        if current == section and key in keys and not stripped.startswith("#"):
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def _old_config(tmp_path: Path) -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    for section, keys in OPTIONAL_KEYS.items():
        text = _drop_keys(text, section, keys)
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_old_config_loads_with_defaults(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _old_config(tmp_path)
    for keys in OPTIONAL_KEYS.values():
        for key in keys:
            assert f"\n{key} =" not in path.read_text(encoding="utf-8")
    with caplog.at_level("INFO", logger="obsilord.config"):
        s = load_settings(path, env_path=tmp_path / "missing.env", environ={})
    assert s.llm.thinking_level == "minimal"
    assert s.llm.turn_timeout_s == 60.0
    assert s.tts.speaker == ""
    assert s.tts.length_scale == 1.0
    assert s.tts.pitch == 1.0
    assert (s.tts.low_shelf_hz, s.tts.low_shelf_db) == (250.0, -6.0)
    assert (s.tts.high_shelf_hz, s.tts.high_shelf_db) == (3500.0, 0.0)
    assert s.stt.beam_size == 1
    assert s.stt.max_utterance_s == 30.0
    assert s.stt.min_utterance_ms == 300
    assert s.stt.download_root == "models"
    assert s.face.skin == "dither"
    assert s.face.character == ""
    lines = [r.getMessage() for r in caplog.records if "using defaults" in r.getMessage()]
    assert len(lines) == 1
    for section, keys in OPTIONAL_KEYS.items():
        for key in keys:
            assert f"{section}.{key}" in lines[0]


def test_full_config_logs_no_defaults(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("INFO", logger="obsilord.config"):
        load_settings(EXAMPLE, env_path=tmp_path / "missing.env", environ={})
    assert not any("using defaults" in r.getMessage() for r in caplog.records)


def test_only_missing_pitch_is_defaulted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = _write_replaced(tmp_path, "pitch = 1.0", "")
    with caplog.at_level("INFO", logger="obsilord.config"):
        s = load_settings(path, env_path=tmp_path / "missing.env", environ={})
    assert s.tts.pitch == 1.0
    [line] = [r.getMessage() for r in caplog.records if "using defaults" in r.getMessage()]
    assert line.endswith(": tts.pitch")


def test_defaulted_keys_still_type_checked_when_given(tmp_path: Path) -> None:
    path = _old_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("[tts]\n", '[tts]\npitch = "high"\n'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"tts\.pitch expected float, got str"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_unknown_key_rejected_in_old_config(tmp_path: Path) -> None:
    path = _old_config(tmp_path)
    path.write_text(
        path.read_text(encoding="utf-8").replace("[tts]\n", "[tts]\nbogus_knob = 1\n"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"unknown keys in \[tts\]: \['bogus_knob'\]"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("server", "port"),
        ("llm", "model"),
        ("llm", "daily_token_cap"),
        ("tts", "voice"),
        ("tts", "provider"),
        ("face", "mouth_thresholds"),
    ],
)
def test_required_keys_still_required(tmp_path: Path, section: str, key: str) -> None:
    text = _drop_keys(EXAMPLE.read_text(encoding="utf-8"), section, [key])
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=rf"missing keys in \[{section}\]: \['{key}'\]"):
        load_settings(path, env_path=tmp_path / "missing.env", environ={})


def _load(path: Path, tmp_path: Path) -> Settings:
    return load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_stt_settings_from_example(tmp_path: Path) -> None:
    s = _load(EXAMPLE, tmp_path)
    assert s.stt.provider == "faster-whisper"
    assert s.stt.model == "small"
    assert (s.stt.device, s.stt.compute_type, s.stt.language) == ("cpu", "int8", "en")
    assert s.stt.beam_size == 1 and type(s.stt.beam_size) is int
    assert s.stt.max_utterance_s == 30.0 and type(s.stt.max_utterance_s) is float
    assert s.stt.min_utterance_ms == 300
    assert s.stt.download_root == "models"
    assert s.stt.enabled
    assert s.stt.language_or_none == "en"


def test_empty_stt_model_disables_voice_input(tmp_path: Path) -> None:
    s = _load(_write_replaced(tmp_path, 'model = "small"', 'model = ""'), tmp_path)
    assert not s.stt.enabled
    s = _load(_write_replaced(tmp_path, 'model = "small"', 'model = "   "'), tmp_path)
    assert not s.stt.enabled


def test_empty_stt_language_means_detect(tmp_path: Path) -> None:
    s = _load(_write_replaced(tmp_path, 'language = "en"', 'language = ""'), tmp_path)
    assert s.stt.language_or_none is None


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("beam_size = 1", "beam_size = 0", r"stt\.beam_size must be in 1\.\.10, got 0"),
        ("beam_size = 1", "beam_size = 11", r"stt\.beam_size must be in 1\.\.10, got 11"),
        ("beam_size = 1", "beam_size = 1.0", r"stt\.beam_size expected int, got float"),
        ("max_utterance_s = 30.0", "max_utterance_s = 0.5",
         r"stt\.max_utterance_s must be in 1\.\.120, got 0\.5"),
        ("max_utterance_s = 30.0", "max_utterance_s = 121",
         r"stt\.max_utterance_s must be in 1\.\.120, got 121"),
        ("max_utterance_s = 30.0", 'max_utterance_s = "30"',
         r"stt\.max_utterance_s expected float, got str"),
        ("min_utterance_ms = 300", "min_utterance_ms = 49",
         r"stt\.min_utterance_ms must be in 50\.\.5000, got 49"),
        ("min_utterance_ms = 300", "min_utterance_ms = 5001",
         r"stt\.min_utterance_ms must be in 50\.\.5000, got 5001"),
        ("min_utterance_ms = 300", "min_utterance_ms = 300.0",
         r"stt\.min_utterance_ms expected int, got float"),
        ('download_root = "models"', 'download_root = ""',
         r"stt\.download_root must not be empty"),
        ('provider = "faster-whisper"', 'provider = "whisper.cpp"',
         r"stt\.provider must be one of \['faster-whisper'\]"),
    ],
)
def test_rejects_bad_stt_values(tmp_path: Path, old: str, new: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _load(_write_replaced(tmp_path, old, new), tmp_path)


def test_rejects_min_utterance_not_below_max(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    text = text.replace("max_utterance_s = 30.0", "max_utterance_s = 2")
    text = text.replace("min_utterance_ms = 300", "min_utterance_ms = 2000")
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"stt\.min_utterance_ms must be shorter"):
        _load(path, tmp_path)


def test_accepts_stt_bounds(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8")
    for old, new in [
        ("beam_size = 1", "beam_size = 10"),
        ("max_utterance_s = 30.0", "max_utterance_s = 120"),
        ("min_utterance_ms = 300", "min_utterance_ms = 5000"),
    ]:
        assert old in text
        text = text.replace(old, new)
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    s = _load(path, tmp_path)
    assert (s.stt.beam_size, s.stt.max_utterance_s, s.stt.min_utterance_ms) == (10, 120.0, 5000)
    path.write_text(
        EXAMPLE.read_text(encoding="utf-8")
        .replace("max_utterance_s = 30.0", "max_utterance_s = 1")
        .replace("min_utterance_ms = 300", "min_utterance_ms = 50"),
        encoding="utf-8",
    )
    s = _load(path, tmp_path)
    assert (s.stt.max_utterance_s, s.stt.min_utterance_ms) == (1.0, 50)


def test_disabled_stt_skips_provider_and_root_checks(tmp_path: Path) -> None:
    text = (
        EXAMPLE.read_text(encoding="utf-8")
        .replace('model = "small"', 'model = ""')
        .replace('provider = "faster-whisper"', 'provider = "none"')
        .replace('download_root = "models"', 'download_root = ""')
        .replace('language = "en"', 'language = "klingon"')
    )
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    s = _load(path, tmp_path)
    assert not s.stt.enabled


@pytest.mark.parametrize("value", ["en", "he", "yue", "haw", "", " "])
def test_accepts_whisper_language(tmp_path: Path, value: str) -> None:
    s = _load(_write_replaced(tmp_path, 'language = "en"', f'language = "{value}"'), tmp_path)
    assert s.stt.language_or_none == (value.strip() or None)


@pytest.mark.parametrize("value", ["xx", "EN", "english", "en-US", "iw"])
def test_rejects_unknown_language(tmp_path: Path, value: str) -> None:
    path = _write_replaced(tmp_path, 'language = "en"', f'language = "{value}"')
    match = rf"stt\.language must be \"\" \(detect\) or a Whisper language code.*'{value}'"
    with pytest.raises(ValueError, match=match):
        _load(path, tmp_path)


def test_face_skin_and_character_from_example(tmp_path: Path) -> None:
    s = _load(EXAMPLE, tmp_path)
    assert s.face.skin == "dither"
    assert s.face.character == ""


def test_accepts_code_skin(tmp_path: Path) -> None:
    s = _load(_write_replaced(tmp_path, 'skin = "dither"', 'skin = "code"'), tmp_path)
    assert s.face.skin == "code"


@pytest.mark.parametrize("value", ["Dither", "3d", "", " dither", "pixel"])
def test_rejects_unknown_skin(tmp_path: Path, value: str) -> None:
    path = _write_replaced(tmp_path, 'skin = "dither"', f'skin = "{value}"')
    match = rf"face\.skin must be one of \['dither', 'code'\], got '{value}'"
    with pytest.raises(ValueError, match=match):
        _load(path, tmp_path)


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('skin = "dither"', "skin = 1", r"face\.skin expected str, got int"),
        ('skin = "dither"', 'skin = ["code"]', r"face\.skin expected str, got list"),
        ('character = ""', "character = true", r"face\.character expected str, got bool"),
        ('character = ""', "character = 5", r"face\.character expected str, got int"),
    ],
)
def test_rejects_wrong_face_types(tmp_path: Path, old: str, new: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _load(_write_replaced(tmp_path, old, new), tmp_path)


def test_character_file_not_checked_by_config(tmp_path: Path) -> None:
    # The file check runs at app startup (face_character.py), not in load_settings.
    path = _write_replaced(
        tmp_path, 'character = ""', 'character = "characters/missing.face.glb"'
    )
    assert _load(path, tmp_path).face.character == "characters/missing.face.glb"


def test_only_missing_face_keys_are_defaulted(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    text = _drop_keys(EXAMPLE.read_text(encoding="utf-8"), "face", ["skin", "character"])
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with caplog.at_level("INFO", logger="obsilord.config"):
        s = _load(path, tmp_path)
    assert (s.face.skin, s.face.character) == ("dither", "")
    [line] = [r.getMessage() for r in caplog.records if "using defaults" in r.getMessage()]
    assert line.endswith(": face.character, face.skin")


def _load(path: Path, tmp_path: Path) -> Settings:
    return load_settings(path, env_path=tmp_path / "missing.env", environ={})


def test_camera_and_vision_from_example(tmp_path: Path) -> None:
    s = _load(EXAMPLE, tmp_path)
    assert s.camera.api == "dshow"
    assert s.camera.timeout_s == 20.0
    assert s.vision.enabled
    assert s.vision.backend == "vulkan"
    assert s.vision.ctx == 2048 and s.vision.cpu_max_edge_px == 512
    assert s.vision == VisionSettings()  # the example spells out every default


def _without_vision_section(tmp_path: Path) -> Path:
    text = EXAMPLE.read_text(encoding="utf-8")
    start = text.index("[vision]")
    end = text.index("[face]")
    path = tmp_path / "config.toml"
    path.write_text(text[:start] + text[end:], encoding="utf-8")
    return path


def test_vision_section_is_optional(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="obsilord.config")
    s = _load(_without_vision_section(tmp_path), tmp_path)
    assert s.vision == VisionSettings()
    assert "vision.backend" in caplog.text  # listed as defaulted


def test_camera_keys_added_in_phase_5_are_optional(tmp_path: Path) -> None:
    path = _write_replaced(tmp_path, 'api = "dshow"\ntimeout_s = 20', "")
    s = _load(path, tmp_path)
    assert s.camera.api == "dshow" and s.camera.timeout_s == 20.0


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ('api = "dshow"', 'api = "v4l2"', "camera.api"),
        ("max_edge_px = 768", "max_edge_px = 32", "camera.max_edge_px"),
        ("warmup_frames = 5", "warmup_frames = -1", "camera.warmup_frames"),
        ("max_per_hour = 30", "max_per_hour = 0", "camera.max_per_hour"),
        ("cooldown_s = 10", "cooldown_s = -1", "camera.cooldown_s"),
        ('backend = "vulkan"', 'backend = "cuda"', "vision.backend"),
        ("ctx = 2048", "ctx = 0", "vision.ctx"),
        ("cpu_max_edge_px = 512", "cpu_max_edge_px = 10", "vision.cpu_max_edge_px"),
        ("start_timeout_s = 30", "start_timeout_s = 0", "vision.start_timeout_s"),
    ],
)
def test_camera_and_vision_ranges(tmp_path: Path, old: str, new: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _load(_write_replaced(tmp_path, old, new), tmp_path)


def test_empty_vision_model_disables_look(tmp_path: Path) -> None:
    path = _write_replaced(
        tmp_path,
        'model = "models/vision/qwen3-vl-2b/Qwen3VL-2B-Instruct-Q4_K_M.gguf"',
        'model = ""',
    )
    assert not _load(path, tmp_path).vision.enabled


def test_debug_section_is_optional_and_off_by_default(tmp_path: Path) -> None:
    s = _load(EXAMPLE, tmp_path)
    assert s.debug.log_prompts is False
    text = EXAMPLE.read_text(encoding="utf-8")
    path = tmp_path / "config.toml"
    path.write_text(text[: text.index("[debug]")], encoding="utf-8")
    assert _load(path, tmp_path).debug.log_prompts is False
    path.write_text(text.replace("log_prompts = false", "log_prompts = true"), encoding="utf-8")
    assert _load(path, tmp_path).debug.log_prompts is True


def test_prompt_log_is_off_until_enabled() -> None:
    from obsilord import promptlog

    assert not promptlog.enabled()
    promptlog.set_enabled(True)
    try:
        assert promptlog.enabled()
    finally:
        promptlog.set_enabled(False)
    assert not promptlog.enabled()


def test_log_prompts_must_be_a_bool(tmp_path: Path) -> None:
    text = EXAMPLE.read_text(encoding="utf-8").replace("log_prompts = false", "log_prompts = 1")
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="debug.log_prompts expected bool, got int"):
        _load(path, tmp_path)


def test_prompt_log_emit_never_raises_and_skips_work_when_off(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from obsilord import promptlog

    calls: list[str] = []

    def broken() -> str:
        calls.append("called")
        raise KeyError("odd part")

    promptlog.emit(broken)  # off: the formatter is not called
    assert calls == []
    caplog.set_level(logging.INFO, logger="obsilord.prompts")
    promptlog.emit(broken)
    assert calls == ["called"]
    assert "prompt log failed (KeyError)" in caplog.text
    promptlog.emit(lambda name: f"hello {name}", "Qwenny")
    assert "hello Qwenny" in caplog.text
