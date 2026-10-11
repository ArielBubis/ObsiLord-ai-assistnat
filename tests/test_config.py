from pathlib import Path

import pytest

from obsilord.config import (
    REPO_ROOT,
    Secrets,
    Settings,
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
    "tts": [
        "speaker",
        "length_scale",
        "pitch",
        "low_shelf_hz",
        "low_shelf_db",
        "high_shelf_hz",
        "high_shelf_db",
    ],
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
