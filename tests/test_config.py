from pathlib import Path

import pytest

from pixelbuddy.config import (
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
