"""[face] character at app startup and GET /face/character.glb."""

import dataclasses
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import obsilord.app as app_module
from fakes import FakeLLM, make_glb, text_only
from obsilord.app import create_app
from obsilord.config import REPO_ROOT, Settings, load_settings

GLB = make_glb({"asset": {"version": "2.0"}, "extensionsUsed": ["KHR_texture_transform"]})
DRACO = make_glb({"asset": {"version": "2.0"}, "extensionsUsed": ["KHR_draco_mesh_compression"]})


def _settings(tmp_path: Path, **face: Any) -> Settings:
    base = text_only(
        load_settings(
            REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
        )
    )
    return dataclasses.replace(base, face=dataclasses.replace(base.face, **face))


def _client(settings: Settings) -> TestClient:
    return TestClient(create_app(settings, llm=FakeLLM(["x"])))


def _errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]


def test_character_served_from_absolute_path(tmp_path: Path) -> None:
    path = tmp_path / "characters" / "nova.face.glb"
    path.parent.mkdir()
    path.write_bytes(GLB)
    resp = _client(_settings(tmp_path, character=str(path))).get("/face/character.glb")
    assert resp.status_code == 200
    assert resp.content == GLB
    assert resp.headers["content-type"] == "model/gltf-binary"
    assert resp.headers["cache-control"] == "no-cache"


def test_relative_character_resolves_against_repo_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    (repo / "web").mkdir(parents=True)  # StaticFiles requires the directory to exist
    (repo / "characters").mkdir()
    (repo / "characters" / "nova.vrm").write_bytes(GLB)
    # The working directory is not the repo root, as when started elsewhere.
    monkeypatch.chdir(tmp_path)
    settings = dataclasses.replace(
        _settings(tmp_path, character="characters/nova.vrm"), repo_root=repo
    )
    resp = _client(settings).get("/face/character.glb")
    assert resp.status_code == 200
    assert resp.content == GLB


@pytest.mark.parametrize("character", ["", "   "])
def test_empty_character_answers_404(
    tmp_path: Path, character: str, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="obsilord.app"):
        client = _client(_settings(tmp_path, character=character))
    assert any("bundled character" in r.getMessage() for r in caplog.records)
    assert client.get("/face/character.glb").status_code == 404


def test_character_removed_after_startup_answers_404(tmp_path: Path) -> None:
    path = tmp_path / "nova.face.glb"
    path.write_bytes(GLB)
    client = _client(_settings(tmp_path, character=str(path)))
    path.unlink()
    assert client.get("/face/character.glb").status_code == 404


def test_missing_character_stops_startup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "characters" / "nope.face.glb"
    with caplog.at_level(logging.ERROR, logger="obsilord.app"):
        with pytest.raises(ValueError, match="does not exist"):
            create_app(_settings(tmp_path, character=str(path)), llm=FakeLLM(["x"]))
    [line] = _errors(caplog)
    assert line.startswith(f"face.character: {path} does not exist")
    assert '[face] character = ""' in line


def test_compressed_character_stops_startup(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "facecap.glb"
    path.write_bytes(DRACO)
    with caplog.at_level(logging.ERROR, logger="obsilord.app"):
        with pytest.raises(ValueError, match="uses Draco compression"):
            create_app(_settings(tmp_path, character=str(path)), llm=FakeLLM(["x"]))
    [line] = _errors(caplog)
    assert "facecap.glb uses Draco compression" in line
    assert "export a .face.glb" in line


def test_character_checked_before_models_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(settings: Settings) -> Any:
        raise AssertionError("no model may load before the character check")

    monkeypatch.setattr(app_module, "_default_llm", boom)
    monkeypatch.setattr(app_module, "_default_tts", boom)
    monkeypatch.setattr(app_module, "_default_stt", boom)
    path = tmp_path / "notes.glb"
    path.write_text("not a model", encoding="utf-8")
    with pytest.raises(ValueError, match="not a binary glTF 2.0 file"):
        create_app(_settings(tmp_path, character=str(path)))


def test_face_config_has_skin_but_not_character(tmp_path: Path) -> None:
    path = tmp_path / "nova.face.glb"
    path.write_bytes(GLB)
    body = _client(_settings(tmp_path, skin="code", character=str(path))).get(
        "/config/face"
    ).json()
    assert body["skin"] == "code"
    assert "character" not in body
    assert str(path) not in str(body)
