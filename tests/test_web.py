"""The page and its static assets are served with usable content types.

Module scripts are refused by browsers unless served with a JavaScript MIME
type, so this checks every web/*.js file explicitly.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from obsilord.app import create_app
from obsilord.config import REPO_ROOT, load_settings

WEB_DIR = REPO_ROOT / "web"
JS_FILES = sorted(p.name for p in WEB_DIR.glob("*.js"))


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    settings = load_settings(
        REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
    )
    return TestClient(create_app(settings))


def test_index_has_canvas_and_status(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "<canvas" in resp.text
    assert 'id="status"' in resp.text
    assert 'type="module"' in resp.text


def test_expected_js_files_exist() -> None:
    for name in ("state.js", "idle.js", "skin_code.js", "face.js", "ws.js"):
        assert name in JS_FILES


@pytest.mark.parametrize("name", JS_FILES)
def test_js_served_with_js_mime(client: TestClient, name: str) -> None:
    resp = client.get(f"/static/{name}")
    assert resp.status_code == 200
    ctype = resp.headers["content-type"].split(";")[0].strip()
    assert ctype in ("text/javascript", "application/javascript"), ctype


def test_css_served_with_css_mime(client: TestClient) -> None:
    resp = client.get("/static/style.css")
    assert resp.status_code == 200
    assert resp.headers["content-type"].split(";")[0].strip() == "text/css"
