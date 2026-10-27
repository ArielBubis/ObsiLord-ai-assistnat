"""The page and its static assets are served with usable content types.

Module scripts are refused by browsers unless served with a JavaScript MIME
type, so this checks every web/*.js file explicitly.
"""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fakes import text_only
from obsilord.app import create_app
from obsilord.config import REPO_ROOT, load_settings

WEB_DIR = REPO_ROOT / "web"
JS_FILES = sorted(p.name for p in WEB_DIR.glob("*.js"))
# Gitignored (license unverified, see planning/PLAN.md): absent in a fresh clone.
BUNDLED_CHARACTER = WEB_DIR / "ai-face" / "characters" / "default.face.glb"
# ES modules the dither skin imports (web/ai-face/, three.js vendored there).
AI_FACE_MODULES = [
    "obsi-lord-skin.js",
    "ai-face.js",
    "render.js",
    "animate.js",
    "normalize.js",
    "vendor/three/build/three.module.min.js",
    "vendor/three/examples/jsm/loaders/GLTFLoader.js",
    "vendor/three/examples/jsm/utils/BufferGeometryUtils.js",
]


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    settings = load_settings(
        REPO_ROOT / "config.example.toml", env_path=tmp_path / "missing.env", environ={}
    )
    return TestClient(create_app(text_only(settings)))


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


@pytest.mark.parametrize("name", AI_FACE_MODULES)
def test_ai_face_modules_served_with_js_mime(client: TestClient, name: str) -> None:
    resp = client.get(f"/static/ai-face/{name}")
    assert resp.status_code == 200
    ctype = resp.headers["content-type"].split(";")[0].strip()
    assert ctype in ("text/javascript", "application/javascript"), ctype


@pytest.mark.skipif(
    not BUNDLED_CHARACTER.is_file(),
    reason="web/ai-face/characters/default.face.glb is absent (gitignored, license unverified)",
)
def test_bundled_character_served(client: TestClient) -> None:
    # The dither skin's fallback when /face/character.glb answers 404.
    resp = client.get("/static/ai-face/characters/default.face.glb")
    assert resp.status_code == 200
    assert resp.content[:4] == b"glTF"


def test_stylesheet_urls_exist() -> None:
    # Fonts are self-hosted; every url() in the stylesheet must be a file in web/.
    text = (WEB_DIR / "style.css").read_text(encoding="utf-8")
    for target in re.findall(r"""url\(\s*["']?([^"')]+)["']?\s*\)""", text):
        assert (WEB_DIR / target).is_file(), f"style.css: {target} is missing"


def test_ids_the_scripts_read_are_in_the_page() -> None:
    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    ids = set(re.findall(r'getElementById\("([\w-]+)"\)', html))
    ids |= set(re.findall(r'getElementById\("([\w-]+)"\)', (WEB_DIR / "ws.js").read_text(encoding="utf-8")))
    ids |= set(re.findall(r'"#([\w-]+)"', (WEB_DIR / "debug.js").read_text(encoding="utf-8")))
    assert ids, "no ids found; the patterns above no longer match the scripts"
    for name in sorted(ids):
        assert f'id="{name}"' in html, f"index.html has no element with id {name}"
