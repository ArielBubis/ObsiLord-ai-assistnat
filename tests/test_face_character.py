"""check_character: the startup check for the [face] character file."""

import struct
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from fakes import GLB_CHUNK_BIN, GLB_CHUNK_JSON, make_glb, make_glb_raw
from obsilord.config import REPO_ROOT
from obsilord.face_character import check_character

ASSET = {"version": "2.0"}
VALID = make_glb({"asset": ASSET})
HINT = "open it in the face creator and export a .face.glb"
BUNDLED = REPO_ROOT / "web" / "ai-face" / "characters" / "default.face.glb"
# The bundled character is gitignored (license unverified, see planning/PLAN.md), so a
# fresh clone does not have it.
NO_BUNDLED = "web/ai-face/characters/default.face.glb is absent (gitignored, license unverified)"


def _write(tmp_path: Path, data: bytes, name: str = "nova.face.glb") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _message(path: Path) -> str:
    with pytest.raises(ValueError) as info:
        check_character(path)
    return str(info.value)


def _doc_message(tmp_path: Path, doc: Any) -> str:
    return _message(_write(tmp_path, make_glb(doc)))


def test_valid_minimal_glb_passes(tmp_path: Path) -> None:
    check_character(_write(tmp_path, VALID))


def test_embedded_buffer_and_image_pass(tmp_path: Path) -> None:
    doc = {
        "asset": ASSET,
        "buffers": [{"byteLength": 4}],
        "bufferViews": [{"buffer": 0, "byteLength": 4}],
        "images": [{"bufferView": 0, "mimeType": "image/png"}],
        "extensionsUsed": ["KHR_texture_transform", "VRMC_vrm"],
        "extensionsRequired": None,
    }
    check_character(_write(tmp_path, make_glb(doc), "model.vrm"))


@pytest.mark.skipif(not BUNDLED.is_file(), reason=NO_BUNDLED)
def test_bundled_default_character_passes() -> None:
    # The page falls back to this file, so it must meet the same rules.
    check_character(BUNDLED)


def test_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "characters" / "nope.face.glb"
    message = _message(path)
    assert message == f"face.character: {path} does not exist"


def test_directory_is_not_a_file(tmp_path: Path) -> None:
    assert _message(tmp_path) == f"face.character: {tmp_path} is not a file"


@pytest.mark.parametrize(
    "data",
    [
        b"just some notes, renamed to .glb\n",
        b"",
        b"glTF",
        VALID[:12],
        VALID[:19],
        make_glb_raw(b'{"asset": {"version": "1.0"}}', version=1),
        make_glb_raw(b'{"asset": {"version": "2.0"}}', version=3),
    ],
    ids=["text", "empty", "magic-only", "header-12", "header-19", "version-1", "version-3"],
)
def test_not_binary_gltf_2(tmp_path: Path, data: bytes) -> None:
    message = _message(_write(tmp_path, data, "notes.glb"))
    assert message == "face.character: notes.glb is not a binary glTF 2.0 file (.glb or .vrm)"


def test_first_chunk_not_json(tmp_path: Path) -> None:
    path = _write(tmp_path, make_glb_raw(b"\x00" * 8, chunk_type=GLB_CHUNK_BIN))
    assert _message(path) == "face.character: nova.face.glb is damaged (no JSON chunk)"


def test_json_chunk_cut_short(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID[:-4])
    assert _message(path) == "face.character: nova.face.glb is damaged (JSON chunk is cut short)"


@pytest.mark.parametrize(
    "raw",
    [b'{"asset": ', b"\xff\xfe{}", b"{} trailing", b"[" * 100_000],
    ids=["truncated", "bad-bytes", "extra-data", "deep-nesting"],
)
def test_unreadable_json(tmp_path: Path, raw: bytes) -> None:
    path = _write(tmp_path, make_glb_raw(raw))
    assert _message(path) == "face.character: nova.face.glb is damaged (unreadable JSON chunk)"


def test_json_with_utf8_bom_is_read(tmp_path: Path) -> None:
    # The browser's TextDecoder drops a BOM, so the check does too.
    check_character(_write(tmp_path, make_glb_raw(b'\xef\xbb\xbf{"asset": {"version": "2.0"}}')))


@pytest.mark.parametrize("doc", [[], [{"asset": ASSET}], None, 5, "glTF"])
def test_json_not_an_object(tmp_path: Path, doc: Any) -> None:
    message = _doc_message(tmp_path, doc)
    assert message == "face.character: nova.face.glb is damaged (JSON chunk is not an object)"


@pytest.mark.parametrize(
    ("ext", "label"),
    [
        ("KHR_draco_mesh_compression", "Draco"),
        ("EXT_meshopt_compression", "meshopt"),
        ("KHR_texture_basisu", "KTX2"),
    ],
)
def test_compression_in_extensions_used(tmp_path: Path, ext: str, label: str) -> None:
    message = _doc_message(tmp_path, {"asset": ASSET, "extensionsUsed": [ext]})
    assert message == f"face.character: nova.face.glb uses {label} compression; {HINT}"


def test_compression_in_extensions_required(tmp_path: Path) -> None:
    doc = {"asset": ASSET, "extensionsRequired": ["EXT_meshopt_compression"]}
    message = _doc_message(tmp_path, doc)
    assert message == f"face.character: nova.face.glb uses meshopt compression; {HINT}"


def test_several_compressions_are_named(tmp_path: Path) -> None:
    doc = {
        "asset": ASSET,
        "extensionsUsed": ["KHR_texture_basisu", "KHR_draco_mesh_compression"],
        "extensionsRequired": ["KHR_draco_mesh_compression"],
    }
    message = _doc_message(tmp_path, doc)
    assert message == f"face.character: nova.face.glb uses Draco and KTX2 compression; {HINT}"


@pytest.mark.parametrize("key", ["extensionsUsed", "extensionsRequired"])
@pytest.mark.parametrize(
    "value", ["KHR_draco_mesh_compression", 5, {"KHR_draco_mesh_compression": True}, [1], [None]]
)
def test_extension_list_not_names_is_damaged(tmp_path: Path, key: str, value: Any) -> None:
    message = _doc_message(tmp_path, {"asset": ASSET, key: value})
    assert message == f"face.character: nova.face.glb is damaged ({key} is not a list of names)"


@pytest.mark.parametrize("key", ["buffers", "images"])
@pytest.mark.parametrize(
    "uri", ["nova.bin", "https://example.com/textures/skin.png", "", "/face/other.bin", "blob:x"]
)
def test_external_uri_refused(tmp_path: Path, key: str, uri: str) -> None:
    message = _doc_message(tmp_path, {"asset": ASSET, key: [{"uri": uri}]})
    assert message == f"face.character: nova.face.glb links to external files; {HINT}"
    if uri:
        assert uri not in message


def test_external_uri_after_embedded_entries_refused(tmp_path: Path) -> None:
    doc = {
        "asset": ASSET,
        "buffers": [{"byteLength": 4}],
        "images": [{"bufferView": 0}, {"uri": "data:image/png;base64,AAAA"}, {"uri": "eyes.png"}],
    }
    assert "links to external files" in _doc_message(tmp_path, doc)


@pytest.mark.parametrize(
    "uri", ["data:application/octet-stream;base64,AAAA", "DATA:application/gltf-buffer;base64,"]
)
def test_data_uri_accepted(tmp_path: Path, uri: str) -> None:
    doc = {
        "asset": ASSET,
        "buffers": [{"uri": uri, "byteLength": 3}],
        "images": [{"uri": "data:image/png;base64,iVBORw0KGgo="}],
    }
    check_character(_write(tmp_path, make_glb(doc)))


@pytest.mark.parametrize("key", ["buffers", "images"])
@pytest.mark.parametrize("value", [{"uri": "data:,"}, "data:,", [5], ["data:,"], [None]])
def test_entries_not_objects_are_damaged(tmp_path: Path, key: str, value: Any) -> None:
    message = _doc_message(tmp_path, {"asset": ASSET, key: value})
    assert message == f"face.character: nova.face.glb is damaged ({key} is not a list of objects)"


@pytest.mark.parametrize("key", ["buffers", "images"])
@pytest.mark.parametrize("uri", [None, 5, ["data:,"]])
def test_uri_not_a_string_is_damaged(tmp_path: Path, key: str, uri: Any) -> None:
    message = _doc_message(tmp_path, {"asset": ASSET, key: [{"uri": uri}]})
    assert message == f"face.character: nova.face.glb is damaged ({key} has a bad uri)"


def test_null_lists_mean_none(tmp_path: Path) -> None:
    doc = {
        "asset": ASSET,
        "extensionsUsed": None,
        "extensionsRequired": None,
        "buffers": None,
        "images": None,
    }
    check_character(_write(tmp_path, make_glb(doc)))


def test_huge_json_chunk_length_is_not_read(tmp_path: Path) -> None:
    # A damaged header claiming a 4 GB JSON chunk in a 100-byte file must be
    # refused from the file size, without allocating the claimed length.
    data = b"glTF" + struct.pack("<III", 2, 100, 0xFFFFFFFF) + struct.pack("<I", GLB_CHUNK_JSON)
    path = _write(tmp_path, data + b"{}".ljust(100 - len(data)))
    tracemalloc.start()
    try:
        message = _message(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert message == "face.character: nova.face.glb is damaged (JSON chunk is cut short)"
    assert peak < 1_000_000, peak


def test_json_chunk_one_byte_past_the_end(tmp_path: Path) -> None:
    raw = b'{"asset": {"version": "2.0"}}   '
    data = b"glTF" + struct.pack("<III", 2, 20 + len(raw), len(raw) + 1)
    path = _write(tmp_path, data + struct.pack("<I", GLB_CHUNK_JSON) + raw)
    assert _message(path) == "face.character: nova.face.glb is damaged (JSON chunk is cut short)"


@pytest.mark.parametrize(
    "doc",
    [
        {},
        {"asset": None},
        {"asset": ["2.0"]},
        {"asset": "2.0"},
        {"asset": {}},
        {"asset": {"version": None}},
        {"asset": {"version": 2}},
        {"asset": {"version": 2.0}},
        {"asset": {"version": ""}},
        {"asset": {"version": "1.0"}},
        {"asset": {"version": "3.0"}},
        {"asset": {"minVersion": "2.0"}},
    ],
)
def test_no_gltf_2_asset_version_is_damaged(tmp_path: Path, doc: Any) -> None:
    message = _doc_message(tmp_path, doc)
    assert message == "face.character: nova.face.glb is damaged (no glTF 2.0 asset version)"


@pytest.mark.parametrize(
    "asset",
    [
        {"version": "2.0"},
        {"version": "2.1", "minVersion": "2.0"},
        {"version": "2.0", "generator": "THREE.GLTFExporter r170"},
    ],
)
def test_gltf_2_asset_versions_pass(tmp_path: Path, asset: dict[str, str]) -> None:
    check_character(_write(tmp_path, make_glb({"asset": asset})))


@pytest.mark.parametrize("token", [b"NaN", b"Infinity", b"-Infinity"])
def test_non_json_constants_are_unreadable(tmp_path: Path, token: bytes) -> None:
    # Python's json accepts these; the browser's JSON.parse does not.
    raw = b'{"asset": {"version": "2.0"}, "extras": {"scale": ' + token + b"}}"
    path = _write(tmp_path, make_glb_raw(raw))
    assert _message(path) == "face.character: nova.face.glb is damaged (unreadable JSON chunk)"


@pytest.mark.parametrize(
    "buffers",
    [
        [{"byteLength": 4}, {"byteLength": 8}],
        [{"uri": "data:,", "byteLength": 0}, {"byteLength": 8}],
        [{"byteLength": 4}, {"uri": "data:,", "byteLength": 0}, {"byteLength": 8}],
    ],
)
def test_later_buffer_without_uri_is_damaged(tmp_path: Path, buffers: list[Any]) -> None:
    message = _doc_message(tmp_path, {"asset": ASSET, "buffers": buffers})
    index = len(buffers) - 1
    assert message == f"face.character: nova.face.glb is damaged (buffers[{index}] has no uri)"


def test_first_buffer_and_any_image_may_omit_uri(tmp_path: Path) -> None:
    doc = {
        "asset": ASSET,
        "buffers": [{"byteLength": 8}, {"uri": "data:,", "byteLength": 0}],
        "images": [{"bufferView": 0}, {"bufferView": 1}, {"bufferView": 2}],
    }
    check_character(_write(tmp_path, make_glb(doc)))
