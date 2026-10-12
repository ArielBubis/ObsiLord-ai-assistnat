"""Startup check for the [face] character file.

The dither skin (web/ai-face/) loads the configured character in the browser:
inspectGlb in web/ai-face/normalize.js reads the binary glTF header and JSON
chunk, refuses Draco, meshopt and KTX2 compression (the page ships no decoders
for them), and then three.js GLTFLoader parses the file, which needs an asset
object with a glTF 2 version. check_character runs the same header, JSON,
asset and compression rules on the server at startup, so a file the page
cannot show stops startup with a readable message instead of silently falling
back to the bundled character in the browser.

The JSON rules follow the browser's JSON.parse: NaN and Infinity, which
Python's json module accepts, are refused.

One rule is stricter than the browser: buffers and images must be embedded.
Buffer 0 may use the .glb BIN chunk (no uri), an image may use a bufferView
(no uri), and anything else must be a data: URI. GLTFLoader would fetch any
other URI, and a buffer after the first with no uri makes it request the
page's own URL, so both are refused: the page must make no requests beyond
its own server, and such a file would not load anyway.

Only the header and JSON chunk are read; the binary payload is not decoded.
The JSON chunk length is checked against the file size before reading, so a
damaged header cannot make the check allocate gigabytes.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, NoReturn

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
GLB_HEADER_LEN = 20  # 12-byte file header plus the first chunk's 8-byte header
CHUNK_JSON = 0x4E4F534A  # "JSON" little-endian

# Extension name -> label used in messages. Same table as normalize.js.
COMPRESSION = {
    "KHR_draco_mesh_compression": "Draco",
    "EXT_meshopt_compression": "meshopt",
    "KHR_texture_basisu": "KTX2",
}

_EXPORT_HINT = "open it in the face creator and export a .face.glb"


def check_character(path: Path) -> None:
    """Raise ValueError with a readable message if the face cannot use this file."""
    if not path.exists():
        raise ValueError(f"face.character: {path} does not exist")
    if not path.is_file():
        raise ValueError(f"face.character: {path} is not a file")
    doc = _read_json_chunk(path)
    name = path.name

    # GLTFLoader refuses a missing asset or a version below 2.
    asset = doc.get("asset")
    version = asset.get("version") if isinstance(asset, dict) else None
    if not isinstance(version, str) or not version.startswith("2"):
        raise ValueError(f"face.character: {name} is damaged (no glTF 2.0 asset version)")

    used = _name_list(doc, "extensionsUsed", name) | _name_list(
        doc, "extensionsRequired", name
    )
    found = [label for ext, label in COMPRESSION.items() if ext in used]
    if found:
        raise ValueError(
            f"face.character: {name} uses {' and '.join(found)} compression; {_EXPORT_HINT}"
        )

    for key in ("buffers", "images"):
        for i, entry in enumerate(_entries(doc, key, name)):
            if "uri" not in entry:
                # Buffer 0 is the BIN chunk; images use a bufferView. A later
                # buffer with no uri makes GLTFLoader request the page URL.
                if key == "buffers" and i > 0:
                    raise ValueError(
                        f"face.character: {name} is damaged (buffers[{i}] has no uri)"
                    )
                continue
            uri = entry["uri"]
            if not isinstance(uri, str):
                raise ValueError(f"face.character: {name} is damaged ({key} has a bad uri)")
            if uri[:5].lower() != "data:":
                raise ValueError(
                    f"face.character: {name} links to external files; {_EXPORT_HINT}"
                )


def _read_json_chunk(path: Path) -> dict[str, Any]:
    """Check the binary glTF 2.0 header and return the parsed JSON chunk."""
    name = path.name
    not_glb = f"face.character: {name} is not a binary glTF 2.0 file (.glb or .vrm)"
    cut_short = f"face.character: {name} is damaged (JSON chunk is cut short)"
    try:
        with path.open("rb") as f:
            head = f.read(GLB_HEADER_LEN)
            if len(head) < GLB_HEADER_LEN or head[:4] != GLB_MAGIC:
                raise ValueError(not_glb)
            if struct.unpack_from("<I", head, 4)[0] != GLB_VERSION:
                raise ValueError(not_glb)
            chunk_len, chunk_type = struct.unpack_from("<II", head, 12)
            if chunk_type != CHUNK_JSON:
                raise ValueError(f"face.character: {name} is damaged (no JSON chunk)")
            # Compare before reading: f.read(n) allocates n bytes up front.
            if chunk_len > os.fstat(f.fileno()).st_size - GLB_HEADER_LEN:
                raise ValueError(cut_short)
            raw = f.read(chunk_len)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise ValueError(f"face.character: cannot read {path} ({reason})") from None
    if len(raw) < chunk_len:
        raise ValueError(cut_short)  # the file shrank while it was read
    try:
        # TextDecoder in the browser drops a UTF-8 BOM and replaces bad bytes.
        doc = json.loads(
            raw.decode("utf-8-sig", errors="replace"), parse_constant=_refuse_constant
        )
    except (ValueError, RecursionError):
        raise ValueError(
            f"face.character: {name} is damaged (unreadable JSON chunk)"
        ) from None
    if not isinstance(doc, dict):
        raise ValueError(f"face.character: {name} is damaged (JSON chunk is not an object)")
    return doc


def _refuse_constant(token: str) -> NoReturn:
    """NaN, Infinity and -Infinity are not JSON; the browser's JSON.parse refuses them."""
    raise ValueError(f"not JSON: {token}")


def _name_list(doc: dict[str, Any], key: str, name: str) -> set[str]:
    """doc[key] as a set of extension names. Absent or null means none."""
    value = doc.get(key)
    if value is None:
        return set()
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"face.character: {name} is damaged ({key} is not a list of names)")
    return set(value)


def _entries(doc: dict[str, Any], key: str, name: str) -> list[dict[str, Any]]:
    """doc[key] as a list of objects. Absent or null means none."""
    value = doc.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ValueError(f"face.character: {name} is damaged ({key} is not a list of objects)")
    return value
