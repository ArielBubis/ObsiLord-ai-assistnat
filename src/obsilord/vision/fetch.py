"""Download the pinned llama.cpp builds and vision model into models/.

Run once by hand from the repo root (the app never downloads programs):

    python -m obsilord.vision.fetch                 # vulkan and cpu builds, model
    python -m obsilord.vision.fetch --backends cpu  # cpu build and model only
    python -m obsilord.vision.fetch --sample        # also the live tests' photo

Sources, looked up 2026-10-27:
- llama.cpp build b11919 (github.com/ggml-org/llama.cpp releases, published
  2026-10-27, marked prerelease like every bNNNN build): prebuilt Windows x64
  zips for the vulkan and cpu backends. sha256 from the release asset digests.
- Qwen/Qwen3-VL-2B-Instruct-GGUF on Hugging Face (tagged license:apache-2.0),
  pinned to commit 52d6c8ff: Q4_K_M weights (1056 MB) and the Q8_0 mmproj
  (424 MB). sha256 from the LFS oids in the repo tree.
- Sample photo for tests/test_look_live.py: bee.jpg (5184x3456, 5.4 MB) from
  the huggingface/documentation-images dataset, pinned to commit 88a3af27.

Every file is checked against its sha256 (a mismatch deletes the download and
stops). Files that are already present and match are skipped. Each zip is
unpacked into its own new directory. The target paths match the [vision]
defaults in config.py.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from obsilord.config import REPO_ROOT

BIN_ROOT = REPO_ROOT / "models" / "llama.cpp"
MODEL_ROOT = REPO_ROOT / "models" / "vision"
LLAMA_BUILD = "b11919"
USER_AGENT = "obsi-lord-fetch"
CHUNK = 1 << 20


@dataclass(frozen=True)
class Remote:
    url: str
    name: str
    sha256: str


def hf_file(repo: str, revision: str, name: str, sha256: str, dataset: bool = False) -> Remote:
    prefix = "datasets/" if dataset else ""
    return Remote(f"https://huggingface.co/{prefix}{repo}/resolve/{revision}/{name}", name, sha256)


def _llama_zip(name: str, sha256: str) -> Remote:
    url = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_BUILD}/{name}"
    return Remote(url, name, sha256)


BINARIES: dict[str, Remote] = {
    "vulkan": _llama_zip(
        f"llama-{LLAMA_BUILD}-bin-win-vulkan-x64.zip",
        "ef40f87e7aa747e757c4c2639fff97ce3e5df0e8e6ccce015595e7a1750fa18f",
    ),
    "cpu": _llama_zip(
        f"llama-{LLAMA_BUILD}-bin-win-cpu-x64.zip",
        "d6f9fe81c14287fd5a26c6bc5083dda371038cbce759afc5049dc7dd76a9b79f",
    ),
}

MODEL_KEY = "qwen3-vl-2b"
_QWEN = ("Qwen/Qwen3-VL-2B-Instruct-GGUF", "52d6c8ffea26cc873ac5ad116f8631268d7eb503")
MODEL = hf_file(
    *_QWEN,
    "Qwen3VL-2B-Instruct-Q4_K_M.gguf",
    "089d75c52f4b7ffc56ba998ffc50aae89fcafc755f9e7208aacca281dca6c2ae",
)
MMPROJ = hf_file(
    *_QWEN,
    "mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
    "f9a68fabba69c3b81e153367b2c7521030b0fa8bb0de400c9599c8e6725f9c82",
)

SAMPLE_DIR = MODEL_ROOT / "samples"
SAMPLE = hf_file(
    "huggingface/documentation-images",
    "88a3af27faf4713e57cec838bd1b9f15b2cf3f4e",
    "bee.jpg",
    "8b21ba78250f852ca5990063866b1ace6432521d0251bde7f8de783b22c99a6d",
    dataset=True,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download(remote: Remote, dest_dir: Path) -> Path:
    """Download to dest_dir/name and check sha256. Skips a file that already matches."""
    dest = dest_dir / remote.name
    if dest.exists() and sha256_file(dest) == remote.sha256:
        print(f"  ok    {dest}")
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    h = hashlib.sha256()
    request = urllib.request.Request(remote.url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as resp, part.open("wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while chunk := resp.read(CHUNK):
            out.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total and (pct := done * 10 // total) != shown:
                shown = pct
                print(f"  {remote.name}: {pct * 10}% of {total / 2**20:.0f} MB", flush=True)
    if h.hexdigest() != remote.sha256:
        part.unlink()
        raise SystemExit(f"sha256 mismatch for {remote.name}: got {h.hexdigest()}")
    part.replace(dest)
    print(f"  saved {dest}")
    return dest


def binary_dir(backend: str) -> Path:
    return BIN_ROOT / f"{LLAMA_BUILD}-{backend}"


def fetch_binaries(backends: list[str]) -> None:
    for backend in backends:
        target = binary_dir(backend)
        if (target / "llama-server.exe").exists():
            print(f"  ok    {target}")
            continue
        remote = BINARIES[backend]
        archive = download(remote, BIN_ROOT / "_zips")
        # Unzipped next to the target and renamed when complete, so an
        # interrupted unzip is never taken for a finished one.
        partial = target.with_name(target.name + ".part")
        shutil.rmtree(partial, ignore_errors=True)
        with zipfile.ZipFile(archive) as z:
            # Python 3.12 drops absolute paths and ".." parts while extracting.
            z.extractall(partial)
        if not (partial / "llama-server.exe").exists():
            raise SystemExit(f"llama-server.exe not found at the top of {remote.name}")
        if target.exists():
            shutil.rmtree(target)  # left by an older run, without llama-server.exe
        partial.rename(target)
        print(f"  unzipped {target}")


def fetch_model() -> None:
    download(MODEL, MODEL_ROOT / MODEL_KEY)
    download(MMPROJ, MODEL_ROOT / MODEL_KEY)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download the look() vision model and server.")
    parser.add_argument(
        "--backends", nargs="+", choices=list(BINARIES), default=list(BINARIES),
        help="llama.cpp builds to fetch (default: vulkan cpu)",
    )
    parser.add_argument(
        "--sample", action="store_true", help="also fetch the sample photo for the live tests"
    )
    args = parser.parse_args(argv)
    print(f"llama.cpp {LLAMA_BUILD}: {', '.join(args.backends)}")
    fetch_binaries(args.backends)
    print(f"model: {MODEL_KEY}")
    fetch_model()
    if args.sample:
        print("sample photo:")
        download(SAMPLE, SAMPLE_DIR)


if __name__ == "__main__":
    main()
