"""Pure TTS helpers: speech text cleanup, frame splitting, speaker resolution."""

import asyncio
import json
from pathlib import Path

import pytest

from obsilord.tts.base import FRAME_MS, frame_bytes, split_frames
from obsilord.tts.piper import PiperTTS, pitch_adjust, read_speaker_id_map, resolve_speaker
from obsilord.tts.text import clean_for_speech, is_speakable

SEMAINE = {"prudence": 0, "spike": 1, "obadiah": 2, "poppy": 3}


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("Hello there.", "Hello there."),
        ("**Bold** and *italic* words.", "Bold and italic words."),
        ("Use `pip install` now.", "Use pip install now."),
        ("## Heading", "Heading"),
        ("_emph_ text", "emph text"),
        ("snake_case name", "snake case name"),
        ("- a bullet item", "a bullet item"),
        ("* star bullet", "star bullet"),
        ("> quoted line", "quoted line"),
        ("See [the docs](https://example.com) here.", "See the docs here."),
        ("  lots   of\t\tspace \n here ", "lots of space here"),
        ("Pi is 3.14, right?", "Pi is 3.14, right?"),
        ("Caf\u00e9 au lait.", "Caf\u00e9 au lait."),
    ],
)
def test_clean_for_speech(raw: str, spoken: str) -> None:
    assert clean_for_speech(raw) == spoken


@pytest.mark.parametrize("raw", ["", "   ", "**", "---", "...", "# ", "`` ``", "- ", "!?"])
def test_unspeakable_becomes_empty(raw: str) -> None:
    assert clean_for_speech(raw) == ""


def test_is_speakable() -> None:
    assert is_speakable("a")
    assert is_speakable("7")
    assert not is_speakable(" .,!-*")


def test_frame_bytes_100ms_at_22050() -> None:
    assert FRAME_MS == 100
    assert frame_bytes(22050) == 4410
    assert frame_bytes(16000) == 3200
    assert frame_bytes(22050, 20) == 882
    assert frame_bytes(10, 1) == 2  # never zero


def test_split_frames_sizes_and_content() -> None:
    pcm = bytes(range(256)) * 40  # 10240 bytes
    frames = split_frames(pcm, 4410)
    assert [len(f) for f in frames] == [4410, 4410, 1420]
    assert b"".join(frames) == pcm
    assert all(len(f) % 2 == 0 for f in frames)


def test_split_frames_exact_and_empty() -> None:
    assert split_frames(b"\x00" * 8820, 4410) == [b"\x00" * 4410] * 2
    assert split_frames(b"", 4410) == []


@pytest.mark.parametrize("size", [0, -2, 3])
def test_split_frames_rejects_bad_size(size: int) -> None:
    with pytest.raises(ValueError):
        split_frames(b"\x00\x00", size)


def test_resolve_speaker_by_name() -> None:
    assert resolve_speaker(SEMAINE, "obadiah") == 2
    assert resolve_speaker(SEMAINE, " prudence ") == 0


def test_resolve_speaker_empty_means_default() -> None:
    assert resolve_speaker(SEMAINE, "") is None
    assert resolve_speaker({}, "") is None


def test_resolve_speaker_unknown_lists_valid_names() -> None:
    with pytest.raises(ValueError, match="'bob'.*obadiah, poppy, prudence, spike"):
        resolve_speaker(SEMAINE, "bob")


def test_resolve_speaker_on_single_speaker_voice() -> None:
    with pytest.raises(ValueError, match="single speaker"):
        resolve_speaker({}, "obadiah")


def _voice_files(tmp_path: Path, speaker_map: dict[str, int] | None) -> Path:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not a real model")
    cfg: dict[str, object] = {"audio": {"sample_rate": 22050}}
    if speaker_map is not None:
        cfg["speaker_id_map"] = speaker_map
    Path(f"{model}.json").write_text(json.dumps(cfg), encoding="utf-8")
    return model


def test_read_speaker_id_map(tmp_path: Path) -> None:
    model = _voice_files(tmp_path, SEMAINE)
    assert read_speaker_id_map(Path(f"{model}.json")) == SEMAINE
    single = tmp_path / "single"
    single.mkdir()
    model2 = _voice_files(single, None)
    assert read_speaker_id_map(Path(f"{model2}.json")) == {}


def test_load_unknown_speaker_fails_before_model_load(tmp_path: Path) -> None:
    # The model file is junk: reaching PiperVoice.load would raise a different error.
    model = _voice_files(tmp_path, SEMAINE)
    with pytest.raises(ValueError, match="valid names: obadiah, poppy, prudence, spike"):
        PiperTTS.load(model, "nobody", 1.0)


def test_load_missing_voice_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match='voice = ""'):
        PiperTTS.load(tmp_path / "missing.onnx", "obadiah", 1.0)


def test_load_missing_voice_json(tmp_path: Path) -> None:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"x")
    with pytest.raises(FileNotFoundError, match="voice.onnx.json"):
        PiperTTS.load(model, "obadiah", 1.0)


class _FakeVoiceConfig:
    sample_rate = 22050


class _FakeVoice:
    """Stands in for piper.PiperVoice: records the SynthesisConfig used."""

    config = _FakeVoiceConfig()

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def synthesize(self, text: str, syn_config: object = None) -> list[object]:
        self.calls.append((text, syn_config))

        class _Chunk:
            audio_int16_bytes = b"\x01\x00" * 5000  # 10000 bytes

        return [_Chunk()]


def test_pitch_adjust() -> None:
    assert pitch_adjust(22050, 1.0, 1.0) == (22050, 1.0)
    rate, ls = pitch_adjust(22050, 1.0, 1.06)
    assert rate == 23373
    assert ls == pytest.approx(1.06)
    rate, ls = pitch_adjust(22050, 1.2, 0.9)
    assert rate == 19845
    assert ls == pytest.approx(1.08)


def test_piper_tts_pitch_106_announces_rate_and_slows_synthesis() -> None:
    voice = _FakeVoice()
    tts = PiperTTS(voice, speaker_id=2, length_scale=1.0, pitch=1.06)
    assert tts.native_rate == 22050
    assert tts.sample_rate == 23373
    assert tts.length_scale_effective == pytest.approx(1.06)
    frames = asyncio.run(_collect(tts, "Hi."))
    # Frame size stays 100 ms at the native rate.
    assert [len(f) for f in frames] == [4410, 4410, 1180]
    [(text, cfg)] = voice.calls
    assert text == "Hi."
    assert cfg.length_scale == pytest.approx(1.06)  # type: ignore[attr-defined]
    assert cfg.speaker_id == 2  # type: ignore[attr-defined]


def test_piper_tts_pitch_one_is_unchanged() -> None:
    voice = _FakeVoice()
    tts = PiperTTS(voice, speaker_id=None, length_scale=0.9)
    assert tts.sample_rate == tts.native_rate == 22050
    assert tts.length_scale_effective == 0.9
    asyncio.run(_collect(tts, "Hi."))
    [(_, cfg)] = voice.calls
    assert cfg.length_scale == 0.9  # type: ignore[attr-defined]
    assert cfg.speaker_id is None  # type: ignore[attr-defined]


async def _collect(tts: PiperTTS, text: str) -> list[bytes]:
    return [f async for f in tts.synthesize(text)]
