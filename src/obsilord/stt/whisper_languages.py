"""Language codes accepted by Whisper models in faster-whisper.

Copied from _LANGUAGE_CODES in faster_whisper/tokenizer.py (faster-whisper
1.2.1, checked 2026-10-12) so config validation does not import faster_whisper
(importing any part of it loads the whole package: ctranslate2, av, tokenizers).
tests/test_stt.py compares this list with the installed package, so a
faster-whisper upgrade that changes the list fails the tests.

No imports on purpose: config.py loads this module at startup.
"""

WHISPER_LANGUAGE_CODES: tuple[str, ...] = (
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo",
    "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es",
    "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "haw",
    "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja",
    "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo",
    "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt",
    "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq",
    "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl",
    "tr", "tt", "uk", "ur", "uz", "vi", "yi", "yo", "zh", "yue",
)
