"""Entry point: python -m obsilord starts uvicorn on 127.0.0.1."""

import logging

import uvicorn

from obsilord.app import create_app
from obsilord.config import load_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Per-request SDK and HTTP lines at INFO are noise. piper logs the input
    # text and phonemes at DEBUG, so it is held at WARNING to keep text out of logs.
    # faster_whisper logs every utterance's duration at INFO (no text): noise.
    for name in ("google_genai", "httpx", "httpcore", "piper", "faster_whisper"):
        logging.getLogger(name).setLevel(logging.WARNING)
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
