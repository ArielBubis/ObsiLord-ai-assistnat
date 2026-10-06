"""Entry point: python -m pixelbuddy starts uvicorn on 127.0.0.1."""

import logging

import uvicorn

from pixelbuddy.app import create_app
from pixelbuddy.config import load_settings


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
