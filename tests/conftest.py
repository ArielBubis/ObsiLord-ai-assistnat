import sys
from pathlib import Path

import pytest

# Make tests/fakes.py importable as "fakes".
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(autouse=True)
def _no_default_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """create_app builds no conversation store from settings in tests, so no
    test writes to the repo's data/ folder. Memory tests pass memory=."""
    import obsilord.app

    monkeypatch.setattr(obsilord.app, "_default_memory", lambda settings: None)
