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


@pytest.fixture(autouse=True)
def _no_default_settings_files(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """create_app keeps the settings file and the activity log of a test in
    its tmp_path, never in the repo's data/ and logs/ folders. Tests that need
    them pass settings_store= and activity=."""
    import obsilord.app
    from obsilord.activity import ActivityLog
    from obsilord.permissions import SettingsStore

    monkeypatch.setattr(
        obsilord.app,
        "_default_settings_store",
        lambda settings: SettingsStore(
            tmp_path / "default-settings.json", obsilord.app.tool_defaults(), settings.camera.photos
        ),
    )
    monkeypatch.setattr(
        obsilord.app,
        "_default_activity",
        lambda settings: ActivityLog(tmp_path / "default-activity.jsonl", settings.memory.retention_days),
    )
