import os

# Must be set before any test module imports app.config / main, so the developer's real
# .env is never loaded into the test process.
os.environ["ASKFOLIO_SKIP_DOTENV"] = "1"

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_heartbeat(tmp_path, monkeypatch):
    """Keep every test off the real /tmp/heartbeat."""
    monkeypatch.setenv("HEARTBEAT_FILE", str(tmp_path / "heartbeat"))
