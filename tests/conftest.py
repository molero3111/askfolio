import os

# Must be set before any test module imports app.config / main, so the developer's real
# .env is never loaded into the test process.
os.environ["ASKFOLIO_SKIP_DOTENV"] = "1"

import pytest


@pytest.fixture(autouse=True)
def _isolated_heartbeat(tmp_path, monkeypatch):
    """Keep every test off the real /tmp/heartbeat."""
    monkeypatch.setenv("HEARTBEAT_FILE", str(tmp_path / "heartbeat"))


@pytest.fixture(autouse=True)
def _no_real_knowledge_json(tmp_path, monkeypatch):
    """Keep the reply guard off the developer's real knowledge JSON."""
    from app import config
    from app.contact_guard import get_guard

    monkeypatch.setattr(config, "INGEST_KNOWLEDGE_JSON", str(tmp_path / "no-cv.json"))
    get_guard.cache_clear()
    yield
    get_guard.cache_clear()


@pytest.fixture(autouse=True)
def _isolated_retry_queue(tmp_path, monkeypatch):
    """Keep every test off a real queue file."""
    from app import config
    from app.retry_queue import get_queue

    monkeypatch.setattr(config, "RETRY_QUEUE_FILE", str(tmp_path / "retry_queue.json"))
    get_queue.cache_clear()
    yield
    get_queue.cache_clear()
