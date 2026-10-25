import importlib

import pytest

from app import config


@pytest.fixture
def reload_config():
    yield lambda: importlib.reload(config)
    importlib.reload(config)


def test_state_path_defaults(monkeypatch, reload_config):
    for k in ("GOOGLE_TOKEN_PATH", "GOOGLE_CREDENTIALS_PATH", "TELEGRAM_OFFSET_FILE"):
        monkeypatch.delenv(k, raising=False)
    c = reload_config()
    assert c.GOOGLE_TOKEN_PATH == "secrets/token.json"
    assert c.GOOGLE_CREDENTIALS_PATH == "secrets/credentials.json"
    assert c.TELEGRAM_OFFSET_FILE == "data/telegram_offset.txt"
