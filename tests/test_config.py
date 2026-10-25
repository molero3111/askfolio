import importlib
from pathlib import Path

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


def test_relative_path_resolves_under_project_root_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    resolved = config.resolve_path("secrets/token.json")
    assert resolved == config.PROJECT_ROOT / "secrets" / "token.json"
    assert (config.PROJECT_ROOT / "app").is_dir()


def test_absolute_path_is_unchanged(tmp_path):
    p = tmp_path / "token.json"
    assert config.resolve_path(str(p)) == p
    assert isinstance(config.resolve_path(str(p)), Path)
