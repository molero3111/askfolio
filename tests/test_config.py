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


def test_booking_disabled_when_env_missing(monkeypatch, reload_config):
    monkeypatch.delenv("BOOKING_ENABLED", raising=False)
    assert reload_config().BOOKING_ENABLED is False


def test_skip_dotenv_flag_is_exposed(monkeypatch, reload_config):
    monkeypatch.setenv("ASKFOLIO_SKIP_DOTENV", "1")
    assert reload_config().SKIP_DOTENV is True
    monkeypatch.delenv("ASKFOLIO_SKIP_DOTENV")
    assert reload_config().SKIP_DOTENV is False
