import importlib
import os
from pathlib import Path

import pytest

from app import config


@pytest.fixture
def reload_config(monkeypatch):
    yield lambda: importlib.reload(config)
    # Never let the teardown reload read the developer's real .env.
    monkeypatch.setenv("ASKFOLIO_SKIP_DOTENV", "1")
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
    calls = []
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: calls.append(a))
    monkeypatch.setenv("ASKFOLIO_SKIP_DOTENV", "1")
    assert reload_config().SKIP_DOTENV is True
    assert calls == []
    monkeypatch.delenv("ASKFOLIO_SKIP_DOTENV")
    assert reload_config().SKIP_DOTENV is False
    assert calls == [(config.PROJECT_ROOT / ".env",)]


def test_blank_langsmith_key_turns_tracing_off(monkeypatch, reload_config):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    reload_config()

    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_langsmith_key_leaves_tracing_env_alone(monkeypatch, reload_config):
    monkeypatch.setenv("LANGSMITH_API_KEY", "fake-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "true")

    reload_config()

    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    # The fixture's teardown reload must not keep the fake key in config
    monkeypatch.delenv("LANGSMITH_API_KEY")
