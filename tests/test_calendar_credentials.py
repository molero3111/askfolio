import logging
from unittest.mock import MagicMock

from app import calendar_client


def _expired_creds():
    creds = MagicMock()
    creds.valid = False
    creds.expired = True
    creds.refresh_token = "r"
    creds.to_json.return_value = '{"token": "new"}'
    return creds


def _setup(monkeypatch, tmp_path, creds):
    token = tmp_path / "secrets" / "token.json"
    token.parent.mkdir()
    token.write_text("{}")
    monkeypatch.setattr(calendar_client, "GOOGLE_TOKEN_PATH", str(token))
    monkeypatch.setattr(
        calendar_client.Credentials, "from_authorized_user_file", lambda *a, **k: creds
    )
    monkeypatch.setattr(calendar_client, "Request", MagicMock())
    return token


def test_refresh_saves_token_privately(monkeypatch, tmp_path):
    creds = _expired_creds()
    token = _setup(monkeypatch, tmp_path, creds)
    assert calendar_client._get_credentials() is creds
    assert token.read_text() == '{"token": "new"}'
    assert token.stat().st_mode & 0o777 == 0o600


def test_save_failure_is_logged_not_raised(monkeypatch, tmp_path, caplog):
    creds = _expired_creds()
    _setup(monkeypatch, tmp_path, creds)

    def boom(path, text):
        raise OSError("disk full")

    monkeypatch.setattr(calendar_client, "atomic_write_secret", boom)
    with caplog.at_level(logging.WARNING):
        assert calendar_client._get_credentials() is creds
    assert any(r.exc_info for r in caplog.records)
