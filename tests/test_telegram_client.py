import pytest
import requests

from app import telegram_client
from app.telegram_client import TelegramError, send_message


class _Resp:
    def __init__(self, status):
        self.status_code = status


def _post_returning(monkeypatch, status):
    calls = []

    def post(url, json=None, timeout=None):
        calls.append(json)
        return _Resp(status)

    monkeypatch.setattr(telegram_client.requests, "post", post)
    return calls


@pytest.mark.parametrize(("status", "temporary"), [(429, True), (500, True), (502, True), (400, False), (403, False)])
def test_send_error_temporary_flag_follows_the_status(monkeypatch, status, temporary):
    _post_returning(monkeypatch, status)
    with pytest.raises(TelegramError) as raised:
        send_message(1, "hi")
    assert raised.value.temporary is temporary
    assert "api.telegram.org" not in str(raised.value)


def test_send_network_error_is_temporary_and_hides_the_url(monkeypatch):
    def post(url, json=None, timeout=None):
        raise requests.ConnectionError(f"failed for {url}")

    monkeypatch.setattr(telegram_client.requests, "post", post)
    with pytest.raises(TelegramError) as raised:
        send_message(1, "hi")
    assert raised.value.temporary is True
    assert "api.telegram.org" not in str(raised.value)


def test_send_allows_a_missing_reply_target(monkeypatch):
    calls = _post_returning(monkeypatch, 200)
    send_message(1, "hi", reply_to_message_id=5)
    assert calls[0]["reply_parameters"] == {"message_id": 5, "allow_sending_without_reply": True}
