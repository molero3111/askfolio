import hashlib
import hmac
import logging

import pytest

from app import analytics, config

KEY = "phc_test_" + "k" * 24
SALT = "s3"


class FakeClient:
    def __init__(self, fail: bool = False):
        self.captures: list[tuple[str, str, dict]] = []
        self.shutdowns = 0
        self.fail = fail

    def capture(self, event, distinct_id=None, properties=None):
        if self.fail:
            raise RuntimeError("posthog down")
        self.captures.append((event, distinct_id, properties))

    def shutdown(self):
        self.shutdowns += 1


@pytest.fixture
def client(monkeypatch):
    """Analytics on, with a fake client; every test starts from a clean module state."""
    fake = FakeClient()
    monkeypatch.setattr(config, "POSTHOG_KEY", KEY)
    monkeypatch.setattr(config, "ANALYTICS_SALT", SALT)
    monkeypatch.setattr(config, "ANALYTICS_ENV", "production")
    monkeypatch.setattr(config, "ANALYTICS_INTERNAL_CHAT_IDS", [])
    monkeypatch.setattr(analytics, "_client", fake)
    return fake


@pytest.mark.parametrize("key,salt", [(KEY, ""), ("", SALT), ("", "")])
def test_disabled_without_key_or_salt(monkeypatch, key, salt):
    monkeypatch.setattr(config, "POSTHOG_KEY", key)
    monkeypatch.setattr(config, "ANALYTICS_SALT", salt)
    monkeypatch.setattr(analytics, "_client", None)
    built = []
    monkeypatch.setattr(analytics, "Posthog", lambda *a, **k: built.append(1))

    assert analytics.enabled() is False
    analytics.track_question(42, "What stack?", "answered", latency_ms=5)
    assert built == []
    assert analytics._client is None


def test_distinct_id_is_salted_hash(client):
    expected = hmac.new(b"s3", b"12345", hashlib.sha256).hexdigest()[:32]
    assert analytics.distinct_id(12345) == expected
    assert analytics.distinct_id("12345") == expected
    assert "12345" not in analytics.distinct_id(12345)


def test_question_event_properties(client):
    analytics.track_question(42, "What stack?", "answered", latency_ms=1200)

    assert len(client.captures) == 1
    event, did, props = client.captures[0]
    assert event == "question_asked"
    assert did == analytics.distinct_id(42)
    assert props["project_source"] == "askfolio"
    assert props["environment"] == "production"
    assert props["is_internal"] is False
    assert props["$ip"] is None
    assert props["$process_person_profile"] is False
    assert props["question"] == "What stack?"
    assert props["outcome"] == "answered"
    assert props["latency_ms"] == 1200
    assert "rate_limit_scope" not in props


def test_rate_limited_carries_scope(client):
    analytics.track_question(42, "hi", "rate_limited", scope="global")

    props = client.captures[0][2]
    assert props["rate_limit_scope"] == "global"
    assert "latency_ms" not in props


def test_internal_chat_is_tagged(client, monkeypatch):
    monkeypatch.setattr(config, "ANALYTICS_INTERNAL_CHAT_IDS", ["42"])
    analytics.track_question(42, "a", "answered", latency_ms=1)
    analytics.track_question(43, "b", "answered", latency_ms=1)

    assert client.captures[0][2]["is_internal"] is True
    assert client.captures[1][2]["is_internal"] is False


def test_clean_question_scrubs_then_caps():
    cleaned = analytics.clean_question("a" * 495 + " jane.doe@example.com")
    assert "@" not in cleaned
    assert "example" not in cleaned
    assert len(cleaned) <= 500

    assert "[contact removed]" in analytics.clean_question("call me at +1 415 555 0100 please")
    assert "555" not in analytics.clean_question("call me at +1 415 555 0100 please")


def test_question_sent_is_cleaned(client):
    analytics.track_question(42, "reach me at jane.doe@example.com", "answered", latency_ms=1)
    assert "example.com" not in client.captures[0][2]["question"]


def test_capture_failure_is_swallowed(client, caplog):
    client.fail = True
    with caplog.at_level(logging.WARNING):
        assert analytics.track_question(42, "secret question text", "answered", latency_ms=1) is None

    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert "secret question text" not in caplog.text


def test_shutdown_is_safe(client, monkeypatch):
    analytics.shutdown()
    analytics.shutdown()
    assert client.shutdowns == 1

    monkeypatch.setattr(config, "POSTHOG_KEY", "")
    analytics.shutdown()  # disabled and no client: nothing to do


def test_client_is_async_with_short_timeout(monkeypatch):
    monkeypatch.setattr(config, "POSTHOG_KEY", KEY)
    monkeypatch.setattr(config, "POSTHOG_HOST", "https://ph.example.test")
    monkeypatch.setattr(analytics, "_client", None)
    seen = {}

    def fake_posthog(key, **kwargs):
        seen["key"] = key
        seen.update(kwargs)
        return FakeClient()

    monkeypatch.setattr(analytics, "Posthog", fake_posthog)
    analytics._get_client()

    assert seen["key"] == KEY
    assert seen["host"] == "https://ph.example.test"
    assert seen["timeout"] == 5
    assert seen["max_queue_size"] == 1000
    assert not seen.get("sync_mode")


def test_redaction_covers_analytics_secrets(monkeypatch):
    import main

    installed = []
    monkeypatch.setattr(main, "install_redaction", lambda secrets: installed.extend(secrets))
    monkeypatch.setattr(main, "install_excepthook", lambda: None)
    monkeypatch.setattr(main.logging, "basicConfig", lambda **k: None)
    monkeypatch.setattr(main.logging, "captureWarnings", lambda flag: None)
    monkeypatch.setattr(main, "POSTHOG_KEY", KEY)
    monkeypatch.setattr(main, "ANALYTICS_SALT", "salt-value-xyz")

    main.configure_process()

    assert KEY in installed
    assert "salt-value-xyz" in installed
