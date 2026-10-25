import pytest

from ingest.migrate import with_retry


def test_retries_connect_errors_then_succeeds():
    calls, sleeps = [], []

    def fake_connect():
        calls.append(1)
        if len(calls) <= 2:
            raise ConnectionRefusedError("refused")
        return "ok"

    assert with_retry(fake_connect, sleep=sleeps.append) == "ok"
    assert len(calls) == 3
    assert sleeps == [1, 2]


def test_gives_up_after_attempts():
    sleeps = []

    def always():
        raise ConnectionRefusedError("refused")

    with pytest.raises(ConnectionRefusedError):
        with_retry(always, attempts=3, sleep=sleeps.append)
    assert sleeps == [1, 2]


def test_non_connect_errors_not_retried():
    sleeps = []

    def bad():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        with_retry(bad, sleep=sleeps.append)
    assert sleeps == []
