import json
import logging
import stat

import pytest

from app import retry_queue
from app.retry_queue import AddResult, RetryQueue, get_queue

MARKER = "fabricated private phrase"


class Clock:
    def __init__(self, now=1000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def path(tmp_path):
    return tmp_path / "queue.json"


def _queue(path, clock, **overrides):
    settings = {"interval": 1800, "after_success": 60, "max_age": 86400, "max_items": 20, "max_per_chat": 3}
    return RetryQueue(path, clock=clock, **{**settings, **overrides})


def test_add_persists_owner_only(path, clock):
    assert _queue(path, clock).add(1, 10, "hello") is AddResult.ADDED

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    reloaded = _queue(path, clock)
    assert len(reloaded) == 1
    clock.now += 1800
    item = reloaded.next_due()
    assert (item.chat_id, item.message_id, item.question, item.attempts) == (1, 10, "hello", 0)


def test_first_add_waits_one_interval(path, clock):
    q = _queue(path, clock)
    q.add(1, 10, "first")
    clock.now += 1000
    q.add(2, 20, "second")  # must not push the first attempt back
    assert q.next_due() is None
    clock.now += 799
    assert q.next_due() is None
    clock.now += 1
    assert q.next_due().question == "first"


def test_empty_queue_has_nothing_due(path, clock):
    assert _queue(path, clock).next_due() is None


def test_duplicate_in_same_chat_is_not_added(path, clock):
    q = _queue(path, clock)
    assert q.add(1, 10, " Hello ") is AddResult.ADDED
    assert q.add(1, 11, "hello") is AddResult.DUPLICATE
    assert len(q) == 1
    assert q.add(2, 20, "hello") is AddResult.ADDED


def test_per_chat_cap(path, clock):
    q = _queue(path, clock)
    for i in range(3):
        assert q.add(1, i, f"q{i}") is AddResult.ADDED
    assert q.add(1, 9, "q9") is AddResult.FULL
    assert q.add(2, 9, "q9") is AddResult.ADDED


def test_total_cap(path, clock):
    q = _queue(path, clock)
    for chat in range(20):
        assert q.add(chat, 1, "q") is AddResult.ADDED
    assert q.add(99, 1, "q") is AddResult.FULL


def test_zero_max_items_is_always_full(path, clock):
    assert _queue(path, clock, max_items=0).add(1, 1, "q") is AddResult.FULL


def test_write_failure_rolls_back(path, clock, monkeypatch):
    def boom(*a):
        raise OSError("read-only")

    q = _queue(path, clock)
    monkeypatch.setattr(retry_queue, "atomic_write_secret", boom)
    assert q.add(1, 10, "hello") is AddResult.FAILED
    assert len(q) == 0
    clock.now += 1800
    assert q.next_due() is None


def test_failed_attempt_reschedules_and_rotates(path, clock):
    q = _queue(path, clock)
    q.add(1, 10, "a")
    q.add(2, 20, "b")
    clock.now += 1800
    a = q.next_due()
    assert a.question == "a"

    q.mark_failed(a)

    assert a.attempts == 1
    assert q.next_due() is None
    clock.now += 1800
    assert q.next_due().question == "b"


def test_attempts_survive_a_reload(path, clock):
    q = _queue(path, clock)
    q.add(1, 10, "a")
    clock.now += 1800
    q.mark_failed(q.next_due())

    clock.now += 1800
    assert _queue(path, clock).next_due().attempts == 1


def test_done_shortens_next_wait(path, clock):
    q = _queue(path, clock)
    q.add(1, 10, "a")
    q.add(2, 20, "b")
    clock.now += 1800

    q.mark_done(q.next_due())

    assert len(q) == 1
    clock.now += 59
    assert q.next_due() is None
    clock.now += 1
    assert q.next_due().question == "b"


def test_emptied_queue_removes_the_file_and_next_add_waits_again(path, clock):
    q = _queue(path, clock)
    q.add(1, 10, "a")
    clock.now += 1800
    q.mark_done(q.next_due())
    assert not path.exists()

    q.add(1, 11, "b")
    clock.now += 1799
    assert q.next_due() is None


def test_pop_expired_at_exactly_max_age(path, clock):
    q = _queue(path, clock)
    q.add(1, 10, "old")
    clock.now += 100
    q.add(2, 20, "young")
    clock.now += 86300

    expired = q.pop_expired()

    assert [i.question for i in expired] == ["old"]
    assert len(q) == 1
    assert len(_queue(path, clock)) == 1


def test_pop_expired_last_item_removes_the_file(path, clock):
    q = _queue(path, clock)
    q.add(1, 10, "old")
    clock.now += 86400
    q.pop_expired()
    assert not path.exists()


def test_failed_removal_write_is_logged_and_kept_in_memory(path, clock, monkeypatch, caplog):
    q = _queue(path, clock)
    q.add(1, 10, MARKER)
    q.add(2, 20, "b")
    clock.now += 1800
    item = q.next_due()

    def boom(*a):
        raise OSError(MARKER)

    monkeypatch.setattr(retry_queue, "atomic_write_secret", boom)
    with caplog.at_level(logging.WARNING):
        q.mark_done(item)

    assert len(q) == 1
    assert "OSError" in caplog.text
    assert MARKER not in caplog.text


@pytest.mark.parametrize("content", ["{not json", "[]", '{"items": "nope"}'])
def test_corrupt_file_starts_empty(path, clock, caplog, content):
    path.write_text(content)
    with caplog.at_level(logging.WARNING):
        q = _queue(path, clock)
    assert len(q) == 0
    assert "[retry_queue]" in caplog.text


def test_invalid_items_are_skipped_without_logging_text(path, clock, caplog):
    good = {"id": "a", "chat_id": 1, "message_id": 10, "question": "fine", "created_at": 1000.0, "attempts": 0}
    bad = {"id": "b", "message_id": 20, "question": MARKER, "created_at": 1000.0, "attempts": 0}
    path.write_text(json.dumps({"version": 1, "next_attempt_at": 0, "items": [good, bad]}))

    with caplog.at_level(logging.WARNING):
        q = _queue(path, clock)

    assert len(q) == 1
    assert "skipped 1" in caplog.text
    assert MARKER not in caplog.text


def test_get_queue_uses_config_and_reports_resumed_items(caplog):
    get_queue().add(1, 10, MARKER)
    get_queue.cache_clear()

    with caplog.at_level(logging.INFO):
        assert len(get_queue()) == 1

    assert "[retry_queue] resumed 1" in caplog.text
    assert MARKER not in caplog.text
